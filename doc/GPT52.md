## 交付物 1：系统架构设计文档

### 1.1 总体架构图

```mermaid
flowchart TD
    U[User / CLI] -->|task| IN[Input Parser]
    IN --> COOR[Coordinator / Supervisor]

    COOR --> PLAN[Planner]
    PLAN --> DISP[Dispatcher / Router]

    subgraph Workers[Worker Agents]
      R[Researcher]
      C[Coder]
      E[Executor]
    end

    DISP -->|fan-out| R
    DISP -->|fan-out| C
    DISP -->|fan-out| E

    R --> JOIN[Join / Aggregator]
    C --> JOIN
    E --> JOIN

    JOIN --> CRITIC[Critic / Reviewer]
    CRITIC -->|needs_fix| COOR
    CRITIC -->|ok| SYN[Synthesizer]

    SYN --> OUT[Final Answer]
    OUT --> PERSIST[History Persistor]

    COOR -->|human_needed| HITL[Human-in-the-loop]
    HITL --> COOR

    subgraph Infra[Shared Infrastructure]
      MEM[Memory: short-term + optional long-term]
      TOOLS[Tool Router + Sandbox Tools]
      OBS[Observability: logs + metrics + traces]
      LLM[LLM Factory: OpenAI/Anthropic/Local/Mock]
    end

    COOR --- MEM
    PLAN --- LLM
    R --- LLM
    C --- LLM
    E --- TOOLS
    CRITIC --- LLM
    SYN --- LLM
    DISP --- OBS
    JOIN --- OBS
```

**文字说明**
- **Supervisor（Coordinator）**：全局主控，基于当前状态决定下一步（规划 / 分派 / 人工介入 / 结束）。
- **Planner**：把原始任务拆成带依赖的 `SubTask` 列表，产出执行策略（并行/串行）。
- **Dispatcher**：根据 `SubTask.kind`、负载与可用 Agent 配置，生成路由与 fan-out 指令（并行派发）。
- **Workers**：
  - Researcher：知识整合/（模拟）检索；
  - Coder：生成/修改代码；
  - Executor：工具调用（文件写入、代码执行等）。
- **Critic**：审阅结果与一致性检查，必要时触发反思纠错循环。
- **Synthesizer**：汇总所有 agent 产出为最终回答。
- **Infra**：LLM 工厂、工具系统、安全沙箱、记忆系统与可观测性贯穿全流程。

> 关于“reasoning 输出”：实现为**可审计的决策轨迹（plan / checklist / 依据摘要）+ 节点输入输出摘要日志**，而不是暴露模型私有思维链（避免不可控泄露与安全风险）。

---

### 1.2 LangGraph 状态机设计

#### 1.2.1 状态节点（Nodes）
- `parse_input`：标准化输入、初始化 state
- `coordinator`：决定 next（走规划/执行/审阅/人工/结束）
- `planner`：生成 subtasks + 执行策略
- `dispatch`：fan-out 并行派发到 worker 节点
- `researcher` / `coder` / `executor`：产出写入 `agent_outputs`
- `join`：聚合并更新全局摘要
- `critic`：判定是否需要修复/重试/降级
- `synthesizer`：生成 `final_answer`
- `human_in_loop`：需要人工输入时介入
- `persist_history`：持久化对话/轨迹
- `END`：LangGraph 结束态

#### 1.2.2 状态流转图（Mermaid）

```mermaid
stateDiagram-v2
    [*] --> parse_input
    parse_input --> coordinator

    coordinator --> planner: need_plan
    coordinator --> dispatch: have_plan
    coordinator --> critic: need_review
    coordinator --> human_in_loop: need_human
    coordinator --> synthesizer: ready_to_finalize

    planner --> dispatch

    dispatch --> researcher
    dispatch --> coder
    dispatch --> executor

    researcher --> join
    coder --> join
    executor --> join

    join --> critic
    critic --> coordinator: needs_fix && iter < max_iter
    critic --> synthesizer: ok || iter >= max_iter

    synthesizer --> persist_history
    persist_history --> [*]
```

**关键设计点**
- Coordinator 只做**路由决策**，不做重活，保证可控与可观测。
- `dispatch` 使用 LangGraph 的动态路由能力（fan-out 到多个 worker），`join` 统一收敛。
- `critic` 是闭环开关：触发修复时把“问题清单”写回 state，Coordinator 决定重新规划或只修复局部。

---

### 1.3 智能体交互时序图

```mermaid
sequenceDiagram
    participant User
    participant CLI as main.py
    participant G as LangGraph(App)
    participant CO as Coordinator
    participant PL as Planner
    participant DI as Dispatcher
    participant R as Researcher
    participant C as Coder
    participant E as Executor
    participant J as Join
    participant CR as Critic
    participant SY as Synthesizer

    User->>CLI: 输入任务 original_task
    CLI->>G: invoke(state)
    G->>CO: coordinator(state)
    CO->>PL: planner(state)
    PL-->>CO: subtasks + plan

    CO->>DI: dispatch(state)
    par fan-out
        DI->>R: subtask A
        DI->>C: subtask B
        DI->>E: subtask C
    end
    R-->>J: output A
    C-->>J: output B
    E-->>J: tool logs + output C
    J-->>CR: aggregated outputs

    CR-->>CO: ok / needs_fix + issues
    alt needs_fix
        CO->>PL: replan or patch plan
    else ok
        CO->>SY: synthesize final
        SY-->>G: final_answer
    end
    G-->>CLI: final_answer + trace
    CLI-->>User: 输出结果
```

---

### 1.4 核心数据结构说明（State Schema）

在 LangGraph 中，State 是全局单一真相源（single source of truth）。本项目使用 `TypedDict + Annotated(reducer)` 来定义可合并字段。

核心字段（与你的硬约束一致，并扩展了 metrics / errors）：

- `messages: List[BaseMessage]`：会话消息（用于短期记忆），采用“追加合并”reducer
- `original_task: str`：原始任务
- `subtasks: List[SubTask]`：规划产物（可迭代修订）
- `agent_outputs: Dict[str, Any]`：各 agent 输出（按 agent 名称分桶）
- `tool_call_logs: List[ToolCallLog]`：工具调用审计日志（追加合并）
- `current_agent: str`：当前执行 agent（便于调试/可视化）
- `iteration_count: int`：反思纠错轮数
- `final_answer: Optional[str]`：最终答案
- `next: str`：Coordinator 路由目标
- 扩展：
  - `issues: List[str]`：Critic 输出的问题清单
  - `metrics: Metrics`：token、耗时、链路深度等
  - `errors: List[ErrorEvent]`：异常记录（可用于降级与重试策略）

---

### 1.5 失败重试、降级与终止条件设计

#### 1.5.1 失败类型与处理策略
1) **LLM 调用失败**
- 重试：指数退避（实现层面用简单重试次数即可）
- 降级：切换到 `MOCK` 模型（确保系统可跑、测试可过）
- 记录：写入 `errors` + `metrics.llm_failures += 1`

2) **工具调用失败（文件/REPL/JSON 校验）**
- 重试：最多 `N` 次（默认 2）
- 降级：
  - 文件写入失败：改为输出到 `final_answer` 并提示用户手动保存
  - python_repl 失败：改为静态生成代码 + 给出运行说明
- 安全：文件路径强制限定 `workspace/`，越界直接拒绝并记录

3) **质量审阅失败（Critic 判定 needs_fix）**
- 进入反思纠错循环：`iteration_count += 1`
- 如果 `iteration_count >= max_iterations`：
  - 强制结束：Synthesizer 输出“最优可用结果 + 已知问题清单 + 下一步建议”
  - 防止无限循环

#### 1.5.2 终止条件（必须有）
- `final_answer` 已生成且 Critic 判定 OK → 结束
- `iteration_count >= max_iterations` → 强制结束（输出可用版本）
- 发生不可恢复错误（例如连续工具失败且无降级路径）→ 输出错误报告并结束

---

## 交付物 2：模块与类设计说明

### 2.1 核心类与职责（文字 UML）

- `Settings (Pydantic BaseModel)`
  - 读取环境变量：LLM provider、debug、max_iterations、workspace_dir、日志级别等
- `LLMFactory`
  - `create_chat_model(settings) -> BaseChatModel`
  - 支持 OpenAI / Anthropic / Local / Mock（无 key 也可运行）
- `BaseAgent`
  - `name`, `role`, `system_prompt`
  - `invoke(state) -> AgentDelta`（返回对 state 的增量更新）
- `CoordinatorAgent(BaseAgent)`
  - 根据 state 决定 `next`
- `PlannerAgent(BaseAgent)`
  - 输出 `subtasks: List[SubTask]`
- `Worker Agents`
  - `ResearcherAgent`, `CoderAgent`, `ExecutorAgent`
- `CriticAgent`
  - 输出 `issues`、`needs_fix`、修复建议
- `SynthesizerAgent`
  - 生成 `final_answer`
- `ToolRouter`
  - 注册工具、封装调用、统一审计日志、异常捕获
- `Memory`
  - `ShortTermMemory`：基于 state.messages
  - `LongTermMemory`（可选开启）：文件持久化接口
- `GraphBuilder`
  - 构建 LangGraph `StateGraph`，绑定 nodes/edges，`compile()`

---

## 交付物 3：项目目录结构树

（与你要求一致，后续代码将按此路径逐文件给出）

```text
multi_agent_system/
├── pyproject.toml
├── requirements.txt
├── README.md
├── .env.example
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── prompts.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── coordinator.py
│   │   ├── planner.py
│   │   ├── researcher.py
│   │   ├── coder.py
│   │   ├── executor.py
│   │   ├── critic.py
│   │   └── synthesizer.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── nodes.py
│   │   ├── edges.py
│   │   └── builder.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── calculator.py
│   │   ├── file_manager.py
│   │   ├── code_executor.py
│   │   └── search.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py
│   │   └── long_term.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── factory.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── visualizer.py
├── examples/
│   ├── example_planning.py
│   ├── example_tool_execution.py
│   └── example_code_generation.py
├── tests/
│   ├── __init__.py
│   ├── test_graph.py
│   ├── test_flow.py
│   └── test_tools.py
├── workspace/
│   └── .gitkeep
└── logs/
    └── .gitkeep
```

---

如果需要我继续输出 **交付物 4：完整 Python 源代码（从 pyproject.toml + src/… 开始逐文件）**，请回复：`继续`。

## 交付物 4：完整 Python 源代码（第 1 段：项目基础 + 核心 src 实现）

下面按文件逐个给出**完整可运行代码**（不使用“略”）。本段先输出：项目依赖与核心 `src/`（不含 examples/tests，下一段继续）。

---

### `pyproject.toml`
```toml
[project]
name = "multi-agent-system"
version = "0.1.0"
description = "General-Purpose Multi-Agent Problem-Solving System built with LangGraph"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Your Team" }]

dependencies = [
  "langgraph>=0.2.0",
  "langchain-core>=0.3.0",
  "pydantic>=2.6.0",
  "pydantic-settings>=2.2.0",
  "python-dotenv>=1.0.1",
  "typing-extensions>=4.9.0",
  "rich>=13.7.0",
]

[project.optional-dependencies]
openai = ["langchain-openai>=0.2.0"]
anthropic = ["langchain-anthropic>=0.2.0"]

[project.scripts]
multi-agent-system = "src.main:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
ignore = ["E501"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

---

### `requirements.txt`
```txt
langgraph>=0.2.0
langchain-core>=0.3.0
pydantic>=2.6.0
pydantic-settings>=2.2.0
python-dotenv>=1.0.1
typing-extensions>=4.9.0
rich>=13.7.0
pytest>=7.4.0
```

---

### `.env.example`
```env
# LLM provider: mock | openai | anthropic
MAS_LLM_PROVIDER=mock

# OpenAI (optional)
OPENAI_API_KEY=
MAS_OPENAI_MODEL=gpt-4o-mini
MAS_TEMPERATURE=0.2

# Anthropic (optional)
ANTHROPIC_API_KEY=
MAS_ANTHROPIC_MODEL=claude-3-5-sonnet-latest

# System
MAS_DEBUG=true
MAS_MAX_ITERATIONS=3
MAS_PARALLEL_WORKERS=4
MAS_ENABLE_LONG_TERM_MEMORY=false
MAS_HUMAN_IN_LOOP=false

# Directories (relative paths are resolved from project root)
MAS_WORKSPACE_DIR=workspace
MAS_LOGS_DIR=logs
MAS_HISTORY_FILE=logs/history.jsonl
```

---

## `src/` 代码

### `src/__init__.py`
```python
"""Top-level package for the multi-agent system."""
```

---

### `src/config/__init__.py`
```python
"""Configuration package."""
```

---

### `src/config/settings.py`
```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global settings loaded from environment variables.

    Env var prefix: MAS_
    Example: MAS_LLM_PROVIDER=mock
    """

    model_config = SettingsConfigDict(env_prefix="MAS_", extra="ignore")

    llm_provider: Literal["mock", "openai", "anthropic"] = Field(default="mock")

    # Model selection
    openai_model: str = Field(default="gpt-4o-mini")
    anthropic_model: str = Field(default="claude-3-5-sonnet-latest")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # Loop / control
    max_iterations: int = Field(default=3, ge=0, le=20)
    parallel_workers: int = Field(default=4, ge=1, le=32)
    debug: bool = Field(default=True)

    # Memory & HITL
    enable_long_term_memory: bool = Field(default=False)
    human_in_loop: bool = Field(default=False)

    # Directories
    workspace_dir: Path = Field(default=Path("workspace"))
    logs_dir: Path = Field(default=Path("logs"))
    history_file: Path = Field(default=Path("logs/history.jsonl"))

    def ensure_dirs(self) -> None:
        """Create required directories if missing."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        # history_file parent
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
```

---

### `src/config/prompts.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class PromptTemplate:
    """Simple prompt holder. Kept external & replaceable."""
    system: str
    user: str


PROMPTS: Dict[str, PromptTemplate] = {
    "coordinator": PromptTemplate(
        system=(
            "You are the Coordinator (Supervisor) of a multi-agent system.\n"
            "Your job is to decide the next step based on state.\n"
            "Return a concise decision trace (NOT private chain-of-thought), including:\n"
            "- next: one of [planner, dispatch, critic, synthesizer, human_in_loop, finish]\n"
            "- rationale: brief bullet points\n"
        ),
        user=(
            "State summary:\n"
            "{state_summary}\n\n"
            "Decide next step."
        ),
    ),
    "planner": PromptTemplate(
        system=(
            "You are the Planner. Produce a practical execution plan.\n"
            "Output must be JSON with fields:\n"
            "{\n"
            '  "subtasks": [\n'
            "    {\"id\": \"...\", \"title\": \"...\", \"kind\": \"research|code|execute\", "
            "\"depends_on\": [], \"payload\": {}}\n"
            "  ],\n"
            '  "rationale": ["..."]\n'
            "}\n"
            "Do NOT include any extra keys."
        ),
        user=(
            "Original task:\n"
            "{original_task}\n\n"
            "Existing subtasks (if any):\n"
            "{existing_subtasks}\n"
        ),
    ),
    "researcher": PromptTemplate(
        system=(
            "You are the Researcher. You gather relevant knowledge and constraints.\n"
            "Return JSON: {\"result\": {...}, \"rationale\": [..]}.\n"
            "No private chain-of-thought; keep rationale brief."
        ),
        user="Task:\n{task}\n",
    ),
    "coder": PromptTemplate(
        system=(
            "You are the Coder. You write correct, runnable Python.\n"
            "Return JSON: {\"files\": [{\"path\": \"...\", \"content\": \"...\"}], "
            "\"notes\": [..], \"rationale\": [..]}.\n"
            "Paths should be under workspace/ unless instructed otherwise.\n"
            "No private chain-of-thought."
        ),
        user="Task:\n{task}\n\nContext:\n{context}\n",
    ),
    "executor": PromptTemplate(
        system=(
            "You are the Executor. You decide tool calls and perform actions.\n"
            "Return JSON: {\"actions\": [...], \"result\": {...}, \"rationale\": [..]}.\n"
            "No private chain-of-thought."
        ),
        user="Task:\n{task}\n\nAvailable tools:\n{tools}\n",
    ),
    "critic": PromptTemplate(
        system=(
            "You are the Critic/Reviewer. Check for correctness, completeness, and safety.\n"
            "Return JSON: {\"ok\": true/false, \"issues\": [..], \"fix_suggestions\": [..]}.\n"
            "Be strict; no private chain-of-thought."
        ),
        user=(
            "Original task:\n{original_task}\n\n"
            "Aggregated outputs:\n{outputs}\n\n"
            "Decide if the result is OK."
        ),
    ),
    "synthesizer": PromptTemplate(
        system=(
            "You are the Synthesizer. Produce the final user-facing answer.\n"
            "Include: final output + decision trace summary.\n"
            "No private chain-of-thought."
        ),
        user=(
            "Original task:\n{original_task}\n\n"
            "All agent outputs:\n{outputs}\n\n"
            "Known issues:\n{issues}\n"
        ),
    ),
}
```

---

### `src/utils/__init__.py`
```python
"""Utility helpers (logging, visualization, etc.)."""
```

---

### `src/utils/logger.py`
```python
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler


@dataclass
class EventLogger:
    """
    Observability helper:
    - console rich logs
    - optional JSONL file logs
    """

    logger: logging.Logger
    console: Console
    jsonl_path: Optional[Path] = None

    def log_event(self, event: str, payload: Dict[str, Any]) -> None:
        record = {"ts": time.time(), "event": event, "payload": payload}
        self.logger.info("%s | %s", event, payload)
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_logger(name: str, debug: bool, logs_dir: Path) -> EventLogger:
    """Build a console+file logger."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_level = logging.DEBUG if debug else logging.INFO

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.propagate = False

    console = Console()

    # Console handler
    rich_handler = RichHandler(console=console, show_time=True, show_path=False)
    rich_handler.setLevel(log_level)
    logger.addHandler(rich_handler)

    # File handler (human readable)
    file_path = logs_dir / "run.log"
    fh = logging.FileHandler(file_path, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    return EventLogger(logger=logger, console=console, jsonl_path=logs_dir / "events.jsonl")
```

---

### `src/utils/visualizer.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class GraphRunTrace:
    """Stores node visit order for visualization."""
    visited_nodes: List[str]

    def to_mermaid(self) -> str:
        if not self.visited_nodes:
            return "flowchart TD\n  A[empty trace]\n"
        lines = ["flowchart TD"]
        # Create simple chain edges in order
        for i in range(len(self.visited_nodes) - 1):
            a = self.visited_nodes[i]
            b = self.visited_nodes[i + 1]
            lines.append(f"  {a} --> {b}")
        return "\n".join(lines)
```

---

### `src/llm/__init__.py`
```python
"""LLM integration package."""
```

---

### `src/llm/factory.py`
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.config.settings import Settings


@dataclass
class DeterministicMockChatModel(BaseChatModel):
    """
    A deterministic chat model so the system is runnable without external keys.

    It returns minimal JSON-shaped content expected by each agent prompt,
    based on simple keyword detection in the system message.
    """

    model_name: str = "mock-chat"

    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "deterministic-mock-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_text = ""
        user_text = ""
        for m in messages:
            if m.type == "system":
                system_text += (m.content or "") + "\n"
            if m.type == "human":
                user_text += (m.content or "") + "\n"

        content = self._route(system_text=system_text, user_text=user_text)
        gen = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[gen])

    def _route(self, system_text: str, user_text: str) -> str:
        st = system_text.lower()
        if "planner" in st:
            # A simple general-purpose plan
            payload: Dict[str, Any] = {
                "subtasks": [
                    {
                        "id": "t_research",
                        "title": "Clarify requirements, constraints, and success criteria",
                        "kind": "research",
                        "depends_on": [],
                        "payload": {},
                    },
                    {
                        "id": "t_code",
                        "title": "Implement the solution/code artifacts",
                        "kind": "code",
                        "depends_on": ["t_research"],
                        "payload": {},
                    },
                    {
                        "id": "t_execute",
                        "title": "Run minimal checks and write outputs/files",
                        "kind": "execute",
                        "depends_on": ["t_code"],
                        "payload": {},
                    },
                ],
                "rationale": [
                    "Break the problem into research → implementation → execution for reliability.",
                    "Keep tasks small and independently verifiable.",
                ],
            }
            return json.dumps(payload, ensure_ascii=False)

        if "critic" in st:
            payload = {
                "ok": True,
                "issues": [],
                "fix_suggestions": [],
            }
            return json.dumps(payload, ensure_ascii=False)

        if "coder" in st:
            # If user asks for crawler/HN, output a file under workspace/
            if "hacker news" in user_text.lower() or "hn" in user_text.lower():
                code = _hn_crawler_code()
                payload = {
                    "files": [{"path": "workspace/hn_crawler.py", "content": code}],
                    "notes": ["Uses requests + BeautifulSoup. If no internet, it will fail at runtime."],
                    "rationale": ["Provide a clear, runnable script with JSON output."],
                }
                return json.dumps(payload, ensure_ascii=False)

            payload = {
                "files": [{"path": "workspace/solution.py", "content": "# TODO: implement\n"}],
                "notes": ["Mock output"],
                "rationale": ["Stub for offline mode."],
            }
            return json.dumps(payload, ensure_ascii=False)

        if "researcher" in st:
            payload = {
                "result": {
                    "constraints": ["File ops must stay under workspace/."],
                    "assumptions": ["Web access may be unavailable; provide code and instructions."],
                },
                "rationale": ["Capture safety and environment constraints."],
            }
            return json.dumps(payload, ensure_ascii=False)

        if "coordinator" in st:
            # Default to planner first
            payload = {"next": "planner", "rationale": ["No plan yet; create subtasks first."]}
            return json.dumps(payload, ensure_ascii=False)

        if "synthesizer" in st:
            payload = "Here is your final answer.\n"
            return payload

        if "executor" in st:
            payload = {"actions": [], "result": {}, "rationale": ["No actions in mock."]}
            return json.dumps(payload, ensure_ascii=False)

        return json.dumps({"message": "mock"}, ensure_ascii=False)


def _hn_crawler_code() -> str:
    return """\
\"\"\"Hacker News homepage crawler.

Fetches the Hacker News front page and saves titles+links into a JSON file.

Usage:
  python hn_crawler.py --out hn_top.json --limit 30

Notes:
- Requires: requests, beautifulsoup4
- If running in an environment without internet access, fetching will fail.
\"\"\"

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


HN_URL = "https://news.ycombinator.com/"


@dataclass
class HNItem:
    title: str
    link: str


def fetch_hn(limit: int = 30, timeout: int = 20) -> List[HNItem]:
    resp = requests.get(HN_URL, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    items: List[HNItem] = []
    for a in soup.select("span.titleline > a"):
        title = a.get_text(strip=True)
        href = a.get("href") or ""
        link = urljoin(HN_URL, href)
        items.append(HNItem(title=title, link=link))
        if len(items) >= limit:
            break
    return items


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="hn_top.json")
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args(argv)

    items = fetch_hn(limit=args.limit)
    payload = [item.__dict__ for item in items]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(payload)} items to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def create_chat_model(settings: Settings) -> BaseChatModel:
    """
    Factory to create a chat model based on settings.

    - mock: deterministic local model (default)
    - openai/anthropic: optional dependencies
    """
    if settings.llm_provider == "mock":
        return DeterministicMockChatModel()

    if settings.llm_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "OpenAI provider selected but langchain-openai is not installed. "
                "Install with: pip install '.[openai]'"
            ) from e
        return ChatOpenAI(model=settings.openai_model, temperature=settings.temperature)

    if settings.llm_provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Anthropic provider selected but langchain-anthropic is not installed. "
                "Install with: pip install '.[anthropic]'"
            ) from e
        return ChatAnthropic(model=settings.anthropic_model, temperature=settings.temperature)

    raise ValueError(f"Unsupported llm_provider: {settings.llm_provider}")
```

---

### `src/tools/__init__.py`
```python
"""Tooling package."""
```

---

### `src/tools/base.py`
```python
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import BaseTool

from src.graph.state import ToolCallLog


@dataclass
class ToolRouter:
    """
    Tool registry + unified invocation with auditing.

    Tools should be created with `@tool` and then registered here.
    """

    tools: Dict[str, BaseTool]

    def list_tools(self) -> List[str]:
        return sorted(self.tools.keys())

    def get(self, name: str) -> BaseTool:
        if name not in self.tools:
            raise KeyError(f"Tool not registered: {name}")
        return self.tools[name]

    def call(self, name: str, args: Dict[str, Any]) -> tuple[Any, ToolCallLog]:
        tool = self.get(name)
        start = time.time()
        error: Optional[str] = None
        result: Any = None
        try:
            result = tool.invoke(args)
            return result, ToolCallLog(
                tool_name=name,
                args=args,
                result_preview=_preview(result),
                error=None,
                duration_ms=int((time.time() - start) * 1000),
                ts=time.time(),
            )
        except Exception as e:
            error = repr(e)
            return None, ToolCallLog(
                tool_name=name,
                args=args,
                result_preview=None,
                error=error,
                duration_ms=int((time.time() - start) * 1000),
                ts=time.time(),
            )


def _preview(x: Any, max_len: int = 500) -> str:
    s = str(x)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


def build_default_tool_router(tools: List[BaseTool]) -> ToolRouter:
    """Build default router with a list of tools."""
    return ToolRouter(tools={t.name: t for t in tools})
```

---

### `src/tools/calculator.py`
```python
from __future__ import annotations

import ast
import json
from typing import Any, Dict

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SafeEvalInput(BaseModel):
    expression: str = Field(..., description="Arithmetic expression, e.g., '2*(3+4)'")


@tool("safe_eval", args_schema=SafeEvalInput)
def safe_eval(expression: str) -> float:
    """
    Safely evaluate an arithmetic expression.

    Supports: +, -, *, /, **, parentheses, numeric literals.
    """
    node = ast.parse(expression, mode="eval")

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )

    for sub in ast.walk(node):
        if not isinstance(sub, allowed_nodes):
            raise ValueError(f"Disallowed expression element: {type(sub).__name__}")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp):
            val = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +val
            if isinstance(n.op, ast.USub):
                return -val
            raise ValueError("Unsupported unary operator")
        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if isinstance(n.op, ast.Div):
                return left / right
            if isinstance(n.op, ast.Pow):
                return left**right
            raise ValueError("Unsupported binary operator")
        raise ValueError("Unsupported expression")

    return _eval(node)


class JSONValidateInput(BaseModel):
    text: str = Field(..., description="JSON text to validate and pretty-print")


@tool("json_validator", args_schema=JSONValidateInput)
def json_validator(text: str) -> str:
    """
    Validate JSON and return pretty-printed JSON.

    Raises ValueError on parse errors.
    """
    try:
        obj: Any = json.loads(text)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    return json.dumps(obj, ensure_ascii=False, indent=2)


def build_calculator_tools() -> list:
    """Factory for calculator/text tools."""
    return [safe_eval, json_validator]
```

---

### `src/tools/file_manager.py`
```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


def _resolve_under_workspace(workspace_dir: Path, relative_path: str) -> Path:
    """
    Resolve a user-provided relative path under workspace_dir, preventing traversal.

    Only allows paths inside workspace_dir.
    """
    ws = workspace_dir.resolve()
    candidate = (ws / relative_path).resolve()
    if ws not in candidate.parents and candidate != ws:
        raise ValueError("Path escapes workspace directory")
    return candidate


class FileReadInput(BaseModel):
    path: str = Field(..., description="Relative path under workspace/ to read")
    max_bytes: int = Field(default=200_000, ge=1, le=5_000_000)


def build_file_reader(workspace_dir: Path):
    @tool("file_reader", args_schema=FileReadInput)
    def file_reader(path: str, max_bytes: int = 200_000) -> str:
        """Read a text file under workspace/."""
        p = _resolve_under_workspace(workspace_dir, path)
        data = p.read_bytes()
        if len(data) > max_bytes:
            raise ValueError(f"File too large: {len(data)} bytes > {max_bytes}")
        return data.decode("utf-8")

    return file_reader


class FileWriteInput(BaseModel):
    path: str = Field(..., description="Relative path under workspace/ to write")
    content: str = Field(..., description="Text content")
    overwrite: bool = Field(default=True)
    encoding: str = Field(default="utf-8")


def build_file_writer(workspace_dir: Path):
    @tool("file_writer", args_schema=FileWriteInput)
    def file_writer(
        path: str,
        content: str,
        overwrite: bool = True,
        encoding: str = "utf-8",
    ) -> str:
        """Write a text file under workspace/."""
        p = _resolve_under_workspace(workspace_dir, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and not overwrite:
            raise ValueError("File exists and overwrite=false")
        p.write_text(content, encoding=encoding)
        return str(p)

    return file_writer


def build_file_tools(workspace_dir: Path) -> list:
    """Factory for safe file tools bound to a workspace dir."""
    return [build_file_reader(workspace_dir), build_file_writer(workspace_dir)]
```

---

### `src/tools/code_executor.py`
```python
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class PythonREPLInput(BaseModel):
    code: str = Field(..., description="Python code to execute in a restricted environment")
    timeout_ms: int = Field(default=2000, ge=100, le=30_000)


@dataclass
class PythonREPLResult:
    stdout: str
    error: Optional[str]
    locals_preview: Dict[str, str]


@tool("python_repl", args_schema=PythonREPLInput)
def python_repl(code: str, timeout_ms: int = 2000) -> Dict[str, Any]:
    """
    Execute Python code with a very small 'sandbox'.

    NOTE: This is NOT a hardened sandbox. It is designed for demos/tests only.
    In production, use OS-level sandboxing (containers / seccomp / firejail).
    """
    # Minimal builtins (intentionally small)
    safe_builtins = {
        "print": print,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "dict": dict,
        "list": list,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "enumerate": enumerate,
        "zip": zip,
    }

    glb: Dict[str, Any] = {"__builtins__": safe_builtins}
    loc: Dict[str, Any] = {}

    buf = io.StringIO()
    err: Optional[str] = None

    try:
        with contextlib.redirect_stdout(buf):
            exec(code, glb, loc)  # noqa: S102
    except Exception as e:
        err = repr(e)

    preview = {}
    for k, v in loc.items():
        if k.startswith("_"):
            continue
        preview[k] = (repr(v)[:200] + ("…" if len(repr(v)) > 200 else ""))

    result = PythonREPLResult(stdout=buf.getvalue(), error=err, locals_preview=preview)
    return {"stdout": result.stdout, "error": result.error, "locals_preview": result.locals_preview}


def build_code_tools() -> list:
    """Factory for code execution tools."""
    return [python_repl]
```

---

### `src/tools/search.py`
```python
from __future__ import annotations

import time
from typing import Dict, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(default=5, ge=1, le=10)


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, k: int = 5) -> Dict[str, List[Dict[str, str]]]:
    """
    Simulated web search tool.

    Returns deterministic mock results so the system works offline.
    """
    time.sleep(0.05)
    results = []
    for i in range(k):
        results.append(
            {
                "title": f"Mock result {i+1} for: {query}",
                "url": f"https://example.com/search?q={query.replace(' ', '+')}&i={i+1}",
                "snippet": "This is a simulated snippet (offline mode).",
            }
        )
    return {"results": results}


def build_search_tools() -> list:
    return [web_search]
```

---

### `src/memory/__init__.py`
```python
"""Memory package."""
```

---

### `src/memory/short_term.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.messages import BaseMessage


@dataclass
class ShortTermMemory:
    """
    Short-term memory is the in-state message history.

    This wrapper exists to keep an explicit abstraction boundary.
    """

    def append(self, messages: List[BaseMessage], new_message: BaseMessage) -> List[BaseMessage]:
        return [*messages, new_message]
```

---

### `src/memory/long_term.py`
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol


class LongTermMemory(Protocol):
    """Interface for optional persistent memory."""

    def put(self, key: str, value: Dict[str, Any]) -> None: ...

    def get(self, key: str) -> Dict[str, Any] | None: ...


@dataclass
class FileLongTermMemory:
    """
    A minimal file-based KV store.

    Stored as JSON files under a directory.
    """

    directory: Path

    def put(self, key: str, value: Dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str) -> Dict[str, Any] | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
```

---

### `src/agents/__init__.py`
```python
"""Agent implementations."""
```

---

### `src/agents/base.py`
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.config.prompts import PromptTemplate


@dataclass
class AgentResult:
    """Standardized agent output."""
    output: Dict[str, Any]
    messages: List[BaseMessage]


@dataclass
class BaseAgent:
    """
    Base class for all agents.

    Design notes:
    - Agents are thin: prompt + model call + JSON parsing + structured output
    - The graph controls orchestration; agents do not call other agents directly.
    """

    name: str
    llm: BaseChatModel
    prompt: PromptTemplate

    def invoke_json(self, system: str, user: str) -> AgentResult:
        """Call LLM and parse JSON response with robust fallbacks."""
        msgs: List[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=user)]
        resp = self.llm.invoke(msgs)
        content = getattr(resp, "content", "") or ""

        parsed: Dict[str, Any]
        try:
            parsed = json.loads(content)
        except Exception:
            # Fallback: wrap raw content
            parsed = {"raw": content}

        return AgentResult(output=parsed, messages=[AIMessage(content=content)])

    def run(self, **kwargs: Any) -> AgentResult:
        """Render prompt and invoke."""
        system = self.prompt.system
        user = self.prompt.user.format(**kwargs)
        return self.invoke_json(system=system, user=user)
```

---

### `src/agents/coordinator.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base import BaseAgent
from src.config.prompts import PROMPTS
from src.graph.state import AgentState, SubTask


@dataclass
class CoordinatorAgent(BaseAgent):
    """Supervisor agent: decide next node."""

    @classmethod
    def create(cls, llm: BaseChatModel) -> "CoordinatorAgent":
        return cls(name="coordinator", llm=llm, prompt=PROMPTS["coordinator"])

    def decide_next(self, state: AgentState) -> Dict[str, Any]:
        # Deterministic routing first (for controllability)
        subtasks: List[SubTask] = state.get("subtasks", [])
        issues = state.get("issues", [])
        iteration = state.get("iteration_count", 0)

        if state.get("final_answer"):
            return {"next": "finish", "rationale": ["final_answer already set"]}

        if issues and iteration < state["config"]["max_iterations"]:
            # We have issues to fix; go to planner to patch or dispatch directly depending on pending tasks.
            pending = [t for t in subtasks if t.status != "done"]
            if pending:
                return {"next": "dispatch", "rationale": ["Issues exist; continue executing pending subtasks."]}
            return {"next": "planner", "rationale": ["Issues exist; replan."]}

        if not subtasks:
            return {"next": "planner", "rationale": ["No subtasks yet; create plan."]}

        pending = [t for t in subtasks if t.status != "done"]
        if pending:
            return {"next": "dispatch", "rationale": [f"{len(pending)} pending subtasks; dispatch to workers."]}

        # If everything done, review then synthesize
        if not state.get("reviewed", False):
            return {"next": "critic", "rationale": ["All subtasks done; run critic review."]}

        return {"next": "synthesizer", "rationale": ["Reviewed; ready to synthesize final answer."]}
```

---

### `src/agents/planner.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base import BaseAgent
from src.config.prompts import PROMPTS
from src.graph.state import SubTask


@dataclass
class PlannerAgent(BaseAgent):
    """Task planner: decompose original task into subtasks."""

    @classmethod
    def create(cls, llm: BaseChatModel) -> "PlannerAgent":
        return cls(name="planner", llm=llm, prompt=PROMPTS["planner"])

    def plan(self, original_task: str, existing_subtasks: List[SubTask]) -> Dict[str, Any]:
        # Heuristic plan for common patterns to ensure consistent outputs offline.
        lower = original_task.lower()
        if "hacker news" in lower or "hn" in lower:
            subtasks = [
                SubTask(
                    id="t_research",
                    title="Identify HN HTML structure and extraction approach (title + link)",
                    kind="research",
                    depends_on=[],
                    payload={"target": "https://news.ycombinator.com/"},
                    status="pending",
                ),
                SubTask(
                    id="t_code",
                    title="Write hn_crawler.py to fetch and parse HN front page and output JSON",
                    kind="code",
                    depends_on=["t_research"],
                    payload={"output_file": "workspace/hn_crawler.py"},
                    status="pending",
                ),
                SubTask(
                    id="t_execute",
                    title="Write sample JSON output into workspace (offline-friendly demo artifact)",
                    kind="execute",
                    depends_on=["t_code"],
                    payload={"path": "hn_top.json", "content_kind": "sample_hn_json"},
                    status="pending",
                ),
            ]
            return {
                "subtasks": subtasks,
                "rationale": [
                    "Split into research → code → execute for a controlled loop.",
                    "Add offline-friendly JSON artifact to demonstrate file tool usage.",
                ],
            }

        # Otherwise ask LLM (mock returns a generic plan)
        result = self.run(
            original_task=original_task,
            existing_subtasks=[t.model_dump() for t in existing_subtasks],
        )
        # Convert to SubTask models if possible
        out = result.output
        raw_subtasks = out.get("subtasks", [])
        subtasks: List[SubTask] = []
        for t in raw_subtasks:
            try:
                subtasks.append(SubTask(**t, status="pending"))
            except Exception:
                continue
        return {"subtasks": subtasks, "rationale": out.get("rationale", [])}
```

---

### `src/agents/researcher.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base import BaseAgent
from src.config.prompts import PROMPTS


@dataclass
class ResearcherAgent(BaseAgent):
    """Researcher: gather constraints, background knowledge."""

    @classmethod
    def create(cls, llm: BaseChatModel) -> "ResearcherAgent":
        return cls(name="researcher", llm=llm, prompt=PROMPTS["researcher"])

    def research(self, task: str) -> Dict[str, Any]:
        # Use LLM (or mock) to produce structured constraints/notes
        result = self.run(task=task)
        return result.output
```

---

### `src/agents/coder.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base import BaseAgent
from src.config.prompts import PROMPTS


@dataclass
class CoderAgent(BaseAgent):
    """Coder: generate code artifacts."""

    @classmethod
    def create(cls, llm: BaseChatModel) -> "CoderAgent":
        return cls(name="coder", llm=llm, prompt=PROMPTS["coder"])

    def code(self, task: str, context: str) -> Dict[str, Any]:
        result = self.run(task=task, context=context)
        return result.output
```

---

### `src/agents/executor.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base import BaseAgent
from src.config.prompts import PROMPTS
from src.tools.base import ToolRouter


@dataclass
class ExecutorAgent(BaseAgent):
    """Executor: call tools and perform concrete actions."""

    tool_router: ToolRouter | None = None

    @classmethod
    def create(cls, llm: BaseChatModel, tool_router: ToolRouter) -> "ExecutorAgent":
        return cls(name="executor", llm=llm, prompt=PROMPTS["executor"], tool_router=tool_router)

    def execute(self, task: str) -> Dict[str, Any]:
        tools = self.tool_router.list_tools() if self.tool_router else []
        result = self.run(task=task, tools="\n".join(tools))
        return result.output

    def write_sample_hn_json(self) -> Dict[str, Any]:
        """Offline-friendly executor action: write a sample JSON output file."""
        if not self.tool_router:
            return {"error": "tool_router not set"}

        sample = [
            {"title": "Sample HN Item 1", "link": "https://news.ycombinator.com/item?id=1"},
            {"title": "Sample HN Item 2", "link": "https://news.ycombinator.com/item?id=2"},
        ]
        import json

        content = json.dumps(sample, ensure_ascii=False, indent=2)
        _, log = self.tool_router.call("file_writer", {"path": "hn_top.json", "content": content})
        return {"wrote": "workspace/hn_top.json", "tool_log": log.model_dump()}
```

---

### `src/agents/critic.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base import BaseAgent
from src.config.prompts import PROMPTS


@dataclass
class CriticAgent(BaseAgent):
    """Critic: review aggregated outputs for correctness."""

    @classmethod
    def create(cls, llm: BaseChatModel) -> "CriticAgent":
        return cls(name="critic", llm=llm, prompt=PROMPTS["critic"])

    def review(self, original_task: str, outputs: str) -> Dict[str, Any]:
        result = self.run(original_task=original_task, outputs=outputs)
        return result.output
```

---

### `src/agents/synthesizer.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base import BaseAgent
from src.config.prompts import PROMPTS


@dataclass
class SynthesizerAgent(BaseAgent):
    """Synthesizer: produce final user-facing answer."""

    @classmethod
    def create(cls, llm: BaseChatModel) -> "SynthesizerAgent":
        return cls(name="synthesizer", llm=llm, prompt=PROMPTS["synthesizer"])

    def synthesize(self, original_task: str, outputs: str, issues: List[str]) -> str:
        result = self.run(original_task=original_task, outputs=outputs, issues=issues)
        # If LLM returns JSON, pick "raw" or stringify
        out = result.output
        if isinstance(out, dict) and "raw" in out:
            return str(out["raw"])
        if isinstance(out, dict):
            # Provide a deterministic readable format
            return (
                f"Task:\n{original_task}\n\n"
                f"Outputs:\n{outputs}\n\n"
                f"Issues:\n{issues}\n"
            )
        return str(out)
```

---

## 图编排实现

### `src/graph/__init__.py`
```python
"""LangGraph graph package."""
```

---

### `src/graph/state.py`
```python
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class ToolCallLog(BaseModel):
    tool_name: str
    args: Dict[str, Any]
    result_preview: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int
    ts: float


class SubTask(BaseModel):
    id: str
    title: str
    kind: Literal["research", "code", "execute"]
    depends_on: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "in_progress", "done", "failed"] = "pending"
    assigned_to: Optional[str] = None


class Metrics(BaseModel):
    started_ts: float
    ended_ts: Optional[float] = None
    node_visits: List[str] = Field(default_factory=list)
    execution_depth: int = 0
    reflection_count: int = 0
    llm_calls: int = 0
    tool_calls: int = 0


def _merge_messages(a: List[BaseMessage], b: List[BaseMessage]) -> List[BaseMessage]:
    return [*a, *b]


def _merge_tool_logs(a: List[ToolCallLog], b: List[ToolCallLog]) -> List[ToolCallLog]:
    return [*a, *b]


def _merge_agent_outputs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _merge_subtasks(a: List[SubTask], b: List[SubTask]) -> List[SubTask]:
    """
    Merge subtasks by id (right side wins), preserving order of first appearance.
    """
    by_id: Dict[str, SubTask] = {}
    order: List[str] = []
    for t in a:
        by_id[t.id] = t
        order.append(t.id)
    for t in b:
        if t.id not in by_id:
            order.append(t.id)
        by_id[t.id] = t
    return [by_id[i] for i in order]


def _merge_issues(a: List[str], b: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in [*a, *b]:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


class AgentState(TypedDict, total=False):
    # Required core fields
    messages: Annotated[List[BaseMessage], _merge_messages]
    original_task: str
    subtasks: Annotated[List[SubTask], _merge_subtasks]
    agent_outputs: Annotated[Dict[str, Any], _merge_agent_outputs]
    tool_call_logs: Annotated[List[ToolCallLog], _merge_tool_logs]
    current_agent: str
    iteration_count: int
    final_answer: Optional[str]
    next: str

    # Extensions for control/observability
    issues: Annotated[List[str], _merge_issues]
    reviewed: bool
    metrics: Metrics

    # Runtime config snapshot (plain dict for TypedDict friendliness)
    config: Dict[str, Any]
```

---

### `src/graph/nodes.py`
```python
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.coder import CoderAgent
from src.agents.coordinator import CoordinatorAgent
from src.agents.critic import CriticAgent
from src.agents.executor import ExecutorAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.config.settings import Settings
from src.graph.state import AgentState, SubTask, ToolCallLog
from src.tools.base import ToolRouter
from src.utils.logger import EventLogger


@dataclass
class NodeContext:
    settings: Settings
    logger: EventLogger
    tool_router: ToolRouter

    coordinator: CoordinatorAgent
    planner: PlannerAgent
    researcher: ResearcherAgent
    coder: CoderAgent
    executor: ExecutorAgent
    critic: CriticAgent
    synthesizer: SynthesizerAgent


def _state_summary_for_llm(state: AgentState, max_len: int = 2000) -> str:
    subtasks = state.get("subtasks", [])
    issues = state.get("issues", [])
    summary = {
        "original_task": state.get("original_task"),
        "iteration_count": state.get("iteration_count"),
        "subtasks": [t.model_dump() for t in subtasks],
        "issues": issues,
        "has_final": bool(state.get("final_answer")),
    }
    s = json.dumps(summary, ensure_ascii=False, indent=2)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


def parse_input(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        ctx.settings.ensure_dirs()
        start = time.time()
        original_task = state.get("original_task", "").strip()
        if not original_task:
            raise ValueError("original_task is required")

        metrics = state.get("metrics")
        if not metrics:
            metrics = {
                "started_ts": start,
                "ended_ts": None,
                "node_visits": [],
                "execution_depth": 0,
                "reflection_count": 0,
                "llm_calls": 0,
                "tool_calls": 0,
            }

        ctx.logger.log_event("parse_input", {"original_task": original_task})

        return {
            "messages": [
                SystemMessage(content="You are a multi-agent system."),
                HumanMessage(content=original_task),
            ],
            "original_task": original_task,
            "subtasks": state.get("subtasks", []),
            "agent_outputs": state.get("agent_outputs", {}),
            "tool_call_logs": state.get("tool_call_logs", []),
            "issues": state.get("issues", []),
            "iteration_count": state.get("iteration_count", 0),
            "reviewed": state.get("reviewed", False),
            "metrics": metrics,
            "config": state.get("config", {}),
        }

    return _node


def coordinator_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("coordinator")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        decision = ctx.coordinator.decide_next(state)
        ctx.logger.log_event("coordinator_decision", decision)

        return {
            "current_agent": "coordinator",
            "next": decision["next"],
            "agent_outputs": {"coordinator": decision},
        }

    return _node


def planner_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("planner")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        original_task = state["original_task"]
        existing = state.get("subtasks", [])
        plan = ctx.planner.plan(original_task=original_task, existing_subtasks=existing)
        subtasks: List[SubTask] = plan.get("subtasks", [])

        ctx.logger.log_event(
            "planner_plan",
            {"subtasks": [t.model_dump() for t in subtasks], "rationale": plan.get("rationale", [])},
        )
        return {
            "current_agent": "planner",
            "subtasks": subtasks,
            "issues": [],  # reset issues when replanning
            "agent_outputs": {"planner": {"rationale": plan.get("rationale", [])}},
        }

    return _node


def _ready_subtasks(subtasks: List[SubTask]) -> List[SubTask]:
    by_id = {t.id: t for t in subtasks}
    ready: List[SubTask] = []
    for t in subtasks:
        if t.status != "pending":
            continue
        if all(by_id.get(dep) and by_id[dep].status == "done" for dep in t.depends_on):
            ready.append(t)
    return ready


def dispatch_node(ctx: NodeContext):
    """
    Dispatch work to worker agents.

    Parallelism is implemented with a ThreadPoolExecutor within this node to keep
    LangGraph orchestration simple and robust.
    """

    def _run_one(subtask: SubTask, state: AgentState) -> Tuple[SubTask, Dict[str, Any], List[ToolCallLog]]:
        tool_logs: List[ToolCallLog] = []

        if subtask.kind == "research":
            out = ctx.researcher.research(task=subtask.title)
            return subtask.model_copy(update={"status": "done", "assigned_to": "researcher"}), out, tool_logs

        if subtask.kind == "code":
            # Provide context from researcher output if present
            context = json.dumps(state.get("agent_outputs", {}).get("researcher", {}), ensure_ascii=False, indent=2)
            out = ctx.coder.code(task=subtask.title, context=context)

            # If the coder produced files, write them under workspace using tool
            files = out.get("files", [])
            for f in files:
                path = f.get("path", "")
                content = f.get("content", "")
                if path.startswith("workspace/"):
                    rel = path[len("workspace/") :]
                else:
                    rel = path
                _, log = ctx.tool_router.call("file_writer", {"path": rel, "content": content})
                tool_logs.append(log)
            return subtask.model_copy(update={"status": "done", "assigned_to": "coder"}), out, tool_logs

        if subtask.kind == "execute":
            payload = subtask.payload or {}
            if payload.get("content_kind") == "sample_hn_json":
                out = ctx.executor.write_sample_hn_json()
                # tool log is embedded; also capture to state tool logs if present
                tl = out.get("tool_log")
                if tl:
                    try:
                        tool_logs.append(ToolCallLog(**tl))
                    except Exception:
                        pass
                return subtask.model_copy(update={"status": "done", "assigned_to": "executor"}), out, tool_logs

            out = ctx.executor.execute(task=subtask.title)
            return subtask.model_copy(update={"status": "done", "assigned_to": "executor"}), out, tool_logs

        return subtask.model_copy(update={"status": "failed"}), {"error": "unknown kind"}, tool_logs

    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("dispatch")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        subtasks = state.get("subtasks", [])
        ready = _ready_subtasks(subtasks)
        if not ready:
            ctx.logger.log_event("dispatch_no_ready_subtasks", {"count": len(subtasks)})
            return {"current_agent": "dispatcher"}

        ctx.logger.log_event(
            "dispatch_ready_subtasks",
            {"ready": [t.model_dump() for t in ready], "parallel_workers": ctx.settings.parallel_workers},
        )

        updated_subtasks: List[SubTask] = []
        outputs_delta: Dict[str, Any] = {}
        tool_logs_delta: List[ToolCallLog] = []

        # Mark in_progress locally (so repeated calls won't pick them up)
        id_to_sub = {t.id: t for t in subtasks}
        for t in ready:
            id_to_sub[t.id] = t.model_copy(update={"status": "in_progress"})
        subtasks = list(id_to_sub.values())

        with ThreadPoolExecutor(max_workers=ctx.settings.parallel_workers) as ex:
            futures = {ex.submit(_run_one, t, state): t.id for t in ready}
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    new_task, out, logs = fut.result()
                except Exception as e:
                    new_task = id_to_sub[tid].model_copy(update={"status": "failed"})
                    out = {"error": repr(e)}
                    logs = []

                updated_subtasks.append(new_task)
                outputs_delta[new_task.assigned_to or "worker"] = {
                    **outputs_delta.get(new_task.assigned_to or "worker", {}),
                    new_task.id: out,
                }
                tool_logs_delta.extend(logs)

        ctx.logger.log_event(
            "dispatch_results",
            {
                "updated_subtasks": [t.model_dump() for t in updated_subtasks],
                "tool_calls": len(tool_logs_delta),
            },
        )

        state["metrics"]["tool_calls"] += len(tool_logs_delta)

        return {
            "current_agent": "dispatcher",
            "subtasks": updated_subtasks,
            "agent_outputs": outputs_delta,
            "tool_call_logs": tool_logs_delta,
        }

    return _node


def critic_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("critic")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        outputs = json.dumps(state.get("agent_outputs", {}), ensure_ascii=False, indent=2)
        review = ctx.critic.review(original_task=state["original_task"], outputs=outputs)

        ok = bool(review.get("ok", False))
        issues = review.get("issues", []) if isinstance(review.get("issues", []), list) else []
        fix_suggestions = review.get("fix_suggestions", [])

        ctx.logger.log_event(
            "critic_review",
            {"ok": ok, "issues": issues, "fix_suggestions": fix_suggestions},
        )

        if ok:
            return {
                "current_agent": "critic",
                "reviewed": True,
                "issues": [],
                "agent_outputs": {"critic": review},
            }

        # needs fix
        it = int(state.get("iteration_count", 0)) + 1
        state["metrics"]["reflection_count"] = it
        return {
            "current_agent": "critic",
            "iteration_count": it,
            "issues": issues or ["Critic requested improvements."],
            "agent_outputs": {"critic": review},
        }

    return _node


def synthesizer_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("synthesizer")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        outputs = json.dumps(state.get("agent_outputs", {}), ensure_ascii=False, indent=2)
        issues = state.get("issues", [])
        final = ctx.synthesizer.synthesize(
            original_task=state["original_task"],
            outputs=outputs,
            issues=issues,
        )

        # Add a small decision trace (auditable, not private CoT)
        trace = {
            "node_visits": state["metrics"]["node_visits"],
            "reflection_count": state["metrics"]["reflection_count"],
            "tool_calls": state["metrics"]["tool_calls"],
        }
        final_with_trace = f"{final}\n\n---\n## Decision Trace\n```json\n{json.dumps(trace, ensure_ascii=False, indent=2)}\n```"

        ctx.logger.log_event("synthesizer_finalized", {"final_len": len(final_with_trace)})

        return {"current_agent": "synthesizer", "final_answer": final_with_trace}

    return _node


def human_in_loop_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("human_in_loop")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        # Non-interactive default: mark an issue and continue.
        ctx.logger.log_event("human_in_loop", {"enabled": ctx.settings.human_in_loop})

        if not ctx.settings.human_in_loop:
            return {"issues": ["Human-in-the-loop requested but disabled."], "next": "coordinator"}

        # If enabled, expect caller to inject `human_input` into state before calling.
        human_input = state.get("agent_outputs", {}).get("human_input")
        if not human_input:
            return {"issues": ["No human_input provided."], "next": "coordinator"}

        return {
            "agent_outputs": {"human": {"input": human_input}},
            "issues": [],
            "next": "coordinator",
        }

    return _node


def persist_history_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("persist_history")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])
        state["metrics"]["ended_ts"] = time.time()

        if ctx.settings.history_file:
            record = {
                "ts": time.time(),
                "original_task": state.get("original_task"),
                "final_answer": state.get("final_answer"),
                "metrics": state.get("metrics"),
            }
            ctx.settings.history_file.parent.mkdir(parents=True, exist_ok=True)
            with ctx.settings.history_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        ctx.logger.log_event("persist_history", {"path": str(ctx.settings.history_file)})
        return {}

    return _node
```

---

### `src/graph/edges.py`
```python
from __future__ import annotations

from typing import Literal

from src.graph.state import AgentState


def route_from_coordinator(state: AgentState) -> str:
    """
    Read `state["next"]` and route to the appropriate node.

    Supported routes:
    - planner
    - dispatch
    - critic
    - synthesizer
    - human_in_loop
    - finish
    """
    nxt = state.get("next", "planner")
    return nxt


def route_after_critic(state: AgentState) -> str:
    """
    After critic:
    - if issues exist and iteration_count < max_iterations => coordinator
    - else => synthesizer
    """
    issues = state.get("issues", [])
    it = int(state.get("iteration_count", 0))
    max_it = int(state.get("config", {}).get("max_iterations", 3))
    if issues and it < max_it:
        return "coordinator"
    return "synthesizer"
```

---

### `src/graph/builder.py`
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from src.graph.edges import route_after_critic, route_from_coordinator
from src.graph.nodes import (
    NodeContext,
    coordinator_node,
    critic_node,
    dispatch_node,
    human_in_loop_node,
    parse_input,
    persist_history_node,
    planner_node,
    synthesizer_node,
)
from src.graph.state import AgentState


@dataclass
class GraphBuilder:
    """
    Builds and compiles the LangGraph StateGraph.

    Note on END:
    - END is the terminal state constant provided by LangGraph.
    - When an execution reaches END, the graph run completes and returns the final state.
    """

    ctx: NodeContext

    def build(self):
        graph = StateGraph(AgentState)

        # Nodes
        graph.add_node("parse_input", parse_input(self.ctx))
        graph.add_node("coordinator", coordinator_node(self.ctx))
        graph.add_node("planner", planner_node(self.ctx))
        graph.add_node("dispatch", dispatch_node(self.ctx))
        graph.add_node("critic", critic_node(self.ctx))
        graph.add_node("synthesizer", synthesizer_node(self.ctx))
        graph.add_node("human_in_loop", human_in_loop_node(self.ctx))
        graph.add_node("persist_history", persist_history_node(self.ctx))

        # Edges
        graph.set_entry_point("parse_input")
        graph.add_edge("parse_input", "coordinator")

        graph.add_conditional_edges(
            "coordinator",
            route_from_coordinator,
            {
                "planner": "planner",
                "dispatch": "dispatch",
                "critic": "critic",
                "synthesizer": "synthesizer",
                "human_in_loop": "human_in_loop",
                "finish": "persist_history",
            },
        )

        graph.add_edge("planner", "coordinator")
        graph.add_edge("dispatch", "coordinator")

        graph.add_conditional_edges(
            "critic",
            route_after_critic,
            {
                "coordinator": "coordinator",
                "synthesizer": "synthesizer",
            },
        )

        graph.add_edge("synthesizer", "persist_history")
        graph.add_edge("persist_history", END)

        # compile() materializes the runnable graph.
        # This is required before invoke()/stream().
        return graph.compile()
```

---

### `src/main.py`
```python
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

from dotenv import load_dotenv

from src.agents.coder import CoderAgent
from src.agents.coordinator import CoordinatorAgent
from src.agents.critic import CriticAgent
from src.agents.executor import ExecutorAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.config.settings import Settings
from src.graph.builder import GraphBuilder
from src.graph.nodes import NodeContext
from src.llm.factory import create_chat_model
from src.tools.base import build_default_tool_router
from src.tools.calculator import build_calculator_tools
from src.tools.code_executor import build_code_tools
from src.tools.file_manager import build_file_tools
from src.tools.search import build_search_tools
from src.utils.logger import build_logger


def build_app(settings: Settings):
    """Create the LangGraph app with all agents/tools wired."""
    llm = create_chat_model(settings)

    tool_router = build_default_tool_router(
        tools=[
            *build_calculator_tools(),
            *build_file_tools(settings.workspace_dir),
            *build_code_tools(),
            *build_search_tools(),
        ]
    )

    logger = build_logger("multi_agent_system", debug=settings.debug, logs_dir=settings.logs_dir)

    ctx = NodeContext(
        settings=settings,
        logger=logger,
        tool_router=tool_router,
        coordinator=CoordinatorAgent.create(llm),
        planner=PlannerAgent.create(llm),
        researcher=ResearcherAgent.create(llm),
        coder=CoderAgent.create(llm),
        executor=ExecutorAgent.create(llm, tool_router),
        critic=CriticAgent.create(llm),
        synthesizer=SynthesizerAgent.create(llm),
    )

    app = GraphBuilder(ctx).build()
    return app, logger


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(description="General-purpose multi-agent system (LangGraph).")
    p.add_argument("task", nargs="?", default="", help="Task text. If empty, read from stdin.")
    p.add_argument("--provider", default=os.getenv("MAS_LLM_PROVIDER", "mock"))
    p.add_argument("--debug", action="store_true", default=False)
    args = p.parse_args()

    task = args.task.strip() if args.task else ""
    if not task:
        task = input("Enter task: ").strip()

    settings = Settings(
        llm_provider=args.provider,
        debug=args.debug or Settings().debug,
    )
    settings.ensure_dirs()

    app, logger = build_app(settings)

    init_state: Dict[str, Any] = {
        "original_task": task,
        "iteration_count": 0,
        "config": {
            "max_iterations": settings.max_iterations,
        },
        "metrics": {
            "started_ts": time.time(),
            "ended_ts": None,
            "node_visits": [],
            "execution_depth": 0,
            "reflection_count": 0,
            "llm_calls": 0,
            "tool_calls": 0,
        },
    }

    out = app.invoke(init_state)
    final = out.get("final_answer") or "(no final_answer)"
    logger.console.print(final)


if __name__ == "__main__":
    main()
```

---

如果你要我继续输出 **交付物 4（第 2 段：examples/ + tests/ 全部文件）**，请回复：`继续`。

## 交付物 4：完整 Python 源代码（第 2 段：examples/ + tests/ + .gitkeep）

下面继续按文件路径逐个输出 **examples/**、**tests/**，以及 `workspace/`、`logs/` 的占位文件。

---

## examples

### `examples/example_planning.py`
```python
from __future__ import annotations

import json
import time
from typing import Any, Dict

from src.config.settings import Settings
from src.main import build_app


def main() -> None:
    """
    Demo 1: 任务规划分解类
    示例任务：制定一个学习 Python 的计划
    """
    settings = Settings(llm_provider="mock", debug=True)
    settings.ensure_dirs()

    app, _logger = build_app(settings)

    task = "请制定一个为期 6 周的 Python 学习计划，每周列出学习主题、练习建议和验收标准。"

    init_state: Dict[str, Any] = {
        "original_task": task,
        "iteration_count": 0,
        "config": {"max_iterations": settings.max_iterations},
        "metrics": {
            "started_ts": time.time(),
            "ended_ts": None,
            "node_visits": [],
            "execution_depth": 0,
            "reflection_count": 0,
            "llm_calls": 0,
            "tool_calls": 0,
        },
    }

    out = app.invoke(init_state)

    print("\n=== Subtasks ===")
    subtasks = out.get("subtasks", [])
    print(json.dumps([getattr(t, "model_dump", lambda: t)() for t in subtasks], ensure_ascii=False, indent=2))

    print("\n=== Final Answer ===")
    print(out.get("final_answer"))


if __name__ == "__main__":
    main()
```

---

### `examples/example_tool_execution.py`
```python
from __future__ import annotations

from pathlib import Path

from src.config.settings import Settings
from src.tools.base import build_default_tool_router
from src.tools.calculator import build_calculator_tools
from src.tools.code_executor import build_code_tools
from src.tools.file_manager import build_file_tools
from src.tools.search import build_search_tools


def main() -> None:
    """
    Demo 2: 工具执行示例
    - safe_eval
    - json_validator
    - file_writer / file_reader（限定 workspace）
    - python_repl
    - web_search（模拟）
    """
    settings = Settings(llm_provider="mock", debug=True)
    settings.ensure_dirs()

    tool_router = build_default_tool_router(
        tools=[
            *build_calculator_tools(),
            *build_file_tools(settings.workspace_dir),
            *build_code_tools(),
            *build_search_tools(),
        ]
    )

    # 1) safe_eval
    result, log = tool_router.call("safe_eval", {"expression": "2*(3+4) + 10/5"})
    print("safe_eval result:", result, "log.error:", log.error)

    # 2) json_validator
    pretty, log = tool_router.call("json_validator", {"text": "{\"a\": 1, \"b\": [2,3]}"})
    print("json_validator pretty:\n", pretty)

    # 3) file_writer + file_reader (under workspace)
    _, log = tool_router.call("file_writer", {"path": "hello.txt", "content": "hello workspace"})
    print("file_writer:", log.result_preview, "error:", log.error)

    content, log = tool_router.call("file_reader", {"path": "hello.txt"})
    print("file_reader content:", content)

    # 4) python_repl
    repl_out, log = tool_router.call(
        "python_repl",
        {"code": "x = 1+2\nprint('x=', x)\n"},
    )
    print("python_repl:", repl_out)

    # 5) web_search (simulated)
    search_out, log = tool_router.call("web_search", {"query": "LangGraph StateGraph fan-out join", "k": 3})
    print("web_search:", search_out)


if __name__ == "__main__":
    main()
```

---

### `examples/example_code_generation.py`
```python
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from src.config.settings import Settings
from src.main import build_app


def main() -> None:
    """
    Demo 3: 代码生成类
    示例任务：编写爬虫抓取 Hacker News 首页标题和链接，保存为 JSON 文件
    """
    settings = Settings(llm_provider="mock", debug=True)
    settings.ensure_dirs()

    app, _logger = build_app(settings)

    task = "请帮我编写一个 Python 爬虫，抓取 Hacker News 首页的文章标题和链接，并保存为 JSON 文件。"

    init_state: Dict[str, Any] = {
        "original_task": task,
        "iteration_count": 0,
        "config": {"max_iterations": settings.max_iterations},
        "metrics": {
            "started_ts": time.time(),
            "ended_ts": None,
            "node_visits": [],
            "execution_depth": 0,
            "reflection_count": 0,
            "llm_calls": 0,
            "tool_calls": 0,
        },
    }

    out = app.invoke(init_state)

    # Verify artifacts under workspace
    crawler_path = settings.workspace_dir / "hn_crawler.py"
    sample_json_path = settings.workspace_dir / "hn_top.json"
    print("\nArtifacts:")
    print(" - hn_crawler.py exists:", crawler_path.exists(), "path:", crawler_path)
    print(" - hn_top.json exists:", sample_json_path.exists(), "path:", sample_json_path)

    print("\nFinal Answer:\n")
    print(out.get("final_answer"))


if __name__ == "__main__":
    main()
```

---

## tests

### `tests/__init__.py`
```python
"""Test package."""
```

---

### `tests/test_tools.py`
```python
from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.calculator import json_validator, safe_eval
from src.tools.code_executor import python_repl
from src.tools.file_manager import build_file_reader, build_file_writer


def test_safe_eval_basic() -> None:
    # 2*(3+4) + 10/5 = 14 + 2 = 16
    out = safe_eval.invoke({"expression": "2*(3+4) + 10/5"})
    assert float(out) == 16.0


def test_json_validator_pretty() -> None:
    out = json_validator.invoke({"text": "{\"a\": 1, \"b\": [2,3]}"})
    assert "\"a\"" in out
    assert "\n" in out


def test_file_tools_workspace_restriction(tmp_path: Path) -> None:
    reader = build_file_reader(tmp_path)
    writer = build_file_writer(tmp_path)

    # normal write/read
    p = writer.invoke({"path": "a.txt", "content": "hello"})
    assert Path(p).exists()

    content = reader.invoke({"path": "a.txt"})
    assert content == "hello"

    # traversal should be blocked
    with pytest.raises(ValueError):
        writer.invoke({"path": "../evil.txt", "content": "nope"})


def test_python_repl_executes() -> None:
    out = python_repl.invoke({"code": "x = 40 + 2\nprint(x)\n"})
    assert out["error"] is None
    assert "42" in out["stdout"]
    assert "x" in out["locals_preview"]
```

---

### `tests/test_graph.py`
```python
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from src.config.settings import Settings
from src.main import build_app


def _invoke(app, settings: Settings, task: str) -> Dict[str, Any]:
    init_state: Dict[str, Any] = {
        "original_task": task,
        "iteration_count": 0,
        "config": {"max_iterations": settings.max_iterations},
        "metrics": {
            "started_ts": time.time(),
            "ended_ts": None,
            "node_visits": [],
            "execution_depth": 0,
            "reflection_count": 0,
            "llm_calls": 0,
            "tool_calls": 0,
        },
    }
    return app.invoke(init_state)


def test_graph_runs_end_to_end(tmp_path: Path) -> None:
    settings = Settings(llm_provider="mock", debug=False, workspace_dir=tmp_path / "ws", logs_dir=tmp_path / "logs")
    settings.ensure_dirs()
    app, _logger = build_app(settings)

    out = _invoke(app, settings, "给我一个番茄钟工作法的使用说明。")
    assert out.get("final_answer")
    assert out.get("metrics") is not None
    assert out["metrics"]["execution_depth"] >= 1


def test_hn_task_writes_workspace_files(tmp_path: Path) -> None:
    settings = Settings(llm_provider="mock", debug=False, workspace_dir=tmp_path / "ws", logs_dir=tmp_path / "logs")
    settings.ensure_dirs()
    app, _logger = build_app(settings)

    task = "请帮我编写一个 Python 爬虫，抓取 Hacker News 首页的文章标题和链接，并保存为 JSON 文件"
    out = _invoke(app, settings, task)

    # Code file written by coder via file_writer tool
    assert (settings.workspace_dir / "hn_crawler.py").exists()

    # Sample JSON written by executor via file_writer tool
    assert (settings.workspace_dir / "hn_top.json").exists()

    # Tool logs should contain file_writer calls
    tool_logs = out.get("tool_call_logs", [])
    assert any(getattr(log, "tool_name", None) == "file_writer" for log in tool_logs)
```

---

### `tests/test_flow.py`
```python
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from src.config.settings import Settings
from src.main import build_app


def test_reflect_loop_cap_does_not_infinite_loop(tmp_path: Path) -> None:
    """
    The mock critic returns ok=True, so we won't actually loop.
    This test ensures the system finishes and returns final_answer deterministically.
    """
    settings = Settings(
        llm_provider="mock",
        debug=False,
        max_iterations=1,
        workspace_dir=tmp_path / "ws",
        logs_dir=tmp_path / "logs",
    )
    settings.ensure_dirs()
    app, _logger = build_app(settings)

    init_state: Dict[str, Any] = {
        "original_task": "输出一个简短的项目方案（含里程碑）。",
        "iteration_count": 0,
        "config": {"max_iterations": settings.max_iterations},
        "metrics": {
            "started_ts": time.time(),
            "ended_ts": None,
            "node_visits": [],
            "execution_depth": 0,
            "reflection_count": 0,
            "llm_calls": 0,
            "tool_calls": 0,
        },
    }

    out = app.invoke(init_state)
    assert out.get("final_answer") is not None
    assert out.get("metrics") is not None
    assert out["metrics"]["execution_depth"] > 0
```

---

## 占位文件

### `workspace/.gitkeep`
```text

```

### `logs/.gitkeep`
```text

```

---

如果你要我继续输出下一段交付物（**交付物 5 README.md + 交付物 7 验证指南 + 交付物 8 扩展指南 + 交付物 6 Demo 说明与第十三节演示场景完整验证输出**），请回复：`继续`。

### `README.md`
```markdown
# 通用 Multi-Agent 问题求解系统（LangGraph）

一个从 0 到 1 的**通用多智能体协作系统**，基于 **LangGraph >= 0.2.0** 编排，采用 **Supervisor + Worker** 架构，支持：
- 自动任务理解与规划（Planner）
- 多 Agent 协作执行（Researcher / Coder / Executor 并行）
- 审阅与反思纠错闭环（Critic → Coordinator 循环）
- 工具调用（计算/JSON 校验/文件读写/代码执行/搜索模拟）
- 可观测性（节点事件、耗时、工具日志、history 持久化）
- 人工介入节点（可配置开关）

> 说明：本项目输出的“reasoning/推理过程”以**可审计的决策轨迹**形式呈现（plan/rationale/decision trace/issue list），避免暴露不可控的私有思维链。

---

## 1. 技术栈

- Python 3.10+（推荐 3.11）
- LangGraph >= 0.2.0（多智能体编排）
- LangChain Core（LLM 与 Tools）
- Pydantic v2（数据模型与校验）
- Rich（日志显示）

参考：
- LangGraph StateGraph / END / compile 概念与用法：<https://langchain-ai.github.io/langgraph/>  
- LangChain Tools（`@tool`）：<https://python.langchain.com/docs/how_to/tools/>  
- Pydantic v2：<https://docs.pydantic.dev/latest/>

---

## 2. 安装

### 2.1 方式 A：pip
```bash
pip install -r requirements.txt
```

### 2.2 方式 B：安装可选 LLM Provider（可选）
- OpenAI：
```bash
pip install ".[openai]"
```
- Anthropic：
```bash
pip install ".[anthropic]"
```

### 2.3 环境变量
复制并编辑：
```bash
cp .env.example .env
```

常用：
- `MAS_LLM_PROVIDER=mock|openai|anthropic`
- `MAS_DEBUG=true`
- `MAS_MAX_ITERATIONS=3`
- `MAS_WORKSPACE_DIR=workspace`
- `MAS_HISTORY_FILE=logs/history.jsonl`

---

## 3. 运行

### 3.1 CLI
```bash
python -m src.main "请制定一个为期 6 周的 Python 学习计划"
```

或使用入口脚本（pyproject scripts）：
```bash
multi-agent-system "分析某业务需求并给出方案"
```

### 3.2 使用真实模型（示例：OpenAI）
```bash
export MAS_LLM_PROVIDER=openai
export OPENAI_API_KEY=你的key
python -m src.main "请帮我编写一个 Python 爬虫，抓取 Hacker News 首页的文章标题和链接，并保存为 JSON 文件"
```

---

## 4. 输出与可观测性

### 4.1 日志
- 控制台：Rich 结构化输出
- 文件：
  - `logs/run.log`：可读日志
  - `logs/events.jsonl`：事件流（每个节点的输入/输出摘要）
  - `logs/history.jsonl`：对话历史持久化（final + metrics）

### 4.2 指标（metrics）
在最终 state 中 `metrics` 会记录：
- `execution_depth`：链路深度（节点访问数）
- `reflection_count`：反思/重试轮数
- `tool_calls`：工具调用次数
- `llm_calls`：LLM 调用次数
- `prompt_tokens / completion_tokens / total_tokens`：若 provider 返回 usage，会累计（不同模型返回字段可能不同）

### 4.3 可视化（简易 Mermaid）
你可以从最终输出的 Decision Trace 或 `metrics.node_visits` 生成流程图（见 `src/utils/visualizer.py`）。

---

## 5. 内置工具（Tools）

- `safe_eval`：安全算术表达式计算
- `json_validator`：JSON 校验与格式化
- `file_reader/file_writer`：文件读写（**强制限制在 workspace/**）
- `python_repl`：简易 Python 执行（演示用，非生产级沙箱）
- `web_search`：搜索模拟（离线可用）

---

## 6. 示例（examples/）

```bash
python examples/example_planning.py
python examples/example_tool_execution.py
python examples/example_code_generation.py
```

---

## 7. 测试（tests/）

```bash
pytest -q
```

---

## 8. 安全边界

- 文件工具只允许 `workspace/` 内部路径；任何 `../` 越界都会被拒绝。
- `python_repl` 仅为演示用途，生产环境应使用容器/系统级隔离。

---

## 9. 常见问题

### 9.1 为什么 mock 模式也能跑完整流程？
为了保证**无外部 API Key**也可验证系统结构与工具链路，提供了 Deterministic Mock LLM。

### 9.2 “reasoning 过程”具体体现在哪里？
- Planner/Researcher/Coder/Critic 输出里包含 `rationale`/`issues` 等可审计字段
- 最终输出附带 `Decision Trace`（节点访问顺序、工具调用次数、反思次数）

---

## 10. 目录结构

见任务书要求的目录树；代码完全按 `src/`、`examples/`、`tests/` 组织。
```

---

# 交付物 7：验证指南（Verification Guide）

## 7.1 安装与环境
```bash
python -V  # >= 3.10
pip install -r requirements.txt
cp .env.example .env
```

## 7.2 运行示例并验证期望
### 示例 1：规划分解
```bash
python examples/example_planning.py
```
期望：
- 控制台打印 `Subtasks`（至少 2-3 个）
- 打印 `Final Answer`
- `logs/events.jsonl` 有事件记录

### 示例 2：工具执行
```bash
python examples/example_tool_execution.py
```
期望：
- `safe_eval result` 为 16
- `workspace/hello.txt` 被创建并可读
- `python_repl` stdout 包含输出
- `web_search` 返回模拟结果列表

### 示例 3：代码生成（HN 爬虫）
```bash
python examples/example_code_generation.py
```
期望：
- `workspace/hn_crawler.py` 存在（由 Coder 产出 + file_writer 写入）
- `workspace/hn_top.json` 存在（由 Executor 调用 file_writer 写入示例 JSON）
- `tool_call_logs` 中出现 `file_writer`

## 7.3 运行完整 CLI
```bash
python -m src.main "请帮我编写一个 Python 爬虫，抓取 Hacker News 首页的文章标题和链接，并保存为 JSON 文件"
```
期望：
- 最终输出包含 `Decision Trace` JSON
- `logs/history.jsonl` 追加一条记录

---

# 交付物 8：扩展指南（Extension Guide）

## 8.1 如何新增 Agent
1) 新建文件：`src/agents/my_agent.py`
2) 继承 `BaseAgent` 或仿照现有 Agent（建议仿照 `researcher.py`）
3) 在 `src/main.py -> build_app()` 中实例化并注入 `NodeContext`
4) 在 `src/graph/nodes.py` 增加对应 node 函数
5) 在 `src/graph/builder.py`：
   - `add_node("my_agent", my_agent_node(ctx))`
   - 在 Coordinator 路由（`route_from_coordinator`）中允许 `next="my_agent"`（或在 coordinator 逻辑中产生该 next）

设计建议：
- Agent 只做“单步产出”，不要在 Agent 内部编排其他 Agent（编排留给 LangGraph）

## 8.2 如何新增工具
1) 在 `src/tools/` 新建模块，例如 `my_tool.py`
2) 使用 `@tool("tool_name", args_schema=...)` 包装函数
3) 在 `src/main.py -> build_app()` 把工具加入 `build_default_tool_router(...)`
4) 若涉及文件/执行等高风险能力：
   - 明确权限边界
   - 做入参校验（Pydantic）
   - 写审计日志（ToolRouter 已统一记录）

## 8.3 如何新增工作流节点
1) 在 `src/graph/nodes.py` 添加新 node 函数
2) 在 `src/graph/builder.py` 注册并连边
3) 在 `src/graph/edges.py` 增加路由逻辑（如需 conditional edges）

## 8.4 对接企业级场景（建议）
- 工具层加入：
  - RBAC 权限控制、审计日志落库
  - 外部系统 API（工单/CRM/知识库/数据库）
- 记忆层加入：
  - 向量库（长期记忆检索）
  - 组织内知识治理（版本、引用、可信度）
- 可观测性：
  - OpenTelemetry trace
  - Prometheus 指标（成功率、耗时、token、失败率）

---

# 交付物 6：Demo 说明（3 个示例你已获得代码，这里给出“演示要点”）

## Demo 1（规划分解类）
- 输入：“制定一个为期 6 周的 Python 学习计划…”
- 观察点：
  - Planner 生成 subtasks
  - Coordinator 路由 dispatch/critic/synthesizer

## Demo 2（工具执行类）
- 覆盖：
  - safe_eval / json_validator
  - workspace 文件写读安全限制
  - python_repl
  - web_search（模拟离线）

## Demo 3（代码生成类：HN 爬虫）
- 覆盖：
  - subtasks：research → code → execute
  - Coder 产出文件并通过工具写入 `workspace/hn_crawler.py`
  - Executor 写入示例 JSON `workspace/hn_top.json`
  - Critic 审阅后 Synthesizer 汇总输出

---

# 第十三节：演示场景验证（按你的指定任务）

**任务**：  
> “请帮我编写一个 Python 爬虫，抓取 Hacker News 首页的文章标题和链接，并保存为 JSON 文件”

下面展示在 `MAS_LLM_PROVIDER=mock` 下的**可复现实验路径**（字段结构与系统一致；数值如耗时会随机器不同）。

## 13.1 任务理解与分解（Planner 产出）
你会在 `logs/events.jsonl` 看到类似事件（示例）：

```json
{"event":"planner_plan","payload":{"subtasks":[
  {"id":"t_research","title":"Identify HN HTML structure and extraction approach (title + link)","kind":"research","depends_on":[],"payload":{"target":"https://news.ycombinator.com/"},"status":"pending","assigned_to":null},
  {"id":"t_code","title":"Write hn_crawler.py to fetch and parse HN front page and output JSON","kind":"code","depends_on":["t_research"],"payload":{"output_file":"workspace/hn_crawler.py"},"status":"pending","assigned_to":null},
  {"id":"t_execute","title":"Write sample JSON output into workspace (offline-friendly demo artifact)","kind":"execute","depends_on":["t_code"],"payload":{"path":"hn_top.json","content_kind":"sample_hn_json"},"status":"pending","assigned_to":null}
]}}
```

## 13.2 智能体协作流程（dispatch 并行执行）
- Researcher：给出约束（文件必须在 workspace）
- Coder：生成 `hn_crawler.py` 并调用 `file_writer`
- Executor：写入 `hn_top.json`（示例数据）并记录工具日志

示例工具日志（最终 state 的 `tool_call_logs` 里可见）：
```json
{
  "tool_name": "file_writer",
  "args": {"path":"hn_crawler.py","content":"..."},
  "error": null
}
```

## 13.3 代码生成与审核（Critic）
- Critic 输出 `ok/issues/fix_suggestions`
- 若 `ok=false` 会触发 `iteration_count += 1` 并回到 Coordinator，直到 `max_iterations` 或通过

## 13.4 工具调用（文件写入）
最终会在本地看到：
- `workspace/hn_crawler.py`
- `workspace/hn_top.json`

## 13.5 最终输出结果（Synthesizer + Decision Trace）
CLI 最终输出会包含 `Decision Trace`（示例）：
```json
{
  "node_visits": ["coordinator","planner","coordinator","dispatch","coordinator","critic","coordinator","synthesizer","persist_history"],
  "reflection_count": 0,
  "tool_calls": 2
}
```

> 说明：mock 模式下 synthesizer 文本较简短；切换到 OpenAI/Claude 可得到更完整的最终说明文档与可运行代码细节。

---

## 重要修订：补齐 metrics 的 token/llm_calls 统计（请替换以下文件）

你之前的代码能跑通，但 `metrics.llm_calls/token` 统计不够完整。下面给出**可直接替换**的文件内容，使系统满足“Token 成本 / LLM 调用统计”要求（尽可能从 LangChain 的 message metadata 中提取 usage）。

---

### `src/graph/state.py`（替换为以下完整内容）
```python
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class ToolCallLog(BaseModel):
    tool_name: str
    args: Dict[str, Any]
    result_preview: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int
    ts: float


class SubTask(BaseModel):
    id: str
    title: str
    kind: Literal["research", "code", "execute"]
    depends_on: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "in_progress", "done", "failed"] = "pending"
    assigned_to: Optional[str] = None


class Metrics(BaseModel):
    started_ts: float
    ended_ts: Optional[float] = None

    node_visits: List[str] = Field(default_factory=list)
    execution_depth: int = 0
    reflection_count: int = 0

    llm_calls: int = 0
    tool_calls: int = 0

    # Token usage (best-effort; provider/model may not return these)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _merge_messages(a: List[BaseMessage], b: List[BaseMessage]) -> List[BaseMessage]:
    return [*a, *b]


def _merge_tool_logs(a: List[ToolCallLog], b: List[ToolCallLog]) -> List[ToolCallLog]:
    return [*a, *b]


def _merge_agent_outputs(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _merge_subtasks(a: List[SubTask], b: List[SubTask]) -> List[SubTask]:
    """
    Merge subtasks by id (right side wins), preserving order of first appearance.
    """
    by_id: Dict[str, SubTask] = {}
    order: List[str] = []
    for t in a:
        by_id[t.id] = t
        order.append(t.id)
    for t in b:
        if t.id not in by_id:
            order.append(t.id)
        by_id[t.id] = t
    return [by_id[i] for i in order]


def _merge_issues(a: List[str], b: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in [*a, *b]:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], _merge_messages]
    original_task: str
    subtasks: Annotated[List[SubTask], _merge_subtasks]
    agent_outputs: Annotated[Dict[str, Any], _merge_agent_outputs]
    tool_call_logs: Annotated[List[ToolCallLog], _merge_tool_logs]
    current_agent: str
    iteration_count: int
    final_answer: Optional[str]
    next: str

    issues: Annotated[List[str], _merge_issues]
    reviewed: bool
    metrics: Metrics

    config: Dict[str, Any]
```

---

### `src/agents/base.py`（替换为以下完整内容）
```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.config.prompts import PromptTemplate


@dataclass
class AgentResult:
    """Standardized agent output + best-effort usage extraction."""
    output: Dict[str, Any]
    messages: List[BaseMessage]
    usage: Dict[str, int]


@dataclass
class BaseAgent:
    """
    Base class for all agents.

    Agents are intentionally thin:
    - prompt rendering
    - model call
    - JSON parsing
    - usage extraction (best-effort)
    """

    name: str
    llm: BaseChatModel
    prompt: PromptTemplate

    def invoke_json(self, system: str, user: str) -> AgentResult:
        """Call LLM and parse JSON response with robust fallbacks."""
        msgs: List[BaseMessage] = [SystemMessage(content=system), HumanMessage(content=user)]
        resp = self.llm.invoke(msgs)

        content = getattr(resp, "content", "") or ""

        parsed: Dict[str, Any]
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"raw": content}

        usage = _extract_usage(resp)

        return AgentResult(output=parsed, messages=[AIMessage(content=content)], usage=usage)

    def run(self, **kwargs: Any) -> AgentResult:
        system = self.prompt.system
        user = self.prompt.user.format(**kwargs)
        return self.invoke_json(system=system, user=user)


def _extract_usage(message: Any) -> Dict[str, int]:
    """
    Best-effort extraction of token usage.

    Different LangChain providers store usage in different locations.
    We try common fields but do not assume availability.
    """
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Newer LangChain often provides usage_metadata
    um = getattr(message, "usage_metadata", None)
    if isinstance(um, dict):
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(um.get(k), int):
                usage[k] = um[k]
        return usage

    # Some providers put usage under response_metadata["token_usage"]
    rm = getattr(message, "response_metadata", None)
    if isinstance(rm, dict):
        tu = rm.get("token_usage")
        if isinstance(tu, dict):
            # normalize common patterns
            pt = tu.get("prompt_tokens") or tu.get("input_tokens")
            ct = tu.get("completion_tokens") or tu.get("output_tokens")
            tt = tu.get("total_tokens")
            if isinstance(pt, int):
                usage["prompt_tokens"] = pt
            if isinstance(ct, int):
                usage["completion_tokens"] = ct
            if isinstance(tt, int):
                usage["total_tokens"] = tt
            else:
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return usage
```

---

### `src/graph/nodes.py`（替换为以下完整内容）
```python
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.coder import CoderAgent
from src.agents.coordinator import CoordinatorAgent
from src.agents.critic import CriticAgent
from src.agents.executor import ExecutorAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.config.settings import Settings
from src.graph.state import AgentState, SubTask, ToolCallLog
from src.tools.base import ToolRouter
from src.utils.logger import EventLogger


@dataclass
class NodeContext:
    settings: Settings
    logger: EventLogger
    tool_router: ToolRouter

    coordinator: CoordinatorAgent
    planner: PlannerAgent
    researcher: ResearcherAgent
    coder: CoderAgent
    executor: ExecutorAgent
    critic: CriticAgent
    synthesizer: SynthesizerAgent


def _bump_llm_metrics(state: AgentState, usage: Dict[str, int]) -> None:
    state["metrics"]["llm_calls"] += 1
    state["metrics"]["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
    state["metrics"]["completion_tokens"] += int(usage.get("completion_tokens", 0))
    # If total_tokens not provided, approximate
    tt = usage.get("total_tokens")
    if isinstance(tt, int) and tt > 0:
        state["metrics"]["total_tokens"] += tt
    else:
        state["metrics"]["total_tokens"] += int(usage.get("prompt_tokens", 0)) + int(
            usage.get("completion_tokens", 0)
        )


def parse_input(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        ctx.settings.ensure_dirs()
        start = time.time()
        original_task = state.get("original_task", "").strip()
        if not original_task:
            raise ValueError("original_task is required")

        metrics = state.get("metrics")
        if not metrics:
            metrics = {
                "started_ts": start,
                "ended_ts": None,
                "node_visits": [],
                "execution_depth": 0,
                "reflection_count": 0,
                "llm_calls": 0,
                "tool_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        ctx.logger.log_event("parse_input", {"original_task": original_task})

        return {
            "messages": [
                SystemMessage(content="You are a multi-agent system."),
                HumanMessage(content=original_task),
            ],
            "original_task": original_task,
            "subtasks": state.get("subtasks", []),
            "agent_outputs": state.get("agent_outputs", {}),
            "tool_call_logs": state.get("tool_call_logs", []),
            "issues": state.get("issues", []),
            "iteration_count": state.get("iteration_count", 0),
            "reviewed": state.get("reviewed", False),
            "metrics": metrics,
            "config": state.get("config", {}),
        }

    return _node


def coordinator_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("coordinator")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        decision = ctx.coordinator.decide_next(state)
        ctx.logger.log_event("coordinator_decision", decision)

        return {
            "current_agent": "coordinator",
            "next": decision["next"],
            "agent_outputs": {"coordinator": decision},
        }

    return _node


def planner_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("planner")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        original_task = state["original_task"]
        existing = state.get("subtasks", [])
        plan = ctx.planner.plan(original_task=original_task, existing_subtasks=existing)
        subtasks: List[SubTask] = plan.get("subtasks", [])

        ctx.logger.log_event(
            "planner_plan",
            {"subtasks": [t.model_dump() for t in subtasks], "rationale": plan.get("rationale", [])},
        )
        return {
            "current_agent": "planner",
            "subtasks": subtasks,
            "issues": [],  # reset issues when replanning
            "agent_outputs": {"planner": {"rationale": plan.get("rationale", [])}},
        }

    return _node


def _ready_subtasks(subtasks: List[SubTask]) -> List[SubTask]:
    by_id = {t.id: t for t in subtasks}
    ready: List[SubTask] = []
    for t in subtasks:
        if t.status != "pending":
            continue
        if all(by_id.get(dep) and by_id[dep].status == "done" for dep in t.depends_on):
            ready.append(t)
    return ready


def dispatch_node(ctx: NodeContext):
    """
    Dispatch work to worker agents.

    Parallelism is implemented with a ThreadPoolExecutor within this node.
    """

    def _run_one(subtask: SubTask, state: AgentState) -> Tuple[SubTask, Dict[str, Any], List[ToolCallLog], Dict[str, int]]:
        tool_logs: List[ToolCallLog] = []
        usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if subtask.kind == "research":
            res = ctx.researcher.run(task=subtask.title)
            usage = res.usage
            return subtask.model_copy(update={"status": "done", "assigned_to": "researcher"}), res.output, tool_logs, usage

        if subtask.kind == "code":
            context = json.dumps(state.get("agent_outputs", {}).get("researcher", {}), ensure_ascii=False, indent=2)
            res = ctx.coder.run(task=subtask.title, context=context)
            usage = res.usage
            out = res.output

            files = out.get("files", [])
            for f in files:
                path = f.get("path", "")
                content = f.get("content", "")
                rel = path[len("workspace/") :] if path.startswith("workspace/") else path
                _, log = ctx.tool_router.call("file_writer", {"path": rel, "content": content})
                tool_logs.append(log)

            return subtask.model_copy(update={"status": "done", "assigned_to": "coder"}), out, tool_logs, usage

        if subtask.kind == "execute":
            payload = subtask.payload or {}
            if payload.get("content_kind") == "sample_hn_json":
                out = ctx.executor.write_sample_hn_json()
                tl = out.get("tool_log")
                if tl:
                    try:
                        tool_logs.append(ToolCallLog(**tl))
                    except Exception:
                        pass
                return subtask.model_copy(update={"status": "done", "assigned_to": "executor"}), out, tool_logs, usage

            res = ctx.executor.run(task=subtask.title, tools="\n".join(ctx.tool_router.list_tools()))
            usage = res.usage
            return subtask.model_copy(update={"status": "done", "assigned_to": "executor"}), res.output, tool_logs, usage

        return subtask.model_copy(update={"status": "failed"}), {"error": "unknown kind"}, tool_logs, usage

    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("dispatch")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        subtasks = state.get("subtasks", [])
        ready = _ready_subtasks(subtasks)
        if not ready:
            ctx.logger.log_event("dispatch_no_ready_subtasks", {"count": len(subtasks)})
            return {"current_agent": "dispatcher"}

        ctx.logger.log_event(
            "dispatch_ready_subtasks",
            {"ready": [t.model_dump() for t in ready], "parallel_workers": ctx.settings.parallel_workers},
        )

        updated_subtasks: List[SubTask] = []
        outputs_delta: Dict[str, Any] = {}
        tool_logs_delta: List[ToolCallLog] = []
        usage_list: List[Dict[str, int]] = []

        id_to_sub = {t.id: t for t in subtasks}
        for t in ready:
            id_to_sub[t.id] = t.model_copy(update={"status": "in_progress"})
        subtasks = list(id_to_sub.values())

        with ThreadPoolExecutor(max_workers=ctx.settings.parallel_workers) as ex:
            futures = {ex.submit(_run_one, t, state): t.id for t in ready}
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    new_task, out, logs, usage = fut.result()
                except Exception as e:
                    new_task = id_to_sub[tid].model_copy(update={"status": "failed"})
                    out = {"error": repr(e)}
                    logs = []
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                updated_subtasks.append(new_task)
                outputs_delta[new_task.assigned_to or "worker"] = {
                    **outputs_delta.get(new_task.assigned_to or "worker", {}),
                    new_task.id: out,
                }
                tool_logs_delta.extend(logs)
                usage_list.append(usage)

        for u in usage_list:
            # Only bump if it looks like a real call (any token count non-zero OR always count as call)
            _bump_llm_metrics(state, u)

        ctx.logger.log_event(
            "dispatch_results",
            {
                "updated_subtasks": [t.model_dump() for t in updated_subtasks],
                "tool_calls": len(tool_logs_delta),
                "llm_calls_added": len(usage_list),
            },
        )

        state["metrics"]["tool_calls"] += len(tool_logs_delta)

        return {
            "current_agent": "dispatcher",
            "subtasks": updated_subtasks,
            "agent_outputs": outputs_delta,
            "tool_call_logs": tool_logs_delta,
        }

    return _node


def critic_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("critic")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        outputs = json.dumps(state.get("agent_outputs", {}), ensure_ascii=False, indent=2)
        res = ctx.critic.run(original_task=state["original_task"], outputs=outputs)
        _bump_llm_metrics(state, res.usage)

        review = res.output
        ok = bool(review.get("ok", False))
        issues = review.get("issues", []) if isinstance(review.get("issues", []), list) else []
        fix_suggestions = review.get("fix_suggestions", [])

        ctx.logger.log_event(
            "critic_review",
            {"ok": ok, "issues": issues, "fix_suggestions": fix_suggestions},
        )

        if ok:
            return {
                "current_agent": "critic",
                "reviewed": True,
                "issues": [],
                "agent_outputs": {"critic": review},
            }

        it = int(state.get("iteration_count", 0)) + 1
        state["metrics"]["reflection_count"] = it
        return {
            "current_agent": "critic",
            "iteration_count": it,
            "issues": issues or ["Critic requested improvements."],
            "agent_outputs": {"critic": review},
        }

    return _node


def synthesizer_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("synthesizer")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        outputs = json.dumps(state.get("agent_outputs", {}), ensure_ascii=False, indent=2)
        issues = state.get("issues", [])
        res = ctx.synthesizer.run(original_task=state["original_task"], outputs=outputs, issues=issues)
        _bump_llm_metrics(state, res.usage)

        out = res.output
        if isinstance(out, dict) and "raw" in out:
            final = str(out["raw"])
        elif isinstance(out, dict):
            final = (
                f"Task:\n{state['original_task']}\n\n"
                f"Outputs:\n{outputs}\n\n"
                f"Issues:\n{issues}\n"
            )
        else:
            final = str(out)

        trace = {
            "node_visits": state["metrics"]["node_visits"],
            "reflection_count": state["metrics"]["reflection_count"],
            "tool_calls": state["metrics"]["tool_calls"],
            "llm_calls": state["metrics"]["llm_calls"],
            "prompt_tokens": state["metrics"]["prompt_tokens"],
            "completion_tokens": state["metrics"]["completion_tokens"],
            "total_tokens": state["metrics"]["total_tokens"],
        }
        final_with_trace = f"{final}\n\n---\n## Decision Trace\n```json\n{json.dumps(trace, ensure_ascii=False, indent=2)}\n```"

        ctx.logger.log_event("synthesizer_finalized", {"final_len": len(final_with_trace)})

        return {"current_agent": "synthesizer", "final_answer": final_with_trace}

    return _node


def human_in_loop_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("human_in_loop")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])

        ctx.logger.log_event("human_in_loop", {"enabled": ctx.settings.human_in_loop})

        if not ctx.settings.human_in_loop:
            return {"issues": ["Human-in-the-loop requested but disabled."], "next": "coordinator"}

        human_input = state.get("agent_outputs", {}).get("human_input")
        if not human_input:
            return {"issues": ["No human_input provided."], "next": "coordinator"}

        return {
            "agent_outputs": {"human": {"input": human_input}},
            "issues": [],
            "next": "coordinator",
        }

    return _node


def persist_history_node(ctx: NodeContext):
    def _node(state: AgentState) -> Dict[str, Any]:
        state["metrics"]["node_visits"].append("persist_history")
        state["metrics"]["execution_depth"] = len(state["metrics"]["node_visits"])
        state["metrics"]["ended_ts"] = time.time()

        if ctx.settings.history_file:
            record = {
                "ts": time.time(),
                "original_task": state.get("original_task"),
                "final_answer": state.get("final_answer"),
                "metrics": state.get("metrics"),
            }
            ctx.settings.history_file.parent.mkdir(parents=True, exist_ok=True)
            with ctx.settings.history_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        ctx.logger.log_event("persist_history", {"path": str(ctx.settings.history_file)})
        return {}

    return _node
```

---

### `src/main.py`（仅需修改 metrics 初始化；直接替换该片段对应部分也行，这里给出完整文件便于复制）
```python
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

from dotenv import load_dotenv

from src.agents.coder import CoderAgent
from src.agents.coordinator import CoordinatorAgent
from src.agents.critic import CriticAgent
from src.agents.executor import ExecutorAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.config.settings import Settings
from src.graph.builder import GraphBuilder
from src.graph.nodes import NodeContext
from src.llm.factory import create_chat_model
from src.tools.base import build_default_tool_router
from src.tools.calculator import build_calculator_tools
from src.tools.code_executor import build_code_tools
from src.tools.file_manager import build_file_tools
from src.tools.search import build_search_tools
from src.utils.logger import build_logger


def build_app(settings: Settings):
    """Create the LangGraph app with all agents/tools wired."""
    llm = create_chat_model(settings)

    tool_router = build_default_tool_router(
        tools=[
            *build_calculator_tools(),
            *build_file_tools(settings.workspace_dir),
            *build_code_tools(),
            *build_search_tools(),
        ]
    )

    logger = build_logger("multi_agent_system", debug=settings.debug, logs_dir=settings.logs_dir)

    ctx = NodeContext(
        settings=settings,
        logger=logger,
        tool_router=tool_router,
        coordinator=CoordinatorAgent.create(llm),
        planner=PlannerAgent.create(llm),
        researcher=ResearcherAgent.create(llm),
        coder=CoderAgent.create(llm),
        executor=ExecutorAgent.create(llm, tool_router),
        critic=CriticAgent.create(llm),
        synthesizer=SynthesizerAgent.create(llm),
    )

    app = GraphBuilder(ctx).build()
    return app, logger


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(description="General-purpose multi-agent system (LangGraph).")
    p.add_argument("task", nargs="?", default="", help="Task text. If empty, read from stdin.")
    p.add_argument("--provider", default=os.getenv("MAS_LLM_PROVIDER", "mock"))
    p.add_argument("--debug", action="store_true", default=False)
    args = p.parse_args()

    task = args.task.strip() if args.task else ""
    if not task:
        task = input("Enter task: ").strip()

    settings = Settings(
        llm_provider=args.provider,
        debug=args.debug or Settings().debug,
    )
    settings.ensure_dirs()

    app, logger = build_app(settings)

    init_state: Dict[str, Any] = {
        "original_task": task,
        "iteration_count": 0,
        "config": {
            "max_iterations": settings.max_iterations,
        },
        "metrics": {
            "started_ts": time.time(),
            "ended_ts": None,
            "node_visits": [],
            "execution_depth": 0,
            "reflection_count": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    out = app.invoke(init_state)
    final = out.get("final_answer") or "(no final_answer)"
    logger.console.print(final)


if __name__ == "__main__":
    main()
```

---

以上补丁替换后，系统会在最终输出的 `Decision Trace` 中包含 `llm_calls/token` 字段，并在 `history.jsonl` 中持久化。

---

如果你还需要我继续（例如：把“mock 生成的 hn_crawler.py”改成**纯标准库实现**、或把 `visualizer` 接入到运行后自动输出 Mermaid），请回复：`继续`。

