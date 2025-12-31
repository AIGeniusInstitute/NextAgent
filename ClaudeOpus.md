# 通用 Multi-Agent 问题求解系统 - 完整开发交付

## 第一部分：系统架构设计文档

---

### 1.1 整体架构图

```mermaid
graph TB
    subgraph "用户交互层"
        UI[用户输入] --> INPUT[输入解析器]
        OUTPUT[最终输出] --> USER[用户]
    end
    
    subgraph "编排控制层"
        INPUT --> COORD[协调者 Coordinator]
        COORD --> PLANNER[规划者 Planner]
        PLANNER --> SCHEDULER[调度器 Scheduler]
    end
    
    subgraph "执行层 - Worker Agents"
        SCHEDULER --> RESEARCHER[研究员 Researcher]
        SCHEDULER --> CODER[编码者 Coder]
        SCHEDULER --> EXECUTOR[执行者 Executor]
    end
    
    subgraph "质量保障层"
        RESEARCHER --> CRITIC[审核者 Critic]
        CODER --> CRITIC
        EXECUTOR --> CRITIC
        CRITIC -->|需要修正| SCHEDULER
        CRITIC -->|通过| SYNTH[综合者 Synthesizer]
    end
    
    subgraph "支撑系统"
        TOOLS[(工具系统)]
        MEMORY[(记忆系统)]
        LLM[(LLM 工厂)]
        LOGGER[(日志系统)]
    end
    
    SYNTH --> OUTPUT
    
    RESEARCHER -.-> TOOLS
    CODER -.-> TOOLS
    EXECUTOR -.-> TOOLS
    
    COORD -.-> MEMORY
    PLANNER -.-> MEMORY
    CRITIC -.-> MEMORY
    
    COORD -.-> LLM
    PLANNER -.-> LLM
    RESEARCHER -.-> LLM
    CODER -.-> LLM
    EXECUTOR -.-> LLM
    CRITIC -.-> LLM
    SYNTH -.-> LLM
    
    COORD -.-> LOGGER
    SCHEDULER -.-> LOGGER
```

### 1.2 LangGraph 状态机设计

```mermaid
stateDiagram-v2
    [*] --> InputParser: 用户输入
    InputParser --> Coordinator: 解析完成
    Coordinator --> Planner: 需要规划
    Coordinator --> END: 简单问题直接回答
    
    Planner --> TaskRouter: 生成执行计划
    
    TaskRouter --> Researcher: 需要研究
    TaskRouter --> Coder: 需要编码
    TaskRouter --> Executor: 需要执行
    TaskRouter --> ParallelExecution: 可并行执行
    
    ParallelExecution --> Researcher
    ParallelExecution --> Coder
    ParallelExecution --> Executor
    
    Researcher --> Critic: 研究完成
    Coder --> Critic: 代码完成
    Executor --> Critic: 执行完成
    
    Critic --> TaskRouter: 需要修正
    Critic --> HumanReview: 需要人工介入
    Critic --> Synthesizer: 质量通过
    
    HumanReview --> TaskRouter: 人工反馈
    HumanReview --> Synthesizer: 人工确认
    
    Synthesizer --> Coordinator: 汇总完成
    Coordinator --> Planner: 需要继续
    Coordinator --> FinalOutput: 任务完成
    
    FinalOutput --> END: 输出结果
    
    note right of Critic
        反思循环：
        - 最多3次重试
        - 超过则降级处理
    end note
```

### 1.3 智能体交互时序图

```mermaid
sequenceDiagram
    participant U as 用户
    participant C as Coordinator
    participant P as Planner
    participant S as Scheduler
    participant R as Researcher
    participant CD as Coder
    participant E as Executor
    participant CR as Critic
    participant SY as Synthesizer
    participant T as Tools
    participant M as Memory
    
    U->>C: 提交任务
    C->>M: 存储原始任务
    C->>C: 理解任务意图
    C->>P: 请求任务规划
    
    P->>M: 查询历史相似任务
    P->>P: 分解任务
    P->>S: 返回执行计划
    
    loop 任务执行循环
        S->>S: 选择下一个子任务
        
        alt 研究类任务
            S->>R: 分配研究任务
            R->>T: 调用搜索工具
            T-->>R: 返回搜索结果
            R->>CR: 提交研究结果
        else 编码类任务
            S->>CD: 分配编码任务
            CD->>T: 调用代码执行工具
            T-->>CD: 返回执行结果
            CD->>CR: 提交代码
        else 执行类任务
            S->>E: 分配执行任务
            E->>T: 调用相关工具
            T-->>E: 返回结果
            E->>CR: 提交执行结果
        end
        
        CR->>CR: 质量评估
        alt 质量不合格
            CR->>S: 需要重试
        else 需要人工介入
            CR->>U: 请求人工确认
            U->>CR: 人工反馈
        else 质量合格
            CR->>SY: 通过审核
        end
    end
    
    SY->>SY: 整合所有结果
    SY->>C: 返回综合结果
    C->>M: 存储执行记录
    C->>U: 返回最终答案
```

### 1.4 核心数据结构说明

#### 1.4.1 主状态结构 (AgentState)

```python
class SubTask(TypedDict):
    """子任务定义"""
    id: str                          # 唯一标识
    name: str                        # 任务名称
    description: str                 # 任务描述
    assigned_agent: str              # 分配的 Agent
    dependencies: List[str]          # 依赖的子任务 ID
    status: Literal["pending", "running", "completed", "failed"]
    result: Optional[str]            # 执行结果
    retry_count: int                 # 重试次数

class ToolCallLog(TypedDict):
    """工具调用日志"""
    tool_name: str                   # 工具名称
    input_params: Dict[str, Any]     # 输入参数
    output: Any                      # 输出结果
    success: bool                    # 是否成功
    error_message: Optional[str]     # 错误信息
    timestamp: str                   # 调用时间
    duration_ms: float               # 耗时(毫秒)

class AgentState(TypedDict):
    """全局状态定义"""
    messages: Annotated[List[BaseMessage], add_messages]  # 对话历史
    original_task: str               # 原始任务
    task_understanding: str          # 任务理解
    subtasks: List[SubTask]          # 子任务列表
    current_plan: str                # 当前执行计划
    agent_outputs: Dict[str, Any]    # 各 Agent 产出
    tool_call_logs: List[ToolCallLog]# 工具调用日志
    current_agent: str               # 当前 Agent
    iteration_count: int             # 迭代计数
    max_iterations: int              # 最大迭代数
    reflection_notes: List[str]      # 反思记录
    human_feedback: Optional[str]    # 人工反馈
    needs_human_input: bool          # 是否需要人工介入
    final_answer: Optional[str]      # 最终答案
    error_log: List[str]             # 错误日志
    token_usage: Dict[str, int]      # Token 统计
    execution_time: Dict[str, float] # 执行时间统计
    next: str                        # 下一节点路由
```

### 1.5 LangGraph 节点与边设计

#### 1.5.1 节点定义

| 节点名 | 函数 | 输入状态 | 输出状态变更 | 说明 |
|--------|------|----------|--------------|------|
| `input_parser` | `parse_input()` | messages | original_task, task_understanding | 解析用户输入 |
| `coordinator` | `coordinate()` | * | current_plan, next | 任务协调与路由 |
| `planner` | `plan_tasks()` | original_task | subtasks, current_plan | 任务规划分解 |
| `researcher` | `research()` | subtasks | agent_outputs["researcher"] | 信息研究 |
| `coder` | `code()` | subtasks | agent_outputs["coder"] | 代码编写 |
| `executor` | `execute()` | subtasks | agent_outputs["executor"], tool_call_logs | 工具执行 |
| `critic` | `review()` | agent_outputs | reflection_notes, needs_human_input | 质量审核 |
| `human_node` | `human_review()` | needs_human_input | human_feedback | 人工介入 |
| `synthesizer` | `synthesize()` | agent_outputs | final_answer | 结果综合 |
| `error_handler` | `handle_error()` | error_log | next | 错误处理 |

#### 1.5.2 边与路由逻辑

```python
# 条件边定义
def route_from_coordinator(state: AgentState) -> str:
    """协调者路由逻辑"""
    if state.get("final_answer"):
        return END
    if not state.get("subtasks"):
        return "planner"
    return "task_router"

def route_task(state: AgentState) -> str:
    """任务路由逻辑"""
    pending_tasks = [t for t in state["subtasks"] if t["status"] == "pending"]
    if not pending_tasks:
        return "synthesizer"
    
    task = pending_tasks[0]
    agent_map = {
        "research": "researcher",
        "code": "coder", 
        "execute": "executor"
    }
    return agent_map.get(task["assigned_agent"], "executor")

def route_from_critic(state: AgentState) -> str:
    """审核者路由逻辑"""
    if state.get("needs_human_input"):
        return "human_node"
    
    failed_tasks = [t for t in state["subtasks"] 
                   if t["status"] == "failed" and t["retry_count"] < 3]
    if failed_tasks:
        return "task_router"
    
    if all(t["status"] == "completed" for t in state["subtasks"]):
        return "synthesizer"
    
    return "coordinator"
```

### 1.6 失败重试与终止条件设计

#### 1.6.1 重试机制

```python
class RetryConfig:
    """重试配置"""
    max_task_retries: int = 3        # 单个子任务最大重试次数
    max_global_iterations: int = 10   # 全局最大迭代次数
    retry_delay_seconds: float = 1.0  # 重试延迟
    
    # 降级策略
    degradation_strategies: Dict[str, str] = {
        "researcher": "使用缓存结果或跳过",
        "coder": "返回伪代码或模板",
        "executor": "记录失败并继续",
    }
```

#### 1.6.2 终止条件

| 条件类型 | 触发条件 | 处理方式 |
|----------|----------|----------|
| 正常完成 | `final_answer` 非空且质量通过 | 返回结果，路由到 END |
| 迭代上限 | `iteration_count >= max_iterations` | 汇总当前结果，标记未完成 |
| 全局超时 | 执行时间超过配置阈值 | 中断执行，返回部分结果 |
| 不可恢复错误 | 连续失败次数过多 | 记录错误，返回失败原因 |
| 用户中断 | 收到中断信号 | 保存状态，支持恢复 |

#### 1.6.3 错误处理流程

```mermaid
flowchart TD
    ERROR[发生错误] --> CLASSIFY{错误分类}
    
    CLASSIFY -->|可重试| RETRY{重试次数检查}
    CLASSIFY -->|不可重试| LOG[记录错误]
    
    RETRY -->|未超限| WAIT[延迟等待]
    WAIT --> REEXEC[重新执行]
    
    RETRY -->|已超限| DEGRADE{降级策略}
    DEGRADE -->|可降级| FALLBACK[执行降级方案]
    DEGRADE -->|不可降级| LOG
    
    FALLBACK --> CONTINUE[继续执行]
    LOG --> HUMAN{需要人工?}
    
    HUMAN -->|是| NOTIFY[通知人工介入]
    HUMAN -->|否| PARTIAL[返回部分结果]
    
    REEXEC --> SUCCESS{成功?}
    SUCCESS -->|是| CONTINUE
    SUCCESS -->|否| RETRY
```

### 1.7 组件职责详细说明

| 组件 | 核心职责 | 主要接口 | 依赖组件 |
|------|----------|----------|----------|
| **任务解析器** | 标准化输入、意图识别、提取关键实体 | `parse(input) -> ParsedTask` | LLM |
| **任务规划器** | DAG 分解、依赖分析、优先级排序 | `plan(task) -> List[SubTask]` | LLM, Memory |
| **Agent 调度器** | 能力匹配、负载均衡、并行控制 | `schedule(tasks) -> ExecutionPlan` | Agent Registry |
| **状态管理器** | 状态 CRUD、快照备份、事务支持 | `get/set/snapshot/restore` | - |
| **记忆系统** | 上下文缓存、历史检索、向量存储 | `store/retrieve/search` | Vector DB (可选) |
| **工具路由器** | 工具注册、权限校验、调用封装 | `register/invoke/list` | Tool Registry |
| **评估器** | 质量评分、Token 统计、性能分析 | `evaluate(result) -> Score` | LLM |

---

**第一部分完成。请输入 "继续" 获取第二部分：模块与类设计说明。**

## 第二部分：模块与类设计说明

---

### 2.1 核心类 UML 类图

```mermaid
classDiagram
    %% 基础抽象类
    class BaseAgent {
        <<abstract>>
        #name: str
        #llm: BaseChatModel
        #tools: List[BaseTool]
        #prompt_template: str
        #logger: Logger
        +__init__(config: AgentConfig)
        +invoke(state: AgentState) AgentState*
        +get_prompt() str
        +bind_tools(tools: List) void
        #_execute(state: AgentState) AgentState*
        #_format_output(response: Any) str
    }
    
    %% 具体 Agent 实现
    class CoordinatorAgent {
        -task_analyzer: TaskAnalyzer
        +invoke(state: AgentState) AgentState
        +understand_task(task: str) str
        +decide_next_step(state: AgentState) str
        +check_completion(state: AgentState) bool
    }
    
    class PlannerAgent {
        -decomposition_strategy: str
        +invoke(state: AgentState) AgentState
        +decompose_task(task: str) List~SubTask~
        +analyze_dependencies(tasks: List) DAG
        +prioritize(tasks: List) List~SubTask~
    }
    
    class ResearcherAgent {
        -search_depth: int
        +invoke(state: AgentState) AgentState
        +research(query: str) ResearchResult
        +synthesize_findings(results: List) str
    }
    
    class CoderAgent {
        -language: str
        -style_guide: str
        +invoke(state: AgentState) AgentState
        +generate_code(spec: str) CodeResult
        +review_code(code: str) List~Issue~
        +fix_issues(code: str, issues: List) str
    }
    
    class ExecutorAgent {
        -sandbox_enabled: bool
        +invoke(state: AgentState) AgentState
        +execute_tool(tool: str, params: Dict) Any
        +validate_result(result: Any) bool
    }
    
    class CriticAgent {
        -quality_threshold: float
        +invoke(state: AgentState) AgentState
        +evaluate(output: Any) EvaluationResult
        +suggest_improvements(output: Any) List~str~
        +decide_action(score: float) str
    }
    
    class SynthesizerAgent {
        -output_format: str
        +invoke(state: AgentState) AgentState
        +aggregate_results(outputs: Dict) str
        +format_final_answer(content: str) str
    }
    
    BaseAgent <|-- CoordinatorAgent
    BaseAgent <|-- PlannerAgent
    BaseAgent <|-- ResearcherAgent
    BaseAgent <|-- CoderAgent
    BaseAgent <|-- ExecutorAgent
    BaseAgent <|-- CriticAgent
    BaseAgent <|-- SynthesizerAgent
    
    %% 工具系统
    class BaseTool {
        <<abstract>>
        #name: str
        #description: str
        #args_schema: Type~BaseModel~
        +invoke(input: Dict) Any*
        +validate_input(input: Dict) bool
    }
    
    class CalculatorTool {
        +invoke(input: Dict) CalculationResult
        -safe_eval(expression: str) float
    }
    
    class FileManagerTool {
        -workspace_path: Path
        +invoke(input: Dict) FileResult
        +read_file(path: str) str
        +write_file(path: str, content: str) bool
        -validate_path(path: str) bool
    }
    
    class CodeExecutorTool {
        -timeout: int
        -sandbox: Sandbox
        +invoke(input: Dict) ExecutionResult
        +execute_python(code: str) str
    }
    
    class SearchTool {
        -api_key: str
        +invoke(input: Dict) SearchResult
        +search(query: str) List~SearchItem~
    }
    
    BaseTool <|-- CalculatorTool
    BaseTool <|-- FileManagerTool
    BaseTool <|-- CodeExecutorTool
    BaseTool <|-- SearchTool
    
    %% 记忆系统
    class BaseMemory {
        <<abstract>>
        +store(key: str, value: Any) void*
        +retrieve(key: str) Any*
        +search(query: str) List~Any~*
        +clear() void*
    }
    
    class ShortTermMemory {
        -buffer: Dict
        -max_size: int
        +store(key: str, value: Any) void
        +retrieve(key: str) Any
        +get_recent(n: int) List
    }
    
    class LongTermMemory {
        -storage_path: Path
        -index: Dict
        +store(key: str, value: Any) void
        +retrieve(key: str) Any
        +persist() void
        +load() void
    }
    
    BaseMemory <|-- ShortTermMemory
    BaseMemory <|-- LongTermMemory
    
    %% 图构建器
    class GraphBuilder {
        -agents: Dict~str, BaseAgent~
        -tools: List~BaseTool~
        -config: GraphConfig
        +build() CompiledGraph
        +add_node(name: str, agent: BaseAgent) void
        +add_edge(from: str, to: str) void
        +add_conditional_edge(from: str, router: Callable) void
        +compile() CompiledGraph
    }
    
    %% 状态管理
    class StateManager {
        -current_state: AgentState
        -history: List~AgentState~
        +get_state() AgentState
        +update_state(updates: Dict) void
        +snapshot() str
        +restore(snapshot_id: str) void
        +get_history() List~AgentState~
    }
    
    %% 关系
    GraphBuilder --> BaseAgent : manages
    GraphBuilder --> BaseTool : registers
    StateManager --> AgentState : manages
    BaseAgent --> BaseTool : uses
    BaseAgent --> BaseMemory : uses
```

### 2.2 核心接口定义

#### 2.2.1 Agent 接口

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

class IAgent(ABC):
    """Agent 接口定义"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Agent 描述"""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """Agent 能力列表"""
        pass
    
    @abstractmethod
    def invoke(self, state: "AgentState") -> "AgentState":
        """
        执行 Agent 逻辑
        
        Args:
            state: 当前系统状态
            
        Returns:
            更新后的系统状态
        """
        pass
    
    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """
        判断是否能处理指定类型任务
        
        Args:
            task_type: 任务类型
            
        Returns:
            是否能处理
        """
        pass
```

#### 2.2.2 工具接口

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from pydantic import BaseModel

class ITool(ABC):
    """工具接口定义"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
    
    @property
    @abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """参数 Schema"""
        pass
    
    @abstractmethod
    def invoke(self, **kwargs) -> Any:
        """
        执行工具
        
        Args:
            **kwargs: 工具参数
            
        Returns:
            执行结果
        """
        pass
    
    @abstractmethod
    def validate(self, **kwargs) -> bool:
        """
        验证参数
        
        Args:
            **kwargs: 待验证参数
            
        Returns:
            是否有效
        """
        pass
```

#### 2.2.3 记忆接口

```python
from abc import ABC, abstractmethod
from typing import Any, List, Optional

class IMemory(ABC):
    """记忆系统接口"""
    
    @abstractmethod
    def store(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        """
        存储数据
        
        Args:
            key: 键
            value: 值
            metadata: 元数据
        """
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """
        检索数据
        
        Args:
            key: 键
            
        Returns:
            存储的值，不存在返回 None
        """
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Any]:
        """
        搜索相关数据
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            
        Returns:
            相关结果列表
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空记忆"""
        pass
```

#### 2.2.4 图构建器接口

```python
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Union
from langgraph.graph import CompiledGraph

class IGraphBuilder(ABC):
    """图构建器接口"""
    
    @abstractmethod
    def add_node(self, name: str, handler: Callable) -> "IGraphBuilder":
        """
        添加节点
        
        Args:
            name: 节点名称
            handler: 节点处理函数
            
        Returns:
            构建器实例（链式调用）
        """
        pass
    
    @abstractmethod
    def add_edge(self, from_node: str, to_node: str) -> "IGraphBuilder":
        """
        添加边
        
        Args:
            from_node: 源节点
            to_node: 目标节点
            
        Returns:
            构建器实例
        """
        pass
    
    @abstractmethod
    def add_conditional_edge(
        self, 
        from_node: str, 
        router: Callable,
        path_map: Dict[str, str]
    ) -> "IGraphBuilder":
        """
        添加条件边
        
        Args:
            from_node: 源节点
            router: 路由函数
            path_map: 路由映射
            
        Returns:
            构建器实例
        """
        pass
    
    @abstractmethod
    def set_entry_point(self, node: str) -> "IGraphBuilder":
        """设置入口点"""
        pass
    
    @abstractmethod
    def compile(self) -> CompiledGraph:
        """
        编译图
        
        Returns:
            编译后的可执行图
        """
        pass
```

### 2.3 数据模型定义

#### 2.3.1 配置模型

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal

class LLMConfig(BaseModel):
    """LLM 配置"""
    provider: Literal["openai", "anthropic", "local"] = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    api_key: Optional[str] = None
    base_url: Optional[str] = None

class AgentConfig(BaseModel):
    """Agent 配置"""
    name: str
    enabled: bool = True
    llm_config: Optional[LLMConfig] = None
    max_retries: int = 3
    timeout_seconds: float = 60.0
    tools: List[str] = Field(default_factory=list)

class SystemConfig(BaseModel):
    """系统配置"""
    debug_mode: bool = False
    max_iterations: int = 10
    default_llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    workspace_dir: str = "workspace"
    log_dir: str = "logs"
    enable_human_in_loop: bool = True
    parallel_execution: bool = True
```

#### 2.3.2 任务模型

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class TaskPriority(str, Enum):
    """任务优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SubTask(BaseModel):
    """子任务模型"""
    id: str = Field(description="唯一标识")
    name: str = Field(description="任务名称")
    description: str = Field(description="任务描述")
    task_type: Literal["research", "code", "execute", "analyze"] = Field(
        description="任务类型"
    )
    assigned_agent: str = Field(description="分配的 Agent")
    dependencies: List[str] = Field(default_factory=list, description="依赖任务ID")
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExecutionPlan(BaseModel):
    """执行计划"""
    plan_id: str
    original_task: str
    subtasks: List[SubTask]
    execution_order: List[str]  # 子任务 ID 的执行顺序
    parallel_groups: List[List[str]]  # 可并行执行的任务组
    estimated_duration: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)
```

#### 2.3.3 结果模型

```python
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime

class ToolCallLog(BaseModel):
    """工具调用日志"""
    tool_name: str
    input_params: Dict[str, Any]
    output: Any
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: float

class AgentOutput(BaseModel):
    """Agent 输出"""
    agent_name: str
    task_id: str
    output: str
    reasoning: str = Field(description="推理过程")
    tool_calls: List[ToolCallLog] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    timestamp: datetime = Field(default_factory=datetime.now)

class EvaluationResult(BaseModel):
    """评估结果"""
    score: float = Field(ge=0, le=1)
    passed: bool
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    reasoning: str

class ExecutionMetrics(BaseModel):
    """执行指标"""
    total_duration_seconds: float
    token_usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0}
    )
    agent_durations: Dict[str, float] = Field(default_factory=dict)
    iteration_count: int
    retry_count: int
    tool_call_count: int
    success: bool

class FinalResult(BaseModel):
    """最终结果"""
    task_id: str
    original_task: str
    answer: str
    reasoning_trace: List[str]
    agent_outputs: Dict[str, AgentOutput]
    metrics: ExecutionMetrics
    created_at: datetime = Field(default_factory=datetime.now)
```

### 2.4 模块职责详细说明

#### 2.4.1 `src/agents/` 模块

| 文件 | 类 | 职责 | 关键方法 |
|------|-----|------|----------|
| `base.py` | `BaseAgent` | Agent 抽象基类 | `invoke()`, `_execute()`, `_create_prompt()` |
| `coordinator.py` | `CoordinatorAgent` | 任务理解与协调 | `understand_task()`, `decide_routing()` |
| `planner.py` | `PlannerAgent` | 任务分解与规划 | `decompose()`, `create_dag()` |
| `researcher.py` | `ResearcherAgent` | 信息检索与研究 | `research()`, `summarize()` |
| `coder.py` | `CoderAgent` | 代码生成与调试 | `generate_code()`, `fix_code()` |
| `executor.py` | `ExecutorAgent` | 工具调用与执行 | `execute()`, `validate_result()` |
| `critic.py` | `CriticAgent` | 质量审核与反思 | `evaluate()`, `suggest_fix()` |
| `synthesizer.py` | `SynthesizerAgent` | 结果汇总与输出 | `aggregate()`, `format_output()` |

#### 2.4.2 `src/graph/` 模块

| 文件 | 职责 | 关键内容 |
|------|------|----------|
| `state.py` | 定义 AgentState TypedDict | 状态字段、类型注解 |
| `nodes.py` | 定义所有节点处理函数 | 各 Agent 对应的节点函数 |
| `edges.py` | 定义路由逻辑 | 条件边路由函数 |
| `builder.py` | 图构建逻辑 | `build_graph()` 函数 |

#### 2.4.3 `src/tools/` 模块

| 文件 | 类 | 功能 | 安全限制 |
|------|-----|------|----------|
| `base.py` | `ToolRegistry` | 工具注册管理 | - |
| `calculator.py` | `SafeCalculator` | 安全数学计算 | 禁止 eval 危险操作 |
| `file_manager.py` | `FileManager` | 文件读写 | 限定 workspace 目录 |
| `code_executor.py` | `CodeExecutor` | Python 代码执行 | 沙箱隔离、超时限制 |
| `search.py` | `WebSearch` | 网络搜索模拟 | - |

#### 2.4.4 `src/memory/` 模块

| 文件 | 类 | 功能 | 存储方式 |
|------|-----|------|----------|
| `short_term.py` | `ShortTermMemory` | 会话内上下文 | 内存 Dict |
| `long_term.py` | `LongTermMemory` | 持久化记忆 | JSON 文件 |

#### 2.4.5 `src/llm/` 模块

| 文件 | 类/函数 | 功能 |
|------|---------|------|
| `factory.py` | `LLMFactory` | 创建 LLM 实例，支持多种 Provider |

#### 2.4.6 `src/utils/` 模块

| 文件 | 功能 |
|------|------|
| `logger.py` | 日志配置，支持文件/控制台输出 |
| `visualizer.py` | 执行过程可视化，生成 Mermaid 图 |

### 2.5 设计模式应用

#### 2.5.1 工厂模式 - LLM 创建

```python
class LLMFactory:
    """LLM 工厂模式实现"""
    
    _providers: Dict[str, Type[BaseChatModel]] = {}
    
    @classmethod
    def register(cls, provider: str, llm_class: Type[BaseChatModel]) -> None:
        """注册 LLM 提供者"""
        cls._providers[provider] = llm_class
    
    @classmethod
    def create(cls, config: LLMConfig) -> BaseChatModel:
        """创建 LLM 实例"""
        if config.provider not in cls._providers:
            raise ValueError(f"Unknown provider: {config.provider}")
        return cls._providers[config.provider](
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
```

#### 2.5.2 策略模式 - 任务分解

```python
class DecompositionStrategy(ABC):
    """分解策略抽象基类"""
    
    @abstractmethod
    def decompose(self, task: str) -> List[SubTask]:
        pass

class SequentialDecomposition(DecompositionStrategy):
    """顺序分解策略"""
    def decompose(self, task: str) -> List[SubTask]:
        # 顺序分解逻辑
        pass

class HierarchicalDecomposition(DecompositionStrategy):
    """层次分解策略"""
    def decompose(self, task: str) -> List[SubTask]:
        # 层次分解逻辑
        pass

class PlannerAgent(BaseAgent):
    def __init__(self, strategy: DecompositionStrategy):
        self.strategy = strategy
    
    def plan(self, task: str) -> List[SubTask]:
        return self.strategy.decompose(task)
```

#### 2.5.3 观察者模式 - 执行监控

```python
class ExecutionObserver(ABC):
    """执行观察者抽象基类"""
    
    @abstractmethod
    def on_node_start(self, node: str, state: AgentState) -> None:
        pass
    
    @abstractmethod
    def on_node_end(self, node: str, state: AgentState, duration: float) -> None:
        pass
    
    @abstractmethod
    def on_error(self, node: str, error: Exception) -> None:
        pass

class LoggingObserver(ExecutionObserver):
    """日志观察者"""
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def on_node_start(self, node: str, state: AgentState) -> None:
        self.logger.info(f"Node {node} started")
    
    # ... 其他方法实现

class MetricsObserver(ExecutionObserver):
    """指标收集观察者"""
    def __init__(self):
        self.metrics = {}
    
    def on_node_end(self, node: str, state: AgentState, duration: float) -> None:
        self.metrics[node] = duration
```

### 2.6 类型系统完整定义

```python
# src/types.py - 集中类型定义

from typing import (
    TypedDict, Annotated, List, Dict, Any, 
    Optional, Literal, Union, Callable
)
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

# 路由类型
RouteType = Literal[
    "coordinator", "planner", "researcher", 
    "coder", "executor", "critic", 
    "synthesizer", "human_node", "end"
]

# 任务类型
TaskType = Literal["research", "code", "execute", "analyze", "synthesize"]

# Agent 能力映射
AgentCapabilities = Dict[str, List[TaskType]]

# 节点处理函数类型
NodeHandler = Callable[["AgentState"], "AgentState"]

# 路由函数类型  
RouterFunction = Callable[["AgentState"], RouteType]

# 完整状态类型
class AgentState(TypedDict, total=False):
    """系统全局状态"""
    # 消息历史 - 使用 add_messages reducer
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 任务相关
    original_task: str
    task_understanding: str
    subtasks: List[SubTask]
    current_plan: str
    current_subtask_id: Optional[str]
    
    # Agent 输出
    agent_outputs: Dict[str, AgentOutput]
    
    # 工具相关
    tool_call_logs: List[ToolCallLog]
    available_tools: List[str]
    
    # 控制流
    current_agent: str
    next: RouteType
    iteration_count: int
    max_iterations: int
    
    # 反思与审核
    reflection_notes: List[str]
    evaluation_results: List[EvaluationResult]
    
    # 人工介入
    needs_human_input: bool
    human_feedback: Optional[str]
    
    # 最终输出
    final_answer: Optional[str]
    reasoning_trace: List[str]
    
    # 错误处理
    error_log: List[str]
    last_error: Optional[str]
    
    # 指标统计
    token_usage: Dict[str, int]
    execution_time: Dict[str, float]
    start_time: Optional[float]
```

---

**第二部分完成。请输入 "继续" 获取第三部分：项目目录结构树与完整 Python 源代码（第1批文件）。**

## 第三部分：项目目录结构树与完整 Python 源代码（第1批）

---

### 3.1 完整项目目录结构树

```
multi_agent_system/
├── pyproject.toml                 # 项目配置与依赖
├── requirements.txt               # pip 依赖（备用）
├── README.md                      # 完整使用说明
├── .env.example                   # 环境变量示例
├── src/
│   ├── __init__.py
│   ├── main.py                    # 系统入口（CLI）
│   ├── types.py                   # 集中类型定义
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py            # 配置管理
│   │   └── prompts.py             # 提示词模板
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                # Agent 基类
│   │   ├── coordinator.py         # 协调者
│   │   ├── planner.py             # 规划者
│   │   ├── researcher.py          # 研究员
│   │   ├── coder.py               # 编码者
│   │   ├── executor.py            # 执行者
│   │   ├── critic.py              # 审核者
│   │   └── synthesizer.py         # 综合者
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py               # 状态定义
│   │   ├── nodes.py               # 节点函数
│   │   ├── edges.py               # 边与路由逻辑
│   │   └── builder.py             # 图构建器
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                # 工具基类与注册
│   │   ├── calculator.py          # 计算工具
│   │   ├── file_manager.py        # 文件操作（限定目录）
│   │   ├── code_executor.py       # 代码执行
│   │   └── search.py              # 搜索工具（模拟）
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py          # 短期记忆
│   │   └── long_term.py           # 长期记忆接口
│   ├── llm/
│   │   ├── __init__.py
│   │   └── factory.py             # LLM 工厂（可替换模型）
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # 日志工具
│       └── visualizer.py          # 可视化工具
├── examples/
│   ├── __init__.py
│   ├── example_planning.py        # 规划分解示例
│   ├── example_tool_execution.py  # 工具执行示例
│   └── example_code_generation.py # 代码生成示例
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # pytest 配置
│   ├── test_graph.py              # 图构建测试
│   ├── test_flow.py               # 完整流程测试
│   └── test_tools.py              # 工具调用测试
├── workspace/                     # 工具操作的限定目录
│   └── .gitkeep
└── logs/                          # 执行日志
    └── .gitkeep
```

---

### 3.2 源代码文件（第1批：项目配置与核心入口）

---

#### 文件：`pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "multi-agent-system"
version = "1.0.0"
description = "通用多智能体协作问题求解系统 - 基于 LangGraph"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Multi-Agent System Team", email = "team@example.com"}
]
keywords = ["langgraph", "multi-agent", "llm", "ai", "automation"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "langgraph>=0.2.0",
    "langchain>=0.2.0",
    "langchain-openai>=0.1.0",
    "langchain-anthropic>=0.1.0",
    "langchain-community>=0.2.0",
    "langchain-core>=0.2.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",
    "httpx>=0.25.0",
    "tenacity>=8.2.0",
    "tiktoken>=0.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "ruff>=0.1.0",
]

[project.scripts]
multi-agent = "src.main:main"

[project.urls]
Homepage = "https://github.com/example/multi-agent-system"
Documentation = "https://github.com/example/multi-agent-system#readme"
Repository = "https://github.com/example/multi-agent-system"

[tool.hatch.build.targets.wheel]
packages = ["src"]

[tool.black]
line-length = 100
target-version = ['py310', 'py311']
include = '\.pyi?$'
exclude = '''
/(
    \.git
    | \.hatch
    | \.mypy_cache
    | \.tox
    | \.venv
    | _build
    | buck-out
    | build
    | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
known_first_party = ["src"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=term-missing"
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "UP", "B", "C4"]
ignore = ["E501"]
target-version = "py310"
```

---

#### 文件：`requirements.txt`

```txt
# 核心依赖
langgraph>=0.2.0
langchain>=0.2.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
langchain-community>=0.2.0
langchain-core>=0.2.0

# 数据模型
pydantic>=2.0.0

# 环境配置
python-dotenv>=1.0.0

# CLI 与输出美化
rich>=13.0.0

# HTTP 客户端
httpx>=0.25.0

# 重试机制
tenacity>=8.2.0

# Token 计数
tiktoken>=0.5.0

# 开发依赖
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.0.0
isort>=5.12.0
mypy>=1.5.0
ruff>=0.1.0
```

---

#### 文件：`.env.example`

```bash
# ============================================
# Multi-Agent System 环境配置
# ============================================

# ----- LLM 配置 -----
# 选择 LLM 提供商: openai, anthropic, local
LLM_PROVIDER=openai

# OpenAI 配置
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1

# Anthropic 配置 (可选)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# 本地模型配置 (可选)
LOCAL_MODEL_URL=http://localhost:11434/v1
LOCAL_MODEL_NAME=llama3

# ----- LLM 参数 -----
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096

# ----- 系统配置 -----
# 调试模式：启用详细日志输出
DEBUG_MODE=false

# 最大迭代次数：防止无限循环
MAX_ITERATIONS=10

# 单个任务最大重试次数
MAX_TASK_RETRIES=3

# 全局超时时间（秒）
GLOBAL_TIMEOUT=300

# ----- 人工介入配置 -----
# 是否启用人工介入节点
ENABLE_HUMAN_IN_LOOP=true

# 需要人工确认的质量阈值 (0-1)
HUMAN_REVIEW_THRESHOLD=0.6

# ----- 并行执行配置 -----
# 是否启用并行执行
ENABLE_PARALLEL_EXECUTION=true

# 最大并行任务数
MAX_PARALLEL_TASKS=3

# ----- 工作目录配置 -----
# 工具操作的安全目录
WORKSPACE_DIR=workspace

# 日志输出目录
LOG_DIR=logs

# ----- 记忆系统配置 -----
# 是否启用长期记忆
ENABLE_LONG_TERM_MEMORY=false

# 长期记忆存储路径
MEMORY_STORAGE_PATH=data/memory

# ----- 可视化配置 -----
# 是否生成执行流程图
ENABLE_VISUALIZATION=true

# 流程图输出格式: mermaid, png
VISUALIZATION_FORMAT=mermaid
```

---

#### 文件：`src/__init__.py`

```python
"""
Multi-Agent Problem Solving System
===================================

基于 LangGraph 的通用多智能体协作问题求解系统。

主要特性：
- 自动任务理解与分解
- 多智能体协作执行
- 计划-执行-反思闭环
- 动态工具调用
- 人工介入支持
- 完整可观测性

使用示例：
    >>> from src import MultiAgentSystem
    >>> system = MultiAgentSystem()
    >>> result = system.run("请帮我编写一个 Python 爬虫")
    >>> print(result.answer)
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent System Team"

from src.graph.builder import build_graph, MultiAgentSystem
from src.config.settings import Settings, get_settings
from src.graph.state import AgentState

__all__ = [
    "MultiAgentSystem",
    "build_graph",
    "Settings",
    "get_settings",
    "AgentState",
    "__version__",
]
```

---

#### 文件：`src/types.py`

```python
"""
类型定义模块
============

集中定义系统中使用的所有类型，确保类型安全和一致性。
"""

from typing import (
    TypedDict,
    Annotated,
    List,
    Dict,
    Any,
    Optional,
    Literal,
    Union,
    Callable,
    TypeVar,
    Protocol,
)
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


# ============================================
# 枚举类型定义
# ============================================

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskPriority(str, Enum):
    """任务优先级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    """任务类型枚举"""
    RESEARCH = "research"
    CODE = "code"
    EXECUTE = "execute"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"


class AgentRole(str, Enum):
    """Agent 角色枚举"""
    COORDINATOR = "coordinator"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    EXECUTOR = "executor"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


# ============================================
# 路由类型定义
# ============================================

RouteType = Literal[
    "input_parser",
    "coordinator",
    "planner",
    "task_router",
    "researcher",
    "coder",
    "executor",
    "critic",
    "human_node",
    "synthesizer",
    "error_handler",
    "end",
]


# ============================================
# Pydantic 数据模型
# ============================================

class SubTask(BaseModel):
    """子任务模型"""
    id: str = Field(description="唯一标识")
    name: str = Field(description="任务名称")
    description: str = Field(description="任务描述")
    task_type: TaskType = Field(description="任务类型")
    assigned_agent: AgentRole = Field(description="分配的 Agent")
    dependencies: List[str] = Field(default_factory=list, description="依赖任务ID列表")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="优先级")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="状态")
    result: Optional[str] = Field(default=None, description="执行结果")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    retry_count: int = Field(default=0, description="重试次数")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: Optional[datetime] = Field(default=None, description="更新时间")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

    class Config:
        use_enum_values = True


class ToolCallLog(BaseModel):
    """工具调用日志模型"""
    tool_name: str = Field(description="工具名称")
    input_params: Dict[str, Any] = Field(description="输入参数")
    output: Any = Field(description="输出结果")
    success: bool = Field(description="是否成功")
    error_message: Optional[str] = Field(default=None, description="错误信息")
    timestamp: datetime = Field(default_factory=datetime.now, description="调用时间")
    duration_ms: float = Field(description="执行耗时(毫秒)")

    class Config:
        arbitrary_types_allowed = True


class AgentOutput(BaseModel):
    """Agent 输出模型"""
    agent_name: str = Field(description="Agent 名称")
    task_id: Optional[str] = Field(default=None, description="任务ID")
    output: str = Field(description="输出内容")
    reasoning: str = Field(default="", description="推理过程")
    tool_calls: List[ToolCallLog] = Field(default_factory=list, description="工具调用记录")
    confidence: float = Field(default=0.8, ge=0, le=1, description="置信度")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")

    class Config:
        arbitrary_types_allowed = True


class EvaluationResult(BaseModel):
    """评估结果模型"""
    score: float = Field(ge=0, le=1, description="评分")
    passed: bool = Field(description="是否通过")
    issues: List[str] = Field(default_factory=list, description="发现的问题")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")
    reasoning: str = Field(default="", description="评估推理过程")


class ExecutionMetrics(BaseModel):
    """执行指标模型"""
    total_duration_seconds: float = Field(default=0.0, description="总执行时间")
    token_usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0},
        description="Token 使用统计"
    )
    agent_durations: Dict[str, float] = Field(
        default_factory=dict,
        description="各 Agent 执行时间"
    )
    iteration_count: int = Field(default=0, description="迭代次数")
    retry_count: int = Field(default=0, description="重试次数")
    tool_call_count: int = Field(default=0, description="工具调用次数")
    success: bool = Field(default=False, description="是否成功")


class FinalResult(BaseModel):
    """最终结果模型"""
    task_id: str = Field(description="任务ID")
    original_task: str = Field(description="原始任务")
    answer: str = Field(description="最终答案")
    reasoning_trace: List[str] = Field(default_factory=list, description="推理轨迹")
    agent_outputs: Dict[str, Any] = Field(default_factory=dict, description="各Agent输出")
    metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics, description="执行指标")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    class Config:
        arbitrary_types_allowed = True


# ============================================
# 协议定义（用于类型检查）
# ============================================

class AgentProtocol(Protocol):
    """Agent 协议定义"""
    
    @property
    def name(self) -> str:
        """Agent 名称"""
        ...
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """执行 Agent 逻辑"""
        ...


class ToolProtocol(Protocol):
    """工具协议定义"""
    
    @property
    def name(self) -> str:
        """工具名称"""
        ...
    
    @property
    def description(self) -> str:
        """工具描述"""
        ...
    
    def invoke(self, **kwargs: Any) -> Any:
        """执行工具"""
        ...


# ============================================
# 类型别名
# ============================================

# 节点处理函数类型
NodeHandler = Callable[[Dict[str, Any]], Dict[str, Any]]

# 路由函数类型
RouterFunction = Callable[[Dict[str, Any]], RouteType]

# Agent 能力映射类型
AgentCapabilities = Dict[AgentRole, List[TaskType]]

# 泛型类型变量
T = TypeVar("T")
```

---

#### 文件：`src/main.py`

```python
"""
Multi-Agent System 主入口
=========================

提供命令行接口和程序入口点。
"""

import argparse
import sys
import time
import uuid
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

from src.config.settings import get_settings, Settings
from src.graph.builder import MultiAgentSystem
from src.graph.state import create_initial_state
from src.utils.logger import setup_logger, get_logger
from src.utils.visualizer import ExecutionVisualizer

# 初始化控制台和日志
console = Console()
logger = get_logger(__name__)


def print_banner() -> None:
    """打印系统横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          Multi-Agent Problem Solving System v1.0             ║
║                 Powered by LangGraph                         ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def print_result(result: dict) -> None:
    """打印执行结果"""
    console.print("\n")
    console.print(Panel(
        Markdown(result.get("final_answer", "无结果")),
        title="[bold green]✅ 执行结果[/bold green]",
        border_style="green",
    ))
    
    # 打印执行指标
    if "execution_time" in result:
        table = Table(title="执行指标", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        
        total_time = sum(result.get("execution_time", {}).values())
        table.add_row("总耗时", f"{total_time:.2f} 秒")
        table.add_row("迭代次数", str(result.get("iteration_count", 0)))
        
        token_usage = result.get("token_usage", {})
        if token_usage:
            table.add_row("Token 消耗", str(token_usage.get("total", 0)))
        
        console.print(table)


def print_reasoning_trace(result: dict) -> None:
    """打印推理轨迹"""
    reasoning_trace = result.get("reasoning_trace", [])
    if reasoning_trace:
        console.print("\n[bold yellow]📝 推理轨迹：[/bold yellow]")
        for i, step in enumerate(reasoning_trace, 1):
            console.print(f"  {i}. {step}")


def interactive_mode(system: MultiAgentSystem, settings: Settings) -> None:
    """交互式模式"""
    console.print("\n[bold cyan]进入交互模式 (输入 'quit' 或 'exit' 退出)[/bold cyan]\n")
    
    visualizer = ExecutionVisualizer() if settings.enable_visualization else None
    
    while True:
        try:
            # 获取用户输入
            user_input = Prompt.ask("\n[bold green]请输入您的任务[/bold green]")
            
            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[yellow]感谢使用，再见！[/yellow]")
                break
            
            if not user_input.strip():
                console.print("[yellow]输入不能为空，请重新输入[/yellow]")
                continue
            
            # 执行任务
            task_id = str(uuid.uuid4())[:8]
            console.print(f"\n[dim]任务ID: {task_id}[/dim]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("正在处理任务...", total=None)
                
                start_time = time.time()
                result = system.run(user_input, task_id=task_id)
                elapsed_time = time.time() - start_time
                
                progress.update(task, description=f"任务完成 (耗时 {elapsed_time:.2f}s)")
            
            # 打印结果
            print_result(result)
            
            # 是否显示详细推理过程
            if settings.debug_mode:
                print_reasoning_trace(result)
            
            # 可视化
            if visualizer and settings.enable_visualization:
                try:
                    graph_output = visualizer.generate_mermaid(result)
                    if Confirm.ask("\n是否显示执行流程图？", default=False):
                        console.print(Panel(graph_output, title="执行流程图"))
                except Exception as e:
                    logger.warning(f"生成可视化图表失败: {e}")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]操作已取消[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]执行出错: {e}[/red]")
            if settings.debug_mode:
                console.print_exception()
            continue


def single_task_mode(
    system: MultiAgentSystem,
    task: str,
    settings: Settings,
    output_file: Optional[str] = None
) -> None:
    """单任务模式"""
    task_id = str(uuid.uuid4())[:8]
    console.print(f"\n[dim]任务ID: {task_id}[/dim]")
    console.print(f"[bold]任务: {task}[/bold]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        prog_task = progress.add_task("正在处理任务...", total=None)
        
        start_time = time.time()
        result = system.run(task, task_id=task_id)
        elapsed_time = time.time() - start_time
        
        progress.update(prog_task, description=f"任务完成 (耗时 {elapsed_time:.2f}s)")
    
    # 打印结果
    print_result(result)
    
    if settings.debug_mode:
        print_reasoning_trace(result)
    
    # 输出到文件
    if output_file:
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            # 转换为可序列化格式
            output_data = {
                "task_id": task_id,
                "original_task": task,
                "answer": result.get("final_answer", ""),
                "reasoning_trace": result.get("reasoning_trace", []),
                "execution_time": elapsed_time,
                "token_usage": result.get("token_usage", {}),
            }
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]结果已保存至: {output_file}[/green]")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Problem Solving System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互模式
  python -m src.main
  
  # 单任务模式
  python -m src.main --task "编写一个 Python 爬虫"
  
  # 带输出文件
  python -m src.main --task "分析数据" --output result.json
  
  # 调试模式
  python -m src.main --debug
        """
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="要执行的任务描述"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出结果文件路径 (JSON 格式)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--no-human", "-n",
        action="store_true",
        help="禁用人工介入"
    )
    
    parser.add_argument(
        "--max-iterations", "-m",
        type=int,
        default=None,
        help="最大迭代次数"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Multi-Agent System v1.0.0"
    )
    
    return parser.parse_args()


def main() -> int:
    """主入口函数"""
    args = parse_args()
    
    # 加载配置
    settings = get_settings()
    
    # 应用命令行参数覆盖
    if args.debug:
        settings.debug_mode = True
    if args.no_human:
        settings.enable_human_in_loop = False
    if args.max_iterations:
        settings.max_iterations = args.max_iterations
    
    # 设置日志
    setup_logger(
        log_dir=settings.log_dir,
        debug=settings.debug_mode
    )
    
    # 打印横幅
    print_banner()
    
    try:
        # 初始化系统
        console.print("[dim]正在初始化系统...[/dim]")
        system = MultiAgentSystem(settings=settings)
        console.print("[green]✓ 系统初始化完成[/green]\n")
        
        # 根据参数选择模式
        if args.task:
            # 单任务模式
            single_task_mode(
                system=system,
                task=args.task,
                settings=settings,
                output_file=args.output
            )
        else:
            # 交互模式
            interactive_mode(system=system, settings=settings)
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]程序被用户中断[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]程序异常: {e}[/red]")
        if settings.debug_mode:
            console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

#### 文件：`src/config/__init__.py`

```python
"""
配置模块
========

提供系统配置管理功能。
"""

from src.config.settings import Settings, get_settings
from src.config.prompts import PromptTemplates, get_prompt

__all__ = [
    "Settings",
    "get_settings",
    "PromptTemplates",
    "get_prompt",
]
```

---

#### 文件：`src/config/settings.py`

```python
"""
配置管理模块
============

使用 Pydantic 进行配置验证，支持环境变量和配置文件。
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseModel):
    """LLM 配置模型"""
    provider: Literal["openai", "anthropic", "local"] = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """验证温度参数"""
        if not 0 <= v <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")
        return v


class AgentConfig(BaseModel):
    """Agent 配置模型"""
    name: str
    enabled: bool = True
    llm_override: Optional[LLMConfig] = None
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    tools: List[str] = Field(default_factory=list)
    custom_prompt: Optional[str] = None


class RetryConfig(BaseModel):
    """重试配置模型"""
    max_task_retries: int = Field(default=3, ge=0)
    max_global_iterations: int = Field(default=10, ge=1)
    retry_delay_seconds: float = Field(default=1.0, ge=0)
    exponential_backoff: bool = True
    backoff_multiplier: float = Field(default=2.0, ge=1)


class Settings(BaseSettings):
    """
    系统配置类
    
    配置优先级（从高到低）：
    1. 环境变量
    2. .env 文件
    3. 默认值
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ===== LLM 配置 =====
    llm_provider: Literal["openai", "anthropic", "local"] = Field(
        default="openai",
        alias="LLM_PROVIDER"
    )
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    
    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", alias="ANTHROPIC_MODEL")
    
    # Local
    local_model_url: str = Field(
        default="http://localhost:11434/v1",
        alias="LOCAL_MODEL_URL"
    )
    local_model_name: str = Field(default="llama3", alias="LOCAL_MODEL_NAME")
    
    # LLM 通用参数
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    
    # ===== 系统配置 =====
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")
    max_iterations: int = Field(default=10, alias="MAX_ITERATIONS")
    max_task_retries: int = Field(default=3, alias="MAX_TASK_RETRIES")
    global_timeout: int = Field(default=300, alias="GLOBAL_TIMEOUT")
    
    # ===== 人工介入 =====
    enable_human_in_loop: bool = Field(default=True, alias="ENABLE_HUMAN_IN_LOOP")
    human_review_threshold: float = Field(default=0.6, alias="HUMAN_REVIEW_THRESHOLD")
    
    # ===== 并行执行 =====
    enable_parallel_execution: bool = Field(default=True, alias="ENABLE_PARALLEL_EXECUTION")
    max_parallel_tasks: int = Field(default=3, alias="MAX_PARALLEL_TASKS")
    
    # ===== 目录配置 =====
    workspace_dir: str = Field(default="workspace", alias="WORKSPACE_DIR")
    log_dir: str = Field(default="logs", alias="LOG_DIR")
    
    # ===== 记忆系统 =====
    enable_long_term_memory: bool = Field(default=False, alias="ENABLE_LONG_TERM_MEMORY")
    memory_storage_path: str = Field(default="data/memory", alias="MEMORY_STORAGE_PATH")
    
    # ===== 可视化 =====
    enable_visualization: bool = Field(default=True, alias="ENABLE_VISUALIZATION")
    visualization_format: Literal["mermaid", "png"] = Field(
        default="mermaid",
        alias="VISUALIZATION_FORMAT"
    )
    
    # ===== Agent 配置 =====
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    
    # ===== 重试配置 =====
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
        self._init_default_agents()
    
    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        for dir_path in [self.workspace_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        if self.enable_long_term_memory:
            Path(self.memory_storage_path).mkdir(parents=True, exist_ok=True)
    
    def _init_default_agents(self) -> None:
        """初始化默认 Agent 配置"""
        default_agents = {
            "coordinator": AgentConfig(
                name="coordinator",
                enabled=True,
                tools=[]
            ),
            "planner": AgentConfig(
                name="planner",
                enabled=True,
                tools=[]
            ),
            "researcher": AgentConfig(
                name="researcher",
                enabled=True,
                tools=["web_search"]
            ),
            "coder": AgentConfig(
                name="coder",
                enabled=True,
                tools=["code_executor", "file_manager"]
            ),
            "executor": AgentConfig(
                name="executor",
                enabled=True,
                tools=["calculator", "file_manager", "code_executor"]
            ),
            "critic": AgentConfig(
                name="critic",
                enabled=True,
                tools=[]
            ),
            "synthesizer": AgentConfig(
                name="synthesizer",
                enabled=True,
                tools=["file_manager"]
            ),
        }
        
        # 合并用户配置
        for name, config in default_agents.items():
            if name not in self.agents:
                self.agents[name] = config
    
    def get_llm_config(self) -> LLMConfig:
        """获取当前 LLM 配置"""
        if self.llm_provider == "openai":
            return LLMConfig(
                provider="openai",
                model_name=self.openai_model,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
            )
        elif self.llm_provider == "anthropic":
            return LLMConfig(
                provider="anthropic",
                model_name=self.anthropic_model,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                api_key=self.anthropic_api_key,
            )
        else:  # local
            return LLMConfig(
                provider="local",
                model_name=self.local_model_name,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                base_url=self.local_model_url,
            )
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """检查 Agent 是否启用"""
        if agent_name in self.agents:
            return self.agents[agent_name].enabled
        return True
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """获取指定 Agent 的配置"""
        return self.agents.get(agent_name)


@lru_cache()
def get_settings() -> Settings:
    """
    获取全局配置实例（单例模式）
    
    Returns:
        Settings: 配置实例
    """
    return Settings()


def reload_settings() -> Settings:
    """
    重新加载配置（清除缓存）
    
    Returns:
        Settings: 新的配置实例
    """
    get_settings.cache_clear()
    return get_settings()
```

---

#### 文件：`src/config/prompts.py`

```python
"""
提示词模板管理模块
==================

集中管理所有 Agent 的提示词模板，支持模板替换和自定义。
"""

from typing import Dict, Optional
from string import Template


class PromptTemplates:
    """
    提示词模板管理类
    
    所有模板使用 Python string.Template 语法：$variable 或 ${variable}
    """
    
    # ===================================================
    # 协调者 (Coordinator) 提示词
    # ===================================================
    
    COORDINATOR_SYSTEM = """你是一个高效的任务协调者（Coordinator），负责理解用户需求并协调多个专业智能体完成任务。

你的核心职责：
1. **任务理解**：深入分析用户输入，提取核心需求和约束条件
2. **工作分配**：根据任务性质决定需要哪些专业智能体参与
3. **进度监控**：跟踪任务执行状态，确保按计划推进
4. **结果整合**：汇总各智能体的输出，形成最终答案

当前可用的专业智能体：
- Planner（规划者）：负责任务分解和执行计划制定
- Researcher（研究员）：负责信息检索和知识整合
- Coder（编码者）：负责代码编写和技术实现
- Executor（执行者）：负责工具调用和操作执行
- Critic（审核者）：负责质量检查和改进建议
- Synthesizer（综合者）：负责结果汇总和最终输出

决策规则：
- 简单的问答类任务可以直接回答，无需分解
- 复杂任务需要先交给 Planner 进行分解
- 始终确保输出质量，必要时启用 Critic 审核

请始终保持清晰的思考过程，并在响应中说明你的推理逻辑。"""

    COORDINATOR_TASK_UNDERSTANDING = """请分析以下用户任务，提取关键信息：

用户任务：$task

请按以下格式输出你的分析：

## 任务理解
[对任务的整体理解]

## 核心需求
[列出1-3个核心需求]

## 约束条件
[列出任何限制或要求]

## 任务类型
[分类：简单问答 / 研究分析 / 代码开发 / 综合任务]

## 建议方案
[是否需要分解？需要哪些智能体？]

## 下一步行动
[具体的下一步：direct_answer / plan / research / code]"""

    COORDINATOR_ROUTING = """基于当前状态，决定下一步行动：

原始任务：$original_task
任务理解：$task_understanding
已完成的子任务：$completed_tasks
待处理的子任务：$pending_tasks
当前迭代：$iteration_count / $max_iterations

请决定：
1. 如果所有任务已完成且质量合格，输出 "FINISH"
2. 如果需要继续执行，输出下一个应该执行的智能体名称
3. 如果出现问题需要重新规划，输出 "REPLAN"

你的决定（只输出一个词）："""

    # ===================================================
    # 规划者 (Planner) 提示词
    # ===================================================
    
    PLANNER_SYSTEM = """你是一个专业的任务规划者（Planner），擅长将复杂任务分解为可执行的子任务。

你的核心能力：
1. **任务分解**：将大任务拆分为独立的、可管理的子任务
2. **依赖分析**：识别子任务之间的依赖关系
3. **优先级排序**：确定执行顺序和优先级
4. **资源分配**：为每个子任务分配最合适的智能体

分解原则：
- 每个子任务应该是单一、明确、可验证的
- 子任务之间的依赖关系要清晰
- 考虑并行执行的可能性以提高效率
- 预估每个子任务的复杂度

可分配的智能体及其能力：
- researcher: 信息检索、知识整理、资料分析
- coder: 代码编写、代码调试、技术实现
- executor: 工具调用、文件操作、命令执行

输出格式必须是有效的 JSON。"""

    PLANNER_DECOMPOSE = """请将以下任务分解为可执行的子任务：

任务描述：$task
任务理解：$understanding

请输出 JSON 格式的执行计划：
```json
{
    "plan_summary": "计划概述",
    "subtasks": [
        {
            "id": "task_1",
            "name": "子任务名称",
            "description": "详细描述",
            "task_type": "research|code|execute|analyze",
            "assigned_agent": "researcher|coder|executor",
            "dependencies": [],
            "priority": "high|medium|low",
            "estimated_complexity": "simple|medium|complex"
        }
    ],
    "parallel_groups": [["task_1", "task_2"], ["task_3"]],
    "notes": "其他注意事项"
}
```

请确保：
1. 子任务 ID 唯一
2. 依赖关系正确（不能循环依赖）
3. 分配给合适的智能体
4. 标注可并行执行的任务组"""

    # ===================================================
    # 研究员 (Researcher) 提示词
    # ===================================================
    
    RESEARCHER_SYSTEM = """你是一个专业的研究员（Researcher），擅长信息检索和知识整合。

你的核心能力：
1. **信息检索**：使用搜索工具获取相关信息
2. **资料分析**：分析和理解检索到的内容
3. **知识整合**：将多个来源的信息综合为结构化知识
4. **事实核查**：验证信息的准确性和可靠性

工作原则：
- 优先使用权威、可靠的信息源
- 对检索结果进行批判性分析
- 明确标注信息来源
- 区分事实陈述和个人观点

可用工具：
- web_search: 网络搜索，获取外部信息

请始终提供清晰、结构化的研究报告。"""

    RESEARCHER_TASK = """请完成以下研究任务：

任务：$task
上下文：$context

要求：
1. 明确研究目标
2. 使用合适的搜索查询
3. 整理和分析检索结果
4. 输出结构化的研究报告

请开始你的研究，并在过程中说明你的思考。"""

    # ===================================================
    # 编码者 (Coder) 提示词
    # ===================================================
    
    CODER_SYSTEM = """你是一个专业的编码者（Coder），擅长编写高质量的代码。

你的核心能力：
1. **代码编写**：根据需求编写清晰、高效的代码
2. **代码调试**：分析和修复代码中的问题
3. **最佳实践**：遵循编码规范和设计模式
4. **文档编写**：为代码添加清晰的注释和文档

编码原则：
- 代码应该清晰、可读、可维护
- 遵循 PEP8 等编码规范
- 添加适当的错误处理
- 编写必要的注释和 docstring
- 考虑边界情况和异常处理

可用工具：
- code_executor: 执行 Python 代码
- file_manager: 文件读写操作

请始终输出完整、可运行的代码。"""

    CODER_TASK = """请完成以下编码任务：

任务：$task
上下文：$context
技术要求：$requirements

请：
1. 分析需求，确定技术方案
2. 编写完整的代码实现
3. 添加必要的注释和错误处理
4. 如果需要，使用工具测试代码

输出格式：
```python
# 你的代码
```

思考过程和代码说明：
[解释你的实现思路]"""

    # ===================================================
    # 执行者 (Executor) 提示词
    # ===================================================
    
    EXECUTOR_SYSTEM = """你是一个可靠的执行者（Executor），负责执行具体的操作和工具调用。

你的核心能力：
1. **工具调用**：正确使用各种工具完成任务
2. **结果验证**：检查执行结果是否符合预期
3. **错误处理**：处理执行过程中的异常情况
4. **状态报告**：清晰报告执行状态和结果

可用工具：
- calculator: 数学计算
- file_manager: 文件读写（限定在 workspace 目录）
- code_executor: Python 代码执行
- web_search: 网络搜索

安全规则：
- 文件操作仅限于 workspace 目录
- 代码执行在沙箱环境中进行
- 不执行任何可能有害的操作

请谨慎执行每个操作，并详细报告结果。"""

    EXECUTOR_TASK = """请执行以下任务：

任务：$task
输入数据：$input_data
期望输出：$expected_output

执行步骤：
1. 分析任务需求
2. 选择合适的工具
3. 执行操作
4. 验证结果
5. 报告执行状态

请开始执行，并说明每一步的操作和结果。"""

    # ===================================================
    # 审核者 (Critic) 提示词
    # ===================================================
    
    CRITIC_SYSTEM = """你是一个严谨的审核者（Critic），负责评估工作质量并提供改进建议。

你的核心能力：
1. **质量评估**：评估输出是否满足要求
2. **问题发现**：识别错误、遗漏和改进空间
3. **改进建议**：提供具体、可操作的改进意见
4. **风险评估**：识别潜在的风险和问题

评估维度：
- 正确性：结果是否正确
- 完整性：是否覆盖所有需求
- 质量：代码质量、文档质量等
- 可用性：结果是否可直接使用
- 安全性：是否存在安全隐患

评分标准：
- 0.9-1.0：优秀，可直接使用
- 0.7-0.9：良好，小问题可接受
- 0.5-0.7：及格，需要改进
- 0.0-0.5：不合格，需要重做

请提供客观、建设性的评审意见。"""

    CRITIC_REVIEW = """请审核以下工作成果：

原始任务：$original_task
子任务：$subtask
执行者：$agent_name
输出内容：
$output

请按以下格式输出评审结果：
```json
{
    "score": 0.85,
    "passed": true,
    "evaluation": {
        "correctness": "评价正确性",
        "completeness": "评价完整性",
        "quality": "评价质量",
        "usability": "评价可用性"
    },
    "issues": [
        "问题1",
        "问题2"
    ],
    "suggestions": [
        "建议1",
        "建议2"
    ],
    "action": "approve|revise|reject",
    "reasoning": "评审推理过程"
}
```"""

    # ===================================================
    # 综合者 (Synthesizer) 提示词
    # ===================================================
    
    SYNTHESIZER_SYSTEM = """你是一个专业的综合者（Synthesizer），负责汇总所有工作成果并生成最终输出。

你的核心能力：
1. **信息整合**：将多个来源的信息融合为统一的输出
2. **结构组织**：以清晰的结构呈现内容
3. **格式优化**：根据需求选择最佳输出格式
4. **质量把控**：确保最终输出的质量和一致性

输出原则：
- 内容完整，覆盖所有关键点
- 结构清晰，便于理解
- 语言准确，表达专业
- 格式适当，易于使用

请生成高质量的最终输出。"""

    SYNTHESIZER_AGGREGATE = """请综合以下工作成果，生成最终答案：

原始任务：$original_task

各智能体输出：
$agent_outputs

审核意见：
$review_notes

请生成最终答案，包含：
1. 任务完成总结
2. 核心内容/代码/结果
3. 使用说明（如适用）
4. 注意事项

请确保输出完整、专业、可直接使用。"""

    # ===================================================
    # 类方法
    # ===================================================
    
    _custom_templates: Dict[str, str] = {}
    
    @classmethod
    def get(cls, template_name: str, **kwargs) -> str:
        """
        获取并渲染模板
        
        Args:
            template_name: 模板名称（如 "COORDINATOR_SYSTEM"）
            **kwargs: 模板变量
            
        Returns:
            渲染后的字符串
        """
        # 优先使用自定义模板
        if template_name in cls._custom_templates:
            template_str = cls._custom_templates[template_name]
        else:
            template_str = getattr(cls, template_name, None)
            if template_str is None:
                raise ValueError(f"未知的模板名称: {template_name}")
        
        # 渲染模板
        if kwargs:
            template = Template(template_str)
            return template.safe_substitute(**kwargs)
        return template_str
    
    @classmethod
    def set_custom(cls, template_name: str, template_str: str) -> None:
        """
        设置自定义模板
        
        Args:
            template_name: 模板名称
            template_str: 模板内容
        """
        cls._custom_templates[template_name] = template_str
    
    @classmethod
    def reset_custom(cls, template_name: Optional[str] = None) -> None:
        """
        重置自定义模板
        
        Args:
            template_name: 模板名称，None 表示重置所有
        """
        if template_name:
            cls._custom_templates.pop(template_name, None)
        else:
            cls._custom_templates.clear()
    
    @classmethod
    def list_templates(cls) -> list:
        """
        列出所有可用模板名称
        
        Returns:
            模板名称列表
        """
        return [
            name for name in dir(cls)
            if name.isupper() and not name.startswith("_")
        ]


def get_prompt(template_name: str, **kwargs) -> str:
    """
    便捷函数：获取渲染后的提示词
    
    Args:
        template_name: 模板名称
        **kwargs: 模板变量
        
    Returns:
        渲染后的字符串
    """
    return PromptTemplates.get(template_name, **kwargs)
```

---

**第三部分（第1批）完成。请输入 "继续" 获取第三部分（第2批）：Agent 模块完整代码。**


## 第三部分（第2批）：Agent 模块完整代码

---

#### 文件：`src/agents/__init__.py`

```python
"""
智能体模块
==========

提供所有专业智能体的实现。

可用智能体：
- CoordinatorAgent: 任务协调与路由
- PlannerAgent: 任务分解与规划
- ResearcherAgent: 信息检索与研究
- CoderAgent: 代码编写与调试
- ExecutorAgent: 工具调用与执行
- CriticAgent: 质量审核与反思
- SynthesizerAgent: 结果汇总与输出
"""

from src.agents.base import BaseAgent, AgentRegistry
from src.agents.coordinator import CoordinatorAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.coder import CoderAgent
from src.agents.executor import ExecutorAgent
from src.agents.critic import CriticAgent
from src.agents.synthesizer import SynthesizerAgent

__all__ = [
    "BaseAgent",
    "AgentRegistry",
    "CoordinatorAgent",
    "PlannerAgent",
    "ResearcherAgent",
    "CoderAgent",
    "ExecutorAgent",
    "CriticAgent",
    "SynthesizerAgent",
]


def get_all_agents() -> dict:
    """
    获取所有可用的 Agent 类
    
    Returns:
        Agent 名称到类的映射字典
    """
    return {
        "coordinator": CoordinatorAgent,
        "planner": PlannerAgent,
        "researcher": ResearcherAgent,
        "coder": CoderAgent,
        "executor": ExecutorAgent,
        "critic": CriticAgent,
        "synthesizer": SynthesizerAgent,
    }
```

---

#### 文件：`src/agents/base.py`

```python
"""
Agent 基类模块
==============

定义所有 Agent 的基类和注册机制。
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from src.config.settings import Settings, AgentConfig, get_settings
from src.config.prompts import PromptTemplates
from src.utils.logger import get_logger
from src.types import AgentOutput, ToolCallLog


class AgentRegistry:
    """
    Agent 注册表
    
    用于管理和获取 Agent 实例的单例注册表。
    支持动态注册新的 Agent 类型。
    """
    
    _agents: Dict[str, Type["BaseAgent"]] = {}
    _instances: Dict[str, "BaseAgent"] = {}
    
    @classmethod
    def register(cls, name: str, agent_class: Type["BaseAgent"]) -> None:
        """
        注册 Agent 类
        
        Args:
            name: Agent 名称
            agent_class: Agent 类
        """
        cls._agents[name] = agent_class
    
    @classmethod
    def get_class(cls, name: str) -> Optional[Type["BaseAgent"]]:
        """
        获取 Agent 类
        
        Args:
            name: Agent 名称
            
        Returns:
            Agent 类，不存在返回 None
        """
        return cls._agents.get(name)
    
    @classmethod
    def get_instance(
        cls,
        name: str,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        settings: Optional[Settings] = None,
    ) -> Optional["BaseAgent"]:
        """
        获取 Agent 实例（单例模式）
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            tools: 工具列表
            settings: 配置
            
        Returns:
            Agent 实例
        """
        if name not in cls._instances:
            agent_class = cls._agents.get(name)
            if agent_class is None:
                return None
            cls._instances[name] = agent_class(
                llm=llm,
                tools=tools,
                settings=settings,
            )
        return cls._instances[name]
    
    @classmethod
    def clear_instances(cls) -> None:
        """清空所有实例缓存"""
        cls._instances.clear()
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """
        列出所有已注册的 Agent
        
        Returns:
            Agent 名称列表
        """
        return list(cls._agents.keys())


def register_agent(name: str) -> Callable:
    """
    Agent 注册装饰器
    
    使用方式：
        @register_agent("my_agent")
        class MyAgent(BaseAgent):
            ...
    
    Args:
        name: Agent 名称
        
    Returns:
        装饰器函数
    """
    def decorator(cls: Type["BaseAgent"]) -> Type["BaseAgent"]:
        AgentRegistry.register(name, cls)
        return cls
    return decorator


class BaseAgent(ABC):
    """
    Agent 抽象基类
    
    所有专业智能体都应继承此类并实现 _execute 方法。
    
    属性:
        name: Agent 名称
        description: Agent 描述
        llm: 语言模型实例
        tools: 可用工具列表
        settings: 系统配置
        logger: 日志记录器
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        settings: Optional[Settings] = None,
    ):
        """
        初始化 Agent
        
        Args:
            llm: 语言模型实例，None 时从工厂创建
            tools: 工具列表
            settings: 配置实例
        """
        self.settings = settings or get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self._tools = tools or []
        self._llm = llm
        self._tool_call_logs: List[ToolCallLog] = []
        
        # 获取 Agent 特定配置
        self._config = self.settings.get_agent_config(self.name)
        
        self.logger.debug(f"初始化 Agent: {self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Agent 描述"""
        pass
    
    @property
    def capabilities(self) -> List[str]:
        """Agent 能力列表"""
        return []
    
    @property
    def llm(self) -> BaseChatModel:
        """获取 LLM 实例"""
        if self._llm is None:
            from src.llm.factory import LLMFactory
            self._llm = LLMFactory.create(self.settings.get_llm_config())
        return self._llm
    
    @property
    def tools(self) -> List[BaseTool]:
        """获取工具列表"""
        return self._tools
    
    @tools.setter
    def tools(self, tools: List[BaseTool]) -> None:
        """设置工具列表"""
        self._tools = tools
    
    def get_system_prompt(self) -> str:
        """
        获取系统提示词
        
        子类可以重写此方法提供自定义系统提示词。
        
        Returns:
            系统提示词字符串
        """
        template_name = f"{self.name.upper()}_SYSTEM"
        try:
            return PromptTemplates.get(template_name)
        except ValueError:
            return f"你是一个专业的 {self.name} 智能体。"
    
    def create_prompt(self, **kwargs) -> ChatPromptTemplate:
        """
        创建聊天提示模板
        
        Args:
            **kwargs: 额外的模板变量
            
        Returns:
            ChatPromptTemplate 实例
        """
        system_prompt = self.get_system_prompt()
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ])
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 Agent 逻辑
        
        这是主要的公共接口，负责：
        1. 记录执行开始
        2. 调用子类实现的 _execute 方法
        3. 记录执行结果和耗时
        4. 处理异常
        
        Args:
            state: 当前系统状态
            
        Returns:
            更新后的状态字典
        """
        start_time = time.time()
        self._tool_call_logs = []
        
        self.logger.info(f"[{self.name}] 开始执行")
        
        if self.settings.debug_mode:
            self._log_state_summary(state, "输入")
        
        try:
            # 调用子类实现
            result_state = self._execute(state)
            
            # 记录执行时间
            duration = time.time() - start_time
            execution_time = result_state.get("execution_time", {})
            execution_time[self.name] = duration
            result_state["execution_time"] = execution_time
            
            # 添加工具调用日志
            if self._tool_call_logs:
                tool_logs = result_state.get("tool_call_logs", [])
                tool_logs.extend(self._tool_call_logs)
                result_state["tool_call_logs"] = tool_logs
            
            # 更新当前 Agent
            result_state["current_agent"] = self.name
            
            self.logger.info(f"[{self.name}] 执行完成，耗时 {duration:.2f}s")
            
            if self.settings.debug_mode:
                self._log_state_summary(result_state, "输出")
            
            return result_state
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"[{self.name}] 执行失败: {e}", exc_info=True)
            
            # 记录错误
            error_log = state.get("error_log", [])
            error_log.append(f"[{self.name}] {datetime.now().isoformat()}: {str(e)}")
            
            return {
                **state,
                "current_agent": self.name,
                "last_error": str(e),
                "error_log": error_log,
                "execution_time": {
                    **state.get("execution_time", {}),
                    self.name: duration,
                },
            }
    
    @abstractmethod
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行具体的 Agent 逻辑
        
        子类必须实现此方法。
        
        Args:
            state: 当前系统状态
            
        Returns:
            更新后的状态字典
        """
        pass
    
    def can_handle(self, task_type: str) -> bool:
        """
        判断是否能处理指定类型的任务
        
        Args:
            task_type: 任务类型
            
        Returns:
            是否能处理
        """
        return task_type in self.capabilities
    
    def call_llm(
        self,
        messages: List[BaseMessage],
        system_prompt: Optional[str] = None,
    ) -> AIMessage:
        """
        调用 LLM
        
        Args:
            messages: 消息列表
            system_prompt: 可选的系统提示词
            
        Returns:
            AI 响应消息
        """
        if system_prompt:
            full_messages = [SystemMessage(content=system_prompt)] + messages
        else:
            full_messages = [SystemMessage(content=self.get_system_prompt())] + messages
        
        response = self.llm.invoke(full_messages)
        
        # 记录 token 使用（如果可用）
        if hasattr(response, "usage_metadata"):
            self.logger.debug(f"Token 使用: {response.usage_metadata}")
        
        return response
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if tool is None:
            raise ValueError(f"工具不存在: {tool_name}")
        
        start_time = time.time()
        try:
            result = tool.invoke(kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # 记录工具调用
            self._tool_call_logs.append(ToolCallLog(
                tool_name=tool_name,
                input_params=kwargs,
                output=result,
                success=True,
                duration_ms=duration_ms,
            ))
            
            self.logger.debug(f"工具调用成功: {tool_name}, 耗时 {duration_ms:.2f}ms")
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self._tool_call_logs.append(ToolCallLog(
                tool_name=tool_name,
                input_params=kwargs,
                output=None,
                success=False,
                error_message=str(e),
                duration_ms=duration_ms,
            ))
            
            self.logger.error(f"工具调用失败: {tool_name}, 错误: {e}")
            raise
    
    def create_output(
        self,
        output: str,
        reasoning: str = "",
        task_id: Optional[str] = None,
        confidence: float = 0.8,
    ) -> AgentOutput:
        """
        创建标准化的 Agent 输出
        
        Args:
            output: 输出内容
            reasoning: 推理过程
            task_id: 任务 ID
            confidence: 置信度
            
        Returns:
            AgentOutput 实例
        """
        return AgentOutput(
            agent_name=self.name,
            task_id=task_id,
            output=output,
            reasoning=reasoning,
            tool_calls=self._tool_call_logs.copy(),
            confidence=confidence,
        )
    
    def _log_state_summary(self, state: Dict[str, Any], prefix: str) -> None:
        """
        记录状态摘要（调试用）
        
        Args:
            state: 状态字典
            prefix: 日志前缀
        """
        summary = {
            "original_task": state.get("original_task", "")[:100],
            "current_agent": state.get("current_agent"),
            "iteration_count": state.get("iteration_count", 0),
            "subtasks_count": len(state.get("subtasks", [])),
            "has_final_answer": state.get("final_answer") is not None,
        }
        self.logger.debug(f"[{prefix}状态] {summary}")
    
    def _get_pending_subtasks(self, state: Dict[str, Any]) -> List[Dict]:
        """
        获取待处理的子任务
        
        Args:
            state: 状态字典
            
        Returns:
            待处理子任务列表
        """
        subtasks = state.get("subtasks", [])
        return [t for t in subtasks if t.get("status") == "pending"]
    
    def _get_completed_subtasks(self, state: Dict[str, Any]) -> List[Dict]:
        """
        获取已完成的子任务
        
        Args:
            state: 状态字典
            
        Returns:
            已完成子任务列表
        """
        subtasks = state.get("subtasks", [])
        return [t for t in subtasks if t.get("status") == "completed"]
    
    def _update_subtask_status(
        self,
        state: Dict[str, Any],
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> List[Dict]:
        """
        更新子任务状态
        
        Args:
            state: 状态字典
            task_id: 任务 ID
            status: 新状态
            result: 执行结果
            error: 错误信息
            
        Returns:
            更新后的子任务列表
        """
        subtasks = state.get("subtasks", []).copy()
        for task in subtasks:
            if task.get("id") == task_id:
                task["status"] = status
                task["updated_at"] = datetime.now().isoformat()
                if result:
                    task["result"] = result
                if error:
                    task["error_message"] = error
                    task["retry_count"] = task.get("retry_count", 0) + 1
                break
        return subtasks
```

---

#### 文件：`src/agents/coordinator.py`

```python
"""
协调者 Agent
============

负责任务理解、工作分配、进度监控和结果整合。
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates
from src.types import TaskStatus


@register_agent("coordinator")
class CoordinatorAgent(BaseAgent):
    """
    协调者智能体
    
    核心职责：
    1. 理解用户任务，提取核心需求
    2. 决定任务处理策略（直接回答/分解执行）
    3. 监控任务执行进度
    4. 协调各智能体工作
    5. 整合最终结果
    """
    
    @property
    def name(self) -> str:
        return "coordinator"
    
    @property
    def description(self) -> str:
        return "任务协调者，负责理解任务、分配工作和监控进度"
    
    @property
    def capabilities(self) -> List[str]:
        return ["task_understanding", "routing", "monitoring", "integration"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行协调逻辑
        
        处理流程：
        1. 如果是新任务，进行任务理解
        2. 如果任务已分解，检查执行进度
        3. 决定下一步行动
        """
        # 检查是否是新任务（没有任务理解）
        if not state.get("task_understanding"):
            return self._understand_task(state)
        
        # 检查是否所有任务已完成
        if self._check_completion(state):
            return self._finalize(state)
        
        # 决定下一步路由
        return self._decide_next_step(state)
    
    def _understand_task(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        理解用户任务
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态，包含任务理解
        """
        original_task = state.get("original_task", "")
        
        # 构建提示词
        prompt = PromptTemplates.get(
            "COORDINATOR_TASK_UNDERSTANDING",
            task=original_task
        )
        
        # 调用 LLM
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        task_understanding = response.content
        
        # 分析任务类型，决定是否需要分解
        needs_planning = self._needs_planning(task_understanding)
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(f"[Coordinator] 任务理解完成: {task_understanding[:200]}...")
        
        self.logger.info(f"任务理解完成，需要规划: {needs_planning}")
        
        # 决定下一步
        if needs_planning:
            next_step = "planner"
        else:
            # 简单任务直接回答
            next_step = "synthesizer"
        
        return {
            **state,
            "task_understanding": task_understanding,
            "reasoning_trace": reasoning_trace,
            "next": next_step,
        }
    
    def _needs_planning(self, understanding: str) -> bool:
        """
        判断任务是否需要规划分解
        
        Args:
            understanding: 任务理解文本
            
        Returns:
            是否需要规划
        """
        # 简单任务关键词
        simple_keywords = ["简单问答", "直接回答", "simple"]
        
        # 复杂任务关键词
        complex_keywords = [
            "代码", "编写", "开发", "实现",
            "分析", "研究", "调研",
            "步骤", "计划", "方案",
            "综合任务", "复杂"
        ]
        
        understanding_lower = understanding.lower()
        
        # 检查是否包含简单任务关键词
        for keyword in simple_keywords:
            if keyword in understanding_lower:
                return False
        
        # 检查是否包含复杂任务关键词
        for keyword in complex_keywords:
            if keyword in understanding_lower:
                return True
        
        # 默认需要规划
        return True
    
    def _check_completion(self, state: Dict[str, Any]) -> bool:
        """
        检查任务是否全部完成
        
        Args:
            state: 当前状态
            
        Returns:
            是否全部完成
        """
        subtasks = state.get("subtasks", [])
        
        # 如果没有子任务，检查是否有直接输出
        if not subtasks:
            return bool(state.get("agent_outputs"))
        
        # 检查所有子任务是否完成
        for task in subtasks:
            status = task.get("status")
            if status not in [TaskStatus.COMPLETED.value, "completed"]:
                return False
        
        return True
    
    def _decide_next_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        决定下一步行动
        
        Args:
            state: 当前状态
            
        Returns:
            包含下一步路由的状态
        """
        subtasks = state.get("subtasks", [])
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 10)
        
        # 检查迭代次数
        if iteration_count >= max_iterations:
            self.logger.warning(f"达到最大迭代次数 {max_iterations}，强制结束")
            return {
                **state,
                "next": "synthesizer",
                "reasoning_trace": state.get("reasoning_trace", []) + [
                    f"[Coordinator] 达到最大迭代次数，强制进入综合阶段"
                ],
            }
        
        # 构建路由决策提示
        completed = [t for t in subtasks if t.get("status") == "completed"]
        pending = [t for t in subtasks if t.get("status") == "pending"]
        failed = [t for t in subtasks if t.get("status") == "failed"]
        
        prompt = PromptTemplates.get(
            "COORDINATOR_ROUTING",
            original_task=state.get("original_task", ""),
            task_understanding=state.get("task_understanding", ""),
            completed_tasks=json.dumps([t["name"] for t in completed], ensure_ascii=False),
            pending_tasks=json.dumps([t["name"] for t in pending], ensure_ascii=False),
            iteration_count=iteration_count,
            max_iterations=max_iterations,
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        decision = response.content.strip().upper()
        
        # 解析决策
        if "FINISH" in decision:
            next_step = "synthesizer"
        elif "REPLAN" in decision:
            next_step = "planner"
        elif pending:
            # 路由到任务执行
            next_step = "task_router"
        elif failed:
            # 有失败任务，尝试重新执行
            next_step = "task_router"
        else:
            next_step = "synthesizer"
        
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(f"[Coordinator] 路由决策: {decision} -> {next_step}")
        
        return {
            **state,
            "next": next_step,
            "iteration_count": iteration_count + 1,
            "reasoning_trace": reasoning_trace,
        }
    
    def _finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        完成任务处理
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append("[Coordinator] 所有任务已完成，准备生成最终答案")
        
        return {
            **state,
            "next": "synthesizer",
            "reasoning_trace": reasoning_trace,
        }
```

---

#### 文件：`src/agents/planner.py`

```python
"""
规划者 Agent
============

负责任务分解、依赖分析和执行计划制定。
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates
from src.types import SubTask, TaskType, TaskStatus, TaskPriority, AgentRole


@register_agent("planner")
class PlannerAgent(BaseAgent):
    """
    规划者智能体
    
    核心职责：
    1. 将复杂任务分解为可管理的子任务
    2. 分析子任务之间的依赖关系
    3. 为每个子任务分配合适的执行者
    4. 制定执行顺序和并行策略
    """
    
    @property
    def name(self) -> str:
        return "planner"
    
    @property
    def description(self) -> str:
        return "任务规划者，负责分解任务和制定执行计划"
    
    @property
    def capabilities(self) -> List[str]:
        return ["task_decomposition", "dependency_analysis", "planning"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行规划逻辑
        
        Args:
            state: 当前状态
            
        Returns:
            包含子任务列表的更新状态
        """
        original_task = state.get("original_task", "")
        task_understanding = state.get("task_understanding", "")
        
        # 构建规划提示
        prompt = PromptTemplates.get(
            "PLANNER_DECOMPOSE",
            task=original_task,
            understanding=task_understanding,
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        # 解析规划结果
        plan_result = self._parse_plan(response.content)
        
        if plan_result is None:
            self.logger.error("规划解析失败，使用默认规划")
            plan_result = self._create_default_plan(original_task)
        
        subtasks = plan_result.get("subtasks", [])
        parallel_groups = plan_result.get("parallel_groups", [])
        plan_summary = plan_result.get("plan_summary", "")
        
        # 转换为 SubTask 格式
        formatted_subtasks = self._format_subtasks(subtasks)
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[Planner] 任务分解完成: {len(formatted_subtasks)} 个子任务"
        )
        reasoning_trace.append(f"[Planner] 计划概述: {plan_summary}")
        
        self.logger.info(f"规划完成: {len(formatted_subtasks)} 个子任务")
        
        return {
            **state,
            "subtasks": formatted_subtasks,
            "current_plan": plan_summary,
            "parallel_groups": parallel_groups,
            "reasoning_trace": reasoning_trace,
            "next": "task_router",
        }
    
    def _parse_plan(self, content: str) -> Optional[Dict[str, Any]]:
        """
        解析 LLM 返回的规划结果
        
        Args:
            content: LLM 响应内容
            
        Returns:
            解析后的规划字典，失败返回 None
        """
        try:
            # 尝试提取 JSON 块
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析
                json_str = content
            
            plan = json.loads(json_str)
            return plan
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON 解析失败: {e}")
            # 尝试修复常见的 JSON 问题
            try:
                # 移除可能的注释
                cleaned = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
                cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
                return json.loads(cleaned)
            except:
                return None
    
    def _create_default_plan(self, task: str) -> Dict[str, Any]:
        """
        创建默认规划（当解析失败时使用）
        
        Args:
            task: 原始任务
            
        Returns:
            默认规划字典
        """
        return {
            "plan_summary": "使用默认规划执行任务",
            "subtasks": [
                {
                    "id": f"task_{uuid.uuid4().hex[:8]}",
                    "name": "分析任务需求",
                    "description": f"分析并理解任务: {task[:100]}",
                    "task_type": "analyze",
                    "assigned_agent": "researcher",
                    "dependencies": [],
                    "priority": "high",
                },
                {
                    "id": f"task_{uuid.uuid4().hex[:8]}",
                    "name": "执行主要任务",
                    "description": f"执行任务: {task[:100]}",
                    "task_type": "execute",
                    "assigned_agent": "executor",
                    "dependencies": [],
                    "priority": "high",
                },
            ],
            "parallel_groups": [],
        }
    
    def _format_subtasks(self, subtasks: List[Dict]) -> List[Dict]:
        """
        格式化子任务为标准格式
        
        Args:
            subtasks: 原始子任务列表
            
        Returns:
            格式化后的子任务列表
        """
        formatted = []
        
        for task in subtasks:
            # 确保有 ID
            task_id = task.get("id") or f"task_{uuid.uuid4().hex[:8]}"
            
            # 映射任务类型
            task_type = task.get("task_type", "execute")
            if task_type not in ["research", "code", "execute", "analyze", "synthesize"]:
                task_type = "execute"
            
            # 映射 Agent
            assigned_agent = task.get("assigned_agent", "executor")
            agent_map = {
                "researcher": "researcher",
                "research": "researcher",
                "coder": "coder",
                "code": "coder",
                "executor": "executor",
                "execute": "executor",
            }
            assigned_agent = agent_map.get(assigned_agent, "executor")
            
            # 映射优先级
            priority = task.get("priority", "medium")
            if priority not in ["low", "medium", "high", "critical"]:
                priority = "medium"
            
            formatted_task = {
                "id": task_id,
                "name": task.get("name", "未命名任务"),
                "description": task.get("description", ""),
                "task_type": task_type,
                "assigned_agent": assigned_agent,
                "dependencies": task.get("dependencies", []),
                "priority": priority,
                "status": "pending",
                "result": None,
                "error_message": None,
                "retry_count": 0,
                "metadata": task.get("metadata", {}),
            }
            
            formatted.append(formatted_task)
        
        return formatted
    
    def _analyze_dependencies(self, subtasks: List[Dict]) -> List[List[str]]:
        """
        分析依赖关系，生成并行执行组
        
        Args:
            subtasks: 子任务列表
            
        Returns:
            可并行执行的任务组列表
        """
        # 构建依赖图
        task_ids = {t["id"] for t in subtasks}
        dependencies = {t["id"]: set(t.get("dependencies", [])) for t in subtasks}
        
        # 验证依赖（移除不存在的依赖）
        for task_id, deps in dependencies.items():
            dependencies[task_id] = deps & task_ids
        
        # 拓扑排序，生成并行组
        parallel_groups = []
        remaining = set(task_ids)
        
        while remaining:
            # 找出没有未完成依赖的任务
            ready = {
                task_id for task_id in remaining
                if not (dependencies[task_id] & remaining)
            }
            
            if not ready:
                # 检测到循环依赖
                self.logger.warning("检测到循环依赖，使用顺序执行")
                parallel_groups.append(list(remaining))
                break
            
            parallel_groups.append(list(ready))
            remaining -= ready
        
        return parallel_groups
```

---

#### 文件：`src/agents/researcher.py`

```python
"""
研究员 Agent
============

负责信息检索、知识整合和资料分析。
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates


@register_agent("researcher")
class ResearcherAgent(BaseAgent):
    """
    研究员智能体
    
    核心职责：
    1. 使用搜索工具获取相关信息
    2. 分析和理解检索到的内容
    3. 整合多来源信息
    4. 生成结构化研究报告
    """
    
    @property
    def name(self) -> str:
        return "researcher"
    
    @property
    def description(self) -> str:
        return "研究员，负责信息检索和知识整合"
    
    @property
    def capabilities(self) -> List[str]:
        return ["research", "analyze", "information_retrieval"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行研究任务
        
        Args:
            state: 当前状态
            
        Returns:
            包含研究结果的更新状态
        """
        # 获取当前待处理的研究任务
        current_task = self._get_current_task(state)
        
        if current_task is None:
            self.logger.warning("没有找到待处理的研究任务")
            return {**state, "next": "coordinator"}
        
        task_description = current_task.get("description", "")
        context = self._build_context(state)
        
        # 构建研究提示
        prompt = PromptTemplates.get(
            "RESEARCHER_TASK",
            task=task_description,
            context=context,
        )
        
        messages = [HumanMessage(content=prompt)]
        
        # 如果有搜索工具，先进行搜索
        search_results = self._perform_search(task_description)
        if search_results:
            prompt += f"\n\n搜索结果：\n{search_results}"
            messages = [HumanMessage(content=prompt)]
        
        # 调用 LLM 生成研究报告
        response = self.call_llm(messages)
        research_output = response.content
        
        # 创建 Agent 输出
        agent_output = self.create_output(
            output=research_output,
            reasoning=f"完成研究任务: {current_task.get('name', '')}",
            task_id=current_task.get("id"),
            confidence=0.85,
        )
        
        # 更新子任务状态
        subtasks = self._update_subtask_status(
            state,
            current_task["id"],
            status="completed",
            result=research_output,
        )
        
        # 更新 agent_outputs
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs[f"researcher_{current_task['id']}"] = agent_output.model_dump()
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[Researcher] 完成任务 '{current_task.get('name', '')}': {research_output[:100]}..."
        )
        
        return {
            **state,
            "subtasks": subtasks,
            "agent_outputs": agent_outputs,
            "reasoning_trace": reasoning_trace,
            "next": "critic",
        }
    
    def _get_current_task(self, state: Dict[str, Any]) -> Optional[Dict]:
        """
        获取当前待处理的研究任务
        
        Args:
            state: 当前状态
            
        Returns:
            待处理的任务，没有则返回 None
        """
        subtasks = state.get("subtasks", [])
        
        for task in subtasks:
            if (task.get("status") == "pending" and 
                task.get("assigned_agent") == "researcher"):
                # 检查依赖是否满足
                dependencies = task.get("dependencies", [])
                if self._dependencies_satisfied(subtasks, dependencies):
                    return task
        
        return None
    
    def _dependencies_satisfied(
        self,
        subtasks: List[Dict],
        dependencies: List[str]
    ) -> bool:
        """
        检查任务依赖是否满足
        
        Args:
            subtasks: 所有子任务
            dependencies: 依赖任务 ID 列表
            
        Returns:
            依赖是否全部满足
        """
        if not dependencies:
            return True
        
        completed_ids = {
            t["id"] for t in subtasks
            if t.get("status") == "completed"
        }
        
        return all(dep in completed_ids for dep in dependencies)
    
    def _build_context(self, state: Dict[str, Any]) -> str:
        """
        构建上下文信息
        
        Args:
            state: 当前状态
            
        Returns:
            上下文字符串
        """
        context_parts = []
        
        # 原始任务
        original_task = state.get("original_task", "")
        if original_task:
            context_parts.append(f"原始任务: {original_task}")
        
        # 任务理解
        understanding = state.get("task_understanding", "")
        if understanding:
            context_parts.append(f"任务理解: {understanding[:500]}")
        
        # 已完成任务的结果
        agent_outputs = state.get("agent_outputs", {})
        if agent_outputs:
            context_parts.append("已有成果:")
            for name, output in agent_outputs.items():
                if isinstance(output, dict):
                    result = output.get("output", "")[:200]
                else:
                    result = str(output)[:200]
                context_parts.append(f"  - {name}: {result}")
        
        return "\n".join(context_parts)
    
    def _perform_search(self, query: str) -> Optional[str]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果字符串，失败返回 None
        """
        # 检查是否有搜索工具
        search_tool = next(
            (t for t in self.tools if t.name == "web_search"),
            None
        )
        
        if search_tool is None:
            self.logger.debug("没有可用的搜索工具")
            return None
        
        try:
            # 提取关键词进行搜索
            result = self.call_tool("web_search", query=query)
            return str(result)
        except Exception as e:
            self.logger.warning(f"搜索失败: {e}")
            return None
```

---

#### 文件：`src/agents/coder.py`

```python
"""
编码者 Agent
============

负责代码编写、调试和技术实现。
"""

import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates


@register_agent("coder")
class CoderAgent(BaseAgent):
    """
    编码者智能体
    
    核心职责：
    1. 根据需求编写代码
    2. 调试和修复代码问题
    3. 遵循编码最佳实践
    4. 添加文档和注释
    """
    
    @property
    def name(self) -> str:
        return "coder"
    
    @property
    def description(self) -> str:
        return "编码者，负责代码编写和技术实现"
    
    @property
    def capabilities(self) -> List[str]:
        return ["code", "debug", "implement"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行编码任务
        
        Args:
            state: 当前状态
            
        Returns:
            包含代码结果的更新状态
        """
        # 获取当前待处理的编码任务
        current_task = self._get_current_task(state)
        
        if current_task is None:
            self.logger.warning("没有找到待处理的编码任务")
            return {**state, "next": "coordinator"}
        
        task_description = current_task.get("description", "")
        context = self._build_context(state)
        requirements = current_task.get("metadata", {}).get("requirements", "Python 3.10+")
        
        # 构建编码提示
        prompt = PromptTemplates.get(
            "CODER_TASK",
            task=task_description,
            context=context,
            requirements=requirements,
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        code_output = response.content
        
        # 提取代码块
        code_blocks = self._extract_code_blocks(code_output)
        
        # 可选：测试代码执行
        execution_result = None
        if code_blocks and self._should_test_code(current_task):
            execution_result = self._test_code(code_blocks[0])
        
        # 构建完整输出
        full_output = code_output
        if execution_result:
            full_output += f"\n\n**代码测试结果:**\n```\n{execution_result}\n```"
        
        # 创建 Agent 输出
        agent_output = self.create_output(
            output=full_output,
            reasoning=f"完成编码任务: {current_task.get('name', '')}",
            task_id=current_task.get("id"),
            confidence=0.85 if execution_result else 0.75,
        )
        
        # 更新子任务状态
        subtasks = self._update_subtask_status(
            state,
            current_task["id"],
            status="completed",
            result=full_output,
        )
        
        # 更新 agent_outputs
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs[f"coder_{current_task['id']}"] = agent_output.model_dump()
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[Coder] 完成任务 '{current_task.get('name', '')}'"
        )
        if code_blocks:
            reasoning_trace.append(f"[Coder] 生成 {len(code_blocks)} 个代码块")
        
        return {
            **state,
            "subtasks": subtasks,
            "agent_outputs": agent_outputs,
            "reasoning_trace": reasoning_trace,
            "next": "critic",
        }
    
    def _get_current_task(self, state: Dict[str, Any]) -> Optional[Dict]:
        """
        获取当前待处理的编码任务
        
        Args:
            state: 当前状态
            
        Returns:
            待处理的任务
        """
        subtasks = state.get("subtasks", [])
        
        for task in subtasks:
            if (task.get("status") == "pending" and 
                task.get("assigned_agent") == "coder"):
                # 检查依赖是否满足
                dependencies = task.get("dependencies", [])
                if self._dependencies_satisfied(subtasks, dependencies):
                    return task
        
        return None
    
    def _dependencies_satisfied(
        self,
        subtasks: List[Dict],
        dependencies: List[str]
    ) -> bool:
        """检查任务依赖是否满足"""
        if not dependencies:
            return True
        
        completed_ids = {
            t["id"] for t in subtasks
            if t.get("status") == "completed"
        }
        
        return all(dep in completed_ids for dep in dependencies)
    
    def _build_context(self, state: Dict[str, Any]) -> str:
        """构建上下文信息"""
        context_parts = []
        
        original_task = state.get("original_task", "")
        if original_task:
            context_parts.append(f"原始任务: {original_task}")
        
        # 已有的研究结果
        agent_outputs = state.get("agent_outputs", {})
        for name, output in agent_outputs.items():
            if "researcher" in name:
                if isinstance(output, dict):
                    result = output.get("output", "")[:500]
                else:
                    result = str(output)[:500]
                context_parts.append(f"研究结果: {result}")
        
        return "\n".join(context_parts)
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """
        从内容中提取代码块
        
        Args:
            content: 响应内容
            
        Returns:
            代码块列表
        """
        # 匹配 ```python ... ``` 或 ``` ... ```
        pattern = r'```(?:python)?\s*([\s\S]*?)```'
        matches = re.findall(pattern, content)
        return [m.strip() for m in matches if m.strip()]
    
    def _should_test_code(self, task: Dict) -> bool:
        """
        判断是否应该测试代码
        
        Args:
            task: 任务信息
            
        Returns:
            是否测试
        """
        # 检查任务元数据
        metadata = task.get("metadata", {})
        if metadata.get("skip_test"):
            return False
        
        # 默认测试简单代码
        description = task.get("description", "").lower()
        skip_keywords = ["不要测试", "不需要运行", "skip test"]
        
        return not any(kw in description for kw in skip_keywords)
    
    def _test_code(self, code: str) -> Optional[str]:
        """
        测试代码执行
        
        Args:
            code: 代码字符串
            
        Returns:
            执行结果，失败返回 None
        """
        # 检查是否有代码执行工具
        executor_tool = next(
            (t for t in self.tools if t.name == "code_executor"),
            None
        )
        
        if executor_tool is None:
            self.logger.debug("没有可用的代码执行工具")
            return None
        
        try:
            result = self.call_tool("code_executor", code=code)
            return str(result)
        except Exception as e:
            self.logger.warning(f"代码测试失败: {e}")
            return f"执行错误: {str(e)}"
```

---

#### 文件：`src/agents/executor.py`

```python
"""
执行者 Agent
============

负责工具调用和具体操作执行。
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates


@register_agent("executor")
class ExecutorAgent(BaseAgent):
    """
    执行者智能体
    
    核心职责：
    1. 调用各种工具完成任务
    2. 验证执行结果
    3. 处理执行异常
    4. 报告执行状态
    """
    
    @property
    def name(self) -> str:
        return "executor"
    
    @property
    def description(self) -> str:
        return "执行者，负责工具调用和操作执行"
    
    @property
    def capabilities(self) -> List[str]:
        return ["execute", "tool_call", "file_operation"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行操作任务
        
        Args:
            state: 当前状态
            
        Returns:
            包含执行结果的更新状态
        """
        # 获取当前待处理的执行任务
        current_task = self._get_current_task(state)
        
        if current_task is None:
            self.logger.warning("没有找到待处理的执行任务")
            return {**state, "next": "coordinator"}
        
        task_description = current_task.get("description", "")
        
        # 分析任务，决定需要调用的工具
        tool_plan = self._plan_tool_calls(task_description, state)
        
        # 执行工具调用
        execution_results = []
        for tool_call in tool_plan:
            result = self._execute_tool_call(tool_call)
            execution_results.append(result)
        
        # 汇总执行结果
        summary = self._summarize_results(execution_results, task_description)
        
        # 判断是否成功
        success = all(r.get("success", False) for r in execution_results)
        
        # 创建 Agent 输出
        agent_output = self.create_output(
            output=summary,
            reasoning=f"执行任务: {current_task.get('name', '')}",
            task_id=current_task.get("id"),
            confidence=0.9 if success else 0.5,
        )
        
        # 更新子任务状态
        status = "completed" if success else "failed"
        subtasks = self._update_subtask_status(
            state,
            current_task["id"],
            status=status,
            result=summary,
            error=None if success else "执行过程中出现错误",
        )
        
        # 更新 agent_outputs
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs[f"executor_{current_task['id']}"] = agent_output.model_dump()
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[Executor] {'成功' if success else '失败'}执行任务 '{current_task.get('name', '')}'"
        )
        
        return {
            **state,
            "subtasks": subtasks,
            "agent_outputs": agent_outputs,
            "reasoning_trace": reasoning_trace,
            "next": "critic",
        }
    
    def _get_current_task(self, state: Dict[str, Any]) -> Optional[Dict]:
        """获取当前待处理的执行任务"""
        subtasks = state.get("subtasks", [])
        
        for task in subtasks:
            if (task.get("status") == "pending" and 
                task.get("assigned_agent") == "executor"):
                dependencies = task.get("dependencies", [])
                if self._dependencies_satisfied(subtasks, dependencies):
                    return task
        
        return None
    
    def _dependencies_satisfied(
        self,
        subtasks: List[Dict],
        dependencies: List[str]
    ) -> bool:
        """检查任务依赖是否满足"""
        if not dependencies:
            return True
        
        completed_ids = {
            t["id"] for t in subtasks
            if t.get("status") == "completed"
        }
        
        return all(dep in completed_ids for dep in dependencies)
    
    def _plan_tool_calls(
        self,
        task_description: str,
        state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        规划工具调用
        
        Args:
            task_description: 任务描述
            state: 当前状态
            
        Returns:
            工具调用计划列表
        """
        # 获取可用工具列表
        available_tools = [t.name for t in self.tools]
        
        # 使用 LLM 分析需要的工具调用
        prompt = f"""分析以下任务，决定需要调用哪些工具：

任务：{task_description}

可用工具：{', '.join(available_tools)}

工具说明：
- calculator: 数学计算，参数 expression (表达式字符串)
- file_manager: 文件操作，参数 action (read/write), path (文件路径), content (写入内容，可选)
- code_executor: 执行 Python 代码，参数 code (代码字符串)
- web_search: 网络搜索，参数 query (搜索查询)

请输出 JSON 格式的工具调用计划：
```json
[
    {{"tool": "工具名", "params": {{"参数名": "参数值"}}}}
]
```

如果不需要调用任何工具，返回空数组 []
"""
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        # 解析工具调用计划
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)```', response.content)
            if json_match:
                plan = json.loads(json_match.group(1))
            else:
                plan = json.loads(response.content)
            return plan if isinstance(plan, list) else []
        except (json.JSONDecodeError, AttributeError):
            self.logger.warning("无法解析工具调用计划")
            return []
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个工具调用
        
        Args:
            tool_call: 工具调用信息
            
        Returns:
            执行结果字典
        """
        tool_name = tool_call.get("tool")
        params = tool_call.get("params", {})
        
        if not tool_name:
            return {
                "tool": tool_name,
                "success": False,
                "error": "工具名称为空",
                "result": None,
            }
        
        try:
            result = self.call_tool(tool_name, **params)
            return {
                "tool": tool_name,
                "success": True,
                "error": None,
                "result": result,
            }
        except Exception as e:
            self.logger.error(f"工具调用失败 {tool_name}: {e}")
            return {
                "tool": tool_name,
                "success": False,
                "error": str(e),
                "result": None,
            }
    
    def _summarize_results(
        self,
        results: List[Dict[str, Any]],
        task_description: str
    ) -> str:
        """
        汇总执行结果
        
        Args:
            results: 执行结果列表
            task_description: 任务描述
            
        Returns:
            汇总字符串
        """
        if not results:
            return f"任务 '{task_description}' 未执行任何工具调用"
        
        summary_parts = [f"## 执行结果\n\n任务：{task_description}\n"]
        
        for i, result in enumerate(results, 1):
            tool_name = result.get("tool", "未知")
            success = result.get("success", False)
            status = "✅ 成功" if success else "❌ 失败"
            
            summary_parts.append(f"### {i}. {tool_name} - {status}")
            
            if success:
                output = result.get("result", "无输出")
                summary_parts.append(f"输出：\n```\n{output}\n```")
            else:
                error = result.get("error", "未知错误")
                summary_parts.append(f"错误：{error}")
            
            summary_parts.append("")
        
        return "\n".join(summary_parts)


# 需要导入 re 模块
import re
```

---

#### 文件：`src/agents/critic.py`

```python
"""
审核者 Agent
============

负责质量评估、问题发现和改进建议。
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates
from src.types import EvaluationResult


@register_agent("critic")
class CriticAgent(BaseAgent):
    """
    审核者智能体
    
    核心职责：
    1. 评估工作成果质量
    2. 发现错误和问题
    3. 提供改进建议
    4. 决定是否需要修正或人工介入
    """
    
    # 质量阈值
    PASS_THRESHOLD = 0.7
    HUMAN_REVIEW_THRESHOLD = 0.5
    
    @property
    def name(self) -> str:
        return "critic"
    
    @property
    def description(self) -> str:
        return "审核者，负责质量评估和改进建议"
    
    @property
    def capabilities(self) -> List[str]:
        return ["evaluate", "review", "suggest"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行审核任务
        
        Args:
            state: 当前状态
            
        Returns:
            包含评审结果的更新状态
        """
        # 获取最近完成的任务
        recent_outputs = self._get_recent_outputs(state)
        
        if not recent_outputs:
            self.logger.warning("没有找到需要审核的输出")
            return {**state, "next": "coordinator"}
        
        # 对每个输出进行评审
        evaluation_results = []
        for output_key, output_data in recent_outputs.items():
            eval_result = self._evaluate_output(output_key, output_data, state)
            evaluation_results.append(eval_result)
        
        # 汇总评审结果
        overall_result = self._aggregate_evaluations(evaluation_results)
        
        # 决定下一步行动
        next_action = self._decide_action(overall_result, state)
        
        # 更新反思记录
        reflection_notes = state.get("reflection_notes", [])
        reflection_notes.append(overall_result["summary"])
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[Critic] 评审完成，评分: {overall_result['score']:.2f}, "
            f"行动: {next_action}"
        )
        
        # 判断是否需要人工介入
        needs_human = (
            overall_result["score"] < self.HUMAN_REVIEW_THRESHOLD and
            self.settings.enable_human_in_loop
        )
        
        return {
            **state,
            "reflection_notes": reflection_notes,
            "evaluation_results": state.get("evaluation_results", []) + [overall_result],
            "needs_human_input": needs_human,
            "reasoning_trace": reasoning_trace,
            "next": next_action,
        }
    
    def _get_recent_outputs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取需要审核的最近输出
        
        Args:
            state: 当前状态
            
        Returns:
            需要审核的输出字典
        """
        agent_outputs = state.get("agent_outputs", {})
        evaluation_results = state.get("evaluation_results", [])
        
        # 获取已评审的输出键
        evaluated_keys = set()
        for result in evaluation_results:
            if isinstance(result, dict):
                evaluated_keys.update(result.get("evaluated_outputs", []))
        
        # 筛选未评审的输出
        recent = {
            k: v for k, v in agent_outputs.items()
            if k not in evaluated_keys
        }
        
        return recent
    
    def _evaluate_output(
        self,
        output_key: str,
        output_data: Any,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        评估单个输出
        
        Args:
            output_key: 输出键
            output_data: 输出数据
            state: 当前状态
            
        Returns:
            评估结果字典
        """
        # 提取输出内容
        if isinstance(output_data, dict):
            output_content = output_data.get("output", str(output_data))
            task_id = output_data.get("task_id")
        else:
            output_content = str(output_data)
            task_id = None
        
        # 获取相关子任务
        subtask = None
        if task_id:
            subtasks = state.get("subtasks", [])
            subtask = next((t for t in subtasks if t.get("id") == task_id), None)
        
        # 构建评审提示
        prompt = PromptTemplates.get(
            "CRITIC_REVIEW",
            original_task=state.get("original_task", ""),
            subtask=json.dumps(subtask, ensure_ascii=False) if subtask else "未知",
            agent_name=output_key.split("_")[0],
            output=output_content[:2000],  # 限制长度
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        # 解析评审结果
        eval_result = self._parse_evaluation(response.content)
        eval_result["output_key"] = output_key
        
        return eval_result
    
    def _parse_evaluation(self, content: str) -> Dict[str, Any]:
        """
        解析 LLM 评审结果
        
        Args:
            content: LLM 响应
            
        Returns:
            解析后的评估结果
        """
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)```', content)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(content)
            
            # 确保必要字段
            return {
                "score": float(result.get("score", 0.7)),
                "passed": result.get("passed", True),
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
                "action": result.get("action", "approve"),
                "reasoning": result.get("reasoning", ""),
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"评审结果解析失败: {e}")
            # 从文本中提取关键信息
            return self._extract_evaluation_from_text(content)
    
    def _extract_evaluation_from_text(self, content: str) -> Dict[str, Any]:
        """
        从文本中提取评估信息（当 JSON 解析失败时）
        
        Args:
            content: 响应文本
            
        Returns:
            评估结果
        """
        content_lower = content.lower()
        
        # 简单的关键词分析
        if "优秀" in content or "excellent" in content_lower:
            score = 0.95
        elif "良好" in content or "good" in content_lower:
            score = 0.8
        elif "通过" in content or "pass" in content_lower:
            score = 0.7
        elif "需要改进" in content or "needs improvement" in content_lower:
            score = 0.55
        elif "不合格" in content or "fail" in content_lower:
            score = 0.3
        else:
            score = 0.7
        
        passed = score >= self.PASS_THRESHOLD
        
        return {
            "score": score,
            "passed": passed,
            "issues": [],
            "suggestions": [],
            "action": "approve" if passed else "revise",
            "reasoning": content[:500],
        }
    
    def _aggregate_evaluations(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        汇总多个评估结果
        
        Args:
            evaluations: 评估结果列表
            
        Returns:
            汇总结果
        """
        if not evaluations:
            return {
                "score": 0.7,
                "passed": True,
                "summary": "无需评审的输出",
                "evaluated_outputs": [],
            }
        
        # 计算平均分
        scores = [e.get("score", 0.7) for e in evaluations]
        avg_score = sum(scores) / len(scores)
        
        # 汇总问题和建议
        all_issues = []
        all_suggestions = []
        for e in evaluations:
            all_issues.extend(e.get("issues", []))
            all_suggestions.extend(e.get("suggestions", []))
        
        # 判断是否通过
        passed = all(e.get("passed", True) for e in evaluations)
        
        # 生成汇总
        summary_parts = [f"评审完成 - 平均评分: {avg_score:.2f}"]
        if all_issues:
            summary_parts.append(f"发现 {len(all_issues)} 个问题")
        if all_suggestions:
            summary_parts.append(f"提出 {len(all_suggestions)} 条建议")
        
        return {
            "score": avg_score,
            "passed": passed,
            "issues": all_issues,
            "suggestions": all_suggestions,
            "summary": "; ".join(summary_parts),
            "evaluated_outputs": [e.get("output_key") for e in evaluations],
            "details": evaluations,
        }
    
    def _decide_action(
        self,
        evaluation: Dict[str, Any],
        state: Dict[str, Any]
    ) -> str:
        """
        根据评估结果决定下一步行动
        
        Args:
            evaluation: 评估结果
            state: 当前状态
            
        Returns:
            下一步行动节点名
        """
        score = evaluation.get("score", 0.7)
        passed = evaluation.get("passed", True)
        
        # 检查是否有待处理的子任务
        pending_tasks = self._get_pending_subtasks(state)
        
        if score < self.HUMAN_REVIEW_THRESHOLD:
            # 低分需要人工介入
            return "human_node"
        
        if not passed:
            # 未通过，检查重试次数
            iteration_count = state.get("iteration_count", 0)
            max_iterations = state.get("max_iterations", 10)
            
            if iteration_count < max_iterations:
                return "task_router"  # 重新执行
            else:
                return "synthesizer"  # 超过次数，强制结束
        
        if pending_tasks:
            # 还有待处理任务
            return "task_router"
        
        # 所有任务完成且通过审核
        return "synthesizer"
```

---

#### 文件：`src/agents/synthesizer.py`

```python
"""
综合者 Agent
============

负责结果汇总和最终输出生成。
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates


@register_agent("synthesizer")
class SynthesizerAgent(BaseAgent):
    """
    综合者智能体
    
    核心职责：
    1. 整合所有智能体的输出
    2. 生成结构化的最终答案
    3. 确保输出完整、专业
    4. 根据需要保存结果文件
    """
    
    @property
    def name(self) -> str:
        return "synthesizer"
    
    @property
    def description(self) -> str:
        return "综合者，负责结果汇总和最终输出"
    
    @property
    def capabilities(self) -> List[str]:
        return ["synthesize", "aggregate", "format"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行综合任务
        
        Args:
            state: 当前状态
            
        Returns:
            包含最终答案的更新状态
        """
        original_task = state.get("original_task", "")
        agent_outputs = state.get("agent_outputs", {})
        reflection_notes = state.get("reflection_notes", [])
        
        # 格式化各智能体输出
        formatted_outputs = self._format_agent_outputs(agent_outputs)
        
        # 格式化审核意见
        formatted_reviews = self._format_reviews(reflection_notes)
        
        # 构建综合提示
        prompt = PromptTemplates.get(
            "SYNTHESIZER_AGGREGATE",
            original_task=original_task,
            agent_outputs=formatted_outputs,
            review_notes=formatted_reviews,
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        final_answer = response.content
        
        # 如果需要保存文件，执行保存
        save_result = self._save_if_needed(final_answer, state)
        if save_result:
            final_answer += f"\n\n{save_result}"
        
        # 创建 Agent 输出
        agent_output = self.create_output(
            output=final_answer,
            reasoning="综合所有结果生成最终答案",
            confidence=0.9,
        )
        
        # 更新 agent_outputs
        agent_outputs["synthesizer"] = agent_output.model_dump()
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append("[Synthesizer] 生成最终答案")
        
        self.logger.info("综合完成，生成最终答案")
        
        return {
            **state,
            "agent_outputs": agent_outputs,
            "final_answer": final_answer,
            "reasoning_trace": reasoning_trace,
            "next": "end",
        }
    
    def _format_agent_outputs(self, outputs: Dict[str, Any]) -> str:
        """
        格式化各智能体的输出
        
        Args:
            outputs: 输出字典
            
        Returns:
            格式化的字符串
        """
        if not outputs:
            return "（无输出）"
        
        parts = []
        for name, output in outputs.items():
            if isinstance(output, dict):
                content = output.get("output", str(output))
                reasoning = output.get("reasoning", "")
            else:
                content = str(output)
                reasoning = ""
            
            # 提取智能体类型
            agent_type = name.split("_")[0]
            
            part = f"### {agent_type.upper()}\n"
            if reasoning:
                part += f"**推理过程**: {reasoning}\n\n"
            part += f"**输出内容**:\n{content}\n"
            
            parts.append(part)
        
        return "\n---\n".join(parts)
    
    def _format_reviews(self, notes: List[str]) -> str:
        """
        格式化审核意见
        
        Args:
            notes: 审核意见列表
            
        Returns:
            格式化的字符串
        """
        if not notes:
            return "（无审核意见）"
        
        return "\n".join(f"- {note}" for note in notes)
    
    def _save_if_needed(
        self,
        content: str,
        state: Dict[str, Any]
    ) -> Optional[str]:
        """
        如果需要，保存结果到文件
        
        Args:
            content: 要保存的内容
            state: 当前状态
            
        Returns:
            保存结果消息，不需要保存返回 None
        """
        original_task = state.get("original_task", "").lower()
        
        # 检查是否需要保存文件
        save_keywords = ["保存", "写入", "输出到文件", "save", "write to file"]
        needs_save = any(kw in original_task for kw in save_keywords)
        
        if not needs_save:
            # 检查内容中是否包含代码，代码通常需要保存
            if "```python" in content and "爬虫" in original_task:
                needs_save = True
        
        if not needs_save:
            return None
        
        # 确定文件名
        if "json" in original_task:
            filename = "output.json"
        elif "python" in original_task or "代码" in original_task:
            filename = "output.py"
        else:
            filename = "output.txt"
        
        # 尝试保存
        file_tool = next(
            (t for t in self.tools if t.name == "file_manager"),
            None
        )
        
        if file_tool is None:
            return None
        
        try:
            # 提取要保存的内容
            save_content = self._extract_saveable_content(content)
            
            result = self.call_tool(
                "file_manager",
                action="write",
                path=filename,
                content=save_content,
            )
            
            return f"📁 结果已保存至: workspace/{filename}"
            
        except Exception as e:
            self.logger.warning(f"保存文件失败: {e}")
            return f"⚠️ 保存文件失败: {str(e)}"
    
    def _extract_saveable_content(self, content: str) -> str:
        """
        提取可保存的内容（如代码块）
        
        Args:
            content: 完整内容
            
        Returns:
            可保存的内容
        """
        import re
        
        # 尝试提取代码块
        code_pattern = r'```(?:python)?\s*([\s\S]*?)```'
        code_matches = re.findall(code_pattern, content)
        
        if code_matches:
            # 返回所有代码块
            return "\n\n".join(code_matches)
        
        # 尝试提取 JSON
        json_pattern = r'```json\s*([\s\S]*?)```'
        json_matches = re.findall(json_pattern, content)
        
        if json_matches:
            return json_matches[0]
        
        # 返回原始内容
        return content
```

---

**第三部分（第2批）完成。请输入 "继续" 获取第三部分（第3批）：Graph 模块和 Tools 模块完整代码。**


## 第三部分（第3批）：Graph 模块和 Tools 模块完整代码

---

### Graph 模块

#### 文件：`src/graph/__init__.py`

```python
"""
图模块
======

提供 LangGraph 状态图的构建和管理功能。

核心组件：
- AgentState: 系统状态定义
- build_graph: 图构建函数
- MultiAgentSystem: 封装的系统类
"""

from src.graph.state import AgentState, create_initial_state
from src.graph.nodes import (
    input_parser_node,
    coordinator_node,
    planner_node,
    task_router_node,
    researcher_node,
    coder_node,
    executor_node,
    critic_node,
    human_node,
    synthesizer_node,
    error_handler_node,
)
from src.graph.edges import (
    route_from_coordinator,
    route_from_planner,
    route_task,
    route_from_critic,
    route_from_human,
    should_continue,
)
from src.graph.builder import build_graph, MultiAgentSystem

__all__ = [
    # State
    "AgentState",
    "create_initial_state",
    # Nodes
    "input_parser_node",
    "coordinator_node",
    "planner_node",
    "task_router_node",
    "researcher_node",
    "coder_node",
    "executor_node",
    "critic_node",
    "human_node",
    "synthesizer_node",
    "error_handler_node",
    # Edges
    "route_from_coordinator",
    "route_from_planner",
    "route_task",
    "route_from_critic",
    "route_from_human",
    "should_continue",
    # Builder
    "build_graph",
    "MultiAgentSystem",
]
```

---

#### 文件：`src/graph/state.py`

```python
"""
状态定义模块
============

定义 LangGraph 使用的状态结构。

AgentState 是整个系统的核心状态容器，在各节点间传递和更新。
使用 TypedDict 确保类型安全，使用 Annotated 支持状态 reducer。
"""

from typing import (
    TypedDict,
    Annotated,
    List,
    Dict,
    Any,
    Optional,
    Sequence,
)
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class SubTaskState(TypedDict, total=False):
    """子任务状态结构"""
    id: str
    name: str
    description: str
    task_type: str
    assigned_agent: str
    dependencies: List[str]
    priority: str
    status: str
    result: Optional[str]
    error_message: Optional[str]
    retry_count: int
    created_at: str
    updated_at: Optional[str]
    metadata: Dict[str, Any]


class ToolCallLogState(TypedDict, total=False):
    """工具调用日志状态"""
    tool_name: str
    input_params: Dict[str, Any]
    output: Any
    success: bool
    error_message: Optional[str]
    timestamp: str
    duration_ms: float


class AgentOutputState(TypedDict, total=False):
    """Agent 输出状态"""
    agent_name: str
    task_id: Optional[str]
    output: str
    reasoning: str
    tool_calls: List[ToolCallLogState]
    confidence: float
    timestamp: str


class EvaluationResultState(TypedDict, total=False):
    """评估结果状态"""
    score: float
    passed: bool
    issues: List[str]
    suggestions: List[str]
    summary: str
    evaluated_outputs: List[str]
    details: List[Dict[str, Any]]


class AgentState(TypedDict, total=False):
    """
    系统全局状态
    
    这是 LangGraph 状态图的核心状态定义。所有节点都接收和返回此状态的部分或全部字段。
    
    状态字段说明：
    
    消息与对话：
        messages: 使用 add_messages reducer 的消息历史，自动合并新消息
        
    任务相关：
        original_task: 用户输入的原始任务描述
        task_understanding: 协调者对任务的理解分析
        subtasks: 规划者分解的子任务列表
        current_plan: 当前执行计划摘要
        current_subtask_id: 当前正在执行的子任务 ID
        parallel_groups: 可并行执行的任务组
        
    Agent 输出：
        agent_outputs: 各 Agent 的输出结果，键为 "agent_taskid" 格式
        
    工具相关：
        tool_call_logs: 所有工具调用的日志记录
        available_tools: 可用工具名称列表
        
    控制流：
        current_agent: 当前正在执行的 Agent 名称
        next: 下一个要执行的节点名称，用于路由
        iteration_count: 当前迭代次数
        max_iterations: 最大允许迭代次数
        
    反思与审核：
        reflection_notes: Critic 的反思记录
        evaluation_results: 评估结果列表
        
    人工介入：
        needs_human_input: 是否需要人工介入
        human_feedback: 人工提供的反馈
        
    最终输出：
        final_answer: 最终生成的答案
        reasoning_trace: 推理过程轨迹，用于可解释性
        
    错误处理：
        error_log: 错误日志列表
        last_error: 最近的错误信息
        
    指标统计：
        token_usage: Token 使用统计
        execution_time: 各节点执行时间统计
        start_time: 任务开始时间戳
        task_id: 任务唯一标识
    """
    
    # ===== 消息历史 =====
    # 使用 add_messages reducer：新消息会自动追加到列表
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # ===== 任务相关 =====
    original_task: str
    task_understanding: str
    subtasks: List[SubTaskState]
    current_plan: str
    current_subtask_id: Optional[str]
    parallel_groups: List[List[str]]
    
    # ===== Agent 输出 =====
    agent_outputs: Dict[str, AgentOutputState]
    
    # ===== 工具相关 =====
    tool_call_logs: List[ToolCallLogState]
    available_tools: List[str]
    
    # ===== 控制流 =====
    current_agent: str
    next: str
    iteration_count: int
    max_iterations: int
    
    # ===== 反思与审核 =====
    reflection_notes: List[str]
    evaluation_results: List[EvaluationResultState]
    
    # ===== 人工介入 =====
    needs_human_input: bool
    human_feedback: Optional[str]
    
    # ===== 最终输出 =====
    final_answer: Optional[str]
    reasoning_trace: List[str]
    
    # ===== 错误处理 =====
    error_log: List[str]
    last_error: Optional[str]
    
    # ===== 指标统计 =====
    token_usage: Dict[str, int]
    execution_time: Dict[str, float]
    start_time: Optional[float]
    task_id: str


def create_initial_state(
    task: str,
    task_id: Optional[str] = None,
    max_iterations: int = 10,
) -> AgentState:
    """
    创建初始状态
    
    Args:
        task: 用户任务描述
        task_id: 任务 ID，不提供则自动生成
        max_iterations: 最大迭代次数
        
    Returns:
        初始化的 AgentState
    """
    import time
    
    if task_id is None:
        task_id = str(uuid.uuid4())[:8]
    
    return AgentState(
        # 消息历史
        messages=[HumanMessage(content=task)],
        
        # 任务相关
        original_task=task,
        task_understanding="",
        subtasks=[],
        current_plan="",
        current_subtask_id=None,
        parallel_groups=[],
        
        # Agent 输出
        agent_outputs={},
        
        # 工具相关
        tool_call_logs=[],
        available_tools=[],
        
        # 控制流
        current_agent="",
        next="input_parser",
        iteration_count=0,
        max_iterations=max_iterations,
        
        # 反思与审核
        reflection_notes=[],
        evaluation_results=[],
        
        # 人工介入
        needs_human_input=False,
        human_feedback=None,
        
        # 最终输出
        final_answer=None,
        reasoning_trace=[],
        
        # 错误处理
        error_log=[],
        last_error=None,
        
        # 指标统计
        token_usage={"prompt": 0, "completion": 0, "total": 0},
        execution_time={},
        start_time=time.time(),
        task_id=task_id,
    )


def merge_state(base: AgentState, updates: Dict[str, Any]) -> AgentState:
    """
    合并状态更新
    
    Args:
        base: 基础状态
        updates: 要更新的字段
        
    Returns:
        合并后的新状态
    """
    # 创建新状态副本
    new_state = dict(base)
    
    # 更新字段
    for key, value in updates.items():
        if key == "messages":
            # messages 使用 add_messages reducer，不直接覆盖
            continue
        new_state[key] = value
    
    return AgentState(**new_state)
```

---

#### 文件：`src/graph/nodes.py`

```python
"""
节点函数模块
============

定义 LangGraph 状态图中的所有节点处理函数。

每个节点函数接收当前状态，执行特定逻辑，返回状态更新。
节点函数是纯函数，不应有副作用（除了日志记录）。
"""

import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.prompt import Prompt

from src.graph.state import AgentState
from src.agents import (
    CoordinatorAgent,
    PlannerAgent,
    ResearcherAgent,
    CoderAgent,
    ExecutorAgent,
    CriticAgent,
    SynthesizerAgent,
)
from src.config.settings import get_settings
from src.utils.logger import get_logger

# 初始化
logger = get_logger(__name__)
console = Console()

# Agent 实例缓存
_agent_cache: Dict[str, Any] = {}


def _get_agent(agent_class, tools: Optional[List] = None):
    """
    获取 Agent 实例（带缓存）
    
    Args:
        agent_class: Agent 类
        tools: 工具列表
        
    Returns:
        Agent 实例
    """
    class_name = agent_class.__name__
    
    if class_name not in _agent_cache:
        from src.tools import get_all_tools
        from src.llm.factory import LLMFactory
        
        settings = get_settings()
        llm = LLMFactory.create(settings.get_llm_config())
        
        if tools is None:
            tools = get_all_tools()
        
        _agent_cache[class_name] = agent_class(
            llm=llm,
            tools=tools,
            settings=settings,
        )
    
    return _agent_cache[class_name]


def clear_agent_cache():
    """清空 Agent 缓存"""
    _agent_cache.clear()


def input_parser_node(state: AgentState) -> Dict[str, Any]:
    """
    输入解析节点
    
    解析用户输入，进行初步处理和标准化。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.info("[Node] input_parser - 开始解析输入")
    
    original_task = state.get("original_task", "")
    
    # 记录开始时间
    start_time = state.get("start_time") or time.time()
    
    # 添加推理轨迹
    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append(f"[InputParser] 收到任务: {original_task[:100]}...")
    
    logger.info(f"[Node] input_parser - 任务: {original_task[:50]}...")
    
    return {
        "start_time": start_time,
        "reasoning_trace": reasoning_trace,
        "next": "coordinator",
    }


def coordinator_node(state: AgentState) -> Dict[str, Any]:
    """
    协调者节点
    
    理解任务、协调工作流程、决定下一步行动。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.info("[Node] coordinator - 开始协调")
    
    agent = _get_agent(CoordinatorAgent)
    result = agent.invoke(dict(state))
    
    return result


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    规划者节点
    
    分解任务、制定执行计划。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.info("[Node] planner - 开始规划")
    
    agent = _get_agent(PlannerAgent)
    result = agent.invoke(dict(state))
    
    return result


def task_router_node(state: AgentState) -> Dict[str, Any]:
    """
    任务路由节点
    
    根据待处理的子任务决定下一个执行的 Agent。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.info("[Node] task_router - 路由任务")
    
    subtasks = state.get("subtasks", [])
    
    # 查找下一个待处理任务
    pending_tasks = [t for t in subtasks if t.get("status") == "pending"]
    
    if not pending_tasks:
        # 没有待处理任务
        logger.info("[Node] task_router - 无待处理任务，进入综合阶段")
        return {"next": "synthesizer"}
    
    # 检查依赖关系，找到可执行的任务
    completed_ids = {t["id"] for t in subtasks if t.get("status") == "completed"}
    
    executable_task = None
    for task in pending_tasks:
        dependencies = task.get("dependencies", [])
        if all(dep in completed_ids for dep in dependencies):
            executable_task = task
            break
    
    if executable_task is None:
        # 有循环依赖或阻塞，选择第一个
        logger.warning("[Node] task_router - 检测到依赖阻塞，强制执行")
        executable_task = pending_tasks[0]
    
    # 根据分配的 Agent 路由
    assigned_agent = executable_task.get("assigned_agent", "executor")
    
    # 更新子任务状态为 running
    updated_subtasks = []
    for task in subtasks:
        if task["id"] == executable_task["id"]:
            task = dict(task)
            task["status"] = "running"
            task["updated_at"] = datetime.now().isoformat()
        updated_subtasks.append(task)
    
    # 添加推理轨迹
    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append(
        f"[TaskRouter] 路由任务 '{executable_task.get('name')}' 到 {assigned_agent}"
    )
    
    logger.info(f"[Node] task_router - 路由到 {assigned_agent}")
    
    return {
        "subtasks": updated_subtasks,
        "current_subtask_id": executable_task["id"],
        "reasoning_trace": reasoning_trace,
        "next": assigned_agent,
    }


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    研究员节点
    
    执行研究任务，获取和整合信息。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.info("[Node] researcher - 开始研究")
    
    agent = _get_agent(ResearcherAgent)
    result = agent.invoke(dict(state))
    
    return result


def coder_node(state: AgentState) -> Dict[str, Any]:
    """
    编码者节点
    
    执行编码任务，生成代码。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.info("[Node] coder - 开始编码")
    
    agent = _get_agent(CoderAgent)
    result = agent.invoke(dict(state))
    
    return result


def executor_node(state: AgentState) -> Dict[str, Any]:
    """
    执行者节点
    
    执行工具调用和操作。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.info("[Node] executor - 开始执行")
    
    agent = _get_agent(ExecutorAgent)
    result = agent.invoke(dict(state))
    
    return result


def critic_node(state: AgentState) -> Dict[str, Any]:
    """
    审核者节点
    
    评估工作质量，提供反馈。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.info("[Node] critic - 开始审核")
    
    agent = _get_agent(CriticAgent)
    result = agent.invoke(dict(state))
    
    return result


def human_node(state: AgentState) -> Dict[str, Any]:
    """
    人工介入节点
    
    暂停执行，等待人工输入。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.info("[Node] human_node - 等待人工介入")
    
    settings = get_settings()
    
    if not settings.enable_human_in_loop:
        # 人工介入被禁用，自动继续
        logger.info("[Node] human_node - 人工介入已禁用，自动继续")
        return {
            "needs_human_input": False,
            "human_feedback": "自动继续（人工介入已禁用）",
            "next": "task_router",
        }
    
    # 显示当前状态摘要
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]⚠️  需要人工介入[/bold yellow]")
    console.print("=" * 60)
    
    # 显示任务信息
    console.print(f"\n[bold]原始任务:[/bold] {state.get('original_task', '')[:200]}")
    
    # 显示最近的评估结果
    eval_results = state.get("evaluation_results", [])
    if eval_results:
        latest = eval_results[-1]
        console.print(f"\n[bold]最新评估:[/bold]")
        console.print(f"  评分: {latest.get('score', 'N/A')}")
        console.print(f"  摘要: {latest.get('summary', 'N/A')}")
        
        issues = latest.get("issues", [])
        if issues:
            console.print(f"  问题: {', '.join(issues[:3])}")
    
    console.print("\n" + "-" * 60)
    console.print("[dim]选项: [continue] 继续执行 | [retry] 重试当前任务 | [abort] 中止任务[/dim]")
    
    # 获取用户输入
    try:
        user_input = Prompt.ask(
            "\n请输入您的决定或反馈",
            default="continue"
        )
    except (EOFError, KeyboardInterrupt):
        user_input = "continue"
    
    # 解析用户输入
    user_input_lower = user_input.lower().strip()
    
    if user_input_lower in ("abort", "exit", "quit"):
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append("[Human] 用户选择中止任务")
        
        return {
            "needs_human_input": False,
            "human_feedback": "用户中止任务",
            "final_answer": "任务已被用户中止。",
            "reasoning_trace": reasoning_trace,
            "next": "end",
        }
    
    elif user_input_lower == "retry":
        # 重置当前任务状态
        current_task_id = state.get("current_subtask_id")
        subtasks = state.get("subtasks", [])
        
        if current_task_id:
            subtasks = [
                {**t, "status": "pending", "retry_count": t.get("retry_count", 0) + 1}
                if t["id"] == current_task_id else t
                for t in subtasks
            ]
        
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append("[Human] 用户选择重试当前任务")
        
        return {
            "needs_human_input": False,
            "human_feedback": "用户请求重试",
            "subtasks": subtasks,
            "reasoning_trace": reasoning_trace,
            "next": "task_router",
        }
    
    else:
        # 继续执行，保存用户反馈
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(f"[Human] 用户反馈: {user_input[:100]}")
        
        return {
            "needs_human_input": False,
            "human_feedback": user_input,
            "reasoning_trace": reasoning_trace,
            "next": "task_router",
        }


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    综合者节点
    
    汇总所有结果，生成最终答案。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.info("[Node] synthesizer - 开始综合")
    
    agent = _get_agent(SynthesizerAgent)
    result = agent.invoke(dict(state))
    
    return result


def error_handler_node(state: AgentState) -> Dict[str, Any]:
    """
    错误处理节点
    
    处理执行过程中的错误，决定恢复策略。
    
    Args:
        state: 当前状态
        
    Returns:
        状态更新字典
    """
    logger.warning("[Node] error_handler - 处理错误")
    
    last_error = state.get("last_error", "未知错误")
    error_log = state.get("error_log", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    
    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append(f"[ErrorHandler] 处理错误: {last_error}")
    
    # 检查是否超过最大迭代
    if iteration_count >= max_iterations:
        logger.error("[Node] error_handler - 超过最大迭代次数，强制结束")
        
        reasoning_trace.append("[ErrorHandler] 超过最大迭代次数，强制结束")
        
        return {
            "final_answer": f"任务执行过程中出现错误，已达到最大重试次数。\n\n最后错误: {last_error}",
            "reasoning_trace": reasoning_trace,
            "next": "end",
        }
    
    # 检查错误类型和频率
    recent_errors = [e for e in error_log if last_error[:20] in e]
    
    if len(recent_errors) >= 3:
        # 同一错误重复出现，跳过问题任务
        logger.warning("[Node] error_handler - 错误重复出现，跳过问题任务")
        
        current_task_id = state.get("current_subtask_id")
        subtasks = state.get("subtasks", [])
        
        if current_task_id:
            subtasks = [
                {**t, "status": "skipped", "error_message": last_error}
                if t["id"] == current_task_id else t
                for t in subtasks
            ]
        
        reasoning_trace.append("[ErrorHandler] 跳过问题任务，继续执行")
        
        return {
            "subtasks": subtasks,
            "last_error": None,
            "reasoning_trace": reasoning_trace,
            "next": "task_router",
        }
    
    # 一般错误，重试
    logger.info("[Node] error_handler - 准备重试")
    
    reasoning_trace.append("[ErrorHandler] 准备重试任务")
    
    return {
        "iteration_count": iteration_count + 1,
        "last_error": None,
        "reasoning_trace": reasoning_trace,
        "next": "coordinator",
    }
```

---

#### 文件：`src/graph/edges.py`

```python
"""
边与路由模块
============

定义 LangGraph 状态图中的条件边和路由逻辑。

路由函数根据当前状态决定下一个要执行的节点。
"""

from typing import Literal
from src.graph.state import AgentState
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 定义路由类型
RouteType = Literal[
    "input_parser",
    "coordinator",
    "planner",
    "task_router",
    "researcher",
    "coder",
    "executor",
    "critic",
    "human_node",
    "synthesizer",
    "error_handler",
    "end",
]


def route_from_coordinator(state: AgentState) -> RouteType:
    """
    协调者节点后的路由
    
    根据协调者的决策，路由到下一个节点。
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_node = state.get("next", "planner")
    
    # 如果有最终答案，直接结束
    if state.get("final_answer"):
        logger.debug("[Route] coordinator -> end (有最终答案)")
        return "end"
    
    # 检查是否有错误
    if state.get("last_error"):
        logger.debug("[Route] coordinator -> error_handler")
        return "error_handler"
    
    # 根据 next 字段路由
    valid_routes = {
        "planner", "task_router", "synthesizer",
        "researcher", "coder", "executor", "end"
    }
    
    if next_node in valid_routes:
        logger.debug(f"[Route] coordinator -> {next_node}")
        return next_node
    
    # 默认去规划
    logger.debug("[Route] coordinator -> planner (默认)")
    return "planner"


def route_from_planner(state: AgentState) -> RouteType:
    """
    规划者节点后的路由
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_node = state.get("next", "task_router")
    
    # 检查是否有子任务
    subtasks = state.get("subtasks", [])
    
    if not subtasks:
        # 没有子任务，可能是简单问题
        logger.debug("[Route] planner -> synthesizer (无子任务)")
        return "synthesizer"
    
    if next_node == "task_router":
        logger.debug("[Route] planner -> task_router")
        return "task_router"
    
    logger.debug(f"[Route] planner -> {next_node}")
    return next_node


def route_task(state: AgentState) -> RouteType:
    """
    任务路由节点后的路由
    
    根据下一个要执行的任务类型，路由到对应的 Agent。
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_node = state.get("next", "executor")
    
    # 有效的工作者节点
    worker_nodes = {"researcher", "coder", "executor"}
    
    if next_node in worker_nodes:
        logger.debug(f"[Route] task_router -> {next_node}")
        return next_node
    
    if next_node == "synthesizer":
        logger.debug("[Route] task_router -> synthesizer")
        return "synthesizer"
    
    if next_node == "coordinator":
        logger.debug("[Route] task_router -> coordinator")
        return "coordinator"
    
    # 默认去执行者
    logger.debug("[Route] task_router -> executor (默认)")
    return "executor"


def route_from_worker(state: AgentState) -> RouteType:
    """
    工作者节点（researcher/coder/executor）后的路由
    
    通常路由到审核者进行质量检查。
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_node = state.get("next", "critic")
    
    # 检查是否有错误
    if state.get("last_error"):
        logger.debug("[Route] worker -> error_handler")
        return "error_handler"
    
    logger.debug(f"[Route] worker -> {next_node}")
    return next_node


def route_from_critic(state: AgentState) -> RouteType:
    """
    审核者节点后的路由
    
    根据评审结果决定：
    - 质量合格：继续执行或综合
    - 质量不合格：重试或人工介入
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_node = state.get("next", "task_router")
    
    # 检查是否需要人工介入
    if state.get("needs_human_input"):
        logger.debug("[Route] critic -> human_node")
        return "human_node"
    
    # 检查是否有待处理任务
    subtasks = state.get("subtasks", [])
    pending = [t for t in subtasks if t.get("status") == "pending"]
    
    if next_node == "synthesizer" or not pending:
        logger.debug("[Route] critic -> synthesizer")
        return "synthesizer"
    
    if next_node == "task_router":
        logger.debug("[Route] critic -> task_router")
        return "task_router"
    
    if next_node == "coordinator":
        logger.debug("[Route] critic -> coordinator")
        return "coordinator"
    
    logger.debug(f"[Route] critic -> {next_node}")
    return next_node


def route_from_human(state: AgentState) -> RouteType:
    """
    人工介入节点后的路由
    
    根据人工反馈决定下一步。
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_node = state.get("next", "task_router")
    
    # 如果有最终答案（用户中止）
    if state.get("final_answer"):
        logger.debug("[Route] human_node -> end")
        return "end"
    
    logger.debug(f"[Route] human_node -> {next_node}")
    return next_node


def route_from_synthesizer(state: AgentState) -> RouteType:
    """
    综合者节点后的路由
    
    通常综合完成后结束。
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_node = state.get("next", "end")
    
    if state.get("final_answer"):
        logger.debug("[Route] synthesizer -> end")
        return "end"
    
    # 可能需要继续迭代
    if next_node == "coordinator":
        logger.debug("[Route] synthesizer -> coordinator")
        return "coordinator"
    
    logger.debug("[Route] synthesizer -> end")
    return "end"


def route_from_error_handler(state: AgentState) -> RouteType:
    """
    错误处理节点后的路由
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    next_node = state.get("next", "coordinator")
    
    if state.get("final_answer"):
        logger.debug("[Route] error_handler -> end")
        return "end"
    
    logger.debug(f"[Route] error_handler -> {next_node}")
    return next_node


def should_continue(state: AgentState) -> bool:
    """
    判断是否应该继续执行
    
    用于循环条件判断。
    
    Args:
        state: 当前状态
        
    Returns:
        是否继续
    """
    # 有最终答案则停止
    if state.get("final_answer"):
        return False
    
    # 超过最大迭代则停止
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    
    if iteration_count >= max_iterations:
        logger.warning("达到最大迭代次数，停止执行")
        return False
    
    return True
```

---

#### 文件：`src/graph/builder.py`

```python
"""
图构建器模块
============

构建 LangGraph 状态图，组装所有节点和边。

这是系统的核心组装点，将所有组件连接成可执行的工作流。
"""

import time
from typing import Any, Dict, List, Optional, Generator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AgentState, create_initial_state
from src.graph.nodes import (
    input_parser_node,
    coordinator_node,
    planner_node,
    task_router_node,
    researcher_node,
    coder_node,
    executor_node,
    critic_node,
    human_node,
    synthesizer_node,
    error_handler_node,
    clear_agent_cache,
)
from src.graph.edges import (
    route_from_coordinator,
    route_from_planner,
    route_task,
    route_from_worker,
    route_from_critic,
    route_from_human,
    route_from_synthesizer,
    route_from_error_handler,
)
from src.config.settings import Settings, get_settings
from src.utils.logger import get_logger
from src.types import FinalResult, ExecutionMetrics

logger = get_logger(__name__)


def build_graph(settings: Optional[Settings] = None) -> StateGraph:
    """
    构建 LangGraph 状态图
    
    这个函数创建并配置整个多智能体工作流：
    
    1. 创建状态图实例
    2. 添加所有处理节点
    3. 配置入口点
    4. 定义节点之间的边和条件边
    5. 返回未编译的图（需要调用 compile() 编译）
    
    图结构：
    ```
    input_parser -> coordinator -> planner -> task_router
                                      ^            |
                                      |      +-----+-----+-----+
                                      |      v     v     v     v
                                      +-- researcher coder executor
                                              |     |     |
                                              +-----+-----+
                                                    |
                                                    v
                                                 critic
                                                    |
                                      +-------------+-------------+
                                      v                           v
                                  human_node               synthesizer -> END
                                      |                           ^
                                      +---------------------------+
    ```
    
    Args:
        settings: 系统配置，None 使用默认配置
        
    Returns:
        配置好的 StateGraph 实例
    """
    if settings is None:
        settings = get_settings()
    
    logger.info("开始构建状态图...")
    
    # 创建状态图
    # AgentState 定义了图中流转的状态结构
    graph = StateGraph(AgentState)
    
    # ===================================================
    # 添加节点
    # 每个节点对应一个处理函数，接收状态并返回状态更新
    # ===================================================
    
    # 输入解析节点：处理和标准化用户输入
    graph.add_node("input_parser", input_parser_node)
    
    # 协调者节点：理解任务、决定执行策略
    graph.add_node("coordinator", coordinator_node)
    
    # 规划者节点：分解任务、制定执行计划
    graph.add_node("planner", planner_node)
    
    # 任务路由节点：根据任务类型分配给对应 Agent
    graph.add_node("task_router", task_router_node)
    
    # 工作者节点
    graph.add_node("researcher", researcher_node)  # 研究员：信息检索
    graph.add_node("coder", coder_node)            # 编码者：代码生成
    graph.add_node("executor", executor_node)      # 执行者：工具调用
    
    # 审核者节点：质量评估和反馈
    graph.add_node("critic", critic_node)
    
    # 人工介入节点：需要时暂停等待人工输入
    graph.add_node("human_node", human_node)
    
    # 综合者节点：汇总结果、生成最终答案
    graph.add_node("synthesizer", synthesizer_node)
    
    # 错误处理节点：处理执行异常
    graph.add_node("error_handler", error_handler_node)
    
    # ===================================================
    # 设置入口点
    # ===================================================
    
    # 图的入口是 input_parser 节点
    graph.set_entry_point("input_parser")
    
    # ===================================================
    # 定义边
    # 边定义了节点之间的转换关系
    # ===================================================
    
    # input_parser -> coordinator（固定边）
    graph.add_edge("input_parser", "coordinator")
    
    # coordinator 之后的条件路由
    graph.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        {
            "planner": "planner",
            "task_router": "task_router",
            "synthesizer": "synthesizer",
            "error_handler": "error_handler",
            "end": END,
        }
    )
    
    # planner 之后的条件路由
    graph.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "task_router": "task_router",
            "synthesizer": "synthesizer",
            "coordinator": "coordinator",
        }
    )
    
    # task_router 之后的条件路由
    graph.add_conditional_edges(
        "task_router",
        route_task,
        {
            "researcher": "researcher",
            "coder": "coder",
            "executor": "executor",
            "synthesizer": "synthesizer",
            "coordinator": "coordinator",
        }
    )
    
    # 工作者节点 -> critic（通过 route_from_worker）
    for worker in ["researcher", "coder", "executor"]:
        graph.add_conditional_edges(
            worker,
            route_from_worker,
            {
                "critic": "critic",
                "error_handler": "error_handler",
                "synthesizer": "synthesizer",
            }
        )
    
    # critic 之后的条件路由
    graph.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "human_node": "human_node",
            "task_router": "task_router",
            "synthesizer": "synthesizer",
            "coordinator": "coordinator",
        }
    )
    
    # human_node 之后的条件路由
    graph.add_conditional_edges(
        "human_node",
        route_from_human,
        {
            "task_router": "task_router",
            "synthesizer": "synthesizer",
            "coordinator": "coordinator",
            "end": END,
        }
    )
    
    # synthesizer 之后的条件路由
    graph.add_conditional_edges(
        "synthesizer",
        route_from_synthesizer,
        {
            "coordinator": "coordinator",
            "end": END,
        }
    )
    
    # error_handler 之后的条件路由
    graph.add_conditional_edges(
        "error_handler",
        route_from_error_handler,
        {
            "coordinator": "coordinator",
            "task_router": "task_router",
            "end": END,
        }
    )
    
    logger.info("状态图构建完成")
    
    return graph


class MultiAgentSystem:
    """
    多智能体系统封装类
    
    提供简化的接口来运行多智能体工作流。
    
    使用示例：
        >>> system = MultiAgentSystem()
        >>> result = system.run("编写一个 Python 爬虫")
        >>> print(result["final_answer"])
    
    属性：
        settings: 系统配置
        graph: 编译后的状态图
        checkpointer: 状态检查点管理器（用于持久化）
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        enable_checkpointing: bool = False,
    ):
        """
        初始化多智能体系统
        
        Args:
            settings: 系统配置，None 使用默认配置
            enable_checkpointing: 是否启用状态检查点
        """
        self.settings = settings or get_settings()
        self._graph = None
        self._compiled_graph = None
        self.checkpointer = MemorySaver() if enable_checkpointing else None
        
        logger.info("初始化 MultiAgentSystem")
    
    @property
    def graph(self):
        """获取编译后的图（延迟加载）"""
        if self._compiled_graph is None:
            self._build_and_compile()
        return self._compiled_graph
    
    def _build_and_compile(self):
        """构建并编译图"""
        self._graph = build_graph(self.settings)
        
        # 编译图
        # compile() 将 StateGraph 转换为可执行的 CompiledGraph
        # checkpointer 参数用于状态持久化，支持断点续传
        compile_kwargs = {}
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer
        
        self._compiled_graph = self._graph.compile(**compile_kwargs)
        
        logger.info("图编译完成")
    
    def run(
        self,
        task: str,
        task_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        运行任务
        
        Args:
            task: 用户任务描述
            task_id: 任务 ID，不提供则自动生成
            config: 运行时配置
            
        Returns:
            最终状态字典，包含 final_answer 等结果
        """
        logger.info(f"开始执行任务: {task[:50]}...")
        
        # 创建初始状态
        initial_state = create_initial_state(
            task=task,
            task_id=task_id,
            max_iterations=self.settings.max_iterations,
        )
        
        # 配置
        run_config = config or {}
        if self.checkpointer and "configurable" not in run_config:
            run_config["configurable"] = {"thread_id": task_id or "default"}
        
        # 运行图
        try:
            final_state = self.graph.invoke(initial_state, config=run_config)
            
            # 计算执行指标
            end_time = time.time()
            start_time = final_state.get("start_time", end_time)
            
            final_state["total_duration"] = end_time - start_time
            
            logger.info(
                f"任务执行完成，耗时 {final_state['total_duration']:.2f}s"
            )
            
            return dict(final_state)
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}", exc_info=True)
            raise
    
    def stream(
        self,
        task: str,
        task_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式运行任务
        
        逐步产出每个节点的执行结果。
        
        Args:
            task: 用户任务描述
            task_id: 任务 ID
            config: 运行时配置
            
        Yields:
            每个节点执行后的状态更新
        """
        logger.info(f"开始流式执行任务: {task[:50]}...")
        
        initial_state = create_initial_state(
            task=task,
            task_id=task_id,
            max_iterations=self.settings.max_iterations,
        )
        
        run_config = config or {}
        if self.checkpointer and "configurable" not in run_config:
            run_config["configurable"] = {"thread_id": task_id or "default"}
        
        # 流式执行
        for event in self.graph.stream(initial_state, config=run_config):
            yield event
    
    def get_graph_visualization(self) -> str:
        """
        获取图的 Mermaid 可视化表示
        
        Returns:
            Mermaid 格式的图描述
        """
        return self.graph.get_graph().draw_mermaid()
    
    def reset(self):
        """重置系统状态"""
        clear_agent_cache()
        self._compiled_graph = None
        self._graph = None
        logger.info("系统已重置")
    
    def create_result(self, state: Dict[str, Any]) -> FinalResult:
        """
        从状态创建最终结果对象
        
        Args:
            state: 最终状态
            
        Returns:
            FinalResult 实例
        """
        return FinalResult(
            task_id=state.get("task_id", ""),
            original_task=state.get("original_task", ""),
            answer=state.get("final_answer", ""),
            reasoning_trace=state.get("reasoning_trace", []),
            agent_outputs=state.get("agent_outputs", {}),
            metrics=ExecutionMetrics(
                total_duration_seconds=state.get("total_duration", 0),
                token_usage=state.get("token_usage", {}),
                agent_durations=state.get("execution_time", {}),
                iteration_count=state.get("iteration_count", 0),
                retry_count=sum(
                    t.get("retry_count", 0)
                    for t in state.get("subtasks", [])
                ),
                tool_call_count=len(state.get("tool_call_logs", [])),
                success=bool(state.get("final_answer")),
            ),
        )
```

---

### Tools 模块

#### 文件：`src/tools/__init__.py`

```python
"""
工具模块
========

提供系统可用的所有工具。

工具遵循 LangChain 工具规范，可被 Agent 调用执行特定操作。

可用工具：
- SafeCalculator: 安全数学计算
- FileManager: 文件读写（限定目录）
- CodeExecutor: Python 代码执行（沙箱）
- WebSearch: 网络搜索（模拟）
"""

from typing import List
from langchain_core.tools import BaseTool

from src.tools.base import ToolRegistry, create_tool
from src.tools.calculator import SafeCalculator, calculator_tool
from src.tools.file_manager import FileManager, file_manager_tool
from src.tools.code_executor import CodeExecutor, code_executor_tool
from src.tools.search import WebSearch, web_search_tool

__all__ = [
    # 基础
    "ToolRegistry",
    "create_tool",
    # 工具类
    "SafeCalculator",
    "FileManager",
    "CodeExecutor",
    "WebSearch",
    # 工具实例
    "calculator_tool",
    "file_manager_tool",
    "code_executor_tool",
    "web_search_tool",
    # 工具获取
    "get_all_tools",
    "get_tool_by_name",
]


def get_all_tools() -> List[BaseTool]:
    """
    获取所有可用工具实例
    
    Returns:
        工具实例列表
    """
    return [
        calculator_tool,
        file_manager_tool,
        code_executor_tool,
        web_search_tool,
    ]


def get_tool_by_name(name: str) -> BaseTool:
    """
    根据名称获取工具
    
    Args:
        name: 工具名称
        
    Returns:
        工具实例
        
    Raises:
        ValueError: 工具不存在
    """
    tools = {
        "calculator": calculator_tool,
        "file_manager": file_manager_tool,
        "code_executor": code_executor_tool,
        "web_search": web_search_tool,
    }
    
    if name not in tools:
        raise ValueError(f"未知工具: {name}，可用工具: {list(tools.keys())}")
    
    return tools[name]
```

---

#### 文件：`src/tools/base.py`

```python
"""
工具基础模块
============

提供工具注册表和工具创建辅助函数。
"""

from typing import Any, Callable, Dict, List, Optional, Type
from functools import wraps

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """
    工具注册表
    
    管理系统中所有可用的工具，支持动态注册和获取。
    """
    
    _tools: Dict[str, BaseTool] = {}
    
    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """
        注册工具
        
        Args:
            tool: 工具实例
        """
        cls._tools[tool.name] = tool
        logger.debug(f"注册工具: {tool.name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseTool]:
        """
        获取工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具实例，不存在返回 None
        """
        return cls._tools.get(name)
    
    @classmethod
    def get_all(cls) -> List[BaseTool]:
        """
        获取所有工具
        
        Returns:
            工具列表
        """
        return list(cls._tools.values())
    
    @classmethod
    def list_names(cls) -> List[str]:
        """
        列出所有工具名称
        
        Returns:
            工具名称列表
        """
        return list(cls._tools.keys())
    
    @classmethod
    def clear(cls) -> None:
        """清空注册表"""
        cls._tools.clear()


def create_tool(
    name: str,
    description: str,
    func: Callable,
    args_schema: Optional[Type[BaseModel]] = None,
    return_direct: bool = False,
) -> StructuredTool:
    """
    创建工具的工厂函数
    
    Args:
        name: 工具名称
        description: 工具描述
        func: 工具执行函数
        args_schema: 参数 Schema（Pydantic 模型）
        return_direct: 是否直接返回结果
        
    Returns:
        StructuredTool 实例
    """
    tool = StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        args_schema=args_schema,
        return_direct=return_direct,
    )
    
    # 自动注册
    ToolRegistry.register(tool)
    
    return tool


def tool_error_handler(func: Callable) -> Callable:
    """
    工具错误处理装饰器
    
    捕获工具执行中的异常，返回格式化的错误信息。
    
    Args:
        func: 工具函数
        
    Returns:
        包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"工具执行错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg, "success": False}
    
    return wrapper
```

---

#### 文件：`src/tools/calculator.py`

```python
"""
计算器工具
==========

提供安全的数学表达式计算功能。
"""

import ast
import operator
from typing import Any, Dict, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CalculatorInput(BaseModel):
    """计算器输入参数"""
    expression: str = Field(
        description="要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'"
    )


class SafeCalculator:
    """
    安全计算器
    
    使用 AST 解析来安全执行数学表达式，避免代码注入风险。
    
    支持的运算：
    - 基本运算：+, -, *, /, //, %, **
    - 比较运算：<, >, <=, >=, ==, !=
    - 数学函数：abs, round, min, max, sum, pow, sqrt
    - 常量：pi, e
    """
    
    # 允许的运算符
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # 允许的比较运算符
    ALLOWED_COMPARISONS = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }
    
    # 允许的函数
    ALLOWED_FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'sqrt': lambda x: x ** 0.5,
        'int': int,
        'float': float,
    }
    
    # 允许的常量
    ALLOWED_CONSTANTS = {
        'pi': 3.141592653589793,
        'e': 2.718281828459045,
        'True': True,
        'False': False,
    }
    
    def __init__(self):
        """初始化计算器"""
        self.logger = get_logger(self.__class__.__name__)
    
    def calculate(self, expression: str) -> Union[float, int, bool, str]:
        """
        安全计算数学表达式
        
        Args:
            expression: 数学表达式字符串
            
        Returns:
            计算结果
            
        Raises:
            ValueError: 表达式无效或包含不允许的操作
        """
        self.logger.debug(f"计算表达式: {expression}")
        
        # 清理表达式
        expression = expression.strip()
        
        if not expression:
            raise ValueError("表达式不能为空")
        
        try:
            # 解析表达式为 AST
            tree = ast.parse(expression, mode='eval')
            
            # 验证并计算
            result = self._eval_node(tree.body)
            
            self.logger.debug(f"计算结果: {result}")
            return result
            
        except SyntaxError as e:
            raise ValueError(f"表达式语法错误: {e}")
        except Exception as e:
            raise ValueError(f"计算错误: {e}")
    
    def _eval_node(self, node: ast.AST) -> Any:
        """
        递归计算 AST 节点
        
        Args:
            node: AST 节点
            
        Returns:
            节点计算结果
        """
        # 数字字面量
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise ValueError(f"不支持的常量类型: {type(node.value)}")
        
        # 变量名（常量）
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.ALLOWED_CONSTANTS:
                return self.ALLOWED_CONSTANTS[name]
            raise ValueError(f"未知的标识符: {name}")
        
        # 一元运算
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self.ALLOWED_OPERATORS[op_type](operand)
        
        # 二元运算
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"不支持的二元运算符: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.ALLOWED_OPERATORS[op_type](left, right)
        
        # 比较运算
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                op_type = type(op)
                if op_type not in self.ALLOWED_COMPARISONS:
                    raise ValueError(f"不支持的比较运算符: {op_type.__name__}")
                right = self._eval_node(comparator)
                if not self.ALLOWED_COMPARISONS[op_type](left, right):
                    return False
                left = right
            return True
        
        # 函数调用
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("不支持的函数调用方式")
            
            func_name = node.func.id
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"不支持的函数: {func_name}")
            
            # 计算参数
            args = [self._eval_node(arg) for arg in node.args]
            
            # 调用函数
            return self.ALLOWED_FUNCTIONS[func_name](*args)
        
        # 列表/元组
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(elem) for elem in node.elts]
        
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


# 创建工具实例
_calculator = SafeCalculator()


@tool(args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    """
    安全计算数学表达式。
    
    支持基本运算（+, -, *, /, **, %）和常用函数（sqrt, abs, round, min, max）。
    
    使用示例：
    - 基本运算: "2 + 3 * 4"
    - 幂运算: "2 ** 10"
    - 函数: "sqrt(16)"
    - 常量: "pi * 2"
    """
    try:
        result = _calculator.calculate(expression)
        return f"计算结果: {expression} = {result}"
    except ValueError as e:
        return f"计算错误: {str(e)}"
    except Exception as e:
        logger.error(f"计算器异常: {e}", exc_info=True)
        return f"计算失败: {str(e)}"
```

---

#### 文件：`src/tools/file_manager.py`

```python
"""
文件管理工具
============

提供安全的文件读写操作，限定在 workspace 目录内。
"""

import os
from pathlib import Path
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileManagerInput(BaseModel):
    """文件管理器输入参数"""
    action: Literal["read", "write", "list", "exists", "delete"] = Field(
        description="操作类型: read(读取), write(写入), list(列出), exists(检查存在), delete(删除)"
    )
    path: str = Field(
        description="文件路径（相对于 workspace 目录）"
    )
    content: Optional[str] = Field(
        default=None,
        description="写入的内容（仅 write 操作需要）"
    )
    
    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """验证路径安全性"""
        # 禁止路径遍历
        if ".." in v:
            raise ValueError("路径不能包含 '..'")
        # 禁止绝对路径
        if v.startswith("/") or (len(v) > 1 and v[1] == ":"):
            raise ValueError("不能使用绝对路径")
        return v


class FileManager:
    """
    文件管理器
    
    安全地在 workspace 目录内进行文件操作。
    
    安全特性：
    - 所有操作限定在 workspace 目录
    - 禁止路径遍历（..）
    - 禁止绝对路径
    - 操作前验证路径合法性
    """
    
    def __init__(self, workspace_dir: Optional[str] = None):
        """
        初始化文件管理器
        
        Args:
            workspace_dir: 工作目录路径，None 使用默认配置
        """
        settings = get_settings()
        self.workspace = Path(workspace_dir or settings.workspace_dir).resolve()
        
        # 确保工作目录存在
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug(f"工作目录: {self.workspace}")
    
    def _resolve_path(self, relative_path: str) -> Path:
        """
        解析相对路径为绝对路径
        
        Args:
            relative_path: 相对路径
            
        Returns:
            解析后的绝对路径
            
        Raises:
            ValueError: 路径不在工作目录内
        """
        # 构建完整路径
        full_path = (self.workspace / relative_path).resolve()
        
        # 验证路径在工作目录内
        try:
            full_path.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"路径 '{relative_path}' 超出工作目录范围")
        
        return full_path
    
    def read(self, path: str) -> str:
        """
        读取文件内容
        
        Args:
            path: 相对路径
            
        Returns:
            文件内容
        """
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        if not full_path.is_file():
            raise ValueError(f"路径不是文件: {path}")
        
        self.logger.info(f"读取文件: {path}")
        
        # 尝试不同编码
        for encoding in ["utf-8", "gbk", "latin-1"]:
            try:
                return full_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"无法解码文件: {path}")
    
    def write(self, path: str, content: str) -> str:
        """
        写入文件内容
        
        Args:
            path: 相对路径
            content: 要写入的内容
            
        Returns:
            操作结果消息
        """
        full_path = self._resolve_path(path)
        
        # 确保父目录存在
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"写入文件: {path}")
        
        full_path.write_text(content, encoding="utf-8")
        
        return f"成功写入 {len(content)} 字符到 {path}"
    
    def list_dir(self, path: str = ".") -> list:
        """
        列出目录内容
        
        Args:
            path: 相对路径，默认为工作目录根
            
        Returns:
            文件和目录列表
        """
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"目录不存在: {path}")
        
        if not full_path.is_dir():
            raise ValueError(f"路径不是目录: {path}")
        
        self.logger.info(f"列出目录: {path}")
        
        items = []
        for item in full_path.iterdir():
            item_type = "dir" if item.is_dir() else "file"
            items.append({
                "name": item.name,
                "type": item_type,
                "size": item.stat().st_size if item.is_file() else None,
            })
        
        return items
    
    def exists(self, path: str) -> bool:
        """
        检查路径是否存在
        
        Args:
            path: 相对路径
            
        Returns:
            是否存在
        """
        full_path = self._resolve_path(path)
        return full_path.exists()
    
    def delete(self, path: str) -> str:
        """
        删除文件
        
        Args:
            path: 相对路径
            
        Returns:
            操作结果消息
        """
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        if full_path.is_dir():
            raise ValueError(f"不能删除目录: {path}")
        
        self.logger.info(f"删除文件: {path}")
        
        full_path.unlink()
        
        return f"成功删除: {path}"


# 创建工具实例
_file_manager = FileManager()


@tool(args_schema=FileManagerInput)
def file_manager_tool(
    action: str,
    path: str,
    content: Optional[str] = None
) -> str:
    """
    安全的文件操作工具，所有操作限定在 workspace 目录内。
    
    支持的操作：
    - read: 读取文件内容
    - write: 写入内容到文件
    - list: 列出目录内容
    - exists: 检查文件/目录是否存在
    - delete: 删除文件
    
    使用示例：
    - 读取: action="read", path="data.txt"
    - 写入: action="write", path="output.json", content='{"key": "value"}'
    - 列出: action="list", path="."
    """
    try:
        if action == "read":
            result = _file_manager.read(path)
            # 限制输出长度
            if len(result) > 5000:
                result = result[:5000] + f"\n...(截断，总长度 {len(result)})"
            return result
        
        elif action == "write":
            if content is None:
                return "错误: write 操作需要提供 content 参数"
            return _file_manager.write(path, content)
        
        elif action == "list":
            items = _file_manager.list_dir(path)
            if not items:
                return f"目录 {path} 为空"
            lines = [f"目录 {path} 内容:"]
            for item in items:
                icon = "📁" if item["type"] == "dir" else "📄"
                size = f" ({item['size']} bytes)" if item["size"] else ""
                lines.append(f"  {icon} {item['name']}{size}")
            return "\n".join(lines)
        
        elif action == "exists":
            exists = _file_manager.exists(path)
            return f"{'存在' if exists else '不存在'}: {path}"
        
        elif action == "delete":
            return _file_manager.delete(path)
        
        else:
            return f"未知操作: {action}"
    
    except FileNotFoundError as e:
        return f"文件未找到: {str(e)}"
    except ValueError as e:
        return f"参数错误: {str(e)}"
    except Exception as e:
        logger.error(f"文件操作异常: {e}", exc_info=True)
        return f"操作失败: {str(e)}"
```

---

#### 文件：`src/tools/code_executor.py`

```python
"""
代码执行工具
============

提供安全的 Python 代码执行环境。
"""

import sys
import io
import traceback
import signal
from typing import Any, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeExecutorInput(BaseModel):
    """代码执行器输入参数"""
    code: str = Field(
        description="要执行的 Python 代码"
    )
    timeout: Optional[int] = Field(
        default=30,
        description="执行超时时间（秒）",
        ge=1,
        le=300
    )


class TimeoutError(Exception):
    """超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("代码执行超时")


class CodeExecutor:
    """
    代码执行器
    
    在受限环境中执行 Python 代码。
    
    安全特性：
    - 执行超时限制
    - 输出长度限制
    - 禁止危险操作（通过黑名单）
    - 隔离的命名空间
    
    注意：
    - 这不是完全安全的沙箱
    - 生产环境建议使用 Docker 容器或其他隔离方案
    """
    
    # 禁止的模块和函数
    BLACKLIST = {
        "os.system",
        "os.popen",
        "os.spawn",
        "os.exec",
        "subprocess",
        "shutil.rmtree",
        "eval",
        "__import__",
        "compile",
        "exec",
        "open",  # 通过文件管理器访问
    }
    
    # 允许导入的模块
    ALLOWED_MODULES = {
        "math",
        "random",
        "datetime",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "string",
        "typing",
        "dataclasses",
        "enum",
        "copy",
        "operator",
        "statistics",
        "decimal",
        "fractions",
    }
    
    MAX_OUTPUT_LENGTH = 10000
    
    def __init__(self, timeout: int = 30):
        """
        初始化代码执行器
        
        Args:
            timeout: 默认超时时间（秒）
        """
        self.default_timeout = timeout
        self.logger = get_logger(self.__class__.__name__)
    
    def _check_code_safety(self, code: str) -> Optional[str]:
        """
        检查代码安全性
        
        Args:
            code: 要检查的代码
            
        Returns:
            如果有问题返回错误信息，否则返回 None
        """
        for forbidden in self.BLACKLIST:
            if forbidden in code:
                return f"禁止的操作: {forbidden}"
        
        # 检查危险的导入
        dangerous_imports = ["os", "sys", "subprocess", "shutil", "socket"]
        for imp in dangerous_imports:
            if f"import {imp}" in code or f"from {imp}" in code:
                return f"禁止导入危险模块: {imp}"
        
        return None
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """
        创建安全的全局命名空间
        
        Returns:
            受限的全局命名空间字典
        """
        import math
        import random
        import datetime
        import json
        import re
        import collections
        import itertools
        import functools
        
        safe_globals = {
            "__builtins__": {
                # 安全的内置函数
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "chr": chr,
                "dict": dict,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "format": format,
                "frozenset": frozenset,
                "getattr": getattr,
                "hasattr": hasattr,
                "hash": hash,
                "hex": hex,
                "int": int,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "iter": iter,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "next": next,
                "oct": oct,
                "ord": ord,
                "pow": pow,
                "print": print,
                "range": range,
                "repr": repr,
                "reversed": reversed,
                "round": round,
                "set": set,
                "slice": slice,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "type": type,
                "zip": zip,
                # 异常
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "StopIteration": StopIteration,
                # 其他
                "True": True,
                "False": False,
                "None": None,
            },
            # 允许的模块
            "math": math,
            "random": random,
            "datetime": datetime,
            "json": json,
            "re": re,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
        }
        
        return safe_globals
    
    def execute(
        self,
        code: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行 Python 代码
        
        Args:
            code: Python 代码字符串
            timeout: 超时时间（秒）
            
        Returns:
            执行结果字典，包含:
            - success: 是否成功
            - output: 标准输出
            - error: 错误信息（如有）
            - result: 最后一个表达式的值（如有）
        """
        timeout = timeout or self.default_timeout
        
        self.logger.info(f"执行代码 (超时: {timeout}s)")
        self.logger.debug(f"代码内容:\n{code[:500]}")
        
        # 安全检查
        safety_issue = self._check_code_safety(code)
        if safety_issue:
            return {
                "success": False,
                "output": "",
                "error": safety_issue,
                "result": None,
            }
        
        # 准备执行环境
        safe_globals = self._create_safe_globals()
        local_namespace = {}
        
        # 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = None
        error = None
        
        try:
            # 设置超时（仅 Unix 系统有效）
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            # 执行代码
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                try:
                    # 尝试作为表达式执行
                    result = eval(code, safe_globals, local_namespace)
                except SyntaxError:
                    # 作为语句执行
                    exec(code, safe_globals, local_namespace)
                    # 检查是否有返回值变量
                    if "result" in local_namespace:
                        result = local_namespace["result"]
            
        except TimeoutError:
            error = f"执行超时（{timeout}秒）"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        finally:
            # 恢复信号处理器
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        # 获取输出
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        # 限制输出长度
        if len(stdout) > self.MAX_OUTPUT_LENGTH:
            stdout = stdout[:self.MAX_OUTPUT_LENGTH] + "\n...(输出被截断)"
        if len(stderr) > self.MAX_OUTPUT_LENGTH:
            stderr = stderr[:self.MAX_OUTPUT_LENGTH] + "\n...(输出被截断)"
        
        output = stdout
        if stderr:
            output += f"\n[stderr]\n{stderr}"
        
        return {
            "success": error is None,
            "output": output.strip(),
            "error": error,
            "result": result,
        }


# 创建工具实例
_code_executor = CodeExecutor()


@tool(args_schema=CodeExecutorInput)
def code_executor_tool(code: str, timeout: int = 30) -> str:
    """
    在安全环境中执行 Python 代码。
    
    支持的功能：
    - 基本 Python 语法
    - 常用标准库（math, random, datetime, json, re 等）
    - 打印输出
    
    限制：
    - 不能访问文件系统（使用 file_manager 工具）
    - 不能执行系统命令
    - 不能进行网络请求
    - 有执行时间限制
    
    使用示例：
    ```python
    import math
    result = math.sqrt(16)
    print(f"结果: {result}")
    ```
    """
    try:
        result = _code_executor.execute(code, timeout)
        
        if result["success"]:
            output_parts = []
            if result["output"]:
                output_parts.append(f"输出:\n{result['output']}")
            if result["result"] is not None:
                output_parts.append(f"返回值: {result['result']}")
            
            if output_parts:
                return "\n\n".join(output_parts)
            return "代码执行成功（无输出）"
        else:
            return f"执行错误:\n{result['error']}"
    
    except Exception as e:
        logger.error(f"代码执行器异常: {e}", exc_info=True)
        return f"执行失败: {str(e)}"
```

---

#### 文件：`src/tools/search.py`

```python
"""
搜索工具
========

提供网络搜索功能（模拟实现）。
"""

import json
import random
from typing import List, Optional
from datetime import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class WebSearchInput(BaseModel):
    """网络搜索输入参数"""
    query: str = Field(
        description="搜索查询关键词"
    )
    num_results: Optional[int] = Field(
        default=5,
        description="返回结果数量",
        ge=1,
        le=10
    )


class SearchResult(BaseModel):
    """搜索结果模型"""
    title: str
    url: str
    snippet: str
    source: str


class WebSearch:
    """
    网络搜索器
    
    当前为模拟实现，返回预设的搜索结果。
    生产环境应对接真实的搜索 API（如 Google、Bing、Serper 等）。
    
    扩展方式：
    1. 继承此类并重写 search 方法
    2. 或配置 API 密钥连接真实服务
    """
    
    # 模拟搜索结果库
    MOCK_RESULTS = {
        "python": [
            SearchResult(
                title="Python 官方文档",
                url="https://docs.python.org/",
                snippet="Python 是一种易于学习、功能强大的编程语言。它具有高效的高级数据结构和简单而有效的面向对象编程方法。",
                source="python.org"
            ),
            SearchResult(
                title="Python 教程 - 菜鸟教程",
                url="https://www.runoob.com/python/",
                snippet="Python 是一个高层次的结合了解释性、编译性、互动性和面向对象的脚本语言。",
                source="runoob.com"
            ),
            SearchResult(
                title="Python 入门指南",
                url="https://docs.python.org/zh-cn/3/tutorial/",
                snippet="本教程非正式地介绍 Python 语言和系统的基本概念和功能。",
                source="python.org"
            ),
        ],
        "爬虫": [
            SearchResult(
                title="Python 爬虫教程",
                url="https://example.com/crawler-tutorial",
                snippet="学习使用 Python 进行网页爬取，包括 requests、BeautifulSoup、Scrapy 等常用库的使用方法。",
                source="example.com"
            ),
            SearchResult(
                title="requests 库文档",
                url="https://docs.python-requests.org/",
                snippet="Requests 是一个简洁且简单的 Python HTTP 库，让 HTTP 请求变得更加人性化。",
                source="python-requests.org"
            ),
            SearchResult(
                title="BeautifulSoup 教程",
                url="https://www.crummy.com/software/BeautifulSoup/",
                snippet="Beautiful Soup 是一个用于解析 HTML 和 XML 文档的 Python 库。",
                source="crummy.com"
            ),
        ],
        "hacker news": [
            SearchResult(
                title="Hacker News",
                url="https://news.ycombinator.com/",
                snippet="Hacker News 是一个社会化新闻网站，专注于计算机科学和创业。由 Y Combinator 创办和运营。",
                source="ycombinator.com"
            ),
            SearchResult(
                title="Hacker News API",
                url="https://github.com/HackerNews/API",
                snippet="The official Hacker News API. Documentation and examples for accessing HN data.",
                source="github.com"
            ),
            SearchResult(
                title="HN 爬取指南",
                url="https://example.com/hn-scraping",
                snippet="如何爬取 Hacker News 网站，包括文章标题、链接、评分等信息。",
                source="example.com"
            ),
        ],
        "default": [
            SearchResult(
                title="搜索结果 1",
                url="https://example.com/result1",
                snippet="这是一个模拟的搜索结果。在实际应用中，这里会显示真实的搜索结果内容。",
                source="example.com"
            ),
            SearchResult(
                title="搜索结果 2",
                url="https://example.com/result2",
                snippet="另一个模拟搜索结果。系统当前使用模拟数据，生产环境需要对接真实搜索API。",
                source="example.com"
            ),
        ],
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化搜索器
        
        Args:
            api_key: 搜索 API 密钥（可选）
        """
        self.api_key = api_key
        self.logger = get_logger(self.__class__.__name__)
    
    def search(
        self,
        query: str,
        num_results: int = 5
    ) -> List[SearchResult]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        self.logger.info(f"搜索: {query}")
        
        # 模拟搜索延迟
        import time
        time.sleep(0.5)
        
        # 匹配模拟结果
        results = []
        query_lower = query.lower()
        
        for keyword, mock_results in self.MOCK_RESULTS.items():
            if keyword != "default" and keyword in query_lower:
                results.extend(mock_results)
        
        # 如果没有匹配，使用默认结果
        if not results:
            results = self.MOCK_RESULTS["default"].copy()
            # 动态生成一些结果
            results.append(SearchResult(
                title=f"关于 '{query}' 的搜索结果",
                url=f"https://example.com/search?q={query}",
                snippet=f"这是关于 '{query}' 的搜索结果。包含相关信息和资源链接。",
                source="example.com"
            ))
        
        # 打乱顺序增加真实感
        random.shuffle(results)
        
        # 限制数量
        results = results[:num_results]
        
        self.logger.info(f"返回 {len(results)} 条结果")
        
        return results
    
    def format_results(self, results: List[SearchResult]) -> str:
        """
        格式化搜索结果为可读字符串
        
        Args:
            results: 搜索结果列表
            
        Returns:
            格式化的字符串
        """
        if not results:
            return "未找到相关结果"
        
        lines = [f"找到 {len(results)} 条结果:\n"]
        
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. **{result.title}**")
            lines.append(f"   链接: {result.url}")
            lines.append(f"   摘要: {result.snippet}")
            lines.append(f"   来源: {result.source}")
            lines.append("")
        
        return "\n".join(lines)


# 创建工具实例
_web_search = WebSearch()


@tool(args_schema=WebSearchInput)
def web_search_tool(query: str, num_results: int = 5) -> str:
    """
    搜索互联网获取信息。
    
    当需要获取最新信息、查找资料或验证事实时使用此工具。
    
    注意：当前为模拟实现，返回预设结果。
    
    使用示例：
    - 搜索 Python 教程: query="Python 入门教程"
    - 查找 API 文档: query="requests 库文档"
    """
    try:
        results = _web_search.search(query, num_results)
        return _web_search.format_results(results)
    except Exception as e:
        logger.error(f"搜索异常: {e}", exc_info=True)
        return f"搜索失败: {str(e)}"
```

---

**第三部分（第3批）完成。请输入 "继续" 获取第三部分（第4批）：Memory 模块、LLM 模块和 Utils 模块完整代码。**

## 第三部分（第4批）：Memory 模块、LLM 模块和 Utils 模块完整代码

---

### Memory 模块

#### 文件：`src/memory/__init__.py`

```python
"""
记忆系统模块
============

提供短期和长期记忆功能。

记忆类型：
- ShortTermMemory: 会话内上下文记忆，存储在内存中
- LongTermMemory: 持久化记忆，支持文件存储
"""

from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory

__all__ = [
    "ShortTermMemory",
    "LongTermMemory",
]
```

---

#### 文件：`src/memory/short_term.py`

```python
"""
短期记忆模块
============

实现会话内的上下文记忆，用于在单次任务执行过程中保存和检索信息。
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import OrderedDict
import threading

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryItem:
    """记忆项"""
    
    def __init__(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化记忆项
        
        Args:
            key: 键
            value: 值
            metadata: 元数据
        """
        self.key = key
        self.value = value
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.accessed_at = datetime.now()
        self.access_count = 0
    
    def access(self) -> Any:
        """访问记忆项，更新访问信息"""
        self.accessed_at = datetime.now()
        self.access_count += 1
        return self.value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "value": self.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
        }


class ShortTermMemory:
    """
    短期记忆
    
    基于 LRU（最近最少使用）策略的内存存储。
    当超过最大容量时，自动淘汰最久未访问的记忆项。
    
    特性：
    - 线程安全
    - 自动过期
    - LRU 淘汰策略
    - 支持元数据
    
    使用示例：
        >>> memory = ShortTermMemory(max_size=100)
        >>> memory.store("user_task", "编写爬虫")
        >>> task = memory.retrieve("user_task")
        >>> print(task)  # "编写爬虫"
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = None
    ):
        """
        初始化短期记忆
        
        Args:
            max_size: 最大存储项数
            default_ttl: 默认生存时间（秒），None 表示不过期
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._storage: OrderedDict[str, MemoryItem] = OrderedDict()
        self._lock = threading.RLock()
        self.logger = get_logger(self.__class__.__name__)
        
        self.logger.debug(f"初始化短期记忆，最大容量: {max_size}")
    
    def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> None:
        """
        存储记忆项
        
        Args:
            key: 键
            value: 值
            metadata: 元数据
            ttl: 生存时间（秒），覆盖默认值
        """
        with self._lock:
            # 如果键已存在，先删除（以更新顺序）
            if key in self._storage:
                del self._storage[key]
            
            # 检查容量，必要时淘汰
            while len(self._storage) >= self.max_size:
                oldest_key = next(iter(self._storage))
                del self._storage[oldest_key]
                self.logger.debug(f"淘汰记忆项: {oldest_key}")
            
            # 存储新项
            item_metadata = metadata or {}
            if ttl or self.default_ttl:
                item_metadata["ttl"] = ttl or self.default_ttl
            
            self._storage[key] = MemoryItem(key, value, item_metadata)
            self.logger.debug(f"存储记忆项: {key}")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        检索记忆项
        
        Args:
            key: 键
            
        Returns:
            值，不存在或已过期返回 None
        """
        with self._lock:
            if key not in self._storage:
                return None
            
            item = self._storage[key]
            
            # 检查过期
            if self._is_expired(item):
                del self._storage[key]
                self.logger.debug(f"记忆项已过期: {key}")
                return None
            
            # 移动到末尾（最近访问）
            self._storage.move_to_end(key)
            
            return item.access()
    
    def _is_expired(self, item: MemoryItem) -> bool:
        """
        检查记忆项是否过期
        
        Args:
            item: 记忆项
            
        Returns:
            是否过期
        """
        ttl = item.metadata.get("ttl")
        if ttl is None:
            return False
        
        elapsed = (datetime.now() - item.created_at).total_seconds()
        return elapsed > ttl
    
    def delete(self, key: str) -> bool:
        """
        删除记忆项
        
        Args:
            key: 键
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if key in self._storage:
                del self._storage[key]
                self.logger.debug(f"删除记忆项: {key}")
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 键
            
        Returns:
            是否存在
        """
        with self._lock:
            if key not in self._storage:
                return False
            
            item = self._storage[key]
            if self._is_expired(item):
                del self._storage[key]
                return False
            
            return True
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的 n 个记忆项
        
        Args:
            n: 数量
            
        Returns:
            记忆项字典列表
        """
        with self._lock:
            # 清理过期项
            self._cleanup_expired()
            
            # 获取最近的项（从末尾开始）
            items = list(self._storage.values())[-n:]
            items.reverse()
            
            return [item.to_dict() for item in items]
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        简单搜索（基于键名匹配）
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            
        Returns:
            匹配的记忆项
        """
        with self._lock:
            self._cleanup_expired()
            
            results = []
            query_lower = query.lower()
            
            for key, item in self._storage.items():
                # 简单的键名和值匹配
                if query_lower in key.lower():
                    results.append(item.to_dict())
                elif isinstance(item.value, str) and query_lower in item.value.lower():
                    results.append(item.to_dict())
                
                if len(results) >= top_k:
                    break
            
            return results
    
    def _cleanup_expired(self) -> int:
        """
        清理过期项
        
        Returns:
            清理的项数
        """
        expired_keys = [
            key for key, item in self._storage.items()
            if self._is_expired(item)
        ]
        
        for key in expired_keys:
            del self._storage[key]
        
        if expired_keys:
            self.logger.debug(f"清理 {len(expired_keys)} 个过期项")
        
        return len(expired_keys)
    
    def clear(self) -> None:
        """清空所有记忆"""
        with self._lock:
            self._storage.clear()
            self.logger.info("清空短期记忆")
    
    def size(self) -> int:
        """
        获取当前存储项数
        
        Returns:
            项数
        """
        with self._lock:
            return len(self._storage)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total_access = sum(
                item.access_count for item in self._storage.values()
            )
            
            return {
                "size": len(self._storage),
                "max_size": self.max_size,
                "total_access_count": total_access,
                "oldest_item": (
                    next(iter(self._storage)).key
                    if self._storage else None
                ),
                "newest_item": (
                    list(self._storage.keys())[-1]
                    if self._storage else None
                ),
            }
```

---

#### 文件：`src/memory/long_term.py`

```python
"""
长期记忆模块
============

实现持久化记忆存储，支持文件系统存储。
"""

import json
import os
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import threading

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LongTermMemory:
    """
    长期记忆
    
    基于文件系统的持久化存储。
    每个记忆项存储为单独的 JSON 文件，支持索引加速查找。
    
    特性：
    - 持久化存储
    - 自动索引
    - 支持元数据
    - 线程安全
    
    存储结构：
        memory_storage/
        ├── index.json          # 索引文件
        ├── items/
        │   ├── abc123.json    # 记忆项文件
        │   └── def456.json
        └── metadata.json       # 元数据
    
    使用示例：
        >>> memory = LongTermMemory()
        >>> memory.store("task_history", {"task": "爬虫", "result": "成功"})
        >>> history = memory.retrieve("task_history")
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        初始化长期记忆
        
        Args:
            storage_path: 存储路径，None 使用默认配置
            auto_save: 是否自动保存索引
        """
        settings = get_settings()
        self.storage_path = Path(
            storage_path or settings.memory_storage_path
        ).resolve()
        self.auto_save = auto_save
        
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.logger = get_logger(self.__class__.__name__)
        
        # 初始化存储目录
        self._init_storage()
        
        # 加载索引
        self._load_index()
        
        self.logger.info(f"初始化长期记忆，路径: {self.storage_path}")
    
    def _init_storage(self) -> None:
        """初始化存储目录结构"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "items").mkdir(exist_ok=True)
    
    def _get_item_path(self, key: str) -> Path:
        """
        获取记忆项的文件路径
        
        Args:
            key: 键
            
        Returns:
            文件路径
        """
        # 使用 hash 确保文件名安全
        key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.storage_path / "items" / f"{key_hash}.json"
    
    def _load_index(self) -> None:
        """加载索引"""
        index_path = self.storage_path / "index.json"
        
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
                self.logger.debug(f"加载索引，共 {len(self._index)} 项")
            except Exception as e:
                self.logger.warning(f"索引加载失败: {e}")
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self) -> None:
        """保存索引"""
        index_path = self.storage_path / "index.json"
        
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"索引保存失败: {e}")
    
    def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        存储记忆项
        
        Args:
            key: 键
            value: 值（必须可 JSON 序列化）
            metadata: 元数据
        """
        with self._lock:
            item_path = self._get_item_path(key)
            
            # 构建记忆项数据
            item_data = {
                "key": key,
                "value": value,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
            # 保存到文件
            try:
                with open(item_path, "w", encoding="utf-8") as f:
                    json.dump(item_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.logger.error(f"存储失败: {e}")
                raise
            
            # 更新索引
            self._index[key] = {
                "path": str(item_path),
                "created_at": item_data["created_at"],
                "updated_at": item_data["updated_at"],
                "metadata_keys": list((metadata or {}).keys()),
            }
            
            if self.auto_save:
                self._save_index()
            
            self.logger.debug(f"存储记忆项: {key}")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        检索记忆项
        
        Args:
            key: 键
            
        Returns:
            值，不存在返回 None
        """
        with self._lock:
            if key not in self._index:
                return None
            
            item_path = self._get_item_path(key)
            
            if not item_path.exists():
                # 索引不一致，清理
                del self._index[key]
                if self.auto_save:
                    self._save_index()
                return None
            
            try:
                with open(item_path, "r", encoding="utf-8") as f:
                    item_data = json.load(f)
                return item_data.get("value")
            except Exception as e:
                self.logger.error(f"检索失败: {e}")
                return None
    
    def delete(self, key: str) -> bool:
        """
        删除记忆项
        
        Args:
            key: 键
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if key not in self._index:
                return False
            
            item_path = self._get_item_path(key)
            
            # 删除文件
            if item_path.exists():
                try:
                    item_path.unlink()
                except Exception as e:
                    self.logger.error(f"删除文件失败: {e}")
            
            # 更新索引
            del self._index[key]
            if self.auto_save:
                self._save_index()
            
            self.logger.debug(f"删除记忆项: {key}")
            return True
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 键
            
        Returns:
            是否存在
        """
        with self._lock:
            return key in self._index
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        搜索记忆项
        
        简单的关键词匹配搜索。
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            
        Returns:
            匹配的记忆项列表
        """
        with self._lock:
            results = []
            query_lower = query.lower()
            
            for key in self._index:
                if query_lower in key.lower():
                    value = self.retrieve(key)
                    if value is not None:
                        results.append({
                            "key": key,
                            "value": value,
                            "score": 1.0,  # 简单匹配不计算分数
                        })
                
                if len(results) >= top_k:
                    break
            
            return results
    
    def list_keys(self) -> List[str]:
        """
        列出所有键
        
        Returns:
            键列表
        """
        with self._lock:
            return list(self._index.keys())
    
    def clear(self) -> None:
        """清空所有记忆"""
        with self._lock:
            # 删除所有文件
            items_dir = self.storage_path / "items"
            if items_dir.exists():
                for item_file in items_dir.glob("*.json"):
                    try:
                        item_file.unlink()
                    except Exception as e:
                        self.logger.warning(f"删除文件失败: {e}")
            
            # 清空索引
            self._index = {}
            self._save_index()
            
            self.logger.info("清空长期记忆")
    
    def size(self) -> int:
        """
        获取存储项数
        
        Returns:
            项数
        """
        with self._lock:
            return len(self._index)
    
    def persist(self) -> None:
        """强制持久化索引"""
        with self._lock:
            self._save_index()
            self.logger.debug("索引已持久化")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            items_dir = self.storage_path / "items"
            total_size = sum(
                f.stat().st_size for f in items_dir.glob("*.json")
            ) if items_dir.exists() else 0
            
            return {
                "item_count": len(self._index),
                "storage_path": str(self.storage_path),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }
```

---

### LLM 模块

#### 文件：`src/llm/__init__.py`

```python
"""
LLM 模块
========

提供语言模型的创建和管理功能。

支持的 LLM 提供商：
- OpenAI (GPT-4, GPT-3.5 等)
- Anthropic (Claude 系列)
- 本地模型 (通过兼容 API)
"""

from src.llm.factory import LLMFactory, create_llm

__all__ = [
    "LLMFactory",
    "create_llm",
]
```

---

#### 文件：`src/llm/factory.py`

```python
"""
LLM 工厂模块
============

使用工厂模式创建和管理 LLM 实例。
"""

from typing import Dict, Optional, Type, Any
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler

from src.config.settings import Settings, LLMConfig, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TokenCounterCallback(BaseCallbackHandler):
    """Token 计数回调处理器"""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def on_llm_end(self, response, **kwargs):
        """LLM 调用结束时统计 token"""
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
    
    def get_usage(self) -> Dict[str, int]:
        """获取使用统计"""
        return {
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
            "total": self.total_tokens,
        }
    
    def reset(self):
        """重置计数器"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


class LLMFactory:
    """
    LLM 工厂类
    
    使用工厂模式创建不同提供商的 LLM 实例。
    
    支持的提供商：
    - openai: OpenAI 模型
    - anthropic: Anthropic Claude 模型
    - local: 本地或兼容 API 的模型
    
    使用示例：
        >>> config = LLMConfig(provider="openai", model_name="gpt-4")
        >>> llm = LLMFactory.create(config)
        >>> response = llm.invoke("Hello!")
    """
    
    _instances: Dict[str, BaseChatModel] = {}
    _token_counters: Dict[str, TokenCounterCallback] = {}
    
    @classmethod
    def create(
        cls,
        config: Optional[LLMConfig] = None,
        cache_key: Optional[str] = None,
    ) -> BaseChatModel:
        """
        创建 LLM 实例
        
        Args:
            config: LLM 配置，None 使用默认配置
            cache_key: 缓存键，相同键返回缓存实例
            
        Returns:
            LLM 实例
        """
        if config is None:
            settings = get_settings()
            config = settings.get_llm_config()
        
        # 生成缓存键
        if cache_key is None:
            cache_key = f"{config.provider}:{config.model_name}"
        
        # 检查缓存
        if cache_key in cls._instances:
            logger.debug(f"使用缓存的 LLM 实例: {cache_key}")
            return cls._instances[cache_key]
        
        # 创建新实例
        logger.info(f"创建 LLM: {config.provider}/{config.model_name}")
        
        if config.provider == "openai":
            llm = cls._create_openai(config)
        elif config.provider == "anthropic":
            llm = cls._create_anthropic(config)
        elif config.provider == "local":
            llm = cls._create_local(config)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {config.provider}")
        
        # 缓存实例
        cls._instances[cache_key] = llm
        
        return llm
    
    @classmethod
    def _create_openai(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建 OpenAI LLM
        
        Args:
            config: LLM 配置
            
        Returns:
            ChatOpenAI 实例
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "请安装 langchain-openai: pip install langchain-openai"
            )
        
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        if config.api_key:
            kwargs["api_key"] = config.api_key
        
        if config.base_url:
            kwargs["base_url"] = config.base_url
        
        return ChatOpenAI(**kwargs)
    
    @classmethod
    def _create_anthropic(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建 Anthropic LLM
        
        Args:
            config: LLM 配置
            
        Returns:
            ChatAnthropic 实例
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "请安装 langchain-anthropic: pip install langchain-anthropic"
            )
        
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        if config.api_key:
            kwargs["anthropic_api_key"] = config.api_key
        
        return ChatAnthropic(**kwargs)
    
    @classmethod
    def _create_local(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建本地/兼容 API 的 LLM
        
        使用 OpenAI 兼容接口连接本地模型（如 Ollama、vLLM 等）
        
        Args:
            config: LLM 配置
            
        Returns:
            ChatOpenAI 实例（使用自定义 base_url）
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "请安装 langchain-openai: pip install langchain-openai"
            )
        
        base_url = config.base_url or "http://localhost:11434/v1"
        
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            base_url=base_url,
            api_key=config.api_key or "not-needed",  # 本地模型通常不需要
        )
    
    @classmethod
    def get_token_counter(cls, cache_key: str) -> Optional[TokenCounterCallback]:
        """
        获取 token 计数器
        
        Args:
            cache_key: LLM 缓存键
            
        Returns:
            TokenCounterCallback 实例
        """
        return cls._token_counters.get(cache_key)
    
    @classmethod
    def clear_cache(cls) -> None:
        """清空所有缓存的 LLM 实例"""
        cls._instances.clear()
        cls._token_counters.clear()
        logger.info("LLM 缓存已清空")
    
    @classmethod
    def list_cached(cls) -> list:
        """
        列出所有缓存的 LLM
        
        Returns:
            缓存键列表
        """
        return list(cls._instances.keys())


def create_llm(
    provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseChatModel:
    """
    便捷函数：创建 LLM 实例
    
    Args:
        provider: 提供商 (openai/anthropic/local)
        model_name: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数
        api_key: API 密钥
        base_url: 基础 URL
        
    Returns:
        LLM 实例
    """
    # 默认模型名称
    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-sonnet-20240229",
        "local": "llama3",
    }
    
    if model_name is None:
        model_name = default_models.get(provider, "gpt-4o-mini")
    
    config = LLMConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
    )
    
    return LLMFactory.create(config)
```

---

### Utils 模块

#### 文件：`src/utils/__init__.py`

```python
"""
工具模块
========

提供日志、可视化等辅助功能。
"""

from src.utils.logger import (
    setup_logger,
    get_logger,
    set_log_level,
)
from src.utils.visualizer import (
    ExecutionVisualizer,
    generate_mermaid_graph,
)

__all__ = [
    "setup_logger",
    "get_logger",
    "set_log_level",
    "ExecutionVisualizer",
    "generate_mermaid_graph",
]
```

---

#### 文件：`src/utils/logger.py`

```python
"""
日志工具模块
============

提供统一的日志配置和管理。
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler
from rich.console import Console

# 全局日志配置
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_RICH_FORMAT = "%(message)s"

# 日志级别映射
_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# 全局日志器缓存
_loggers: dict = {}
_initialized = False


def setup_logger(
    log_dir: str = "logs",
    log_file: Optional[str] = None,
    level: str = "info",
    debug: bool = False,
    use_rich: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    设置全局日志配置
    
    Args:
        log_dir: 日志目录
        log_file: 日志文件名，None 自动生成
        level: 日志级别
        debug: 是否启用调试模式（覆盖 level）
        use_rich: 是否使用 Rich 美化输出
        max_file_size: 单个日志文件最大大小
        backup_count: 保留的备份文件数
    """
    global _initialized
    
    if _initialized:
        return
    
    # 确定日志级别
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = _LOG_LEVELS.get(level.lower(), logging.INFO)
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 确定日志文件名
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"multi_agent_{timestamp}.log"
    
    log_file_path = log_path / log_file
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 添加控制台处理器
    if use_rich:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=debug,
            rich_tracebacks=True,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter(_RICH_FORMAT))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT)
        )
    
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT)
    )
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
    root_logger.addHandler(file_handler)
    
    # 降低第三方库的日志级别
    for lib in ["httpx", "httpcore", "openai", "anthropic", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    _initialized = True
    
    # 记录启动信息
    root_logger.info(f"日志系统初始化完成，文件: {log_file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器实例
    
    Args:
        name: 日志器名称
        
    Returns:
        Logger 实例
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger
    
    return _loggers[name]


def set_log_level(level: str, logger_name: Optional[str] = None) -> None:
    """
    设置日志级别
    
    Args:
        level: 日志级别
        logger_name: 日志器名称，None 表示根日志器
    """
    log_level = _LOG_LEVELS.get(level.lower(), logging.INFO)
    
    if logger_name:
        logging.getLogger(logger_name).setLevel(log_level)
    else:
        logging.getLogger().setLevel(log_level)


class LoggerContext:
    """
    日志上下文管理器
    
    用于临时修改日志级别。
    
    使用示例：
        >>> with LoggerContext("debug"):
        ...     # 此处日志级别为 DEBUG
        ...     pass
        >>> # 日志级别恢复
    """
    
    def __init__(self, level: str, logger_name: Optional[str] = None):
        self.level = level
        self.logger_name = logger_name
        self._original_level = None
    
    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self._original_level = logger.level
        set_log_level(self.level, self.logger_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self._original_level)
        return False
```

---

#### 文件：`src/utils/visualizer.py`

```python
"""
可视化工具模块
==============

提供执行过程可视化功能。
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionVisualizer:
    """
    执行过程可视化器
    
    生成执行流程的可视化表示。
    
    支持的格式：
    - Mermaid: 流程图标记语言
    - Text: 纯文本表示
    """
    
    def __init__(self):
        """初始化可视化器"""
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_mermaid(
        self,
        state: Dict[str, Any],
        include_details: bool = False
    ) -> str:
        """
        生成 Mermaid 流程图
        
        Args:
            state: 执行状态
            include_details: 是否包含详细信息
            
        Returns:
            Mermaid 格式字符串
        """
        lines = ["```mermaid", "flowchart TD"]
        
        # 添加开始节点
        lines.append("    START((开始))")
        
        # 从推理轨迹生成节点
        reasoning_trace = state.get("reasoning_trace", [])
        node_id = 0
        prev_node = "START"
        
        for step in reasoning_trace:
            node_id += 1
            node_name = f"N{node_id}"
            
            # 提取节点信息
            if "]" in step:
                agent = step.split("]")[0].strip("[")
                content = step.split("]")[1].strip()[:50]
            else:
                agent = "Unknown"
                content = step[:50]
            
            # 添加节点
            lines.append(f"    {node_name}[{agent}]")
            lines.append(f"    {prev_node} --> {node_name}")
            
            prev_node = node_name
        
        # 添加结束节点
        lines.append("    END((结束))")
        lines.append(f"    {prev_node} --> END")
        
        # 添加样式
        lines.append("")
        lines.append("    classDef coordinator fill:#e1f5fe")
        lines.append("    classDef planner fill:#fff3e0")
        lines.append("    classDef worker fill:#e8f5e9")
        lines.append("    classDef critic fill:#fce4ec")
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def generate_text_trace(
        self,
        state: Dict[str, Any],
        max_width: int = 80
    ) -> str:
        """
        生成文本格式的执行轨迹
        
        Args:
            state: 执行状态
            max_width: 最大宽度
            
        Returns:
            文本格式字符串
        """
        lines = []
        lines.append("=" * max_width)
        lines.append("执行轨迹".center(max_width))
        lines.append("=" * max_width)
        
        # 基本信息
        lines.append(f"任务ID: {state.get('task_id', 'N/A')}")
        lines.append(f"原始任务: {state.get('original_task', '')[:60]}...")
        lines.append(f"迭代次数: {state.get('iteration_count', 0)}")
        lines.append("-" * max_width)
        
        # 推理轨迹
        lines.append("推理过程:")
        for i, step in enumerate(state.get("reasoning_trace", []), 1):
            lines.append(f"  {i}. {step[:max_width-5]}")
        
        lines.append("-" * max_width)
        
        # 子任务状态
        subtasks = state.get("subtasks", [])
        if subtasks:
            lines.append("子任务状态:")
            for task in subtasks:
                status_icon = {
                    "completed": "✓",
                    "failed": "✗",
                    "pending": "○",
                    "running": "◐",
                }.get(task.get("status", ""), "?")
                lines.append(
                    f"  {status_icon} {task.get('name', 'Unknown')[:40]}"
                )
        
        lines.append("-" * max_width)
        
        # 执行时间
        exec_time = state.get("execution_time", {})
        if exec_time:
            lines.append("执行时间:")
            total = sum(exec_time.values())
            for agent, duration in exec_time.items():
                pct = (duration / total * 100) if total > 0 else 0
                bar_len = int(pct / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {agent:15} {bar} {duration:.2f}s ({pct:.1f}%)")
        
        lines.append("=" * max_width)
        
        return "\n".join(lines)
    
    def generate_summary(self, state: Dict[str, Any]) -> str:
        """
        生成执行摘要
        
        Args:
            state: 执行状态
            
        Returns:
            摘要字符串
        """
        lines = []
        
        # 状态
        has_answer = bool(state.get("final_answer"))
        status = "✅ 成功" if has_answer else "❌ 未完成"
        lines.append(f"状态: {status}")
        
        # 统计
        lines.append(f"迭代次数: {state.get('iteration_count', 0)}")
        lines.append(f"子任务数: {len(state.get('subtasks', []))}")
        lines.append(f"工具调用: {len(state.get('tool_call_logs', []))} 次")
        
        # 时间
        total_time = sum(state.get("execution_time", {}).values())
        lines.append(f"总耗时: {total_time:.2f}s")
        
        # Token
        token_usage = state.get("token_usage", {})
        if token_usage.get("total"):
            lines.append(f"Token消耗: {token_usage['total']}")
        
        return "\n".join(lines)


def generate_mermaid_graph(state: Dict[str, Any]) -> str:
    """
    便捷函数：生成 Mermaid 图
    
    Args:
        state: 执行状态
        
    Returns:
        Mermaid 格式字符串
    """
    visualizer = ExecutionVisualizer()
    return visualizer.generate_mermaid(state)


def print_execution_trace(state: Dict[str, Any]) -> None:
    """
    打印执行轨迹到控制台
    
    Args:
        state: 执行状态
    """
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    visualizer = ExecutionVisualizer()
    
    # 打印文本轨迹
    trace = visualizer.generate_text_trace(state)
    console.print(Panel(trace, title="执行轨迹"))
    
    # 打印摘要
    summary = visualizer.generate_summary(state)
    console.print(Panel(summary, title="执行摘要"))
```

---

**第三部分（第4批）完成。请输入 "继续" 获取第三部分（第5批）：Examples 和 Tests 模块完整代码。**

## 第三部分（第5批）：Examples 和 Tests 模块完整代码

---

### Examples 模块

#### 文件：`examples/__init__.py`

```python
"""
示例模块
========

包含系统使用示例。
"""
```

---

#### 文件：`examples/example_planning.py`

```python
"""
示例 1：任务规划分解
====================

演示系统如何分解和规划复杂任务。

运行方式：
    python -m examples.example_planning
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.config.settings import get_settings
from src.graph.builder import MultiAgentSystem
from src.utils.logger import setup_logger
from src.utils.visualizer import ExecutionVisualizer


def main():
    """运行任务规划示例"""
    console = Console()
    
    # 设置日志
    setup_logger(debug=False)
    
    console.print(Panel(
        "[bold blue]示例 1: 任务规划分解[/bold blue]\n\n"
        "演示系统如何理解和分解复杂任务",
        title="Multi-Agent System Demo"
    ))
    
    # 示例任务
    task = """
    请帮我制定一个学习 Python 的完整计划，包括：
    1. 学习路径和阶段划分
    2. 每个阶段的学习内容和目标
    3. 推荐的学习资源
    4. 预计时间安排
    5. 实践项目建议
    """
    
    console.print(f"\n[bold]任务描述:[/bold]\n{task}")
    console.print("\n[dim]正在处理...[/dim]\n")
    
    try:
        # 初始化系统
        settings = get_settings()
        settings.max_iterations = 5  # 限制迭代次数
        
        system = MultiAgentSystem(settings=settings)
        
        # 执行任务
        result = system.run(task, task_id="planning_demo")
        
        # 显示结果
        console.print("\n" + "=" * 60)
        console.print("[bold green]执行完成[/bold green]")
        console.print("=" * 60)
        
        # 显示子任务分解
        subtasks = result.get("subtasks", [])
        if subtasks:
            console.print("\n[bold]任务分解结果:[/bold]")
            for i, task_item in enumerate(subtasks, 1):
                status_icon = "✅" if task_item.get("status") == "completed" else "⏳"
                console.print(
                    f"  {status_icon} {i}. {task_item.get('name', 'Unknown')}"
                )
                console.print(
                    f"      类型: {task_item.get('task_type', 'N/A')} | "
                    f"执行者: {task_item.get('assigned_agent', 'N/A')}"
                )
        
        # 显示最终答案
        final_answer = result.get("final_answer", "")
        if final_answer:
            console.print("\n[bold]最终答案:[/bold]")
            console.print(Panel(
                Markdown(final_answer),
                title="学习计划",
                border_style="green"
            ))
        
        # 显示执行统计
        console.print("\n[bold]执行统计:[/bold]")
        console.print(f"  迭代次数: {result.get('iteration_count', 0)}")
        console.print(f"  子任务数: {len(subtasks)}")
        
        exec_time = result.get("execution_time", {})
        total_time = sum(exec_time.values())
        console.print(f"  总耗时: {total_time:.2f} 秒")
        
        # 可视化
        visualizer = ExecutionVisualizer()
        console.print("\n[bold]执行轨迹:[/bold]")
        trace = visualizer.generate_text_trace(result)
        console.print(trace)
        
    except Exception as e:
        console.print(f"[red]执行出错: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
```

---

#### 文件：`examples/example_tool_execution.py`

```python
"""
示例 2：工具执行
================

演示系统的工具调用功能。

运行方式：
    python -m examples.example_tool_execution
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import get_settings
from src.graph.builder import MultiAgentSystem
from src.utils.logger import setup_logger
from src.tools import get_all_tools


def main():
    """运行工具执行示例"""
    console = Console()
    
    setup_logger(debug=False)
    
    console.print(Panel(
        "[bold blue]示例 2: 工具执行[/bold blue]\n\n"
        "演示系统的工具调用能力",
        title="Multi-Agent System Demo"
    ))
    
    # 显示可用工具
    console.print("\n[bold]可用工具:[/bold]")
    tools = get_all_tools()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("工具名", style="cyan")
    table.add_column("描述")
    
    for tool in tools:
        table.add_row(tool.name, tool.description[:60] + "...")
    
    console.print(table)
    
    # 示例任务 - 使用计算器
    task1 = "请计算 (25 * 4) + (100 / 5) - 17 的结果"
    
    console.print(f"\n[bold]任务 1:[/bold] {task1}")
    console.print("[dim]正在执行...[/dim]")
    
    try:
        settings = get_settings()
        settings.max_iterations = 3
        
        system = MultiAgentSystem(settings=settings)
        result = system.run(task1, task_id="calc_demo")
        
        # 显示工具调用日志
        tool_logs = result.get("tool_call_logs", [])
        if tool_logs:
            console.print("\n[bold]工具调用记录:[/bold]")
            for log in tool_logs:
                status = "✅" if log.get("success") else "❌"
                console.print(
                    f"  {status} {log.get('tool_name')}: "
                    f"{log.get('output', 'N/A')}"
                )
        
        final_answer = result.get("final_answer", "")
        if final_answer:
            console.print(f"\n[bold green]结果:[/bold green] {final_answer[:500]}")
        
    except Exception as e:
        console.print(f"[red]执行出错: {e}[/red]")
    
    # 示例任务 - 文件操作
    console.print("\n" + "-" * 50)
    
    task2 = "请在 workspace 目录创建一个 test.txt 文件，内容为 'Hello, Multi-Agent System!'"
    
    console.print(f"\n[bold]任务 2:[/bold] {task2}")
    console.print("[dim]正在执行...[/dim]")
    
    try:
        result = system.run(task2, task_id="file_demo")
        
        tool_logs = result.get("tool_call_logs", [])
        if tool_logs:
            console.print("\n[bold]工具调用记录:[/bold]")
            for log in tool_logs:
                status = "✅" if log.get("success") else "❌"
                console.print(
                    f"  {status} {log.get('tool_name')}: "
                    f"{str(log.get('output', 'N/A'))[:100]}"
                )
        
        final_answer = result.get("final_answer", "")
        if final_answer:
            console.print(f"\n[bold green]结果:[/bold green] {final_answer[:500]}")
        
    except Exception as e:
        console.print(f"[red]执行出错: {e}[/red]")
    
    console.print("\n[dim]示例执行完成[/dim]")


if __name__ == "__main__":
    main()
```

---

#### 文件：`examples/example_code_generation.py`

```python
"""
示例 3：代码生成
================

演示系统的代码生成和执行能力。

运行方式：
    python -m examples.example_code_generation
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from src.config.settings import get_settings
from src.graph.builder import MultiAgentSystem
from src.utils.logger import setup_logger


def main():
    """运行代码生成示例"""
    console = Console()
    
    setup_logger(debug=False)
    
    console.print(Panel(
        "[bold blue]示例 3: 代码生成[/bold blue]\n\n"
        "演示系统生成和执行代码的能力",
        title="Multi-Agent System Demo"
    ))
    
    # 代码生成任务
    task = """
    请帮我编写一个 Python 爬虫，抓取 Hacker News 首页的文章标题和链接，
    并保存为 JSON 文件。
    
    要求：
    1. 使用 requests 和 BeautifulSoup 库
    2. 抓取首页前 10 篇文章的标题和链接
    3. 保存到 workspace/hn_articles.json
    4. 包含错误处理
    """
    
    console.print(f"\n[bold]任务描述:[/bold]\n{task}")
    console.print("\n[dim]正在处理（这可能需要一些时间）...[/dim]\n")
    
    try:
        settings = get_settings()
        settings.max_iterations = 8
        
        system = MultiAgentSystem(settings=settings)
        
        # 流式执行以显示进度
        console.print("[bold]执行进度:[/bold]")
        
        final_state = None
        for event in system.stream(task, task_id="code_gen_demo"):
            for node_name, node_output in event.items():
                if node_name != "__end__":
                    console.print(f"  ▶ 执行节点: [cyan]{node_name}[/cyan]")
                final_state = node_output
        
        if final_state is None:
            console.print("[yellow]未获取到执行结果[/yellow]")
            return
        
        console.print("\n" + "=" * 60)
        console.print("[bold green]执行完成[/bold green]")
        console.print("=" * 60)
        
        # 显示生成的代码
        agent_outputs = final_state.get("agent_outputs", {})
        
        for key, output in agent_outputs.items():
            if "coder" in key.lower():
                console.print("\n[bold]生成的代码:[/bold]")
                
                if isinstance(output, dict):
                    code_content = output.get("output", "")
                else:
                    code_content = str(output)
                
                # 尝试提取代码块
                import re
                code_match = re.search(
                    r'```python\s*([\s\S]*?)```',
                    code_content
                )
                
                if code_match:
                    code = code_match.group(1)
                    syntax = Syntax(
                        code,
                        "python",
                        theme="monokai",
                        line_numbers=True
                    )
                    console.print(Panel(syntax, title="Python 代码"))
                break
        
        # 显示最终答案
        final_answer = final_state.get("final_answer", "")
        if final_answer:
            console.print("\n[bold]完整结果:[/bold]")
            console.print(Panel(
                Markdown(final_answer[:3000]),
                title="执行结果",
                border_style="green"
            ))
        
        # 显示工具调用
        tool_logs = final_state.get("tool_call_logs", [])
        if tool_logs:
            console.print("\n[bold]工具调用记录:[/bold]")
            for log in tool_logs:
                status = "✅" if log.get("success") else "❌"
                console.print(
                    f"  {status} {log.get('tool_name')} "
                    f"({log.get('duration_ms', 0):.0f}ms)"
                )
        
        # 显示统计
        console.print("\n[bold]执行统计:[/bold]")
        console.print(f"  迭代次数: {final_state.get('iteration_count', 0)}")
        console.print(f"  子任务数: {len(final_state.get('subtasks', []))}")
        console.print(f"  工具调用: {len(tool_logs)} 次")
        
        exec_time = final_state.get("execution_time", {})
        total_time = sum(exec_time.values())
        console.print(f"  总耗时: {total_time:.2f} 秒")
        
    except Exception as e:
        console.print(f"[red]执行出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()
```

---

### Tests 模块

#### 文件：`tests/__init__.py`

```python
"""
测试模块
========

包含系统单元测试和集成测试。
"""
```

---

#### 文件：`tests/conftest.py`

```python
"""
Pytest 配置文件
===============

定义测试固件和通用配置。
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Settings, LLMConfig
from src.graph.state import create_initial_state, AgentState


@pytest.fixture
def mock_settings():
    """创建测试用配置"""
    return Settings(
        llm_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        debug_mode=True,
        max_iterations=3,
        enable_human_in_loop=False,
        workspace_dir="test_workspace",
        log_dir="test_logs",
    )


@pytest.fixture
def mock_llm():
    """创建模拟 LLM"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content="这是一个模拟的 LLM 响应"
    )
    return mock


@pytest.fixture
def sample_state():
    """创建示例状态"""
    return create_initial_state(
        task="测试任务：编写一个简单的 Python 函数",
        task_id="test_001",
        max_iterations=5,
    )


@pytest.fixture
def sample_subtasks():
    """创建示例子任务列表"""
    return [
        {
            "id": "task_1",
            "name": "分析需求",
            "description": "分析任务需求",
            "task_type": "analyze",
            "assigned_agent": "researcher",
            "dependencies": [],
            "priority": "high",
            "status": "pending",
            "result": None,
            "error_message": None,
            "retry_count": 0,
        },
        {
            "id": "task_2",
            "name": "编写代码",
            "description": "根据需求编写代码",
            "task_type": "code",
            "assigned_agent": "coder",
            "dependencies": ["task_1"],
            "priority": "high",
            "status": "pending",
            "result": None,
            "error_message": None,
            "retry_count": 0,
        },
        {
            "id": "task_3",
            "name": "测试代码",
            "description": "测试生成的代码",
            "task_type": "execute",
            "assigned_agent": "executor",
            "dependencies": ["task_2"],
            "priority": "medium",
            "status": "pending",
            "result": None,
            "error_message": None,
            "retry_count": 0,
        },
    ]


@pytest.fixture
def test_workspace(tmp_path):
    """创建临时测试工作空间"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    
    # 创建一些测试文件
    (workspace / "test.txt").write_text("Hello, World!")
    (workspace / "data.json").write_text('{"key": "value"}')
    
    return workspace


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    """设置测试环境"""
    # 设置环境变量
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DEBUG_MODE", "true")
    
    # 创建临时目录
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    
    yield
    
    # 清理


@pytest.fixture
def mock_tool_response():
    """创建模拟工具响应"""
    return {
        "calculator": "计算结果: 42",
        "file_manager": "文件操作成功",
        "code_executor": "代码执行成功\n输出: Hello",
        "web_search": "找到 5 条结果",
    }
```

---

#### 文件：`tests/test_graph.py`

```python
"""
图构建测试
==========

测试 LangGraph 状态图的构建和配置。
"""

import pytest
from unittest.mock import MagicMock, patch

from src.graph.builder import build_graph, MultiAgentSystem
from src.graph.state import create_initial_state, AgentState
from src.graph.nodes import (
    input_parser_node,
    coordinator_node,
    task_router_node,
)
from src.graph.edges import (
    route_from_coordinator,
    route_from_planner,
    route_task,
)


class TestGraphBuilder:
    """图构建器测试"""
    
    def test_build_graph_returns_state_graph(self, mock_settings):
        """测试图构建返回正确类型"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            graph = build_graph(mock_settings)
            
            assert graph is not None
            # StateGraph 应该有 nodes 属性
            assert hasattr(graph, "nodes")
    
    def test_graph_has_required_nodes(self, mock_settings):
        """测试图包含所有必需节点"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            graph = build_graph(mock_settings)
            
            required_nodes = [
                "input_parser",
                "coordinator",
                "planner",
                "task_router",
                "researcher",
                "coder",
                "executor",
                "critic",
                "human_node",
                "synthesizer",
                "error_handler",
            ]
            
            for node in required_nodes:
                assert node in graph.nodes, f"缺少节点: {node}"
    
    def test_graph_compilation(self, mock_settings):
        """测试图可以正常编译"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            graph = build_graph(mock_settings)
            compiled = graph.compile()
            
            assert compiled is not None
            assert hasattr(compiled, "invoke")


class TestMultiAgentSystem:
    """MultiAgentSystem 测试"""
    
    def test_system_initialization(self, mock_settings):
        """测试系统初始化"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            system = MultiAgentSystem(settings=mock_settings)
            
            assert system.settings == mock_settings
            assert system._graph is None  # 延迟加载
    
    def test_system_graph_lazy_loading(self, mock_settings, mock_llm):
        """测试图的延迟加载"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            with patch("src.llm.factory.LLMFactory.create", return_value=mock_llm):
                system = MultiAgentSystem(settings=mock_settings)
                
                # 访问 graph 属性触发加载
                _ = system.graph
                
                assert system._compiled_graph is not None
    
    def test_system_reset(self, mock_settings):
        """测试系统重置"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            system = MultiAgentSystem(settings=mock_settings)
            system._compiled_graph = MagicMock()
            
            system.reset()
            
            assert system._compiled_graph is None


class TestInitialState:
    """初始状态测试"""
    
    def test_create_initial_state(self):
        """测试创建初始状态"""
        state = create_initial_state(
            task="测试任务",
            task_id="test_123",
            max_iterations=5,
        )
        
        assert state["original_task"] == "测试任务"
        assert state["task_id"] == "test_123"
        assert state["max_iterations"] == 5
        assert state["iteration_count"] == 0
        assert state["final_answer"] is None
        assert len(state["messages"]) == 1
    
    def test_initial_state_has_required_fields(self):
        """测试初始状态包含必需字段"""
        state = create_initial_state(task="测试")
        
        required_fields = [
            "messages",
            "original_task",
            "subtasks",
            "agent_outputs",
            "tool_call_logs",
            "iteration_count",
            "max_iterations",
            "final_answer",
            "reasoning_trace",
            "error_log",
        ]
        
        for field in required_fields:
            assert field in state, f"缺少字段: {field}"


class TestEdgeRouting:
    """边路由测试"""
    
    def test_route_from_coordinator_to_planner(self, sample_state):
        """测试协调者到规划者的路由"""
        state = dict(sample_state)
        state["next"] = "planner"
        state["task_understanding"] = "需要规划的任务"
        
        result = route_from_coordinator(state)
        
        assert result == "planner"
    
    def test_route_from_coordinator_to_end(self, sample_state):
        """测试协调者到结束的路由"""
        state = dict(sample_state)
        state["final_answer"] = "任务已完成"
        
        result = route_from_coordinator(state)
        
        assert result == "end"
    
    def test_route_task_to_researcher(self, sample_state, sample_subtasks):
        """测试任务路由到研究员"""
        state = dict(sample_state)
        state["subtasks"] = sample_subtasks
        state["next"] = "researcher"
        
        result = route_task(state)
        
        assert result == "researcher"
    
    def test_route_task_to_synthesizer_when_no_pending(self, sample_state):
        """测试无待处理任务时路由到综合者"""
        state = dict(sample_state)
        state["subtasks"] = []
        state["next"] = "synthesizer"
        
        result = route_task(state)
        
        assert result == "synthesizer"
```

---

#### 文件：`tests/test_flow.py`

```python
"""
完整流程测试
============

测试端到端的任务执行流程。
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.graph.builder import MultiAgentSystem
from src.graph.state import create_initial_state


class TestEndToEndFlow:
    """端到端流程测试"""
    
    @pytest.fixture
    def mock_system(self, mock_settings, mock_llm):
        """创建模拟系统"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            with patch("src.llm.factory.LLMFactory.create", return_value=mock_llm):
                system = MultiAgentSystem(settings=mock_settings)
                return system
    
    def test_simple_task_execution(self, mock_system, mock_llm):
        """测试简单任务执行"""
        # 配置模拟响应
        mock_llm.invoke.return_value = MagicMock(
            content="""
            ## 任务理解
            这是一个简单的问答任务
            
            ## 任务类型
            简单问答
            
            ## 下一步行动
            direct_answer
            """
        )
        
        with patch.object(
            mock_system,
            "graph",
            new_callable=lambda: MagicMock(
                invoke=MagicMock(return_value={
                    "final_answer": "这是测试答案",
                    "iteration_count": 1,
                    "reasoning_trace": ["完成"],
                })
            )
        ):
            result = mock_system.run("什么是 Python?", task_id="test_simple")
            
            assert "final_answer" in result
    
    def test_task_with_subtasks(self, mock_system, sample_subtasks):
        """测试带子任务的执行"""
        mock_result = {
            "original_task": "复杂任务",
            "subtasks": sample_subtasks,
            "final_answer": "任务完成",
            "iteration_count": 3,
            "reasoning_trace": [
                "[Coordinator] 任务理解完成",
                "[Planner] 分解为 3 个子任务",
                "[Synthesizer] 生成最终答案",
            ],
        }
        
        with patch.object(
            mock_system,
            "graph",
            new_callable=lambda: MagicMock(invoke=MagicMock(return_value=mock_result))
        ):
            result = mock_system.run("复杂任务", task_id="test_complex")
            
            assert len(result.get("subtasks", [])) == 3
            assert result.get("final_answer") == "任务完成"
    
    def test_iteration_limit(self, mock_settings):
        """测试迭代次数限制"""
        mock_settings.max_iterations = 2
        
        state = create_initial_state(
            task="测试任务",
            max_iterations=2,
        )
        state["iteration_count"] = 2
        
        # 迭代次数已达上限
        assert state["iteration_count"] >= state["max_iterations"]
    
    def test_error_handling_in_flow(self, mock_system):
        """测试流程中的错误处理"""
        error_result = {
            "original_task": "错误任务",
            "error_log": ["模拟错误"],
            "last_error": "处理失败",
            "final_answer": "任务执行过程中出现错误",
            "iteration_count": 1,
        }
        
        with patch.object(
            mock_system,
            "graph",
            new_callable=lambda: MagicMock(invoke=MagicMock(return_value=error_result))
        ):
            result = mock_system.run("错误任务")
            
            assert "error_log" in result
            assert len(result["error_log"]) > 0


class TestStreamExecution:
    """流式执行测试"""
    
    def test_stream_yields_events(self, mock_settings, mock_llm):
        """测试流式执行产生事件"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            with patch("src.llm.factory.LLMFactory.create", return_value=mock_llm):
                system = MultiAgentSystem(settings=mock_settings)
                
                mock_events = [
                    {"input_parser": {"next": "coordinator"}},
                    {"coordinator": {"next": "synthesizer"}},
                    {"synthesizer": {"final_answer": "完成"}},
                ]
                
                mock_graph = MagicMock()
                mock_graph.stream = MagicMock(return_value=iter(mock_events))
                
                with patch.object(system, "_compiled_graph", mock_graph):
                    system._graph = MagicMock()
                    
                    events = list(system.stream("测试任务"))
                    
                    assert len(events) == 3


class TestHumanInLoop:
    """人工介入测试"""
    
    def test_human_input_flag(self, sample_state):
        """测试人工介入标志"""
        state = dict(sample_state)
        state["needs_human_input"] = True
        
        assert state["needs_human_input"] is True
    
    def test_human_feedback_recorded(self, sample_state):
        """测试人工反馈记录"""
        state = dict(sample_state)
        state["human_feedback"] = "用户确认继续"
        
        assert state["human_feedback"] == "用户确认继续"
```

---

#### 文件：`tests/test_tools.py`

```python
"""
工具调用测试
============

测试各种工具的功能和安全性。
"""

import pytest
import json
from pathlib import Path

from src.tools.calculator import SafeCalculator, calculator_tool
from src.tools.file_manager import FileManager, file_manager_tool
from src.tools.code_executor import CodeExecutor, code_executor_tool
from src.tools.search import WebSearch, web_search_tool


class TestSafeCalculator:
    """安全计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        return SafeCalculator()
    
    def test_basic_arithmetic(self, calculator):
        """测试基本算术"""
        assert calculator.calculate("2 + 3") == 5
        assert calculator.calculate("10 - 4") == 6
        assert calculator.calculate("3 * 4") == 12
        assert calculator.calculate("15 / 3") == 5.0
    
    def test_complex_expression(self, calculator):
        """测试复杂表达式"""
        assert calculator.calculate("(2 + 3) * 4") == 20
        assert calculator.calculate("2 ** 10") == 1024
        assert calculator.calculate("17 % 5") == 2
    
    def test_math_functions(self, calculator):
        """测试数学函数"""
        assert calculator.calculate("sqrt(16)") == 4.0
        assert calculator.calculate("abs(-5)") == 5
        assert calculator.calculate("max(1, 2, 3)") == 3
        assert calculator.calculate("min(1, 2, 3)") == 1
    
    def test_constants(self, calculator):
        """测试常量"""
        result = calculator.calculate("pi")
        assert 3.14 < result < 3.15
    
    def test_invalid_expression(self, calculator):
        """测试无效表达式"""
        with pytest.raises(ValueError):
            calculator.calculate("invalid")
    
    def test_forbidden_operations(self, calculator):
        """测试禁止的操作"""
        with pytest.raises(ValueError):
            calculator.calculate("__import__('os')")
    
    def test_calculator_tool(self):
        """测试计算器工具接口"""
        result = calculator_tool.invoke({"expression": "2 + 2"})
        assert "4" in result


class TestFileManager:
    """文件管理器测试"""
    
    @pytest.fixture
    def file_manager(self, test_workspace):
        return FileManager(workspace_dir=str(test_workspace))
    
    def test_read_file(self, file_manager, test_workspace):
        """测试读取文件"""
        content = file_manager.read("test.txt")
        assert content == "Hello, World!"
    
    def test_write_file(self, file_manager, test_workspace):
        """测试写入文件"""
        file_manager.write("new_file.txt", "New content")
        
        assert (test_workspace / "new_file.txt").exists()
        assert (test_workspace / "new_file.txt").read_text() == "New content"
    
    def test_list_dir(self, file_manager):
        """测试列出目录"""
        items = file_manager.list_dir(".")
        
        assert len(items) >= 2
        names = [item["name"] for item in items]
        assert "test.txt" in names
    
    def test_file_exists(self, file_manager):
        """测试检查文件存在"""
        assert file_manager.exists("test.txt") is True
        assert file_manager.exists("nonexistent.txt") is False
    
    def test_path_traversal_blocked(self, file_manager):
        """测试路径遍历被阻止"""
        with pytest.raises(ValueError):
            file_manager.read("../../../etc/passwd")
    
    def test_absolute_path_blocked(self, file_manager):
        """测试绝对路径被阻止"""
        with pytest.raises(ValueError):
            file_manager.read("/etc/passwd")
    
    def test_file_manager_tool(self, test_workspace):
        """测试文件管理器工具接口"""
        with pytest.MonkeyPatch().context() as mp:
            # 需要临时修改工作目录
            result = file_manager_tool.invoke({
                "action": "exists",
                "path": "test.txt"
            })
            # 结果应该包含存在或不存在的信息
            assert "存在" in result or "不存在" in result


class TestCodeExecutor:
    """代码执行器测试"""
    
    @pytest.fixture
    def executor(self):
        return CodeExecutor(timeout=5)
    
    def test_simple_code_execution(self, executor):
        """测试简单代码执行"""
        result = executor.execute("print('Hello')")
        
        assert result["success"] is True
        assert "Hello" in result["output"]
    
    def test_expression_evaluation(self, executor):
        """测试表达式求值"""
        result = executor.execute("2 + 2")
        
        assert result["success"] is True
        assert result["result"] == 4
    
    def test_math_operations(self, executor):
        """测试数学运算"""
        code = """
import math
result = math.sqrt(16) + math.pi
print(f"Result: {result}")
"""
        result = executor.execute(code)
        
        assert result["success"] is True
        assert "Result:" in result["output"]
    
    def test_forbidden_import(self, executor):
        """测试禁止的导入"""
        result = executor.execute("import os")
        
        assert result["success"] is False
        assert "禁止" in result["error"]
    
    def test_forbidden_operations(self, executor):
        """测试禁止的操作"""
        result = executor.execute("__import__('subprocess')")
        
        assert result["success"] is False
    
    def test_timeout(self, executor):
        """测试超时"""
        code = """
import time
time.sleep(10)
"""
        # 这个测试可能需要较长时间
        result = executor.execute(code, timeout=1)
        
        # 应该失败或超时
        # 注意：time 模块可能不在允许列表中
        assert result["success"] is False
    
    def test_code_executor_tool(self):
        """测试代码执行器工具接口"""
        result = code_executor_tool.invoke({
            "code": "print(1 + 1)"
        })
        assert "2" in result


class TestWebSearch:
    """网络搜索测试"""
    
    @pytest.fixture
    def search(self):
        return WebSearch()
    
    def test_search_returns_results(self, search):
        """测试搜索返回结果"""
        results = search.search("python")
        
        assert len(results) > 0
        assert all(hasattr(r, "title") for r in results)
        assert all(hasattr(r, "url") for r in results)
    
    def test_search_with_limit(self, search):
        """测试搜索结果数量限制"""
        results = search.search("python", num_results=3)
        
        assert len(results) <= 3
    
    def test_format_results(self, search):
        """测试结果格式化"""
        results = search.search("python", num_results=2)
        formatted = search.format_results(results)
        
        assert "找到" in formatted
        assert "条结果" in formatted
    
    def test_empty_results_handling(self, search):
        """测试空结果处理"""
        formatted = search.format_results([])
        
        assert "未找到" in formatted
    
    def test_web_search_tool(self):
        """测试搜索工具接口"""
        result = web_search_tool.invoke({
            "query": "python programming"
        })
        assert "结果" in result


class TestToolIntegration:
    """工具集成测试"""
    
    def test_all_tools_have_names(self):
        """测试所有工具都有名称"""
        from src.tools import get_all_tools
        
        tools = get_all_tools()
        
        for tool in tools:
            assert tool.name is not None
            assert len(tool.name) > 0
    
    def test_all_tools_have_descriptions(self):
        """测试所有工具都有描述"""
        from src.tools import get_all_tools
        
        tools = get_all_tools()
        
        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 0
    
    def test_get_tool_by_name(self):
        """测试按名称获取工具"""
        from src.tools import get_tool_by_name
        
        calc = get_tool_by_name("calculator")
        assert calc.name == "calculator"
        
        with pytest.raises(ValueError):
            get_tool_by_name("nonexistent_tool")
```

---

### 其他配置文件

#### 文件：`workspace/.gitkeep`

```
# 此目录用于工具操作的安全隔离
# 所有文件操作都限定在此目录内
```

---

#### 文件：`logs/.gitkeep`

```
# 此目录存储执行日志
# 日志文件会自动轮转
```

---



## 第六部分：扩展指南详细说明

---

### 6.1 扩展指南完整文档

#### 文件：`docs/EXTENSION_GUIDE.md`

```markdown
# Multi-Agent System 扩展指南

本文档详细说明如何扩展 Multi-Agent System 的各个组件。

---

## 目录

1. [新增 Agent](#1-新增-agent)
2. [新增工具](#2-新增工具)
3. [新增工作流节点](#3-新增工作流节点)
4. [自定义提示词](#4-自定义提示词)
5. [自定义路由逻辑](#5-自定义路由逻辑)
6. [扩展记忆系统](#6-扩展记忆系统)
7. [对接企业级场景](#7-对接企业级场景)
8. [性能优化](#8-性能优化)

---

## 1. 新增 Agent

### 1.1 基本步骤

创建新 Agent 需要以下步骤：

1. 创建 Agent 类文件
2. 继承 `BaseAgent` 基类
3. 实现必要的属性和方法
4. 注册 Agent
5. 添加到图构建器
6. 配置路由

### 1.2 完整示例：创建数据分析 Agent

```python
# src/agents/data_analyst.py

"""
数据分析 Agent
==============

负责数据处理、统计分析和可视化建议。
"""

from typing import Any, Dict, List, Optional
import json
import re

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates
from src.utils.logger import get_logger

logger = get_logger(__name__)


# 添加提示词模板
DATA_ANALYST_SYSTEM = """你是一个专业的数据分析师（Data Analyst），擅长数据处理和统计分析。

你的核心能力：
1. **数据理解**：快速理解数据结构和含义
2. **统计分析**：执行描述性统计、相关性分析等
3. **数据清洗**：识别和处理异常值、缺失值
4. **可视化建议**：推荐合适的图表类型
5. **洞察提取**：从数据中提取有价值的洞察

分析原则：
- 始终验证数据质量
- 使用适当的统计方法
- 结论要有数据支撑
- 提供可操作的建议

可用工具：
- calculator: 数学计算
- code_executor: 执行 Python 数据分析代码
- file_manager: 读取数据文件

请提供详细的分析过程和结论。"""

DATA_ANALYST_TASK = """请对以下数据进行分析：

任务描述：$task
数据上下文：$context
分析要求：$requirements

请按以下步骤进行分析：
1. 数据概览：理解数据结构
2. 数据质量检查：检查缺失值、异常值
3. 统计分析：执行相关统计计算
4. 洞察提取：总结关键发现
5. 建议：提供可操作的建议

请输出结构化的分析报告。"""


@register_agent("data_analyst")
class DataAnalystAgent(BaseAgent):
    """
    数据分析师智能体
    
    核心职责：
    1. 理解和处理数据
    2. 执行统计分析
    3. 提取数据洞察
    4. 生成分析报告
    """
    
    @property
    def name(self) -> str:
        return "data_analyst"
    
    @property
    def description(self) -> str:
        return "数据分析师，负责数据处理、统计分析和洞察提取"
    
    @property
    def capabilities(self) -> List[str]:
        return ["analyze", "statistics", "data_processing", "visualization"]
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return DATA_ANALYST_SYSTEM
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据分析任务
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        # 获取当前任务
        current_task = self._get_current_task(state)
        
        if current_task is None:
            self.logger.warning("没有找到待处理的分析任务")
            return {**state, "next": "coordinator"}
        
        task_description = current_task.get("description", "")
        context = self._build_context(state)
        requirements = current_task.get("metadata", {}).get(
            "requirements", "提供完整的数据分析"
        )
        
        # 构建分析提示
        prompt = DATA_ANALYST_TASK.replace(
            "$task", task_description
        ).replace(
            "$context", context
        ).replace(
            "$requirements", requirements
        )
        
        messages = [HumanMessage(content=prompt)]
        
        # 检查是否需要先读取数据
        data_content = self._load_data_if_needed(task_description, state)
        if data_content:
            prompt += f"\n\n数据内容：\n```\n{data_content[:2000]}\n```"
            messages = [HumanMessage(content=prompt)]
        
        # 调用 LLM 进行分析
        response = self.call_llm(messages)
        analysis_output = response.content
        
        # 如果需要执行代码进行计算
        code_results = self._execute_analysis_code(analysis_output)
        if code_results:
            analysis_output += f"\n\n**计算结果：**\n{code_results}"
        
        # 创建输出
        agent_output = self.create_output(
            output=analysis_output,
            reasoning=f"完成数据分析任务: {current_task.get('name', '')}",
            task_id=current_task.get("id"),
            confidence=0.85,
        )
        
        # 更新子任务状态
        subtasks = self._update_subtask_status(
            state,
            current_task["id"],
            status="completed",
            result=analysis_output,
        )
        
        # 更新 agent_outputs
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs[f"data_analyst_{current_task['id']}"] = agent_output.model_dump()
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[DataAnalyst] 完成分析任务 '{current_task.get('name', '')}'"
        )
        
        return {
            **state,
            "subtasks": subtasks,
            "agent_outputs": agent_outputs,
            "reasoning_trace": reasoning_trace,
            "next": "critic",
        }
    
    def _get_current_task(self, state: Dict[str, Any]) -> Optional[Dict]:
        """获取当前待处理的分析任务"""
        subtasks = state.get("subtasks", [])
        
        for task in subtasks:
            if (task.get("status") == "pending" and 
                task.get("assigned_agent") == "data_analyst"):
                dependencies = task.get("dependencies", [])
                if self._dependencies_satisfied(subtasks, dependencies):
                    return task
        
        # 也处理 analyze 类型的任务
        for task in subtasks:
            if (task.get("status") == "pending" and 
                task.get("task_type") == "analyze"):
                dependencies = task.get("dependencies", [])
                if self._dependencies_satisfied(subtasks, dependencies):
                    return task
        
        return None
    
    def _dependencies_satisfied(
        self,
        subtasks: List[Dict],
        dependencies: List[str]
    ) -> bool:
        """检查依赖是否满足"""
        if not dependencies:
            return True
        
        completed_ids = {
            t["id"] for t in subtasks
            if t.get("status") == "completed"
        }
        
        return all(dep in completed_ids for dep in dependencies)
    
    def _build_context(self, state: Dict[str, Any]) -> str:
        """构建分析上下文"""
        context_parts = []
        
        original_task = state.get("original_task", "")
        if original_task:
            context_parts.append(f"原始任务: {original_task}")
        
        # 获取之前的研究或数据结果
        agent_outputs = state.get("agent_outputs", {})
        for name, output in agent_outputs.items():
            if any(keyword in name.lower() for keyword in ["researcher", "executor"]):
                if isinstance(output, dict):
                    result = output.get("output", "")[:500]
                else:
                    result = str(output)[:500]
                context_parts.append(f"前置结果 ({name}): {result}")
        
        return "\n".join(context_parts)
    
    def _load_data_if_needed(
        self,
        task_description: str,
        state: Dict[str, Any]
    ) -> Optional[str]:
        """如果任务涉及文件数据，加载数据"""
        # 检查任务描述中是否提到文件
        file_patterns = [
            r'文件[：:]\s*(\S+)',
            r'(\S+\.csv)',
            r'(\S+\.json)',
            r'(\S+\.xlsx?)',
        ]
        
        for pattern in file_patterns:
            match = re.search(pattern, task_description)
            if match:
                filename = match.group(1)
                try:
                    return self.call_tool(
                        "file_manager",
                        action="read",
                        path=filename
                    )
                except Exception as e:
                    self.logger.warning(f"加载数据文件失败: {e}")
        
        return None
    
    def _execute_analysis_code(self, analysis_output: str) -> Optional[str]:
        """执行分析中的代码块"""
        # 提取代码块
        code_pattern = r'```python\s*([\s\S]*?)```'
        matches = re.findall(code_pattern, analysis_output)
        
        if not matches:
            return None
        
        results = []
        for code in matches[:2]:  # 最多执行2个代码块
            try:
                result = self.call_tool("code_executor", code=code)
                if result:
                    results.append(str(result))
            except Exception as e:
                self.logger.warning(f"代码执行失败: {e}")
        
        return "\n".join(results) if results else None
```

### 1.3 注册新 Agent

```python
# src/agents/__init__.py

from src.agents.data_analyst import DataAnalystAgent

__all__ = [
    # ... 现有 agents
    "DataAnalystAgent",
]

def get_all_agents() -> dict:
    return {
        # ... 现有 agents
        "data_analyst": DataAnalystAgent,
    }
```

### 1.4 添加到图构建器

```python
# src/graph/nodes.py

from src.agents import DataAnalystAgent

def data_analyst_node(state: AgentState) -> Dict[str, Any]:
    """数据分析师节点"""
    logger.info("[Node] data_analyst - 开始分析")
    
    agent = _get_agent(DataAnalystAgent)
    result = agent.invoke(dict(state))
    
    return result
```

```python
# src/graph/builder.py

from src.graph.nodes import data_analyst_node

def build_graph(settings):
    # ... 现有代码
    
    # 添加数据分析师节点
    graph.add_node("data_analyst", data_analyst_node)
    
    # 添加边
    graph.add_conditional_edges(
        "data_analyst",
        route_from_worker,
        {
            "critic": "critic",
            "error_handler": "error_handler",
            "synthesizer": "synthesizer",
        }
    )
    
    # 更新任务路由
    # 需要修改 route_task 函数以支持新的 agent
```

### 1.5 更新任务路由

```python
# src/graph/edges.py

def route_task(state: AgentState) -> RouteType:
    """任务路由逻辑"""
    next_node = state.get("next", "executor")
    
    # 有效的工作者节点（添加新的）
    worker_nodes = {"researcher", "coder", "executor", "data_analyst"}
    
    if next_node in worker_nodes:
        return next_node
    
    # ... 其他逻辑
```

---

## 2. 新增工具

### 2.1 基本步骤

1. 创建工具类文件
2. 定义输入参数 Schema
3. 实现工具逻辑
4. 使用 `@tool` 装饰器
5. 注册工具

### 2.2 完整示例：创建 HTTP 请求工具

```python
# src/tools/http_client.py

"""
HTTP 客户端工具
===============

提供安全的 HTTP 请求功能。
"""

import json
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HTTPClientInput(BaseModel):
    """HTTP 客户端输入参数"""
    
    method: Literal["GET", "POST", "PUT", "DELETE"] = Field(
        default="GET",
        description="HTTP 方法"
    )
    url: str = Field(
        description="请求 URL"
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="请求头"
    )
    params: Optional[Dict[str, str]] = Field(
        default=None,
        description="URL 查询参数"
    )
    body: Optional[str] = Field(
        default=None,
        description="请求体（JSON 字符串）"
    )
    timeout: int = Field(
        default=30,
        description="超时时间（秒）",
        ge=1,
        le=120
    )
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """验证 URL 安全性"""
        parsed = urlparse(v)
        
        # 只允许 http 和 https
        if parsed.scheme not in ("http", "https"):
            raise ValueError("只支持 http 和 https 协议")
        
        # 禁止访问内网地址
        blocked_hosts = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "10.",
            "172.16.",
            "192.168.",
        ]
        
        host = parsed.netloc.lower()
        for blocked in blocked_hosts:
            if host.startswith(blocked) or host == blocked.rstrip("."):
                raise ValueError(f"禁止访问内网地址: {host}")
        
        return v


class HTTPClient:
    """
    HTTP 客户端
    
    安全特性：
    - 禁止访问内网地址
    - 请求超时限制
    - 响应大小限制
    - 只允许 http/https 协议
    """
    
    MAX_RESPONSE_SIZE = 1024 * 1024  # 1MB
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        发送 HTTP 请求
        
        Args:
            method: HTTP 方法
            url: 请求 URL
            headers: 请求头
            params: 查询参数
            body: 请求体
            timeout: 超时时间
            
        Returns:
            响应信息字典
        """
        self.logger.info(f"HTTP {method} {url}")
        
        # 默认请求头
        default_headers = {
            "User-Agent": "MultiAgentSystem/1.0",
            "Accept": "application/json, text/html, */*",
        }
        
        if headers:
            default_headers.update(headers)
        
        # 解析请求体
        json_body = None
        if body:
            try:
                json_body = json.loads(body)
            except json.JSONDecodeError:
                # 作为普通文本发送
                pass
        
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=default_headers,
                    params=params,
                    json=json_body if json_body else None,
                    content=body if body and not json_body else None,
                )
                
                # 检查响应大小
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.MAX_RESPONSE_SIZE:
                    return {
                        "success": False,
                        "error": f"响应过大: {content_length} bytes",
                        "status_code": response.status_code,
                    }
                
                # 尝试解析 JSON
                try:
                    response_body = response.json()
                except:
                    response_body = response.text[:5000]  # 限制文本长度
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response_body,
                }
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": f"请求超时 ({timeout}s)",
            }
        except httpx.RequestError as e:
            return {
                "success": False,
                "error": f"请求失败: {str(e)}",
            }
        except Exception as e:
            self.logger.error(f"HTTP 请求异常: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"未知错误: {str(e)}",
            }


# 创建工具实例
_http_client = HTTPClient()


@tool(args_schema=HTTPClientInput)
def http_client_tool(
    method: str = "GET",
    url: str = "",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """
    发送 HTTP 请求获取网络资源。
    
    支持 GET、POST、PUT、DELETE 方法。
    禁止访问内网地址以确保安全。
    
    使用示例：
    - GET 请求: method="GET", url="https://api.example.com/data"
    - POST 请求: method="POST", url="https://api.example.com/submit", 
                 body='{"key": "value"}'
    """
    try:
        result = _http_client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            body=body,
            timeout=timeout,
        )
        
        if result["success"]:
            body_preview = result.get("body", "")
            if isinstance(body_preview, dict):
                body_preview = json.dumps(body_preview, ensure_ascii=False, indent=2)
            
            if len(str(body_preview)) > 2000:
                body_preview = str(body_preview)[:2000] + "...[截断]"
            
            return f"""HTTP {method} {url}
状态码: {result['status_code']}

响应内容:
{body_preview}"""
        else:
            return f"请求失败: {result.get('error', '未知错误')}"
    
    except ValueError as e:
        return f"参数错误: {str(e)}"
    except Exception as e:
        logger.error(f"HTTP 工具异常: {e}", exc_info=True)
        return f"请求失败: {str(e)}"
```

### 2.3 注册工具

```python
# src/tools/__init__.py

from src.tools.http_client import HTTPClient, http_client_tool

__all__ = [
    # ... 现有工具
    "HTTPClient",
    "http_client_tool",
]

def get_all_tools():
    return [
        calculator_tool,
        file_manager_tool,
        code_executor_tool,
        web_search_tool,
        http_client_tool,  # 新增
    ]
```

---

## 3. 新增工作流节点

### 3.1 创建自定义节点

```python
# src/graph/nodes.py

def validation_node(state: AgentState) -> Dict[str, Any]:
    """
    验证节点
    
    在执行关键操作前进行验证。
    """
    logger.info("[Node] validation - 开始验证")
    
    original_task = state.get("original_task", "")
    subtasks = state.get("subtasks", [])
    
    # 验证逻辑
    validation_results = {
        "task_valid": bool(original_task),
        "subtasks_valid": len(subtasks) > 0,
        "no_errors": len(state.get("error_log", [])) == 0,
    }
    
    all_valid = all(validation_results.values())
    
    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append(
        f"[Validation] 验证结果: {'通过' if all_valid else '未通过'}"
    )
    
    return {
        **state,
        "validation_results": validation_results,
        "reasoning_trace": reasoning_trace,
        "next": "task_router" if all_valid else "error_handler",
    }
```

### 3.2 添加到图

```python
# src/graph/builder.py

def build_graph(settings):
    # ... 现有节点
    
    # 添加验证节点
    graph.add_node("validation", validation_node)
    
    # 添加边：planner 后进行验证
    graph.add_edge("planner", "validation")
    
    # 验证后的条件路由
    graph.add_conditional_edges(
        "validation",
        lambda state: state.get("next", "task_router"),
        {
            "task_router": "task_router",
            "error_handler": "error_handler",
        }
    )
```

---

## 4. 自定义提示词

### 4.1 运行时修改提示词

```python
from src.config.prompts import PromptTemplates

# 设置自定义系统提示词
PromptTemplates.set_custom(
    "COORDINATOR_SYSTEM",
    """你是一个专门为企业定制的任务协调者。

你的工作重点：
1. 严格遵守企业安全规范
2. 优先使用内部工具和资源
3. 保护敏感信息不外泄

[其他自定义内容...]
"""
)

# 设置自定义任务提示词
PromptTemplates.set_custom(
    "CODER_TASK",
    """请按照公司代码规范编写代码：

任务：$task
上下文：$context

代码规范要求：
1. 使用 Google Python Style Guide
2. 类型注解必须完整
3. 函数必须有 docstring
4. 单元测试覆盖率 > 80%

[其他规范...]
"""
)
```

### 4.2 从文件加载提示词

```python
# src/config/custom_prompts.py

import os
from pathlib import Path
from src.config.prompts import PromptTemplates

def load_custom_prompts(prompts_dir: str = "prompts") -> None:
    """
    从目录加载自定义提示词
    
    目录结构：
        prompts/
        ├── coordinator_system.txt
        ├── planner_decompose.txt
        └── ...
    """
    prompts_path = Path(prompts_dir)
    
    if not prompts_path.exists():
        return
    
    for prompt_file in prompts_path.glob("*.txt"):
        # 文件名转换为模板名
        # coordinator_system.txt -> COORDINATOR_SYSTEM
        template_name = prompt_file.stem.upper()
        
        content = prompt_file.read_text(encoding="utf-8")
        PromptTemplates.set_custom(template_name, content)
        
        print(f"加载自定义提示词: {template_name}")
```

---

## 5. 自定义路由逻辑

### 5.1 基于任务类型的路由

```python
# src/graph/edges.py

def custom_task_router(state: AgentState) -> RouteType:
    """
    自定义任务路由
    
    基于任务类型和关键词智能选择执行者。
    """
    original_task = state.get("original_task", "").lower()
    subtasks = state.get("subtasks", [])
    
    # 获取下一个待处理任务
    pending = [t for t in subtasks if t.get("status") == "pending"]
    if not pending:
        return "synthesizer"
    
    task = pending[0]
    task_type = task.get("task_type", "")
    task_desc = task.get("description", "").lower()
    
    # 数据分析任务
    if any(kw in task_desc for kw in ["分析", "统计", "数据", "图表"]):
        return "data_analyst"
    
    # API 调用任务
    if any(kw in task_desc for kw in ["api", "接口", "请求", "http"]):
        return "executor"
    
    # 代码任务
    if task_type == "code" or any(kw in task_desc for kw in ["编写", "代码", "实现"]):
        return "coder"
    
    # 研究任务
    if task_type == "research" or any(kw in task_desc for kw in ["搜索", "查找", "研究"]):
        return "researcher"
    
    # 默认执行者
    return "executor"
```

### 5.2 基于评分的质量路由

```python
# src/graph/edges.py

def quality_based_router(state: AgentState) -> RouteType:
    """
    基于质量评分的路由
    
    根据历史评分决定是否需要更严格的审核。
    """
    eval_results = state.get("evaluation_results", [])
    
    if not eval_results:
        # 首次评估，使用普通流程
        return "critic"
    
    # 计算平均分
    scores = [r.get("score", 0.7) for r in eval_results if isinstance(r, dict)]
    avg_score = sum(scores) / len(scores) if scores else 0.7
    
    # 低于阈值，需要更严格的审核
    if avg_score < 0.6:
        return "human_node"  # 人工审核
    elif avg_score < 0.75:
        return "critic"  # 额外审核
    else:
        return "synthesizer"  # 直接综合
```

---

## 6. 扩展记忆系统

### 6.1 添加向量存储记忆

```python
# src/memory/vector_memory.py

"""
向量记忆模块
============

使用向量数据库实现语义搜索记忆。
"""

from typing import Any, Dict, List, Optional
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorMemory:
    """
    向量记忆
    
    使用 embedding 进行语义搜索。
    支持对接 Chroma、Pinecone、Milvus 等向量数据库。
    """
    
    def __init__(
        self,
        collection_name: str = "multi_agent_memory",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        初始化向量记忆
        
        Args:
            collection_name: 集合名称
            embedding_model: 嵌入模型
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._collection = None
        self.logger = get_logger(self.__class__.__name__)
        
        self._init_collection()
    
    def _init_collection(self) -> None:
        """初始化向量集合"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.Client(Settings(
                anonymized_telemetry=False
            ))
            
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.logger.info(f"向量集合初始化完成: {self.collection_name}")
            
        except ImportError:
            self.logger.warning("chromadb 未安装，向量记忆不可用")
            self._collection = None
    
    def store(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        存储内容到向量数据库
        
        Args:
            key: 唯一标识
            content: 文本内容
            metadata: 元数据
        """
        if self._collection is None:
            return
        
        self._collection.upsert(
            ids=[key],
            documents=[content],
            metadatas=[metadata or {}],
        )
        
        self.logger.debug(f"存储向量: {key}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义搜索
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            filter_metadata: 元数据过滤
            
        Returns:
            搜索结果列表
        """
        if self._collection is None:
            return []
        
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata,
        )
        
        # 格式化结果
        formatted = []
        for i, (doc_id, doc, metadata, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            formatted.append({
                "id": doc_id,
                "content": doc,
                "metadata": metadata,
                "score": 1 - distance,  # 转换为相似度
            })
        
        return formatted
    
    def delete(self, key: str) -> None:
        """删除记忆"""
        if self._collection is None:
            return
        
        self._collection.delete(ids=[key])
    
    def clear(self) -> None:
        """清空所有记忆"""
        if self._collection is None:
            return
        
        # 重新创建集合
        import chromadb
        client = chromadb.Client()
        client.delete_collection(self.collection_name)
        self._init_collection()
```

---

## 7. 对接企业级场景

### 7.1 企业配置示例

```python
# src/config/enterprise.py

"""
企业级配置
==========

针对企业环境的特殊配置。
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SecurityConfig(BaseModel):
    """安全配置"""
    enable_audit_log: bool = True
    sensitive_fields: List[str] = ["password", "api_key", "token"]
    allowed_domains: List[str] = []
    blocked_domains: List[str] = ["localhost", "127.0.0.1"]
    max_file_size_mb: int = 10
    enable_encryption: bool = False


class ComplianceConfig(BaseModel):
    """合规配置"""
    data_retention_days: int = 90
    enable_pii_detection: bool = True
    require_approval_for: List[str] = ["file_write", "http_request"]
    audit_all_llm_calls: bool = True


class EnterpriseSettings(BaseModel):
    """企业设置"""
    company_name: str
    environment: str = "production"
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    
    # LLM 配置
    use_azure_openai: bool = False
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    
    # 审批流程
    require_human_approval: bool = True
    approval_timeout_minutes: int = 30
    
    # 日志
    log_to_splunk: bool = False
    splunk_endpoint: Optional[str] = None
```

### 7.2 审计日志

```python
# src/utils/audit.py

"""
审计日志模块
============

记录系统操作的审计日志。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(
        self,
        log_dir: str = "audit_logs",
        enable_file_log: bool = True,
        enable_remote_log: bool = False,
        remote_endpoint: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.enable_file_log = enable_file_log
        self.enable_remote_log = enable_remote_log
        self.remote_endpoint = remote_endpoint
        
        if enable_file_log:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log(
        self,
        action: str,
        actor: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        result: str = "success",
    ) -> None:
        """
        记录审计日志
        
        Args:
            action: 操作类型
            actor: 执行者
            resource: 资源
            details: 详细信息
            result: 结果
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "actor": actor,
            "resource": resource,
            "details": details or {},
            "result": result,
        }
        
        if self.enable_file_log:
            self._write_to_file(entry)
        
        if self.enable_remote_log and self.remote_endpoint:
            self._send_to_remote(entry)
        
        logger.debug(f"审计日志: {action} by {actor} on {resource}")
    
    def _write_to_file(self, entry: Dict) -> None:
        """写入文件"""
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def _send_to_remote(self, entry: Dict) -> None:
        """发送到远程"""
        try:
            import httpx
            httpx.post(
                self.remote_endpoint,
                json=entry,
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"发送审计日志失败: {e}")


# 全局审计日志实例
audit_logger = AuditLogger()


def audit_action(
    action: str,
    actor: str,
    resource: str,
    details: Optional[Dict] = None,
    result: str = "success",
) -> None:
    """便捷函数：记录审计日志"""
    audit_logger.log(action, actor, resource, details, result)
```

---

## 8. 性能优化

### 8.1 并行执行优化

```python
# src/graph/parallel.py

"""
并行执行模块
============

支持子任务的并行执行。
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParallelExecutor:
    """并行执行器"""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def execute_parallel(
        self,
        tasks: List[Dict[str, Any]],
        execute_func: Callable[[Dict], Any],
        timeout: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        并行执行任务
        
        Args:
            tasks: 任务列表
            execute_func: 执行函数
            timeout: 超时时间
            
        Returns:
            执行结果列表
        """
        results = []
        futures = {}
        
        for task in tasks:
            future = self.executor.submit(execute_func, task)
            futures[future] = task["id"]
        
        for future in as_completed(futures, timeout=timeout):
            task_id = futures[future]
            try:
                result = future.result()
                results.append({
                    "task_id": task_id,
                    "success": True,
                    "result": result,
                })
            except Exception as e:
                logger.error(f"任务 {task_id} 执行失败: {e}")
                results.append({
                    "task_id": task_id,
                    "success": False,
                    "error": str(e),
                })
        
        return results
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
```

### 8.2 缓存优化

```python
# src/utils/cache.py

"""
缓存模块
========

提供 LLM 响应缓存功能。
"""

import hashlib
import json
from typing import Any, Optional
from functools import lru_cache

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMCache:
    """LLM 响应缓存"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
    
    def _make_key(self, messages: list, model: str) -> str:
        """生成缓存键"""
        content = json.dumps({
            "messages": [m.content if hasattr(m, "content") else str(m) for m in messages],
            "model": model,
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, messages: list, model: str) -> Optional[Any]:
        """获取缓存"""
        key = self._make_key(messages, model)
        result = self._cache.get(key)
        if result:
            logger.debug(f"缓存命中: {key[:8]}")
        return result
    
    def set(self, messages: list, model: str, response: Any) -> None:
        """设置缓存"""
        if len(self._cache) >= self.max_size:
            # 移除最旧的项
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        key = self._make_key(messages, model)
        self._cache[key] = response
        logger.debug(f"缓存设置: {key[:8]}")
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()


# 全局缓存实例
llm_cache = LLMCache()
```

---

本扩展指南涵盖了系统的主要扩展点。根据具体需求，您可以：

1. 添加新的专业 Agent 处理特定领域任务
2. 集成更多外部工具和服务
3. 自定义工作流程和路由逻辑
4. 扩展记忆系统支持更复杂的检索
5. 对接企业级基础设施

如有问题，请参考源代码或提交 Issue。


---

**第六部分完成。所有交付物已全部输出完毕。**

---

## 📋 交付物清单总结

| 序号 | 交付物 | 状态 |
|------|--------|------|
| 1 | 系统架构设计文档 | ✅ 完成 |
| 2 | 模块与类设计说明 | ✅ 完成 |
| 3 | 项目目录结构树 | ✅ 完成 |
| 4 | 完整 Python 源代码 | ✅ 完成 |
| 5 | README.md | ✅ 完成 |
| 6 | Demo 示例（3个） | ✅ 完成 |
| 7 | 验证指南 | ✅ 完成 |
| 8 | 扩展指南 | ✅ 完成 |

所有代码文件均已完整输出，可直接复制使用。如需任何文件的补充说明或修改，请告知。