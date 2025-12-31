# NextAgent：通用 Multi-Agent 问题求解系统 - 完整开发任务书

## 角色设定

你是一位精通 LangChain/LangGraph 的**资深 Multi-Agent 系统架构师**与 **Python 全栈工程负责人**，拥有 10 年以上分布式系统与 AI 应用开发经验。

---

## 一、项目目标

从 0 到 1 设计并实现一个**通用多智能体协作系统（General-Purpose Multi-Agent Problem-Solving System）**：
- 能够自动理解任意用户任务输入
- 智能规划、分解、分派给不同专业智能体执行
- 通过协作、审阅、反思、纠错闭环
- 最终输出高质量结果

> 定位：类似 AutoGPT + LangGraph 的**可控增强版**，支持工具调用与可观测性。

---

## 二、技术栈（硬约束）

| 项目 | 要求 |
|------|------|
| 语言 | Python 3.10+（推荐 3.11） |
| 核心框架 | **LangGraph >= 0.2.0**（必须基于此实现多智能体编排） |
| LLM 集成 | LangChain 生态（支持 OpenAI / Claude / 本地模型），通过环境变量配置 |
| 数据模型 | Pydantic V2 + 完整类型注解（Type Hints） |
| 依赖管理 | `pyproject.toml`（支持 poetry / uv / pip 一键安装） |
| 代码规范 | PEP8 + docstring + 详细注释 |
| 设计模式 | 工厂模式、策略模式、观察者模式（按需） |

---

## 三、系统架构设计要求

### 3.1 架构模式
采用 **Supervisor + Worker 混合架构**：
- 主控节点（Coordinator/Supervisor）负责任务理解、规划与路由
- 多个专业 Worker Agent 执行具体子任务
- 支持循环交互直到任务完成（FINISH 状态）

### 3.2 必须输出的架构内容

1. **系统总体架构图**（Mermaid 流程图 + 文字说明）
2. **状态机设计**（StateGraph 定义 + 状态流转）
3. **智能体交互时序图**
4. **核心数据结构说明**（State Schema）
5. **失败重试与终止条件设计**

### 3.3 核心组件

| 组件 | 职责 |
|------|------|
| 任务解析器 | 输入标准化、意图识别 |
| 任务规划器 | 任务分解、依赖分析、执行计划生成 |
| Agent 调度器 | 能力路由、负载分配、并行/串行控制 |
| 状态管理器 | 全局状态维护、上下文传递 |
| 记忆系统 | 短期记忆（会话内）+ 长期记忆（可选持久化） |
| 工具路由器 | 工具注册、调用封装、权限控制 |
| 评估器 | 结果质量评估、Token 成本统计 |

---

## 四、智能体角色定义（至少包含）

| 角色 | 英文名 | 核心职责 |
|------|--------|----------|
| 协调者 | Coordinator/Supervisor | 任务理解、工作分配、进度监控、结果整合 |
| 规划者 | Planner | 任务拆解、执行计划制定、依赖排序 |
| 研究员 | Researcher | 信息检索、知识整合、资料分析 |
| 执行者 | Executor | 工具调用、代码执行、具体操作 |
| 编码者 | Coder | 代码编写、调试、技术实现 |
| 审核者 | Critic/Reviewer | 质量检查、错误发现、改进建议 |
| 综合者 | Synthesizer | 结果汇总、最终输出生成 |

**要求**：
- 角色可通过配置开启/关闭/调整数量
- 提示词模板外置管理，支持替换
- 新增 Agent 只需继承基类并注册

---

## 五、工作流设计

### 5.1 核心链路

```
用户输入 → 输入解析 → 任务规划 → 任务分解 
    → 并行/串行协作执行 → 交叉审阅 → 反思纠错（循环）
    → 结果汇总 → 最终输出
```

### 5.2 功能要求

**核心功能（必须实现）**：
- [√] 多 Agent 协作对话
- [√] 任务自动分解与规划
- [√] 动态任务编排（根据问题类型自动规划路径）
- [√] 智能体通信机制（消息传递、状态共享）
- [√] 循环与条件分支（迭代优化、条件判断）
- [√] 动态工具调用
- [√] 执行结果汇总与输出
- [√] 错误恢复机制（失败重试、降级策略、异常处理）

**高级功能（必须实现）**：
- [√] 计划/执行/反思/纠错链路（Plan-Execute-Reflect Loop）
- [√] Agent 间并行执行
- [√] 人工介入节点（Human-in-the-loop）
- [√] 对话历史持久化
- [√] 执行过程可视化/日志追踪
- [√] LLM 输出必须显示 reasoning 过程

---

## 六、状态与记忆系统

### 6.1 Graph State 必须包含
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]          # 对话历史
    original_task: str                   # 原始问题
    subtasks: List[SubTask]              # 子任务列表
    agent_outputs: Dict[str, Any]        # 每个 agent 的产出
    tool_call_logs: List[ToolCallLog]    # 工具调用日志
    current_agent: str                   # 当前执行 agent
    iteration_count: int                 # 迭代次数
    final_answer: Optional[str]          # 最终答案
    next: str                            # 下一节点路由
```

### 6.2 记忆机制
- **短期记忆**：会话内上下文，必须实现
- **长期记忆**：文件持久化或 KV 存储，提供接口（默认可关闭）
- **反思缓存**：存储历史反思结果，避免重复错误

---

## 七、工具系统

### 7.1 内置工具示例（至少实现）

| 工具类型 | 示例工具 | 说明 |
|----------|----------|------|
| 计算/文本 | safe_eval, json_validator | 安全表达式计算、JSON 校验格式化 |
| 文件操作 | file_reader, file_writer | **限定 `workspace/` 目录**，禁止任意路径 |
| 代码执行 | python_repl | 沙箱执行 Python 代码 |
| 信息检索 | web_search（模拟） | 网络搜索模拟 |

### 7.2 工具系统要求
- 统一封装（使用 `@tool` 装饰器）
- 入参校验（Pydantic）
- 异常处理与日志记录
- **安全边界限制**（文件操作限定目录）

---

## 八、可观测性与调试

- [x] 每个节点的输入/输出摘要打印
- [x] 执行耗时统计
- [x] 错误栈记录
- [x] 可配置开关（DEBUG 模式）
- [x] 产出 `logs/` 目录或控制台可读执行轨迹
- [x] 支持执行过程可视化（可选 Mermaid 图）

---

## 九、项目结构（必须遵循）

```
multi_agent_system/
├── pyproject.toml                 # 项目配置与依赖
├── requirements.txt               # pip 依赖（备用）
├── README.md                      # 完整使用说明
├── .env.example                   # 环境变量示例
├── src/
│   ├── __init__.py
│   ├── main.py                    # 系统入口（CLI）
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
│   │   ├── base.py                # 工具基类
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
│   ├── example_planning.py        # 规划分解示例
│   ├── example_tool_execution.py  # 工具执行示例
│   └── example_code_generation.py # 代码生成示例
├── tests/
│   ├── __init__.py
│   ├── test_graph.py              # 图构建测试
│   ├── test_flow.py               # 完整流程测试
│   └── test_tools.py              # 工具调用测试
├── workspace/                     # 工具操作的限定目录
│   └── .gitkeep
└── logs/                          # 执行日志
    └── .gitkeep
```

---

## 十、交付物清单（按顺序输出）

### 必须全部给出，不可省略：

1. **系统架构设计文档**
   - 整体架构图（Mermaid）
   - 组件职责说明
   - 数据流说明
   - LangGraph 节点/边设计
   - 状态结构定义
   - 失败重试与终止条件

2. **模块与类设计说明**
   - 核心类 UML 或文字结构
   - 接口定义

3. **项目目录结构树**

4. **完整 Python 源代码**
   - 按文件分别输出
   - 使用 Markdown 代码块
   - 代码块上方写明文件路径
   - **所有文件必须完整，不可用"略""自行补充"**

5. **README.md**
   - 项目介绍
   - 安装步骤
   - 环境配置
   - 运行命令
   - 示例演示

6. **Demo 示例（至少 3 个）**
   - 示例 1：任务规划分解类（如"制定一个学习 Python 的计划"）
   - 示例 2：代码生成类（如"编写爬虫抓取 Hacker News"）
   - 示例 3：综合分析类（如"分析某业务需求并给出方案"）

7. **验证指南**
   - 安装命令
   - 环境变量设置
   - 运行示例
   - 期望输出说明

8. **扩展指南**
   - 如何新增 Agent
   - 如何新增工具
   - 如何新增工作流节点
   - 如何对接企业级场景（可选）

---

## 十一、评估指标（系统需支持统计）

| 指标 | 说明 |
|------|------|
| 任务成功率 | 成功完成的任务比例 |
| Token 成本 | 单次任务 Token 消耗 |
| 执行链路深度 | 从输入到输出经过的节点数 |
| 反思次数 | Critic 触发的重试轮数 |
| 执行耗时 | 端到端时间 |

---

## 十二、重要约束（必须遵守）

1. **不要省略任何关键文件**，不要用 "略""自行补充""..."
2. **所有代码必须可直接运行**，不要伪代码
3. **所有示例和测试必须可执行**
4. **文件操作必须限定安全目录**（`workspace/`）
5. **关键设计决策请说明理由**
6. **代码必须包含完整类型注解和 docstring**
7. **LangGraph 的 END 状态和 compile() 流程必须有注释说明**

---

## 十三、演示场景验证

完成开发后，请使用以下场景验证系统功能：

**任务**："请帮我编写一个 Python 爬虫，抓取 Hacker News 首页的文章标题和链接，并保存为 JSON 文件"

展示完整的：
- 任务理解与分解过程
- 智能体协作流程
- 代码生成与审核
- 工具调用（文件写入）
- 最终执行结果

---

现在请开始分段完整输出所有交付物。每段完成之后，提示用户继续输出下一段。
