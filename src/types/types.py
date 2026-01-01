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

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"

class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TaskType(str, Enum):
    RESEARCH = "research"
    CODE = "code"
    EXECUTE = "execute"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"

class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    EXECUTOR = "executor"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"

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

class SubTask(BaseModel):
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
    score: float = Field(ge=0, le=1, description="评分")
    passed: bool = Field(description="是否通过")
    issues: List[str] = Field(default_factory=list, description="发现的问题")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")
    reasoning: str = Field(default="", description="评估推理过程")

class ExecutionMetrics(BaseModel):
    total_duration_seconds: float = Field(default=0.0, description="总执行时间")
    token_usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt": 0, "completion": 0, "total": 0},
        description="Token 使用统计"
    )
    agent_durations: Dict[str, float] = Field(default_factory=dict, description="各 Agent 执行时间")
    iteration_count: int = Field(default=0, description="迭代次数")
    retry_count: int = Field(default=0, description="重试次数")
    tool_call_count: int = Field(default=0, description="工具调用次数")
    success: bool = Field(default=False, description="是否成功")

class FinalResult(BaseModel):
    task_id: str = Field(description="任务ID")
    original_task: str = Field(description="原始任务")
    answer: str = Field(description="最终答案")
    reasoning_trace: List[str] = Field(default_factory=list, description="推理轨迹")
    agent_outputs: Dict[str, Any] = Field(default_factory=dict, description="各Agent输出")
    metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics, description="执行指标")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    class Config:
        arbitrary_types_allowed = True

class AgentProtocol(Protocol):
    @property
    def name(self) -> str: ...
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]: ...

class ToolProtocol(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    def invoke(self, **kwargs: Any) -> Any: ...

NodeHandler = Callable[[Dict[str, Any]], Dict[str, Any]]
RouterFunction = Callable[[Dict[str, Any]], RouteType]
AgentCapabilities = Dict[AgentRole, List[TaskType]]
T = TypeVar("T")


