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