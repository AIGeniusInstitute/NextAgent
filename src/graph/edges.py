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