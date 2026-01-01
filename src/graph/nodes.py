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
    
    result["iteration_count"] = state.get("iteration_count", 0) + 1
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