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