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