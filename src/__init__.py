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