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