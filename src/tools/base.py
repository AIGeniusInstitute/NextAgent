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