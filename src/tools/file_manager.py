"""
文件管理工具
============

提供安全的文件读写操作，限定在 workspace 目录内。
"""

import os
from pathlib import Path
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileManagerInput(BaseModel):
    """文件管理器输入参数"""
    action: Literal["read", "write", "list", "exists", "delete"] = Field(
        description="操作类型: read(读取), write(写入), list(列出), exists(检查存在), delete(删除)"
    )
    path: str = Field(
        description="文件路径（相对于 workspace 目录）"
    )
    content: Optional[str] = Field(
        default=None,
        description="写入的内容（仅 write 操作需要）"
    )
    
    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """验证路径安全性"""
        # 禁止路径遍历
        if ".." in v:
            raise ValueError("路径不能包含 '..'")
        # 禁止绝对路径
        if v.startswith("/") or (len(v) > 1 and v[1] == ":"):
            raise ValueError("不能使用绝对路径")
        return v


class FileManager:
    """
    文件管理器
    
    安全地在 workspace 目录内进行文件操作。
    
    安全特性：
    - 所有操作限定在 workspace 目录
    - 禁止路径遍历（..）
    - 禁止绝对路径
    - 操作前验证路径合法性
    """
    
    def __init__(self, workspace_dir: Optional[str] = None):
        """
        初始化文件管理器
        
        Args:
            workspace_dir: 工作目录路径，None 使用默认配置
        """
        settings = get_settings()
        self.workspace = Path(workspace_dir or settings.workspace_dir).resolve()
        
        # 确保工作目录存在
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger(self.__class__.__name__)
        self.logger.debug(f"工作目录: {self.workspace}")
    
    def _resolve_path(self, relative_path: str) -> Path:
        """
        解析相对路径为绝对路径
        
        Args:
            relative_path: 相对路径
            
        Returns:
            解析后的绝对路径
            
        Raises:
            ValueError: 路径不在工作目录内
        """
        # 构建完整路径
        full_path = (self.workspace / relative_path).resolve()
        
        # 验证路径在工作目录内
        try:
            full_path.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"路径 '{relative_path}' 超出工作目录范围")
        
        return full_path
    
    def read(self, path: str) -> str:
        """
        读取文件内容
        
        Args:
            path: 相对路径
            
        Returns:
            文件内容
        """
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        if not full_path.is_file():
            raise ValueError(f"路径不是文件: {path}")
        
        self.logger.info(f"读取文件: {path}")
        
        # 尝试不同编码
        for encoding in ["utf-8", "gbk", "latin-1"]:
            try:
                return full_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"无法解码文件: {path}")
    
    def write(self, path: str, content: str) -> str:
        """
        写入文件内容
        
        Args:
            path: 相对路径
            content: 要写入的内容
            
        Returns:
            操作结果消息
        """
        full_path = self._resolve_path(path)
        
        # 确保父目录存在
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"写入文件: {path}")
        
        full_path.write_text(content, encoding="utf-8")
        
        return f"成功写入 {len(content)} 字符到 {path}"
    
    def list_dir(self, path: str = ".") -> list:
        """
        列出目录内容
        
        Args:
            path: 相对路径，默认为工作目录根
            
        Returns:
            文件和目录列表
        """
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"目录不存在: {path}")
        
        if not full_path.is_dir():
            raise ValueError(f"路径不是目录: {path}")
        
        self.logger.info(f"列出目录: {path}")
        
        items = []
        for item in full_path.iterdir():
            item_type = "dir" if item.is_dir() else "file"
            items.append({
                "name": item.name,
                "type": item_type,
                "size": item.stat().st_size if item.is_file() else None,
            })
        
        return items
    
    def exists(self, path: str) -> bool:
        """
        检查路径是否存在
        
        Args:
            path: 相对路径
            
        Returns:
            是否存在
        """
        full_path = self._resolve_path(path)
        return full_path.exists()
    
    def delete(self, path: str) -> str:
        """
        删除文件
        
        Args:
            path: 相对路径
            
        Returns:
            操作结果消息
        """
        full_path = self._resolve_path(path)
        
        if not full_path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        if full_path.is_dir():
            raise ValueError(f"不能删除目录: {path}")
        
        self.logger.info(f"删除文件: {path}")
        
        full_path.unlink()
        
        return f"成功删除: {path}"


# 创建工具实例
_file_manager = FileManager()


@tool(args_schema=FileManagerInput)
def file_manager_tool(
    action: str,
    path: str,
    content: Optional[str] = None
) -> str:
    """
    安全的文件操作工具，所有操作限定在 workspace 目录内。
    
    支持的操作：
    - read: 读取文件内容
    - write: 写入内容到文件
    - list: 列出目录内容
    - exists: 检查文件/目录是否存在
    - delete: 删除文件
    
    使用示例：
    - 读取: action="read", path="data.txt"
    - 写入: action="write", path="output.json", content='{"key": "value"}'
    - 列出: action="list", path="."
    """
    try:
        if action == "read":
            result = _file_manager.read(path)
            # 限制输出长度
            if len(result) > 5000:
                result = result[:5000] + f"\n...(截断，总长度 {len(result)})"
            return result
        
        elif action == "write":
            if content is None:
                return "错误: write 操作需要提供 content 参数"
            return _file_manager.write(path, content)
        
        elif action == "list":
            items = _file_manager.list_dir(path)
            if not items:
                return f"目录 {path} 为空"
            lines = [f"目录 {path} 内容:"]
            for item in items:
                icon = "📁" if item["type"] == "dir" else "📄"
                size = f" ({item['size']} bytes)" if item["size"] else ""
                lines.append(f"  {icon} {item['name']}{size}")
            return "\n".join(lines)
        
        elif action == "exists":
            exists = _file_manager.exists(path)
            return f"{'存在' if exists else '不存在'}: {path}"
        
        elif action == "delete":
            return _file_manager.delete(path)
        
        else:
            return f"未知操作: {action}"
    
    except FileNotFoundError as e:
        return f"文件未找到: {str(e)}"
    except ValueError as e:
        return f"参数错误: {str(e)}"
    except Exception as e:
        logger.error(f"文件操作异常: {e}", exc_info=True)
        return f"操作失败: {str(e)}"