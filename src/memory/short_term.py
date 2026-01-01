"""
短期记忆模块
============

实现会话内的上下文记忆，用于在单次任务执行过程中保存和检索信息。
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import OrderedDict
import threading

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MemoryItem:
    """记忆项"""
    
    def __init__(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化记忆项
        
        Args:
            key: 键
            value: 值
            metadata: 元数据
        """
        self.key = key
        self.value = value
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.accessed_at = datetime.now()
        self.access_count = 0
    
    def access(self) -> Any:
        """访问记忆项，更新访问信息"""
        self.accessed_at = datetime.now()
        self.access_count += 1
        return self.value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "value": self.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
        }


class ShortTermMemory:
    """
    短期记忆
    
    基于 LRU（最近最少使用）策略的内存存储。
    当超过最大容量时，自动淘汰最久未访问的记忆项。
    
    特性：
    - 线程安全
    - 自动过期
    - LRU 淘汰策略
    - 支持元数据
    
    使用示例：
        >>> memory = ShortTermMemory(max_size=100)
        >>> memory.store("user_task", "编写爬虫")
        >>> task = memory.retrieve("user_task")
        >>> print(task)  # "编写爬虫"
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = None
    ):
        """
        初始化短期记忆
        
        Args:
            max_size: 最大存储项数
            default_ttl: 默认生存时间（秒），None 表示不过期
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._storage: OrderedDict[str, MemoryItem] = OrderedDict()
        self._lock = threading.RLock()
        self.logger = get_logger(self.__class__.__name__)
        
        self.logger.debug(f"初始化短期记忆，最大容量: {max_size}")
    
    def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> None:
        """
        存储记忆项
        
        Args:
            key: 键
            value: 值
            metadata: 元数据
            ttl: 生存时间（秒），覆盖默认值
        """
        with self._lock:
            # 如果键已存在，先删除（以更新顺序）
            if key in self._storage:
                del self._storage[key]
            
            # 检查容量，必要时淘汰
            while len(self._storage) >= self.max_size:
                oldest_key = next(iter(self._storage))
                del self._storage[oldest_key]
                self.logger.debug(f"淘汰记忆项: {oldest_key}")
            
            # 存储新项
            item_metadata = metadata or {}
            if ttl or self.default_ttl:
                item_metadata["ttl"] = ttl or self.default_ttl
            
            self._storage[key] = MemoryItem(key, value, item_metadata)
            self.logger.debug(f"存储记忆项: {key}")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        检索记忆项
        
        Args:
            key: 键
            
        Returns:
            值，不存在或已过期返回 None
        """
        with self._lock:
            if key not in self._storage:
                return None
            
            item = self._storage[key]
            
            # 检查过期
            if self._is_expired(item):
                del self._storage[key]
                self.logger.debug(f"记忆项已过期: {key}")
                return None
            
            # 移动到末尾（最近访问）
            self._storage.move_to_end(key)
            
            return item.access()
    
    def _is_expired(self, item: MemoryItem) -> bool:
        """
        检查记忆项是否过期
        
        Args:
            item: 记忆项
            
        Returns:
            是否过期
        """
        ttl = item.metadata.get("ttl")
        if ttl is None:
            return False
        
        elapsed = (datetime.now() - item.created_at).total_seconds()
        return elapsed > ttl
    
    def delete(self, key: str) -> bool:
        """
        删除记忆项
        
        Args:
            key: 键
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if key in self._storage:
                del self._storage[key]
                self.logger.debug(f"删除记忆项: {key}")
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 键
            
        Returns:
            是否存在
        """
        with self._lock:
            if key not in self._storage:
                return False
            
            item = self._storage[key]
            if self._is_expired(item):
                del self._storage[key]
                return False
            
            return True
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的 n 个记忆项
        
        Args:
            n: 数量
            
        Returns:
            记忆项字典列表
        """
        with self._lock:
            # 清理过期项
            self._cleanup_expired()
            
            # 获取最近的项（从末尾开始）
            items = list(self._storage.values())[-n:]
            items.reverse()
            
            return [item.to_dict() for item in items]
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        简单搜索（基于键名匹配）
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            
        Returns:
            匹配的记忆项
        """
        with self._lock:
            self._cleanup_expired()
            
            results = []
            query_lower = query.lower()
            
            for key, item in self._storage.items():
                # 简单的键名和值匹配
                if query_lower in key.lower():
                    results.append(item.to_dict())
                elif isinstance(item.value, str) and query_lower in item.value.lower():
                    results.append(item.to_dict())
                
                if len(results) >= top_k:
                    break
            
            return results
    
    def _cleanup_expired(self) -> int:
        """
        清理过期项
        
        Returns:
            清理的项数
        """
        expired_keys = [
            key for key, item in self._storage.items()
            if self._is_expired(item)
        ]
        
        for key in expired_keys:
            del self._storage[key]
        
        if expired_keys:
            self.logger.debug(f"清理 {len(expired_keys)} 个过期项")
        
        return len(expired_keys)
    
    def clear(self) -> None:
        """清空所有记忆"""
        with self._lock:
            self._storage.clear()
            self.logger.info("清空短期记忆")
    
    def size(self) -> int:
        """
        获取当前存储项数
        
        Returns:
            项数
        """
        with self._lock:
            return len(self._storage)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取记忆统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total_access = sum(
                item.access_count for item in self._storage.values()
            )
            
            return {
                "size": len(self._storage),
                "max_size": self.max_size,
                "total_access_count": total_access,
                "oldest_item": (
                    next(iter(self._storage)).key
                    if self._storage else None
                ),
                "newest_item": (
                    list(self._storage.keys())[-1]
                    if self._storage else None
                ),
            }