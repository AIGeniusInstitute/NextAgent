"""
长期记忆模块
============

实现持久化记忆存储，支持文件系统存储。
"""

import json
import os
import hashlib
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import threading

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LongTermMemory:
    """
    长期记忆
    
    基于文件系统的持久化存储。
    每个记忆项存储为单独的 JSON 文件，支持索引加速查找。
    
    特性：
    - 持久化存储
    - 自动索引
    - 支持元数据
    - 线程安全
    
    存储结构：
        memory_storage/
        ├── index.json          # 索引文件
        ├── items/
        │   ├── abc123.json    # 记忆项文件
        │   └── def456.json
        └── metadata.json       # 元数据
    
    使用示例：
        >>> memory = LongTermMemory()
        >>> memory.store("task_history", {"task": "爬虫", "result": "成功"})
        >>> history = memory.retrieve("task_history")
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_save: bool = True
    ):
        """
        初始化长期记忆
        
        Args:
            storage_path: 存储路径，None 使用默认配置
            auto_save: 是否自动保存索引
        """
        settings = get_settings()
        self.storage_path = Path(
            storage_path or settings.memory_storage_path
        ).resolve()
        self.auto_save = auto_save
        
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.logger = get_logger(self.__class__.__name__)
        
        # 初始化存储目录
        self._init_storage()
        
        # 加载索引
        self._load_index()
        
        self.logger.info(f"初始化长期记忆，路径: {self.storage_path}")
    
    def _init_storage(self) -> None:
        """初始化存储目录结构"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "items").mkdir(exist_ok=True)
    
    def _get_item_path(self, key: str) -> Path:
        """
        获取记忆项的文件路径
        
        Args:
            key: 键
            
        Returns:
            文件路径
        """
        # 使用 hash 确保文件名安全
        key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
        return self.storage_path / "items" / f"{key_hash}.json"
    
    def _load_index(self) -> None:
        """加载索引"""
        index_path = self.storage_path / "index.json"
        
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
                self.logger.debug(f"加载索引，共 {len(self._index)} 项")
            except Exception as e:
                self.logger.warning(f"索引加载失败: {e}")
                self._index = {}
        else:
            self._index = {}
    
    def _save_index(self) -> None:
        """保存索引"""
        index_path = self.storage_path / "index.json"
        
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"索引保存失败: {e}")
    
    def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        存储记忆项
        
        Args:
            key: 键
            value: 值（必须可 JSON 序列化）
            metadata: 元数据
        """
        with self._lock:
            item_path = self._get_item_path(key)
            
            # 构建记忆项数据
            item_data = {
                "key": key,
                "value": value,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            
            # 保存到文件
            try:
                with open(item_path, "w", encoding="utf-8") as f:
                    json.dump(item_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.logger.error(f"存储失败: {e}")
                raise
            
            # 更新索引
            self._index[key] = {
                "path": str(item_path),
                "created_at": item_data["created_at"],
                "updated_at": item_data["updated_at"],
                "metadata_keys": list((metadata or {}).keys()),
            }
            
            if self.auto_save:
                self._save_index()
            
            self.logger.debug(f"存储记忆项: {key}")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        检索记忆项
        
        Args:
            key: 键
            
        Returns:
            值，不存在返回 None
        """
        with self._lock:
            if key not in self._index:
                return None
            
            item_path = self._get_item_path(key)
            
            if not item_path.exists():
                # 索引不一致，清理
                del self._index[key]
                if self.auto_save:
                    self._save_index()
                return None
            
            try:
                with open(item_path, "r", encoding="utf-8") as f:
                    item_data = json.load(f)
                return item_data.get("value")
            except Exception as e:
                self.logger.error(f"检索失败: {e}")
                return None
    
    def delete(self, key: str) -> bool:
        """
        删除记忆项
        
        Args:
            key: 键
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if key not in self._index:
                return False
            
            item_path = self._get_item_path(key)
            
            # 删除文件
            if item_path.exists():
                try:
                    item_path.unlink()
                except Exception as e:
                    self.logger.error(f"删除文件失败: {e}")
            
            # 更新索引
            del self._index[key]
            if self.auto_save:
                self._save_index()
            
            self.logger.debug(f"删除记忆项: {key}")
            return True
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 键
            
        Returns:
            是否存在
        """
        with self._lock:
            return key in self._index
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        搜索记忆项
        
        简单的关键词匹配搜索。
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            
        Returns:
            匹配的记忆项列表
        """
        with self._lock:
            results = []
            query_lower = query.lower()
            
            for key in self._index:
                if query_lower in key.lower():
                    value = self.retrieve(key)
                    if value is not None:
                        results.append({
                            "key": key,
                            "value": value,
                            "score": 1.0,  # 简单匹配不计算分数
                        })
                
                if len(results) >= top_k:
                    break
            
            return results
    
    def list_keys(self) -> List[str]:
        """
        列出所有键
        
        Returns:
            键列表
        """
        with self._lock:
            return list(self._index.keys())
    
    def clear(self) -> None:
        """清空所有记忆"""
        with self._lock:
            # 删除所有文件
            items_dir = self.storage_path / "items"
            if items_dir.exists():
                for item_file in items_dir.glob("*.json"):
                    try:
                        item_file.unlink()
                    except Exception as e:
                        self.logger.warning(f"删除文件失败: {e}")
            
            # 清空索引
            self._index = {}
            self._save_index()
            
            self.logger.info("清空长期记忆")
    
    def size(self) -> int:
        """
        获取存储项数
        
        Returns:
            项数
        """
        with self._lock:
            return len(self._index)
    
    def persist(self) -> None:
        """强制持久化索引"""
        with self._lock:
            self._save_index()
            self.logger.debug("索引已持久化")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            items_dir = self.storage_path / "items"
            total_size = sum(
                f.stat().st_size for f in items_dir.glob("*.json")
            ) if items_dir.exists() else 0
            
            return {
                "item_count": len(self._index),
                "storage_path": str(self.storage_path),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }