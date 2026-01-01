"""
日志工具模块
============

提供统一的日志配置和管理。
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler
from rich.console import Console

# 全局日志配置
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_RICH_FORMAT = "%(message)s"

# 日志级别映射
_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# 全局日志器缓存
_loggers: dict = {}
_initialized = False


def setup_logger(
    log_dir: str = "logs",
    log_file: Optional[str] = None,
    level: str = "info",
    debug: bool = False,
    use_rich: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    设置全局日志配置
    
    Args:
        log_dir: 日志目录
        log_file: 日志文件名，None 自动生成
        level: 日志级别
        debug: 是否启用调试模式（覆盖 level）
        use_rich: 是否使用 Rich 美化输出
        max_file_size: 单个日志文件最大大小
        backup_count: 保留的备份文件数
    """
    global _initialized
    
    if _initialized:
        return
    
    # 确定日志级别
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = _LOG_LEVELS.get(level.lower(), logging.INFO)
    
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 确定日志文件名
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"multi_agent_{timestamp}.log"
    
    log_file_path = log_path / log_file
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 添加控制台处理器
    if use_rich:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=debug,
            rich_tracebacks=True,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter(_RICH_FORMAT))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT)
        )
    
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT)
    )
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
    root_logger.addHandler(file_handler)
    
    # 降低第三方库的日志级别
    for lib in ["httpx", "httpcore", "openai", "anthropic", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    _initialized = True
    
    # 记录启动信息
    root_logger.info(f"日志系统初始化完成，文件: {log_file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器实例
    
    Args:
        name: 日志器名称
        
    Returns:
        Logger 实例
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger
    
    return _loggers[name]


def set_log_level(level: str, logger_name: Optional[str] = None) -> None:
    """
    设置日志级别
    
    Args:
        level: 日志级别
        logger_name: 日志器名称，None 表示根日志器
    """
    log_level = _LOG_LEVELS.get(level.lower(), logging.INFO)
    
    if logger_name:
        logging.getLogger(logger_name).setLevel(log_level)
    else:
        logging.getLogger().setLevel(log_level)


class LoggerContext:
    """
    日志上下文管理器
    
    用于临时修改日志级别。
    
    使用示例：
        >>> with LoggerContext("debug"):
        ...     # 此处日志级别为 DEBUG
        ...     pass
        >>> # 日志级别恢复
    """
    
    def __init__(self, level: str, logger_name: Optional[str] = None):
        self.level = level
        self.logger_name = logger_name
        self._original_level = None
    
    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self._original_level = logger.level
        set_log_level(self.level, self.logger_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self._original_level)
        return False