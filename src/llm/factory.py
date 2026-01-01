"""
LLM 工厂模块
============

使用工厂模式创建和管理 LLM 实例。
"""

from typing import Dict, Optional, Type, Any
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler

from src.config.settings import Settings, LLMConfig, get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TokenCounterCallback(BaseCallbackHandler):
    """Token 计数回调处理器"""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
    
    def on_llm_end(self, response, **kwargs):
        """LLM 调用结束时统计 token"""
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
    
    def get_usage(self) -> Dict[str, int]:
        """获取使用统计"""
        return {
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
            "total": self.total_tokens,
        }
    
    def reset(self):
        """重置计数器"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0


class LLMFactory:
    """
    LLM 工厂类
    
    使用工厂模式创建不同提供商的 LLM 实例。
    
    支持的提供商：
    - openai: OpenAI 模型
    - anthropic: Anthropic Claude 模型
    - local: 本地或兼容 API 的模型
    
    使用示例：
        >>> config = LLMConfig(provider="openai", model_name="gpt-4")
        >>> llm = LLMFactory.create(config)
        >>> response = llm.invoke("Hello!")
    """
    
    _instances: Dict[str, BaseChatModel] = {}
    _token_counters: Dict[str, TokenCounterCallback] = {}
    
    @classmethod
    def create(
        cls,
        config: Optional[LLMConfig] = None,
        cache_key: Optional[str] = None,
    ) -> BaseChatModel:
        """
        创建 LLM 实例
        
        Args:
            config: LLM 配置，None 使用默认配置
            cache_key: 缓存键，相同键返回缓存实例
            
        Returns:
            LLM 实例
        """
        if config is None:
            settings = get_settings()
            config = settings.get_llm_config()
        
        # 生成缓存键
        if cache_key is None:
            cache_key = f"{config.provider}:{config.model_name}"
        
        # 检查缓存
        if cache_key in cls._instances:
            logger.debug(f"使用缓存的 LLM 实例: {cache_key}")
            return cls._instances[cache_key]
        
        # 创建新实例
        logger.info(f"创建 LLM: {config.provider}/{config.model_name}")
        
        if config.provider == "openai":
            llm = cls._create_openai(config)
        elif config.provider == "anthropic":
            llm = cls._create_anthropic(config)
        elif config.provider == "local":
            llm = cls._create_local(config)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {config.provider}")
        
        # 缓存实例
        cls._instances[cache_key] = llm
        
        return llm
    
    @classmethod
    def _create_openai(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建 OpenAI LLM
        
        Args:
            config: LLM 配置
            
        Returns:
            ChatOpenAI 实例
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "请安装 langchain-openai: pip install langchain-openai"
            )
        
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        if config.api_key:
            kwargs["api_key"] = config.api_key
        
        if config.base_url:
            kwargs["base_url"] = config.base_url
        
        return ChatOpenAI(**kwargs)
    
    @classmethod
    def _create_anthropic(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建 Anthropic LLM
        
        Args:
            config: LLM 配置
            
        Returns:
            ChatAnthropic 实例
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "请安装 langchain-anthropic: pip install langchain-anthropic"
            )
        
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        if config.api_key:
            kwargs["anthropic_api_key"] = config.api_key
        
        return ChatAnthropic(**kwargs)
    
    @classmethod
    def _create_local(cls, config: LLMConfig) -> BaseChatModel:
        """
        创建本地/兼容 API 的 LLM
        
        使用 OpenAI 兼容接口连接本地模型（如 Ollama、vLLM 等）
        
        Args:
            config: LLM 配置
            
        Returns:
            ChatOpenAI 实例（使用自定义 base_url）
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "请安装 langchain-openai: pip install langchain-openai"
            )
        
        base_url = config.base_url or "http://localhost:11434/v1"
        
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            base_url=base_url,
            api_key=config.api_key or "not-needed",  # 本地模型通常不需要
        )
    
    @classmethod
    def get_token_counter(cls, cache_key: str) -> Optional[TokenCounterCallback]:
        """
        获取 token 计数器
        
        Args:
            cache_key: LLM 缓存键
            
        Returns:
            TokenCounterCallback 实例
        """
        return cls._token_counters.get(cache_key)
    
    @classmethod
    def clear_cache(cls) -> None:
        """清空所有缓存的 LLM 实例"""
        cls._instances.clear()
        cls._token_counters.clear()
        logger.info("LLM 缓存已清空")
    
    @classmethod
    def list_cached(cls) -> list:
        """
        列出所有缓存的 LLM
        
        Returns:
            缓存键列表
        """
        return list(cls._instances.keys())


def create_llm(
    provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseChatModel:
    """
    便捷函数：创建 LLM 实例
    
    Args:
        provider: 提供商 (openai/anthropic/local)
        model_name: 模型名称
        temperature: 温度参数
        max_tokens: 最大 token 数
        api_key: API 密钥
        base_url: 基础 URL
        
    Returns:
        LLM 实例
    """
    # 默认模型名称
    default_models = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-sonnet-20240229",
        "local": "llama3",
    }
    
    if model_name is None:
        model_name = default_models.get(provider, "gpt-4o-mini")
    
    config = LLMConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
    )
    
    return LLMFactory.create(config)