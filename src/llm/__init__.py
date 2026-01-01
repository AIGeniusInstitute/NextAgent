"""
LLM 模块
========

提供语言模型的创建和管理功能。

支持的 LLM 提供商：
- OpenAI (GPT-4, GPT-3.5 等)
- Anthropic (Claude 系列)
- 本地模型 (通过兼容 API)
"""

from src.llm.factory import LLMFactory, create_llm

__all__ = [
    "LLMFactory",
    "create_llm",
]