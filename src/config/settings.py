"""
配置模块
========
提供系统配置管理功能。
"""
from src.config.settings import Settings, get_settings
from src.config.prompts import PromptTemplates, get_prompt

__all__ = [
    "Settings",
    "get_settings",
    "PromptTemplates",
    "get_prompt",
]

"""
配置管理模块
============
使用 Pydantic 进行配置验证，支持环境变量和配置文件。
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "local"] = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0 <= v <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")
        return v

class AgentConfig(BaseModel):
    name: str
    enabled: bool = True
    llm_override: Optional[LLMConfig] = None
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    tools: List[str] = Field(default_factory=list)
    custom_prompt: Optional[str] = None

class RetryConfig(BaseModel):
    max_task_retries: int = Field(default=3, ge=0)
    max_global_iterations: int = Field(default=10, ge=1)
    retry_delay_seconds: float = Field(default=1.0, ge=0)
    exponential_backoff: bool = True
    backoff_multiplier: float = Field(default=2.0, ge=1)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    llm_provider: Literal["openai", "anthropic", "local"] = Field(default="openai", alias="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", alias="ANTHROPIC_MODEL")
    local_model_url: str = Field(default="http://localhost:11434/v1", alias="LOCAL_MODEL_URL")
    local_model_name: str = Field(default="llama3", alias="LOCAL_MODEL_NAME")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, alias="LLM_MAX_TOKENS")
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")
    max_iterations: int = Field(default=10, alias="MAX_ITERATIONS")
    max_task_retries: int = Field(default=3, alias="MAX_TASK_RETRIES")
    global_timeout: int = Field(default=300, alias="GLOBAL_TIMEOUT")
    enable_human_in_loop: bool = Field(default=True, alias="ENABLE_HUMAN_IN_LOOP")
    human_review_threshold: float = Field(default=0.6, alias="HUMAN_REVIEW_THRESHOLD")
    enable_parallel_execution: bool = Field(default=True, alias="ENABLE_PARALLEL_EXECUTION")
    max_parallel_tasks: int = Field(default=3, alias="MAX_PARALLEL_TASKS")
    workspace_dir: str = Field(default="workspace", alias="WORKSPACE_DIR")
    log_dir: str = Field(default="logs", alias="LOG_DIR")
    enable_long_term_memory: bool = Field(default=False, alias="ENABLE_LONG_TERM_MEMORY")
    memory_storage_path: str = Field(default="data/memory", alias="MEMORY_STORAGE_PATH")
    enable_visualization: bool = Field(default=True, alias="ENABLE_VISUALIZATION")
    visualization_format: Literal["mermaid", "png"] = Field(default="mermaid", alias="VISUALIZATION_FORMAT")
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
        self._init_default_agents()

    def _ensure_directories(self) -> None:
        for dir_path in [self.workspace_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        if self.enable_long_term_memory:
            Path(self.memory_storage_path).mkdir(parents=True, exist_ok=True)

    def _init_default_agents(self) -> None:
        default_agents = {
            "coordinator": AgentConfig(name="coordinator", enabled=True, tools=[]),
            "planner": AgentConfig(name="planner", enabled=True, tools=[]),
            "researcher": AgentConfig(name="researcher", enabled=True, tools=["web_search"]),
            "coder": AgentConfig(name="coder", enabled=True, tools=["code_executor", "file_manager"]),
            "executor": AgentConfig(name="executor", enabled=True, tools=["calculator", "file_manager", "code_executor"]),
            "critic": AgentConfig(name="critic", enabled=True, tools=[]),
            "synthesizer": AgentConfig(name="synthesizer", enabled=True, tools=["file_manager"]),
        }
        for name, config in default_agents.items():
            if name not in self.agents:
                self.agents[name] = config

    def get_llm_config(self) -> LLMConfig:
        if self.llm_provider == "openai":
            return LLMConfig(provider="openai", model_name=self.openai_model, temperature=self.llm_temperature, max_tokens=self.llm_max_tokens, api_key=self.openai_api_key, base_url=self.openai_base_url)
        elif self.llm_provider == "anthropic":
            return LLMConfig(provider="anthropic", model_name=self.anthropic_model, temperature=self.llm_temperature, max_tokens=self.llm_max_tokens, api_key=self.anthropic_api_key)
        else:
            return LLMConfig(provider="local", model_name=self.local_model_name, temperature=self.llm_temperature, max_tokens=self.llm_max_tokens, base_url=self.local_model_url)

    def is_agent_enabled(self, agent_name: str) -> bool:
        if agent_name in self.agents:
            return self.agents[agent_name].enabled
        return True

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        return self.agents.get(agent_name)

@lru_cache()
def get_settings() -> Settings:
    return Settings()

def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()



    