"""
配置管理模块
============

使用 Pydantic 进行配置验证，支持环境变量和配置文件。
"""
from dotenv import load_dotenv
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class LLMConfig(BaseModel):
    """LLM 配置模型"""
    provider: Literal["openai", "anthropic", "local"] = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, gt=0)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """验证温度参数"""
        if not 0 <= v <= 2:
            raise ValueError("temperature 必须在 0 到 2 之间")
        return v


class AgentConfig(BaseModel):
    """Agent 配置模型"""
    name: str
    enabled: bool = True
    llm_override: Optional[LLMConfig] = None
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    tools: List[str] = Field(default_factory=list)
    custom_prompt: Optional[str] = None


class RetryConfig(BaseModel):
    """重试配置模型"""
    max_task_retries: int = Field(default=3, ge=0)
    max_global_iterations: int = Field(default=10, ge=1)
    retry_delay_seconds: float = Field(default=1.0, ge=0)
    exponential_backoff: bool = True
    backoff_multiplier: float = Field(default=2.0, ge=1)


class Settings(BaseSettings):
    """
    系统配置类
    
    配置优先级（从高到低）：
    1. 环境变量
    2. .env 文件
    3. 默认值
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ===== LLM 配置 =====
    llm_provider: Literal["openai", "anthropic", "local"] = Field(
        default="openai",
        env="LLM_PROVIDER"
    )
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")

    # Anthropic
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-sonnet-20240229", env="ANTHROPIC_MODEL")
    
    # Local
    local_model_url: str = Field(
        default="http://localhost:11434/v1",
        env="LOCAL_MODEL_URL"
    )
    local_model_name: str = Field(default="llama3", env="LOCAL_MODEL_NAME")
    
    # LLM 通用参数
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=4096, env="LLM_MAX_TOKENS")
    
    # ===== 系统配置 =====
    debug_mode: bool = Field(default=False, env="DEBUG_MODE")
    max_iterations: int = Field(default=10, env="MAX_ITERATIONS")
    max_task_retries: int = Field(default=3, env="MAX_TASK_RETRIES")
    global_timeout: int = Field(default=300, env="GLOBAL_TIMEOUT")
    
    # ===== 人工介入 =====
    enable_human_in_loop: bool = Field(default=True, env="ENABLE_HUMAN_IN_LOOP")
    human_review_threshold: float = Field(default=0.6, env="HUMAN_REVIEW_THRESHOLD")
    
    # ===== 并行执行 =====
    enable_parallel_execution: bool = Field(default=True, env="ENABLE_PARALLEL_EXECUTION")
    max_parallel_tasks: int = Field(default=3, env="MAX_PARALLEL_TASKS")
    
    # ===== 目录配置 =====
    workspace_dir: str = Field(default="workspace", env="WORKSPACE_DIR")
    log_dir: str = Field(default="logs", env="LOG_DIR")
    
    # ===== 记忆系统 =====
    enable_long_term_memory: bool = Field(default=False, env="ENABLE_LONG_TERM_MEMORY")
    memory_storage_path: str = Field(default="data/memory", env="MEMORY_STORAGE_PATH")
    
    # ===== 可视化 =====
    enable_visualization: bool = Field(default=True, env="ENABLE_VISUALIZATION")
    visualization_format: Literal["mermaid", "png"] = Field(
        default="mermaid",
        env="VISUALIZATION_FORMAT"
    )
    
    # ===== Agent 配置 =====
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)
    
    # ===== 重试配置 =====
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
        self._init_default_agents()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.openai_base_url = os.environ.get("OPENAI_BASE_URL")
        self.openai_model = os.environ.get("OPENAI_MODEL")

    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        for dir_path in [self.workspace_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        if self.enable_long_term_memory:
            Path(self.memory_storage_path).mkdir(parents=True, exist_ok=True)
    
    def _init_default_agents(self) -> None:
        """初始化默认 Agent 配置"""
        default_agents = {
            "coordinator": AgentConfig(
                name="coordinator",
                enabled=True,
                tools=[]
            ),
            "planner": AgentConfig(
                name="planner",
                enabled=True,
                tools=[]
            ),
            "researcher": AgentConfig(
                name="researcher",
                enabled=True,
                tools=["web_search"]
            ),
            "coder": AgentConfig(
                name="coder",
                enabled=True,
                tools=["code_executor", "file_manager"]
            ),
            "executor": AgentConfig(
                name="executor",
                enabled=True,
                tools=["calculator", "file_manager", "code_executor"]
            ),
            "critic": AgentConfig(
                name="critic",
                enabled=True,
                tools=[]
            ),
            "synthesizer": AgentConfig(
                name="synthesizer",
                enabled=True,
                tools=["file_manager"]
            ),
        }
        
        # 合并用户配置
        for name, config in default_agents.items():
            if name not in self.agents:
                self.agents[name] = config



    def get_llm_config(self) -> LLMConfig:
        llm_provider = os.environ.get("LLM_PROVIDER")

        temperature = float(os.environ.get("LLM_TEMPERATURE"))
        max_tokens = int(os.environ.get("LLM_MAX_TOKENS"))


        """获取当前 LLM 配置"""
        if llm_provider == "openai":

            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            model_name = os.environ.get("OPENAI_MODEL")

            return LLMConfig(
                provider=llm_provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url,
            )

        elif llm_provider == "anthropic":
            return LLMConfig(
                provider=llm_provider,
                model_name=os.environ.get("ANTHROPIC_MODEL_NAME", self.anthropic_model),
                temperature=float(os.environ.get("LLM_TEMPERATURE", self.llm_temperature)),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", self.llm_max_tokens)),
                api_key=os.environ.get("ANTHROPIC_API_KEY", self.anthropic_api_key),
            )
        else:  # local
            return LLMConfig(
                provider=llm_provider,
                model_name=os.environ.get("LOCAL_MODEL_NAME", self.local_model_name),
                temperature=float(os.environ.get("LLM_TEMPERATURE", self.llm_temperature)),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", self.llm_max_tokens)),
                base_url=os.environ.get("LOCAL_MODEL_URL", self.local_model_url),
            )
    
    def is_agent_enabled(self, agent_name: str) -> bool:
        """检查 Agent 是否启用"""
        if agent_name in self.agents:
            return self.agents[agent_name].enabled
        return True
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """获取指定 Agent 的配置"""
        return self.agents.get(agent_name)


# 你可以用下面命令在本机快速查看加载结果：
# python -c 'from src.config.settings import reload_settings; s=reload_settings(); print("API_KEY=", s.openai_api_key, "BASE_URL=", s.openai_base_url, "PROVIDER=", s.llm_provider)'

@lru_cache()
def get_settings() -> Settings:
    """
    获取全局配置实例（单例模式）
    
    Returns:
        Settings: 配置实例
    """
    settings = Settings()

    return settings


def reload_settings() -> Settings:
    """
    重新加载配置（清除缓存）
    
    Returns:
        Settings: 新的配置实例
    """
    get_settings.cache_clear()
    return get_settings()