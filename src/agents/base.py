"""
Agent 基类模块
==============

定义所有 Agent 的基类和注册机制。
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from src.config.settings import Settings, AgentConfig, get_settings
from src.config.prompts import PromptTemplates
from src.utils.logger import get_logger
from src.types import AgentOutput, ToolCallLog


class AgentRegistry:
    """
    Agent 注册表
    
    用于管理和获取 Agent 实例的单例注册表。
    支持动态注册新的 Agent 类型。
    """
    
    _agents: Dict[str, Type["BaseAgent"]] = {}
    _instances: Dict[str, "BaseAgent"] = {}
    
    @classmethod
    def register(cls, name: str, agent_class: Type["BaseAgent"]) -> None:
        """
        注册 Agent 类
        
        Args:
            name: Agent 名称
            agent_class: Agent 类
        """
        cls._agents[name] = agent_class
    
    @classmethod
    def get_class(cls, name: str) -> Optional[Type["BaseAgent"]]:
        """
        获取 Agent 类
        
        Args:
            name: Agent 名称
            
        Returns:
            Agent 类，不存在返回 None
        """
        return cls._agents.get(name)
    
    @classmethod
    def get_instance(
        cls,
        name: str,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        settings: Optional[Settings] = None,
    ) -> Optional["BaseAgent"]:
        """
        获取 Agent 实例（单例模式）
        
        Args:
            name: Agent 名称
            llm: LLM 实例
            tools: 工具列表
            settings: 配置
            
        Returns:
            Agent 实例
        """
        if name not in cls._instances:
            agent_class = cls._agents.get(name)
            if agent_class is None:
                return None
            cls._instances[name] = agent_class(
                llm=llm,
                tools=tools,
                settings=settings,
            )
        return cls._instances[name]
    
    @classmethod
    def clear_instances(cls) -> None:
        """清空所有实例缓存"""
        cls._instances.clear()
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """
        列出所有已注册的 Agent
        
        Returns:
            Agent 名称列表
        """
        return list(cls._agents.keys())


def register_agent(name: str) -> Callable:
    """
    Agent 注册装饰器
    
    使用方式：
        @register_agent("my_agent")
        class MyAgent(BaseAgent):
            ...
    
    Args:
        name: Agent 名称
        
    Returns:
        装饰器函数
    """
    def decorator(cls: Type["BaseAgent"]) -> Type["BaseAgent"]:
        AgentRegistry.register(name, cls)
        return cls
    return decorator


class BaseAgent(ABC):
    """
    Agent 抽象基类
    
    所有专业智能体都应继承此类并实现 _execute 方法。
    
    属性:
        name: Agent 名称
        description: Agent 描述
        llm: 语言模型实例
        tools: 可用工具列表
        settings: 系统配置
        logger: 日志记录器
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[BaseTool]] = None,
        settings: Optional[Settings] = None,
    ):
        """
        初始化 Agent
        
        Args:
            llm: 语言模型实例，None 时从工厂创建
            tools: 工具列表
            settings: 配置实例
        """
        self.settings = settings or get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self._tools = tools or []
        self._llm = llm
        self._tool_call_logs: List[ToolCallLog] = []
        
        # 获取 Agent 特定配置
        self._config = self.settings.get_agent_config(self.name)
        
        self.logger.debug(f"初始化 Agent: {self.name}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Agent 描述"""
        pass
    
    @property
    def capabilities(self) -> List[str]:
        """Agent 能力列表"""
        return []
    
    @property
    def llm(self) -> BaseChatModel:
        """获取 LLM 实例"""
        if self._llm is None:
            from src.llm.factory import LLMFactory
            self._llm = LLMFactory.create(self.settings.get_llm_config())
        return self._llm
    
    @property
    def tools(self) -> List[BaseTool]:
        """获取工具列表"""
        return self._tools
    
    @tools.setter
    def tools(self, tools: List[BaseTool]) -> None:
        """设置工具列表"""
        self._tools = tools
    
    def get_system_prompt(self) -> str:
        """
        获取系统提示词
        
        子类可以重写此方法提供自定义系统提示词。
        
        Returns:
            系统提示词字符串
        """
        template_name = f"{self.name.upper()}_SYSTEM"
        try:
            return PromptTemplates.get(template_name)
        except ValueError:
            return f"你是一个专业的 {self.name} 智能体。"
    
    def create_prompt(self, **kwargs) -> ChatPromptTemplate:
        """
        创建聊天提示模板
        
        Args:
            **kwargs: 额外的模板变量
            
        Returns:
            ChatPromptTemplate 实例
        """
        system_prompt = self.get_system_prompt()
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}"),
        ])
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 Agent 逻辑
        
        这是主要的公共接口，负责：
        1. 记录执行开始
        2. 调用子类实现的 _execute 方法
        3. 记录执行结果和耗时
        4. 处理异常
        
        Args:
            state: 当前系统状态
            
        Returns:
            更新后的状态字典
        """
        start_time = time.time()
        self._tool_call_logs = []
        
        self.logger.info(f"[{self.name}] 开始执行")
        
        if self.settings.debug_mode:
            self._log_state_summary(state, "输入")
        
        try:
            # 调用子类实现
            result_state = self._execute(state)
            
            # 记录执行时间
            duration = time.time() - start_time
            execution_time = result_state.get("execution_time", {})
            execution_time[self.name] = duration
            result_state["execution_time"] = execution_time
            
            # 添加工具调用日志
            if self._tool_call_logs:
                tool_logs = result_state.get("tool_call_logs", [])
                tool_logs.extend(self._tool_call_logs)
                result_state["tool_call_logs"] = tool_logs
            
            # 更新当前 Agent
            result_state["current_agent"] = self.name
            
            self.logger.info(f"[{self.name}] 执行完成，耗时 {duration:.2f}s")
            
            if self.settings.debug_mode:
                self._log_state_summary(result_state, "输出")
            
            return result_state
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"[{self.name}] 执行失败: {e}", exc_info=True)
            
            # 记录错误
            error_log = state.get("error_log", [])
            error_log.append(f"[{self.name}] {datetime.now().isoformat()}: {str(e)}")
            
            return {
                **state,
                "current_agent": self.name,
                "last_error": str(e),
                "error_log": error_log,
                "execution_time": {
                    **state.get("execution_time", {}),
                    self.name: duration,
                },
            }
    
    @abstractmethod
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行具体的 Agent 逻辑
        
        子类必须实现此方法。
        
        Args:
            state: 当前系统状态
            
        Returns:
            更新后的状态字典
        """
        pass
    
    def can_handle(self, task_type: str) -> bool:
        """
        判断是否能处理指定类型的任务
        
        Args:
            task_type: 任务类型
            
        Returns:
            是否能处理
        """
        return task_type in self.capabilities
    
    def call_llm(
        self,
        messages: List[BaseMessage],
        system_prompt: Optional[str] = None,
    ) -> AIMessage:
        """
        调用 LLM
        
        Args:
            messages: 消息列表
            system_prompt: 可选的系统提示词
            
        Returns:
            AI 响应消息
        """
        if system_prompt:
            full_messages = [SystemMessage(content=system_prompt)] + messages
        else:
            full_messages = [SystemMessage(content=self.get_system_prompt())] + messages
        
        response = self.llm.invoke(full_messages)
        
        # 记录 token 使用（如果可用）
        if hasattr(response, "usage_metadata"):
            self.logger.debug(f"Token 使用: {response.usage_metadata}")
        
        return response
    
    def call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            **kwargs: 工具参数
            
        Returns:
            工具执行结果
        """
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if tool is None:
            raise ValueError(f"工具不存在: {tool_name}")
        
        start_time = time.time()
        try:
            result = tool.invoke(kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # 记录工具调用
            self._tool_call_logs.append(ToolCallLog(
                tool_name=tool_name,
                input_params=kwargs,
                output=result,
                success=True,
                duration_ms=duration_ms,
            ))
            
            self.logger.debug(f"工具调用成功: {tool_name}, 耗时 {duration_ms:.2f}ms")
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self._tool_call_logs.append(ToolCallLog(
                tool_name=tool_name,
                input_params=kwargs,
                output=None,
                success=False,
                error_message=str(e),
                duration_ms=duration_ms,
            ))
            
            self.logger.error(f"工具调用失败: {tool_name}, 错误: {e}")
            raise
    
    def create_output(
        self,
        output: str,
        reasoning: str = "",
        task_id: Optional[str] = None,
        confidence: float = 0.8,
    ) -> AgentOutput:
        """
        创建标准化的 Agent 输出
        
        Args:
            output: 输出内容
            reasoning: 推理过程
            task_id: 任务 ID
            confidence: 置信度
            
        Returns:
            AgentOutput 实例
        """
        return AgentOutput(
            agent_name=self.name,
            task_id=task_id,
            output=output,
            reasoning=reasoning,
            tool_calls=self._tool_call_logs.copy(),
            confidence=confidence,
        )
    
    def _log_state_summary(self, state: Dict[str, Any], prefix: str) -> None:
        """
        记录状态摘要（调试用）
        
        Args:
            state: 状态字典
            prefix: 日志前缀
        """
        summary = {
            "original_task": state.get("original_task", "")[:100],
            "current_agent": state.get("current_agent"),
            "iteration_count": state.get("iteration_count", 0),
            "subtasks_count": len(state.get("subtasks", [])),
            "has_final_answer": state.get("final_answer") is not None,
        }
        self.logger.debug(f"[{prefix}状态] {summary}")
    
    def _get_pending_subtasks(self, state: Dict[str, Any]) -> List[Dict]:
        """
        获取待处理的子任务
        
        Args:
            state: 状态字典
            
        Returns:
            待处理子任务列表
        """
        subtasks = state.get("subtasks", [])
        return [t for t in subtasks if t.get("status") == "pending"]
    
    def _get_completed_subtasks(self, state: Dict[str, Any]) -> List[Dict]:
        """
        获取已完成的子任务
        
        Args:
            state: 状态字典
            
        Returns:
            已完成子任务列表
        """
        subtasks = state.get("subtasks", [])
        return [t for t in subtasks if t.get("status") == "completed"]
    
    def _update_subtask_status(
        self,
        state: Dict[str, Any],
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> List[Dict]:
        """
        更新子任务状态
        
        Args:
            state: 状态字典
            task_id: 任务 ID
            status: 新状态
            result: 执行结果
            error: 错误信息
            
        Returns:
            更新后的子任务列表
        """
        subtasks = state.get("subtasks", []).copy()
        for task in subtasks:
            if task.get("id") == task_id:
                task["status"] = status
                task["updated_at"] = datetime.now().isoformat()
                if result:
                    task["result"] = result
                if error:
                    task["error_message"] = error
                    task["retry_count"] = task.get("retry_count", 0) + 1
                break
        return subtasks