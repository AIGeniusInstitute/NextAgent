"""
图构建器模块
============

构建 LangGraph 状态图，组装所有节点和边。

这是系统的核心组装点，将所有组件连接成可执行的工作流。
"""

import time
from typing import Any, Dict, List, Optional, Generator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import AgentState, create_initial_state
from src.graph.nodes import (
    input_parser_node,
    coordinator_node,
    planner_node,
    task_router_node,
    researcher_node,
    coder_node,
    executor_node,
    critic_node,
    human_node,
    synthesizer_node,
    error_handler_node,
    clear_agent_cache,
)
from src.graph.edges import (
    route_from_coordinator,
    route_from_planner,
    route_task,
    route_from_worker,
    route_from_critic,
    route_from_human,
    route_from_synthesizer,
    route_from_error_handler,
)
from src.config.settings import Settings, get_settings
from src.utils.logger import get_logger
from src.types import FinalResult, ExecutionMetrics

logger = get_logger(__name__)


def build_graph(settings: Optional[Settings] = None) -> StateGraph:
    """
    构建 LangGraph 状态图
    
    这个函数创建并配置整个多智能体工作流：
    
    1. 创建状态图实例
    2. 添加所有处理节点
    3. 配置入口点
    4. 定义节点之间的边和条件边
    5. 返回未编译的图（需要调用 compile() 编译）
    
    图结构：
    ```
    input_parser -> coordinator -> planner -> task_router
                                      ^            |
                                      |      +-----+-----+-----+
                                      |      v     v     v     v
                                      +-- researcher coder executor
                                              |     |     |
                                              +-----+-----+
                                                    |
                                                    v
                                                 critic
                                                    |
                                      +-------------+-------------+
                                      v                           v
                                  human_node               synthesizer -> END
                                      |                           ^
                                      +---------------------------+
    ```
    
    Args:
        settings: 系统配置，None 使用默认配置
        
    Returns:
        配置好的 StateGraph 实例
    """
    if settings is None:
        settings = get_settings()
    
    logger.info("开始构建状态图...")
    
    # 创建状态图
    # AgentState 定义了图中流转的状态结构
    graph = StateGraph(AgentState)
    
    # ===================================================
    # 添加节点
    # 每个节点对应一个处理函数，接收状态并返回状态更新
    # ===================================================
    
    # 输入解析节点：处理和标准化用户输入
    graph.add_node("input_parser", input_parser_node)
    
    # 协调者节点：理解任务、决定执行策略
    graph.add_node("coordinator", coordinator_node)
    
    # 规划者节点：分解任务、制定执行计划
    graph.add_node("planner", planner_node)
    
    # 任务路由节点：根据任务类型分配给对应 Agent
    graph.add_node("task_router", task_router_node)
    
    # 工作者节点
    graph.add_node("researcher", researcher_node)  # 研究员：信息检索
    graph.add_node("coder", coder_node)            # 编码者：代码生成
    graph.add_node("executor", executor_node)      # 执行者：工具调用
    
    # 审核者节点：质量评估和反馈
    graph.add_node("critic", critic_node)
    
    # 人工介入节点：需要时暂停等待人工输入
    graph.add_node("human_node", human_node)
    
    # 综合者节点：汇总结果、生成最终答案
    graph.add_node("synthesizer", synthesizer_node)
    
    # 错误处理节点：处理执行异常
    graph.add_node("error_handler", error_handler_node)
    
    # ===================================================
    # 设置入口点
    # ===================================================
    
    # 图的入口是 input_parser 节点
    graph.set_entry_point("input_parser")
    
    # ===================================================
    # 定义边
    # 边定义了节点之间的转换关系
    # ===================================================
    
    # input_parser -> coordinator（固定边）
    graph.add_edge("input_parser", "coordinator")
    
    # coordinator 之后的条件路由
    graph.add_conditional_edges(
        "coordinator",
        route_from_coordinator,
        {
            "planner": "planner",
            "task_router": "task_router",
            "synthesizer": "synthesizer",
            "error_handler": "error_handler",
            "end": END,
        }
    )
    
    # planner 之后的条件路由
    graph.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "task_router": "task_router",
            "synthesizer": "synthesizer",
            "coordinator": "coordinator",
        }
    )
    
    # task_router 之后的条件路由
    graph.add_conditional_edges(
        "task_router",
        route_task,
        {
            "researcher": "researcher",
            "coder": "coder",
            "executor": "executor",
            "synthesizer": "synthesizer",
            "coordinator": "coordinator",
        }
    )
    
    # 工作者节点 -> critic（通过 route_from_worker）
    for worker in ["researcher", "coder", "executor"]:
        graph.add_conditional_edges(
            worker,
            route_from_worker,
            {
                "critic": "critic",
                "error_handler": "error_handler",
                "synthesizer": "synthesizer",
                "coordinator": "coordinator",
            }
        )
    
    # critic 之后的条件路由
    graph.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "human_node": "human_node",
            "task_router": "task_router",
            "synthesizer": "synthesizer",
            "coordinator": "coordinator",
        }
    )
    
    # human_node 之后的条件路由
    graph.add_conditional_edges(
        "human_node",
        route_from_human,
        {
            "task_router": "task_router",
            "synthesizer": "synthesizer",
            "coordinator": "coordinator",
            "end": END,
        }
    )
    
    # synthesizer 之后的条件路由
    graph.add_conditional_edges(
        "synthesizer",
        route_from_synthesizer,
        {
            "coordinator": "coordinator",
            "end": END,
        }
    )
    
    # error_handler 之后的条件路由
    graph.add_conditional_edges(
        "error_handler",
        route_from_error_handler,
        {
            "coordinator": "coordinator",
            "task_router": "task_router",
            "end": END,
        }
    )
    
    logger.info("状态图构建完成")
    
    return graph


class MultiAgentSystem:
    """
    多智能体系统封装类
    
    提供简化的接口来运行多智能体工作流。
    
    使用示例：
        >>> system = MultiAgentSystem()
        >>> result = system.run("编写一个 Python 爬虫")
        >>> print(result["final_answer"])
    
    属性：
        settings: 系统配置
        graph: 编译后的状态图
        checkpointer: 状态检查点管理器（用于持久化）
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        enable_checkpointing: bool = False,
    ):
        """
        初始化多智能体系统
        
        Args:
            settings: 系统配置，None 使用默认配置
            enable_checkpointing: 是否启用状态检查点
        """
        self.settings = settings or get_settings()
        self._graph = None
        self._compiled_graph = None
        self.checkpointer = MemorySaver() if enable_checkpointing else None
        
        logger.info("初始化 MultiAgentSystem")
    
    @property
    def graph(self):
        """获取编译后的图（延迟加载）"""
        if self._compiled_graph is None:
            self._build_and_compile()
        return self._compiled_graph
    
    def _build_and_compile(self):
        """构建并编译图"""
        self._graph = build_graph(self.settings)
        
        # 编译图
        # compile() 将 StateGraph 转换为可执行的 CompiledGraph
        # checkpointer 参数用于状态持久化，支持断点续传
        compile_kwargs = {}
        if self.checkpointer:
            compile_kwargs["checkpointer"] = self.checkpointer
        
        self._compiled_graph = self._graph.compile(**compile_kwargs)
        
        logger.info("图编译完成")
    
    def run(
        self,
        task: str,
        task_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        运行任务
        
        Args:
            task: 用户任务描述
            task_id: 任务 ID，不提供则自动生成
            config: 运行时配置
            
        Returns:
            最终状态字典，包含 final_answer 等结果
        """
        logger.info(f"开始执行任务: {task[:50]}...")
        
        # 创建初始状态
        initial_state = create_initial_state(
            task=task,
            task_id=task_id,
            max_iterations=self.settings.max_iterations,
        )
        
        # 配置
        run_config = config or {}
        if self.checkpointer and "configurable" not in run_config:
            run_config["configurable"] = {"thread_id": task_id or "default"}
        if "recursion_limit" not in run_config:
            run_config["recursion_limit"] = self.settings.max_iterations
        
        try:
            final_state = self.graph.invoke(initial_state, config=run_config)
            
            # 计算执行指标
            end_time = time.time()
            start_time = final_state.get("start_time", end_time)
            
            final_state["total_duration"] = end_time - start_time
            
            logger.info(
                f"任务执行完成，耗时 {final_state['total_duration']:.2f}s"
            )
            
            return dict(final_state)
            
        except Exception as e:
            logger.error(f"任务执行失败: {e}", exc_info=True)
            raise
    
    def stream(
        self,
        task: str,
        task_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式运行任务
        
        逐步产出每个节点的执行结果。
        
        Args:
            task: 用户任务描述
            task_id: 任务 ID
            config: 运行时配置
            
        Yields:
            每个节点执行后的状态更新
        """
        logger.info(f"开始流式执行任务: {task[:50]}...")
        
        initial_state = create_initial_state(
            task=task,
            task_id=task_id,
            max_iterations=self.settings.max_iterations,
        )
        
        run_config = config or {}
        if self.checkpointer and "configurable" not in run_config:
            run_config["configurable"] = {"thread_id": task_id or "default"}
        if "recursion_limit" not in run_config:
            run_config["recursion_limit"] = self.settings.graph_recursion_limit
        
        for event in self.graph.stream(initial_state, config=run_config):
            yield event
    
    def get_graph_visualization(self) -> str:
        """
        获取图的 Mermaid 可视化表示
        
        Returns:
            Mermaid 格式的图描述
        """
        return self.graph.get_graph().draw_mermaid()
    
    def reset(self):
        """重置系统状态"""
        clear_agent_cache()
        self._compiled_graph = None
        self._graph = None
        logger.info("系统已重置")
    
    def create_result(self, state: Dict[str, Any]) -> FinalResult:
        """
        从状态创建最终结果对象
        
        Args:
            state: 最终状态
            
        Returns:
            FinalResult 实例
        """
        return FinalResult(
            task_id=state.get("task_id", ""),
            original_task=state.get("original_task", ""),
            answer=state.get("final_answer", ""),
            reasoning_trace=state.get("reasoning_trace", []),
            agent_outputs=state.get("agent_outputs", {}),
            metrics=ExecutionMetrics(
                total_duration_seconds=state.get("total_duration", 0),
                token_usage=state.get("token_usage", {}),
                agent_durations=state.get("execution_time", {}),
                iteration_count=state.get("iteration_count", 0),
                retry_count=sum(
                    t.get("retry_count", 0)
                    for t in state.get("subtasks", [])
                ),
                tool_call_count=len(state.get("tool_call_logs", [])),
                success=bool(state.get("final_answer")),
            ),
        )