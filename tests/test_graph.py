"""
图构建测试
==========

测试 LangGraph 状态图的构建和配置。
"""

import pytest
from unittest.mock import MagicMock, patch

from src.graph.builder import build_graph, MultiAgentSystem
from src.graph.state import create_initial_state, AgentState
from src.graph.nodes import (
    input_parser_node,
    coordinator_node,
    task_router_node,
)
from src.graph.edges import (
    route_from_coordinator,
    route_from_planner,
    route_task,
)


class TestGraphBuilder:
    """图构建器测试"""
    
    def test_build_graph_returns_state_graph(self, mock_settings):
        """测试图构建返回正确类型"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            graph = build_graph(mock_settings)
            
            assert graph is not None
            # StateGraph 应该有 nodes 属性
            assert hasattr(graph, "nodes")
    
    def test_graph_has_required_nodes(self, mock_settings):
        """测试图包含所有必需节点"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            graph = build_graph(mock_settings)
            
            required_nodes = [
                "input_parser",
                "coordinator",
                "planner",
                "task_router",
                "researcher",
                "coder",
                "executor",
                "critic",
                "human_node",
                "synthesizer",
                "error_handler",
            ]
            
            for node in required_nodes:
                assert node in graph.nodes, f"缺少节点: {node}"
    
    def test_graph_compilation(self, mock_settings):
        """测试图可以正常编译"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            graph = build_graph(mock_settings)
            compiled = graph.compile()
            
            assert compiled is not None
            assert hasattr(compiled, "invoke")


class TestMultiAgentSystem:
    """MultiAgentSystem 测试"""
    
    def test_system_initialization(self, mock_settings):
        """测试系统初始化"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            system = MultiAgentSystem(settings=mock_settings)
            
            assert system.settings == mock_settings
            assert system._graph is None  # 延迟加载
    
    def test_system_graph_lazy_loading(self, mock_settings, mock_llm):
        """测试图的延迟加载"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            with patch("src.llm.factory.LLMFactory.create", return_value=mock_llm):
                system = MultiAgentSystem(settings=mock_settings)
                
                # 访问 graph 属性触发加载
                _ = system.graph
                
                assert system._compiled_graph is not None
    
    def test_system_reset(self, mock_settings):
        """测试系统重置"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            system = MultiAgentSystem(settings=mock_settings)
            system._compiled_graph = MagicMock()
            
            system.reset()
            
            assert system._compiled_graph is None


class TestInitialState:
    """初始状态测试"""
    
    def test_create_initial_state(self):
        """测试创建初始状态"""
        state = create_initial_state(
            task="测试任务",
            task_id="test_123",
            max_iterations=5,
        )
        
        assert state["original_task"] == "测试任务"
        assert state["task_id"] == "test_123"
        assert state["max_iterations"] == 5
        assert state["iteration_count"] == 0
        assert state["final_answer"] is None
        assert len(state["messages"]) == 1
    
    def test_initial_state_has_required_fields(self):
        """测试初始状态包含必需字段"""
        state = create_initial_state(task="测试")
        
        required_fields = [
            "messages",
            "original_task",
            "subtasks",
            "agent_outputs",
            "tool_call_logs",
            "iteration_count",
            "max_iterations",
            "final_answer",
            "reasoning_trace",
            "error_log",
        ]
        
        for field in required_fields:
            assert field in state, f"缺少字段: {field}"


class TestEdgeRouting:
    """边路由测试"""
    
    def test_route_from_coordinator_to_planner(self, sample_state):
        """测试协调者到规划者的路由"""
        state = dict(sample_state)
        state["next"] = "planner"
        state["task_understanding"] = "需要规划的任务"
        
        result = route_from_coordinator(state)
        
        assert result == "planner"
    
    def test_route_from_coordinator_to_end(self, sample_state):
        """测试协调者到结束的路由"""
        state = dict(sample_state)
        state["final_answer"] = "任务已完成"
        
        result = route_from_coordinator(state)
        
        assert result == "end"
    
    def test_route_task_to_researcher(self, sample_state, sample_subtasks):
        """测试任务路由到研究员"""
        state = dict(sample_state)
        state["subtasks"] = sample_subtasks
        state["next"] = "researcher"
        
        result = route_task(state)
        
        assert result == "researcher"
    
    def test_route_task_to_synthesizer_when_no_pending(self, sample_state):
        """测试无待处理任务时路由到综合者"""
        state = dict(sample_state)
        state["subtasks"] = []
        state["next"] = "synthesizer"
        
        result = route_task(state)
        
        assert result == "synthesizer"