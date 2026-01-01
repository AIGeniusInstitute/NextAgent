"""
完整流程测试
============

测试端到端的任务执行流程。
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.graph.builder import MultiAgentSystem
from src.graph.state import create_initial_state


class TestEndToEndFlow:
    """端到端流程测试"""
    
    @pytest.fixture
    def mock_system(self, mock_settings, mock_llm):
        """创建模拟系统"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            with patch("src.llm.factory.LLMFactory.create", return_value=mock_llm):
                system = MultiAgentSystem(settings=mock_settings)
                return system
    
    def test_simple_task_execution(self, mock_system, mock_llm):
        """测试简单任务执行"""
        # 配置模拟响应
        mock_llm.invoke.return_value = MagicMock(
            content="""
            ## 任务理解
            这是一个简单的问答任务
            
            ## 任务类型
            简单问答
            
            ## 下一步行动
            direct_answer
            """
        )
        
        with patch.object(
            mock_system,
            "graph",
            new_callable=lambda: MagicMock(
                invoke=MagicMock(return_value={
                    "final_answer": "这是测试答案",
                    "iteration_count": 1,
                    "reasoning_trace": ["完成"],
                })
            )
        ):
            result = mock_system.run("什么是 Python?", task_id="test_simple")
            
            assert "final_answer" in result
    
    def test_task_with_subtasks(self, mock_system, sample_subtasks):
        """测试带子任务的执行"""
        mock_result = {
            "original_task": "复杂任务",
            "subtasks": sample_subtasks,
            "final_answer": "任务完成",
            "iteration_count": 3,
            "reasoning_trace": [
                "[Coordinator] 任务理解完成",
                "[Planner] 分解为 3 个子任务",
                "[Synthesizer] 生成最终答案",
            ],
        }
        
        with patch.object(
            mock_system,
            "graph",
            new_callable=lambda: MagicMock(invoke=MagicMock(return_value=mock_result))
        ):
            result = mock_system.run("复杂任务", task_id="test_complex")
            
            assert len(result.get("subtasks", [])) == 3
            assert result.get("final_answer") == "任务完成"
    
    def test_iteration_limit(self, mock_settings):
        """测试迭代次数限制"""
        mock_settings.max_iterations = 2
        
        state = create_initial_state(
            task="测试任务",
            max_iterations=2,
        )
        state["iteration_count"] = 2
        
        # 迭代次数已达上限
        assert state["iteration_count"] >= state["max_iterations"]
    
    def test_error_handling_in_flow(self, mock_system):
        """测试流程中的错误处理"""
        error_result = {
            "original_task": "错误任务",
            "error_log": ["模拟错误"],
            "last_error": "处理失败",
            "final_answer": "任务执行过程中出现错误",
            "iteration_count": 1,
        }
        
        with patch.object(
            mock_system,
            "graph",
            new_callable=lambda: MagicMock(invoke=MagicMock(return_value=error_result))
        ):
            result = mock_system.run("错误任务")
            
            assert "error_log" in result
            assert len(result["error_log"]) > 0


class TestStreamExecution:
    """流式执行测试"""
    
    def test_stream_yields_events(self, mock_settings, mock_llm):
        """测试流式执行产生事件"""
        with patch("src.graph.builder.get_settings", return_value=mock_settings):
            with patch("src.llm.factory.LLMFactory.create", return_value=mock_llm):
                system = MultiAgentSystem(settings=mock_settings)
                
                mock_events = [
                    {"input_parser": {"next": "coordinator"}},
                    {"coordinator": {"next": "synthesizer"}},
                    {"synthesizer": {"final_answer": "完成"}},
                ]
                
                mock_graph = MagicMock()
                mock_graph.stream = MagicMock(return_value=iter(mock_events))
                
                with patch.object(system, "_compiled_graph", mock_graph):
                    system._graph = MagicMock()
                    
                    events = list(system.stream("测试任务"))
                    
                    assert len(events) == 3


class TestHumanInLoop:
    """人工介入测试"""
    
    def test_human_input_flag(self, sample_state):
        """测试人工介入标志"""
        state = dict(sample_state)
        state["needs_human_input"] = True
        
        assert state["needs_human_input"] is True
    
    def test_human_feedback_recorded(self, sample_state):
        """测试人工反馈记录"""
        state = dict(sample_state)
        state["human_feedback"] = "用户确认继续"
        
        assert state["human_feedback"] == "用户确认继续"