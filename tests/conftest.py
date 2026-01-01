"""
Pytest 配置文件
===============

定义测试固件和通用配置。
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Settings, LLMConfig
from src.graph.state import create_initial_state, AgentState


@pytest.fixture
def mock_settings():
    """创建测试用配置"""
    return Settings(
        llm_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        debug_mode=True,
        max_iterations=3,
        enable_human_in_loop=False,
        workspace_dir="test_workspace",
        log_dir="test_logs",
    )


@pytest.fixture
def mock_llm():
    """创建模拟 LLM"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content="这是一个模拟的 LLM 响应"
    )
    return mock


@pytest.fixture
def sample_state():
    """创建示例状态"""
    return create_initial_state(
        task="测试任务：编写一个简单的 Python 函数",
        task_id="test_001",
        max_iterations=5,
    )


@pytest.fixture
def sample_subtasks():
    """创建示例子任务列表"""
    return [
        {
            "id": "task_1",
            "name": "分析需求",
            "description": "分析任务需求",
            "task_type": "analyze",
            "assigned_agent": "researcher",
            "dependencies": [],
            "priority": "high",
            "status": "pending",
            "result": None,
            "error_message": None,
            "retry_count": 0,
        },
        {
            "id": "task_2",
            "name": "编写代码",
            "description": "根据需求编写代码",
            "task_type": "code",
            "assigned_agent": "coder",
            "dependencies": ["task_1"],
            "priority": "high",
            "status": "pending",
            "result": None,
            "error_message": None,
            "retry_count": 0,
        },
        {
            "id": "task_3",
            "name": "测试代码",
            "description": "测试生成的代码",
            "task_type": "execute",
            "assigned_agent": "executor",
            "dependencies": ["task_2"],
            "priority": "medium",
            "status": "pending",
            "result": None,
            "error_message": None,
            "retry_count": 0,
        },
    ]


@pytest.fixture
def test_workspace(tmp_path):
    """创建临时测试工作空间"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    
    # 创建一些测试文件
    (workspace / "test.txt").write_text("Hello, World!")
    (workspace / "data.json").write_text('{"key": "value"}')
    
    return workspace


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    """设置测试环境"""
    # 设置环境变量
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DEBUG_MODE", "true")
    
    # 创建临时目录
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    
    yield
    
    # 清理


@pytest.fixture
def mock_tool_response():
    """创建模拟工具响应"""
    return {
        "calculator": "计算结果: 42",
        "file_manager": "文件操作成功",
        "code_executor": "代码执行成功\n输出: Hello",
        "web_search": "找到 5 条结果",
    }