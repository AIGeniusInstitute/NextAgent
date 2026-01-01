"""
工具调用测试
============

测试各种工具的功能和安全性。
"""

import pytest
import json
from pathlib import Path

from src.tools.calculator import SafeCalculator, calculator_tool
from src.tools.file_manager import FileManager, file_manager_tool
from src.tools.code_executor import CodeExecutor, code_executor_tool
from src.tools.search import WebSearch, web_search_tool


class TestSafeCalculator:
    """安全计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        return SafeCalculator()
    
    def test_basic_arithmetic(self, calculator):
        """测试基本算术"""
        assert calculator.calculate("2 + 3") == 5
        assert calculator.calculate("10 - 4") == 6
        assert calculator.calculate("3 * 4") == 12
        assert calculator.calculate("15 / 3") == 5.0
    
    def test_complex_expression(self, calculator):
        """测试复杂表达式"""
        assert calculator.calculate("(2 + 3) * 4") == 20
        assert calculator.calculate("2 ** 10") == 1024
        assert calculator.calculate("17 % 5") == 2
    
    def test_math_functions(self, calculator):
        """测试数学函数"""
        assert calculator.calculate("sqrt(16)") == 4.0
        assert calculator.calculate("abs(-5)") == 5
        assert calculator.calculate("max(1, 2, 3)") == 3
        assert calculator.calculate("min(1, 2, 3)") == 1
    
    def test_constants(self, calculator):
        """测试常量"""
        result = calculator.calculate("pi")
        assert 3.14 < result < 3.15
    
    def test_invalid_expression(self, calculator):
        """测试无效表达式"""
        with pytest.raises(ValueError):
            calculator.calculate("invalid")
    
    def test_forbidden_operations(self, calculator):
        """测试禁止的操作"""
        with pytest.raises(ValueError):
            calculator.calculate("__import__('os')")
    
    def test_calculator_tool(self):
        """测试计算器工具接口"""
        result = calculator_tool.invoke({"expression": "2 + 2"})
        assert "4" in result


class TestFileManager:
    """文件管理器测试"""
    
    @pytest.fixture
    def file_manager(self, test_workspace):
        return FileManager(workspace_dir=str(test_workspace))
    
    def test_read_file(self, file_manager, test_workspace):
        """测试读取文件"""
        content = file_manager.read("test.txt")
        assert content == "Hello, World!"
    
    def test_write_file(self, file_manager, test_workspace):
        """测试写入文件"""
        file_manager.write("new_file.txt", "New content")
        
        assert (test_workspace / "new_file.txt").exists()
        assert (test_workspace / "new_file.txt").read_text() == "New content"
    
    def test_list_dir(self, file_manager):
        """测试列出目录"""
        items = file_manager.list_dir(".")
        
        assert len(items) >= 2
        names = [item["name"] for item in items]
        assert "test.txt" in names
    
    def test_file_exists(self, file_manager):
        """测试检查文件存在"""
        assert file_manager.exists("test.txt") is True
        assert file_manager.exists("nonexistent.txt") is False
    
    def test_path_traversal_blocked(self, file_manager):
        """测试路径遍历被阻止"""
        with pytest.raises(ValueError):
            file_manager.read("../../../etc/passwd")
    
    def test_absolute_path_blocked(self, file_manager):
        """测试绝对路径被阻止"""
        with pytest.raises(ValueError):
            file_manager.read("/etc/passwd")
    
    def test_file_manager_tool(self, test_workspace):
        """测试文件管理器工具接口"""
        with pytest.MonkeyPatch().context() as mp:
            # 需要临时修改工作目录
            result = file_manager_tool.invoke({
                "action": "exists",
                "path": "test.txt"
            })
            # 结果应该包含存在或不存在的信息
            assert "存在" in result or "不存在" in result


class TestCodeExecutor:
    """代码执行器测试"""
    
    @pytest.fixture
    def executor(self):
        return CodeExecutor(timeout=5)
    
    def test_simple_code_execution(self, executor):
        """测试简单代码执行"""
        result = executor.execute("print('Hello')")
        
        assert result["success"] is True
        assert "Hello" in result["output"]
    
    def test_expression_evaluation(self, executor):
        """测试表达式求值"""
        result = executor.execute("2 + 2")
        
        assert result["success"] is True
        assert result["result"] == 4
    
    def test_math_operations(self, executor):
        """测试数学运算"""
        code = """
import math
result = math.sqrt(16) + math.pi
print(f"Result: {result}")
"""
        result = executor.execute(code)
        
        assert result["success"] is True
        assert "Result:" in result["output"]
    
    def test_forbidden_import(self, executor):
        """测试禁止的导入"""
        result = executor.execute("import os")
        
        assert result["success"] is False
        assert "禁止" in result["error"]
    
    def test_forbidden_operations(self, executor):
        """测试禁止的操作"""
        result = executor.execute("__import__('subprocess')")
        
        assert result["success"] is False
    
    def test_timeout(self, executor):
        """测试超时"""
        code = """
import time
time.sleep(10)
"""
        # 这个测试可能需要较长时间
        result = executor.execute(code, timeout=1)
        
        # 应该失败或超时
        # 注意：time 模块可能不在允许列表中
        assert result["success"] is False
    
    def test_code_executor_tool(self):
        """测试代码执行器工具接口"""
        result = code_executor_tool.invoke({
            "code": "print(1 + 1)"
        })
        assert "2" in result


class TestWebSearch:
    """网络搜索测试"""
    
    @pytest.fixture
    def search(self):
        return WebSearch()
    
    def test_search_returns_results(self, search):
        """测试搜索返回结果"""
        results = search.search("python")
        
        assert len(results) > 0
        assert all(hasattr(r, "title") for r in results)
        assert all(hasattr(r, "url") for r in results)
    
    def test_search_with_limit(self, search):
        """测试搜索结果数量限制"""
        results = search.search("python", num_results=3)
        
        assert len(results) <= 3
    
    def test_format_results(self, search):
        """测试结果格式化"""
        results = search.search("python", num_results=2)
        formatted = search.format_results(results)
        
        assert "找到" in formatted
        assert "条结果" in formatted
    
    def test_empty_results_handling(self, search):
        """测试空结果处理"""
        formatted = search.format_results([])
        
        assert "未找到" in formatted
    
    def test_web_search_tool(self):
        """测试搜索工具接口"""
        result = web_search_tool.invoke({
            "query": "python programming"
        })
        assert "结果" in result


class TestToolIntegration:
    """工具集成测试"""
    
    def test_all_tools_have_names(self):
        """测试所有工具都有名称"""
        from src.tools import get_all_tools
        
        tools = get_all_tools()
        
        for tool in tools:
            assert tool.name is not None
            assert len(tool.name) > 0
    
    def test_all_tools_have_descriptions(self):
        """测试所有工具都有描述"""
        from src.tools import get_all_tools
        
        tools = get_all_tools()
        
        for tool in tools:
            assert tool.description is not None
            assert len(tool.description) > 0
    
    def test_get_tool_by_name(self):
        """测试按名称获取工具"""
        from src.tools import get_tool_by_name
        
        calc = get_tool_by_name("calculator")
        assert calc.name == "calculator"
        
        with pytest.raises(ValueError):
            get_tool_by_name("nonexistent_tool")