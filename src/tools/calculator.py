"""
计算器工具
==========

提供安全的数学表达式计算功能。
"""

import ast
import operator
from typing import Any, Dict, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CalculatorInput(BaseModel):
    """计算器输入参数"""
    expression: str = Field(
        description="要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'"
    )


class SafeCalculator:
    """
    安全计算器
    
    使用 AST 解析来安全执行数学表达式，避免代码注入风险。
    
    支持的运算：
    - 基本运算：+, -, *, /, //, %, **
    - 比较运算：<, >, <=, >=, ==, !=
    - 数学函数：abs, round, min, max, sum, pow, sqrt
    - 常量：pi, e
    """
    
    # 允许的运算符
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # 允许的比较运算符
    ALLOWED_COMPARISONS = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
    }
    
    # 允许的函数
    ALLOWED_FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'sqrt': lambda x: x ** 0.5,
        'int': int,
        'float': float,
    }
    
    # 允许的常量
    ALLOWED_CONSTANTS = {
        'pi': 3.141592653589793,
        'e': 2.718281828459045,
        'True': True,
        'False': False,
    }
    
    def __init__(self):
        """初始化计算器"""
        self.logger = get_logger(self.__class__.__name__)
    
    def calculate(self, expression: str) -> Union[float, int, bool, str]:
        """
        安全计算数学表达式
        
        Args:
            expression: 数学表达式字符串
            
        Returns:
            计算结果
            
        Raises:
            ValueError: 表达式无效或包含不允许的操作
        """
        self.logger.debug(f"计算表达式: {expression}")
        
        # 清理表达式
        expression = expression.strip()
        
        if not expression:
            raise ValueError("表达式不能为空")
        
        try:
            # 解析表达式为 AST
            tree = ast.parse(expression, mode='eval')
            
            # 验证并计算
            result = self._eval_node(tree.body)
            
            self.logger.debug(f"计算结果: {result}")
            return result
            
        except SyntaxError as e:
            raise ValueError(f"表达式语法错误: {e}")
        except Exception as e:
            raise ValueError(f"计算错误: {e}")
    
    def _eval_node(self, node: ast.AST) -> Any:
        """
        递归计算 AST 节点
        
        Args:
            node: AST 节点
            
        Returns:
            节点计算结果
        """
        # 数字字面量
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)):
                return node.value
            raise ValueError(f"不支持的常量类型: {type(node.value)}")
        
        # 变量名（常量）
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.ALLOWED_CONSTANTS:
                return self.ALLOWED_CONSTANTS[name]
            raise ValueError(f"未知的标识符: {name}")
        
        # 一元运算
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self.ALLOWED_OPERATORS[op_type](operand)
        
        # 二元运算
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"不支持的二元运算符: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.ALLOWED_OPERATORS[op_type](left, right)
        
        # 比较运算
        if isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                op_type = type(op)
                if op_type not in self.ALLOWED_COMPARISONS:
                    raise ValueError(f"不支持的比较运算符: {op_type.__name__}")
                right = self._eval_node(comparator)
                if not self.ALLOWED_COMPARISONS[op_type](left, right):
                    return False
                left = right
            return True
        
        # 函数调用
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("不支持的函数调用方式")
            
            func_name = node.func.id
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"不支持的函数: {func_name}")
            
            # 计算参数
            args = [self._eval_node(arg) for arg in node.args]
            
            # 调用函数
            return self.ALLOWED_FUNCTIONS[func_name](*args)
        
        # 列表/元组
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(elem) for elem in node.elts]
        
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")


# 创建工具实例
_calculator = SafeCalculator()


@tool(args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    """
    安全计算数学表达式。
    
    支持基本运算（+, -, *, /, **, %）和常用函数（sqrt, abs, round, min, max）。
    
    使用示例：
    - 基本运算: "2 + 3 * 4"
    - 幂运算: "2 ** 10"
    - 函数: "sqrt(16)"
    - 常量: "pi * 2"
    """
    try:
        result = _calculator.calculate(expression)
        return f"计算结果: {expression} = {result}"
    except ValueError as e:
        return f"计算错误: {str(e)}"
    except Exception as e:
        logger.error(f"计算器异常: {e}", exc_info=True)
        return f"计算失败: {str(e)}"