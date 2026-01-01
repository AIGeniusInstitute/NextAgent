"""
代码执行工具
============

提供安全的 Python 代码执行环境。
"""

import sys
import io
import traceback
import signal
from typing import Any, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeExecutorInput(BaseModel):
    """代码执行器输入参数"""
    code: str = Field(
        description="要执行的 Python 代码"
    )
    timeout: Optional[int] = Field(
        default=30,
        description="执行超时时间（秒）",
        ge=1,
        le=300
    )


class TimeoutError(Exception):
    """超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("代码执行超时")


class CodeExecutor:
    """
    代码执行器
    
    在受限环境中执行 Python 代码。
    
    安全特性：
    - 执行超时限制
    - 输出长度限制
    - 禁止危险操作（通过黑名单）
    - 隔离的命名空间
    
    注意：
    - 这不是完全安全的沙箱
    - 生产环境建议使用 Docker 容器或其他隔离方案
    """
    
    # 禁止的模块和函数
    BLACKLIST = {
        "os.system",
        "os.popen",
        "os.spawn",
        "os.exec",
        "subprocess",
        "shutil.rmtree",
        "eval",
        "__import__",
        "compile",
        "exec",
        "open",  # 通过文件管理器访问
    }
    
    # 允许导入的模块
    ALLOWED_MODULES = {
        "math",
        "random",
        "datetime",
        "json",
        "re",
        "collections",
        "itertools",
        "functools",
        "string",
        "typing",
        "dataclasses",
        "enum",
        "copy",
        "operator",
        "statistics",
        "decimal",
        "fractions",
    }
    
    MAX_OUTPUT_LENGTH = 10000
    
    def __init__(self, timeout: int = 30):
        """
        初始化代码执行器
        
        Args:
            timeout: 默认超时时间（秒）
        """
        self.default_timeout = timeout
        self.logger = get_logger(self.__class__.__name__)
    
    def _check_code_safety(self, code: str) -> Optional[str]:
        """
        检查代码安全性
        
        Args:
            code: 要检查的代码
            
        Returns:
            如果有问题返回错误信息，否则返回 None
        """
        for forbidden in self.BLACKLIST:
            if forbidden in code:
                return f"禁止的操作: {forbidden}"
        
        # 检查危险的导入
        dangerous_imports = ["os", "sys", "subprocess", "shutil", "socket"]
        for imp in dangerous_imports:
            if f"import {imp}" in code or f"from {imp}" in code:
                return f"禁止导入危险模块: {imp}"
        
        return None
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """
        创建安全的全局命名空间
        
        Returns:
            受限的全局命名空间字典
        """
        import math
        import random
        import datetime
        import json
        import re
        import collections
        import itertools
        import functools
        
        safe_globals = {
            "__builtins__": {
                # 安全的内置函数
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "chr": chr,
                "dict": dict,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "format": format,
                "frozenset": frozenset,
                "getattr": getattr,
                "hasattr": hasattr,
                "hash": hash,
                "hex": hex,
                "int": int,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "iter": iter,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "next": next,
                "oct": oct,
                "ord": ord,
                "pow": pow,
                "print": print,
                "range": range,
                "repr": repr,
                "reversed": reversed,
                "round": round,
                "set": set,
                "slice": slice,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "type": type,
                "zip": zip,
                # 异常
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "StopIteration": StopIteration,
                # 其他
                "True": True,
                "False": False,
                "None": None,
            },
            # 允许的模块
            "math": math,
            "random": random,
            "datetime": datetime,
            "json": json,
            "re": re,
            "collections": collections,
            "itertools": itertools,
            "functools": functools,
        }
        
        return safe_globals
    
    def execute(
        self,
        code: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行 Python 代码
        
        Args:
            code: Python 代码字符串
            timeout: 超时时间（秒）
            
        Returns:
            执行结果字典，包含:
            - success: 是否成功
            - output: 标准输出
            - error: 错误信息（如有）
            - result: 最后一个表达式的值（如有）
        """
        timeout = timeout or self.default_timeout
        
        self.logger.info(f"执行代码 (超时: {timeout}s)")
        self.logger.debug(f"代码内容:\n{code[:500]}")
        
        # 安全检查
        safety_issue = self._check_code_safety(code)
        if safety_issue:
            return {
                "success": False,
                "output": "",
                "error": safety_issue,
                "result": None,
            }
        
        # 准备执行环境
        safe_globals = self._create_safe_globals()
        local_namespace = {}
        
        # 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = None
        error = None
        
        try:
            # 设置超时（仅 Unix 系统有效）
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
            
            # 执行代码
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                try:
                    # 尝试作为表达式执行
                    result = eval(code, safe_globals, local_namespace)
                except SyntaxError:
                    # 作为语句执行
                    exec(code, safe_globals, local_namespace)
                    # 检查是否有返回值变量
                    if "result" in local_namespace:
                        result = local_namespace["result"]
            
        except TimeoutError:
            error = f"执行超时（{timeout}秒）"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        finally:
            # 恢复信号处理器
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        # 获取输出
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        # 限制输出长度
        if len(stdout) > self.MAX_OUTPUT_LENGTH:
            stdout = stdout[:self.MAX_OUTPUT_LENGTH] + "\n...(输出被截断)"
        if len(stderr) > self.MAX_OUTPUT_LENGTH:
            stderr = stderr[:self.MAX_OUTPUT_LENGTH] + "\n...(输出被截断)"
        
        output = stdout
        if stderr:
            output += f"\n[stderr]\n{stderr}"
        
        return {
            "success": error is None,
            "output": output.strip(),
            "error": error,
            "result": result,
        }


# 创建工具实例
_code_executor = CodeExecutor()


@tool(args_schema=CodeExecutorInput)
def code_executor_tool(code: str, timeout: int = 30) -> str:
    """
    在安全环境中执行 Python 代码。
    
    支持的功能：
    - 基本 Python 语法
    - 常用标准库（math, random, datetime, json, re 等）
    - 打印输出
    
    限制：
    - 不能访问文件系统（使用 file_manager 工具）
    - 不能执行系统命令
    - 不能进行网络请求
    - 有执行时间限制
    
    使用示例：
    ```python
    import math
    result = math.sqrt(16)
    print(f"结果: {result}")
    ```
    """
    try:
        result = _code_executor.execute(code, timeout)
        
        if result["success"]:
            output_parts = []
            if result["output"]:
                output_parts.append(f"输出:\n{result['output']}")
            if result["result"] is not None:
                output_parts.append(f"返回值: {result['result']}")
            
            if output_parts:
                return "\n\n".join(output_parts)
            return "代码执行成功（无输出）"
        else:
            return f"执行错误:\n{result['error']}"
    
    except Exception as e:
        logger.error(f"代码执行器异常: {e}", exc_info=True)
        return f"执行失败: {str(e)}"