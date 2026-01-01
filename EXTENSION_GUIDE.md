# Multi-Agent System 扩展指南

本文档详细说明如何扩展 Multi-Agent System 的各个组件。

---

## 目录

1. [新增 Agent](#1-新增-agent)
2. [新增工具](#2-新增工具)
3. [新增工作流节点](#3-新增工作流节点)
4. [自定义提示词](#4-自定义提示词)
5. [自定义路由逻辑](#5-自定义路由逻辑)
6. [扩展记忆系统](#6-扩展记忆系统)
7. [对接企业级场景](#7-对接企业级场景)
8. [性能优化](#8-性能优化)

---

## 1. 新增 Agent

### 1.1 基本步骤

创建新 Agent 需要以下步骤：

1. 创建 Agent 类文件
2. 继承 `BaseAgent` 基类
3. 实现必要的属性和方法
4. 注册 Agent
5. 添加到图构建器
6. 配置路由

### 1.2 完整示例：创建数据分析 Agent

```python
# src/agents/data_analyst.py

"""
数据分析 Agent
==============

负责数据处理、统计分析和可视化建议。
"""

from typing import Any, Dict, List, Optional
import json
import re

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates
from src.utils.logger import get_logger

logger = get_logger(__name__)


# 添加提示词模板
DATA_ANALYST_SYSTEM = """你是一个专业的数据分析师（Data Analyst），擅长数据处理和统计分析。

你的核心能力：
1. **数据理解**：快速理解数据结构和含义
2. **统计分析**：执行描述性统计、相关性分析等
3. **数据清洗**：识别和处理异常值、缺失值
4. **可视化建议**：推荐合适的图表类型
5. **洞察提取**：从数据中提取有价值的洞察

分析原则：
- 始终验证数据质量
- 使用适当的统计方法
- 结论要有数据支撑
- 提供可操作的建议

可用工具：
- calculator: 数学计算
- code_executor: 执行 Python 数据分析代码
- file_manager: 读取数据文件

请提供详细的分析过程和结论。"""

DATA_ANALYST_TASK = """请对以下数据进行分析：

任务描述：$task
数据上下文：$context
分析要求：$requirements

请按以下步骤进行分析：
1. 数据概览：理解数据结构
2. 数据质量检查：检查缺失值、异常值
3. 统计分析：执行相关统计计算
4. 洞察提取：总结关键发现
5. 建议：提供可操作的建议

请输出结构化的分析报告。"""


@register_agent("data_analyst")
class DataAnalystAgent(BaseAgent):
    """
    数据分析师智能体
    
    核心职责：
    1. 理解和处理数据
    2. 执行统计分析
    3. 提取数据洞察
    4. 生成分析报告
    """
    
    @property
    def name(self) -> str:
        return "data_analyst"
    
    @property
    def description(self) -> str:
        return "数据分析师，负责数据处理、统计分析和洞察提取"
    
    @property
    def capabilities(self) -> List[str]:
        return ["analyze", "statistics", "data_processing", "visualization"]
    
    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return DATA_ANALYST_SYSTEM
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据分析任务
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        # 获取当前任务
        current_task = self._get_current_task(state)
        
        if current_task is None:
            self.logger.warning("没有找到待处理的分析任务")
            return {**state, "next": "coordinator"}
        
        task_description = current_task.get("description", "")
        context = self._build_context(state)
        requirements = current_task.get("metadata", {}).get(
            "requirements", "提供完整的数据分析"
        )
        
        # 构建分析提示
        prompt = DATA_ANALYST_TASK.replace(
            "$task", task_description
        ).replace(
            "$context", context
        ).replace(
            "$requirements", requirements
        )
        
        messages = [HumanMessage(content=prompt)]
        
        # 检查是否需要先读取数据
        data_content = self._load_data_if_needed(task_description, state)
        if data_content:
            prompt += f"\n\n数据内容：\n```\n{data_content[:2000]}\n```"
            messages = [HumanMessage(content=prompt)]
        
        # 调用 LLM 进行分析
        response = self.call_llm(messages)
        analysis_output = response.content
        
        # 如果需要执行代码进行计算
        code_results = self._execute_analysis_code(analysis_output)
        if code_results:
            analysis_output += f"\n\n**计算结果：**\n{code_results}"
        
        # 创建输出
        agent_output = self.create_output(
            output=analysis_output,
            reasoning=f"完成数据分析任务: {current_task.get('name', '')}",
            task_id=current_task.get("id"),
            confidence=0.85,
        )
        
        # 更新子任务状态
        subtasks = self._update_subtask_status(
            state,
            current_task["id"],
            status="completed",
            result=analysis_output,
        )
        
        # 更新 agent_outputs
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs[f"data_analyst_{current_task['id']}"] = agent_output.model_dump()
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[DataAnalyst] 完成分析任务 '{current_task.get('name', '')}'"
        )
        
        return {
            **state,
            "subtasks": subtasks,
            "agent_outputs": agent_outputs,
            "reasoning_trace": reasoning_trace,
            "next": "critic",
        }
    
    def _get_current_task(self, state: Dict[str, Any]) -> Optional[Dict]:
        """获取当前待处理的分析任务"""
        subtasks = state.get("subtasks", [])
        
        for task in subtasks:
            if (task.get("status") == "pending" and 
                task.get("assigned_agent") == "data_analyst"):
                dependencies = task.get("dependencies", [])
                if self._dependencies_satisfied(subtasks, dependencies):
                    return task
        
        # 也处理 analyze 类型的任务
        for task in subtasks:
            if (task.get("status") == "pending" and 
                task.get("task_type") == "analyze"):
                dependencies = task.get("dependencies", [])
                if self._dependencies_satisfied(subtasks, dependencies):
                    return task
        
        return None
    
    def _dependencies_satisfied(
        self,
        subtasks: List[Dict],
        dependencies: List[str]
    ) -> bool:
        """检查依赖是否满足"""
        if not dependencies:
            return True
        
        completed_ids = {
            t["id"] for t in subtasks
            if t.get("status") == "completed"
        }
        
        return all(dep in completed_ids for dep in dependencies)
    
    def _build_context(self, state: Dict[str, Any]) -> str:
        """构建分析上下文"""
        context_parts = []
        
        original_task = state.get("original_task", "")
        if original_task:
            context_parts.append(f"原始任务: {original_task}")
        
        # 获取之前的研究或数据结果
        agent_outputs = state.get("agent_outputs", {})
        for name, output in agent_outputs.items():
            if any(keyword in name.lower() for keyword in ["researcher", "executor"]):
                if isinstance(output, dict):
                    result = output.get("output", "")[:500]
                else:
                    result = str(output)[:500]
                context_parts.append(f"前置结果 ({name}): {result}")
        
        return "\n".join(context_parts)
    
    def _load_data_if_needed(
        self,
        task_description: str,
        state: Dict[str, Any]
    ) -> Optional[str]:
        """如果任务涉及文件数据，加载数据"""
        # 检查任务描述中是否提到文件
        file_patterns = [
            r'文件[：:]\s*(\S+)',
            r'(\S+\.csv)',
            r'(\S+\.json)',
            r'(\S+\.xlsx?)',
        ]
        
        for pattern in file_patterns:
            match = re.search(pattern, task_description)
            if match:
                filename = match.group(1)
                try:
                    return self.call_tool(
                        "file_manager",
                        action="read",
                        path=filename
                    )
                except Exception as e:
                    self.logger.warning(f"加载数据文件失败: {e}")
        
        return None
    
    def _execute_analysis_code(self, analysis_output: str) -> Optional[str]:
        """执行分析中的代码块"""
        # 提取代码块
        code_pattern = r'```python\s*([\s\S]*?)```'
        matches = re.findall(code_pattern, analysis_output)
        
        if not matches:
            return None
        
        results = []
        for code in matches[:2]:  # 最多执行2个代码块
            try:
                result = self.call_tool("code_executor", code=code)
                if result:
                    results.append(str(result))
            except Exception as e:
                self.logger.warning(f"代码执行失败: {e}")
        
        return "\n".join(results) if results else None
```

### 1.3 注册新 Agent

```python
# src/agents/__init__.py

from src.agents.data_analyst import DataAnalystAgent

__all__ = [
    # ... 现有 agents
    "DataAnalystAgent",
]

def get_all_agents() -> dict:
    return {
        # ... 现有 agents
        "data_analyst": DataAnalystAgent,
    }
```

### 1.4 添加到图构建器

```python
# src/graph/nodes.py

from src.agents import DataAnalystAgent

def data_analyst_node(state: AgentState) -> Dict[str, Any]:
    """数据分析师节点"""
    logger.info("[Node] data_analyst - 开始分析")
    
    agent = _get_agent(DataAnalystAgent)
    result = agent.invoke(dict(state))
    
    return result
```

```python
# src/graph/builder.py

from src.graph.nodes import data_analyst_node

def build_graph(settings):
    # ... 现有代码
    
    # 添加数据分析师节点
    graph.add_node("data_analyst", data_analyst_node)
    
    # 添加边
    graph.add_conditional_edges(
        "data_analyst",
        route_from_worker,
        {
            "critic": "critic",
            "error_handler": "error_handler",
            "synthesizer": "synthesizer",
        }
    )
    
    # 更新任务路由
    # 需要修改 route_task 函数以支持新的 agent
```

### 1.5 更新任务路由

```python
# src/graph/edges.py

def route_task(state: AgentState) -> RouteType:
    """任务路由逻辑"""
    next_node = state.get("next", "executor")
    
    # 有效的工作者节点（添加新的）
    worker_nodes = {"researcher", "coder", "executor", "data_analyst"}
    
    if next_node in worker_nodes:
        return next_node
    
    # ... 其他逻辑
```

---

## 2. 新增工具

### 2.1 基本步骤

1. 创建工具类文件
2. 定义输入参数 Schema
3. 实现工具逻辑
4. 使用 `@tool` 装饰器
5. 注册工具

### 2.2 完整示例：创建 HTTP 请求工具

```python
# src/tools/http_client.py

"""
HTTP 客户端工具
===============

提供安全的 HTTP 请求功能。
"""

import json
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HTTPClientInput(BaseModel):
    """HTTP 客户端输入参数"""
    
    method: Literal["GET", "POST", "PUT", "DELETE"] = Field(
        default="GET",
        description="HTTP 方法"
    )
    url: str = Field(
        description="请求 URL"
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="请求头"
    )
    params: Optional[Dict[str, str]] = Field(
        default=None,
        description="URL 查询参数"
    )
    body: Optional[str] = Field(
        default=None,
        description="请求体（JSON 字符串）"
    )
    timeout: int = Field(
        default=30,
        description="超时时间（秒）",
        ge=1,
        le=120
    )
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """验证 URL 安全性"""
        parsed = urlparse(v)
        
        # 只允许 http 和 https
        if parsed.scheme not in ("http", "https"):
            raise ValueError("只支持 http 和 https 协议")
        
        # 禁止访问内网地址
        blocked_hosts = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "10.",
            "172.16.",
            "192.168.",
        ]
        
        host = parsed.netloc.lower()
        for blocked in blocked_hosts:
            if host.startswith(blocked) or host == blocked.rstrip("."):
                raise ValueError(f"禁止访问内网地址: {host}")
        
        return v


class HTTPClient:
    """
    HTTP 客户端
    
    安全特性：
    - 禁止访问内网地址
    - 请求超时限制
    - 响应大小限制
    - 只允许 http/https 协议
    """
    
    MAX_RESPONSE_SIZE = 1024 * 1024  # 1MB
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        发送 HTTP 请求
        
        Args:
            method: HTTP 方法
            url: 请求 URL
            headers: 请求头
            params: 查询参数
            body: 请求体
            timeout: 超时时间
            
        Returns:
            响应信息字典
        """
        self.logger.info(f"HTTP {method} {url}")
        
        # 默认请求头
        default_headers = {
            "User-Agent": "MultiAgentSystem/1.0",
            "Accept": "application/json, text/html, */*",
        }
        
        if headers:
            default_headers.update(headers)
        
        # 解析请求体
        json_body = None
        if body:
            try:
                json_body = json.loads(body)
            except json.JSONDecodeError:
                # 作为普通文本发送
                pass
        
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=default_headers,
                    params=params,
                    json=json_body if json_body else None,
                    content=body if body and not json_body else None,
                )
                
                # 检查响应大小
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.MAX_RESPONSE_SIZE:
                    return {
                        "success": False,
                        "error": f"响应过大: {content_length} bytes",
                        "status_code": response.status_code,
                    }
                
                # 尝试解析 JSON
                try:
                    response_body = response.json()
                except:
                    response_body = response.text[:5000]  # 限制文本长度
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response_body,
                }
                
        except httpx.TimeoutException:
            return {
                "success": False,
                "error": f"请求超时 ({timeout}s)",
            }
        except httpx.RequestError as e:
            return {
                "success": False,
                "error": f"请求失败: {str(e)}",
            }
        except Exception as e:
            self.logger.error(f"HTTP 请求异常: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"未知错误: {str(e)}",
            }


# 创建工具实例
_http_client = HTTPClient()


@tool(args_schema=HTTPClientInput)
def http_client_tool(
    method: str = "GET",
    url: str = "",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """
    发送 HTTP 请求获取网络资源。
    
    支持 GET、POST、PUT、DELETE 方法。
    禁止访问内网地址以确保安全。
    
    使用示例：
    - GET 请求: method="GET", url="https://api.example.com/data"
    - POST 请求: method="POST", url="https://api.example.com/submit", 
                 body='{"key": "value"}'
    """
    try:
        result = _http_client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            body=body,
            timeout=timeout,
        )
        
        if result["success"]:
            body_preview = result.get("body", "")
            if isinstance(body_preview, dict):
                body_preview = json.dumps(body_preview, ensure_ascii=False, indent=2)
            
            if len(str(body_preview)) > 2000:
                body_preview = str(body_preview)[:2000] + "...[截断]"
            
            return f"""HTTP {method} {url}
状态码: {result['status_code']}

响应内容:
{body_preview}"""
        else:
            return f"请求失败: {result.get('error', '未知错误')}"
    
    except ValueError as e:
        return f"参数错误: {str(e)}"
    except Exception as e:
        logger.error(f"HTTP 工具异常: {e}", exc_info=True)
        return f"请求失败: {str(e)}"
```

### 2.3 注册工具

```python
# src/tools/__init__.py

from src.tools.http_client import HTTPClient, http_client_tool

__all__ = [
    # ... 现有工具
    "HTTPClient",
    "http_client_tool",
]

def get_all_tools():
    return [
        calculator_tool,
        file_manager_tool,
        code_executor_tool,
        web_search_tool,
        http_client_tool,  # 新增
    ]
```

---

## 3. 新增工作流节点

### 3.1 创建自定义节点

```python
# src/graph/nodes.py

def validation_node(state: AgentState) -> Dict[str, Any]:
    """
    验证节点
    
    在执行关键操作前进行验证。
    """
    logger.info("[Node] validation - 开始验证")
    
    original_task = state.get("original_task", "")
    subtasks = state.get("subtasks", [])
    
    # 验证逻辑
    validation_results = {
        "task_valid": bool(original_task),
        "subtasks_valid": len(subtasks) > 0,
        "no_errors": len(state.get("error_log", [])) == 0,
    }
    
    all_valid = all(validation_results.values())
    
    reasoning_trace = state.get("reasoning_trace", [])
    reasoning_trace.append(
        f"[Validation] 验证结果: {'通过' if all_valid else '未通过'}"
    )
    
    return {
        **state,
        "validation_results": validation_results,
        "reasoning_trace": reasoning_trace,
        "next": "task_router" if all_valid else "error_handler",
    }
```

### 3.2 添加到图

```python
# src/graph/builder.py

def build_graph(settings):
    # ... 现有节点
    
    # 添加验证节点
    graph.add_node("validation", validation_node)
    
    # 添加边：planner 后进行验证
    graph.add_edge("planner", "validation")
    
    # 验证后的条件路由
    graph.add_conditional_edges(
        "validation",
        lambda state: state.get("next", "task_router"),
        {
            "task_router": "task_router",
            "error_handler": "error_handler",
        }
    )
```

---

## 4. 自定义提示词

### 4.1 运行时修改提示词

```python
from src.config.prompts import PromptTemplates

# 设置自定义系统提示词
PromptTemplates.set_custom(
    "COORDINATOR_SYSTEM",
    """你是一个专门为企业定制的任务协调者。

你的工作重点：
1. 严格遵守企业安全规范
2. 优先使用内部工具和资源
3. 保护敏感信息不外泄

[其他自定义内容...]
"""
)

# 设置自定义任务提示词
PromptTemplates.set_custom(
    "CODER_TASK",
    """请按照公司代码规范编写代码：

任务：$task
上下文：$context

代码规范要求：
1. 使用 Google Python Style Guide
2. 类型注解必须完整
3. 函数必须有 docstring
4. 单元测试覆盖率 > 80%

[其他规范...]
"""
)
```

### 4.2 从文件加载提示词

```python
# src/config/custom_prompts.py

import os
from pathlib import Path
from src.config.prompts import PromptTemplates

def load_custom_prompts(prompts_dir: str = "prompts") -> None:
    """
    从目录加载自定义提示词
    
    目录结构：
        prompts/
        ├── coordinator_system.txt
        ├── planner_decompose.txt
        └── ...
    """
    prompts_path = Path(prompts_dir)
    
    if not prompts_path.exists():
        return
    
    for prompt_file in prompts_path.glob("*.txt"):
        # 文件名转换为模板名
        # coordinator_system.txt -> COORDINATOR_SYSTEM
        template_name = prompt_file.stem.upper()
        
        content = prompt_file.read_text(encoding="utf-8")
        PromptTemplates.set_custom(template_name, content)
        
        print(f"加载自定义提示词: {template_name}")
```

---

## 5. 自定义路由逻辑

### 5.1 基于任务类型的路由

```python
# src/graph/edges.py

def custom_task_router(state: AgentState) -> RouteType:
    """
    自定义任务路由
    
    基于任务类型和关键词智能选择执行者。
    """
    original_task = state.get("original_task", "").lower()
    subtasks = state.get("subtasks", [])
    
    # 获取下一个待处理任务
    pending = [t for t in subtasks if t.get("status") == "pending"]
    if not pending:
        return "synthesizer"
    
    task = pending[0]
    task_type = task.get("task_type", "")
    task_desc = task.get("description", "").lower()
    
    # 数据分析任务
    if any(kw in task_desc for kw in ["分析", "统计", "数据", "图表"]):
        return "data_analyst"
    
    # API 调用任务
    if any(kw in task_desc for kw in ["api", "接口", "请求", "http"]):
        return "executor"
    
    # 代码任务
    if task_type == "code" or any(kw in task_desc for kw in ["编写", "代码", "实现"]):
        return "coder"
    
    # 研究任务
    if task_type == "research" or any(kw in task_desc for kw in ["搜索", "查找", "研究"]):
        return "researcher"
    
    # 默认执行者
    return "executor"
```

### 5.2 基于评分的质量路由

```python
# src/graph/edges.py

def quality_based_router(state: AgentState) -> RouteType:
    """
    基于质量评分的路由
    
    根据历史评分决定是否需要更严格的审核。
    """
    eval_results = state.get("evaluation_results", [])
    
    if not eval_results:
        # 首次评估，使用普通流程
        return "critic"
    
    # 计算平均分
    scores = [r.get("score", 0.7) for r in eval_results if isinstance(r, dict)]
    avg_score = sum(scores) / len(scores) if scores else 0.7
    
    # 低于阈值，需要更严格的审核
    if avg_score < 0.6:
        return "human_node"  # 人工审核
    elif avg_score < 0.75:
        return "critic"  # 额外审核
    else:
        return "synthesizer"  # 直接综合
```

---

## 6. 扩展记忆系统

### 6.1 添加向量存储记忆

```python
# src/memory/vector_memory.py

"""
向量记忆模块
============

使用向量数据库实现语义搜索记忆。
"""

from typing import Any, Dict, List, Optional
import json

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorMemory:
    """
    向量记忆
    
    使用 embedding 进行语义搜索。
    支持对接 Chroma、Pinecone、Milvus 等向量数据库。
    """
    
    def __init__(
        self,
        collection_name: str = "multi_agent_memory",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        初始化向量记忆
        
        Args:
            collection_name: 集合名称
            embedding_model: 嵌入模型
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._collection = None
        self.logger = get_logger(self.__class__.__name__)
        
        self._init_collection()
    
    def _init_collection(self) -> None:
        """初始化向量集合"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.Client(Settings(
                anonymized_telemetry=False
            ))
            
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.logger.info(f"向量集合初始化完成: {self.collection_name}")
            
        except ImportError:
            self.logger.warning("chromadb 未安装，向量记忆不可用")
            self._collection = None
    
    def store(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        存储内容到向量数据库
        
        Args:
            key: 唯一标识
            content: 文本内容
            metadata: 元数据
        """
        if self._collection is None:
            return
        
        self._collection.upsert(
            ids=[key],
            documents=[content],
            metadatas=[metadata or {}],
        )
        
        self.logger.debug(f"存储向量: {key}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义搜索
        
        Args:
            query: 搜索查询
            top_k: 返回数量
            filter_metadata: 元数据过滤
            
        Returns:
            搜索结果列表
        """
        if self._collection is None:
            return []
        
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata,
        )
        
        # 格式化结果
        formatted = []
        for i, (doc_id, doc, metadata, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            formatted.append({
                "id": doc_id,
                "content": doc,
                "metadata": metadata,
                "score": 1 - distance,  # 转换为相似度
            })
        
        return formatted
    
    def delete(self, key: str) -> None:
        """删除记忆"""
        if self._collection is None:
            return
        
        self._collection.delete(ids=[key])
    
    def clear(self) -> None:
        """清空所有记忆"""
        if self._collection is None:
            return
        
        # 重新创建集合
        import chromadb
        client = chromadb.Client()
        client.delete_collection(self.collection_name)
        self._init_collection()
```

---

## 7. 对接企业级场景

### 7.1 企业配置示例

```python
# src/config/enterprise.py

"""
企业级配置
==========

针对企业环境的特殊配置。
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SecurityConfig(BaseModel):
    """安全配置"""
    enable_audit_log: bool = True
    sensitive_fields: List[str] = ["password", "api_key", "token"]
    allowed_domains: List[str] = []
    blocked_domains: List[str] = ["localhost", "127.0.0.1"]
    max_file_size_mb: int = 10
    enable_encryption: bool = False


class ComplianceConfig(BaseModel):
    """合规配置"""
    data_retention_days: int = 90
    enable_pii_detection: bool = True
    require_approval_for: List[str] = ["file_write", "http_request"]
    audit_all_llm_calls: bool = True


class EnterpriseSettings(BaseModel):
    """企业设置"""
    company_name: str
    environment: str = "production"
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    
    # LLM 配置
    use_azure_openai: bool = False
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    
    # 审批流程
    require_human_approval: bool = True
    approval_timeout_minutes: int = 30
    
    # 日志
    log_to_splunk: bool = False
    splunk_endpoint: Optional[str] = None
```

### 7.2 审计日志

```python
# src/utils/audit.py

"""
审计日志模块
============

记录系统操作的审计日志。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(
        self,
        log_dir: str = "audit_logs",
        enable_file_log: bool = True,
        enable_remote_log: bool = False,
        remote_endpoint: Optional[str] = None,
    ):
        self.log_dir = Path(log_dir)
        self.enable_file_log = enable_file_log
        self.enable_remote_log = enable_remote_log
        self.remote_endpoint = remote_endpoint
        
        if enable_file_log:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log(
        self,
        action: str,
        actor: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        result: str = "success",
    ) -> None:
        """
        记录审计日志
        
        Args:
            action: 操作类型
            actor: 执行者
            resource: 资源
            details: 详细信息
            result: 结果
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "actor": actor,
            "resource": resource,
            "details": details or {},
            "result": result,
        }
        
        if self.enable_file_log:
            self._write_to_file(entry)
        
        if self.enable_remote_log and self.remote_endpoint:
            self._send_to_remote(entry)
        
        logger.debug(f"审计日志: {action} by {actor} on {resource}")
    
    def _write_to_file(self, entry: Dict) -> None:
        """写入文件"""
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def _send_to_remote(self, entry: Dict) -> None:
        """发送到远程"""
        try:
            import httpx
            httpx.post(
                self.remote_endpoint,
                json=entry,
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"发送审计日志失败: {e}")


# 全局审计日志实例
audit_logger = AuditLogger()


def audit_action(
    action: str,
    actor: str,
    resource: str,
    details: Optional[Dict] = None,
    result: str = "success",
) -> None:
    """便捷函数：记录审计日志"""
    audit_logger.log(action, actor, resource, details, result)
```

---

## 8. 性能优化

### 8.1 并行执行优化

```python
# src/graph/parallel.py

"""
并行执行模块
============

支持子任务的并行执行。
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ParallelExecutor:
    """并行执行器"""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def execute_parallel(
        self,
        tasks: List[Dict[str, Any]],
        execute_func: Callable[[Dict], Any],
        timeout: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        并行执行任务
        
        Args:
            tasks: 任务列表
            execute_func: 执行函数
            timeout: 超时时间
            
        Returns:
            执行结果列表
        """
        results = []
        futures = {}
        
        for task in tasks:
            future = self.executor.submit(execute_func, task)
            futures[future] = task["id"]
        
        for future in as_completed(futures, timeout=timeout):
            task_id = futures[future]
            try:
                result = future.result()
                results.append({
                    "task_id": task_id,
                    "success": True,
                    "result": result,
                })
            except Exception as e:
                logger.error(f"任务 {task_id} 执行失败: {e}")
                results.append({
                    "task_id": task_id,
                    "success": False,
                    "error": str(e),
                })
        
        return results
    
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)
```

### 8.2 缓存优化

```python
# src/utils/cache.py

"""
缓存模块
========

提供 LLM 响应缓存功能。
"""

import hashlib
import json
from typing import Any, Optional
from functools import lru_cache

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMCache:
    """LLM 响应缓存"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
    
    def _make_key(self, messages: list, model: str) -> str:
        """生成缓存键"""
        content = json.dumps({
            "messages": [m.content if hasattr(m, "content") else str(m) for m in messages],
            "model": model,
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, messages: list, model: str) -> Optional[Any]:
        """获取缓存"""
        key = self._make_key(messages, model)
        result = self._cache.get(key)
        if result:
            logger.debug(f"缓存命中: {key[:8]}")
        return result
    
    def set(self, messages: list, model: str, response: Any) -> None:
        """设置缓存"""
        if len(self._cache) >= self.max_size:
            # 移除最旧的项
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        key = self._make_key(messages, model)
        self._cache[key] = response
        logger.debug(f"缓存设置: {key[:8]}")
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()


# 全局缓存实例
llm_cache = LLMCache()
```

---

本扩展指南涵盖了系统的主要扩展点。根据具体需求，您可以：

1. 添加新的专业 Agent 处理特定领域任务
2. 集成更多外部工具和服务
3. 自定义工作流程和路由逻辑
4. 扩展记忆系统支持更复杂的检索
5. 对接企业级基础设施

如有问题，请参考源代码或提交 Issue。