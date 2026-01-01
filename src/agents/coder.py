"""
编码者 Agent
============

负责代码编写、调试和技术实现。
"""

import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates


@register_agent("coder")
class CoderAgent(BaseAgent):
    """
    编码者智能体
    
    核心职责：
    1. 根据需求编写代码
    2. 调试和修复代码问题
    3. 遵循编码最佳实践
    4. 添加文档和注释
    """
    
    @property
    def name(self) -> str:
        return "coder"
    
    @property
    def description(self) -> str:
        return "编码者，负责代码编写和技术实现"
    
    @property
    def capabilities(self) -> List[str]:
        return ["code", "debug", "implement"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行编码任务
        
        Args:
            state: 当前状态
            
        Returns:
            包含代码结果的更新状态
        """
        # 获取当前待处理的编码任务
        current_task = self._get_current_task(state)
        
        if current_task is None:
            self.logger.warning("没有找到待处理的编码任务")
            return {**state, "next": "coordinator"}
        
        task_description = current_task.get("description", "")
        context = self._build_context(state)
        requirements = current_task.get("metadata", {}).get("requirements", "Python 3.10+")
        
        # 构建编码提示
        prompt = PromptTemplates.get(
            "CODER_TASK",
            task=task_description,
            context=context,
            requirements=requirements,
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        code_output = response.content
        
        # 提取代码块
        code_blocks = self._extract_code_blocks(code_output)
        
        # 可选：测试代码执行
        execution_result = None
        if code_blocks and self._should_test_code(current_task):
            execution_result = self._test_code(code_blocks[0])
        
        # 构建完整输出
        full_output = code_output
        if execution_result:
            full_output += f"\n\n**代码测试结果:**\n```\n{execution_result}\n```"
        
        # 创建 Agent 输出
        agent_output = self.create_output(
            output=full_output,
            reasoning=f"完成编码任务: {current_task.get('name', '')}",
            task_id=current_task.get("id"),
            confidence=0.85 if execution_result else 0.75,
        )
        
        # 更新子任务状态
        subtasks = self._update_subtask_status(
            state,
            current_task["id"],
            status="completed",
            result=full_output,
        )
        
        # 更新 agent_outputs
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs[f"coder_{current_task['id']}"] = agent_output.model_dump()
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[Coder] 完成任务 '{current_task.get('name', '')}'"
        )
        if code_blocks:
            reasoning_trace.append(f"[Coder] 生成 {len(code_blocks)} 个代码块")
        
        return {
            **state,
            "subtasks": subtasks,
            "agent_outputs": agent_outputs,
            "reasoning_trace": reasoning_trace,
            "next": "critic",
        }
    
    def _get_current_task(self, state: Dict[str, Any]) -> Optional[Dict]:
        """
        获取当前待处理的编码任务
        
        Args:
            state: 当前状态
            
        Returns:
            待处理的任务
        """
        subtasks = state.get("subtasks", [])
        
        for task in subtasks:
            if (task.get("status") == "pending" and 
                task.get("assigned_agent") == "coder"):
                # 检查依赖是否满足
                dependencies = task.get("dependencies", [])
                if self._dependencies_satisfied(subtasks, dependencies):
                    return task
        
        return None
    
    def _dependencies_satisfied(
        self,
        subtasks: List[Dict],
        dependencies: List[str]
    ) -> bool:
        """检查任务依赖是否满足"""
        if not dependencies:
            return True
        
        completed_ids = {
            t["id"] for t in subtasks
            if t.get("status") == "completed"
        }
        
        return all(dep in completed_ids for dep in dependencies)
    
    def _build_context(self, state: Dict[str, Any]) -> str:
        """构建上下文信息"""
        context_parts = []
        
        original_task = state.get("original_task", "")
        if original_task:
            context_parts.append(f"原始任务: {original_task}")
        
        # 已有的研究结果
        agent_outputs = state.get("agent_outputs", {})
        for name, output in agent_outputs.items():
            if "researcher" in name:
                if isinstance(output, dict):
                    result = output.get("output", "")[:500]
                else:
                    result = str(output)[:500]
                context_parts.append(f"研究结果: {result}")
        
        return "\n".join(context_parts)
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """
        从内容中提取代码块
        
        Args:
            content: 响应内容
            
        Returns:
            代码块列表
        """
        # 匹配 ```python ... ``` 或 ``` ... ```
        pattern = r'```(?:python)?\s*([\s\S]*?)```'
        matches = re.findall(pattern, content)
        return [m.strip() for m in matches if m.strip()]
    
    def _should_test_code(self, task: Dict) -> bool:
        """
        判断是否应该测试代码
        
        Args:
            task: 任务信息
            
        Returns:
            是否测试
        """
        # 检查任务元数据
        metadata = task.get("metadata", {})
        if metadata.get("skip_test"):
            return False
        
        # 默认测试简单代码
        description = task.get("description", "").lower()
        skip_keywords = ["不要测试", "不需要运行", "skip test"]
        
        return not any(kw in description for kw in skip_keywords)
    
    def _test_code(self, code: str) -> Optional[str]:
        """
        测试代码执行
        
        Args:
            code: 代码字符串
            
        Returns:
            执行结果，失败返回 None
        """
        # 检查是否有代码执行工具
        executor_tool = next(
            (t for t in self.tools if t.name == "code_executor"),
            None
        )
        
        if executor_tool is None:
            self.logger.debug("没有可用的代码执行工具")
            return None
        
        try:
            result = self.call_tool("code_executor", code=code)
            return str(result)
        except Exception as e:
            self.logger.warning(f"代码测试失败: {e}")
            return f"执行错误: {str(e)}"