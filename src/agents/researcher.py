"""
研究员 Agent
============

负责信息检索、知识整合和资料分析。
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates


@register_agent("researcher")
class ResearcherAgent(BaseAgent):
    """
    研究员智能体
    
    核心职责：
    1. 使用搜索工具获取相关信息
    2. 分析和理解检索到的内容
    3. 整合多来源信息
    4. 生成结构化研究报告
    """
    
    @property
    def name(self) -> str:
        return "researcher"
    
    @property
    def description(self) -> str:
        return "研究员，负责信息检索和知识整合"
    
    @property
    def capabilities(self) -> List[str]:
        return ["research", "analyze", "information_retrieval"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行研究任务
        
        Args:
            state: 当前状态
            
        Returns:
            包含研究结果的更新状态
        """
        # 获取当前待处理的研究任务
        current_task = self._get_current_task(state)
        
        if current_task is None:
            self.logger.warning("没有找到待处理的研究任务")
            return {**state, "next": "coordinator"}
        
        task_description = current_task.get("description", "")
        context = self._build_context(state)
        
        # 构建研究提示
        prompt = PromptTemplates.get(
            "RESEARCHER_TASK",
            task=task_description,
            context=context,
        )
        
        messages = [HumanMessage(content=prompt)]
        
        # 如果有搜索工具，先进行搜索
        search_results = self._perform_search(task_description)
        if search_results:
            prompt += f"\n\n搜索结果：\n{search_results}"
            messages = [HumanMessage(content=prompt)]
        
        # 调用 LLM 生成研究报告
        response = self.call_llm(messages)
        research_output = response.content
        
        # 创建 Agent 输出
        agent_output = self.create_output(
            output=research_output,
            reasoning=f"完成研究任务: {current_task.get('name', '')}",
            task_id=current_task.get("id"),
            confidence=0.85,
        )
        
        # 更新子任务状态
        subtasks = self._update_subtask_status(
            state,
            current_task["id"],
            status="completed",
            result=research_output,
        )
        
        # 更新 agent_outputs
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs[f"researcher_{current_task['id']}"] = agent_output.model_dump()
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[Researcher] 完成任务 '{current_task.get('name', '')}': {research_output[:100]}..."
        )
        
        return {
            **state,
            "subtasks": subtasks,
            "agent_outputs": agent_outputs,
            "reasoning_trace": reasoning_trace,
            "next": "critic",
        }
    
    def _get_current_task(self, state: Dict[str, Any]) -> Optional[Dict]:
        """
        获取当前待处理的研究任务
        
        Args:
            state: 当前状态
            
        Returns:
            待处理的任务，没有则返回 None
        """
        subtasks = state.get("subtasks", [])
        
        for task in subtasks:
            if (task.get("status") == "pending" and 
                task.get("assigned_agent") == "researcher"):
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
        """
        检查任务依赖是否满足
        
        Args:
            subtasks: 所有子任务
            dependencies: 依赖任务 ID 列表
            
        Returns:
            依赖是否全部满足
        """
        if not dependencies:
            return True
        
        completed_ids = {
            t["id"] for t in subtasks
            if t.get("status") == "completed"
        }
        
        return all(dep in completed_ids for dep in dependencies)
    
    def _build_context(self, state: Dict[str, Any]) -> str:
        """
        构建上下文信息
        
        Args:
            state: 当前状态
            
        Returns:
            上下文字符串
        """
        context_parts = []
        
        # 原始任务
        original_task = state.get("original_task", "")
        if original_task:
            context_parts.append(f"原始任务: {original_task}")
        
        # 任务理解
        understanding = state.get("task_understanding", "")
        if understanding:
            context_parts.append(f"任务理解: {understanding[:500]}")
        
        # 已完成任务的结果
        agent_outputs = state.get("agent_outputs", {})
        if agent_outputs:
            context_parts.append("已有成果:")
            for name, output in agent_outputs.items():
                if isinstance(output, dict):
                    result = output.get("output", "")[:200]
                else:
                    result = str(output)[:200]
                context_parts.append(f"  - {name}: {result}")
        
        return "\n".join(context_parts)
    
    def _perform_search(self, query: str) -> Optional[str]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            
        Returns:
            搜索结果字符串，失败返回 None
        """
        # 检查是否有搜索工具
        search_tool = next(
            (t for t in self.tools if t.name == "web_search"),
            None
        )
        
        if search_tool is None:
            self.logger.debug("没有可用的搜索工具")
            return None
        
        try:
            # 提取关键词进行搜索
            result = self.call_tool("web_search", query=query)
            return str(result)
        except Exception as e:
            self.logger.warning(f"搜索失败: {e}")
            return None