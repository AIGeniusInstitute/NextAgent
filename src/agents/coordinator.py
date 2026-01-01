"""
协调者 Agent
============

负责任务理解、工作分配、进度监控和结果整合。
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates
from src.types import TaskStatus


@register_agent("coordinator")
class CoordinatorAgent(BaseAgent):
    """
    协调者智能体
    
    核心职责：
    1. 理解用户任务，提取核心需求
    2. 决定任务处理策略（直接回答/分解执行）
    3. 监控任务执行进度
    4. 协调各智能体工作
    5. 整合最终结果
    """
    
    @property
    def name(self) -> str:
        return "coordinator"
    
    @property
    def description(self) -> str:
        return "任务协调者，负责理解任务、分配工作和监控进度"
    
    @property
    def capabilities(self) -> List[str]:
        return ["task_understanding", "routing", "monitoring", "integration"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行协调逻辑
        
        处理流程：
        1. 如果是新任务，进行任务理解
        2. 如果任务已分解，检查执行进度
        3. 决定下一步行动
        """
        # 检查是否是新任务（没有任务理解）
        if not state.get("task_understanding"):
            return self._understand_task(state)
        
        # 检查是否所有任务已完成
        if self._check_completion(state):
            return self._finalize(state)
        
        # 决定下一步路由
        return self._decide_next_step(state)
    
    def _understand_task(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        理解用户任务
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态，包含任务理解
        """
        original_task = state.get("original_task", "")
        
        # 构建提示词
        prompt = PromptTemplates.get(
            "COORDINATOR_TASK_UNDERSTANDING",
            task=original_task
        )
        
        # 调用 LLM
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        task_understanding = response.content
        
        # 分析任务类型，决定是否需要分解
        needs_planning = self._needs_planning(task_understanding)
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(f"[Coordinator] 任务理解完成: {task_understanding[:200]}...")
        
        self.logger.info(f"任务理解完成，需要规划: {needs_planning}")
        
        # 决定下一步
        if needs_planning:
            next_step = "planner"
        else:
            # 简单任务直接回答
            next_step = "synthesizer"
        
        return {
            **state,
            "task_understanding": task_understanding,
            "reasoning_trace": reasoning_trace,
            "next": next_step,
        }
    
    def _needs_planning(self, understanding: str) -> bool:
        """
        判断任务是否需要规划分解
        
        Args:
            understanding: 任务理解文本
            
        Returns:
            是否需要规划
        """
        # 简单任务关键词
        simple_keywords = ["简单问答", "直接回答", "simple"]
        
        # 复杂任务关键词
        complex_keywords = [
            "代码", "编写", "开发", "实现",
            "分析", "研究", "调研",
            "步骤", "计划", "方案",
            "综合任务", "复杂"
        ]
        
        understanding_lower = understanding.lower()
        
        # 检查是否包含简单任务关键词
        for keyword in simple_keywords:
            if keyword in understanding_lower:
                return False
        
        # 检查是否包含复杂任务关键词
        for keyword in complex_keywords:
            if keyword in understanding_lower:
                return True
        
        # 默认需要规划
        return True
    
    def _check_completion(self, state: Dict[str, Any]) -> bool:
        """
        检查任务是否全部完成
        
        Args:
            state: 当前状态
            
        Returns:
            是否全部完成
        """
        subtasks = state.get("subtasks", [])
        
        # 如果没有子任务，检查是否有直接输出
        if not subtasks:
            return bool(state.get("agent_outputs"))
        
        # 检查所有子任务是否完成
        for task in subtasks:
            status = task.get("status")
            if status not in [TaskStatus.COMPLETED.value, "completed"]:
                return False
        
        return True
    
    def _decide_next_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        决定下一步行动
        
        Args:
            state: 当前状态
            
        Returns:
            包含下一步路由的状态
        """
        subtasks = state.get("subtasks", [])
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 10)
        
        # 检查迭代次数
        if iteration_count >= max_iterations:
            self.logger.warning(f"达到最大迭代次数 {max_iterations}，强制结束")
            return {
                **state,
                "next": "synthesizer",
                "reasoning_trace": state.get("reasoning_trace", []) + [
                    f"[Coordinator] 达到最大迭代次数，强制进入综合阶段"
                ],
            }
        
        # 构建路由决策提示
        completed = [t for t in subtasks if t.get("status") == "completed"]
        pending = [t for t in subtasks if t.get("status") == "pending"]
        failed = [t for t in subtasks if t.get("status") == "failed"]
        
        prompt = PromptTemplates.get(
            "COORDINATOR_ROUTING",
            original_task=state.get("original_task", ""),
            task_understanding=state.get("task_understanding", ""),
            completed_tasks=json.dumps([t["name"] for t in completed], ensure_ascii=False),
            pending_tasks=json.dumps([t["name"] for t in pending], ensure_ascii=False),
            iteration_count=iteration_count,
            max_iterations=max_iterations,
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        decision = response.content.strip().upper()
        
        # 解析决策
        if "FINISH" in decision:
            next_step = "synthesizer"
        elif "REPLAN" in decision:
            next_step = "planner"
        elif pending:
            # 路由到任务执行
            next_step = "task_router"
        elif failed:
            # 有失败任务，尝试重新执行
            next_step = "task_router"
        else:
            next_step = "synthesizer"
        
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(f"[Coordinator] 路由决策: {decision} -> {next_step}")
        
        return {
            **state,
            "next": next_step,
            "iteration_count": iteration_count + 1,
            "reasoning_trace": reasoning_trace,
        }
    
    def _finalize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        完成任务处理
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append("[Coordinator] 所有任务已完成，准备生成最终答案")
        
        return {
            **state,
            "next": "synthesizer",
            "reasoning_trace": reasoning_trace,
        }