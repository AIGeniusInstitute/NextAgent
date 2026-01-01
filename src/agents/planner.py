"""
规划者 Agent
============

负责任务分解、依赖分析和执行计划制定。
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates
from src.types import SubTask, TaskType, TaskStatus, TaskPriority, AgentRole


@register_agent("planner")
class PlannerAgent(BaseAgent):
    """
    规划者智能体
    
    核心职责：
    1. 将复杂任务分解为可管理的子任务
    2. 分析子任务之间的依赖关系
    3. 为每个子任务分配合适的执行者
    4. 制定执行顺序和并行策略
    """
    
    @property
    def name(self) -> str:
        return "planner"
    
    @property
    def description(self) -> str:
        return "任务规划者，负责分解任务和制定执行计划"
    
    @property
    def capabilities(self) -> List[str]:
        return ["task_decomposition", "dependency_analysis", "planning"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行规划逻辑
        
        Args:
            state: 当前状态
            
        Returns:
            包含子任务列表的更新状态
        """
        original_task = state.get("original_task", "")
        task_understanding = state.get("task_understanding", "")
        
        # 构建规划提示
        prompt = PromptTemplates.get(
            "PLANNER_DECOMPOSE",
            task=original_task,
            understanding=task_understanding,
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        # 解析规划结果
        plan_result = self._parse_plan(response.content)
        
        if plan_result is None:
            self.logger.error("规划解析失败，使用默认规划")
            plan_result = self._create_default_plan(original_task)
        
        subtasks = plan_result.get("subtasks", [])
        parallel_groups = plan_result.get("parallel_groups", [])
        plan_summary = plan_result.get("plan_summary", "")
        
        # 转换为 SubTask 格式
        formatted_subtasks = self._format_subtasks(subtasks)
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[Planner] 任务分解完成: {len(formatted_subtasks)} 个子任务"
        )
        reasoning_trace.append(f"[Planner] 计划概述: {plan_summary}")
        
        self.logger.info(f"规划完成: {len(formatted_subtasks)} 个子任务")
        
        return {
            **state,
            "subtasks": formatted_subtasks,
            "current_plan": plan_summary,
            "parallel_groups": parallel_groups,
            "reasoning_trace": reasoning_trace,
            "next": "task_router",
        }
    
    def _parse_plan(self, content: str) -> Optional[Dict[str, Any]]:
        """
        解析 LLM 返回的规划结果
        
        Args:
            content: LLM 响应内容
            
        Returns:
            解析后的规划字典，失败返回 None
        """
        try:
            # 尝试提取 JSON 块
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析
                json_str = content
            
            plan = json.loads(json_str)
            return plan
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON 解析失败: {e}")
            # 尝试修复常见的 JSON 问题
            try:
                # 移除可能的注释
                cleaned = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
                cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
                return json.loads(cleaned)
            except:
                return None
    
    def _create_default_plan(self, task: str) -> Dict[str, Any]:
        """
        创建默认规划（当解析失败时使用）
        
        Args:
            task: 原始任务
            
        Returns:
            默认规划字典
        """
        return {
            "plan_summary": "使用默认规划执行任务",
            "subtasks": [
                {
                    "id": f"task_{uuid.uuid4().hex[:8]}",
                    "name": "分析任务需求",
                    "description": f"分析并理解任务: {task[:100]}",
                    "task_type": "analyze",
                    "assigned_agent": "researcher",
                    "dependencies": [],
                    "priority": "high",
                },
                {
                    "id": f"task_{uuid.uuid4().hex[:8]}",
                    "name": "执行主要任务",
                    "description": f"执行任务: {task[:100]}",
                    "task_type": "execute",
                    "assigned_agent": "executor",
                    "dependencies": [],
                    "priority": "high",
                },
            ],
            "parallel_groups": [],
        }
    
    def _format_subtasks(self, subtasks: List[Dict]) -> List[Dict]:
        """
        格式化子任务为标准格式
        
        Args:
            subtasks: 原始子任务列表
            
        Returns:
            格式化后的子任务列表
        """
        formatted = []
        
        for task in subtasks:
            # 确保有 ID
            task_id = task.get("id") or f"task_{uuid.uuid4().hex[:8]}"
            
            # 映射任务类型
            task_type = task.get("task_type", "execute")
            if task_type not in ["research", "code", "execute", "analyze", "synthesize"]:
                task_type = "execute"
            
            # 映射 Agent
            assigned_agent = task.get("assigned_agent", "executor")
            agent_map = {
                "researcher": "researcher",
                "research": "researcher",
                "coder": "coder",
                "code": "coder",
                "executor": "executor",
                "execute": "executor",
            }
            assigned_agent = agent_map.get(assigned_agent, "executor")
            
            # 映射优先级
            priority = task.get("priority", "medium")
            if priority not in ["low", "medium", "high", "critical"]:
                priority = "medium"
            
            formatted_task = {
                "id": task_id,
                "name": task.get("name", "未命名任务"),
                "description": task.get("description", ""),
                "task_type": task_type,
                "assigned_agent": assigned_agent,
                "dependencies": task.get("dependencies", []),
                "priority": priority,
                "status": "pending",
                "result": None,
                "error_message": None,
                "retry_count": 0,
                "metadata": task.get("metadata", {}),
            }
            
            formatted.append(formatted_task)
        
        return formatted
    
    def _analyze_dependencies(self, subtasks: List[Dict]) -> List[List[str]]:
        """
        分析依赖关系，生成并行执行组
        
        Args:
            subtasks: 子任务列表
            
        Returns:
            可并行执行的任务组列表
        """
        # 构建依赖图
        task_ids = {t["id"] for t in subtasks}
        dependencies = {t["id"]: set(t.get("dependencies", [])) for t in subtasks}
        
        # 验证依赖（移除不存在的依赖）
        for task_id, deps in dependencies.items():
            dependencies[task_id] = deps & task_ids
        
        # 拓扑排序，生成并行组
        parallel_groups = []
        remaining = set(task_ids)
        
        while remaining:
            # 找出没有未完成依赖的任务
            ready = {
                task_id for task_id in remaining
                if not (dependencies[task_id] & remaining)
            }
            
            if not ready:
                # 检测到循环依赖
                self.logger.warning("检测到循环依赖，使用顺序执行")
                parallel_groups.append(list(remaining))
                break
            
            parallel_groups.append(list(ready))
            remaining -= ready
        
        return parallel_groups