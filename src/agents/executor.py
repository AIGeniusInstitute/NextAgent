"""
执行者 Agent
============

负责工具调用和具体操作执行。
"""

import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates

import re

@register_agent("executor")
class ExecutorAgent(BaseAgent):
    """
    执行者智能体
    
    核心职责：
    1. 调用各种工具完成任务
    2. 验证执行结果
    3. 处理执行异常
    4. 报告执行状态
    """
    
    @property
    def name(self) -> str:
        return "executor"
    
    @property
    def description(self) -> str:
        return "执行者，负责工具调用和操作执行"
    
    @property
    def capabilities(self) -> List[str]:
        return ["execute", "tool_call", "file_operation"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行操作任务
        
        Args:
            state: 当前状态
            
        Returns:
            包含执行结果的更新状态
        """
        # 获取当前待处理的执行任务
        current_task = self._get_current_task(state)
        
        if current_task is None:
            self.logger.warning("没有找到待处理的执行任务")
            return {**state, "next": "coordinator"}
        
        task_description = current_task.get("description", "")
        
        # 分析任务，决定需要调用的工具
        tool_plan = self._plan_tool_calls(task_description, state)
        
        # 执行工具调用
        execution_results = []
        for tool_call in tool_plan:
            result = self._execute_tool_call(tool_call)
            execution_results.append(result)
        
        # 汇总执行结果
        summary = self._summarize_results(execution_results, task_description)
        
        llm_result = self._llm_execution_reasoning(task_description, summary, state)
        final_output = f"{summary}\n\n## LLM 执行推理\n\n{llm_result}"
        
        # 判断是否成功
        success = all(r.get("success", False) for r in execution_results)
        
        # 创建 Agent 输出
        agent_output = self.create_output(
            output=final_output,
            reasoning=f"执行任务: {current_task.get('name', '')}",
            task_id=current_task.get("id"),
            confidence=0.9 if success else 0.5,
        )
        
        # 更新子任务状态
        status = "completed" if success else "failed"
        subtasks = self._update_subtask_status(
            state,
            current_task["id"],
            status=status,
            result=final_output,
            error=None if success else "执行过程中出现错误",
        )
        
        # 更新 agent_outputs
        agent_outputs = state.get("agent_outputs", {})
        agent_outputs[f"executor_{current_task['id']}"] = agent_output.model_dump()
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[Executor] {'成功' if success else '失败'}执行任务 '{current_task.get('name', '')}'"
        )
        
        return {
            **state,
            "subtasks": subtasks,
            "agent_outputs": agent_outputs,
            "reasoning_trace": reasoning_trace,
            "next": "critic",
        }
    
    def _get_current_task(self, state: Dict[str, Any]) -> Optional[Dict]:
        """获取当前待处理的执行任务"""
        subtasks = state.get("subtasks", [])
        
        for task in subtasks:
            if (task.get("status") == "pending" and 
                task.get("assigned_agent") == "executor"):
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
    
    def _plan_tool_calls(
        self,
        task_description: str,
        state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        规划工具调用
        
        Args:
            task_description: 任务描述
            state: 当前状态
            
        Returns:
            工具调用计划列表
        """
        # 获取可用工具列表
        available_tools = [t.name for t in self.tools]
        
        # 使用 LLM 分析需要的工具调用
        prompt = f"""分析以下任务，决定需要调用哪些工具：

任务：{task_description}

可用工具：{', '.join(available_tools)}

工具说明：
- calculator: 数学计算，参数 expression (表达式字符串)
- file_manager: 文件操作，参数 action (read/write), path (文件路径), content (写入内容，可选)
- code_executor: 执行 Python 代码，参数 code (代码字符串)
- web_search: 网络搜索，参数 query (搜索查询)

请输出 JSON 格式的工具调用计划：
```json
[
    {{"tool": "工具名", "params": {{"参数名": "参数值"}}}}
]
```

如果不需要调用任何工具，返回空数组 []
"""
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        # 解析工具调用计划
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)```', response.content)
            if json_match:
                plan = json.loads(json_match.group(1))
            else:
                plan = json.loads(response.content)
            return plan if isinstance(plan, list) else []
        except (json.JSONDecodeError, AttributeError):
            self.logger.warning("无法解析工具调用计划")
            return []
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个工具调用
        
        Args:
            tool_call: 工具调用信息
            
        Returns:
            执行结果字典
        """
        tool_name = tool_call.get("tool")
        params = tool_call.get("params", {})
        
        if not tool_name:
            return {
                "tool": tool_name,
                "success": False,
                "error": "工具名称为空",
                "result": None,
            }
        
        try:
            result = self.call_tool(tool_name, **params)
            return {
                "tool": tool_name,
                "success": True,
                "error": None,
                "result": result,
            }
        except Exception as e:
            self.logger.error(f"工具调用失败 {tool_name}: {e}")
            return {
                "tool": tool_name,
                "success": False,
                "error": str(e),
                "result": None,
            }
    
    def _summarize_results(
        self,
        results: List[Dict[str, Any]],
        task_description: str
    ) -> str:
        """
        汇总执行结果
        
        Args:
            results: 执行结果列表
            task_description: 任务描述
            
        Returns:
            汇总字符串
        """
        if not results:
            return f"任务 '{task_description}' 未执行任何工具调用"
        
        summary_parts = [f"## 执行结果\n\n任务：{task_description}\n"]
        
        for i, result in enumerate(results, 1):
            tool_name = result.get("tool", "未知")
            success = result.get("success", False)
            status = "✅ 成功" if success else "❌ 失败"
            
            summary_parts.append(f"### {i}. {tool_name} - {status}")
            
            if success:
                output = result.get("result", "无输出")
                summary_parts.append(f"输出：\n```\n{output}\n```")
            else:
                error = result.get("error", "未知错误")
                summary_parts.append(f"错误：{error}")
            
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def _llm_execution_reasoning(
        self,
        task_description: str,
        tool_summary: str,
        state: Dict[str, Any],
    ) -> str:
        """
        使用 LLM 进行执行推理，基于任务描述和工具执行结果生成结果
        """
        prompt = PromptTemplates.get(
            "EXECUTOR_TASK",
            task=task_description,
            input_data=tool_summary,
            expected_output=state.get("expected_output", ""),
        )
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        return getattr(response, "content", "")

