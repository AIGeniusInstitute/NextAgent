"""
审核者 Agent
============

负责质量评估、问题发现和改进建议。
"""

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates
from src.types import EvaluationResult


@register_agent("critic")
class CriticAgent(BaseAgent):
    """
    审核者智能体
    
    核心职责：
    1. 评估工作成果质量
    2. 发现错误和问题
    3. 提供改进建议
    4. 决定是否需要修正或人工介入
    """
    
    # 质量阈值
    PASS_THRESHOLD = 0.7
    HUMAN_REVIEW_THRESHOLD = 0.5
    
    @property
    def name(self) -> str:
        return "critic"
    
    @property
    def description(self) -> str:
        return "审核者，负责质量评估和改进建议"
    
    @property
    def capabilities(self) -> List[str]:
        return ["evaluate", "review", "suggest"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行审核任务
        
        Args:
            state: 当前状态
            
        Returns:
            包含评审结果的更新状态
        """
        # 获取最近完成的任务
        recent_outputs = self._get_recent_outputs(state)
        
        if not recent_outputs:
            self.logger.warning("没有找到需要审核的输出")
            return {**state, "next": "coordinator"}
        
        # 对每个输出进行评审
        evaluation_results = []
        for output_key, output_data in recent_outputs.items():
            eval_result = self._evaluate_output(output_key, output_data, state)
            evaluation_results.append(eval_result)
        
        # 汇总评审结果
        overall_result = self._aggregate_evaluations(evaluation_results)
        
        # 决定下一步行动
        next_action = self._decide_action(overall_result, state)
        
        # 更新反思记录
        reflection_notes = state.get("reflection_notes", [])
        reflection_notes.append(overall_result["summary"])
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"[Critic] 评审完成，评分: {overall_result['score']:.2f}, "
            f"行动: {next_action}"
        )
        
        # 判断是否需要人工介入
        needs_human = (
            overall_result["score"] < self.HUMAN_REVIEW_THRESHOLD and
            self.settings.enable_human_in_loop
        )
        
        return {
            **state,
            "reflection_notes": reflection_notes,
            "evaluation_results": state.get("evaluation_results", []) + [overall_result],
            "needs_human_input": needs_human,
            "reasoning_trace": reasoning_trace,
            "next": next_action,
        }
    
    def _get_recent_outputs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取需要审核的最近输出
        
        Args:
            state: 当前状态
            
        Returns:
            需要审核的输出字典
        """
        agent_outputs = state.get("agent_outputs", {})
        evaluation_results = state.get("evaluation_results", [])
        
        # 获取已评审的输出键
        evaluated_keys = set()
        for result in evaluation_results:
            if isinstance(result, dict):
                evaluated_keys.update(result.get("evaluated_outputs", []))
        
        # 筛选未评审的输出
        recent = {
            k: v for k, v in agent_outputs.items()
            if k not in evaluated_keys
        }
        
        return recent
    
    def _evaluate_output(
        self,
        output_key: str,
        output_data: Any,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        评估单个输出
        
        Args:
            output_key: 输出键
            output_data: 输出数据
            state: 当前状态
            
        Returns:
            评估结果字典
        """
        # 提取输出内容
        if isinstance(output_data, dict):
            output_content = output_data.get("output", str(output_data))
            task_id = output_data.get("task_id")
        else:
            output_content = str(output_data)
            task_id = None
        
        # 获取相关子任务
        subtask = None
        if task_id:
            subtasks = state.get("subtasks", [])
            subtask = next((t for t in subtasks if t.get("id") == task_id), None)
        
        # 构建评审提示
        prompt = PromptTemplates.get(
            "CRITIC_REVIEW",
            original_task=state.get("original_task", ""),
            subtask=json.dumps(subtask, ensure_ascii=False) if subtask else "未知",
            agent_name=output_key.split("_")[0],
            output=output_content[:2000],  # 限制长度
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        # 解析评审结果
        eval_result = self._parse_evaluation(response.content)
        eval_result["output_key"] = output_key
        
        return eval_result
    
    def _parse_evaluation(self, content: str) -> Dict[str, Any]:
        """
        解析 LLM 评审结果
        
        Args:
            content: LLM 响应
            
        Returns:
            解析后的评估结果
        """
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)```', content)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(content)
            
            # 确保必要字段
            return {
                "score": float(result.get("score", 0.7)),
                "passed": result.get("passed", True),
                "issues": result.get("issues", []),
                "suggestions": result.get("suggestions", []),
                "action": result.get("action", "approve"),
                "reasoning": result.get("reasoning", ""),
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"评审结果解析失败: {e}")
            # 从文本中提取关键信息
            return self._extract_evaluation_from_text(content)
    
    def _extract_evaluation_from_text(self, content: str) -> Dict[str, Any]:
        """
        从文本中提取评估信息（当 JSON 解析失败时）
        
        Args:
            content: 响应文本
            
        Returns:
            评估结果
        """
        content_lower = content.lower()
        
        # 简单的关键词分析
        if "优秀" in content or "excellent" in content_lower:
            score = 0.95
        elif "良好" in content or "good" in content_lower:
            score = 0.8
        elif "通过" in content or "pass" in content_lower:
            score = 0.7
        elif "需要改进" in content or "needs improvement" in content_lower:
            score = 0.55
        elif "不合格" in content or "fail" in content_lower:
            score = 0.3
        else:
            score = 0.7
        
        passed = score >= self.PASS_THRESHOLD
        
        return {
            "score": score,
            "passed": passed,
            "issues": [],
            "suggestions": [],
            "action": "approve" if passed else "revise",
            "reasoning": content[:500],
        }
    
    def _aggregate_evaluations(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        汇总多个评估结果
        
        Args:
            evaluations: 评估结果列表
            
        Returns:
            汇总结果
        """
        if not evaluations:
            return {
                "score": 0.7,
                "passed": True,
                "summary": "无需评审的输出",
                "evaluated_outputs": [],
            }
        
        # 计算平均分
        scores = [e.get("score", 0.7) for e in evaluations]
        avg_score = sum(scores) / len(scores)
        
        # 汇总问题和建议
        all_issues = []
        all_suggestions = []
        for e in evaluations:
            all_issues.extend(e.get("issues", []))
            all_suggestions.extend(e.get("suggestions", []))
        
        # 判断是否通过
        passed = all(e.get("passed", True) for e in evaluations)
        
        # 生成汇总
        summary_parts = [f"评审完成 - 平均评分: {avg_score:.2f}"]
        if all_issues:
            summary_parts.append(f"发现 {len(all_issues)} 个问题")
        if all_suggestions:
            summary_parts.append(f"提出 {len(all_suggestions)} 条建议")
        
        return {
            "score": avg_score,
            "passed": passed,
            "issues": all_issues,
            "suggestions": all_suggestions,
            "summary": "; ".join(summary_parts),
            "evaluated_outputs": [e.get("output_key") for e in evaluations],
            "details": evaluations,
        }
    
    def _decide_action(
        self,
        evaluation: Dict[str, Any],
        state: Dict[str, Any]
    ) -> str:
        """
        根据评估结果决定下一步行动
        
        Args:
            evaluation: 评估结果
            state: 当前状态
            
        Returns:
            下一步行动节点名
        """
        score = evaluation.get("score", 0.7)
        passed = evaluation.get("passed", True)
        
        # 检查是否有待处理的子任务
        pending_tasks = self._get_pending_subtasks(state)
        
        if score < self.HUMAN_REVIEW_THRESHOLD:
            # 低分需要人工介入
            return "human_node"
        
        if not passed:
            # 未通过，检查重试次数
            iteration_count = state.get("iteration_count", 0)
            max_iterations = state.get("max_iterations", 10)
            
            if iteration_count < max_iterations:
                return "task_router"  # 重新执行
            else:
                return "synthesizer"  # 超过次数，强制结束
        
        if pending_tasks:
            # 还有待处理任务
            return "task_router"
        
        # 所有任务完成且通过审核
        return "synthesizer"