"""
综合者 Agent
============

负责结果汇总和最终输出生成。
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.agents.base import BaseAgent, register_agent
from src.config.prompts import PromptTemplates


@register_agent("synthesizer")
class SynthesizerAgent(BaseAgent):
    """
    综合者智能体
    
    核心职责：
    1. 整合所有智能体的输出
    2. 生成结构化的最终答案
    3. 确保输出完整、专业
    4. 根据需要保存结果文件
    """
    
    @property
    def name(self) -> str:
        return "synthesizer"
    
    @property
    def description(self) -> str:
        return "综合者，负责结果汇总和最终输出"
    
    @property
    def capabilities(self) -> List[str]:
        return ["synthesize", "aggregate", "format"]
    
    def _execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行综合任务
        
        Args:
            state: 当前状态
            
        Returns:
            包含最终答案的更新状态
        """
        original_task = state.get("original_task", "")
        agent_outputs = state.get("agent_outputs", {})
        reflection_notes = state.get("reflection_notes", [])
        
        # 格式化各智能体输出
        formatted_outputs = self._format_agent_outputs(agent_outputs)
        
        # 格式化审核意见
        formatted_reviews = self._format_reviews(reflection_notes)
        
        # 构建综合提示
        prompt = PromptTemplates.get(
            "SYNTHESIZER_AGGREGATE",
            original_task=original_task,
            agent_outputs=formatted_outputs,
            review_notes=formatted_reviews,
        )
        
        messages = [HumanMessage(content=prompt)]
        response = self.call_llm(messages)
        
        final_answer = response.content
        
        # 如果需要保存文件，执行保存
        save_result = self._save_if_needed(final_answer, state)
        if save_result:
            final_answer += f"\n\n{save_result}"
        
        # 创建 Agent 输出
        agent_output = self.create_output(
            output=final_answer,
            reasoning="综合所有结果生成最终答案",
            confidence=0.9,
        )
        
        # 更新 agent_outputs
        agent_outputs["synthesizer"] = agent_output.model_dump()
        
        # 更新推理轨迹
        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append("[Synthesizer] 生成最终答案")
        
        self.logger.info("综合完成，生成最终答案")
        
        return {
            **state,
            "agent_outputs": agent_outputs,
            "final_answer": final_answer,
            "reasoning_trace": reasoning_trace,
            "next": "end",
        }
    
    def _format_agent_outputs(self, outputs: Dict[str, Any]) -> str:
        """
        格式化各智能体的输出
        
        Args:
            outputs: 输出字典
            
        Returns:
            格式化的字符串
        """
        if not outputs:
            return "（无输出）"
        
        parts = []
        for name, output in outputs.items():
            if isinstance(output, dict):
                content = output.get("output", str(output))
                reasoning = output.get("reasoning", "")
            else:
                content = str(output)
                reasoning = ""
            
            # 提取智能体类型
            agent_type = name.split("_")[0]
            
            part = f"### {agent_type.upper()}\n"
            if reasoning:
                part += f"**推理过程**: {reasoning}\n\n"
            part += f"**输出内容**:\n{content}\n"
            
            parts.append(part)
        
        return "\n---\n".join(parts)
    
    def _format_reviews(self, notes: List[str]) -> str:
        """
        格式化审核意见
        
        Args:
            notes: 审核意见列表
            
        Returns:
            格式化的字符串
        """
        if not notes:
            return "（无审核意见）"
        
        return "\n".join(f"- {note}" for note in notes)
    
    def _save_if_needed(
        self,
        content: str,
        state: Dict[str, Any]
    ) -> Optional[str]:
        """
        如果需要，保存结果到文件
        
        Args:
            content: 要保存的内容
            state: 当前状态
            
        Returns:
            保存结果消息，不需要保存返回 None
        """
        original_task = state.get("original_task", "").lower()
        
        # 检查是否需要保存文件
        save_keywords = ["保存", "写入", "输出到文件", "save", "write to file"]
        needs_save = any(kw in original_task for kw in save_keywords)
        
        if not needs_save:
            # 检查内容中是否包含代码，代码通常需要保存
            if "```python" in content and "爬虫" in original_task:
                needs_save = True
        
        if not needs_save:
            return None
        
        # 确定文件名
        if "json" in original_task:
            filename = "output.json"
        elif "python" in original_task or "代码" in original_task:
            filename = "output.py"
        else:
            filename = "output.txt"
        
        # 尝试保存
        file_tool = next(
            (t for t in self.tools if t.name == "file_manager"),
            None
        )
        
        if file_tool is None:
            return None
        
        try:
            # 提取要保存的内容
            save_content = self._extract_saveable_content(content)
            
            result = self.call_tool(
                "file_manager",
                action="write",
                path=filename,
                content=save_content,
            )
            
            return f"📁 结果已保存至: workspace/{filename}"
            
        except Exception as e:
            self.logger.warning(f"保存文件失败: {e}")
            return f"⚠️ 保存文件失败: {str(e)}"
    
    def _extract_saveable_content(self, content: str) -> str:
        """
        提取可保存的内容（如代码块）
        
        Args:
            content: 完整内容
            
        Returns:
            可保存的内容
        """
        import re
        
        # 尝试提取代码块
        code_pattern = r'```(?:python)?\s*([\s\S]*?)```'
        code_matches = re.findall(code_pattern, content)
        
        if code_matches:
            # 返回所有代码块
            return "\n\n".join(code_matches)
        
        # 尝试提取 JSON
        json_pattern = r'```json\s*([\s\S]*?)```'
        json_matches = re.findall(json_pattern, content)
        
        if json_matches:
            return json_matches[0]
        
        # 返回原始内容
        return content