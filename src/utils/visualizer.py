"""
可视化工具模块
==============

提供执行过程可视化功能。
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionVisualizer:
    """
    执行过程可视化器
    
    生成执行流程的可视化表示。
    
    支持的格式：
    - Mermaid: 流程图标记语言
    - Text: 纯文本表示
    """
    
    def __init__(self):
        """初始化可视化器"""
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_mermaid(
        self,
        state: Dict[str, Any],
        include_details: bool = False
    ) -> str:
        """
        生成 Mermaid 流程图
        
        Args:
            state: 执行状态
            include_details: 是否包含详细信息
            
        Returns:
            Mermaid 格式字符串
        """
        lines = ["```mermaid", "flowchart TD"]
        
        # 添加开始节点
        lines.append("    START((开始))")
        
        # 从推理轨迹生成节点
        reasoning_trace = state.get("reasoning_trace", [])
        node_id = 0
        prev_node = "START"
        
        for step in reasoning_trace:
            node_id += 1
            node_name = f"N{node_id}"
            
            # 提取节点信息
            if "]" in step:
                agent = step.split("]")[0].strip("[")
                content = step.split("]")[1].strip()[:50]
            else:
                agent = "Unknown"
                content = step[:50]
            
            # 添加节点
            lines.append(f"    {node_name}[{agent}]")
            lines.append(f"    {prev_node} --> {node_name}")
            
            prev_node = node_name
        
        # 添加结束节点
        lines.append("    END((结束))")
        lines.append(f"    {prev_node} --> END")
        
        # 添加样式
        lines.append("")
        lines.append("    classDef coordinator fill:#e1f5fe")
        lines.append("    classDef planner fill:#fff3e0")
        lines.append("    classDef worker fill:#e8f5e9")
        lines.append("    classDef critic fill:#fce4ec")
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def generate_text_trace(
        self,
        state: Dict[str, Any],
        max_width: int = 80
    ) -> str:
        """
        生成文本格式的执行轨迹
        
        Args:
            state: 执行状态
            max_width: 最大宽度
            
        Returns:
            文本格式字符串
        """
        lines = []
        lines.append("=" * max_width)
        lines.append("执行轨迹".center(max_width))
        lines.append("=" * max_width)
        
        # 基本信息
        lines.append(f"任务ID: {state.get('task_id', 'N/A')}")
        lines.append(f"原始任务: {state.get('original_task', '')[:60]}...")
        lines.append(f"迭代次数: {state.get('iteration_count', 0)}")
        lines.append("-" * max_width)
        
        # 推理轨迹
        lines.append("推理过程:")
        for i, step in enumerate(state.get("reasoning_trace", []), 1):
            lines.append(f"  {i}. {step[:max_width-5]}")
        
        lines.append("-" * max_width)
        
        # 子任务状态
        subtasks = state.get("subtasks", [])
        if subtasks:
            lines.append("子任务状态:")
            for task in subtasks:
                status_icon = {
                    "completed": "✓",
                    "failed": "✗",
                    "pending": "○",
                    "running": "◐",
                }.get(task.get("status", ""), "?")
                lines.append(
                    f"  {status_icon} {task.get('name', 'Unknown')[:40]}"
                )
        
        lines.append("-" * max_width)
        
        # 执行时间
        exec_time = state.get("execution_time", {})
        if exec_time:
            lines.append("执行时间:")
            total = sum(exec_time.values())
            for agent, duration in exec_time.items():
                pct = (duration / total * 100) if total > 0 else 0
                bar_len = int(pct / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(f"  {agent:15} {bar} {duration:.2f}s ({pct:.1f}%)")
        
        lines.append("=" * max_width)
        
        return "\n".join(lines)
    
    def generate_summary(self, state: Dict[str, Any]) -> str:
        """
        生成执行摘要
        
        Args:
            state: 执行状态
            
        Returns:
            摘要字符串
        """
        lines = []
        
        # 状态
        has_answer = bool(state.get("final_answer"))
        status = "✅ 成功" if has_answer else "❌ 未完成"
        lines.append(f"状态: {status}")
        
        # 统计
        lines.append(f"迭代次数: {state.get('iteration_count', 0)}")
        lines.append(f"子任务数: {len(state.get('subtasks', []))}")
        lines.append(f"工具调用: {len(state.get('tool_call_logs', []))} 次")
        
        # 时间
        total_time = sum(state.get("execution_time", {}).values())
        lines.append(f"总耗时: {total_time:.2f}s")
        
        # Token
        token_usage = state.get("token_usage", {})
        if token_usage.get("total"):
            lines.append(f"Token消耗: {token_usage['total']}")
        
        return "\n".join(lines)


def generate_mermaid_graph(state: Dict[str, Any]) -> str:
    """
    便捷函数：生成 Mermaid 图
    
    Args:
        state: 执行状态
        
    Returns:
        Mermaid 格式字符串
    """
    visualizer = ExecutionVisualizer()
    return visualizer.generate_mermaid(state)


def print_execution_trace(state: Dict[str, Any]) -> None:
    """
    打印执行轨迹到控制台
    
    Args:
        state: 执行状态
    """
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    visualizer = ExecutionVisualizer()
    
    # 打印文本轨迹
    trace = visualizer.generate_text_trace(state)
    console.print(Panel(trace, title="执行轨迹"))
    
    # 打印摘要
    summary = visualizer.generate_summary(state)
    console.print(Panel(summary, title="执行摘要"))