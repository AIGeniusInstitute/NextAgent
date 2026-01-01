"""
示例 1：任务规划分解
====================

演示系统如何分解和规划复杂任务。

运行方式：
    python -m examples.example_planning
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.config.settings import get_settings
from src.graph.builder import MultiAgentSystem
from src.utils.logger import setup_logger
from src.utils.visualizer import ExecutionVisualizer


def main():
    """运行任务规划示例"""
    console = Console()
    
    # 设置日志
    setup_logger(debug=False)
    
    console.print(Panel(
        "[bold blue]示例 1: 任务规划分解[/bold blue]\n\n"
        "演示系统如何理解和分解复杂任务",
        title="Multi-Agent System Demo"
    ))
    
    # 示例任务
    task = """
    请帮我制定一个学习 Python 的完整计划，包括：
    1. 学习路径和阶段划分
    2. 每个阶段的学习内容和目标
    3. 推荐的学习资源
    4. 预计时间安排
    5. 实践项目建议
    """
    
    console.print(f"\n[bold]任务描述:[/bold]\n{task}")
    console.print("\n[dim]正在处理...[/dim]\n")
    
    try:
        # 初始化系统
        settings = get_settings()
        settings.max_iterations = 5  # 限制迭代次数
        
        system = MultiAgentSystem(settings=settings)
        
        # 执行任务
        result = system.run(task, task_id="planning_demo")
        
        # 显示结果
        console.print("\n" + "=" * 60)
        console.print("[bold green]执行完成[/bold green]")
        console.print("=" * 60)
        
        # 显示子任务分解
        subtasks = result.get("subtasks", [])
        if subtasks:
            console.print("\n[bold]任务分解结果:[/bold]")
            for i, task_item in enumerate(subtasks, 1):
                status_icon = "✅" if task_item.get("status") == "completed" else "⏳"
                console.print(
                    f"  {status_icon} {i}. {task_item.get('name', 'Unknown')}"
                )
                console.print(
                    f"      类型: {task_item.get('task_type', 'N/A')} | "
                    f"执行者: {task_item.get('assigned_agent', 'N/A')}"
                )
        
        # 显示最终答案
        final_answer = result.get("final_answer", "")
        if final_answer:
            console.print("\n[bold]最终答案:[/bold]")
            console.print(Panel(
                Markdown(final_answer),
                title="学习计划",
                border_style="green"
            ))
        
        # 显示执行统计
        console.print("\n[bold]执行统计:[/bold]")
        console.print(f"  迭代次数: {result.get('iteration_count', 0)}")
        console.print(f"  子任务数: {len(subtasks)}")
        
        exec_time = result.get("execution_time", {})
        total_time = sum(exec_time.values())
        console.print(f"  总耗时: {total_time:.2f} 秒")
        
        # 可视化
        visualizer = ExecutionVisualizer()
        console.print("\n[bold]执行轨迹:[/bold]")
        trace = visualizer.generate_text_trace(result)
        console.print(trace)
        
    except Exception as e:
        console.print(f"[red]执行出错: {e}[/red]")
        raise


if __name__ == "__main__":
    main()