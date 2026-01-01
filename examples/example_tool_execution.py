"""
示例 2：工具执行
================

演示系统的工具调用功能。

运行方式：
    python -m examples.example_tool_execution
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import get_settings
from src.graph.builder import MultiAgentSystem
from src.utils.logger import setup_logger
from src.tools import get_all_tools


def main():
    """运行工具执行示例"""
    console = Console()
    
    setup_logger(debug=False)
    
    console.print(Panel(
        "[bold blue]示例 2: 工具执行[/bold blue]\n\n"
        "演示系统的工具调用能力",
        title="Multi-Agent System Demo"
    ))
    
    # 显示可用工具
    console.print("\n[bold]可用工具:[/bold]")
    tools = get_all_tools()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("工具名", style="cyan")
    table.add_column("描述")
    
    for tool in tools:
        table.add_row(tool.name, tool.description[:60] + "...")
    
    console.print(table)
    
    # 示例任务 - 使用计算器
    task1 = "请计算 (25 * 4) + (100 / 5) - 17 的结果"
    
    console.print(f"\n[bold]任务 1:[/bold] {task1}")
    console.print("[dim]正在执行...[/dim]")
    
    try:
        settings = get_settings()
        settings.max_iterations = 3
        
        system = MultiAgentSystem(settings=settings)
        result = system.run(task1, task_id="calc_demo")
        
        # 显示工具调用日志
        tool_logs = result.get("tool_call_logs", [])
        if tool_logs:
            console.print("\n[bold]工具调用记录:[/bold]")
            for log in tool_logs:
                status = "✅" if log.get("success") else "❌"
                console.print(
                    f"  {status} {log.get('tool_name')}: "
                    f"{log.get('output', 'N/A')}"
                )
        
        final_answer = result.get("final_answer", "")
        if final_answer:
            console.print(f"\n[bold green]结果:[/bold green] {final_answer[:500]}")
        
    except Exception as e:
        console.print(f"[red]执行出错: {e}[/red]")
    
    # 示例任务 - 文件操作
    console.print("\n" + "-" * 50)
    
    task2 = "请在 workspace 目录创建一个 test.txt 文件，内容为 'Hello, Multi-Agent System!'"
    
    console.print(f"\n[bold]任务 2:[/bold] {task2}")
    console.print("[dim]正在执行...[/dim]")
    
    try:
        result = system.run(task2, task_id="file_demo")
        
        tool_logs = result.get("tool_call_logs", [])
        if tool_logs:
            console.print("\n[bold]工具调用记录:[/bold]")
            for log in tool_logs:
                status = "✅" if log.get("success") else "❌"
                console.print(
                    f"  {status} {log.get('tool_name')}: "
                    f"{str(log.get('output', 'N/A'))[:100]}"
                )
        
        final_answer = result.get("final_answer", "")
        if final_answer:
            console.print(f"\n[bold green]结果:[/bold green] {final_answer[:500]}")
        
    except Exception as e:
        console.print(f"[red]执行出错: {e}[/red]")
    
    console.print("\n[dim]示例执行完成[/dim]")


if __name__ == "__main__":
    main()