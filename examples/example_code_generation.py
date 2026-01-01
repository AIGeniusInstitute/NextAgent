"""
示例 3：代码生成
================

演示系统的代码生成和执行能力。

运行方式：
    python -m examples.example_code_generation
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from src.config.settings import get_settings
from src.graph.builder import MultiAgentSystem
from src.utils.logger import setup_logger


def main():
    """运行代码生成示例"""
    console = Console()
    
    setup_logger(debug=False)
    
    console.print(Panel(
        "[bold blue]示例 3: 代码生成[/bold blue]\n\n"
        "演示系统生成和执行代码的能力",
        title="Multi-Agent System Demo"
    ))
    
    # 代码生成任务
    task = """
    请帮我编写一个 Python 爬虫，抓取 Hacker News 首页的文章标题和链接，
    并保存为 JSON 文件。
    
    要求：
    1. 使用 requests 和 BeautifulSoup 库
    2. 抓取首页前 10 篇文章的标题和链接
    3. 保存到 workspace/hn_articles.json
    4. 包含错误处理
    """
    
    console.print(f"\n[bold]任务描述:[/bold]\n{task}")
    console.print("\n[dim]正在处理（这可能需要一些时间）...[/dim]\n")
    
    try:
        settings = get_settings()
        settings.max_iterations = 8
        
        system = MultiAgentSystem(settings=settings)
        
        # 流式执行以显示进度
        console.print("[bold]执行进度:[/bold]")
        
        final_state = None
        for event in system.stream(task, task_id="code_gen_demo"):
            for node_name, node_output in event.items():
                if node_name != "__end__":
                    console.print(f"  ▶ 执行节点: [cyan]{node_name}[/cyan]")
                final_state = node_output
        
        if final_state is None:
            console.print("[yellow]未获取到执行结果[/yellow]")
            return
        
        console.print("\n" + "=" * 60)
        console.print("[bold green]执行完成[/bold green]")
        console.print("=" * 60)
        
        # 显示生成的代码
        agent_outputs = final_state.get("agent_outputs", {})
        
        for key, output in agent_outputs.items():
            if "coder" in key.lower():
                console.print("\n[bold]生成的代码:[/bold]")
                
                if isinstance(output, dict):
                    code_content = output.get("output", "")
                else:
                    code_content = str(output)
                
                # 尝试提取代码块
                import re
                code_match = re.search(
                    r'```python\s*([\s\S]*?)```',
                    code_content
                )
                
                if code_match:
                    code = code_match.group(1)
                    syntax = Syntax(
                        code,
                        "python",
                        theme="monokai",
                        line_numbers=True
                    )
                    console.print(Panel(syntax, title="Python 代码"))
                break
        
        # 显示最终答案
        final_answer = final_state.get("final_answer", "")
        if final_answer:
            console.print("\n[bold]完整结果:[/bold]")
            console.print(Panel(
                Markdown(final_answer[:3000]),
                title="执行结果",
                border_style="green"
            ))
        
        # 显示工具调用
        tool_logs = final_state.get("tool_call_logs", [])
        if tool_logs:
            console.print("\n[bold]工具调用记录:[/bold]")
            for log in tool_logs:
                status = "✅" if log.get("success") else "❌"
                console.print(
                    f"  {status} {log.get('tool_name')} "
                    f"({log.get('duration_ms', 0):.0f}ms)"
                )
        
        # 显示统计
        console.print("\n[bold]执行统计:[/bold]")
        console.print(f"  迭代次数: {final_state.get('iteration_count', 0)}")
        console.print(f"  子任务数: {len(final_state.get('subtasks', []))}")
        console.print(f"  工具调用: {len(tool_logs)} 次")
        
        exec_time = final_state.get("execution_time", {})
        total_time = sum(exec_time.values())
        console.print(f"  总耗时: {total_time:.2f} 秒")
        
    except Exception as e:
        console.print(f"[red]执行出错: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    main()