"""
Multi-Agent Problem Solving System
===================================

基于 LangGraph 的通用多智能体协作问题求解系统。

主要特性：
- 自动任务理解与分解
- 多智能体协作执行
- 计划-执行-反思闭环
- 动态工具调用
- 人工介入支持
- 完整可观测性

使用示例：
    >>> from src import MultiAgentSystem
    >>> system = MultiAgentSystem()
    >>> result = system.run("请帮我编写一个 Python 爬虫")
    >>> print(result.answer)
"""

__version__ = "1.0.0"
__author__ = "Multi-Agent System Team"

from src.graph.builder import build_graph, MultiAgentSystem
from src.config.settings import Settings, get_settings
from src.graph.state import AgentState

__all__ = [
    "MultiAgentSystem",
    "build_graph",
    "Settings",
    "get_settings",
    "AgentState",
    "__version__",
]

"""
Multi-Agent System 主入口
=========================

提供命令行接口和程序入口点。
"""

import argparse
import sys
import time
import uuid
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

from src.config.settings import get_settings, Settings
from src.graph.builder import MultiAgentSystem
from src.graph.state import create_initial_state
from src.utils.logger import setup_logger, get_logger
from src.utils.visualizer import ExecutionVisualizer

console = Console()
logger = get_logger(__name__)

def print_banner() -> None:
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          Multi-Agent Problem Solving System v1.0             ║
║                 Powered by LangGraph                         ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")

def print_result(result: dict) -> None:
    console.print("\n")
    console.print(Panel(
        Markdown(result.get("final_answer", "无结果")),
        title="[bold green]✅ 执行结果[/bold green]",
        border_style="green",
    ))
    if "execution_time" in result:
        table = Table(title="执行指标", show_header=True, header_style="bold magenta")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        total_time = sum(result.get("execution_time", {}).values())
        table.add_row("总耗时", f"{total_time:.2f} 秒")
        table.add_row("迭代次数", str(result.get("iteration_count", 0)))
        token_usage = result.get("token_usage", {})
        if token_usage:
            table.add_row("Token 消耗", str(token_usage.get("total", 0)))
        console.print(table)

def print_reasoning_trace(result: dict) -> None:
    reasoning_trace = result.get("reasoning_trace", [])
    if reasoning_trace:
        console.print("\n[bold yellow]📝 推理轨迹：[/bold yellow]")
        for i, step in enumerate(reasoning_trace, 1):
            console.print(f"  {i}. {step}")

def interactive_mode(system: MultiAgentSystem, settings: Settings) -> None:
    console.print("\n[bold cyan]进入交互模式 (输入 'quit' 或 'exit' 退出)[/bold cyan]\n")
    visualizer = ExecutionVisualizer() if settings.enable_visualization else None
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]请输入您的任务[/bold green]")
            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[yellow]感谢使用，再见！[/yellow]")
                break
            if not user_input.strip():
                console.print("[yellow]输入不能为空，请重新输入[/yellow]")
                continue
            task_id = str(uuid.uuid4())[:8]
            console.print(f"\n[dim]任务ID: {task_id}[/dim]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("正在处理任务...", total=None)
                start_time = time.time()
                result = system.run(user_input, task_id=task_id)
                elapsed_time = time.time() - start_time
                progress.update(task, description=f"任务完成 (耗时 {elapsed_time:.2f}s)")
            print_result(result)
            if settings.debug_mode:
                print_reasoning_trace(result)
            if visualizer and settings.enable_visualization:
                try:
                    graph_output = visualizer.generate_mermaid(result)
                    if Confirm.ask("\n是否显示执行流程图？", default=False):
                        console.print(Panel(graph_output, title="执行流程图"))
                except Exception as e:
                    logger.warning(f"生成可视化图表失败: {e}")
        except KeyboardInterrupt:
            console.print("\n[yellow]操作已取消[/yellow]")
            continue
        except Exception as e:
            console.print(f"[red]执行出错: {e}[/red]")
            if settings.debug_mode:
                console.print_exception()
            continue

def single_task_mode(
    system: MultiAgentSystem,
    task: str,
    settings: Settings,
    output_file: Optional[str] = None
) -> None:
    task_id = str(uuid.uuid4())[:8]
    console.print(f"\n[dim]任务ID: {task_id}[/dim]")
    console.print(f"[bold]任务: {task}[/bold]\n")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        prog_task = progress.add_task("正在处理任务...", total=None)
        start_time = time.time()
        result = system.run(task, task_id=task_id)
        elapsed_time = time.time() - start_time
        progress.update(prog_task, description=f"任务完成 (耗时 {elapsed_time:.2f}s)")
    print_result(result)
    if settings.debug_mode:
        print_reasoning_trace(result)
    if output_file:
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            output_data = {
                "task_id": task_id,
                "original_task": task,
                "answer": result.get("final_answer", ""),
                "reasoning_trace": result.get("reasoning_trace", []),
                "execution_time": elapsed_time,
                "token_usage": result.get("token_usage", {}),
            }
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]结果已保存至: {output_file}[/green]")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Problem Solving System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互模式
  python -m src.main
  
  # 单任务模式
  python -m src.main --task "编写一个 Python 爬虫"
  
  # 带输出文件
  python -m src.main --task "分析数据" --output result.json
  
  # 调试模式
  python -m src.main --debug
        """
    )
    parser.add_argument("--task", "-t", type=str, help="要执行的任务描述")
    parser.add_argument("--output", "-o", type=str, help="输出结果文件路径 (JSON 格式)")
    parser.add_argument("--debug", "-d", action="store_true", help="启用调试模式")
    parser.add_argument("--config", "-c", type=str, help="配置文件路径")
    parser.add_argument("--no-human", "-n", action="store_true", help="禁用人工介入")
    parser.add_argument("--max-iterations", "-m", type=int, default=None, help="最大迭代次数")
    parser.add_argument("--version", "-v", action="version", version="Multi-Agent System v1.0.0")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    settings = get_settings()
    if args.debug:
        settings.debug_mode = True
    if args.no_human:
        settings.enable_human_in_loop = False
    if args.max_iterations:
        settings.max_iterations = args.max_iterations
    setup_logger(log_dir=settings.log_dir, debug=settings.debug_mode)
    print_banner()
    try:
        console.print("[dim]正在初始化系统...[/dim]")
        system = MultiAgentSystem(settings=settings)
        console.print("[green]✓ 系统初始化完成[/green]\n")
        if args.task:
            single_task_mode(system=system, task=args.task, settings=settings, output_file=args.output)
        else:
            interactive_mode(system=system, settings=settings)
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]程序被用户中断[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]程序异常: {e}[/red]")
        if settings.debug_mode:
            console.print_exception()
        return 1

if __name__ == "__main__":
    sys.exit(main())



    