from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import time

from infra.logging.recent_logs import add_log, clear_logs, get_recent_logs, set_refresh_callback
from infra.logging.runtime.base import Runtime

console = Console()

class RichRuntime(Runtime):
    def __init__(self):
        self.progress = self._create_progress()
        self.tasks = {}
        self.live: Live | None = None

    def _create_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            auto_refresh=False,
        )

    def _render(self):
        lines = get_recent_logs()
        if lines:
            logs_renderable = "\n".join(lines)
        else:
            logs_renderable = "[dim]No logs yet[/dim]"
        return Group(
            self.progress,
            Panel(logs_renderable, title="Recent Logs (last 3)", border_style="cyan"),
        )

    def _refresh(self):
        if self.live is not None:
            self.live.update(self._render(), refresh=True)

    def start(self, msg: str):
        # Each pipeline phase (initialize/run) should render its own fresh task list.
        self.progress = self._create_progress()
        self.tasks = {}
        clear_logs()
        set_refresh_callback(self._refresh)
        console.print(f"\n[bold cyan]🚀 {msg}...[/bold cyan]\n")
        self.live = Live(self._render(), console=console, auto_refresh=False, transient=False)
        self.live.start()
        self._refresh()

    def stop(self, msg: str = None):
        set_refresh_callback(None)
        if self.live is not None:
            self._refresh()
            self.live.stop()
            self.live = None
        if msg:
            console.print(f"\n[bold green]✅ {msg}[/bold green]\n")

    def add_step(self, name):
        task_id = self.progress.add_task(name, total=1, start=False)
        self.tasks[name] = task_id
        self._refresh()

    def run_step(self, name, func, *args, **kwargs):
        task_id = self.tasks[name]
        self.progress.start_task(task_id)
        self._refresh()

        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = round(time.time() - start, 2)
            self.progress.update(
                task_id,
                description=f"[green]✔ {name} ({duration}s)",
                completed=1,
            )
            self.progress.stop_task(task_id)
            self._refresh()
            return result
        except Exception:
            duration = round(time.time() - start, 2)
            self.progress.update(
                task_id,
                description=f"[red]✖ {name} ({duration}s)",
                completed=1,
            )
            self.progress.stop_task(task_id)
            self._refresh()
            raise

    def log(self, message: str):
        add_log(str(message))
