from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import time

from infra.logging.runtime.base import Runtime

console = Console()

class RichRuntime(Runtime):
    def __init__(self):
        self.progress = self._create_progress()
        self.tasks = {}

    def _create_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        )

    def start(self, msg: str):
        # Each pipeline phase (initialize/run) should render its own fresh task list.
        self.progress = self._create_progress()
        self.tasks = {}
        console.print(f"\n[bold cyan]🚀 {msg}...[/bold cyan]\n")
        self.progress.start()

    def stop(self, msg: str = None):
        self.progress.stop()
        if msg:
            console.print(f"\n[bold green]✅ {msg}[/bold green]\n")

    def add_step(self, name):
        task_id = self.progress.add_task(name, start=False)
        self.tasks[name] = task_id

    def run_step(self, name, func, *args, **kwargs):
        task_id = self.tasks[name]
        self.progress.start_task(task_id)

        start = time.time()
        result = func(*args, **kwargs)
        duration = round(time.time() - start, 2)

        self.progress.update(
            task_id,
            description=f"[green]✔ {name} ({duration}s)",
            completed=1
        )

        return result

    def log(self, message: str):
        console.print(message)
