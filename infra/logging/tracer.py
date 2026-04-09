import time
from functools import wraps
from rich.panel import Panel
from rich import box
from infra.logging.formatters import console

def trace(component_name):
    def decorator(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            start = time.time()

            console.print(
                Panel.fit(
                    f"[bold cyan]START[/bold cyan] → {component_name}",
                    box=box.ROUNDED
                )
            )

            result = func(state, *args, **kwargs)

            duration = round(time.time() - start, 3)

            console.print(
                Panel.fit(
                    f"[bold green]END[/bold green] → {component_name} ({duration}s)",
                    box=box.ROUNDED
                )
            )

            return result
        return wrapper
    return decorator