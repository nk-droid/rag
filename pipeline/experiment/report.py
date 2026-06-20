from pathlib import Path
from typing import Any
from rich.table import Table
from rich.console import Console

from infra.storage.experiment_store import ExperimentStore

def build_comparison(
    run_dir: Path,
    metric_names: list[str],
    store: ExperimentStore | None = None,
) -> dict[str, Any]:
    store = store or ExperimentStore()
    variants = store.list_variants(run_dir)

    rows: dict[str, dict[str, Any]] = {}
    for variant in variants:
        metrics = store.load_metrics(run_dir, variant)
        rows[variant] = {name: metrics.get(name, {}) for name in metric_names}

    best: dict[str, str | None] = {}
    for name in metric_names:
        scored = [
            (variant, cells[name].get("value"))
            for variant, cells in rows.items()
            if isinstance(cells.get(name), dict) and cells[name].get("value") is not None
        ]
        if not scored:
            best[name] = None
            continue

        higher = next(
            (
                rows[v][name].get("higher_is_better", True)
                for v, _ in scored
                if rows[v][name]
            ),
            True,
        )
        best[name] = (max if higher else min)(scored, key=lambda item: item[1])[0]

    return {"metrics": metric_names, "variants": rows, "best": best}

def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}"
    return str(value)

def render_console(comparison: dict[str, Any], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title="Variant comparison", show_lines=False)
    table.add_column("variant", style="bold")
    for name in comparison["metrics"]:
        table.add_column(name, justify="right")

    for variant, cells in comparison["variants"].items():
        row = [variant]
        for name in comparison["metrics"]:
            value = cells.get(name, {}).get("value")
            text = _format(value)
            if comparison["best"].get(name) == variant and value is not None:
                text = f"[green]{text}[/green]"
            row.append(text)
        table.add_row(*row)

    console.print(table)

def render_markdown(comparison: dict[str, Any]) -> str:
    metrics = comparison["metrics"]
    header = "| variant | " + " | ".join(metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)
    lines = [header, sep]
    for variant, cells in comparison["variants"].items():
        values = []
        for name in metrics:
            value = cells.get(name, {}).get("value")
            text = _format(value)
            if comparison["best"].get(name) == variant and value is not None:
                text = f"**{text}**"
            values.append(text)
        lines.append(f"| {variant} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"