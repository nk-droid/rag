from pathlib import Path
from rich.console import Console
from argparse import ArgumentParser

from components.evaluation.dataset import check_metric_requirements, load_dataset
from components.evaluation.metrics import aggregate
from components.evaluation.ragas_metrics import RAGAS_METRIC_NAMES, ragas_aggregate_batch
from pipeline.config import load_config
from infra.storage.experiment_store import ExperimentStore
from pipeline.experiment.config import Experiment, load_experiment
from pipeline.experiment.report import build_comparison, render_console, render_markdown
from pipeline.experiment.runner import run_experiment

console = Console()

def _compute_and_store(
    store: ExperimentStore,
    run_dir: Path,
    metric_names: list[str],
) -> dict:
    lexical_names = [n for n in metric_names if n not in RAGAS_METRIC_NAMES]
    ragas_names = [n for n in metric_names if n in RAGAS_METRIC_NAMES]

    variants = store.list_variants(run_dir)
    records_by_variant = {v: store.load_runs(run_dir, v) for v in variants}
    metrics_by_variant = {
        v: aggregate(records_by_variant[v], lexical_names) for v in variants
    }

    if ragas_names:
        ragas_config = load_config([])  # base.yaml → evaluation.ragas settings
        ragas_by_variant = ragas_aggregate_batch(
            records_by_variant, ragas_names, config=ragas_config
        )
        for v in variants:
            metrics_by_variant[v].update(ragas_by_variant.get(v, {}))

    for v in variants:
        store.write_variant_metrics(run_dir, v, metrics_by_variant[v])

    comparison = build_comparison(run_dir, metric_names, store=store)
    store.write_comparison(run_dir, comparison)
    (run_dir / "comparison.md").write_text(render_markdown(comparison), encoding="utf-8")
    return comparison

def _metric_names(run_dir: Path, store: ExperimentStore, override: list[str] | None) -> list[str]:
    if override:
        return override
    manifest = store.load_manifest(run_dir)
    metrics = manifest.get("experiment", {}).get("metrics")
    return list(metrics) if metrics else []

def cmd_run(args) -> None:
    experiment: Experiment = load_experiment(args.experiment)
    samples = load_dataset(experiment.dataset)

    for warning in check_metric_requirements(samples, experiment.metrics):
        console.print(f"[yellow]warning:[/yellow] {warning}")

    console.print(
        f"[bold]{experiment.name}[/bold]: {len(experiment.variants)} variants "
        f"x {len(samples)} questions (parallelism={experiment.parallelism})"
    )

    experiment_dict = experiment.to_dict()
    sample_records = [s.to_record() for s in samples]
    results = run_experiment(experiment_dict, sample_records)

    store = ExperimentStore(args.root)
    run_dir = store.create_run(experiment_dict)
    for result in results:
        store.write_variant_runs(run_dir, result)
        if result.get("error"):
            console.print(f"[red]{result['variant']}: {result['error']}[/red]")

    comparison = _compute_and_store(store, run_dir, experiment.metrics)
    render_console(comparison, console)
    console.print(f"\n[cyan]Run stored at[/cyan] {run_dir}")

def cmd_metrics(args) -> None:
    run_dir = Path(args.run_dir)
    store = ExperimentStore(args.root)
    metric_names = _metric_names(run_dir, store, args.metrics)
    if not metric_names:
        console.print("[red]No metrics specified and none found in manifest.[/red]")
        raise SystemExit(1)
    comparison = _compute_and_store(store, run_dir, metric_names)
    render_console(comparison, console)
    console.print(f"\n[cyan]Updated metrics at[/cyan] {run_dir}")

def cmd_report(args) -> None:
    run_dir = Path(args.run_dir)
    store = ExperimentStore(args.root)
    metric_names = _metric_names(run_dir, store, args.metrics)
    comparison = build_comparison(run_dir, metric_names, store=store)
    render_console(comparison, console)

def main() -> None:
    parser = ArgumentParser(description="Run and compare RAG pipeline variants.")
    parser.add_argument(
        "--root", default="data/experiments", help="Experiment storage root."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an experiment and compute metrics.")
    run.add_argument("--experiment", required=True, help="Path to experiment YAML.")
    run.set_defaults(func=cmd_run)

    metrics = sub.add_parser("metrics", help="Recompute metrics from stored runs.")
    metrics.add_argument("run_dir", help="Path to a stored run directory.")
    metrics.add_argument("--metrics", nargs="*", default=None, help="Override metric list.")
    metrics.set_defaults(func=cmd_metrics)

    report = sub.add_parser("report", help="Print comparison from stored metrics.")
    report.add_argument("run_dir", help="Path to a stored run directory.")
    report.add_argument("--metrics", nargs="*", default=None, help="Override metric list.")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()