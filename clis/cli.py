from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.json import JSON

from components.ingestion.repo_cloner import RepoCloneError, RepoCloner, RepoClonerSettings
from pipeline.config import load_config
from pipeline.orchestrator import RAGOrchestrator
from pipeline.results import extract_answer
from pipeline.validator import validate_config
from pipeline.workspace import apply_workspace

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "configs" / "pipeline"

def _config_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))

def _get_query() -> str:
    return console.input("[yellow]Enter your query: [/yellow]").strip()

def _get_source_path() -> str:
    return console.input("[yellow]Enter source directory/file path: [/yellow]").strip()

def _list_pipelines() -> None:
    console.print("[bold cyan]Available pipelines[/bold cyan]")
    for path in sorted(PIPELINE_DIR.glob("*.yaml")):
        console.print(f"  - {path.stem}")

def _require_config(*parts: str) -> str:
    path = _config_path(*parts)
    if not Path(path).exists():
        console.print(f"[red]Config not found:[/red] {Path(path).relative_to(REPO_ROOT)}")
        if parts[1] == "pipeline":
            console.print("Run [cyan]--list-pipelines[/cyan] to see available pipelines.")
        raise SystemExit(1)
    return path

def _build_config(args) -> dict[str, Any]:
    pipeline_name = args.pipeline
    if pipeline_name is None:
        pipeline_name = "repo_hybrid_graph" if args.repo_url else "custom"

    config = load_config(
        [
            _require_config("configs", "pipeline", f"{pipeline_name}.yaml"),
            _require_config("configs", "runtime", f"{args.runtime}.yaml"),
            _require_config("configs", "env", f"{args.env}.yaml"),
        ]
    )
    if args.save_intermediate:
        config.setdefault("intermediate", {})["enabled"] = True

    return apply_workspace(config)

def _validate_or_exit(config: dict[str, Any]) -> None:
    errors = validate_config(config)
    if not errors:
        return
    console.print("[red]Pipeline config is invalid:[/red]")
    for error in errors:
        console.print(f"  - {error}")
    raise SystemExit(1)

def _resolve_sources(args) -> list[str]:
    if args.repo_url:
        cloner = RepoCloner(RepoClonerSettings())

        try:
            checkout = cloner.clone_or_update(
                repo_url=args.repo_url,
                branch=args.branch,
                source_id=args.source_id,
                access_token=args.access_token,
            )
        except RepoCloneError as error:
            console.print(f"[red]Repo clone failed:[/red] {error}")
            raise SystemExit(1) from error

        console.print("[green]Repository ready[/green]")
        console.print(f"Source ID: [cyan]{checkout.source_id}[/cyan]")
        console.print(f"Branch: [cyan]{checkout.branch}[/cyan]")
        console.print(f"Commit: [cyan]{checkout.commit_sha}[/cyan]")
        console.print(f"Path: [cyan]{checkout.working_tree}[/cyan]")

        return [str(checkout.working_tree)]

    if args.source:
        return [args.source]

    source = _get_source_path()
    if not source:
        console.print("[red]Source path is required when --repo-url is not provided.[/red]")
        raise SystemExit(1)

    return [source]

def _chunk_path(chunk: Any) -> str:
    metadata = getattr(chunk, "metadata", {}) or {}
    return str(
        metadata.get("relative_path")
        or metadata.get("path")
        or metadata.get("source")
        or "unknown"
    )

def _unique_paths(chunks: list[Any], limit: int = 10) -> list[str]:
    seen = set()
    paths = []
    for chunk in chunks:
        path = _chunk_path(chunk)
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
        if len(paths) >= limit:
            break
    return paths

def _print_evidence_summary(state: dict[str, Any]) -> None:
    retrieved = state.get("retrieved", []) or []
    expanded = state.get("graph_expanded", []) or []

    console.print("\n[bold cyan]Evidence summary[/bold cyan]")
    console.print(f"Retrieved chunks: {len(retrieved)}")
    console.print(f"Graph-expanded chunks: {len(expanded)}")

    console.print("\n[bold]Top retrieved files[/bold]")
    for path in _unique_paths(retrieved):
        console.print(f"  - {path}")

    console.print("\n[bold]Top graph-expanded files[/bold]")
    for path in _unique_paths(expanded):
        console.print(f"  - {path}")

def main(args) -> None:
    if args.list_pipelines:
        _list_pipelines()
        return

    config = _build_config(args)
    _validate_or_exit(config)

    if args.validate_only:
        console.print("[green]Pipeline config is valid.[/green]")
        console.print(f"Workspace: [cyan]{config.get('workspace', {}).get('path')}[/cyan]")
        return

    sources = _resolve_sources(args)
    query = args.query or _get_query()

    if not query:
        console.print("[red]Query is required.[/red]")
        raise SystemExit(1)

    orchestrator = RAGOrchestrator(config)

    state: dict[str, Any] = {
        "sources": sources,
        "query": query,
    }

    if args.top_k is not None:
        state["top_k"] = args.top_k
    if args.run_id:
        state["intermediate_run_id"] = args.run_id

    if not args.skip_init:
        state = orchestrator.initialize(state)
    else:
        console.print("[yellow]Skipping init pipeline. Existing indexes will be used.[/yellow]")

    state["query"] = query
    state["sources"] = sources

    state = orchestrator.run(state)

    if args.show_state or args.repo_url:
        _print_evidence_summary(state)

    answer = extract_answer(state)
    console.print("\n[bold green]Answer[/bold green]")
    console.print(answer)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(answer, encoding="utf-8")
        console.print(f"\n[bold cyan]Answer written to[/bold cyan] {output_path}")

    if args.show_state:
        console.print("\n[bold cyan]Final state[/bold cyan]")
        console.print(
            JSON.from_data(
                {
                    "workspace": config.get("workspace", {}).get("id"),
                    "retriever": state.get("retriever"),
                    "ranker": state.get("ranker"),
                    "chunker": state.get("chunker"),
                    "indexer": state.get("indexer"),
                    "indexed_count": state.get("indexed_count"),
                    "retrieved_count": len(state.get("retrieved", []) or []),
                    "graph_expanded_count": len(state.get("graph_expanded", []) or []),
                    "init_skipped": state.get("init_skipped"),
                    "intermediate_run_id": state.get("intermediate_run_id"),
                    "intermediate_path": state.get("intermediate_path"),
                }
            )
        )

    if state.get("intermediate_path"):
        console.print(f"\n[bold cyan]Intermediate[/bold cyan] {state['intermediate_path']}")

def cli_main() -> None:
    parser = ArgumentParser(description="Run modular RAG pipelines from CLI.")

    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        help="Pipeline config name. Defaults to repo_hybrid_graph for --repo-url, otherwise custom.",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default="cli",
        help="Runtime config name.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        help="Environment config name.",
    )

    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List available pipeline configs and exit.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the composed pipeline config and exit without running.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Local file or directory source.",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default=None,
        help="Git repository URL to clone/index.",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Git branch/ref to use for --repo-url.",
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default=None,
        help="Stable local source ID for cloned repo, e.g. autopr.",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        default=None,
        help="Optional GitHub token for private HTTPS repos. Do not commit or log this.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to ask. If omitted, CLI prompts interactively.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override retrieval top_k.",
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip indexing/init and use existing indexes.",
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Print useful final state metadata.",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Write step-by-step snapshots under data/intermediate.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID/folder name for intermediate snapshots.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the final answer/result to this file path.",
    )

    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
