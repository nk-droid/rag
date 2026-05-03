from argparse import ArgumentParser
from pipeline.config import load_config
from pipeline.orchestrator import RAGOrchestrator
# from infra.logging.logger import get_logger
from rich.console import Console
from rich.json import JSON

console = Console()
# logger = get_logger(__name__, "DEBUG")

def _get_query():
    return console.input("[yellow]Enter your query: [/yellow]")

def _get_sources():
    return console.input("[yellow]Enter the source directory path: [/yellow]")

def main(args):
    config = load_config([
        f"configs/pipeline/{args.pipeline}.yaml",
        f"configs/runtime/{args.runtime}.yaml",
        f"configs/env/{args.env}.yaml"
    ])

    orchestrator = RAGOrchestrator(config)
    sources = _get_sources()
    state = {"sources": sources}
    state = orchestrator.initialize(state)

    state["query"] = _get_query()
    state["sources"] = sources

    state = orchestrator.run(state)
    console.print(f"[green]{state['parsed_output'].answer}[/green]")

    # if args.run_evaluation:
    #     from scripts.evaluate import run_evaluation
    #     evaluation_results = run_evaluation(orchestrator, evaluator)
    #     console.print(JSON.from_data(evaluation_results))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pipeline", type=str, help="which pipeline to use")
    parser.add_argument("--runtime", type=str, help="which runtime to use")
    parser.add_argument("--env", type=str, help="which env to use")
    parser.add_argument("-e", "--eval", action="store_true", help="whether to run evaluation after main")

    args = parser.parse_args()
    main(args)
