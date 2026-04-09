from argparse import ArgumentParser
from pipeline.config import load_config
from pipeline.orchestrator import RAGOrchestrator
# from infra.logging.logger import get_logger
from rich.console import Console
from rich.json import JSON

console = Console()
# logger = get_logger(__name__, "DEBUG")

def main(args):
    config = load_config([
        f"configs/pipeline/{args.pipeline}.yaml",
        f"configs/runtime/{args.runtime}.yaml",
        f"configs/env/{args.env}.yaml"
    ])
    
    # console.print(JSON.from_data(config))

    orchestrator = RAGOrchestrator(config)
    state = {
        "query": "Give one search method with the details of its working. How it can be used in a RAG system?",
        "sources": "/Users/nidhishkumar/Personal/rag/data/raw/docs"
    }
    state = orchestrator.initialize(state)

    state = orchestrator.run(state)
    print(state["parsed_output"].answer)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pipeline", type=str, help="which pipeline to use")
    parser.add_argument("--runtime", type=str, help="which runtime to use")
    parser.add_argument("--env", type=str, help="which env to use")

    args = parser.parse_args()
    main(args)
