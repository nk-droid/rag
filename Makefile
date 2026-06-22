setup:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov --cov-report=term-missing --cov-fail-under=90

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-e2e:
	pytest -m e2e

validate:
	rag --pipeline simple --validate-only
	rag --pipeline custom --validate-only
	rag --pipeline advanced --validate-only
	rag --pipeline repo_hybrid_graph --validate-only

demo-doc:
	rag --source data/raw/docs-short --pipeline custom --query "Summarize the main ideas in these documents"

demo-repo:
	rag --repo-url https://github.com/nk-droid/AutoPR --source-id autopr --pipeline repo_hybrid_graph --show-state --query "If Redis is removed, which exact files need to change and why?"

eval:
	rag-eval run --experiment configs/experiments/example.yaml

clone-autopr:
	python -c "from components.ingestion.repo_cloner import RepoCloner, RepoClonerSettings; c = RepoCloner(RepoClonerSettings()).clone_or_update(repo_url='https://github.com/nk-droid/AutoPR', branch='main', source_id='autopr'); print('Cloned', c.commit_sha, 'to', c.working_tree)"

eval-autopr: clone-autopr
	rag-eval run --experiment configs/experiments/autopr.yaml

clean:
	rm -rf data/repos data/workspaces data/indices data/intermediate data/experiments data/uploads
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete