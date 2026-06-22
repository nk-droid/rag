setup:
	pip install -e ".[dev]"

test:
	pytest

validate:
	python clis/cli.py --pipeline simple --validate-only
	python clis/cli.py --pipeline custom --validate-only
	python clis/cli.py --pipeline advanced --validate-only
	python clis/cli.py --pipeline repo_hybrid_graph --validate-only

demo-doc:
	python clis/cli.py --source data/raw/docs-short --pipeline custom --query "Summarize the main ideas in these documents"

demo-repo:
	python clis/cli.py --repo-url https://github.com/nk-droid/AutoPR --source-id autopr --pipeline repo_hybrid_graph --show-state --query "If Redis is removed, which exact files need to change and why?"

eval:
	python clis/eval_cli.py run --experiment configs/experiments/example.yaml

clone-autopr:
	python -c "from components.ingestion.repo_cloner import RepoCloner, RepoClonerSettings; c = RepoCloner(RepoClonerSettings()).clone_or_update(repo_url='https://github.com/nk-droid/AutoPR', branch='main', source_id='autopr'); print('Cloned', c.commit_sha, 'to', c.working_tree)"

eval-autopr: clone-autopr
	python clis/eval_cli.py run --experiment configs/experiments/autopr.yaml

clean:
	rm -rf data/repos data/workspaces data/indices data/intermediate data/experiments data/uploads
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete