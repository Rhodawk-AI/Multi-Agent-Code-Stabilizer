.PHONY: install test lint run bootstrap audit stabilize clean

install:
	pip install pdm && pdm install

test:
	pdm run pytest tests/ -v --tb=short

lint:
	pdm run ruff check .
	pdm run mypy .

bootstrap:
	pdm run rhodawk-ai-code-stabilizer bootstrap $(REPO_PATH)

audit:
	pdm run rhodawk-ai-code-stabilizer audit $(REPO_URL) --path $(REPO_PATH) --output audit_report.md

stabilize:
	pdm run rhodawk-ai-code-stabilizer stabilize $(REPO_URL) --path $(REPO_PATH)

status:
	pdm run rhodawk-ai-code-stabilizer status $(REPO_PATH)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
