.PHONY: gitleaks commit-checks help

help:
	@echo "gitleaks        — scan repo for leaked secrets"
	@echo "commit-checks   — run all pre-commit hooks on all files"

.git/hooks/pre-commit:
	pre-commit install

gitleaks: .git/hooks/pre-commit
	pre-commit run gitleaks --all-files

commit-checks: .git/hooks/pre-commit
	pre-commit run --all-files
