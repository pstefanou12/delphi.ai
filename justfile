# Author: pstefanou12@
# Task runner for delphi.ai.
#
# Every recipe goes through poetry so the environment is identical locally,
# in pre-commit, and in CI. Do not invoke ruff/pylint/mypy/pytest directly.

set shell := ["bash", "-euo", "pipefail", "-c"]

# List the available recipes.
default:
    @just --list

# Install the project together with its dev dependencies.
install:
    poetry install --with dev

# Re-resolve dependencies and rewrite poetry.lock.
lock:
    poetry lock

# Sync the environment to poetry.lock exactly, removing stale packages.
sync:
    poetry sync --with dev

# Format the codebase in place.
fmt:
    poetry run ruff format delphi/ tests/

# Check formatting without modifying files.
fmt-check:
    poetry run ruff format --check delphi/ tests/

# Lint and apply safe fixes.
lint:
    poetry run ruff check --fix delphi/ tests/

# Lint without modifying files.
lint-check:
    poetry run ruff check delphi/ tests/

# Detect logical errors with pylint.
pylint:
    poetry run pylint --errors-only delphi/ tests/

# Type-check with mypy.
typecheck:
    poetry run mypy delphi/ tests/

# Run the test suite. Extra arguments are forwarded to pytest.
test *ARGS:
    poetry run pytest {{ARGS}}

# Branch coverage of delphi/. Extra arguments are forwarded to pytest.
cov *ARGS:
    poetry run pytest --cov=delphi --cov-branch \
        --cov-report=term-missing {{ARGS}}

# Install the pre-push git hooks.
hooks:
    poetry run pre-commit install --hook-type pre-push

# Static gate: exactly what pre-commit runs on push. Excludes tests.
check: fmt-check lint-check pylint typecheck

# Full gate: the static checks plus the test suite.
ci: check test

# Remove tool caches.
clean:
    rm -rf .mypy_cache .pytest_cache .ruff_cache
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
