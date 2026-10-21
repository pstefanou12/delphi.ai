# Author: pstefanou12@
# Task runner for delphi.ai.
#
# Every recipe goes through poetry so the environment is identical locally,
# in pre-commit, and in CI. Do not invoke ruff/pylint/mypy/pytest directly.

set shell := ["bash", "-euo", "pipefail", "-c"]
# Forward recipe arguments through "$@" so a quoted one survives, which
# {{ARGS}} does not: it splits on whitespace and turns -m "not slow" into two.
set positional-arguments

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
    poetry run ruff format delphi/

# Check formatting without modifying files.
fmt-check:
    poetry run ruff format --check delphi/

# Lint and apply safe fixes.
lint:
    poetry run ruff check --fix delphi/

# Lint without modifying files.
lint-check:
    poetry run ruff check delphi/

# Detect logical errors with pylint.
pylint:
    poetry run pylint --errors-only delphi/

# Type-check with mypy.
typecheck:
    poetry run mypy delphi/

# Run the tests affected by local changes, one per core. pytest-testmon
# records the code each test executes and reruns only tests whose code changed
# since the last run; the first run, and every run in CI, runs the whole suite.
# `--testmon-noselect` forces a full local run, `-m "not slow"` skips the long
# fits, and `-n 0` runs serially, which `-s` needs. Extra arguments are
# forwarded to pytest.
test *ARGS:
    # Python 3.14's default sys.monitoring core credits a line only to the
    # first test that runs it, so testmon would miss most dependencies.
    if [ -n "${CI:-}" ]; then poetry run pytest -n auto --dist worksteal "$@"; else COVERAGE_CORE=ctrace poetry run pytest --testmon -n auto --dist worksteal "$@"; fi

# Branch coverage of delphi/. Extra arguments are forwarded to pytest.
cov *ARGS:
    poetry run pytest --cov=delphi --cov-branch \
        --cov-report=term-missing "$@"

# Install the pre-push git hooks.
hooks:
    poetry run pre-commit install --hook-type pre-push

# Static gate: exactly what pre-commit runs on push. Does not run the suite.
check: fmt-check lint-check pylint typecheck

# Full gate: the static checks plus the test suite.
ci: check test

# Remove tool caches.
clean:
    rm -rf .mypy_cache .pytest_cache .ruff_cache
    find . -type d -name __pycache__ -prune -exec rm -rf {} +
