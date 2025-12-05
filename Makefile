.PHONY: init check lint format test clean

# Initialize development environment
init:
	cd trainer && uv sync --dev
	cd trainer && uv run pre-commit install
	cd trainer && uv run pre-commit install --hook-type commit-msg

# Run all pre-commit checks
check:
	cd trainer && uv run pre-commit run --all-files

# Lint Python code
lint-python:
	cd trainer && uv run ruff check .

# Format code
format-python:
	cd trainer && uv run ruff format .

format-rust:
	cd agent && cargo fmt --all

format: format-python format-rust

# Run tests
test-python:
	cd trainer && uv run pytest

test-rust:
	cd agent && cargo test --all

test: test-python test-rust

# Lint Rust code
lint-rust:
	cd agent && cargo clippy --all-targets --all-features -- -D warnings

lint: lint-python lint-rust

# Clean build artifacts
clean:
	cd trainer && rm -rf .ruff_cache .mypy_cache .pytest_cache **/__pycache__
	cd agent && cargo clean
