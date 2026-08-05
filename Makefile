.PHONY: init doctor build-rust check lint format test clean

# Initialize development environment
init:
	cd trainer && uv sync --locked --dev --no-install-package reversi-zero-rs
	./scripts/with-torch-env uv sync --project trainer --locked --dev --no-build-isolation-package reversi-zero-rs --reinstall-package reversi-zero-rs
	cd trainer && uv run --no-sync pre-commit install
	cd trainer && uv run --no-sync pre-commit install --hook-type commit-msg
	$(MAKE) doctor

# Verify the toolchain and the Python/Rust bridge.
doctor:
	uv --version
	rustc --version
	cargo --version
	trainer/.venv/bin/python --version
	./scripts/with-torch-env trainer/.venv/bin/python trainer/scripts/doctor.py

# Build Rust components against the PyTorch installed in trainer/.venv.
build-rust:
	./scripts/with-torch-env cargo build --manifest-path agent/Cargo.toml --release

# Run all pre-commit checks
check:
	cd trainer && uv run --no-sync pre-commit run --all-files

# Lint Python code
lint-python:
	cd trainer && uv run --no-sync ruff check .

# Format code
format-python:
	cd trainer && uv run --no-sync ruff format .

format-rust:
	cd agent && cargo fmt --all

format: format-python format-rust

# Run tests
test-python:
	cd trainer && ../scripts/with-torch-env uv run --no-sync pytest

test-rust:
	./scripts/with-torch-env cargo test --manifest-path agent/Cargo.toml --all

test: test-python test-rust

# Lint Rust code
lint-rust:
	./scripts/with-torch-env cargo clippy --manifest-path agent/Cargo.toml --all-targets --all-features -- -D warnings

lint: lint-python lint-rust

# Clean build artifacts
clean:
	cd trainer && rm -rf .ruff_cache .mypy_cache .pytest_cache **/__pycache__
	cd agent && cargo clean
