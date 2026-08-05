"""Verify that Python and Rust use the same PyTorch/CUDA environment."""

from __future__ import annotations

import sys

import reversi_zero_rs
import torch


def main() -> None:
    python_cuda_built = torch.backends.cuda.is_built()
    python_cuda_available = torch.cuda.is_available()
    rust_cuda_built = reversi_zero_rs.cuda_built()
    rust_cuda_available = reversi_zero_rs.cuda_available()

    print(f"torch {torch.__version__}")
    print(f"reversi_zero_rs {reversi_zero_rs.__file__}")
    print(
        f"CUDA support: Python built={python_cuda_built}, Rust built={rust_cuda_built}"
    )
    print(
        "CUDA runtime: "
        f"Python available={python_cuda_available}, Rust available={rust_cuda_available}"
    )
    if python_cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if python_cuda_built != rust_cuda_built:
        sys.exit(
            "Python/Rust CUDA build mismatch. Rebuild the extension with `make init`."
        )
    if python_cuda_available != rust_cuda_available:
        sys.exit(
            "Python/Rust CUDA runtime mismatch. Rebuild with `make init` and check the driver."
        )


if __name__ == "__main__":
    main()
