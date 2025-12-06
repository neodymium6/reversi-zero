from pathlib import Path
from typing import Literal

import torch

from reversi_zero_trainer.models.dummy import DummyReversiNet, ResNetReversiNet


def main() -> None:
    model_type: Literal["dummy", "resnet"] = "resnet"  # change to "dummy" if needed
    channels = 64
    num_blocks = 6

    root = Path(__file__).resolve().parents[2]
    models_dir = root / "models" / "ts"
    models_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "resnet":
        model = ResNetReversiNet(
            in_channels=3, channels=channels, num_blocks=num_blocks
        )
        suffix = f"resnet_c{channels}_b{num_blocks}"
    else:
        model = DummyReversiNet(in_channels=3)
        suffix = "dummy"

    model.eval()

    example_input = torch.zeros(1, 3, 8, 8, dtype=torch.float32)

    scripted = torch.jit.trace(model, example_input)

    out_path = models_dir / f"latest_{suffix}.pt"
    if not isinstance(scripted, torch.jit.ScriptModule):
        raise TypeError("Expected scripted model to be a ScriptModule")
    scripted.save(str(out_path))
    print(f"Saved TorchScript model to {out_path}")


if __name__ == "__main__":
    main()
