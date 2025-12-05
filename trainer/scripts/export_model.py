from pathlib import Path

import torch

from reversi_zero_trainer.models.dummy import DummyReversiNet


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    models_dir = root / "models" / "ts"
    models_dir.mkdir(parents=True, exist_ok=True)

    model = DummyReversiNet(in_channels=3)
    model.eval()

    example_input = torch.zeros(1, 3, 8, 8, dtype=torch.float32)

    scripted = torch.jit.trace(model, example_input)

    out_path = models_dir / "latest.pt"
    if not isinstance(scripted, torch.jit.ScriptModule):
        raise TypeError("Expected scripted model to be a ScriptModule")
    scripted.save(str(out_path))
    print(f"Saved TorchScript model to {out_path}")


if __name__ == "__main__":
    main()
