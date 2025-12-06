import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyReversiNet(nn.Module):
    """
    8x8 Reversi board simple network.
    - Input: (B, C, 8, 8)  (C is assumed to be 3 channels for now)
    - Output:
        policy: (B, 64)  logits for each square
        value:  (B, 1)   scalar evaluation of the board
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 64),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))

        policy_logits = self.policy_head(h)  # (B, 64)
        value = self.value_head(h)  # (B, 1)
        return policy_logits, value


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(x + h)


class ResNetReversiNet(nn.Module):
    """
    Shallow ResNet for 8x8 Reversi.
    - Input: (B, C, 8, 8)
    - Output: policy (B, 64), value (B, 1)

    Args:
        in_channels: input feature channels (default 3)
        channels: internal channel width (default 64)
        num_blocks: number of residual blocks (default 6)
    """

    def __init__(self, in_channels: int = 3, channels: int = 64, num_blocks: int = 6):
        super().__init__()
        self.stem = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 64),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        h = F.relu(self.stem(x))
        h = self.blocks(h)
        policy_logits = self.policy_head(h)  # (B, 64)
        value = self.value_head(h)  # (B, 1)
        return policy_logits, value
