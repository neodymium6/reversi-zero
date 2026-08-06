"""
AlphaZero training loop implementation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from reversi_zero_trainer.data import (
    ReplayBufferDataset,
    SelfPlayDataset,
    SymmetryAugmentedDataset,
)

TrainingDataSource = Path | str | Sequence[Path | str]

WSD_WARMUP_FRACTION = 0.02
WSD_DECAY_FRACTION = 0.15
WSD_MIN_LR_RATIO = 0.01


class WSDScheduler:
    """Warmup-stable-decay schedule over the progress of a complete run."""

    def __init__(self, optimizer: Optimizer, total_iterations: int) -> None:
        if total_iterations <= 0:
            raise ValueError("total_iterations must be > 0")
        self.optimizer = optimizer
        self.total_iterations = total_iterations
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.iteration = 0
        self.steps_in_iteration = 1
        self.step_in_iteration = 0
        self._apply_progress(0.0)

    @staticmethod
    def factor(progress: float) -> float:
        """Return the LR multiplier at normalized run progress."""
        progress = min(max(progress, 0.0), 1.0)
        if progress < WSD_WARMUP_FRACTION:
            warmup_progress = progress / WSD_WARMUP_FRACTION
            return WSD_MIN_LR_RATIO + (1.0 - WSD_MIN_LR_RATIO) * warmup_progress

        decay_start = 1.0 - WSD_DECAY_FRACTION
        if progress <= decay_start:
            return 1.0

        decay_progress = (progress - decay_start) / WSD_DECAY_FRACTION
        return 1.0 - (1.0 - WSD_MIN_LR_RATIO) * decay_progress

    def _apply_progress(self, progress: float) -> None:
        factor = self.factor(progress)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def begin_iteration(self, iteration: int, steps_in_iteration: int) -> None:
        """Start one training iteration with a known number of optimizer steps."""
        if not 0 <= iteration < self.total_iterations:
            raise ValueError("iteration must be within the configured run")
        if steps_in_iteration <= 0:
            raise ValueError("steps_in_iteration must be > 0")
        self.iteration = iteration
        self.steps_in_iteration = steps_in_iteration
        self.step_in_iteration = 0
        self._apply_progress(iteration / self.total_iterations)

    def step(self) -> None:
        """Advance the schedule after one optimizer step."""
        self.step_in_iteration += 1
        progress = (
            self.iteration + self.step_in_iteration / self.steps_in_iteration
        ) / self.total_iterations
        self._apply_progress(progress)

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_iterations": self.total_iterations,
            "base_lrs": self.base_lrs,
            "iteration": self.iteration,
            "steps_in_iteration": self.steps_in_iteration,
            "step_in_iteration": self.step_in_iteration,
            "last_lrs": [float(group["lr"]) for group in self.optimizer.param_groups],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        stored_total = int(state_dict["total_iterations"])
        if stored_total != self.total_iterations:
            raise ValueError(
                "WSD total_iterations mismatch: "
                f"checkpoint={stored_total}, configured={self.total_iterations}"
            )
        self.base_lrs = [float(lr) for lr in state_dict["base_lrs"]]
        self.iteration = int(state_dict["iteration"])
        self.steps_in_iteration = int(state_dict["steps_in_iteration"])
        self.step_in_iteration = int(state_dict["step_in_iteration"])
        last_lrs = [float(lr) for lr in state_dict["last_lrs"]]
        if len(last_lrs) != len(self.optimizer.param_groups):
            raise ValueError("WSD optimizer parameter group count mismatch")
        for group, learning_rate in zip(self.optimizer.param_groups, last_lrs):
            group["lr"] = learning_rate


@dataclass
class TrainingConfig:
    """Configuration for AlphaZero training."""

    # Training
    batch_size: int = 256
    num_workers: int = 4
    num_epochs: int = 1
    learning_rate: float = 0.001
    lr_schedule: Literal["constant", "wsd"] = "constant"
    lr_schedule_iterations: int | None = None
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    symmetry_augmentation: Literal[1, 2, 4, 8] = 8
    dtype: Literal["float32", "bfloat16"] = "float32"

    # Device
    device: Literal["cuda", "cpu"] = "cuda"

    # Checkpointing
    checkpoint_dir: Path | str = "checkpoints"
    save_every_n_epochs: int = 1

    def __post_init__(self) -> None:
        if self.symmetry_augmentation not in (1, 2, 4, 8):
            raise ValueError("symmetry_augmentation must be one of 1, 2, 4, or 8")
        if self.dtype == "bfloat16" and self.device != "cuda":
            raise ValueError("bfloat16 training requires CUDA")
        if self.lr_schedule == "wsd" and (
            self.lr_schedule_iterations is None or self.lr_schedule_iterations <= 0
        ):
            raise ValueError("WSD requires lr_schedule_iterations > 0")
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class AlphaZeroTrainer:
    """
    Trainer for AlphaZero-style networks.

    Combines policy loss (cross-entropy with MCTS distribution) and
    value loss (MSE with game outcome).

    This trainer is designed to be reusable across multiple self-play iterations.
    You can call train() multiple times with different data paths.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optimizer | None = None,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = (
            WSDScheduler(self.optimizer, config.lr_schedule_iterations)
            if config.lr_schedule == "wsd" and config.lr_schedule_iterations is not None
            else None
        )

        # Training state (persists across multiple train() calls)
        self.batch_step = 0  # Total number of batches processed
        self.total_epochs_trained = 0

    def _create_dataloaders(
        self, data_path: TrainingDataSource
    ) -> tuple[DataLoader, DataLoader]:
        """Create augmented training and unmodified evaluation dataloaders."""
        evaluation_dataset: SelfPlayDataset | ReplayBufferDataset
        if isinstance(data_path, (Path, str)):
            evaluation_dataset = SelfPlayDataset(data_path)
        else:
            evaluation_dataset = ReplayBufferDataset(list(data_path))
        training_dataset = (
            evaluation_dataset
            if self.config.symmetry_augmentation == 1
            else SymmetryAugmentedDataset(
                evaluation_dataset,
                symmetry_count=self.config.symmetry_augmentation,
            )
        )
        common_options = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.device == "cuda",
        }
        training_dataloader = DataLoader(
            training_dataset,
            shuffle=True,
            **common_options,
        )
        evaluation_dataloader = DataLoader(
            evaluation_dataset,
            shuffle=False,
            **common_options,
        )
        return training_dataloader, evaluation_dataloader

    def compute_loss(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined AlphaZero loss.

        Args:
            policy_logits: (B, 64) raw policy logits
            value_pred: (B, 1) predicted value
            target_policy: (B, 64) target MCTS visit distribution
            target_value: (B,) target game outcome

        Returns:
            total_loss: combined loss
            policy_loss: cross-entropy loss
            value_loss: MSE loss
        """
        # Policy loss: cross-entropy between MCTS distribution and network output
        # Note: target_policy is already a probability distribution from MCTS
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policy * log_probs).sum(dim=1).mean()

        # Value loss: MSE between predicted value and actual outcome
        value_pred = value_pred.squeeze(1)  # (B, 1) -> (B,)
        value_loss = F.mse_loss(value_pred, target_value)

        # Combined loss
        total_loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.value_loss_weight * value_loss
        )

        return total_loss, policy_loss, value_loss

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data

        Returns:
            Dictionary of average metrics for the epoch.
        """
        self.model.train()

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        for batch_idx, (states, target_policies, target_values) in enumerate(
            dataloader
        ):
            # Move to device
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            # BF16 autocast accelerates CUDA tensor-core training while keeping
            # model parameters, optimizer state, and loss accumulation in FP32.
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.config.dtype == "bfloat16",
            ):
                policy_logits, value_pred = self.model(states)
                total_loss, policy_loss, value_loss = self.compute_loss(
                    policy_logits, value_pred, target_policies, target_values
                )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Accumulate metrics
            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            num_batches += 1
            self.batch_step += 1

        # Return average metrics
        return {
            "loss/total": total_loss_sum / num_batches,
            "loss/policy": policy_loss_sum / num_batches,
            "loss/value": value_loss_sum / num_batches,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate model on full dataset without training.

        Args:
            dataloader: DataLoader for evaluation data

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        # Track value predictions for analysis
        value_preds = []
        value_targets = []

        for states, target_policies, target_values in dataloader:
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            # Forward pass
            policy_logits, value_pred = self.model(states)

            # Compute loss
            total_loss, policy_loss, value_loss = self.compute_loss(
                policy_logits, value_pred, target_policies, target_values
            )

            # Accumulate metrics
            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            num_batches += 1

            # Track predictions
            value_preds.append(value_pred.squeeze(1).cpu())
            value_targets.append(target_values.cpu())

        # Concatenate all predictions
        all_preds = torch.cat(value_preds)
        all_targets = torch.cat(value_targets)

        # Compute additional metrics
        value_mae = (all_preds - all_targets).abs().mean().item()
        value_correlation = torch.corrcoef(torch.stack([all_preds, all_targets]))[
            0, 1
        ].item()

        return {
            "eval/loss_total": total_loss_sum / num_batches,
            "eval/loss_policy": policy_loss_sum / num_batches,
            "eval/loss_value": value_loss_sum / num_batches,
            "eval/value_mae": value_mae,
            "eval/value_correlation": value_correlation,
        }

    def save_checkpoint(self, epoch: int, filename: str | None = None) -> Path:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            filename: Optional custom filename (default: checkpoint_epoch_{epoch}.pt)

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        if not isinstance(self.config.checkpoint_dir, Path):
            self.config.checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_path = self.config.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "total_epochs_trained": self.total_epochs_trained,
            "batch_step": self.batch_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            ),
            "config": self.config,
        }

        next_path = checkpoint_path.with_name(f".{checkpoint_path.name}.next")
        try:
            torch.save(checkpoint, next_path)  # nosec B614
            next_path.replace(checkpoint_path)
        finally:
            next_path.unlink(missing_ok=True)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        """
        Load model checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )  # nosec B614

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_state = checkpoint.get("lr_scheduler_state_dict")
        if scheduler_state is not None:
            if self.lr_scheduler is None:
                raise ValueError("Checkpoint uses WSD but trainer does not")
            self.lr_scheduler.load_state_dict(scheduler_state)
        self.total_epochs_trained = checkpoint["total_epochs_trained"]
        self.batch_step = checkpoint["batch_step"]

        # Restore config if available
        if "config" in checkpoint:
            self.config = checkpoint["config"]
            if "symmetry_augmentation" not in vars(self.config):
                self.config.symmetry_augmentation = 1
            if "dtype" not in vars(self.config):
                self.config.dtype = "float32"
            if "lr_schedule" not in vars(self.config):
                self.config.lr_schedule = "constant"
            if "lr_schedule_iterations" not in vars(self.config):
                self.config.lr_schedule_iterations = None
            # Ensure checkpoint_dir is a Path object
            self.config.checkpoint_dir = Path(self.config.checkpoint_dir)

    def train(
        self,
        data_path: TrainingDataSource,
        num_epochs: int | None = None,
        schedule_iteration: int | None = None,
    ) -> Generator[dict[str, float], None, None]:
        """
        Run training loop, yielding metrics for each epoch.

        This method can be called multiple times with different data paths
        to continue training on new self-play data.

        Args:
            data_path: One or more directories containing self-play NPY files
            num_epochs: Number of epochs to train. If None, uses config.num_epochs
            schedule_iteration: Zero-based run iteration required by WSD

        Yields:
            Dictionary containing training and evaluation metrics for each epoch

        Example:
            # Train on initial data
            for metrics in trainer.train("data/selfplay_iter1"):
                logger.log_metric("loss", metrics["loss/total"])

            # Later, train on new data
            for metrics in trainer.train("data/selfplay_iter2"):
                logger.log_metric("loss", metrics["loss/total"])
        """
        training_dataloader, evaluation_dataloader = self._create_dataloaders(data_path)

        if num_epochs is None:
            num_epochs = self.config.num_epochs

        if self.lr_scheduler is not None:
            if schedule_iteration is None:
                raise ValueError("WSD training requires schedule_iteration")
            self.lr_scheduler.begin_iteration(
                schedule_iteration,
                len(training_dataloader) * num_epochs,
            )

        for epoch in range(num_epochs):
            # Train one epoch
            train_metrics = self.train_epoch(training_dataloader)

            # Evaluate
            eval_metrics = self.evaluate(evaluation_dataloader)

            # Combine metrics
            metrics = {
                **train_metrics,
                **eval_metrics,
                "epoch": self.total_epochs_trained,
                "batch_step": self.batch_step,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }

            self.total_epochs_trained += 1

            yield metrics
