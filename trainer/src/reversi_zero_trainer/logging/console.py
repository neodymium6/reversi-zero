"""Console logger implementation using Rich for beautiful output."""

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table

from .base import BaseLogger, register_logger
from .config import BaseLoggerConfig, ConsoleConfig, LoggerKind


@register_logger(LoggerKind.CONSOLE)
class ConsoleLogger(BaseLogger):
    """Logger that prints to console/stdout using Rich for colorful output.

    Features:
    - Colorful metric and parameter logging
    - Optional parameter table display
    - Optional timestamps
    """

    def __init__(self, cfg: BaseLoggerConfig) -> None:
        """Initialize console logger.

        Args:
            cfg: Configuration (must be ConsoleConfig)

        Raises:
            TypeError: If cfg is not a ConsoleConfig instance
        """
        self.cfg: ConsoleConfig = self._as_console_cfg(cfg)
        self.console = Console()
        self.params: dict[str, Any] = {}
        self.params_logged = False
        self.last_step: int | None = None

    @staticmethod
    def _as_console_cfg(cfg: BaseLoggerConfig) -> ConsoleConfig:
        """Narrow the config type to ConsoleConfig.

        Args:
            cfg: Base logger configuration

        Returns:
            ConsoleConfig instance

        Raises:
            TypeError: If cfg is not a ConsoleConfig instance
        """
        if not isinstance(cfg, ConsoleConfig):
            raise TypeError(
                f"ConsoleLogger requires ConsoleConfig, but got {type(cfg).__name__}"
            )
        return cfg

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a metric to console with rich formatting.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        if not self.cfg.verbose:
            return

        # Show params table on first metric log
        if not self.params_logged and self.params and self.cfg.show_params_table:
            self._show_params_table()
            self.params_logged = True

        # Print separator when step changes
        if step is not None and self.last_step is not None and step != self.last_step:
            self.console.print("[dim]" + "─" * 80 + "[/dim]")

        if step is not None:
            self.last_step = step

        timestamp_str = ""
        if self.cfg.show_timestamp:
            timestamp_str = f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] "

        # Format with fixed width for alignment
        # Name: left-aligned, 30 chars
        # Value: right-aligned, consistent decimal places
        name_formatted = f"{name:<30}"

        if step is not None:
            self.console.print(
                f"{timestamp_str}[cyan]●[/cyan] [bold blue]{name_formatted}[/bold blue] "
                f"[bold green]{value:>12.6f}[/bold green] "
                f"[dim](step=[yellow]{step:>6}[/yellow])[/dim]"
            )
        else:
            self.console.print(
                f"{timestamp_str}[cyan]●[/cyan] [bold blue]{name_formatted}[/bold blue] "
                f"[bold green]{value:>12.6f}[/bold green]"
            )

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to console.

        Args:
            key: Parameter name
            value: Parameter value

        Raises:
            RuntimeError: If called after metrics have been logged
        """
        if not self.cfg.verbose:
            return

        # Validate: params must be logged before metrics
        if self.params_logged:
            raise RuntimeError(
                f"Cannot log parameter '{key}' after metrics have been logged. "
                "All parameters must be logged before logging any metrics."
            )

        # Store params for table display later
        self.params[key] = value

        # Also print individually if not using table
        if not self.cfg.show_params_table:
            self.console.print(
                f"[magenta]▸[/magenta] [bold]{key}[/bold]=[cyan]{value}[/cyan]"
            )

    def _show_params_table(self) -> None:
        """Display all logged parameters in a rich table."""
        if not self.params:
            return

        table = Table(title="[bold]Configuration Parameters[/bold]", show_header=True)
        table.add_column("Parameter", style="bold blue", no_wrap=True)
        table.add_column("Value", style="green")

        for key, value in self.params.items():
            table.add_row(key, str(value))

        self.console.print(table)
        self.console.print()  # Add blank line after table

    def finish(self) -> None:
        """Finish logging and show final summary."""
        if self.cfg.verbose and self.params and not self.params_logged:
            # If params were logged but no metrics, still show the table
            if self.cfg.show_params_table:
                self._show_params_table()
