"""Logging utilities for the project."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """Get or create a logger with standard formatting.

    Parameters
    ----------
    name : str
        Name of the logger (typically __name__ from calling module)
    level : int, default=logging.INFO
        Logging level (e.g., logging.DEBUG, logging.INFO)
    log_file : str, optional
        If provided, also log to this file

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if logger has no handlers (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            # Create parent directory if needed
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def set_log_level(logger: logging.Logger, level: int) -> None:
    """Set logging level for a logger and all its handlers.

    Parameters
    ----------
    logger : logging.Logger
        Logger to configure
    level : int
        New logging level
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
