import logging
import sys
from datetime import datetime
from typing import ClassVar


class Colors:
    """ANSI escape codes for terminal colors."""

    GREY = "\x1b[38;5;240m"
    BLUE = "\x1b[38;5;39m"
    YELLOW = "\x1b[38;5;220m"
    RED = "\x1b[38;5;196m"
    GREEN = "\x1b[38;5;82m"
    BOLD = "\x1b[1m"
    RESET = "\x1b[0m"


class Formatter(logging.Formatter):
    """A formatter for attractor package logs."""

    level_colors: ClassVar[dict[int, str]] = {
        logging.DEBUG: Colors.GREY,
        logging.INFO: Colors.BLUE,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.level_colors.get(record.levelno, Colors.RESET)
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        logger_name = record.name.split(".")[-1]
        prefix = (
            f"{Colors.GREY}{timestamp}{Colors.RESET} "
            f"{color}{record.levelname:8}{Colors.RESET} "
            f"{Colors.GREEN}{logger_name:12}{Colors.RESET} "
        )

        message = record.getMessage()
        if record.exc_info:
            if not message.endswith("\n"):
                message += "\n"
            message += self.formatException(record.exc_info)

        return f"{prefix} │ {message}"


root_logger = logging.getLogger("attractors")
root_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(Formatter())
root_logger.addHandler(console_handler)
root_logger.propagate = False


def setup_logger(name: str | None = None) -> logging.Logger:
    """
    Set up a logger with beautiful formatting.

    This will create a child logger that inherits settings from the root logger.

    Args:
        name: The name for the logger (defaults to the module name)

    Returns:
        A configured logger instance

    Example:
        >>> logger = setup_logger(name=__name__)
        >>> logger.info("Processing system parameters...")
        >>> logger.warning("Numerical instability detected")
        >>> logger.error("Failed to integrate system", exc_info=True)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(Formatter())
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def set_log_level(level: str | int) -> None:
    """
    Set the logging level for all attractors loggers.

    Args:
        level: Either a string ('DEBUG', 'INFO', etc) or logging constant

    Example:
        >>> set_log_level("DEBUG")  # Using string
        >>> set_log_level(logging.INFO)  # Using logging constant
    """
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level

    root_logger.setLevel(numeric_level)


default_logger = setup_logger(name="attractors")
