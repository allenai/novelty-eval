import logging
import os
from datetime import datetime
from colorama import Fore, Style


class ColoredFormatter(logging.Formatter):
    """Custom formatter to color log levels in the console."""

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT
    }

    def format(self, record):
        # We create a copy of the record so we don't mutate the original
        # which acts as the source for the FileHandler (which shouldn't be colored)
        record = logging.makeLogRecord(record.__dict__)

        log_color = self.COLORS.get(record.levelname, "")
        reset = Style.RESET_ALL
        record.levelname = f"{log_color}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(output_dir: str, console_level: str = "INFO") -> logging.Logger:
    """Set up a logger that writes to output_dir/logs/timestamp with colored console output."""

    # 1. Setup Directories
    logs_base_dir = os.path.join(output_dir, 'logs')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(logs_base_dir, timestamp)

    os.makedirs(run_dir, exist_ok=True)

    # 2. Configure Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Removed [%(task)s] from format
    log_format = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s"

    # 3. Create Handlers

    # A. File Handler (Clean text, saved to run.log)
    log_file_path = os.path.join(run_dir, 'run.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.setLevel(logging.DEBUG)

    # B. Console Handler (Colored text, printed to stdout)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColoredFormatter(log_format))
    stream_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))

    # 4. Attach Handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # 5. Attach runtime context (Optional but useful)
    # This allows you to do `logger.run_dir` elsewhere in your code
    # to save artifacts (plots, jsons) in the same folder as the logs.
    logger.run_dir = run_dir

    logger.info(f"Logger initialized. Saving logs to: {run_dir}")

    return logger


def attach_run_log(logger: logging.Logger, artifacts_dir: str) -> None:
    """Add a second file handler that writes to artifacts_dir/run.log."""
    log_format = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s"
    log_file_path = os.path.join(artifacts_dir, "run.log")
    handler = logging.FileHandler(log_file_path)
    handler.setFormatter(logging.Formatter(log_format))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info(f"Run log attached: {log_file_path}")