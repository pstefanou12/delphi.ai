import logging
import json
import time
from pathlib import Path


class delphiLogger:
    """
    Structured logger for delphi and Trainer systems.
    Uses Python's logging infrastructure for console/file logging,
    and keeps JSONL records for experiment reproducibility.
    """
    def __init__(self, save_dir=None, run_name=None, level=logging.INFO):
        self.save_dir = save_dir
        self.run_name = run_name or time.strftime("%Y%m%d-%H%M%S")

        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(level)

        # Prevent duplicate handlers (VERY important in notebooks)
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            if self.save_dir is not None:
                self.save_dir = Path(self.save_dir)
                self.save_dir.mkdir(parents=True, exist_ok=True)

                # Standard log file (human readable)
                text_log_path = self.save_dir / f"{self.run_name}.log"

                # Structured log file (JSONL)
                self.jsonl_path = self.save_dir / f"{self.run_name}.jsonl"

                # File handler
                file_handler = logging.FileHandler(text_log_path)
                file_formatter = logging.Formatter(
                    "[%(asctime)s] %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)

        self._history = []

    def log(self, level=20, **data):
        """
        Log a dict of experiment data (to console, text file, and JSONL).
        level 20 is the int code for logging.INFO. Full list of logging levels here: 
        https://docs.python.org/3/library/logging.html#logging-levels
        """
        import ipdb; ipdb.set_trace()
        # Write structured data to JSONL
        record = {"timestamp": time.time(), **data}
        self._history.append(record)

        if self.save_dir is not None:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        # Human-readable logging via Python logging
        pretty = ", ".join(f"{k}={v}" for k, v in data.items())
        import ipdb; ipdb.set_trace()
        self.logger.log(level, pretty)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def log_training_step(self, epoch, loss):
        data = {
                'epoch': epoch, 
                'loss': float(loss)
            }
        self.log(data=data)

    def log_metrics(self, **metrics):
        self.log(data=metrics)

    def log_model(self, model):
        params = sum(p.numel() for p in model.parameters())
        self.log(data=params)

    def get_history(self): 
        return self._history