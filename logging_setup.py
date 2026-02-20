import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(app_name: str = "app", log_file: str = "backend.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (rotates)
    fh = RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=3)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info("%s logging initialized", app_name)
