from __future__ import annotations
import logging
from pathlib import Path

DATA_DIR = Path("data")
LOG_DIR = Path("logs")
STATE_DIR = Path("state")
ARTIFACT_DIR = Path("artifacts")

def configure_logging(name: str, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        return
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(process)d %(name)s %(message)s"
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)
    handler = logging.FileHandler(file_path, encoding="utf-8")
    handler.setFormatter(formatter)
    root.addHandler(handler)
