from __future__ import annotations
import os, secrets
from pathlib import Path

def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)

def load_or_create_secret(env_name: str, path: Path, required: bool) -> str:
    value = os.environ.get(env_name, "").strip()
    if value:
        _write(path, value)
        return value
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value:
            os.chmod(path, 0o600)
            return value
    except FileNotFoundError:
        pass
    if not required:
        return ""
    value = secrets.token_urlsafe(32)
    _write(path, value)
    return value
