from __future__ import annotations
import os
from pathlib import Path

REQUIRED_MODEL_FILES = (
    "config.json",
    "preprocessor.json",
    "assets/tokenizer/vocabulary.spm",
)
WEIGHTS_MONOLITHIC = "model.weights.h5"
WEIGHTS_INDEX = "model.weights.json"
WEIGHTS_GLOB = "model_*.weights.h5"
DEFAULT_PRESET = "gemma4_instruct_31b"

def weights_path(path: Path) -> Path:
    mono = path / WEIGHTS_MONOLITHIC
    if mono.is_file():
        return mono
    index = path / WEIGHTS_INDEX
    if index.is_file() and any(path.glob(WEIGHTS_GLOB)):
        return index
    raise FileNotFoundError(f"No strict Keras weights entry found under {path}")

def model_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not all((path / rel).is_file() for rel in REQUIRED_MODEL_FILES):
        return False
    try:
        weights_path(path)
    except FileNotFoundError:
        return False
    return True

def _version_key(path: Path):
    try:
        return (1, int(path.name), str(path))
    except ValueError:
        return (0, path.name, str(path))

def discover_from_base(base: Path) -> Path | None:
    base = base.expanduser()
    if model_complete(base):
        return base.resolve()
    if not base.is_dir():
        return None
    candidates = [p for p in base.iterdir() if model_complete(p)]
    return (
        sorted(candidates, key=_version_key, reverse=True)[0].resolve()
        if candidates else None
    )

def resolve_model_path(
    preferred: Path | None,
    input_root: Path = Path("/kaggle/input"),
    preset_name: str = DEFAULT_PRESET,
) -> Path:
    if preferred is not None:
        found = discover_from_base(preferred)
        if found is not None:
            return found

    candidates: list[Path] = []
    for root in (input_root / "models", input_root):
        if not root.is_dir():
            continue
        for dirpath, dirnames, _files in os.walk(root):
            current = Path(dirpath)
            if current.name == preset_name:
                found = discover_from_base(current)
                if found is not None:
                    candidates.append(found)
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", "__pycache__", ".cache"}
            ]
    if candidates:
        return sorted(set(candidates), key=_version_key, reverse=True)[0]

    preferred_text = str(preferred) if preferred else "<unset>"
    raise FileNotFoundError(
        "Could not find attached complete Keras Gemma 4 preset. "
        f"preset={preset_name!r}, preferred={preferred_text!r}, "
        f"input_root={str(input_root)!r}"
    )
