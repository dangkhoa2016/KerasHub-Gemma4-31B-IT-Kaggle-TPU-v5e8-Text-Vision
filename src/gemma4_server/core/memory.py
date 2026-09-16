from __future__ import annotations
import json, os, threading, time
from pathlib import Path
from typing import Callable

GIB = 1024 ** 3

def cgroup_memory_current_gib(
    path: Path = Path("/sys/fs/cgroup/memory.current"),
) -> float | None:
    try:
        return int(path.read_text(encoding="utf-8").strip()) / GIB
    except (OSError, ValueError):
        return None

def cgroup_snapshot() -> dict:
    def read(path):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            return None
    cur = read("/sys/fs/cgroup/memory.current")
    peak = read("/sys/fs/cgroup/memory.peak")
    return {
        "current_bytes": cur,
        "current_gib": round(cur / GIB, 6) if cur is not None else None,
        "peak_bytes": peak,
        "peak_gib": round(peak / GIB, 6) if peak is not None else None,
    }

class MemoryGuardMonitor:
    def __init__(
        self,
        *,
        guard_gib: float = 300.0,
        interval_seconds: float = 1.0,
        breach_path: Path = Path("state/memory-guard-breach.json"),
        sampler: Callable[[], float | None] = cgroup_memory_current_gib,
        terminator: Callable[[int], object] = os._exit,
    ):
        self.guard_gib = float(guard_gib)
        self.interval_seconds = float(interval_seconds)
        self.breach_path = Path(breach_path)
        self.sampler = sampler
        self.terminator = terminator
        self._stop = threading.Event()
        self._thread = None
        self._triggered = False
        self._phase = "startup"
        self._lock = threading.Lock()

    def set_phase(self, phase: str):
        self._phase = str(phase)

    def sample_once(self, label="sample"):
        value = self.sampler()
        if value is None or value < self.guard_gib:
            return value
        with self._lock:
            if self._triggered:
                return value
            self._triggered = True
            payload = {
                "label": label,
                "phase": self._phase,
                "current_gib": round(float(value), 3),
                "guard_gib": self.guard_gib,
                "exit_code": 20,
                "timestamp": time.time(),
            }
            self.breach_path.parent.mkdir(parents=True, exist_ok=True)
            self.breach_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.terminator(20)
        return value

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sample_once("periodic")
            except Exception:
                continue

    def start(self):
        self.breach_path.unlink(missing_ok=True)
        self.sample_once("monitor_start")
        self._thread = threading.Thread(
            target=self._run, name="memory-guard", daemon=True
        )
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self._thread = None
