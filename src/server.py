#!/usr/bin/env python3
from __future__ import annotations
import logging, multiprocessing as mp, signal
from waitress import serve

from gemma4_server.api.app import Runtime, create_app
from gemma4_server.core.config import Config
from gemma4_server.core.paths import (
    LOG_DIR, configure_logging
)
from gemma4_server.workers.manager import (
    GenerationManager
)

logger = logging.getLogger("gemma4_server")

class CoordinatorShutdown(BaseException):
    pass

def _signal(signum, _frame):
    raise CoordinatorShutdown(signum)

def main():
    mp.freeze_support()
    configure_logging(
        "api", LOG_DIR / "server.log"
    )
    try:
        config = Config.from_env()
    except Exception:
        logger.exception("Invalid configuration")
        return 2

    manager = GenerationManager(config)
    runtime = Runtime(config, manager)
    app = create_app(runtime)

    old = {
        s: signal.getsignal(s)
        for s in (signal.SIGTERM, signal.SIGINT)
    }
    for s in old:
        signal.signal(s, _signal)

    try:
        manager.start_async()
        serve(
            app,
            host=config.host,
            port=config.port,
            threads=max(
                4, config.max_queue_size + 3
            ),
            channel_timeout=max(
                120,
                int(config.request_timeout) + 60,
            ),
        )
    except CoordinatorShutdown:
        logger.info(
            "Coordinator shutdown requested"
        )
    finally:
        for s,h in old.items():
            signal.signal(s,h)
        manager.shutdown(
            False, config.shutdown_timeout
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
