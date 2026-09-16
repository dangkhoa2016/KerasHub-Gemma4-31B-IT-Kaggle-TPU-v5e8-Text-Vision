# G8 lifecycle inventory

`g8-lifecycle-inventory.json` is the detailed component map. It inventories GenerationManager, model_worker_main, JobStore, Flask handlers, task/result queues, worker process, collector, monitor/watchdog, restart thread, and model engine with constructor/start/stop/restart/identity/lock/state/load/cleanup locations.

Observed worker states only: `starting`, `loading`, `ready`, `busy`, `failed`, `stopped`. Manager-derived health states only: `loading`, `ready`, `restarting`, `unavailable`.

The source boundary is clear: `GenerationManager._start_worker()` creates one `tpu-0` spawn-process and increments its generation; `model_worker_main()` owns TPU imports and `Gemma4TPUEngine.load()`; `_collect()` delegates result messages to `_handle()`; and `JobStore` owns persisted async state. The generation check in `_handle()` happens before event dispatch.
