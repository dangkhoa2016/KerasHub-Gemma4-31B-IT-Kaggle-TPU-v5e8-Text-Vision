# Troubleshooting

## TPU device count is not 8

Verify the Kaggle accelerator is TPU v5e-8 / v5litepod-8 and that no stale TPU
process owns the device handles. Do not run multiple independent JAX TPU
clients against the same allocation.

## Worker is not ready

Inspect `logs/server.stdout.log`, `/health/ready`, and the worker load timeout.
Model load can legitimately take many minutes.

## First request is slow

The first compatible request may compile a large JAX executable. Keep the
worker alive and retry the same request shape before diagnosing hot latency.

## Memory climbs during load

Avoid concurrent model loads. Check cgroup memory and checkpoint page cache.
A second 31B load in the same host can exceed the available memory.

## 401 Unauthorized

Provide the configured API key through `Authorization: Bearer ...` or
`X-API-Key`. Restart additionally requires `X-Restart-Secret`.

## Image request rejected

Check request size, image byte limit, pixel limit and content type. Multipart
requests must contain the `image` field.
