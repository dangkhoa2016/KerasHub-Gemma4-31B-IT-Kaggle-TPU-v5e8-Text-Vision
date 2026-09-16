# G6 Memory Envelope

The observed final authority cgroup envelope was:

```text
memory.max=354334801920 bytes
```

This is the observed Kaggle authority cgroup envelope, not a universal TPU v5e-8 specification. The historical 300 GiB admission floor is an authority-run gate, not a hardware specification.

All final G3, G4, and G5 authority records show the same `memory.max`. No `memory.peak` value was recorded, so `memory_peak_status=NOT_RECORDED`.
