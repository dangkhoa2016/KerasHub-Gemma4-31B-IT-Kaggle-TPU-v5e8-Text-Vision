# G4 Entry Resolution

```text
G3_PASS_FROZEN=true
G4_SCOPE_RESOLUTION=PASS
```

## Repository-supported scope

The canonical roadmap defines G4 as:

> `G4  native-vs-split generation characterization`

Source: `docs/ROADMAP.md`, roadmap gate list, line 23 (also repeated in
`docs/ROADMAP.vi.md`, line 23).

The architecture document narrows the supported comparison: the initial G3/G5
path is marked `keras_hub_native_unvalidated`; G4 may replace it with a
Gemma4-native split prefill/decode implementation after real TPU
characterization. Source: `docs/ARCHITECTURE.md`, lines 29-31, with the same
scope in `docs/ARCHITECTURE.vi.md`, lines 26-27.

## Resolution

G4 is therefore the characterization of the native KerasHub generation path
against the repository's possible Gemma4-native split prefill/decode path.
No new gate, model behavior, or speculative requirement is introduced. The
kickoff records the scope and starts planning only; no expensive TPU workload
is launched by this closeout.
