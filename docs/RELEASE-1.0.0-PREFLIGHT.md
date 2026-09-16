# v1.0.0 Release Preflight

This is a preparation checklist. It does not create a tag, GitHub release, or
public publication during the final CPU phase.

## Required before publication

- [ ] G9 PRIME, HOT-1, and HOT-2 pass with direct evidence.
- [ ] G10 fresh Kaggle Restart Session → Run All passes at the frozen SHA.
- [ ] README and README.vi describe the same current gate state.
- [ ] G8 final evidence paths and SHA-256 manifests are present.
- [ ] Release notes include architecture, usage, Kaggle setup, known
  limitations, G8/G9/G10 evidence references, and asset checksums.
- [ ] Repository secret/private-material audit passes.
- [ ] Canonical CI passes on the exact release candidate SHA.

## Publication commands prepared for the later authorized phase

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
gh release create v1.0.0 --generate-notes --verify-tag
```

These commands are documentation only here. Until G10 closes, the required
state remains:

```text
TAG=false
RELEASE=false
PUBLIC_V1_0_0=false
```
