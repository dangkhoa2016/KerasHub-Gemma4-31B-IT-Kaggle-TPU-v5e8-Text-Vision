from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Iterable, Mapping


GIB = 1024**3
MINIMUM_AUTHORITY_MEMORY_GIB = 300
EXPECTED_CANDIDATE_A_SHAPE = (262144, 5376)
EXPECTED_CANDIDATE_A_DTYPE = "bfloat16"
EXPECTED_CANDIDATE_A_SHARD_COUNT = 8
EXPECTED_CANDIDATE_A_SHARD_SHAPE = (32768, 5376)
EXPECTED_CANDIDATE_A_SPEC = "P('model','batch')"


class AuthorityGateError(RuntimeError):
    """A pre-authority contract or environment gate failed."""


class CandidateAVerificationError(RuntimeError):
    """The loaded model does not satisfy the frozen Candidate-A contract."""


def _normal_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def hardware_gate(
    model_path: str | Path,
    accel_paths: Iterable[str | Path],
    memory_max_bytes: int | None,
    minimum_memory_gib: int = MINIMUM_AUTHORITY_MEMORY_GIB,
) -> dict[str, object]:
    """Validate the shell-visible hardware prerequisites without JAX imports."""
    model = Path(model_path)
    accel = [str(path) for path in accel_paths]
    minimum_bytes = int(minimum_memory_gib) * GIB
    result = {
        "MODEL_PATH": str(model),
        "MODEL_PATH_EXISTS": model.exists(),
        "ACCELERATOR_PATHS": accel,
        "ACCELERATOR_PRESENT": bool(accel),
        "MEMORY_MAX_BYTES": memory_max_bytes,
        "MINIMUM_AUTHORITY_MEMORY_GIB": int(minimum_memory_gib),
        "MEMORY_FLOOR_PASS": (
            memory_max_bytes is not None and int(memory_max_bytes) >= minimum_bytes
        ),
    }
    if not model.exists():
        raise AuthorityGateError(f"MODEL_PATH does not exist: {model}")
    if not accel:
        raise AuthorityGateError("TPU accelerator device node /dev/accel* is absent")
    if not result["MEMORY_FLOOR_PASS"]:
        raise AuthorityGateError(
            f"cgroup memory.max is below {minimum_memory_gib} GiB: {memory_max_bytes!r}"
        )
    result["HARDWARE_GATE"] = "PASS"
    return result


def _read_requirements(path: str | Path) -> list[tuple[str, str]]:
    requirements = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if not match:
            raise AuthorityGateError(f"requirements line is not exact-pinned: {line!r}")
        requirements.append((_normal_name(match.group(1)), match.group(2)))
    return requirements


def exact_runtime_versions(
    requirements_path: str | Path,
    installed: Mapping[str, str],
) -> dict[str, str]:
    """Require every manifest distribution to have its exact installed version."""
    installed_normalized = {
        _normal_name(name): str(version) for name, version in installed.items()
    }
    result = {}
    for name, expected in _read_requirements(requirements_path):
        observed = installed_normalized.get(name)
        if observed != expected:
            raise AuthorityGateError(
                f"runtime baseline mismatch for {name}: expected {expected}, found {observed}"
            )
        result[name] = observed
    return result


def _parse_dependency(raw: str) -> tuple[str, str | None, bool]:
    try:
        from packaging.markers import default_environment
        from packaging.requirements import Requirement

        requirement = Requirement(raw)
        if requirement.marker is not None and not requirement.marker.evaluate(default_environment()):
            return _normal_name(requirement.name), None, False
        return _normal_name(requirement.name), str(requirement.specifier) or None, True
    except ImportError:
        requirement, _, marker = raw.partition(";")
        if marker and "extra" in marker:
            return "", None, False
        requirement = re.sub(r"\[[^]]*\]", "", requirement.strip())
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*((?:==|!=|~=|>=|<=|>|<)?)\s*([^\s,]+)?", requirement)
        if not match:
            raise AuthorityGateError(f"dependency metadata is not parseable: {raw!r}")
        specifier = (match.group(2) or "") + (match.group(3) or "")
        return _normal_name(match.group(1)), specifier or None, True


def _version_key(version: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"[.+-]", version.lower()))


def _satisfies(version: str, specifier: str | None) -> bool:
    if not specifier:
        return True
    for item in specifier.split(","):
        match = re.fullmatch(r"(==|!=|~=|>=|<=|>|<)\s*(.+)", item.strip())
        if not match:
            return False
        operator, expected = match.groups()
        if operator in {"==", "!="} and expected.endswith(".*"):
            prefix = expected[:-1]
            matches = version == prefix[:-1] or version.startswith(prefix)
            return matches if operator == "==" else not matches
        actual_key = _version_key(version)
        expected_key = _version_key(expected)
        if operator == "==" and actual_key != expected_key:
            return False
        if operator == "!=" and actual_key == expected_key:
            return False
        if operator == ">=" and actual_key < expected_key:
            return False
        if operator == "<=" and actual_key > expected_key:
            return False
        if operator == ">" and actual_key <= expected_key:
            return False
        if operator == "<" and actual_key >= expected_key:
            return False
        if operator == "~=" and actual_key < expected_key:
            return False
    return True


def project_dependency_gate(
    distributions: Mapping[str, Mapping[str, object]],
    roots: Iterable[str],
) -> dict[str, object]:
    """Validate the metadata closure rooted at the frozen runtime packages."""
    normalized = {_normal_name(name): metadata for name, metadata in distributions.items()}
    pending = deque(_normal_name(root) for root in roots)
    checked = []
    while pending:
        name = pending.popleft()
        if name in checked:
            continue
        metadata = normalized.get(name)
        if metadata is None:
            raise AuthorityGateError(f"project dependency is missing: {name}")
        checked.append(name)
        for raw in metadata.get("requires", ()) or ():
            dependency, specifier, active = _parse_dependency(str(raw))
            if not active:
                continue
            dependency_metadata = normalized.get(dependency)
            if dependency_metadata is None:
                raise AuthorityGateError(
                    f"project dependency is missing: {dependency} required by {name}"
                )
            if not _satisfies(str(dependency_metadata.get("version", "")), specifier):
                raise AuthorityGateError(
                    f"project dependency version mismatch for {dependency}: {raw}"
                )
            pending.append(dependency)
    return {
        "pass": True,
        "checked_distributions": checked,
        "PROJECT_SCOPED_DEPENDENCY_GATE": "PASS",
    }


def _weight_path(weight) -> str:
    return str(getattr(weight, "path", getattr(weight, "name", "")))


def _canonical_spec(value) -> str:
    raw = repr(value)
    try:
        parts = tuple(value)
    except TypeError:
        parts = tuple(re.findall(r"['\"]([^'\"]+)['\"]", raw))
    if parts == ("model", "batch"):
        return EXPECTED_CANDIDATE_A_SPEC
    return raw


def verify_candidate_a(model) -> dict[str, object]:
    """Verify the frozen token-embedding Candidate-A placement after load."""
    embedding = next(
        (weight for weight in getattr(model, "weights", ()) if _weight_path(weight).endswith("token_embedding/embeddings")),
        None,
    )
    if embedding is None:
        raise CandidateAVerificationError("token embedding weight is absent")
    value = getattr(embedding, "value", embedding)
    shape = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", "")).lower()
    shards = list(getattr(value, "addressable_shards", ()))
    shard_shapes = [
        [int(dimension) for dimension in getattr(getattr(shard, "data", shard), "shape", ())]
        for shard in shards
    ]
    spec = _canonical_spec(getattr(getattr(value, "sharding", None), "spec", None))
    if shape != EXPECTED_CANDIDATE_A_SHAPE:
        raise CandidateAVerificationError(f"token embedding shape mismatch: {shape!r}")
    if EXPECTED_CANDIDATE_A_DTYPE not in dtype:
        raise CandidateAVerificationError(f"token embedding dtype mismatch: {dtype!r}")
    if len(shards) != EXPECTED_CANDIDATE_A_SHARD_COUNT:
        raise CandidateAVerificationError(f"token embedding shard count mismatch: {len(shards)}")
    if any(tuple(shape) != EXPECTED_CANDIDATE_A_SHARD_SHAPE for shape in shard_shapes):
        raise CandidateAVerificationError(f"token embedding shard shapes mismatch: {shard_shapes!r}")
    if spec != EXPECTED_CANDIDATE_A_SPEC:
        raise CandidateAVerificationError(f"token embedding sharding spec mismatch: {spec!r}")
    return {
        "TOKEN_EMBEDDING_SHAPE": list(shape),
        "TOKEN_EMBEDDING_DTYPE": EXPECTED_CANDIDATE_A_DTYPE,
        "TOKEN_EMBEDDING_ADDRESSABLE_SHARDS": len(shards),
        "TOKEN_EMBEDDING_SHARD_SHAPES": shard_shapes,
        "TOKEN_EMBEDDING_SHARDING_SPEC": spec,
        "CANDIDATE_A_SHARDING_VERIFIED": True,
    }
