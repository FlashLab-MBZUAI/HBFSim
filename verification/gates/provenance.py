#!/usr/bin/env python3
"""Fail unless every published scenario parameter has auditable provenance."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


# These values describe the workload presented to the simulator, not hardware
# or a simulated policy. They must still be explicit in every resolved config,
# but assigning them a hardware evidence grade would be category error.
WORKLOAD_KEYS: dict[str, str] = {
    "line-size": "Request size used when a trace line omits an explicit byte count.",
    "interarrival-ns": "Synthetic arrival spacing used when a trace line omits at=.",
}

KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")


def load_json_strict(path: Path) -> dict:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
        value: dict = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{path}: duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=reject_duplicates)


def published_config_keys(
    config_root: Path,
) -> tuple[set[str], list[Path], dict[str, dict[str, str]]]:
    paths = sorted(config_root.rglob("*.cfg"))
    if not paths:
        raise ValueError(f"no published .cfg files found in {config_root}")
    keys: set[str] = set()
    configs: dict[str, dict[str, str]] = {}
    for path in paths:
        per_file: set[str] = set()
        values: dict[str, str] = {}
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{line_no}: expected key=value")
            key, value = (part.strip() for part in line.split("=", 1))
            if not KEY_PATTERN.fullmatch(key):
                raise ValueError(f"{path}:{line_no}: invalid config key {key!r}")
            if not value:
                raise ValueError(f"{path}:{line_no}: empty value for {key}")
            if key in per_file:
                raise ValueError(f"{path}:{line_no}: duplicate config key {key}")
            per_file.add(key)
            values[key] = value
            keys.add(key)
        relative_path = path.relative_to(config_root).as_posix()
        if relative_path in configs:
            raise ValueError(f"duplicate config path: {relative_path}")
        configs[relative_path] = values
    return keys, paths, configs


def nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_ocp_disclosure(path, disclosures, sources, configs, review_date):
    record = disclosures.get("ocp_hbf_v070", {})
    artifact = sources.get(record.get("source"), {})
    if record.get("artifact_status") != "retrieved_and_page_count_verified" or artifact.get("pages") != 130:
        raise ValueError(f"{path}: missing verified 130-page OCP artifact")
    if artifact.get("sha256") != "307531eb8053f00cbeccbc907ddff0a9c4fe6f9d0066a077ce33b0ac99312da3":
        raise ValueError(f"{path}: OCP artifact identity changed")
    for grade, channels, rate in ((1, 8, 8), (2, 16, 16), (3, 16, 32)):
        values = record["grades"][str(grade)]
        expected = 64 * rate / 8 * 0.75
        if values["payload_GBps_per_channel"] != expected or values["payload_GBps_per_stack"] != expected * channels:
            raise ValueError(f"{path}: grade {grade} double-counted efficiency or lane capacity")
        overlay = configs[f"overlays/hbf/ocp-v070-grade{grade}.cfg"]
        if int(overlay["hbf-speed-grade"]) != grade or int(overlay["hbf-channels"]) != channels:
            raise ValueError(f"{path}: grade {grade} profile mismatch")
    if not record.get("known_discrepancy") or disclosures["hbm4"].get("full_normative_timing_tables_obtained") is not False:
        raise ValueError(f"{path}: evidence limits were lost")


def validate_registry(
    path: Path,
    published: set[str],
    configs: dict[str, dict[str, str]],
) -> tuple[int, int]:
    data = load_json_strict(path)
    if data.get("schema") != {
        "name": "hbfsim.parameter-provenance",
        "version": 4,
    }:
        raise ValueError(f"{path}: unsupported or missing schema")
    review_date = data.get("as_of")
    if not isinstance(review_date, str) or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", review_date) is None:
        raise ValueError(f"{path}: as_of must be an ISO calendar date")

    grades = data.get("evidence_grades")
    if not isinstance(grades, dict) or not grades:
        raise ValueError(f"{path}: evidence_grades must be a non-empty object")
    for grade, description in grades.items():
        if not nonempty_string(grade) or not nonempty_string(description):
            raise ValueError(f"{path}: evidence grade names/descriptions must be non-empty")

    sources = data.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"{path}: sources must be an object")
    for source_id, source in sources.items():
        if not nonempty_string(source_id) or not isinstance(source, dict):
            raise ValueError(f"{path}: invalid source record {source_id!r}")
        for field in ("title", "publisher"):
            if not nonempty_string(source.get(field)):
                raise ValueError(f"{path}: source {source_id} needs non-empty {field}")
        if not (nonempty_string(source.get("url")) or
                nonempty_string(source.get("status"))):
            raise ValueError(
                f"{path}: source {source_id} needs a URL or an explicit status")

    validate_ocp_disclosure(
        path, data.get("disclosures"), sources, configs, review_date)

    parameters = data.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        raise ValueError(f"{path}: parameters must be a non-empty array")
    coverage: dict[str, list[int]] = defaultdict(list)
    for index, parameter in enumerate(parameters):
        where = f"{path}: parameters[{index}]"
        if not isinstance(parameter, dict):
            raise ValueError(f"{where} must be an object")
        config_keys = parameter.get("config_keys")
        if not isinstance(config_keys, list) or not config_keys:
            raise ValueError(f"{where}.config_keys must be a non-empty array")
        if len(config_keys) != len(set(config_keys)):
            raise ValueError(f"{where}.config_keys contains duplicates")
        for key in config_keys:
            if not isinstance(key, str) or not KEY_PATTERN.fullmatch(key):
                raise ValueError(f"{where}: invalid config key {key!r}")
            coverage[key].append(index)
        grade = parameter.get("grade")
        if grade not in grades:
            raise ValueError(f"{where}: unknown evidence grade {grade!r}")
        source = parameter.get("source")
        source_list = parameter.get("sources")
        if source is not None and source_list is not None:
            raise ValueError(
                f"{where}: source and sources are mutually exclusive"
            )
        if source_list is not None:
            if (
                not isinstance(source_list, list)
                or not source_list
                or len(source_list) != len(set(source_list))
                or not all(nonempty_string(item) for item in source_list)
            ):
                raise ValueError(
                    f"{where}.sources must be a non-empty unique string array"
                )
            source_refs = source_list
        elif source is None:
            source_refs = []
        elif nonempty_string(source):
            source_refs = [source]
        else:
            raise ValueError(f"{where}.source must be a non-empty string or null")
        for source_ref in source_refs:
            if source_ref not in sources:
                raise ValueError(
                    f"{where}: unknown source reference {source_ref!r}"
                )
        if (
            grade in {"vendor_published", "literature_derived"}
            and not source_refs
        ):
            raise ValueError(f"{where}: grade {grade} requires a source")
        if not nonempty_string(parameter.get("unit")):
            raise ValueError(f"{where}.unit must be non-empty")
        if not nonempty_string(parameter.get("notes")):
            raise ValueError(f"{where}.notes must be non-empty")
        if "profile_value" not in parameter or parameter["profile_value"] is None:
            raise ValueError(f"{where}.profile_value must be present")

    overlaps = data.get("allowed_overlaps", {})
    if not isinstance(overlaps, dict):
        raise ValueError(f"{path}: allowed_overlaps must be an object")
    for key, reason in overlaps.items():
        if not nonempty_string(reason):
            raise ValueError(f"{path}: allowed overlap {key!r} needs a reason")
        if len(coverage.get(key, [])) < 2:
            raise ValueError(f"{path}: stale allowed overlap for non-overlapping key {key}")
    for key, matches in coverage.items():
        if len(matches) > 1 and key not in overlaps:
            raise ValueError(
                f"registry key has ambiguous provenance {matches}: {key}")

    stale_workload_keys = set(WORKLOAD_KEYS) - published
    if stale_workload_keys:
        raise ValueError(
            "workload-key whitelist contains unpublished keys: " +
            ", ".join(sorted(stale_workload_keys)))
    for key, reason in WORKLOAD_KEYS.items():
        if not nonempty_string(reason):
            raise ValueError(f"workload key {key} needs an explanatory reason")
        if key in coverage:
            raise ValueError(
                f"{key} is both workload-whitelisted and provenance-graded; choose one")

    for key in sorted(published - set(WORKLOAD_KEYS)):
        matches = coverage.get(key, [])
        if not matches:
            raise ValueError(f"published config key lacks provenance: {key}")

    return len(coverage), len(overlaps)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path,
                        default=Path("configs"))
    parser.add_argument("--registry", type=Path,
                        default=Path("configs/parameter-provenance.json"))
    args = parser.parse_args()

    published, paths, configs = published_config_keys(args.config_root)
    registered, overlaps = validate_registry(args.registry, published, configs)
    print(
        f"parameter provenance PASS: {len(published)} published keys across "
        f"{len(paths)} configs; {registered} registered keys; "
        f"{len(WORKLOAD_KEYS)} workload keys; {overlaps} allowed overlaps")
    for key, reason in WORKLOAD_KEYS.items():
        print(f"  workload-only {key}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
