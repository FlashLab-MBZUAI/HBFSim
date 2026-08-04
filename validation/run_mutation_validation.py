#!/usr/bin/env python3
"""Prove foundational gates kill intentional production-code defects."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPORT_SCHEMA = {
    "name": "hbfsim.validation.mutation-report",
    "version": 1,
}


@dataclass(frozen=True)
class Mutation:
    mutation_id: str
    description: str
    source: str
    old: str
    new: str
    replacement_count: int
    cases: tuple[str, ...]
    gate: str = "canonical"
    critical: bool = True

    def patch_digest(self) -> str:
        payload = json.dumps(
            {
                "id": self.mutation_id,
                "source": self.source,
                "old": self.old,
                "new": self.new,
                "replacement_count": self.replacement_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


MUTATIONS = (
    Mutation(
        "MUT-HBM-01",
        "suppress PRE when an open bank conflicts with the requested row",
        "src/physical/hbm/hbm_device.cpp",
        "if (bank.has_open_row && bank.open_row != addr.row) {",
        "if (bank.has_open_row && false) {",
        1,
        ("hbm.row-miss-hit-conflict",),
    ),
    Mutation(
        "MUT-HBM-02",
        "use short bank-group spacing where tCCD_L is required",
        "src/physical/hbm/hbm_device.cpp",
        "issue_ns + tccd_l_ns_,",
        "issue_ns + tccd_s_ns_,",
        2,
        ("hbm.row-miss-hit-conflict",),
    ),
    Mutation(
        "MUT-HBM-03",
        "route every address through one pseudo-channel scheduler state",
        "src/physical/hbm/hbm_device.cpp",
        """std::size_t HbmDevice::pseudo_channel_index(const HbmAddress& addr) const {
    return (static_cast<std::size_t>(addr.stack) * config_.channels_per_stack + addr.channel) *
        config_.pseudo_channels_per_channel + addr.pseudo_channel;
}""",
        """std::size_t HbmDevice::pseudo_channel_index(const HbmAddress& addr) const {
    (void)addr;
    return 0;
}""",
        1,
        ("hbm.dual-pch-overlap",),
    ),
    Mutation(
        "MUT-HBM-04",
        "drop the partial-edge signature from the exact pseudo-channel "
        "state-class partition",
        "src/physical/hbm/hbm_device.cpp",
        """classes[{pseudo_channel_class_ids_.at(index),
                 edge_signatures[index]}].push_back(index);""",
        """classes[{pseudo_channel_class_ids_.at(index),
                 0}].push_back(index);""",
        1,
        ("physical.hbm-frfcfs",),
        "physical",
    ),
    Mutation(
        "MUT-HBF-01",
        "return ECC issue-slot completion instead of response latency",
        "src/physical/hbf/hbf_device.cpp",
        "const double finish_ns = issue.start_ns + latency_ns;",
        "const double finish_ns = issue.finish_ns;",
        1,
        ("hbf.read-single",),
    ),
    Mutation(
        "MUT-HBF-02",
        "collapse every HBF plane onto one media-resource state",
        "src/physical/hbf/hbf_device.cpp",
        """std::size_t HbfDevice::plane_index(const HbfAddress& addr) const {
    std::size_t index = die_index(addr);
    index = index * config_.planes_per_die + addr.plane;
    return index;
}""",
        """std::size_t HbfDevice::plane_index(const HbfAddress& addr) const {
    (void)addr;
    return 0;
}""",
        1,
        ("hbf.read-cross-plane",),
    ),
    Mutation(
        "MUT-HBF-03",
        "publish an L2P update before its media program can commit",
        "src/physical/hbf/hbf_device.cpp",
        """schedule_lpn_mapping_commit(
                    lpn, new_ppn, page_done, program_commit_sequence);""",
        """schedule_lpn_mapping_commit(
                    lpn, new_ppn, 0.0, program_commit_sequence);""",
        1,
        ("hbf.write-full-drain",),
    ),
    Mutation(
        "MUT-FTL-01",
        "overwrite an LPN without invalidating its old physical page",
        "src/physical/hbf/hbf_device.cpp",
        """if (current != lpn_to_ppn_.end() && current->second != new_ppn) {
                    invalidate_ppn(current->second);
                } else if (current == lpn_to_ppn_.end() &&
                           compact_lpn_ppn(lpn)) {
                    retire_compact_page(lpn, PageOwner::Logical);
                }
                lpn_to_ppn_[lpn] = new_ppn;""",
        """if (current != lpn_to_ppn_.end() && current->second != new_ppn) {
                    (void)current;
                } else if (current == lpn_to_ppn_.end() &&
                           compact_lpn_ppn(lpn)) {
                    retire_compact_page(lpn, PageOwner::Logical);
                }
                lpn_to_ppn_[lpn] = new_ppn;""",
        1,
        ("hbf.overwrite-no-gc",),
    ),
    Mutation(
        "MUT-FTL-02",
        "relocate live GC data without redirecting its logical mapping",
        "src/physical/hbf/hbf_device.cpp",
        """        } else {
            lpn_to_ppn_[lpn] = new_ppn;
            mark_mapping_page_dirty(
                mapping_vpn_for_lpn(lpn), mapping_done);
        }""",
        """        } else {
            mark_mapping_page_dirty(
                mapping_vpn_for_lpn(lpn), mapping_done);
        }""",
        1,
        ("hbf.gc-live-relocation",),
    ),
    Mutation(
        "MUT-FTL-03",
        "omit mapping-checkpoint media bytes from physical-write accounting",
        "src/physical/hbf/hbf_device.cpp",
        """stats_.physical_write_bytes += config_.page_size_bytes;
    stats_.mapping_program_payload_bytes += config_.page_size_bytes;""",
        """stats_.physical_write_bytes += 0;
    stats_.mapping_program_payload_bytes += config_.page_size_bytes;""",
        1,
        ("hbf.write-full-drain",),
    ),
    Mutation(
        "MUT-HYB-01",
        "serialize each mixed-tier transaction behind all prior tier work",
        "src/physical/hybrid/direct_composition.cpp",
        """segment.arrival_ns = std::max(
                window_admit_ns,
                foreground_admit_frontier_ns);""",
        """segment.arrival_ns = std::max({
                window_admit_ns,
                foreground_admit_frontier_ns,
                result.finish_ns});""",
        1,
        ("hybrid.independent-overlap",),
    ),
    Mutation(
        "MUT-HYB-02",
        "let layer-streaming backing DMA bypass its completion-order "
        "credit pool",
        "src/physical/hybrid/layer_streaming_composition.cpp",
        """        return limit == 0 || inflight_backing_transactions_ < limit;""",
        """        (void)limit;
        return true;""",
        1,
        ("physical.layer-backing-credit",),
        "physical",
    ),
    Mutation(
        "MUT-METRIC-01",
        "sum child completion timestamps instead of taking their maximum",
        "src/physical/hybrid/direct_composition.cpp",
        """parent.finish_ns = std::max(
                    parent.finish_ns,
                    transaction_finish_ns);""",
        "parent.finish_ns += transaction_finish_ns;",
        1,
        ("hybrid.parent-split",),
    ),
    Mutation(
        "MUT-EXT-01",
        "omit returned completion protocol bytes from every S2M transfer",
        "src/physical/external/external_backing_device.cpp",
        "timing.s2m_protocol_bytes = config_.completion_bytes;",
        "timing.s2m_protocol_bytes = 0;",
        1,
        ("external.read-single", "external.write-single"),
    ),
    Mutation(
        "MUT-EXT-02",
        "replace earliest-gap serial scheduling with a call-order frontier",
        "src/physical/external/external_backing_device.cpp",
        "double start_ns = ready_ns;",
        """double start_ns = intervals_.empty() ?
        ready_ns :
        std::max(ready_ns, intervals_.rbegin()->second);""",
        1,
        ("external.future-gap-backfill",),
    ),
    Mutation(
        "MUT-BEH-01",
        "promote a reuse-filtered page on its first observation",
        "src/physical/hybrid/behavioral_tiering_composition.cpp",
        """            insert_or_update_ghost(page, 1);
            return {false, 1};""",
        """            insert_or_update_ghost(page, 1);
            return {true, 1};""",
        1,
        (
            "behavioral.composition-unit",
            "behavioral.independent-differential",
        ),
        "behavioral",
    ),
    Mutation(
        "MUT-BEH-02",
        "discard dirty state after an HBM write",
        "src/physical/hybrid/behavioral_tiering_composition.cpp",
        """        auto& resident = found->second;
        resident.busy = true;
        const auto& request = requests_.at(transaction->request_index);
        if (request.op == Op::Write) {
            resident.dirty = true;
        }""",
        """        auto& resident = found->second;
        resident.busy = true;
        const auto& request = requests_.at(transaction->request_index);
        if (request.op == Op::Write) {
            resident.dirty = false;
        }""",
        1,
        (
            "behavioral.composition-unit",
            "behavioral.independent-differential",
        ),
        "behavioral",
    ),
    Mutation(
        "MUT-BEH-03",
        "route scratch annotations differently from the same address stream",
        "src/physical/hybrid/behavioral_tiering_composition.cpp",
        """    void plan_transaction(const std::shared_ptr<Transaction>& transaction) {
        const auto& request = requests_.at(transaction->request_index);
        auto resident = policy_residents_.find(transaction->page);""",
        """    void plan_transaction(const std::shared_ptr<Transaction>& transaction) {
        const auto& request = requests_.at(transaction->request_index);
        if (request.kind == SemanticKind::Scratch) {
            transaction->planned_action =
                BehavioralDecisionAction::HbfBypass;
            ++result_.placement.hbf_bypasses;
            record_decision(
                transaction, BehavioralDecisionAction::HbfBypass, 0);
            return;
        }
        auto resident = policy_residents_.find(transaction->page);""",
        1,
        (
            "behavioral.composition-unit",
            "behavioral.independent-differential",
        ),
        "behavioral",
    ),
    Mutation(
        "MUT-BEH-04",
        "evict the most-recently used idle HBM page instead of the LRU page",
        "src/physical/hybrid/behavioral_tiering_composition.cpp",
        """            if (resident.last_touch < current.last_touch ||
                (resident.last_touch == current.last_touch && page < victim)) {""",
        """            if (resident.last_touch > current.last_touch ||
                (resident.last_touch == current.last_touch && page < victim)) {""",
        1,
        (
            "behavioral.composition-unit",
            "behavioral.independent-differential",
        ),
        "behavioral",
    ),
    Mutation(
        "MUT-BEH-05",
        "skip the HBF-to-HBM data-transfer link during a promotion fill",
        "src/physical/hybrid/behavioral_tiering_composition.cpp",
        """            [this, page, slot] { issue_fill_d2d(page, slot); });""",
        """            [this, page, slot] { issue_hbm_install(page, slot); });""",
        1,
        (
            "behavioral.composition-unit",
            "behavioral.independent-differential",
        ),
        "behavioral",
    ),
    Mutation(
        "MUT-BEH-06",
        "let a replacement overtake earlier decisions on its victim page",
        "src/physical/hybrid/behavioral_tiering_composition.cpp",
        """        if (victim != planned_page_observations_.end() &&
            !victim->second.empty() &&
            victim->second.front() < transaction->decision_observation) {
            return *transaction->planned_victim;
        }""",
        """        if (victim != planned_page_observations_.end() &&
            !victim->second.empty() &&
            victim->second.front() < transaction->decision_observation &&
            !transaction->planned_victim.has_value()) {
            return *transaction->planned_victim;
        }""",
        1,
        (
            "behavioral.composition-unit",
            "behavioral.independent-differential",
        ),
        "behavioral",
    ),
    Mutation(
        "MUT-BEH-07",
        "ignore explicit phase dependencies when offering behavioral requests",
        "src/physical/hybrid/behavioral_tiering_composition.cpp",
        """            const auto offered_arrival_ns =
                std::max(source_arrival_ns, dependency_ready_ns);""",
        """            const auto offered_arrival_ns =
                dependency_ready_ns >= 0.0 ?
                    source_arrival_ns :
                    dependency_ready_ns;""",
        1,
        (
            "behavioral.composition-unit",
            "behavioral.independent-differential",
        ),
        "behavioral",
    ),
    Mutation(
        "MUT-BEH-08",
        "let one multi-page parent monopolize the foreground window",
        "src/physical/hybrid/behavioral_tiering_composition.cpp",
        """        parent.admission_ready = true;
        admission_ready_parents_.push_back(request_index);""",
        """        parent.admission_ready = true;
        admission_ready_parents_.push_front(request_index);""",
        1,
        (
            "behavioral.composition-unit",
            "behavioral.independent-differential",
        ),
        "behavioral",
    ),
    Mutation(
        "MUT-BEH-09",
        "let a planned HBM hit overtake its page's earlier promotion",
        "src/physical/hybrid/behavioral_tiering_composition.cpp",
        """        if (own->second.front() < transaction->decision_observation) {
            return transaction->page;
        }""",
        """        if (own->second.front() < transaction->decision_observation &&
            transaction->decision_observation == 0) {
            return transaction->page;
        }""",
        1,
        (
            "behavioral.composition-unit",
            "behavioral.independent-differential",
        ),
        "behavioral",
    ),
    Mutation(
        "MUT-BEH-10",
        "omit the victim-page tombstone for a planned replacement",
        "src/physical/hybrid/behavioral_tiering_composition.cpp",
        "            victim_observations.push_back(observation);",
        """            if (observation == 0) {
                victim_observations.push_back(observation);
            }""",
        1,
        (
            "behavioral.composition-unit",
            "behavioral.independent-differential",
        ),
        "behavioral",
    ),
)


def _run(
    command: list[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _diagnostic(result: subprocess.CompletedProcess[str]) -> str:
    combined = (
        f"command={' '.join(result.args)}\n"
        f"exit_code={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return combined[-8000:]


def _git(root: Path, *arguments: str) -> str:
    result = _run(["git", *arguments], cwd=root)
    if result.returncode:
        raise RuntimeError(_diagnostic(result))
    return result.stdout.strip()


def _archive_head(root: Path, destination: Path) -> tuple[str, str]:
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if archive.returncode:
        raise RuntimeError(
            "git archive failed:\n"
            + archive.stderr.decode(errors="replace"))
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
        # The archive is generated from this repository's own committed tree,
        # not accepted from an external input.
        tar.extractall(destination)
    return commit, tree


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _case_paths(source_root: Path, mutation: Mutation) -> list[Path]:
    if mutation.gate != "canonical":
        raise RuntimeError(
            f"{mutation.mutation_id}: non-canonical mutation has no case files")
    return [
        source_root / "validation/cases" / f"{case_id}.json"
        for case_id in mutation.cases
    ]


def _validation_command(
    source_root: Path,
    probe: Path,
    cases: list[Path],
    artifact_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        str(source_root / "validation/run_validation.py"),
        "--probe",
        str(probe),
        "--artifact-dir",
        str(artifact_dir),
    ]
    for case in cases:
        command.extend(["--case", str(case)])
    return command


def _apply_mutation(path: Path, mutation: Mutation) -> str:
    original = path.read_text()
    observed = original.count(mutation.old)
    if observed != mutation.replacement_count:
        raise RuntimeError(
            f"{mutation.mutation_id}: expected "
            f"{mutation.replacement_count} replacement site(s), "
            f"found {observed}")
    path.write_text(original.replace(mutation.old, mutation.new))
    return original


def _configure(
    source_root: Path,
    build_root: Path,
    runtime_root: Path,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(build_root),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=ON",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={runtime_root}",
            "-DCMAKE_CXX_FLAGS=-Wall -Wextra -Wpedantic -Werror",
        ],
        cwd=source_root,
    )


def _build(
    source_root: Path,
    build_root: Path,
    parallel: int,
    targets: tuple[str, ...],
    *,
    clean_first: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = ["cmake", "--build", str(build_root)]
    if clean_first:
        command.append("--clean-first")
    command.append("--target")
    command.extend(targets)
    command.extend(["--parallel", str(parallel)])
    return _run(command, cwd=source_root)


def _targets_for(mutations: list[Mutation]) -> tuple[str, ...]:
    targets: list[str] = []
    if any(mutation.gate == "canonical" for mutation in mutations):
        targets.append("validation_probe")
    if any(mutation.gate == "behavioral" for mutation in mutations):
        targets.extend((
            "behavioral_tiering_composition_test",
            "scenario_compare",
        ))
    if any(mutation.gate == "physical" for mutation in mutations):
        targets.append("physical_probe")
    unknown = {
        mutation.gate
        for mutation in mutations
        if mutation.gate not in {"canonical", "behavioral", "physical"}
    }
    if unknown:
        raise RuntimeError(f"unsupported mutation gates: {sorted(unknown)}")
    return tuple(targets)


def _behavioral_commands(
    source_root: Path,
    runtime_root: Path,
    artifact_dir: Path,
) -> tuple[tuple[str, list[str]], ...]:
    return (
        (
            "behavioral.composition-unit",
            [str(runtime_root / "behavioral_tiering_composition_test")],
        ),
        (
            "behavioral.independent-differential",
            [
                sys.executable,
                "-B",
                str(source_root / "validation/run_behavioral_differential.py"),
                "--scenario-compare",
                str(runtime_root / "scenario_compare"),
                "--config",
                str(
                    source_root
                    / "configs/scenario_compare/server-4k-hbf4x.cfg"
                ),
                "--seeds",
                "8",
                "--operations",
                "28",
                "--report",
                str(artifact_dir / "behavioral-differential.json"),
            ],
        ),
    )


def _physical_commands(
    runtime_root: Path,
) -> tuple[tuple[str, list[str]], ...]:
    return (
        (
            "physical.hbm-frfcfs",
            [str(runtime_root / "physical_probe"), "hbm-frfcfs"],
        ),
        (
            "physical.layer-backing-credit",
            [
                str(runtime_root / "physical_probe"),
                "composition-reuse-routing",
            ],
        ),
    )


def _run_gate(
    *,
    mutation: Mutation,
    source_root: Path,
    runtime_root: Path,
    artifact_dir: Path,
) -> tuple[subprocess.CompletedProcess[str], str | None]:
    if mutation.gate == "canonical":
        result = _run(
            _validation_command(
                source_root,
                runtime_root / "validation_probe",
                _case_paths(source_root, mutation),
                artifact_dir,
            ),
            cwd=source_root,
        )
        return result, mutation.cases[0] if result.returncode else None
    if mutation.gate == "behavioral":
        diagnostics: list[str] = []
        last: subprocess.CompletedProcess[str] | None = None
        for case_id, command in _behavioral_commands(
            source_root, runtime_root, artifact_dir
        ):
            if case_id not in mutation.cases:
                continue
            last = _run(command, cwd=source_root)
            diagnostics.append(_diagnostic(last))
            if last.returncode:
                last.stdout = "\n\n".join(diagnostics)
                last.stderr = ""
                return last, case_id
        if last is None:
            raise RuntimeError(
                f"{mutation.mutation_id}: behavioral gate has no commands")
        last.stdout = "\n\n".join(diagnostics)
        last.stderr = ""
        return last, None
    if mutation.gate == "physical":
        for case_id, command in _physical_commands(runtime_root):
            if case_id != mutation.cases[0]:
                continue
            result = _run(command, cwd=source_root)
            return result, case_id if result.returncode else None
        raise RuntimeError(
            f"{mutation.mutation_id}: physical gate has no command")
    raise RuntimeError(
        f"{mutation.mutation_id}: unsupported gate {mutation.gate!r}")


def _run_gate_set(
    *,
    mutations: list[Mutation],
    source_root: Path,
    runtime_root: Path,
    artifact_dir: Path,
) -> subprocess.CompletedProcess[str] | None:
    canonical = [
        mutation for mutation in mutations
        if mutation.gate == "canonical"
    ]
    if canonical:
        cases = sorted({
            case
            for mutation in canonical
            for case in _case_paths(source_root, mutation)
        })
        result = _run(
            _validation_command(
                source_root,
                runtime_root / "validation_probe",
                cases,
                artifact_dir / "canonical",
            ),
            cwd=source_root,
        )
        if result.returncode:
            return result
    behavioral = [
        mutation for mutation in mutations
        if mutation.gate == "behavioral"
    ]
    if behavioral:
        selected_cases = {
            case for mutation in behavioral for case in mutation.cases
        }
        for case_id, command in _behavioral_commands(
            source_root,
            runtime_root,
            artifact_dir / "behavioral",
        ):
            if case_id not in selected_cases:
                continue
            result = _run(command, cwd=source_root)
            if result.returncode:
                return result
    physical = [
        mutation for mutation in mutations
        if mutation.gate == "physical"
    ]
    if physical:
        selected_cases = {
            case for mutation in physical for case in mutation.cases
        }
        for case_id, command in _physical_commands(runtime_root):
            if case_id not in selected_cases:
                continue
            result = _run(command, cwd=source_root)
            if result.returncode:
                return result
    return None


def run_mutations(
    *,
    repository: Path,
    report_path: Path,
    artifact_dir: Path,
    parallel: int,
    selected_ids: set[str] | None,
) -> dict[str, Any]:
    tracked_status = _git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status:
        raise RuntimeError(
            "mutation evidence requires a clean tracked worktree; "
            "commit or restore tracked changes first")
    selected = [
        mutation
        for mutation in MUTATIONS
        if selected_ids is None or mutation.mutation_id in selected_ids
    ]
    if not selected:
        raise RuntimeError("no mutations selected")
    known_ids = {mutation.mutation_id for mutation in MUTATIONS}
    if selected_ids is not None and selected_ids - known_ids:
        raise RuntimeError(
            f"unknown mutation IDs: {sorted(selected_ids - known_ids)}")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "source_commit": "",
        "source_tree": "",
        "mutations": [],
        "summary": {},
    }
    with tempfile.TemporaryDirectory(
        prefix="hbfsim-mutation-validation-"
    ) as directory:
        temp_root = Path(directory)
        source_root = temp_root / "source"
        build_root = temp_root / "build"
        runtime_root = temp_root / "bin"
        source_root.mkdir()
        runtime_root.mkdir()
        commit, tree = _archive_head(repository, source_root)
        report["source_commit"] = commit
        report["source_tree"] = tree

        configure = _configure(source_root, build_root, runtime_root)
        if configure.returncode:
            raise RuntimeError(
                "mutation baseline configure failed:\n"
                + _diagnostic(configure))
        targets = _targets_for(selected)
        baseline_build = _build(
            source_root, build_root, parallel, targets)
        if baseline_build.returncode:
            raise RuntimeError(
                "mutation baseline build failed:\n"
                + _diagnostic(baseline_build))
        baseline = _run_gate_set(
            mutations=selected,
            source_root=source_root,
            runtime_root=runtime_root,
            artifact_dir=artifact_dir / "baseline",
        )
        if baseline is not None:
            raise RuntimeError(
                "unmutated baseline did not pass selected cases:\n"
                + _diagnostic(baseline))

        for mutation in selected:
            source_path = source_root / mutation.source
            started = time.monotonic()
            entry: dict[str, Any] = {
                "id": mutation.mutation_id,
                "description": mutation.description,
                "critical": mutation.critical,
                "source": mutation.source,
                "patch_digest": mutation.patch_digest(),
                "cases": list(mutation.cases),
                "status": "invalid_patch",
                "killed_by": None,
                "duration_seconds": 0.0,
                "diagnostic": "",
            }
            original: str | None = None
            try:
                original = _apply_mutation(source_path, mutation)
                # A clean build is intentional. Rapid replace/restore cycles
                # can share one filesystem timestamp tick and otherwise let an
                # incremental generator reuse an unmutated object.
                build = _build(
                    source_root,
                    build_root,
                    parallel,
                    _targets_for([mutation]),
                    clean_first=True,
                )
                if build.returncode:
                    entry["status"] = "invalid_build"
                    entry["diagnostic"] = _diagnostic(build)
                else:
                    validation, killed_by = _run_gate(
                        mutation=mutation,
                        source_root=source_root,
                        runtime_root=runtime_root,
                        artifact_dir=(
                            artifact_dir / mutation.mutation_id
                        ),
                    )
                    if validation.returncode:
                        entry["status"] = "killed"
                        entry["killed_by"] = killed_by
                    else:
                        entry["status"] = "survived"
                    entry["diagnostic"] = _diagnostic(validation)
            except (OSError, RuntimeError) as error:
                entry["status"] = "invalid_patch"
                entry["diagnostic"] = str(error)
            finally:
                if original is not None:
                    source_path.write_text(original)
                    if source_path.read_text() != original:
                        entry["status"] = "restore_failed"
                        entry["diagnostic"] = (
                            f"{mutation.mutation_id}: source restore "
                            "verification failed"
                        )
                entry["duration_seconds"] = round(
                    time.monotonic() - started, 6)
                report["mutations"].append(entry)
                _atomic_write_json(report_path, report)
                print(
                    f"{entry['status'].upper()} {mutation.mutation_id}: "
                    f"{', '.join(mutation.cases)}",
                    flush=True,
                )

        final_build = _build(
            source_root,
            build_root,
            parallel,
            targets,
            clean_first=True,
        )
        if final_build.returncode:
            raise RuntimeError(
                "restored mutation baseline build failed:\n"
                + _diagnostic(final_build))
        restored_baseline = _run_gate_set(
            mutations=selected,
            source_root=source_root,
            runtime_root=runtime_root,
            artifact_dir=artifact_dir / "restored-baseline",
        )
        if restored_baseline is not None:
            raise RuntimeError(
                "restored mutation baseline validation failed:\n"
                + _diagnostic(restored_baseline))

    killed = sum(
        entry["status"] == "killed" for entry in report["mutations"])
    total = len(report["mutations"])
    critical_total = sum(
        entry["critical"] for entry in report["mutations"])
    critical_killed = sum(
        entry["critical"] and entry["status"] == "killed"
        for entry in report["mutations"]
    )
    report["summary"] = {
        "total": total,
        "killed": killed,
        "survived": sum(
            entry["status"] == "survived"
            for entry in report["mutations"]
        ),
        "invalid": sum(
            entry["status"] not in {"killed", "survived"}
            for entry in report["mutations"]
        ),
        "kill_rate": killed / total,
        "critical_total": critical_total,
        "critical_killed": critical_killed,
        "passed": (
            killed / total >= 0.9
            and critical_killed == critical_total
        ),
    }
    _atomic_write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("build/foundational-mutation-report.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("build/foundational-mutation-artifacts"),
    )
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument(
        "--mutation",
        action="append",
        dest="mutations",
    )
    args = parser.parse_args()
    if args.parallel <= 0:
        parser.error("--parallel must be positive")
    repository = args.repository.resolve()
    if not (repository / ".git").exists():
        parser.error(f"not a Git worktree: {repository}")
    try:
        report = run_mutations(
            repository=repository,
            report_path=args.report.resolve(),
            artifact_dir=args.artifact_dir.resolve(),
            parallel=args.parallel,
            selected_ids=(
                set(args.mutations) if args.mutations else None
            ),
        )
    except RuntimeError as error:
        print(f"foundational mutation validation failed: {error}",
              file=sys.stderr)
        return 1
    summary = report["summary"]
    if not summary["passed"]:
        print(
            "foundational mutation validation failed: "
            f"killed={summary['killed']}/{summary['total']}, "
            f"critical={summary['critical_killed']}/"
            f"{summary['critical_total']}, "
            f"report={args.report.resolve()}",
            file=sys.stderr,
        )
        return 1
    print(
        "PASS foundational mutation validation: "
        f"killed {summary['killed']}/{summary['total']} production "
        f"mutations; report={args.report.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
