#!/usr/bin/env python3
"""Independent state oracle for behavior-only HBM/HBF page placement.

This module intentionally knows nothing about HBFSim device timing or C++
implementation structure. It defines the causal policy state machine used by
the differential gate when one foreground page transaction is outstanding.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable


FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
UINT64_MAX = (1 << 64) - 1
ACTION_CODE = {
    "hbm-hit": 0,
    "hbf-bypass": 1,
    "promote": 2,
}


@dataclass(frozen=True)
class Access:
    page: int
    op: str
    bytes: int = 4096


def _hash_word(fingerprint: int, word: int) -> int:
    if word < 0 or word >= 1 << 64:
        raise ValueError(f"fingerprint word is outside uint64: {word}")
    for byte in word.to_bytes(8, "little"):
        fingerprint ^= byte
        fingerprint = (fingerprint * FNV_PRIME) & ((1 << 64) - 1)
    return fingerprint


def simulate(
    accesses: Iterable[Access],
    *,
    policy: str,
    hbm_pages: int,
    history_pages: int,
    promotion_threshold: int = 2,
    page_size: int = 4096,
) -> dict[str, int | list[str]]:
    if policy not in {"always-admit", "reuse-filtered"}:
        raise ValueError(f"unsupported behavioral policy: {policy}")
    if hbm_pages <= 0 or history_pages <= 0 or page_size <= 0:
        raise ValueError("capacity and page size must be positive")
    if policy == "reuse-filtered" and promotion_threshold < 2:
        raise ValueError("reuse-filtered threshold must be at least two")

    resident: dict[int, dict[str, int | bool]] = {}
    ghosts: OrderedDict[int, int] = OrderedDict()
    actions: list[str] = []
    next_slot = 0
    stats: dict[str, int | list[str]] = {
        "page_observations": 0,
        "hbm_hits": 0,
        "hbf_bypasses": 0,
        "cold_misses": 0,
        "history_hits": 0,
        "promotions": 0,
        "promotions_with_backing_fill": 0,
        "promotions_without_backing_fill": 0,
        "clean_evictions": 0,
        "dirty_evictions": 0,
        "history_evictions": 0,
        "hbm_foreground_bytes": 0,
        "hbf_bypass_bytes": 0,
        "backing_fill_pages": 0,
        "backing_fill_bytes": 0,
        "hbm_install_pages": 0,
        "hbm_install_bytes": 0,
        "dirty_writeback_pages": 0,
        "dirty_writeback_bytes": 0,
        "drain_writeback_pages": 0,
        "drain_writeback_bytes": 0,
        "peak_resident_pages": 0,
        "final_resident_pages": 0,
        "final_dirty_pages": 0,
        "peak_history_pages": 0,
        "decision_fingerprint": FNV_OFFSET,
        "actions": actions,
    }

    def insert_ghost(page: int, observations: int) -> None:
        if page in ghosts:
            del ghosts[page]
        while len(ghosts) >= history_pages:
            ghosts.popitem(last=False)
            stats["history_evictions"] += 1
        ghosts[page] = observations
        stats["peak_history_pages"] = max(
            int(stats["peak_history_pages"]), len(ghosts))

    for request_index, access in enumerate(accesses):
        if access.op not in {"R", "W"}:
            raise ValueError(f"unsupported operation: {access.op}")
        if access.page < 0 or access.bytes <= 0:
            raise ValueError("access page/bytes must be nonnegative/positive")
        observation = int(stats["page_observations"]) + 1
        stats["page_observations"] = observation

        history_observations = 0
        slot: int | None = None
        victim: int | None = None
        victim_dirty = False
        if access.page in resident:
            action = "hbm-hit"
            stats["hbm_hits"] += 1
            entry = resident[access.page]
            slot = int(entry["slot"])
            entry["last_touch"] = observation
            if access.op == "W":
                entry["dirty"] = True
            stats["hbm_foreground_bytes"] += access.bytes
        else:
            promote = policy == "always-admit"
            if policy == "always-admit":
                stats["cold_misses"] += 1
                history_observations = 1
            elif access.page not in ghosts:
                stats["cold_misses"] += 1
                history_observations = 1
                insert_ghost(access.page, history_observations)
            else:
                stats["history_hits"] += 1
                history_observations = ghosts.pop(access.page) + 1
                promote = history_observations >= promotion_threshold
                if not promote:
                    insert_ghost(access.page, history_observations)

            if not promote:
                action = "hbf-bypass"
                stats["hbf_bypasses"] += 1
                stats["hbf_bypass_bytes"] += access.bytes
            else:
                action = "promote"
                stats["promotions"] += 1
                if access.page in ghosts:
                    del ghosts[access.page]
                if len(resident) == hbm_pages:
                    victim = min(
                        resident,
                        key=lambda page: (
                            int(resident[page]["last_touch"]), page))
                    victim_state = resident.pop(victim)
                    slot = int(victim_state["slot"])
                    victim_dirty = bool(victim_state["dirty"])
                    if victim_dirty:
                        stats["dirty_evictions"] += 1
                        stats["dirty_writeback_pages"] += 1
                        stats["dirty_writeback_bytes"] += page_size
                    else:
                        stats["clean_evictions"] += 1
                    if policy == "reuse-filtered":
                        insert_ghost(victim, promotion_threshold - 1)
                else:
                    slot = next_slot
                    next_slot += 1
                if slot is None:
                    raise AssertionError("oracle promotion lost its HBM slot")
                needs_fill = access.op == "R"
                if needs_fill:
                    stats["promotions_with_backing_fill"] += 1
                    stats["backing_fill_pages"] += 1
                    stats["backing_fill_bytes"] += page_size
                    stats["hbm_install_pages"] += 1
                    stats["hbm_install_bytes"] += page_size
                else:
                    stats["promotions_without_backing_fill"] += 1
                    stats["hbm_foreground_bytes"] += access.bytes
                resident[access.page] = {
                    "dirty": access.op == "W",
                    "last_touch": observation,
                    "slot": slot,
                }
                stats["peak_resident_pages"] = max(
                    int(stats["peak_resident_pages"]), len(resident))

        actions.append(action)
        fingerprint = int(stats["decision_fingerprint"])
        for word in (
            observation,
            request_index,
            0,  # one full-page transaction per generated request
            access.page,
            0 if access.op == "R" else 1,
            ACTION_CODE[action],
            history_observations,
            UINT64_MAX if slot is None else slot,
            UINT64_MAX if victim is None else victim,
            1 if victim_dirty else 0,
        ):
            fingerprint = _hash_word(fingerprint, word)
        stats["decision_fingerprint"] = fingerprint

    dirty_at_drain = sum(
        bool(state["dirty"]) for state in resident.values())
    stats["drain_writeback_pages"] = dirty_at_drain
    stats["drain_writeback_bytes"] = dirty_at_drain * page_size
    stats["dirty_writeback_pages"] += dirty_at_drain
    stats["dirty_writeback_bytes"] += dirty_at_drain * page_size
    stats["final_resident_pages"] = len(resident)
    stats["final_dirty_pages"] = 0
    return stats
