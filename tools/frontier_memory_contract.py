#!/usr/bin/env python3
"""Canonical compact Frontier allocator-state contract shared by HBFSim tools."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


MEMORY_CONTRACT_SCHEMA_VERSION = 4
ALLOCATOR_STATE_SCHEMA_VERSION = 1
RESIDENCY_PLAN_SCHEMA_VERSION = 1


def canonical_allocator_state(
    request_id: str,
    blocks: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the exact JSON object covered by ``allocator_state_sha256``."""

    normalized_blocks: list[dict[str, Any]] = []
    for block in blocks:
        normalized_blocks.append(
            {
                "block_id": int(block["block_id"]),
                "block_hash": (
                    None
                    if block.get("block_hash") is None
                    else int(block["block_hash"])
                ),
                "ref_count": int(block["ref_count"]),
            }
        )
    return {
        "schema_version": ALLOCATOR_STATE_SCHEMA_VERSION,
        "request_id": str(request_id),
        "blocks": normalized_blocks,
    }


def allocator_state_sha256(
    request_id: str,
    blocks: Iterable[Mapping[str, Any]],
) -> str:
    """Hash one request's ordered physical-block ownership snapshot."""

    encoded = json.dumps(
        canonical_allocator_state(request_id, blocks),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
