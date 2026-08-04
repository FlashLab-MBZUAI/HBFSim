#!/usr/bin/env python3
"""Apply or verify the one supported HBFSim patch on a pinned Frontier checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_MANIFEST = REPOSITORY_ROOT / "integrations/frontier/manifest.json"


class IntegrationError(ValueError):
    """The Frontier checkout or tracked integration bundle is not exact."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrationError(
            f"cannot load {description} {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise IntegrationError(f"{description} must be a JSON object: {path}")
    return value


def _string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise IntegrationError(f"{description} must be a non-empty string")
    return value


def _sha256(value: Any, description: str) -> str:
    result = _string(value, description)
    if (
        len(result) != 64
        or result != result.lower()
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise IntegrationError(
            f"{description} must be 64 lowercase hexadecimal characters"
        )
    return result


def validate_bundle() -> tuple[dict[str, Any], Path, bytes]:
    manifest = _load_object(BUNDLE_MANIFEST, "integration manifest")
    if manifest.get("schema") != {
        "name": "hbfsim.frontier_integration_patch",
        "version": 1,
    }:
        raise IntegrationError("unsupported Frontier integration manifest schema")
    upstream = manifest.get("upstream")
    patch = manifest.get("patch")
    contracts = manifest.get("contracts")
    post_hashes = manifest.get("post_apply_sha256")
    if not isinstance(upstream, dict) or not isinstance(patch, dict):
        raise IntegrationError("integration manifest has malformed source metadata")
    if not isinstance(contracts, dict) or not isinstance(post_hashes, dict):
        raise IntegrationError("integration manifest has malformed contract metadata")
    _string(manifest.get("integration_id"), "integration_id")
    revision = _string(upstream.get("revision"), "upstream revision")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise IntegrationError("upstream revision must be a full lowercase git SHA")
    if contracts != {
        "memory_contract_schema_version": 4,
        "allocator_state_schema_version": 1,
        "residency_plan_schema_version": 1,
        "kv_lifecycle_schema_version": 1,
    }:
        raise IntegrationError("unexpected Frontier integration contract versions")

    patch_relative = Path(_string(patch.get("path"), "patch path"))
    patch_path = (REPOSITORY_ROOT / patch_relative).resolve()
    try:
        patch_path.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise IntegrationError("patch path escapes the HBFSim repository") from error
    if not patch_path.is_file():
        raise IntegrationError(f"Frontier integration patch is missing: {patch_path}")
    patch_bytes = patch_path.read_bytes()
    if patch.get("bytes") != len(patch_bytes):
        raise IntegrationError("Frontier integration patch byte size mismatch")
    if _sha256_file(patch_path) != _sha256(
        patch.get("sha256"), "patch sha256"
    ):
        raise IntegrationError("Frontier integration patch digest mismatch")

    expected_paths = set()
    for path_text, digest in post_hashes.items():
        normalized = Path(_string(path_text, "post-apply path")).as_posix()
        if (
            normalized != path_text
            or normalized.startswith("../")
            or normalized.startswith("/")
        ):
            raise IntegrationError(f"invalid post-apply path: {path_text!r}")
        _sha256(digest, f"post-apply digest for {path_text}")
        expected_paths.add(path_text)
    patch_paths = {
        match.group(1)
        for match in re.finditer(
            rb"^diff --git a/(\S+) b/\S+$",
            patch_bytes,
            flags=re.MULTILINE,
        )
    }
    decoded_patch_paths = {path.decode("utf-8") for path in patch_paths}
    if decoded_patch_paths != expected_paths:
        raise IntegrationError(
            "patch paths do not equal the manifest post-apply file set"
        )
    return manifest, patch_path, patch_bytes


def _git(frontier_dir: Path, *arguments: str, text: bool = True) -> str | bytes:
    process = subprocess.run(
        ["git", "-C", str(frontier_dir), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    if process.returncode != 0:
        stderr = (
            process.stderr.strip()
            if text
            else process.stderr.decode("utf-8", errors="replace").strip()
        )
        raise IntegrationError(
            f"git {' '.join(arguments)} failed in {frontier_dir}: {stderr}"
        )
    return process.stdout


def _require_base_revision(
    frontier_dir: Path,
    manifest: dict[str, Any],
) -> None:
    actual = str(_git(frontier_dir, "rev-parse", "HEAD")).strip()
    expected = str(manifest["upstream"]["revision"])
    if actual != expected:
        raise IntegrationError(
            f"Frontier HEAD must be {expected}, got {actual}"
        )


def _require_clean(frontier_dir: Path) -> None:
    status = str(
        _git(frontier_dir, "status", "--porcelain", "--untracked-files=all")
    )
    if status:
        raise IntegrationError(
            "Frontier checkout must be completely clean before patching:\n"
            + status.rstrip()
        )


def _verify_applied(
    frontier_dir: Path,
    manifest: dict[str, Any],
    patch_bytes: bytes,
) -> dict[str, Any]:
    unstaged = str(_git(frontier_dir, "diff", "--name-only"))
    untracked = str(
        _git(frontier_dir, "ls-files", "--others", "--exclude-standard")
    )
    if unstaged or untracked:
        raise IntegrationError(
            "verified Frontier integration may not contain unstaged or "
            "untracked files"
        )
    staged_diff = bytes(
        _git(
            frontier_dir,
            "diff",
            "--cached",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            text=False,
        )
    )
    if staged_diff != patch_bytes:
        raise IntegrationError(
            "Frontier staged diff is not byte-identical to the tracked patch"
        )
    check = subprocess.run(
        ["git", "-C", str(frontier_dir), "diff", "--cached", "--check"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if check.returncode != 0:
        raise IntegrationError("Frontier staged patch fails diff --check")
    actual_paths = set(
        str(_git(frontier_dir, "diff", "--cached", "--name-only"))
        .strip()
        .splitlines()
    )
    expected_hashes = manifest["post_apply_sha256"]
    if actual_paths != set(expected_hashes):
        raise IntegrationError("Frontier staged paths differ from the manifest")
    for relative, expected_digest in expected_hashes.items():
        path = frontier_dir / relative
        if not path.is_file() or _sha256_file(path) != expected_digest:
            raise IntegrationError(
                f"post-apply Frontier file digest mismatch: {relative}"
            )
    return {
        "schema": {
            "name": "hbfsim.frontier_integration_receipt",
            "version": 1,
        },
        "result": "pass",
        "integration_id": manifest["integration_id"],
        "upstream": manifest["upstream"],
        "patch": manifest["patch"],
        "contracts": manifest["contracts"],
        "staged_diff_sha256": _sha256_bytes(staged_diff),
        "post_apply_sha256": expected_hashes,
    }


def prepare(
    *,
    action: str,
    frontier_dir: Path,
) -> dict[str, Any]:
    manifest, patch_path, patch_bytes = validate_bundle()
    frontier_dir = frontier_dir.resolve()
    if not frontier_dir.is_dir():
        raise IntegrationError(
            f"Frontier checkout does not exist: {frontier_dir}"
        )
    _require_base_revision(frontier_dir, manifest)
    if action in {"check", "apply"}:
        _require_clean(frontier_dir)
        process = subprocess.run(
            [
                "git",
                "-C",
                str(frontier_dir),
                "apply",
                "--check",
                "--whitespace=error-all",
                str(patch_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if process.returncode != 0:
            raise IntegrationError(
                "tracked Frontier patch does not apply exactly:\n"
                + process.stdout.rstrip()
            )
        if action == "check":
            return {
                "schema": {
                    "name": "hbfsim.frontier_integration_check",
                    "version": 1,
                },
                "result": "pass",
                "integration_id": manifest["integration_id"],
                "upstream": manifest["upstream"],
                "patch": manifest["patch"],
            }
        apply_process = subprocess.run(
            [
                "git",
                "-C",
                str(frontier_dir),
                "apply",
                "--index",
                "--whitespace=error-all",
                str(patch_path),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if apply_process.returncode != 0:
            raise IntegrationError(
                "failed to apply tracked Frontier patch atomically:\n"
                + apply_process.stdout.rstrip()
            )
    elif action != "verify":
        raise IntegrationError(f"unsupported integration action: {action}")
    return _verify_applied(frontier_dir, manifest, patch_bytes)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "apply", "verify"))
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    arguments = parser.parse_args()
    try:
        result = prepare(
            action=arguments.action,
            frontier_dir=arguments.frontier_dir,
        )
        if arguments.receipt is not None:
            _atomic_write_json(arguments.receipt, result)
    except (IntegrationError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
