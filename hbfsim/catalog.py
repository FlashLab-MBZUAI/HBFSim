"""Discover the configuration profiles shipped in ``configs/`` by short name.

A *system* is a complete hardware profile, an *overlay* is a one-purpose
override applied after it, and a *policy* holds settings for the reference
runner's placement policies. Short names are paths relative to the kind's
directory without ``.cfg`` (``miniquick/4hbm-4hbf``, ``hbf/ocp-v070-grade3``);
an unambiguous file stem (``ocp-v070-grade3``) or any ``.cfg`` path also works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
from pathlib import Path
import re
from typing import Iterable, Mapping

from hbfsim.workspace import HbfsimError, repository_root


KIND_DIRECTORIES = {
    "system": "systems",
    "overlay": "overlays",
    "policy": "policies",
}

# The reference runner's policy suite, in the order its --help lists them.
# tests/python/test_hbfsim_frontdoor.py keeps this in step with the C++ source.
SCENARIOS: dict[str, str] = {
    "all-hbm": "Every request is served by HBM (the fast upper bound).",
    "all-hbf": "Every request goes through the logical HBF path: FTL, mapping, write buffer, GC.",
    "flat": "One address space split at flat-hbm-bytes: low addresses in HBM, the rest in HBF.",
    "direct-read": "Read-mostly data is read from static physical HBF; mutable data stays in HBM.",
    "demand-fill": "Behavior-only baseline: every observed page is admitted to an HBM tier.",
    "reuse-filtered": "Behavior-only policy: a page is promoted to HBM only after observed reuse.",
    "hbf-streaming": "Layer weights are double-buffered from HBF into HBM.",
    "external-streaming": "Layer weights are double-buffered from the external backing device into HBM.",
}
# Scenarios that are meaningful on every shipped system profile and the demo trace.
CORE_SCENARIOS = ("all-hbm", "all-hbf", "flat", "direct-read", "hbf-streaming")


@dataclass(frozen=True)
class Profile:
    """One ``.cfg`` file under ``configs/``."""

    kind: str
    name: str
    path: Path
    description: str
    values: Mapping[str, str] = field(repr=False)


def parse_config(path: Path) -> dict[str, str]:
    """Parse ``key=value`` records, ignoring ``#`` comments (later keys win)."""

    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise HbfsimError(f"{path}:{number}: expected key=value, got {raw!r}")
        values[key.strip()] = value.strip()
    return values


def _leading_comment(path: Path) -> str:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped.startswith("#"):
            break
        text = stripped.lstrip("#").strip()
        if not text:
            if lines:
                break
            continue
        lines.append(text)
    paragraph = " ".join(lines)
    sentence = re.split(r"(?<=[.;])\s", paragraph, maxsplit=1)[0]
    return sentence.rstrip(".;") if sentence else ""


def _gib(value: float) -> str:
    gib = value / 2**30
    if gib >= 1024:
        return f"{gib / 1024:.3g} TiB"
    if gib >= 1:
        return f"{gib:.4g} GiB"
    return f"{value / 2**20:.4g} MiB"


def hbf_raw_capacity_bytes(values: Mapping[str, str]) -> int | None:
    """Raw HBF bytes declared by a profile, or ``None`` if not self-contained."""

    try:
        if "hbf-capacity-bytes" in values:
            return int(values["hbf-capacity-bytes"])
        if "hbf-capacity-ratio" in values and "hbm-capacity-bytes" in values:
            return int(float(values["hbf-capacity-ratio"]) * int(values["hbm-capacity-bytes"]))
        keys = ("hbf-stacks", "hbf-channels", "hbf-dies-per-channel", "hbf-planes-per-die",
                "hbf-blocks-per-plane", "hbf-pages-per-block", "hbf-page-size")
        if all(key in values for key in keys):
            product = 1
            for key in keys:
                product *= int(values[key])
            return product
    except ValueError:
        return None
    return None


def describe_system(values: Mapping[str, str]) -> str:
    """One-line topology summary such as ``4x HBM (128 GiB) + 4x HBF G2 (512 GiB)``."""

    parts = []
    if "hbm-stacks" in values:
        hbm = f"{values['hbm-stacks']}x HBM"
        if "hbm-capacity-bytes" in values:
            hbm += f" ({_gib(int(values['hbm-capacity-bytes']))})"
        parts.append(hbm)
    if "hbf-stacks" in values:
        hbf = f"{values['hbf-stacks']}x HBF"
        if "hbf-speed-grade" in values:
            hbf += f" grade {values['hbf-speed-grade']}"
        capacity = hbf_raw_capacity_bytes(values)
        if capacity:
            hbf += f" ({_gib(capacity)} raw)"
        parts.append(hbf)
    return " + ".join(parts)


def _shorten(text: str, limit: int = 96) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _profiles(kind: str) -> list[Profile]:
    directory = repository_root() / "configs" / KIND_DIRECTORIES[kind]
    profiles = []
    for path in directory.rglob("*.cfg"):
        name = path.relative_to(directory).with_suffix("").as_posix()
        values = parse_config(path)
        description = _leading_comment(path)
        if kind == "system":
            topology = describe_system(values)
            if name.startswith("miniquick/"):
                topology += ", capacity-scaled"
            description = topology or description
            if default_policy(path) is None and not name.startswith("miniquick/"):
                description += " (frontend fragment, no reference policy)"
        elif len(description) < 40 and values:
            # Terse headers (e.g. the speed-grade overlays) read better with their keys.
            keys = ", ".join(f"{key}={value}" for key, value in list(values.items())[:3])
            description = f"{description} ({keys})" if description else keys
        profiles.append(Profile(kind, name, path, _shorten(description), values))
    return sorted(profiles, key=lambda profile: profile.name)


def profiles(kind: str) -> list[Profile]:
    """Every shipped profile of one kind: ``system``, ``overlay`` or ``policy``."""

    if kind not in KIND_DIRECTORIES:
        raise HbfsimError(f"unknown profile kind {kind!r}; use one of {', '.join(KIND_DIRECTORIES)}")
    return _profiles(kind)


def resolve(kind: str, name: str | Path) -> Path:
    """Resolve a short name, unique stem, or path to an existing ``.cfg`` file."""

    candidate = Path(name).expanduser()
    if candidate.suffix == ".cfg" or candidate.is_file():
        if candidate.is_file():
            return candidate.resolve()
        rooted = repository_root() / candidate
        if rooted.is_file():
            return rooted.resolve()
        raise HbfsimError(f"{kind} config not found: {name}")
    text = str(name).strip().removesuffix(".cfg")
    available = profiles(kind)
    for profile in available:
        if profile.name == text:
            return profile.path
    # A unique file stem or path suffix is enough: ocp-v070-grade3, mapping/block.
    matches = [profile for profile in available
               if profile.path.stem == text or profile.name.endswith("/" + text)]
    if len(matches) == 1:
        return matches[0].path
    names = [profile.name for profile in available]
    if len(matches) > 1:
        raise HbfsimError(
            f"{kind} name {text!r} is ambiguous; use one of: "
            + ", ".join(profile.name for profile in matches)
        )
    close = difflib.get_close_matches(text, names, n=3)
    close += [p.name for p in available
              if p.path.stem in difflib.get_close_matches(text, [p.path.stem], n=1)]
    close += [name for name in names if text.lower() in name.lower()]
    close = list(dict.fromkeys(close))[:3]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise HbfsimError(
        f"unknown {kind} {text!r}.{hint} List them with `python3 -m hbfsim list {kind}s`."
    )


def default_policy(system: Path) -> Path | None:
    """The reference-policy profile paired with a shipped system, if any.

    Shipped pairs share a file name (``systems/X.cfg`` and
    ``policies/reference/X.cfg``). The pairing is a front-door convenience; the
    simulator itself never infers a policy from a file name.
    """

    root = repository_root() / "configs"
    try:
        relative = system.resolve().relative_to((root / "systems").resolve())
    except ValueError:
        return None
    candidate = root / "policies" / "reference" / relative
    return candidate.resolve() if candidate.is_file() else None


def scenario_list(selection: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize a scenario selection: ``None``/``core``, ``all``, or names."""

    if selection is None:
        return CORE_SCENARIOS
    items = [selection] if isinstance(selection, str) else list(selection)
    names: list[str] = []
    for item in items:
        for token in str(item).split(","):
            token = token.strip()
            if not token:
                continue
            if token == "core":
                names.extend(CORE_SCENARIOS)
            elif token == "all":
                names.extend(SCENARIOS)
            elif token in SCENARIOS:
                names.append(token)
            else:
                close = difflib.get_close_matches(token, list(SCENARIOS), n=2)
                hint = f" Did you mean: {', '.join(close)}?" if close else ""
                raise HbfsimError(
                    f"unknown scenario {token!r}.{hint} "
                    "List them with `python3 -m hbfsim list scenarios`."
                )
    if not names:
        raise HbfsimError("select at least one scenario")
    return tuple(dict.fromkeys(names))
