#!/usr/bin/env python3
"""Generate deterministic, auditable synthetic memory traces.

The configured working set is a set of request-sized logical pages.  It is
independent of ``address_span_bytes`` and its placement is explicit.  Patterns
select an access order over that fixed working set; they do not silently
change its placement or size.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


UINT64_MAX = (1 << 64) - 1
PATTERNS = ("sequential", "random-permutation", "modular-stride", "hotspot")
PLACEMENTS = ("linear", "stratified-random")
KINDS = (
    "unknown",
    "model_weights",
    "shared_context",
    "generated_context",
    "scratch",
    "metadata",
)
MASK64 = UINT64_MAX


@dataclass(frozen=True)
class GeneratorConfig:
    """Complete synthetic workload contract used by the CLI and runners."""

    pattern: str
    pages: int
    operations: int | None = None
    passes: int | None = None
    request_bytes: int = 4096
    base_address: int = 0
    address_span_bytes: int | None = None
    read_percent: Decimal | int | str = Decimal(100)
    seed: int = 1
    kind: str = "unknown"
    at_ns: float | int | None = 0
    stride_pages: int | None = None
    hot_page_percent: Decimal | int | str = Decimal(20)
    hot_access_percent: Decimal | int | str = Decimal(80)
    phase_comments: bool = False
    layer_per_pass: bool = False
    placement: str = "linear"
    placement_seed: int = 1
    # Optional residue-balancing contract for stratified placement.  Every
    # consecutive placement_modulus strata receive every slot residue modulo
    # this value exactly once, in a seeded random permutation.  Equal stratum
    # widths must be integer multiples of this value.
    placement_modulus: int | None = None
    # Optional exact page weights over equal-sized contiguous address regions.
    # This preserves one common trace across nested capacity boundaries while
    # making every routed traffic share explicit in the manifest.  It changes
    # placement only; access order remains controlled solely by ``seed``.
    placement_region_weights: tuple[int, ...] | None = None


def _decimal(value: Decimal | int | str, name: str) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a finite decimal percentage") from None
    if not result.is_finite():
        raise ValueError(f"{name} must be a finite decimal percentage")
    return result


def _exact_share(total: int, percent: Decimal | int | str, name: str) -> int:
    value = _decimal(percent, name)
    if value < 0 or value > 100:
        raise ValueError(f"{name} must be in [0, 100]")
    numerator, denominator = value.as_integer_ratio()
    scaled = total * numerator
    divisor = denominator * 100
    if scaled % divisor:
        raise ValueError(
            f"{name}={value} does not select an exact integer count from {total}")
    return scaled // divisor


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


class _SplitMix64Stream:
    """Small version-stable PRNG used only for deterministic trace shuffling."""

    def __init__(self, seed: int) -> None:
        self._state = seed & MASK64

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & MASK64
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    def below(self, bound: int) -> int:
        """Uniformly sample [0, bound) without modulo bias."""
        if bound <= 0:
            raise ValueError("random bound must be positive")
        threshold = (1 << 64) % bound
        while True:
            value = self.next_u64()
            if value >= threshold:
                return value % bound


def _fisher_yates(size: int, seed: int, salt: int, cycle: int) -> list[int]:
    """Return a deterministic Fisher-Yates permutation for one epoch."""
    order = list(range(size))
    entropy = seed ^ salt ^ _splitmix64(cycle)
    random = _SplitMix64Stream(entropy)
    for upper in range(size - 1, 0, -1):
        other = random.below(upper + 1)
        order[upper], order[other] = order[other], order[upper]
    return order


def _stratified_random_slots(
    pages: int,
    span_slots: int,
    seed: int,
    modulus: int | None = None,
) -> list[int]:
    """Choose one randomized slot from each non-overlapping address stratum.

    Integer boundaries partition the complete span without gaps.  A separate
    SplitMix64 stream is derived for every logical page, so changing one
    stratum's width cannot shift the random stream used by another stratum.
    With ``modulus``, every consecutive modulus-sized group of strata receives
    every slot residue exactly once in a seeded Fisher-Yates permutation.  The
    validator requires each stratum width to be a multiple of the modulus, so
    the choice among same-residue candidates remains unbiased over all slots.
    """
    slots: list[int] = []
    salt = 0xD1B54A32D192ED03
    residues = (
        _PermutationCycles(modulus, seed, 0xDB4F0B9175AE2165)
        if modulus is not None else None)
    for page in range(pages):
        begin = page * span_slots // pages
        end = (page + 1) * span_slots // pages
        random = _SplitMix64Stream(seed ^ salt ^ _splitmix64(page))
        if residues is None:
            slots.append(begin + random.below(end - begin))
            continue
        residue = residues.at(page)
        first = begin + ((residue - begin) % modulus)
        if first >= end:
            raise ValueError(
                f"stratum {page} contains no slot congruent to {residue} "
                f"modulo {modulus}")
        candidates = 1 + (end - 1 - first) // modulus
        slots.append(first + random.below(candidates) * modulus)
    return slots


def _weighted_stratified_random_slots(
    pages: int,
    span_slots: int,
    seed: int,
    weights: tuple[int, ...],
) -> tuple[list[int], list[int]]:
    """Place exact page shares into equal-sized contiguous address regions."""
    total_weight = sum(weights)
    region_width = span_slots // len(weights)
    slots: list[int] = []
    page_counts: list[int] = []
    for region, weight in enumerate(weights):
        count = pages * weight // total_weight
        page_counts.append(count)
        region_seed = seed ^ _splitmix64(
            0xA0761D6478BD642F ^ region)
        slots.extend(
            region * region_width + slot
            for slot in _stratified_random_slots(
                count, region_width, region_seed))
    if len(slots) != pages:
        raise ValueError("placement region weights did not conserve pages")
    return slots, page_counts


class _PermutationCycles:
    """Lazily retain only the currently consumed Fisher-Yates epoch."""

    def __init__(self, size: int, seed: int, salt: int) -> None:
        if size <= 0:
            raise ValueError("permutation size must be positive")
        self._size = size
        self._seed = seed
        self._salt = salt
        self._cycle = -1
        self._order: list[int] = []

    def at(self, ordinal: int) -> int:
        cycle, position = divmod(ordinal, self._size)
        if cycle != self._cycle:
            self._order = _fisher_yates(
                self._size, self._seed, self._salt, cycle)
            self._cycle = cycle
        return self._order[position]


def _validate(config: GeneratorConfig) -> tuple[int, int, int, int, int]:
    for name, value in (("pages", config.pages),
                        ("request_bytes", config.request_bytes),
                        ("base_address", config.base_address),
                        ("seed", config.seed),
                        ("placement_seed", config.placement_seed)):
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
    if config.pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {', '.join(PATTERNS)}")
    if config.placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {', '.join(PLACEMENTS)}")
    if config.pages <= 0:
        raise ValueError("pages must be positive")
    if config.request_bytes <= 0:
        raise ValueError("request_bytes must be positive")
    if config.base_address < 0 or config.base_address > UINT64_MAX:
        raise ValueError("base_address must fit uint64_t")
    if config.base_address % config.request_bytes:
        raise ValueError("base_address must be aligned to request_bytes")
    if config.seed < 0 or config.seed > UINT64_MAX:
        raise ValueError("seed must fit uint64_t")
    if config.placement_seed < 0 or config.placement_seed > UINT64_MAX:
        raise ValueError("placement_seed must fit uint64_t")
    if config.placement_modulus is not None:
        if (type(config.placement_modulus) is not int or
                config.placement_modulus <= 0):
            raise ValueError("placement_modulus must be a positive integer")
        if config.placement != "stratified-random":
            raise ValueError(
                "placement_modulus is only valid for stratified-random")
        if config.pages % config.placement_modulus:
            raise ValueError(
                "pages must be divisible by placement_modulus")
    if config.placement_region_weights is not None:
        weights = config.placement_region_weights
        if (not isinstance(weights, tuple) or len(weights) < 2 or
                any(type(weight) is not int or weight <= 0
                    for weight in weights)):
            raise ValueError(
                "placement_region_weights must contain at least two "
                "positive integers")
        if config.placement != "stratified-random":
            raise ValueError(
                "placement_region_weights is only valid for "
                "stratified-random")
        if config.placement_modulus is not None:
            raise ValueError(
                "placement_region_weights and placement_modulus cannot "
                "be combined")
        total_weight = sum(weights)
        if any(config.pages * weight % total_weight for weight in weights):
            raise ValueError(
                "placement_region_weights must select exact integer page "
                "counts")
    if config.kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")

    if (config.operations is None) == (config.passes is None):
        raise ValueError("set exactly one of operations or passes")
    if config.operations is not None:
        if type(config.operations) is not int or config.operations <= 0:
            raise ValueError("operations must be a positive integer")
        operations = config.operations
    else:
        if type(config.passes) is not int or config.passes <= 0:
            raise ValueError("passes must be a positive integer")
        operations = config.pages * config.passes
    if operations > UINT64_MAX:
        raise ValueError("operation count overflows uint64_t")
    if operations > UINT64_MAX // config.request_bytes:
        raise ValueError("total request bytes overflow uint64_t")

    minimum_span = config.pages * config.request_bytes
    if minimum_span > UINT64_MAX:
        raise ValueError("working-set byte size overflows uint64_t")
    span = minimum_span if config.address_span_bytes is None else config.address_span_bytes
    if type(span) is not int or span <= 0:
        raise ValueError("address_span_bytes must be a positive integer")
    if span % config.request_bytes:
        raise ValueError("address_span_bytes must be aligned to request_bytes")
    if span < minimum_span:
        raise ValueError("address_span_bytes cannot be smaller than the working set")
    if span - 1 > UINT64_MAX - config.base_address:
        raise ValueError("base_address + address_span_bytes overflows uint64_t")
    span_slots = span // config.request_bytes
    if (config.placement_region_weights is not None and
            span_slots % len(config.placement_region_weights)):
        raise ValueError(
            "address span slots must be divisible by the placement region "
            "count")
    if config.placement_modulus is not None:
        modulus = config.placement_modulus
        if span_slots % config.pages:
            raise ValueError(
                "address span slots must be divisible by pages when "
                "placement_modulus is set")
        stratum_width = span_slots // config.pages
        if stratum_width < modulus:
            raise ValueError(
                "every address stratum must contain at least "
                "placement_modulus slots")
        if stratum_width % modulus:
            raise ValueError(
                "every address stratum width must be divisible by "
                "placement_modulus for unbiased slot selection")

    read_ops = _exact_share(operations, config.read_percent, "read_percent")
    if config.at_ns is not None:
        try:
            at_ns = float(config.at_ns)
        except (TypeError, ValueError):
            raise ValueError("at_ns must be a finite non-negative number or None") from None
        if not math.isfinite(at_ns) or at_ns < 0:
            raise ValueError("at_ns must be a finite non-negative number or None")

    if config.pattern == "modular-stride":
        if type(config.stride_pages) is not int or config.stride_pages <= 0:
            raise ValueError("modular-stride requires positive stride_pages")
        if math.gcd(config.stride_pages, config.pages) != 1:
            raise ValueError("stride_pages must be coprime with pages")
    elif config.stride_pages is not None:
        raise ValueError("stride_pages is only valid for modular-stride")

    hot_pages = 0
    hot_ops = 0
    if config.pattern == "hotspot":
        hot_pages = _exact_share(config.pages, config.hot_page_percent,
                                 "hot_page_percent")
        hot_ops = _exact_share(operations, config.hot_access_percent,
                               "hot_access_percent")
        if hot_pages == 0 and hot_ops != 0:
            raise ValueError("hot accesses require at least one hot page")
        if hot_pages == config.pages and hot_ops != operations:
            raise ValueError("cold accesses require at least one cold page")
    return operations, span, span_slots, read_ops, hot_pages


def _selected(index: int, selected: int, total: int) -> bool:
    """Evenly distribute an exact selected count over ``total`` positions."""
    return ((index + 1) * selected) // total > (index * selected) // total


def _format_number(value: float | int) -> str:
    number = float(value)
    if number == 0.0:
        return "0"
    return format(number, ".17g")


def generate(config: GeneratorConfig) -> tuple[list[str], dict[str, Any]]:
    """Return trace lines (without terminators) and their exact manifest."""
    operations, span, span_slots, read_ops, hot_pages = _validate(config)
    hot_ops = (_exact_share(operations, config.hot_access_percent,
                            "hot_access_percent")
               if config.pattern == "hotspot" else 0)
    random_pages = _PermutationCycles(
        config.pages, config.seed, 0xE7037ED1A0B428DB)
    hotspot_membership = (_PermutationCycles(
        config.pages, config.seed, 0x589965CC75374CC3)
        if config.pattern == "hotspot" else None)
    hotspot_hot_order = (_PermutationCycles(
        hot_pages, config.seed, 0x8EBC6AF09C88C6E3)
        if config.pattern == "hotspot" and hot_pages > 0 else None)
    cold_pages = config.pages - hot_pages
    hotspot_cold_order = (_PermutationCycles(
        cold_pages, config.seed, 0x1D8E4E27C47D124F)
        if config.pattern == "hotspot" and cold_pages > 0 else None)
    placement_region_page_counts: list[int] | None = None
    if config.placement_region_weights is not None:
        stratified_slots, placement_region_page_counts = (
            _weighted_stratified_random_slots(
                config.pages,
                span_slots,
                config.placement_seed,
                config.placement_region_weights,
            ))
    else:
        stratified_slots = (_stratified_random_slots(
            config.pages, span_slots, config.placement_seed,
            config.placement_modulus)
            if config.placement == "stratified-random" else None)

    def physical_address(logical_page: int) -> int:
        if stratified_slots is None:
            # floor() gives distinct, evenly spaced slots because span_slots
            # is at least pages.  This regular mapping is intentional for
            # sequential and modular-stride experiments.
            slot = logical_page * span_slots // config.pages
        else:
            slot = stratified_slots[logical_page]
        return config.base_address + slot * config.request_bytes

    def logical_page(index: int, hot_ordinal: int, cold_ordinal: int) -> int:
        if config.pattern == "sequential":
            return index % config.pages
        if config.pattern == "random-permutation":
            return random_pages.at(index)
        if config.pattern == "modular-stride":
            assert config.stride_pages is not None
            return ((config.seed % config.pages) +
                    index * config.stride_pages) % config.pages
        if _selected(index, hot_ops, operations):
            assert hotspot_hot_order is not None
            assert hotspot_membership is not None
            rank = hotspot_hot_order.at(hot_ordinal)
            # Randomize the hot/cold page membership, not merely access order.
            return hotspot_membership.at(rank)
        assert hotspot_cold_order is not None
        assert hotspot_membership is not None
        rank = hot_pages + hotspot_cold_order.at(cold_ordinal)
        return hotspot_membership.at(rank)

    lines = [
        (f"# hbfsim synthetic pattern={config.pattern} ops={operations} "
         f"pages={config.pages} request_bytes={config.request_bytes} "
         f"address_span_bytes={span} seed={config.seed} "
         f"placement={config.placement} "
         f"placement_seed={config.placement_seed} "
         f"placement_modulus={config.placement_modulus} "
         f"placement_region_weights={config.placement_region_weights}")
    ]
    unique_pages: set[int] = set()
    hot_ordinal = 0
    cold_ordinal = 0
    previous_phase = -1
    for index in range(operations):
        phase = index // config.pages
        if config.phase_comments and phase != previous_phase:
            lines.append(f"# phase={phase} begin_op={index}")
            previous_phase = phase
        is_hot = config.pattern == "hotspot" and _selected(index, hot_ops, operations)
        page = logical_page(index, hot_ordinal, cold_ordinal)
        if is_hot:
            hot_ordinal += 1
        elif config.pattern == "hotspot":
            cold_ordinal += 1
        unique_pages.add(page)
        op = "R" if _selected(index, read_ops, operations) else "W"
        suffix = f" {config.kind}" if config.kind != "unknown" else ""
        if config.at_ns is not None:
            suffix += f" at={_format_number(config.at_ns)}"
        if config.layer_per_pass:
            suffix += f" layer={phase}"
        lines.append(
            f"0x{physical_address(page):x} {op} {config.request_bytes}{suffix}")

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    write_ops = operations - read_ops
    manifest: dict[str, Any] = {
        "schema": {"name": "hbfsim.synthetic_trace.manifest", "version": 3},
        "pattern": config.pattern,
        "seed": config.seed,
        "placement": config.placement,
        "placement_seed": config.placement_seed,
        "placement_modulus": config.placement_modulus,
        "placement_region_weights": config.placement_region_weights,
        "placement_region_page_counts": placement_region_page_counts,
        "layer_per_pass": config.layer_per_pass,
        "ops": operations,
        "read_ops": read_ops,
        "write_ops": write_ops,
        "request_bytes": config.request_bytes,
        "bytes": operations * config.request_bytes,
        "read_bytes": read_ops * config.request_bytes,
        "write_bytes": write_ops * config.request_bytes,
        "working_set_pages": config.pages,
        "unique_pages": len(unique_pages),
        "base_address": config.base_address,
        "address_span_bytes": span,
        "read_percent": str(_decimal(config.read_percent, "read_percent")),
        "kind": config.kind,
        "at_ns": config.at_ns,
        "passes": config.passes,
        "stride_pages": config.stride_pages,
        "hot_page_percent": (
            str(_decimal(config.hot_page_percent, "hot_page_percent"))
            if config.pattern == "hotspot" else None),
        "hot_access_percent": (
            str(_decimal(config.hot_access_percent, "hot_access_percent"))
            if config.pattern == "hotspot" else None),
        "hot_pages": hot_pages if config.pattern == "hotspot" else None,
        "hot_ops": hot_ops if config.pattern == "hotspot" else None,
        "trace_digest": {
            "algorithm": "sha256",
            "value": hashlib.sha256(payload).hexdigest(),
        },
        "trace_file_bytes": len(payload),
    }
    return lines, manifest


def _write_temp(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path


def publish(output: Path, manifest_path: Path, lines: list[str],
            manifest: dict[str, Any]) -> None:
    """Publish a trace and its digest-binding manifest, manifest last."""
    output = output.resolve()
    manifest_path = manifest_path.resolve()
    if output == manifest_path:
        raise ValueError("trace and manifest paths must be different")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    expected = manifest.get("trace_digest", {}).get("value")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("manifest digest does not match generated trace")
    published_manifest = dict(manifest)
    published_manifest["trace_path"] = str(output)
    manifest_data = (json.dumps(published_manifest, indent=2, sort_keys=True) +
                     "\n").encode("utf-8")
    trace_temp: Path | None = None
    manifest_temp: Path | None = None
    try:
        trace_temp = _write_temp(output, payload)
        manifest_temp = _write_temp(manifest_path, manifest_data)
        os.replace(trace_temp, output)
        trace_temp = None
        # The manifest is the commit record and must always be published last.
        os.replace(manifest_temp, manifest_path)
        manifest_temp = None
    finally:
        if trace_temp is not None:
            trace_temp.unlink(missing_ok=True)
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)


def _percentage(text: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation:
        raise argparse.ArgumentTypeError("must be a decimal percentage") from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pattern", choices=PATTERNS, required=True)
    parser.add_argument("--pages", type=int, required=True)
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--ops", type=int)
    count.add_argument("--passes", type=int)
    parser.add_argument("--bytes", dest="request_bytes", type=int, default=4096)
    parser.add_argument("--base", dest="base_address",
                        type=lambda value: int(value, 0), default=0)
    parser.add_argument("--address-span", dest="address_span_bytes",
                        type=lambda value: int(value, 0))
    parser.add_argument(
        "--placement", choices=PLACEMENTS, default="linear",
        help="map logical pages regularly or randomly within uniform strata")
    parser.add_argument("--placement-seed",
                        type=lambda value: int(value, 0), default=1,
                        help="independent seed for stratified-random placement")
    parser.add_argument(
        "--placement-modulus",
        type=lambda value: int(value, 0),
        help=("balance slot residues in seeded permutations; requires "
              "stratified-random placement and equal, modulus-divisible "
              "strata"),
    )
    parser.add_argument(
        "--placement-region-weights",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        help=("comma-separated positive integer page weights over equal-sized "
              "contiguous address regions; requires stratified-random"),
    )
    parser.add_argument("--read-percent", type=_percentage, default=Decimal(100))
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=1)
    parser.add_argument("--kind", choices=KINDS, default="unknown")
    parser.add_argument("--at", dest="at_ns", type=float, default=0)
    parser.add_argument("--stride-pages", type=int)
    parser.add_argument("--hot-page-percent", type=_percentage, default=Decimal(20))
    parser.add_argument("--hot-access-percent", type=_percentage, default=Decimal(80))
    parser.add_argument("--phase-comments", action="store_true")
    parser.add_argument(
        "--layer-per-pass", action="store_true",
        help="append layer=N, using each working-set pass as one layer")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = GeneratorConfig(
        pattern=args.pattern,
        pages=args.pages,
        operations=args.ops,
        passes=args.passes if args.ops is None else None,
        request_bytes=args.request_bytes,
        base_address=args.base_address,
        address_span_bytes=args.address_span_bytes,
        placement=args.placement,
        placement_seed=args.placement_seed,
        placement_modulus=args.placement_modulus,
        placement_region_weights=args.placement_region_weights,
        read_percent=args.read_percent,
        seed=args.seed,
        kind=args.kind,
        at_ns=args.at_ns,
        stride_pages=args.stride_pages,
        hot_page_percent=args.hot_page_percent,
        hot_access_percent=args.hot_access_percent,
        phase_comments=args.phase_comments,
        layer_per_pass=args.layer_per_pass,
    )
    if config.operations is None and config.passes is None:
        config = replace(config, passes=1)
    try:
        lines, manifest = generate(config)
        manifest_path = args.manifest or Path(str(args.output) + ".manifest.json")
        publish(args.output, manifest_path, lines, manifest)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
