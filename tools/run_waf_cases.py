#!/usr/bin/env python3
"""Verify the canonical HBF write-amplification factor.

Self-contained: generates deterministic traces, runs scenario_compare, and
judges every case against truths that do not come from the model itself:

  identities (every case, exact):
      emitted data_programs == write_ops - coalesced_writes
      page_programs == data_programs + mapping_programs + gc_relocations
      physical_write_bytes == page_programs x page_size
      logical_write_bytes  == trace bytes
      reported waf
          == physical_write_bytes / logical_write_bytes
  sharp regimes (exact):
      seq-wrap  — sequential circular overwrite invalidates blocks in fill
                  order, so greedy victims are FULLY invalid:
                  gc_relocations MUST be 0 and WAF is approximately
                  1 + mapping share for full-page writes.
      kv-fifo   — 8 append streams with FIFO expiry (the write pattern
                  large-KV serving actually produces): streams advance at
                  one global rate, blocks expire in fill order, relocations
                  are a jitter tail.
      subpage   — 64 B writes to distinct mapped pages: one RMW program
                  each. WAF is 64 x (1 + mapping/N), proving that its
                  denominator remains logical workload bytes.
  structural/random-churn guards:
      rand-churn at u_eff 58% / 90% — uniform random overwrite, the
      adversarial accounting stir (NOT a workload claim). Both cases must
      execute GC with non-zero live-page relocation, remain below a
      conservative FIFO-derived ceiling, and increase both WAF and average
      victim valid fraction as u_eff rises.

FIFO fluid bound (first-principles, so the band is independent truth):
with footprint U pages uniformly overwritten over a cycling physical pool
of P pages cleaned in fill order, a block cleans after one pool turnover
= P*(1-v) logical writes, during which a page survives with probability
exp(-P*(1-v)/U); so with rho = U/P the victim valid fraction solves
    v = exp(-(1 - v) / rho)
    waf_fifo = 1 / (1 - v).
Greedy victim selection dominates FIFO.  The fluid result is therefore used
only as a conservative finite-size sanity ceiling (fixed point x 1.15), not
as a finite-run expected value.  In particular, it cannot justify a lower
WAF bound for a finite deterministic run: greedy selection can find victims
far below the FIFO-average valid fraction.  The lower structural floor is
only WAF >= 1, while non-zero relocation and the cross-occupancy relations
prove that the intended GC path was exercised.

Geometry discipline: greedy GC needs a real block population to choose
from — starved geometries (10-20 blocks/plane near full) degenerate to
WAF near pages_per_block and verify nothing. This suite
runs 16 planes x
64 blocks x 256 pages x 4 KiB (1 GiB). Per plane, 1 free block is GC
reserve and 1 is the mapping-role active, so the DATA POOL is 62
blocks/plane = 253,952 pages and u_eff = footprint / data pool.
Footprints are whole blocks per plane. Controller DRAM uses the exact 2 MiB
required by the complete resident L2P table (512 mapping pages). Foreground
translation never reads a mapping page from flash; dirty checkpoint pages are
programmed only when the run drains.

Every trace explicitly declares its initial logical image with one read per
footprint page before churn. Initial state therefore comes from workload
data, never from future-write lookahead; the verification identities below
continue to count only the write phase. Runs use a one-request closed-loop
window and require each buffered write to flush before acknowledgement, so
overwrite completion makes invalid pages visible before subsequent
allocation/GC decisions. This suite verifies steady-state media accounting,
not controller queue-depth or burst-absorption behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from plot_time_breakdown import write_visualization
from time_breakdown_report import SummaryInput, write_time_breakdown_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.certificate import (  # noqa: E402
    EXPLORATORY_VALIDATION,
    CertificateError,
    VerifiedCertificate,
    attach_certificate_to_summary_file,
    ensure_exploratory_summary,
    verify_certificate,
)
from validation.contracts import load_json_strict  # noqa: E402

PAGE = 4096
OOB = 224
PLANES = 16
BLOCKS_PER_PLANE = 64
PAGES_PER_BLOCK = 256
DATA_POOL_PAGES = PLANES * (BLOCKS_PER_PLANE - 2) * PAGES_PER_BLOCK  # 253,952
TOTAL_PAGES = PLANES * BLOCKS_PER_PLANE * PAGES_PER_BLOCK
MAPPING_ENTRIES_PER_PAGE = 512
GEOMETRY = (
    "--hbf-stacks", "1",
    "--hbf-channels", "2",
    "--hbf-dies-per-channel", "2",
    "--hbf-planes-per-die", "4",
    "--hbf-blocks-per-plane", str(BLOCKS_PER_PLANE),
    "--hbf-pages-per-block", str(PAGES_PER_BLOCK),
    "--hbf-page-size", str(PAGE),
    "--hbf-oob-bytes", str(OOB),
    "--hbf-ecc-decode-raw-bw", "135",
    "--hbf-ecc-encode-raw-bw", "135",
    "--hbf-channel-bw", "270",
    "--hbf-hbio-bw", "512",
    "--hbf-tsv-bw", "548",
    "--hbf-media-lanes-per-plane", "16",
    "--hbf-subarrays-per-plane", "32",
    "--hbf-page-buffer-banks-per-plane", "16",
    "--hbf-write-coalescing", "true",
    "--hbf-write-buffer-completion-requires-flush", "true",
    "--hbf-write-buffer-pages", "256",
    "--hbf-write-buffer-flush-threshold-pages", "128",
    "--hbf-gc-low-watermark-pages", "1024",
    "--hbf-gc-reserved-free-blocks-per-plane", "1",
    "--hbf-ctrl-dram-bytes", "0",
)
DEFAULT_OUT = Path("out/write-amplification-cases")
TIME_BREAKDOWN_CONFIG = "inline-waf-geometry"
WAF_CSV_FIELDS = (
    "case",
    "u_eff",
    "waf",
    "waf_window_lo",
    "waf_window_hi",
    "logical_write_bytes",
    "physical_write_bytes",
    "data_programs",
    "page_programs",
    "mapping_programs",
    "gc_relocations",
    "gc_reclaimed_invalid_pages",
    "gc_victim_valid_fraction",
    "coalesced_writes",
    "gc_runs",
    "block_erases",
    "free_pages",
    "accounting_verified",
    "verdict",
)


def fifo_waf_bound(rho: float) -> float:
    """Solve v = exp(-(1-v)/rho); return full-page WAF 1/(1-v)."""
    v = 0.5
    for _ in range(100000):
        nxt = math.exp(-(1.0 - v) / rho)
        if abs(nxt - v) < 1e-12:
            v = nxt
            break
        v = nxt
    return 1.0 / (1.0 - v)


def lcg(state: int) -> int:
    return (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)


@dataclass(frozen=True)
class AmplificationCase:
    key: str
    name: str
    footprint_blocks: int      # whole blocks per plane
    rounds: int                # ops = rounds x footprint
    waf_window: tuple[float, float]
    expect_zero_relocations: bool
    expect_gc_runs: bool
    expect_nonzero_relocations: bool = False
    write_bytes: int = PAGE
    note: str = ""

    @property
    def footprint_pages(self) -> int:
        return self.footprint_blocks * PLANES * PAGES_PER_BLOCK

    @property
    def ops(self) -> int:
        return self.rounds * self.footprint_pages

    @property
    def u_eff(self) -> float:
        return self.footprint_pages / DATA_POOL_PAGES


# Band caps from the fluid bound (computed at import; values noted):
#   rho = 36/62 = 0.581 -> program amp ~= 1.43 ; cap ~= 1.65
#   rho = 56/62 = 0.903 -> program amp ~= 5.34 ; cap ~= 6.14
CASES = (
    AmplificationCase("seq-wrap", "顺序循环覆盖", 54, 3, (1.00, 1.02),
            expect_zero_relocations=True, expect_gc_runs=True,
            note="失效跟随块填充序:victim 全无效,搬迁=0"),
    AmplificationCase("kv-fifo", "KV append+FIFO 过期", 54, 3, (1.00, 1.10),
            expect_zero_relocations=False, expect_gc_runs=True,
            note="大 KV 写的真实形态;流同速推进,近零搬迁"),
    AmplificationCase("rand-churn-58", "随机覆盖 u_eff=58%", 36, 5,
            (1.00, round(fifo_waf_bound(36 / 62) * 1.15, 2)),
            expect_zero_relocations=False, expect_gc_runs=True,
            expect_nonzero_relocations=True,
            note="记账搅拌;非零搬迁证明 GC 真被行使"),
    AmplificationCase("rand-churn-90", "随机覆盖 u_eff=90%", 56, 4,
            (1.00, round(fifo_waf_bound(56 / 62) * 1.15, 2)),
            expect_zero_relocations=False, expect_gc_runs=True,
            expect_nonzero_relocations=True,
            note="greedy<=FIFO 流体界 x1.15;必须单调高于 u_eff=58%"),
    AmplificationCase("subpage", "子页 64B RMW", 2, 1, (63.90, 64.60),
            expect_zero_relocations=True, expect_gc_runs=False,
            write_bytes=64,
            note="字节放大=64x(1+mapping/N);program放大=1+mapping/N"),
)



def generate_trace(path: Path, case: AmplificationCase) -> int:
    """Declare the initial image, then deterministic churn; return write bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = 20260707
    fp = case.footprint_pages
    # Read-first is the simulator's explicit initial-image declaration: these
    # pages exist before the measured overwrite phase. Do not infer an image
    # from addresses that only appear in future writes.
    lines = [f"0x{i * PAGE:x} R {PAGE}" for i in range(fp)]
    if case.key == "seq-wrap" or case.key == "subpage":
        for i in range(case.ops):
            lines.append(f"0x{(i % fp) * PAGE:x} W {case.write_bytes}")
    elif case.key == "kv-fifo":
        streams = 8
        per_stream = fp // streams
        cursors = [0] * streams
        for _ in range(case.ops):
            state = lcg(state)
            s = state % streams
            addr = (s * per_stream + cursors[s]) * PAGE
            cursors[s] = (cursors[s] + 1) % per_stream
            lines.append(f"0x{addr:x} W {case.write_bytes}")
    elif case.key.startswith("rand-churn"):
        for _ in range(case.ops):
            state = lcg(state)
            lines.append(f"0x{(state % fp) * PAGE:x} W {case.write_bytes}")
    else:
        raise ValueError(case.key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Write-amplification verification trace: {case.key}\n")
        handle.write("\n".join(lines))
        handle.write("\n")
    return case.ops * case.write_bytes


def run_case(
    binary: Path,
    case: AmplificationCase,
    out_dir: Path,
    validation_certificate: VerifiedCertificate | None = None,
) -> dict:
    trace = out_dir / f"{case.key}.trace"
    logical_bytes = generate_trace(trace, case)
    summary = (out_dir / f"{case.key}.summary.json").resolve()
    summary.unlink(missing_ok=True)
    cmd = [str(Path(binary).resolve()), "--trace", str(trace.resolve()),
           "--scenarios", "all-HBF", *GEOMETRY,
           "--max-outstanding-requests", "1",
           "--summary-json", str(summary)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        detail = (f"stderr:\n{proc.stderr[-2000:].strip()}\n"
                  f"stdout tail:\n{proc.stdout[-2000:].strip()}")
        raise RuntimeError(
            f"{case.key}: scenario_compare exited {proc.returncode}:\n{detail}")
    if not summary.exists():
        raise RuntimeError(
            f"{case.key}: scenario_compare succeeded without writing {summary}")

    if validation_certificate is None:
        data = load_json_strict(summary)
        ensure_exploratory_summary(data)
    else:
        data = attach_certificate_to_summary_file(
            summary, validation_certificate)
    if data.get("schema") != {
            "name": "hbfsim.scenario_compare.summary", "version": 16}:
        raise RuntimeError(f"{case.key}: unsupported or missing summary schema")
    if data.get("sanity") != "PASS":
        raise RuntimeError(f"{case.key}: summary reports SANITY={data.get('sanity')!r}")
    matches = [s for s in data.get("scenarios", []) if s.get("name") == "all-HBF"]
    if len(matches) != 1:
        raise RuntimeError(f"{case.key}: expected one all-HBF result, got {len(matches)}")
    scenario = matches[0]
    mapping_entries_per_page = data["config"]["hbf"]["mapping_entries_per_page"]
    if mapping_entries_per_page != MAPPING_ENTRIES_PER_PAGE:
        raise RuntimeError(
            f"{case.key}: mapping entries/page {mapping_entries_per_page} != "
            f"verification constant {MAPPING_ENTRIES_PER_PAGE}")
    return {
        "logical_bytes": logical_bytes,
        "stats": scenario["hbf_stats"],
        "mapping_entries_per_page": mapping_entries_per_page,
        "summary": data,
        "summary_path": summary,
    }


def verify_regime(
    case: AmplificationCase,
    waf: float | None,
    relocations: int,
    gc_runs: int,
) -> list[str]:
    """Check only independently stated per-case regime expectations."""
    failures: list[str] = []
    lo, hi = case.waf_window
    if waf is None or not lo <= waf <= hi:
        failures.append(f"WAF {waf} outside window [{lo}, {hi}]")
    if case.expect_zero_relocations and relocations != 0:
        failures.append(
            f"expected zero GC relocations, got {relocations}")
    if case.expect_nonzero_relocations and relocations == 0:
        failures.append("expected non-zero GC relocations")
    if case.expect_gc_runs and gc_runs == 0:
        failures.append("expected GC to run under capacity pressure")
    if not case.expect_gc_runs and gc_runs != 0:
        failures.append(f"expected no GC, got {gc_runs} runs")
    return failures


def verify_random_churn_relation(
    details: dict[str, dict],
) -> list[str]:
    """Metamorphic checks across the fixed low/high-occupancy churn cases."""
    low = details["rand-churn-58"]
    high = details["rand-churn-90"]
    failures: list[str] = []
    if high["waf"] <= low["waf"]:
        failures.append("WAF at u_eff=90% must exceed u_eff=58%")
    if (high["gc_victim_valid_fraction"] <=
            low["gc_victim_valid_fraction"]):
        failures.append(
            "average GC-victim valid fraction at u_eff=90% must exceed "
            "u_eff=58%")
    return failures


def verify(case: AmplificationCase, result: dict) -> tuple[list[str], dict]:
    stats = result["stats"]
    failures: list[str] = []
    programs = stats["page_programs"]
    data_programs = stats["data_programs"]
    mapping = stats["mapping_page_programs"]
    reloc = stats["gc_relocations"]
    coalesced = stats["write_buffer_hits"]
    # --- accounting identities ---
    if programs != data_programs + mapping + reloc:
        failures.append(
            f"page_programs {programs} != data {data_programs} + mapping "
            f"{mapping} + relocation {reloc}")
    if stats["physical_write_bytes"] != programs * PAGE:
        failures.append(
            f"physical_write_bytes {stats['physical_write_bytes']} != "
            f"programs x page {programs * PAGE}")
    expected_component_bytes = {
        "data_program_payload_bytes": data_programs * PAGE,
        "mapping_program_payload_bytes": mapping * PAGE,
        "gc_relocation_payload_bytes": reloc * PAGE,
    }
    for key, expected in expected_component_bytes.items():
        if stats.get(key) != expected:
            failures.append(
                f"{key} {stats.get(key)!r} != independently derived "
                f"{expected}")
    if sum(expected_component_bytes.values()) != stats["physical_write_bytes"]:
        failures.append(
            "data + mapping + GC payload bytes do not conserve physical writes")
    if data_programs != case.ops - coalesced:
        failures.append(
            f"data programs {data_programs} != ops {case.ops} - coalesced "
            f"{coalesced}")
    if stats["logical_write_bytes"] != result["logical_bytes"]:
        failures.append(
            f"logical_write_bytes {stats['logical_write_bytes']} != trace "
            f"bytes {result['logical_bytes']}")
    mapping_pages = math.ceil(
        case.footprint_pages / result["mapping_entries_per_page"])
    expected_page_reads = (
        case.footprint_pages + reloc +
        (case.ops if case.write_bytes < PAGE else 0))
    expected_free_pages = (
        TOTAL_PAGES - case.footprint_pages - mapping_pages - programs +
        stats["block_erases"] * PAGES_PER_BLOCK)
    expected_valid_pages = case.footprint_pages + mapping_pages
    expected_invalid_pages = (
        stats["invalidations"] -
        stats["gc_runs"] * PAGES_PER_BLOCK)
    if stats.get("accounting_verified") is not True:
        failures.append("runtime HBF accounting audit was not verified")
    if stats["total_pages"] != TOTAL_PAGES:
        failures.append(
            f"total_pages {stats['total_pages']} != geometry {TOTAL_PAGES}")
    if stats["mapping_entries"] != case.footprint_pages:
        failures.append(
            f"mapping_entries {stats['mapping_entries']} != footprint "
            f"{case.footprint_pages}")
    expected_mapping_lookups = case.footprint_pages + case.ops
    expected_mapping_updates = data_programs + stats["gc_data_relocations"]
    if stats["mapping_lookup_ops"] != expected_mapping_lookups:
        failures.append(
            f"resident mapping lookups {stats['mapping_lookup_ops']} != "
            f"initial reads + writes {expected_mapping_lookups}")
    if stats["mapping_user_lookup_ops"] != expected_mapping_lookups or \
            stats["mapping_gc_lookup_ops"] != 0:
        failures.append("resident mapping lookup sources do not conserve")
    if stats["mapping_update_ops"] != expected_mapping_updates:
        failures.append(
            f"resident mapping updates {stats['mapping_update_ops']} != "
            f"data programs + GC data relocations {expected_mapping_updates}")
    if (stats["mapping_user_update_ops"] != data_programs or
            stats["mapping_gc_update_ops"] !=
            stats["gc_data_relocations"]):
        failures.append("resident mapping update sources do not conserve")
    if mapping != mapping_pages:
        failures.append(f"mapping programs {mapping} != {mapping_pages}")
    if stats["page_reads"] != expected_page_reads:
        failures.append(
            f"page_reads {stats['page_reads']} != initial data "
            f"{case.footprint_pages} + GC {reloc} + "
            f"RMW {case.ops if case.write_bytes < PAGE else 0}")
    if stats["physical_read_bytes"] != stats["page_reads"] * PAGE:
        failures.append("physical_read_bytes does not equal page_reads x page_size")
    if stats["block_erases"] != stats["gc_runs"]:
        failures.append("block_erases must equal gc_runs when the trace has no raw erase")
    if (stats["gc_data_relocations"] + stats["gc_mapping_relocations"] !=
            reloc):
        failures.append("GC data + mapping relocation counts do not conserve total")
    if (reloc + stats["gc_reclaimed_invalid_pages"] !=
            stats["gc_runs"] * PAGES_PER_BLOCK):
        failures.append("GC relocated + reclaimed pages do not fill every victim")
    if (stats["gc_runs"] and
            stats["gc_reclaimed_invalid_pages"] < stats["gc_runs"]):
        failures.append("every GC victim must reclaim at least one invalid page")
    if stats["free_pages"] != expected_free_pages:
        failures.append(
            f"free_pages {stats['free_pages']} != independent capacity "
            f"accounting {expected_free_pages}")
    if stats["valid_pages"] != expected_valid_pages:
        failures.append(
            f"valid_pages {stats['valid_pages']} != live data+mapping "
            f"{expected_valid_pages}")
    if expected_invalid_pages < 0 or stats["invalid_pages"] != expected_invalid_pages:
        failures.append(
            f"invalid_pages {stats['invalid_pages']} != invalidations minus "
            f"erased GC victims {expected_invalid_pages}")
    if stats["pending_program_pages"] != 0:
        failures.append("drained run retained pending program pages")
    if stats["pending_mapping_publications"] != 0:
        failures.append("drained run retained pending mapping publications")
    if stats["static_unmaterialized_pages"] != 0:
        failures.append("WAF-only run unexpectedly retained static reservations")
    if (stats["free_pages"] + stats["valid_pages"] +
            stats["invalid_pages"] != TOTAL_PAGES):
        failures.append("free + valid + invalid pages do not conserve capacity")
    expected_flash_transactions = (
        stats["page_reads"] + programs + stats["block_erases"])
    if (stats["flash_scheduler_enqueues"] != expected_flash_transactions or
            stats["flash_scheduler_issues"] != expected_flash_transactions):
        failures.append("flash scheduler counts do not conserve media operations")
    if (stats["ecc_decode_ops"] != stats["page_reads"] or
            stats["ecc_encode_ops"] != programs):
        failures.append("ECC operation counts do not conserve page reads/programs")
    waf = (
        stats["physical_write_bytes"] / stats["logical_write_bytes"]
        if stats["logical_write_bytes"] else None)
    # The summary JSON serializes ratios to 6 decimals: compare at half-ulp.
    reported_waf = stats["waf"]
    if waf is None:
        if reported_waf is not None:
            failures.append("zero-byte denominator must serialize WAF as null")
    elif reported_waf is None or abs(waf - reported_waf) > 5e-7:
        failures.append(
            f"reported WAF {reported_waf} != {waf}")
    raw_codeword_bytes = stats["ecc_encode_codeword_bytes"]
    expected_raw_codeword_bytes = programs * (PAGE + OOB)
    if raw_codeword_bytes != expected_raw_codeword_bytes:
        failures.append(
            f"raw codeword writes {raw_codeword_bytes} != programs x "
            f"(page + OOB) {expected_raw_codeword_bytes}")
    # --- independently stated regime expectations ---
    failures.extend(
        verify_regime(case, waf, reloc, stats["gc_runs"]))
    victim_valid_fraction = (
        reloc / (stats["gc_runs"] * PAGES_PER_BLOCK)
        if stats["gc_runs"] else 0.0)
    detail = {
        "waf": waf,
        "logical_write_bytes": stats["logical_write_bytes"],
        "physical_write_bytes": stats["physical_write_bytes"],
        "data_programs": data_programs,
        "page_programs": programs,
        "mapping_programs": mapping,
        "gc_relocations": reloc,
        "gc_reclaimed_invalid_pages": stats["gc_reclaimed_invalid_pages"],
        "gc_victim_valid_fraction": victim_valid_fraction,
        "coalesced_writes": coalesced,
        "gc_runs": stats["gc_runs"],
        "block_erases": stats["block_erases"],
        "free_pages": stats["free_pages"],
        "accounting_verified": stats["accounting_verified"],
    }
    return failures, detail


def time_breakdown_input(
    case: AmplificationCase,
    result: dict,
) -> SummaryInput:
    """Attach deterministic suite identity without mutating canonical JSON."""
    return SummaryInput(
        label=case.key,
        summary=result["summary"],
        source=str(Path(result["summary_path"]).resolve()),
        scenario_metadata={
            "all-HBF": {
                "case": case.key,
                "config": TIME_BREAKDOWN_CONFIG,
            },
        },
    )


def verification_csv_row(
    case: AmplificationCase,
    detail: dict,
    verdict: str,
) -> dict[str, object]:
    """Build a named row so header and value order cannot silently diverge."""
    lo, hi = case.waf_window
    return {
        "case": case.key,
        "u_eff": f"{case.u_eff:.4f}",
        "waf": f"{detail['waf']:.6f}",
        "waf_window_lo": lo,
        "waf_window_hi": hi,
        "logical_write_bytes": detail["logical_write_bytes"],
        "physical_write_bytes": detail["physical_write_bytes"],
        "data_programs": detail["data_programs"],
        "page_programs": detail["page_programs"],
        "mapping_programs": detail["mapping_programs"],
        "gc_relocations": detail["gc_relocations"],
        "gc_reclaimed_invalid_pages": detail["gc_reclaimed_invalid_pages"],
        "gc_victim_valid_fraction":
            f"{detail['gc_victim_valid_fraction']:.6f}",
        "coalesced_writes": detail["coalesced_writes"],
        "gc_runs": detail["gc_runs"],
        "block_erases": detail["block_erases"],
        "free_pages": detail["free_pages"],
        "accounting_verified": str(detail["accounting_verified"]).lower(),
        "verdict": verdict,
    }


def console_audit_header() -> str:
    return (
        f"{'case':14s} {'u_eff':>6s} {'WAF':>11s} "
        f"{'WAF窗口[下,上]':>16s} {'data页':>8s} "
        f"{'all程序':>8s} {'mapping页':>9s} {'搬迁页':>8s} "
        f"{'回收无效':>8s} {'victim有效':>10s} "
        f"{'合并':>6s} {'gc次':>6s} {'擦除':>6s} "
        f"{'free页':>8s} {'审计':>5s}  判定"
    )


def console_audit_row(
    case: AmplificationCase,
    detail: dict,
    verdict: str,
) -> str:
    lo, hi = case.waf_window
    audit = "yes" if detail["accounting_verified"] else "no"
    return (
        f"{case.key:14s} {case.u_eff:6.1%} "
        f"{detail['waf']:11.4f} "
        f"[{lo:>5.2f},{hi:>6.2f}] {detail['data_programs']:8d} "
        f"{detail['page_programs']:8d} {detail['mapping_programs']:9d} "
        f"{detail['gc_relocations']:8d} "
        f"{detail['gc_reclaimed_invalid_pages']:8d} "
        f"{detail['gc_victim_valid_fraction']:10.2%} "
        f"{detail['coalesced_writes']:6d} {detail['gc_runs']:6d} "
        f"{detail['block_erases']:6d} {detail['free_pages']:8d} "
        f"{audit:>5s}  {verdict}"
    )



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-compare", type=Path,
                        default=Path("build/scenario_compare"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--validation-certificate",
        type=Path,
        help=(
            "attach a verified foundational certificate; omitted runs remain "
            "explicitly exploratory"
        ),
    )
    args = parser.parse_args()
    args.scenario_compare = args.scenario_compare.resolve()
    if not args.scenario_compare.is_file():
        parser.error(f"scenario_compare not found: {args.scenario_compare}")
    validation_certificate: VerifiedCertificate | None = None
    if args.validation_certificate is not None:
        try:
            validation_certificate = verify_certificate(
                args.validation_certificate,
                repository=ROOT,
                scenario_compare=args.scenario_compare,
            )
        except CertificateError as error:
            parser.error(f"invalid validation certificate: {error}")
    if args.out_dir is None:
        args.out_dir = DEFAULT_OUT
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / "write-amplification-verification.csv"
    time_csv_path = args.out_dir / "time-breakdown.csv"
    time_markdown_path = args.out_dir / "time-breakdown.md"
    time_html_path = args.out_dir / "time-breakdown.html"
    table_path.unlink(missing_ok=True)
    time_csv_path.unlink(missing_ok=True)
    time_markdown_path.unlink(missing_ok=True)
    time_html_path.unlink(missing_ok=True)

    rows = []
    time_inputs = []
    failed = False
    for case in CASES:
        result = run_case(
            args.scenario_compare,
            case,
            args.out_dir,
            validation_certificate,
        )
        time_inputs.append(time_breakdown_input(case, result))
        failures, detail = verify(case, result)
        verdict = "PASS" if not failures else "FAIL: " + "; ".join(failures)
        failed = failed or bool(failures)
        rows.append((case, detail, verdict))
        print(
            f"[done] {case.key}: WAF={detail['waf']:.4f} "
            f"({verdict.split(':')[0]})",
            flush=True)

    time_report = write_time_breakdown_report(
        time_inputs, time_csv_path, time_markdown_path)
    write_visualization(
        time_inputs,
        time_html_path,
        title="HBFSim GC/WAF timing",
    )
    print(
        f"wrote: {time_csv_path}, {time_markdown_path}, and {time_html_path} "
        f"({len(time_report.overviews)} scenarios)")

    by_key = {row[0].key: row for row in rows}
    relation_failures = verify_random_churn_relation(
        {key: row[1] for key, row in by_key.items()})
    if relation_failures:
        failed = True
        rows.append((AmplificationCase(
            "monotone", "随机覆盖单调性", 0, 0, (0, 0), False, False),
            {
                "waf": 0.0,
                "logical_write_bytes": 0,
                "physical_write_bytes": 0,
                "data_programs": 0,
                "page_programs": 0,
                "mapping_programs": 0,
                "gc_relocations": 0,
                "gc_reclaimed_invalid_pages": 0,
                "gc_victim_valid_fraction": 0.0,
                "coalesced_writes": 0,
                "gc_runs": 0,
                "block_erases": 0,
                "free_pages": 0,
                "accounting_verified": False,
            },
            "FAIL: " + "; ".join(relation_failures)))

    print("\n" + console_audit_header())
    csv_rows = []
    for case, d, verdict in rows:
        print(console_audit_row(case, d, verdict))
        csv_rows.append(verification_csv_row(case, d, verdict))
    temporary = table_path.with_suffix(table_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=WAF_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    temporary.replace(table_path)
    print(f"wrote: {table_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
