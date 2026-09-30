"""Test support: run every HBF mapping organization through one persistent engine session.

`run()` replays the same synthetic traffic through the seven organizations in
`tests/fixtures/mapping-organizations/`, checkpoints, restarts from the saved
image and reports per-phase mapping snapshots; `mapping_observations()` checks
the engine's foreground / terminal-drain mapping decomposition. Configured
timings only, no measured hardware.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from hbfsim_client.simulation_session import ResolvedSystemConfig, SimulationSession
from hbfsim_client.transaction_protocol import Transaction

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "tests/fixtures/mapping-organizations"
VARIANTS = ("resident", "page-cache", "entry-cache", "block", "block-log", "extent", "object-segment")


def submit(session: SimulationSession, name: str, accesses: list[tuple[str, int, int]]) -> dict:
    """One outstanding request, including across chunks: compare equal ordering."""
    transactions = []
    for index, (op, address, size) in enumerate(accesses):
        transactions.append(Transaction(id=f"{name}-{index}", target="HBF_LOGICAL", op=op,
            addr=address, bytes=size, issue_ns=0,
            dependencies=() if index == 0 else (transactions[-1].id,)))
    result = session.run(transactions, completions=False)
    return dict(elapsed_ns=result.elapsed_ns, transactions=len(transactions))


def run(simulator: Path, output: Path, *, operations: int = 4096,
        logical_pages: int = 6144, blocks: int = 256, pages_per_block: int = 32,
        seed: int = 19, buffer_pages: int = 0, flush_threshold: int = 0,
        flush_on_completion: bool = False, variants: tuple[str, ...] = VARIANTS) -> dict:
    if operations < 1 or logical_pages < 2 or blocks < 16 or pages_per_block < 2:
        raise ValueError("comparison needs positive traffic and nontrivial NAND geometry")
    if flush_threshold > buffer_pages or (flush_on_completion and not buffer_pages):
        raise ValueError("flush policy requires a sufficient write buffer")
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    if logical_pages < pages_per_block:
        raise ValueError("logical capacity must hold a full block")
    workloads = {
        "block-writes": [("W", (i % (logical_pages // pages_per_block)) * pages_per_block * 4096,
                          pages_per_block * 4096) for i in range(max(1, operations // pages_per_block))],
        "sequential": [("W", (i % logical_pages) * 4096, 4096) for i in range(operations)],
        "random": [("W", rng.randrange(logical_pages) * 4096, 4096) for _ in range(operations)],
        "fragments": [("W", (i // 64 % min(16, logical_pages)) * 4096 + i % 64 * 64, 64)
                      for i in range(max(64, operations // 4))],
        "reads": [("R", rng.randrange(logical_pages) * 4096, 64) for _ in range(operations // 4 + 1)],
    }
    trace_hash = hashlib.sha256(json.dumps(workloads, sort_keys=True).encode()).hexdigest()
    rows = []
    for variant in variants:
        directory = output / variant
        directory.mkdir(exist_ok=True)
        override = directory / "workload.cfg"
        # Cache keeps two pages after directory, scratch, GC and payload
        # reservations. Its budget is explicit; other primary indexes derive
        # their own worst-case budget. Report both reservation and actual use.
        directory_bytes = ((blocks * pages_per_block + 511) // 512) * 8
        cache_budget = directory_bytes + 4 * (4096 + 16) + 4096 + buffer_pages * 4096
        override.write_text(
            f"hbf-blocks-per-plane={blocks}\nhbf-pages-per-block={pages_per_block}\n"
            f"hbf-logical-capacity-bytes={logical_pages * 4096}\n"
            f"hbf-write-coalescing={'true' if buffer_pages else 'false'}\n"
            f"hbf-write-buffer-pages={buffer_pages}\n"
            f"hbf-write-buffer-flush-threshold-pages={flush_threshold}\n"
            f"hbf-write-buffer-completion-requires-flush={'true' if flush_on_completion else 'false'}\n"
            f"hbf-ctrl-dram-bytes={cache_budget if variant in {'page-cache', 'entry-cache'} else 0}\n")
        profiles = (ROOT / "configs/systems/eight-stack-baseline.cfg", PROFILES / "base.cfg",
                    PROFILES / f"{variant}.cfg", override)
        config = ResolvedSystemConfig.load(profiles).resolve(simulator)
        phases = []
        image = directory / "checkpoint.image"
        with SimulationSession(simulator_path=simulator, system_config=config,
                enable_hbm=True, enable_hbf=True, hbf_wear_output_prefix=directory / "wear") as session:
            if variant == "object-segment":
                session.hbf_object_command("CREATE", "create", object_id=1, first_lpn=0, page_count=logical_pages)
            for first in range(0, logical_pages, 256):
                submit(session, f"initial-{first}", [("W", p * 4096, 4096)
                    for p in range(first, min(first + 256, logical_pages))])
            session.checkpoint("initial")
            for name, trace in workloads.items():
                before = session.hbf_mapping_snapshot(f"before-{name}")
                for first in range(0, len(trace), 256):
                    submit(session, f"{name}-{first}", trace[first:first + 256])
                foreground = session.hbf_mapping_snapshot(f"foreground-{name}")
                session.checkpoint(f"persist-{name}")
                after = session.hbf_mapping_snapshot(f"after-{name}")
                logical = after["logical_write_bytes"] - before["logical_write_bytes"]
                phases.append(dict(name=name, before=before, foreground=foreground, after=after,
                    write_amplification=None if not logical else
                        (after["physical_write_bytes"] - before["physical_write_bytes"]) / logical))
            # Expiry uses the object directory where available; the page/extent
            # variants receive the same logical deallocation and replacement.
            if variant == "object-segment":
                session.hbf_object_command("SEAL", "seal", object_id=1)
            session.checkpoint_image("save", image)
            saved = session.hbf_mapping_snapshot("saved")
        with SimulationSession(simulator_path=simulator, system_config=config,
                enable_hbm=True, enable_hbf=True, initial_hbf_persistent_image=image,
                hbf_wear_output_prefix=directory / "resumed-wear") as resumed:
            boot = resumed.hbf_mapping_snapshot("boot")
            submit(resumed, "read-restored", [("R", 0, 64), ("R", (logical_pages - 1) * 4096, 64)])
            if variant == "object-segment":
                resumed.hbf_object_command("DELETE", "delete", object_id=1)
                resumed.hbf_object_command("CREATE", "recreate", object_id=2, first_lpn=0, page_count=logical_pages)
            else:
                resumed.invalidate_hbf_pages("expire", first_lpn=0, page_count=logical_pages)
            submit(resumed, "replace", [("W", 0, 64), ("R", 0, 64)])
            resumed.checkpoint("replacement")
            recovered = resumed.hbf_mapping_snapshot("recovered")
        row = dict(variant=variant, phases=phases, saved=saved, recovery_boot=boot,
                   recovery_final=recovered, checkpoint_bytes=image.stat().st_size,
                   profiles=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in profiles])
        (directory / "result.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        rows.append(row)
    report = dict(schema="hbfsim.mapping_organizations.v1", seed=seed, trace_sha256=trace_hash,
        logical_pages=logical_pages, blocks=blocks, pages_per_block=pages_per_block,
        operations=operations, buffer_pages=buffer_pages, flush_threshold=flush_threshold,
        flush_on_completion=flush_on_completion,
        simulator_sha256=hashlib.sha256(simulator.read_bytes()).hexdigest(),
        scope="Serial requests; identical NAND geometry, logical capacity and trace; configured timings. "
              "Cache budget holds two mapping pages; other policies derive their reservation. "
              "Foreground plus explicit checkpoint costs are reported separately. Recovery is quiescent image restart.",
        results=rows)
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


# --- mapping observation decomposition -------------------------------------
PROGRAM_KINDS = ('data', 'copy', 'padding', 'metadata')
MEDIA_COUNTERS = ('logical_read_bytes', 'logical_write_bytes', 'physical_read_bytes',
                  'physical_write_bytes', 'page_reads', 'page_programs', 'block_erases')
MAINTENANCE = {
    'page': ('page_gc_and_static_wear_leveling',
             {'gc_runs': 'page_gc_runs', 'static_wear_leveling_runs': 'static_wl_runs'},
             {'page_gc': 'page_gc_copy_programs', 'static_wear_leveling': 'static_wl_copy_programs'}),
    'block': ('block_replacement', {'block_replacements': 'full_merges'},
              {'block_replacement': 'copy_programs'}),
    'block-log': ('block_log_merge',
                  {name: name for name in ('switch_merges', 'partial_merges', 'full_merges')},
                  {'block_log_merge': 'copy_programs'}),
    'extent': ('segment_cleaning', {'cleaned_segments': 'cleaned_segments'},
               {'segment_cleaning': 'copy_programs'}),
    'object-segment': ('segment_cleaning', {'cleaned_segments': 'cleaned_segments'},
                       {'segment_cleaning': 'copy_programs'}),
}


def _count(snapshot, field):
    value = snapshot[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'mapping observation has no nonnegative count for {field}')
    return value


def _window(before, after, start, end):
    organization = after['organization']
    if (before['organization'] != organization or
            before['page_size_bytes'] != after['page_size_bytes']):
        raise ValueError('mapping observation changed organization or NAND page size')
    page_bytes = _count(after, 'page_size_bytes')
    if not page_bytes:
        raise ValueError('mapping observation has an empty NAND page')

    def delta(field):
        value = _count(after, field) - _count(before, field)
        if value < 0:
            raise ValueError(f'mapping counter decreased across {start}/{end}: {field}')
        return value

    counters = {field: delta(field) for field in MEDIA_COUNTERS}
    programs = {kind: delta(f'{kind}_programs') for kind in PROGRAM_KINDS}
    if (sum(programs.values()) != counters['page_programs'] or
            counters['page_programs'] * page_bytes != counters['physical_write_bytes']):
        raise ValueError(f'mapping program decomposition diverged across {start}/{end}')
    kind, event_fields, copy_fields = MAINTENANCE[organization]
    copies = {cause: delta(field) for cause, field in copy_fields.items()}
    if sum(copies.values()) != programs['copy']:
        raise ValueError(f'mapping copy attribution diverged across {start}/{end}')
    written = counters['logical_write_bytes']
    return dict(start=start, end=end, **counters, programs=programs,
        program_payload_bytes={kind: count * page_bytes for kind, count in programs.items()},
        nand_per_logical_write_byte=counters['physical_write_bytes'] / written if written else None,
        maintenance=dict(kind=kind, events={name: delta(field) for name, field in event_fields.items()},
                         copy_programs_by_cause=copies))


def mapping_observations(initial, final_measurement):
    """Use only matching snapshots for each physical-write decomposition.

    Native generic GC counters describe the Page FTL. Segment cleaning and
    block replacements retain their own event names and are never inferred
    from generic gc_runs=0. Initial state can include timed recovery work.
    """
    boundaries = final_measurement['hbf_mapping_observations']
    foreground, post_drain = boundaries['foreground'], boundaries['post_drain']
    if initial is None:
        if foreground is not None or post_drain is not None:
            raise ValueError('missing initial HBF mapping observation')
        return None
    windows = {
        'foreground': _window(initial, foreground, 'initial', 'foreground'),
        'terminal_drain': _window(foreground, post_drain, 'foreground', 'post_drain'),
        'total': _window(initial, post_drain, 'initial', 'post_drain'),
    }
    final_native = final_measurement['device_workload_totals']['hbf']
    drain_native = final_measurement['end_of_session_drain']['device_delta']['hbf']
    for field in MEDIA_COUNTERS:
        if (_count(post_drain, field) != _count(final_native, field) or
                windows['terminal_drain'][field] != _count(drain_native, field)):
            raise ValueError(f'mapping and native end-of-session boundaries differ: {field}')
    return dict(schema='hbfsim.mapping_observations.v1', organization=initial['organization'],
        initial=initial, foreground=foreground, post_drain=post_drain, windows=windows,
        terminal_drain_ns=final_measurement['end_of_session_drain']['tail_ns'],
        semantics='Foreground excludes terminal drain; total includes it. Programs are attributed by the '
                  'primary mapping implementation. Maintenance events use organization-specific names; '
                  'a block replacement is not a Page GC run. Non-applicable Page-only snapshot counts are null.')
