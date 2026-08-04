#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
probe="${1:-"${repo_root}/build/physical_probe"}"
out_dir="${2:-"${repo_root}/out/physical-guards"}"

if [[ ! -x "${probe}" ]]; then
  cmake -S "${repo_root}" -B "${repo_root}/build"
  cmake --build "${repo_root}/build" -j "${HBFSIM_BUILD_JOBS:-4}"
  probe="${repo_root}/build/physical_probe"
fi

mkdir -p "${out_dir}"
log="${out_dir}/physical-guards.log"
: > "${log}"

run_case() {
  local name="$1"
  shift
  local output="${out_dir}/${name}.txt"
  {
    printf '\n[guard] %s\n' "${name}"
    "${probe}" "$@"
  } | tee "${output}"
  cat "${output}" >> "${log}"
}

run_trace_case() {
  local name="$1"
  local probe_name="$2"
  local output="${out_dir}/${name}.txt"
  {
    printf '\n[guard] %s\n' "${name}"
    "${probe}" "${probe_name}" \
      --waterfall \
      --waterfall-limit 160 \
      --trace "${out_dir}/${name}.trace.json" \
      --folded "${out_dir}/${name}.folded"
  } | tee "${output}"
  cat "${output}" >> "${log}"
}

require_pattern() {
  local file="$1"
  local pattern="$2"
  local message="$3"
  if ! grep -Eq "${pattern}" "${file}"; then
    printf '[guard][fail] %s\n' "${message}" >&2
    printf '[guard][fail] see %s\n' "${file}" >&2
    exit 1
  fi
}

require_speedup_gt() {
  local file="$1"
  local threshold="$2"
  local label="$3"
  local speedup
  speedup="$(awk '/speedup/ {gsub("x", "", $3); print $3; exit}' "${file}")"
  if ! awk -v speedup="${speedup:-0}" -v threshold="${threshold}" 'BEGIN { exit !(speedup > threshold) }'; then
    printf '[guard][fail] expected %s speedup > %sx, got %s\n' \
      "${label}" "${threshold}" "${speedup:-missing}" >&2
    printf '[guard][fail] see %s\n' "${file}" >&2
    exit 1
  fi
}

require_metric_gt() {
  local file="$1"
  local metric="$2"
  local threshold="$3"
  local label="$4"
  local value
  value="$(awk -v metric="${metric}" '{
    for (i = 1; i <= NF; ++i) {
      if (index($i, metric "=") == 1) {
        split($i, parts, "=")
        value = parts[2]
      }
    }
  } END { print value }' "${file}")"
  if ! awk -v value="${value:-0}" -v threshold="${threshold}" 'BEGIN { exit !(value > threshold) }'; then
    printf '[guard][fail] expected %s %s > %s, got %s\n' \
      "${label}" "${metric}" "${threshold}" "${value:-missing}" >&2
    printf '[guard][fail] see %s\n' "${file}" >&2
    exit 1
  fi
}

require_refresh_penalty() {
  local file="$1"
  local line
  local count
  local stall
  line="$(awk '/refresh on:/ {seen=1; next} seen && /HBM stats:/ {print; exit}' "${file}")"
  count="$(printf '%s\n' "${line}" | sed -n 's/.*refresh_count=\([0-9][0-9]*\).*/\1/p')"
  stall="$(printf '%s\n' "${line}" | sed -n 's/.*refresh_stall_ns=\([0-9.][0-9.]*\).*/\1/p')"
  if ! awk -v count="${count:-0}" -v stall="${stall:-0}" 'BEGIN { exit !(count > 0 && stall > 0.0) }'; then
    printf '[guard][fail] expected refresh-on run to report positive refresh count and stall\n' >&2
    printf '[guard][fail] got line: %s\n' "${line:-missing}" >&2
    printf '[guard][fail] see %s\n' "${file}" >&2
    exit 1
  fi
}

run_case hbm-interface hbm-interface
for marker in \
  hbm_interface_derivation_ok \
  hbm_address_interleave_ok \
  hbm_address_golden_vectors_ok \
  hbm_uint64_mapping_ok \
  hbm_command_clock_alignment_ok \
  hbm_tccd_analytical_ok \
  hbm_long_train_clock_stability_ok \
  hbm_stride_distribution_ok \
  hbm_sequential_peak_reachable \
  hbm_interface_metamorphic_ok \
  hbm_invalid_profile_rejected; do
  require_pattern "${out_dir}/hbm-interface.txt" \
    "${marker}=yes" \
    "hbm-interface contract failed: ${marker}"
done

run_trace_case hbm-row hbm-row
require_pattern "${out_dir}/hbm-row.txt" \
  'row_hits=4 row_misses=1 row_conflicts=1' \
  'hbm-row should count one miss/conflict plus four derived full-channel BL8 bursts'
require_pattern "${out_dir}/hbm-row.txt" \
  'hbm_command[[:space:]]+PRE.*closes_row' \
  'hbm-row conflict should issue a row-closing PRE before the next ACT'

run_case hbm-boundaries hbm-boundaries
require_pattern "${out_dir}/hbm-boundaries.txt" \
  'hbm_stage_work_child_aggregation_ok=yes' \
  'hbm-boundaries must aggregate every physical burst into canonical stage work'
require_pattern "${out_dir}/hbm-boundaries.txt" \
  'hbm_stage_work_trace_independent_ok=yes' \
  'hbm canonical stage work must not depend on trace collection'

run_case hbm-channels hbm-channels
require_speedup_gt "${out_dir}/hbm-channels.txt" 1.5 "multi-channel HBM"
require_pattern "${out_dir}/hbm-channels.txt" \
  'active_pch=4/4' \
  'hbm-channels should use all four pseudo-channels in the 4-channel run'

run_case hbm-refresh hbm-refresh
require_refresh_penalty "${out_dir}/hbm-refresh.txt"

run_case hbm-tfaw hbm-tfaw
require_pattern "${out_dir}/hbm-tfaw.txt" \
  'tfaw_ok=yes' \
  'hbm-tfaw should sustain at most four activations per tFAW window'

run_case hbm-frfcfs hbm-frfcfs
require_pattern "${out_dir}/hbm-frfcfs.txt" \
  'frfcfs_hit_first=yes' \
  'hbm-frfcfs: a queued row hit should bypass an older conflict'
require_pattern "${out_dir}/hbm-frfcfs.txt" \
  'qd1_in_order=yes' \
  'hbm-frfcfs: QD1 issue must keep arrival order'
require_pattern "${out_dir}/hbm-frfcfs.txt" \
  'no_clairvoyance=yes' \
  'hbm-frfcfs: hits arriving after the oldest could issue must not bypass it'
require_pattern "${out_dir}/hbm-frfcfs.txt" \
  'starvation_cap_ok=yes' \
  'hbm-frfcfs: the age cap must bound how far hits bypass the oldest request'
require_pattern "${out_dir}/hbm-frfcfs.txt" \
  'mixed_rw_readiness_ok=yes' \
  'hbm-frfcfs: a row hit blocked by R/W gates must not delay a ready conflict'
require_pattern "${out_dir}/hbm-frfcfs.txt" \
  'same_arrival_bounded_bypass_ok=yes' \
  'hbm-frfcfs: equal-timestamp hits must have a bounded bypass count'
require_pattern "${out_dir}/hbm-frfcfs.txt" \
  'large_request_streaming_ok=yes' \
  'hbm-frfcfs: large parents must stream through the bounded controller queue'

run_trace_case hbm-turnaround hbm-turnaround
require_pattern "${out_dir}/hbm-turnaround.txt" \
  'derived_burst_gates_ok=yes' \
  'hbm-turnaround should conserve RD/WR gates using the one derived burst duration'

run_trace_case hbf-rounding hbf-rounding
require_pattern "${out_dir}/hbf-rounding.txt" \
  'logical_write=128 physical_write=4096' \
  'hbf-rounding should expose data-page plus mapping-page physical writes'
require_pattern "${out_dir}/hbf-rounding.txt" \
  'mapping_programs=1' \
  'hbf-rounding should program one mapping page'

run_case hbf-basic hbf-basic
require_pattern "${out_dir}/hbf-basic.txt" \
  'hbf_stage_work_exact_once_ok=yes' \
  'hbf-basic must aggregate every issue exactly once into canonical stage work'
require_pattern "${out_dir}/hbf-basic.txt" \
  'hbf_resource_busy_capacity_ok=yes' \
  'hbf-basic must expose exclusive resource busy work with capacity counts'
require_pattern "${out_dir}/hbf-basic.txt" \
  'hbf_time_accounting_trace_independent_ok=yes' \
  'hbf canonical time accounting must not depend on trace collection'

run_trace_case hbf-resident-mapping hbf-resident-mapping
require_pattern "${out_dir}/hbf-resident-mapping.txt" \
  'resident_mapping_no_flash_read_ok=yes' \
  'hbf-resident-mapping must read data pages only'
require_pattern "${out_dir}/hbf-resident-mapping.txt" \
  'mapping_lookups=3' \
  'hbf-resident-mapping must perform one controller-DRAM lookup per read'
require_pattern "${out_dir}/hbf-resident-mapping.txt" \
  'logical_read=1536 physical_read=1536' \
  'hbf-resident-mapping physical read bytes must equal logical data bytes'
require_pattern "${out_dir}/hbf-resident-mapping.txt" \
  'mapping_programs=0' \
  'read-only resident mapping must not create checkpoint programs'

run_trace_case hbf-gc hbf-gc
require_pattern "${out_dir}/hbf-gc.txt" \
  'gc_runs=[1-9][0-9]*' \
  'hbf-gc should trigger at least one real GC run'
require_pattern "${out_dir}/hbf-gc.txt" \
  'gc_relocations=[1-9][0-9]*' \
  'hbf-gc should relocate at least one valid page'
require_pattern "${out_dir}/hbf-gc.txt" \
  'mappings=3' \
  'hbf-gc should not count mapping metadata as user LPN entries'
require_pattern "${out_dir}/hbf-gc.txt" \
  'gc_relocation_mapping_persistence_ok=yes' \
  'hbf-gc relocation must update resident L2P and persist its dirty checkpoint'
require_pattern "${out_dir}/hbf-gc.txt" \
  'gc_role_headroom_no_false_trigger_ok=yes' \
  'hbf-gc must not debit whole-block headroom for appends into an active role block'
require_pattern "${out_dir}/hbf-gc.txt" \
  'gc_active_relocation_headroom_converges_ok=yes' \
  'hbf-gc must count active-GC leftovers and converge without repeated relocation'
require_pattern "${out_dir}/hbf-gc.txt" \
  'gc_multiplane_opening_debit_ok=yes' \
  'hbf-gc must charge a whole-block debit to the exact multi-plane opening request'

run_case hbf-parallelism hbf-parallelism
require_speedup_gt "${out_dir}/hbf-parallelism.txt" 2.5 "multi-plane HBF"
require_pattern "${out_dir}/hbf-parallelism.txt" \
  'active_planes=4/4' \
  'hbf-parallelism should use all four planes in the 4-plane run'
require_metric_gt "${out_dir}/hbf-parallelism.txt" media_parallelism 3.5 "multi-plane HBF"

run_case hbf-media-lanes hbf-media-lanes
require_speedup_gt "${out_dir}/hbf-media-lanes.txt" 2.5 "multi-lane HBF read"
require_pattern "${out_dir}/hbf-media-lanes.txt" \
  'active_media_lanes=4/4' \
  'hbf-media-lanes should use all four internal media lanes in the 4-lane run'
require_metric_gt "${out_dir}/hbf-media-lanes.txt" read_lane_parallelism 3.5 "multi-lane HBF read"

run_case hbf-page-buffer-banks hbf-page-buffer-banks
require_speedup_gt "${out_dir}/hbf-page-buffer-banks.txt" 2.5 "multi-bank HBF page buffer"
require_pattern "${out_dir}/hbf-page-buffer-banks.txt" \
  'active_page_buffer_banks=4/4' \
  'hbf-page-buffer-banks should use all four page-buffer banks in the 4-bank run'
require_metric_gt "${out_dir}/hbf-page-buffer-banks.txt" page_buffer_bank_parallelism 3.5 \
  "multi-bank HBF page buffer"

run_trace_case hbf-subarrays hbf-subarrays
require_speedup_gt "${out_dir}/hbf-subarrays.txt" 1.5 "multi-subarray HBF read"
require_pattern "${out_dir}/hbf-subarrays.txt" \
  'subarray[0-9]' \
  'hbf-subarrays should expose derived subarray trace entities'
require_pattern "${out_dir}/hbf-subarrays.txt" \
  'one subarray serializes eight rounds; four subarrays batch into ~3 rounds' \
  'hbf-subarrays should state the batch round expectation'
require_metric_gt "${out_dir}/hbf-subarrays.txt" subarray_read_parallelism 1.5 \
  "cross-subarray HBF read"

run_case hbf-exact-calendar hbf-exact-calendar
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'resource_calendar_backfill_over_64_ok=yes' \
  'hbf-exact-calendar must preserve every reusable idle interval beyond 64 gaps'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'sense_round_backfill_over_64_ok=yes' \
  'hbf-exact-calendar must preserve every joinable future sense round beyond 64 rounds'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'plane_barrier_backfill_ok=yes' \
  'hbf-exact-calendar must backfill reads before later-ready full-plane programs'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'plane_barrier_no_straddle_ok=yes' \
  'hbf-exact-calendar must not let a read data path straddle a full-plane program'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'full_plane_program_reverse_backfill_ok=yes' \
  'hbf-exact-calendar must backfill a short program before a future read'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'full_plane_erase_reverse_backfill_ok=yes' \
  'hbf-exact-calendar must backfill a short erase before a future read'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'read_buffer_future_fill_merge_ok=yes' \
  'hbf-exact-calendar must merge a read with its in-flight read-buffer fill'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'read_buffer_delayed_lookup_no_false_hit_ok=yes' \
  'hbf-exact-calendar must not claim a fill after an intervening eviction opportunity'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'read_buffer_partial_overlay_exact_bytes_ok=yes' \
  'hbf-exact-calendar must charge exact bytes for partial read-buffer overlays'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'program_invalidates_read_buffer_ok=yes' \
  'hbf-exact-calendar must retire an erased-value cache line when its PPN is programmed'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'erase_epoch_no_resurrection_ok=yes' \
  'hbf-exact-calendar must drop stale mapping publications after erase'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'erase_read_buffer_epoch_ok=yes' \
  'hbf-exact-calendar must retire read-buffer data with its erased block epoch'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'erase_pending_mapping_dependency_ok=yes' \
  'hbf-exact-calendar must retain the erase dependency of a retired pending mapping'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'resident_mapping_ignores_checkpoint_erase_ok=yes' \
  'hbf-exact-calendar resident lookup must not wait on a persistent checkpoint erase'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'erase_waits_target_program_ok=yes' \
  'hbf-exact-calendar must order a destructive erase after an issued program to its block'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'raw_read_waits_target_program_ok=yes' \
  'hbf-exact-calendar must order a raw physical read after an issued program to its PPN'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'erase_waits_target_read_ok=yes' \
  'hbf-exact-calendar must order a destructive erase after an issued read from its block'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'targeted_state_no_reverse_time_travel_ok=yes' \
  'hbf-exact-calendar targeted waits must preserve program-before-mapping state order'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'erase_plane_call_order_invariant_ok=yes' \
  'hbf-exact-calendar independent-plane timing must not depend on API call order'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'erase_stack_isolation_ok=yes' \
  'hbf-exact-calendar must keep erase state barriers local to one stack'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'erase_same_stack_plane_isolation_ok=yes' \
  'hbf-exact-calendar must not serialize an erase with another plane in the same stack'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'multipage_cross_stack_erase_dependency_ok=yes' \
  'hbf-exact-calendar must honor every stack touched by a multi-page request'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'raw_physical_ownership_ok=yes' \
  'hbf-exact-calendar must isolate raw physical blocks from FTL ownership'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'static_reservation_fail_closed_ok=yes' \
  'hbf-exact-calendar static reservations must validate before mutating state'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'resident_mapping_budget_fail_closed_ok=yes' \
  'hbf-exact-calendar must reject controller DRAM smaller than the full L2P table'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'resident_mapping_pipeline_ok=yes' \
  'hbf-exact-calendar must pipeline resident-map issues without serializing response latency'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'resident_mapping_future_dirty_drain_ok=yes' \
  'hbf-exact-calendar drain must persist future resident-map dirty generations'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'write_buffer_generation_lifetime_ok=yes' \
  'hbf-exact-calendar must keep WB slot ownership aligned with generation readability'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'gc_inflight_ownership_pin_ok=yes' \
  'hbf-exact-calendar must pin in-flight program and mapping targets against GC'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'drain_causal_barrier_ok=yes' \
  'hbf-exact-calendar must reject issue/drain time travel without mutating state'
require_pattern "${out_dir}/hbf-exact-calendar.txt" \
  'drain_future_state_barrier_ok=yes' \
  'hbf-exact-calendar must order same-arrival work after a future-completing drain'

run_trace_case hbf-ecc-pipeline hbf-ecc-pipeline
for marker in \
  ecc_first_read_latency_ok \
  ecc_same_die_ii_ok \
  ecc_cross_die_parallel_ok \
  ecc_oob_codeword_ok \
  ecc_raw_payload_path_ok \
  ecc_internal_route_ok \
  ecc_partial_erased_fill_ok \
  ecc_unmapped_erased_read_ok \
  ecc_invalid_write_no_side_effect_ok \
  ecc_erase_then_program_ok \
  ecc_saturation_ok \
  ecc_shared_read_write_issue_ok \
  ecc_invalid_config_rejected; do
  require_pattern "${out_dir}/hbf-ecc-pipeline.txt" \
    "${marker}=yes" \
    "hbf-ecc-pipeline contract failed: ${marker}"
done
require_pattern "${out_dir}/hbf-ecc-pipeline.txt" \
  'ecc_issue[[:space:]]+user/ecc_decode_issue' \
  'hbf-ecc-pipeline trace must expose decode initiation occupancy'
require_pattern "${out_dir}/hbf-ecc-pipeline.txt" \
  'ecc_latency[[:space:]]+user/ecc_decode_latency' \
  'hbf-ecc-pipeline trace must expose overlappable decode response latency'

run_case hbf-io hbf-io
require_pattern "${out_dir}/hbf-io.txt" \
  'hbio_isolated_bottleneck_ok=yes' \
  'hbf-io must isolate external payload bandwidth from raw TSV/channel/ECC'

run_trace_case hbf-tsu hbf-tsu
require_pattern "${out_dir}/hbf-tsu.txt" \
  'flash_scheduler_issue_read' \
  'hbf-tsu should expose read transaction scheduler issue'
require_pattern "${out_dir}/hbf-tsu.txt" \
  'flash_scheduler_issue_program' \
  'hbf-tsu should expose program transaction scheduler issue'
require_pattern "${out_dir}/hbf-tsu.txt" \
  'flash_scheduler_issue_erase' \
  'hbf-tsu should expose erase transaction scheduler issue'
require_pattern "${out_dir}/hbf-tsu.txt" \
  'flash_sched_enq=3 flash_sched_issue=3' \
  'hbf-tsu should account exactly three flash scheduler transactions'

run_trace_case hbf-coalesce hbf-coalesce
require_pattern "${out_dir}/hbf-coalesce.txt" \
  'logical_write=128 physical_write=1024' \
  'hbf-coalesce should turn two 64B writes into one data page plus one mapping page'
require_pattern "${out_dir}/hbf-coalesce.txt" \
  'write_buffer_hits=1 write_buffer_misses=1 write_buffer_flushes=1 write_buffer_read_hits=1' \
  'hbf-coalesce should stage, merge, read-hit, and flush the write buffer'
require_pattern "${out_dir}/hbf-coalesce.txt" \
  'mapping_programs=1' \
  'hbf-coalesce should commit one mapping page'

run_case hbf-write-backpressure hbf-write-backpressure
require_pattern "${out_dir}/hbf-write-backpressure.txt" \
  'burst_slot_waits=0' \
  'hbf-write-backpressure: writes inside the SRAM slot budget must not wait'
require_pattern "${out_dir}/hbf-write-backpressure.txt" \
  'sustained_slot_waits=12' \
  'hbf-write-backpressure: every admit past the budget must wait for a program'
require_pattern "${out_dir}/hbf-write-backpressure.txt" \
  'admit_floor_ok=yes' \
  'hbf-write-backpressure: last admit must sit within one program of the serialized floor'

run_trace_case hbf-mapping-writeback hbf-mapping-writeback
require_pattern "${out_dir}/hbf-mapping-writeback.txt" \
  'logical_write=1536 physical_write=2048' \
  'hbf-mapping-writeback should write three data pages plus one mapping page'
require_pattern "${out_dir}/hbf-mapping-writeback.txt" \
  'page_programs=4' \
  'hbf-mapping-writeback should program three data pages and one mapping page'
require_pattern "${out_dir}/hbf-mapping-writeback.txt" \
  'mapping_programs=1' \
  'hbf-mapping-writeback should persist the shared dirty mapping page once at drain'

run_case composition-kind-blind composition-kind-blind
require_pattern "${out_dir}/composition-kind-blind.txt" \
  'kind_blind_ok=yes' \
  'composition-kind-blind: kind-agnostic presets must route identically with/without labels'
require_pattern "${out_dir}/composition-kind-blind.txt" \
  'kind_refines_direct_reads=yes' \
  'composition-kind-blind: read-only-direct labels may only refine reads toward HBM'

run_case composition-coop-write composition-coop-write
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_burst_ok=yes' \
  'composition-coop-write: bursts within the HBM write region complete at HBM speed'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_backpressure_ok=yes' \
  'composition-coop-write: a full region must throttle to destage/program pace'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_streamed_ok=yes' \
  'composition-coop-write: streamed (layer-wise) destage must conserve programs'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_streamed_foreground_tie_priority_ok=yes' \
  'composition-coop-write: same-time streamed background yields to foreground HBM'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_subpage_merge_ok=yes' \
  'composition-coop-write: same-page sub-page writes share one finite slot and exact dirty ranges'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_cross_page_subpage_ok=yes' \
  'composition-coop-write: unaligned cross-page writes split into byte-exact page residents'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_astra_subpage_sizes_ok=yes' \
  'composition-coop-write: 1536B/2304B ASTRA writes preserve host and dirty-union bytes'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_cross_page_one_slot_stream_ok=yes' \
  'composition-coop-write: a 2304B cross-page write streams through one page slot'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_oversized_write_streamed_ok=yes' \
  'composition-coop-write: requests larger than the region stream through finite page slots'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_slot_reuse_coherence_ok=yes' \
  'composition-coop-write: a released slot must not serve a stale logical page'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'coop_invalid_region_geometry_rejected=yes' \
  'composition-coop-write: slot region size/base must be full-page aligned'
require_pattern "${out_dir}/composition-coop-write.txt" \
  'base_die_link_fixed_latency_work_ok=yes' \
  'composition-coop-write: base-die link fixed response latency work must be conserved'

run_case composition-reuse-routing composition-reuse-routing
require_pattern "${out_dir}/composition-reuse-routing.txt" \
  'reuse_flat_boundary_split_ok=yes' \
  'composition-reuse-routing: flat requests must split at the tier boundary'
require_pattern "${out_dir}/composition-reuse-routing.txt" \
  'reuse_readonly_ok=yes' \
  'composition-reuse-routing: read-only-direct keeps written data on HBM'
require_pattern "${out_dir}/composition-reuse-routing.txt" \
  'reuse_parked_ok=yes' \
  'composition-reuse-routing: pages parked in the HBM write region read from HBM'
require_pattern "${out_dir}/composition-reuse-routing.txt" \
  'closed_loop_monotonic_admission_ok=yes' \
  'composition-reuse-routing: bounded-window admission must stay monotonic'
require_pattern "${out_dir}/composition-reuse-routing.txt" \
  'direct_front_end_admission_wait_ok=yes' \
  'composition-reuse-routing: direct closed-loop admission wait work must be conserved'

printf '\n[guard] physical scenario guards passed\n'
printf '[guard] logs:   %s\n' "${log}"
printf '[guard] traces: %s/*.trace.json\n' "${out_dir}"
