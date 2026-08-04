#pragma once

#include "physical/physical_types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hbfsim::physical::hbf {

inline constexpr std::string_view kPlacementMappingScheme =
    "page-striped-stack-local-mapping-v2";

struct HbfConfig {
    std::uint32_t stacks = 1;
    std::uint32_t channels_per_stack = 8;
    std::uint32_t dies_per_channel = 4;
    std::uint32_t planes_per_die = 16;
    std::uint32_t blocks_per_plane = 512;
    std::uint32_t pages_per_block = 1024;
    std::uint64_t page_size_bytes = 2048;
    // Out-of-band (spare) bytes per page: ECC parity, reverse LPN mapping,
    // block status. Shares the page's wordline, so array timings are
    // unchanged. Flash-side raw transfers and ECC move page + OOB; decoded
    // SRAM and the external HBIO move payload only. Logical/physical byte
    // statistics stay data-area based.
    std::uint64_t oob_bytes_per_page = 128;
    std::uint32_t media_lanes_per_plane = 16;
    // 0 derives subarrays from media_lanes_per_plane.
    std::uint32_t subarrays_per_plane = 0;
    std::uint32_t page_buffer_banks_per_plane = 1;

    double t_read_page_ns = 4000.0;
    double t_program_page_ns = 75000.0;
    double t_erase_block_ns = 2'000'000.0;
    double t_program_verify_ns = 5000.0;
    // ECC response latency is independent of pipeline throughput. The raw
    // bandwidth includes data + OOB codeword bytes and sets the initiation
    // interval of one shared decode/encode issue port per flash die.
    // The library default 34 GB/s provisions the default 32-die stack for its
    // 1024 GB/s HBIO after 2048 B + 128 B codeword overhead. Published
    // profiles override it explicitly. These are assumptions, not vendor ECC
    // measurements.
    double ecc_decode_latency_ns = 500.0;
    double ecc_encode_latency_ns = 500.0;
    double ecc_decode_raw_bandwidth_GBps_per_die = 34.0;
    double ecc_encode_raw_bandwidth_GBps_per_die = 34.0;
    // Flash channel/media/page-buffer data bandwidths are raw codeword rates
    // (payload + OOB). The stack TSV uses one shared timeline for internal
    // command bytes and those raw codewords. HBIO is external payload
    // bandwidth and logic SRAM holds decoded payload.
    double channel_bandwidth_GBps = 136.0;
    double hb_io_bandwidth_GBps = 1024.0;
    double tsv_bandwidth_GBps = 1120.0;
    double media_lane_bandwidth_GBps = 2048.0;
    double logic_sram_bandwidth_GBps = 2048.0;
    double page_buffer_bandwidth_GBps = 2048.0;
    std::uint64_t command_address_bytes = 64;
    // Per-request dispatch cost of the stack's logic scheduler. HBM-class
    // PHYs issue a command every ~2 ns; at 4 KiB pages a 20 ns dispatch would
    // cap a stack at 200 GB/s and silently dominate the read fabric.
    double logic_scheduler_issue_ns = 2.0;
    double address_generation_ns = 5.0;
    // Every stack keeps its complete L2P table resident in controller DRAM.
    // Accesses are pipelined: issue_ns occupies the single modeled port while
    // latency_ns is the overlappable response latency.
    double ctrl_dram_latency_ns = 100.0;
    double ctrl_dram_issue_ns = 1.0;
    double mapping_update_ns = 25.0;
    double free_page_allocation_ns = 10.0;
    double flash_tsu_issue_ns = 10.0;
    // Global controller-DRAM budget, divided evenly across stacks. Zero
    // derives exactly enough capacity for the complete page-level mapping
    // table. An explicit undersized budget is rejected.
    std::uint64_t ctrl_dram_bytes = 0;
    // Number of L2P entries represented by one persistent mapping checkpoint
    // page. It also determines the resident entry width:
    // page_size_bytes / mapping_entries_per_page.
    std::uint64_t mapping_entries_per_page = 512;
    // Exploratory batch-synchronous sense activation per quadrant/plane,
    // inspired by public HBF material. false selects an optimistic independent-
    // subarray sensitivity bound; neither scheduler is a calibrated product fact.
    bool batch_activation = true;
    // Exploratory logic-die read buffer (per stack): pages are served from
    // SRAM on a hit, with no flash transaction. Capacity per stack; 0 disables.
    // Public HBF material does not specify its capacity.
    std::uint64_t read_buffer_pages = 1024;
    // Finite controller credits for page-granular foreground reads, per HBF
    // stack. A large memory-object request is split into page transactions and
    // every child holds one credit from translation admission through its
    // user-visible completion. This prevents one giant parent request from
    // bypassing the intended outstanding-work bound. The value is an explicit
    // architecture assumption, not a published HBF product parameter.
    std::uint64_t page_read_queue_depth_per_stack = 4096;
    bool write_coalescing_enabled = false;
    bool write_buffer_completion_requires_flush = false;
    std::uint64_t write_buffer_pages = 1024;
    std::uint64_t write_buffer_flush_threshold_pages = 0;
    bool auto_gc_enabled = true;
    std::uint64_t gc_low_watermark_pages = 0;
    std::uint64_t gc_hard_watermark_pages = 0;
    std::uint64_t gc_reserved_free_blocks_per_plane = 0;
    double gc_wear_leveling_weight = 0.0;
};

struct ResidentMappingCapacity {
    std::uint64_t pages_per_stack = 0;
    std::uint64_t bytes_per_stack = 0;
    std::uint64_t total_bytes = 0;
};

[[nodiscard]] ResidentMappingCapacity derive_resident_mapping_capacity(
    const HbfConfig& config);

struct HbfAddress {
    std::uint32_t stack = 0;
    std::uint32_t channel = 0;
    std::uint32_t die = 0;
    std::uint32_t plane = 0;
    std::uint32_t block = 0;
    std::uint32_t page = 0;
    std::uint64_t offset = 0;

    [[nodiscard]] std::string path() const;
};

struct HbfStats {
    std::uint64_t read_requests = 0;
    std::uint64_t program_requests = 0;
    std::uint64_t erase_requests = 0;
    std::uint64_t logical_read_bytes = 0;
    std::uint64_t logical_write_bytes = 0;
    std::uint64_t physical_read_bytes = 0;
    std::uint64_t physical_write_bytes = 0;
    // Payload-byte decomposition of physical_write_bytes. These counters are
    // updated at the actual program event, after any controller coalescing.
    // They deliberately exclude OOB; ecc_encode_codeword_bytes below is the
    // corresponding raw page+OOB quantity.
    std::uint64_t data_program_payload_bytes = 0;
    std::uint64_t mapping_program_payload_bytes = 0;
    std::uint64_t gc_relocation_payload_bytes = 0;
    std::uint64_t page_reads = 0;
    // Foreground data-page programs after write-buffer coalescing. Excludes
    // mapping-page programs and every GC relocation program. Together:
    // page_programs == data_programs + mapping_page_programs + gc_relocations.
    std::uint64_t data_programs = 0;
    std::uint64_t page_programs = 0;
    std::uint64_t block_erases = 0;
    std::uint64_t mapping_entries = 0;
    std::uint64_t invalidations = 0;
    // The complete L2P table is capacity-resident from time zero; these expose
    // its exact footprint and the pipelined per-stack DRAM service.
    std::uint64_t resident_mapping_table_bytes = 0;
    std::uint64_t resident_mapping_table_bytes_per_stack = 0;
    std::uint64_t resident_mapping_pages_per_stack = 0;
    std::uint64_t mapping_lookup_ops = 0;
    std::uint64_t mapping_user_lookup_ops = 0;
    std::uint64_t mapping_gc_lookup_ops = 0;
    std::uint64_t mapping_update_ops = 0;
    std::uint64_t mapping_user_update_ops = 0;
    std::uint64_t mapping_gc_update_ops = 0;
    std::uint64_t mapping_dram_wait_ops = 0;
    double mapping_dram_wait_ns = 0.0;
    double mapping_dram_wait_max_ns = 0.0;
    double mapping_dram_issue_busy_ns = 0.0;
    std::uint64_t mapping_dram_resources = 0;
    std::uint64_t mapping_page_programs = 0;
    std::uint64_t static_reserved_pages = 0;
    // Initial logical images are pre-existing media state: they consume
    // physical data/mapping pages but never count as workload writes or WAF.
    // Dense ranges use a compact directory and materialize only pages that
    // are overwritten or relocated.
    std::uint64_t initial_logical_data_pages = 0;
    std::uint64_t initial_mapping_pages = 0;
    std::uint64_t compact_initial_logical_data_pages = 0;
    std::uint64_t compact_initial_mapping_pages = 0;
    std::uint64_t compact_live_logical_data_pages = 0;
    std::uint64_t compact_live_mapping_pages = 0;
    std::uint64_t compact_retired_logical_data_pages = 0;
    std::uint64_t compact_retired_mapping_pages = 0;
    std::uint64_t write_buffer_hits = 0;
    std::uint64_t write_buffer_misses = 0;
    std::uint64_t read_buffer_hits = 0;
    std::uint64_t read_buffer_misses = 0;
    std::uint64_t read_buffer_read_bytes = 0;
    std::uint64_t write_buffer_flushes = 0;
    std::uint64_t write_buffer_merged_bytes = 0;
    std::uint64_t write_buffer_read_hits = 0;
    std::uint64_t write_buffer_read_bytes = 0;
    // Writes that had to wait for an SRAM slot: slots stay occupied until
    // the flushed page has actually been programmed, so sustained overload
    // is admitted at program rate (real capacity backpressure).
    std::uint64_t write_buffer_slot_wait_ops = 0;
    double write_buffer_slot_wait_ns = 0.0;
    std::uint64_t read_splits = 0;
    std::uint64_t read_split_pages = 0;
    std::uint64_t page_read_admission_events = 0;
    std::uint64_t page_read_admission_waited_pages = 0;
    double page_read_admission_wait_ns = 0.0;
    double page_read_admission_max_wait_ns = 0.0;
    std::uint64_t flash_scheduler_enqueues = 0;
    std::uint64_t flash_scheduler_issues = 0;
    std::uint64_t gc_runs = 0;
    std::uint64_t gc_relocations = 0;
    // Relocation and reclaim accounting is kept separate from foreground and
    // mapping-page programs. Every full GC victim must satisfy:
    //   gc_data_relocations + gc_mapping_relocations == gc_relocations
    //   gc_relocations + gc_reclaimed_invalid_pages
    //       == gc_runs * pages_per_block
    std::uint64_t gc_data_relocations = 0;
    std::uint64_t gc_mapping_relocations = 0;
    std::uint64_t gc_reclaimed_invalid_pages = 0;
    std::uint64_t gc_user_blocked_runs = 0;
    // Lazily recomputed structural audit of every physical page. Static
    // reservations consume whole blocks; pages without a materialized static
    // image are therefore reported explicitly instead of being mislabeled as
    // free or valid.
    std::uint64_t total_pages = 0;
    std::uint64_t free_pages = 0;
    std::uint64_t valid_pages = 0;
    std::uint64_t invalid_pages = 0;
    std::uint64_t pending_program_pages = 0;
    std::uint64_t pending_mapping_publications = 0;
    std::uint64_t static_unmaterialized_pages = 0;
    bool accounting_verified = false;
    double ecc_issue_busy_ns = 0.0;
    double ecc_decode_queue_wait_ns = 0.0;
    double ecc_encode_queue_wait_ns = 0.0;
    double ecc_decode_latency_work_ns = 0.0;
    double ecc_encode_latency_work_ns = 0.0;
    double ecc_decode_issue_busy_ns = 0.0;
    double ecc_encode_issue_busy_ns = 0.0;
    std::uint64_t ecc_decode_ops = 0;
    std::uint64_t ecc_encode_ops = 0;
    std::uint64_t ecc_codeword_bytes = 0;
    std::uint64_t ecc_decode_codeword_bytes = 0;
    std::uint64_t ecc_encode_codeword_bytes = 0;
    std::uint64_t active_ecc_dies = 0;
    std::uint64_t max_ecc_inflight_per_die = 0;
    double max_ecc_issue_busy_ns = 0.0;
    double avg_active_ecc_issue_busy_ns = 0.0;
    double media_busy_ns = 0.0;
    double hb_io_command_busy_ns = 0.0;
    double hb_io_data_busy_ns = 0.0;
    double sequencer_busy_ns = 0.0;
    double page_buffer_bank_busy_ns = 0.0;
    double read_lane_busy_ns = 0.0;
    double subarray_read_busy_ns = 0.0;
    double first_arrival_ns = std::numeric_limits<double>::infinity();
    double finish_ns = 0.0;
    std::uint64_t stacks = 0;
    std::uint64_t channels = 0;
    std::uint64_t active_channels = 0;
    std::uint64_t dies = 0;
    std::uint64_t active_dies = 0;
    std::uint64_t planes = 0;
    std::uint64_t active_planes = 0;
    std::uint64_t media_lanes = 0;
    std::uint64_t active_media_lanes = 0;
    std::uint64_t subarrays = 0;
    std::uint64_t active_subarrays = 0;
    std::uint64_t page_buffer_banks = 0;
    std::uint64_t active_page_buffer_banks = 0;
    std::uint64_t max_plane_ops = 0;
    std::uint64_t max_media_lane_reads = 0;
    std::uint64_t max_subarray_reads = 0;
    std::uint64_t max_page_buffer_bank_reads = 0;
    std::uint64_t max_die_transactions = 0;
    double max_plane_media_busy_ns = 0.0;
    double avg_active_plane_media_busy_ns = 0.0;
    double avg_active_plane_ops = 0.0;
    double max_media_lane_busy_ns = 0.0;
    double avg_active_media_lane_busy_ns = 0.0;
    double avg_active_media_lane_reads = 0.0;
    double max_subarray_busy_ns = 0.0;
    double avg_active_subarray_busy_ns = 0.0;
    double avg_active_subarray_reads = 0.0;
    double max_page_buffer_bank_busy_ns = 0.0;
    double avg_active_page_buffer_bank_busy_ns = 0.0;
    double avg_active_page_buffer_bank_reads = 0.0;
    double max_channel_busy_ns = 0.0;
    double avg_active_channel_busy_ns = 0.0;
    double avg_active_die_transactions = 0.0;
    // Sum of every top-level issue/drain Breakdown exactly once, including
    // all internal mapping/GC work attributed to that causal operation.
    // Stages may overlap and therefore must not be summed as elapsed time.
    Breakdown stage_work;
    // Exclusive resource reservation work. Dividing by resource count and
    // active wall span yields bounded utilization for the named resource.
    double logic_ingress_busy_ns = 0.0;
    double tsv_busy_ns = 0.0;
    double sram_busy_ns = 0.0;
    double flash_source_queue_busy_ns = 0.0;
    double channel_command_busy_ns = 0.0;
    double channel_data_busy_ns = 0.0;
    std::uint64_t logic_ingress_resources = 0;
    std::uint64_t tsv_resources = 0;
    std::uint64_t sram_resources = 0;
    std::uint64_t flash_source_queue_resources = 0;
    std::uint64_t channel_command_resources = 0;
    std::uint64_t channel_data_resources = 0;

    // Canonical WAF: programmed media payload bytes divided by user logical
    // write bytes. This includes data-page rounding, mapping-page programs,
    // and GC relocations, but deliberately excludes OOB/ECC and transport
    // framing. A zero host-write denominator is undefined, never WAF=0.
    [[nodiscard]] std::optional<double> waf() const;
    [[nodiscard]] double active_span_ns() const;
    [[nodiscard]] double media_utilization() const;
    [[nodiscard]] double io_utilization() const;
    [[nodiscard]] double media_parallelism() const;
    [[nodiscard]] double read_lane_parallelism() const;
    [[nodiscard]] double subarray_read_parallelism() const;
    [[nodiscard]] double channel_parallelism() const;
    [[nodiscard]] double hbio_parallelism() const;
    [[nodiscard]] double hbio_command_utilization() const;
    [[nodiscard]] double hbio_data_utilization() const;
    [[nodiscard]] double sequencer_parallelism() const;
    [[nodiscard]] double ecc_issue_parallelism() const;
    [[nodiscard]] double ecc_issue_utilization() const;
    [[nodiscard]] double plane_media_skew() const;
    [[nodiscard]] double plane_op_skew() const;
    [[nodiscard]] double media_lane_skew() const;
    [[nodiscard]] double media_lane_read_skew() const;
    [[nodiscard]] double subarray_busy_skew() const;
    [[nodiscard]] double subarray_read_skew() const;
    [[nodiscard]] double page_buffer_bank_parallelism() const;
    [[nodiscard]] double page_buffer_bank_skew() const;
    [[nodiscard]] double page_buffer_bank_read_skew() const;
    [[nodiscard]] double channel_busy_skew() const;
    [[nodiscard]] double die_transaction_skew() const;
};

struct DirtyRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    // Last-writer provenance for these bytes. Write-buffer ranges remain
    // disjoint, so an overwrite can replace the source of only the bytes it
    // actually updates without relabeling the rest of the buffered page.
    HeatmapTrafficSource heatmap_source = HeatmapTrafficSource::Direct;
};

// Read-only, deterministic state exported for independent validation.  This
// deliberately contains semantic FTL state rather than scheduler internals:
// callers can prove mapping/page/block conservation without changing time,
// materializing callbacks, or depending on hash-table iteration order.
struct HbfAuditMapping {
    std::uint64_t key = 0;
    std::uint64_t ppn = 0;
};

struct HbfAuditPage {
    std::uint64_t ppn = 0;
    std::string status;
    std::string owner;
    std::uint64_t logical_key = 0;
    std::uint64_t block_epoch = 0;
};

struct HbfAuditBlock {
    std::uint64_t block = 0;
    std::string role;
    std::uint32_t valid_pages = 0;
    std::uint32_t invalid_pages = 0;
    std::uint32_t free_pages = 0;
    std::uint32_t next_page = 0;
    std::uint32_t erase_count = 0;
    std::uint32_t pending_program_pages = 0;
    std::uint32_t pending_mapping_publications = 0;
    std::uint64_t epoch = 0;
    bool erase_pending = false;
};

struct HbfAuditSnapshot {
    std::vector<HbfAuditMapping> logical_mappings;
    std::vector<HbfAuditMapping> mapping_pages;
    std::vector<HbfAuditPage> materialized_pages;
    std::vector<HbfAuditBlock> blocks;
    std::vector<std::uint64_t> free_pages_per_stack;
    std::vector<std::uint64_t> data_allocation_cursors;
    std::vector<std::uint64_t> mapping_allocation_cursors;
    std::vector<std::uint64_t> gc_allocation_cursors;
    std::vector<std::uint64_t> dirty_mapping_vpns;
    std::uint64_t free_pages = 0;
    std::uint64_t pending_dirty_mapping_events = 0;
    std::uint64_t pending_lpn_updates = 0;
    std::uint64_t pending_vpn_updates = 0;
    std::uint64_t pending_commits = 0;
    std::uint64_t write_buffer_entries = 0;
    std::uint64_t inflight_buffered_generations = 0;
    std::uint64_t pending_physical_programs = 0;
    std::uint64_t pending_physical_erases = 0;

    [[nodiscard]] bool quiescent() const {
        return dirty_mapping_vpns.empty() &&
            pending_dirty_mapping_events == 0 &&
            pending_lpn_updates == 0 &&
            pending_vpn_updates == 0 &&
            pending_commits == 0 &&
            write_buffer_entries == 0 &&
            inflight_buffered_generations == 0 &&
            pending_physical_programs == 0 &&
            pending_physical_erases == 0;
    }
};

class HbfDevice {
public:
    explicit HbfDevice(
        HbfConfig config,
        AddressHeatmap* address_heatmap = nullptr);

    [[nodiscard]] HbfAddress decode(std::uint64_t addr) const;
    [[nodiscard]] std::uint64_t encode(const HbfAddress& addr) const;
    [[nodiscard]] PhysicalCompletion issue(const PhysicalRequest& request);
    [[nodiscard]] PhysicalCompletion drain_pending(
        std::string id,
        double arrival_ns,
        TraceConfig trace = {});
    void prepopulate_logical_pages(const std::vector<std::uint64_t>& lpns);
    // Dense read-only initial images are common in capacity sweeps. This
    // compact contiguous range form preserves logical translation and the
    // exact FTL physical placement, including partial boundary mapping pages,
    // while avoiding one hash-table entry per data page. Once installed, the
    // device accepts logical reads/drain only.
    void prepopulate_read_only_logical_page_range(
        std::uint64_t first_lpn,
        std::uint64_t page_count);
    // Mutable dense initial images use the same exact FTL placement as
    // prepopulate_logical_pages(), but keep untouched L2P/data state compact.
    // First overwrite and GC relocation materialize only the affected pages.
    void prepopulate_mutable_logical_page_range(
        std::uint64_t first_lpn,
        std::uint64_t page_count);
    // Raw/static data and the managed FTL may never own the same NAND block.
    // Reserve every block touched by these encoded PPNs before issuing work.
    // Repeated reservations are idempotent; intersecting live FTL state fails.
    void reserve_static_physical_pages(const std::vector<std::uint64_t>& ppns);
    // Reserve already-decoded physical block indices without materializing
    // page states. This is used by digest-bound immutable object populations
    // whose untouched pages still consume whole NAND blocks.
    void reserve_static_physical_block_indices(
        const std::vector<std::size_t>& block_indices);
    // Dense static images use the same block coordinate on every fabric
    // plane. Reserve an inclusive source-block extent without materializing
    // one page-table entry per immutable page.
    void reserve_static_physical_block_extent(
        std::uint32_t first_block,
        std::uint32_t block_count);
    // Aggregate, mapping, and parallelism stats are recomputed lazily here:
    // refreshing them on every issue() scans live/pending mappings and every
    // subarray/lane/bank, which dominates large runs.
    [[nodiscard]] const HbfStats& stats() const {
        refresh_parallel_stats();
        return stats_;
    }
    [[nodiscard]] const HbfConfig& config() const { return config_; }
    [[nodiscard]] HbfAuditSnapshot audit_snapshot() const;
    void attach_address_heatmap(AddressHeatmap& address_heatmap) {
        if (last_issue_arrival_ns_) {
            throw std::runtime_error(
                "HBF address heatmap must be attached before the first request");
        }
        address_heatmap_ = &address_heatmap;
    }
    // Which stack owns a logical page (shared-nothing FTL partition); the
    // composition layer needs this to route per-stack base-die-link traffic.
    [[nodiscard]] std::size_t stack_for_logical_page(std::uint64_t lpn) const {
        return stack_for_lpn(lpn);
    }

private:
    enum class TransactionSource {
        User,
        Mapping,
        GC,
        Prepopulate,
    };

    enum class TransactionKind {
        Read,
        Program,
        Erase,
    };

    enum class MappingAccessKind {
        Lookup,
        Update,
    };

    // A user read returns decoded payload over HBIO. Internal controller reads
    // (GC and read-modify-write) end in logic-die SRAM instead.
    enum class ReadPayloadRoute {
        External,
        Internal,
    };

    enum class BlockRole {
        Free,
        StaticReadOnly,
        RawPhysical,
        Data,
        Mapping,
        GC,
    };

    enum class PageStatus {
        Erased,
        StaticReadOnly,
        Valid,
        Invalid,
    };

    enum class PageOwner {
        Unassigned,
        Logical,
        Mapping,
        RawPhysical,
        StaticReadOnly,
    };

    // Exact serial-resource reservation calendar. Besides the busy-until
    // frontier it retains every still-usable idle interval behind that
    // frontier. The augmented treap indexes each subtree's largest gap, so
    // out-of-call-order reservations can find the earliest fitting interval
    // without a linear scan and without discarding physical capacity.
    struct ResourceTimeline {
        struct Gap {
            double begin_ns = 0.0;
            double end_ns = 0.0;
        };

        struct GapNode {
            Gap gap;
            std::uint64_t priority = 0;
            double max_duration_ns = 0.0;
            std::unique_ptr<GapNode> left;
            std::unique_ptr<GapNode> right;
        };

        ResourceTimeline() = default;
        ResourceTimeline(ResourceTimeline&&) noexcept = default;
        ResourceTimeline& operator=(ResourceTimeline&&) noexcept = default;
        ResourceTimeline(const ResourceTimeline&) = delete;
        ResourceTimeline& operator=(const ResourceTimeline&) = delete;

        double ready_ns = 0.0;
        // Sum of all non-overlapping reservations on this one resource.
        double reserved_work_ns = 0.0;

        [[nodiscard]] std::optional<Gap> first_fitting_gap(
            double earliest_ns,
            double duration_ns) const;
        [[nodiscard]] bool can_reserve_exact(
            double begin_ns,
            double duration_ns) const;
        [[nodiscard]] double preview_start(
            double earliest_ns,
            double duration_ns) const;
        void consume_gap(const Gap& gap, double begin_ns, double end_ns);
        void insert_gap(double begin_ns, double end_ns);
        void prune_before(double causal_watermark_ns);

    private:
        std::unique_ptr<GapNode> gap_root_;
        std::uint64_t priority_state_ = 0x9e3779b97f4a7c15ULL;
        double pruned_through_ns_ = 0.0;

        [[nodiscard]] std::uint64_t next_priority();
        static void refresh(GapNode& node);
        static void split(
            std::unique_ptr<GapNode> root,
            double key,
            std::unique_ptr<GapNode>& lower,
            std::unique_ptr<GapNode>& upper);
        static void insert_node(
            std::unique_ptr<GapNode>& root,
            std::unique_ptr<GapNode> node);
        static void erase_node(std::unique_ptr<GapNode>& root, double key);
        [[nodiscard]] static std::unique_ptr<GapNode> merge(
            std::unique_ptr<GapNode> lower,
            std::unique_ptr<GapNode> upper);
        [[nodiscard]] static const GapNode* predecessor(
            const GapNode* root,
            double key);
        [[nodiscard]] static const GapNode* successor(
            const GapNode* root,
            double key);
        [[nodiscard]] static const GapNode* first_fitting_from(
            const GapNode* root,
            double minimum_begin_ns,
            double duration_ns);
    };

    struct PlaneState {
        struct BusyWindow {
            double begin_ns = 0.0;
            double end_ns = 0.0;
        };
        struct SubarrayState {
            ResourceTimeline timeline;
            double busy_ns = 0.0;
            std::uint64_t read_count = 0;
        };

        struct MediaLaneState {
            ResourceTimeline timeline;
            double busy_ns = 0.0;
            std::uint64_t read_count = 0;
        };

        struct PageBufferBankState {
            ResourceTimeline timeline;
            double busy_ns = 0.0;
            std::uint64_t read_count = 0;
        };

        // Batch-synchronous activation rounds: all senses in a round share
        // [start, start + tR), with one slot per subarray. Each subarray keeps
        // the exact ordered set of future rounds it may still join. Past
        // entries are reclaimed only behind the nondecreasing issue-arrival
        // watermark, where no later transaction can become ready.
        std::vector<std::set<double>> available_sense_rounds_by_subarray;
        double sense_rounds_pruned_through_ns = 0.0;
        ResourceTimeline sense_round_calendar;
        std::vector<BusyWindow> full_plane_windows;

        double media_busy_ns = 0.0;
        // Union of array-busy intervals. Batch reads share one tR window and
        // independent subarrays may overlap; summing per-request tR would
        // produce impossible >100% normalized plane utilization.
        std::vector<BusyWindow> media_busy_windows;
        std::uint64_t read_count = 0;
        std::uint64_t program_count = 0;
        std::uint64_t erase_count = 0;
        std::vector<SubarrayState> subarrays;
        std::vector<MediaLaneState> media_lanes;
        std::vector<PageBufferBankState> page_buffer_banks;
        std::deque<std::size_t> free_blocks;
        std::optional<std::size_t> active_data_block;
        std::optional<std::size_t> active_mapping_block;
        std::optional<std::size_t> active_gc_block;
    };

    struct DieState {
        struct EccInflightInterval {
            double start_ns = 0.0;
            double finish_ns = 0.0;
        };

        ResourceTimeline sequencer;
        // Decode and encode conservatively share one pipelined issue port per
        // die. Only the initiation interval occupies this calendar; response
        // latency may overlap later codewords.
        ResourceTimeline ecc_issue;
        std::vector<EccInflightInterval> ecc_inflight_intervals;
        double ecc_issue_busy_ns = 0.0;
        std::uint64_t ecc_decode_ops = 0;
        std::uint64_t ecc_encode_ops = 0;
        double sequencer_busy_ns = 0.0;
        std::uint64_t transaction_count = 0;
        // Per-source TSU slots issue whichever transaction is ready first
        // (backfill): in-order issue would let one late transaction park
        // the queue while the die sits idle.
        std::array<ResourceTimeline, 4> source_queues{};
    };

    struct ChannelState {
        ResourceTimeline command;
        ResourceTimeline data;
        double command_busy_ns = 0.0;
        double data_busy_ns = 0.0;
        std::uint64_t command_count = 0;
        std::uint64_t data_count = 0;
    };

    struct LogicDieState {
        // Ingress admission accepts requests in ARRIVAL order: a backfilled
        // timeline of issue slots. A bare busy-until cursor ordered requests
        // by CALL order instead — one late-arriving background op (e.g. a
        // staged backing write) parked the cursor in the future and every
        // later-called but earlier-arriving read queued behind it (measured:
        // staged fills waiting >1 ms of phantom ingress queue).
        ResourceTimeline ingress;
        ResourceTimeline hb_io_command;
        ResourceTimeline hb_io_data;
        double hb_io_command_busy_ns = 0.0;
        double hb_io_data_busy_ns = 0.0;
        ResourceTimeline tsv;
        // Every reservation on this finite SRAM calendar must contribute the
        // same duration to Breakdown::sram_staging_ns. stats() enforces that
        // aggregate work-conservation invariant.
        ResourceTimeline sram;
        // Single pipelined controller-DRAM issue port for the stack-resident
        // L2P table. Only issue_ns occupies the port; response latency overlaps
        // later accesses.
        ResourceTimeline mapping_dram_issue;
        // Logic-die read buffer: PPN-keyed LRU of pages whose data sits in
        // base-die SRAM (decoded); a hit never touches the flash array.
        struct ReadBufferEntry {
            double ready_ns = 0.0;
            std::uint64_t block_epoch = 0;
            std::set<std::pair<double, std::uint64_t>> touches;
        };
        std::unordered_map<std::uint64_t, ReadBufferEntry> read_buffer;
        // Exact auxiliary indexes for read-buffer capacity enforcement.  The
        // prefix contains the smallest capacity+1 ready times, which makes
        // "more than capacity entries are visible at t" an O(1) boundary
        // test after O(log n) updates.  The touch index provides the exact
        // temporal LRU victim without rescanning every cache line.
        using ReadBufferReadyKey = std::pair<double, std::uint64_t>;
        using ReadBufferTouch = std::pair<double, std::uint64_t>;
        using ReadBufferLruKey =
            std::pair<ReadBufferTouch, std::uint64_t>;
        std::set<ReadBufferReadyKey> read_buffer_ready_prefix;
        std::set<ReadBufferReadyKey> read_buffer_ready_suffix;
        std::set<ReadBufferLruKey> read_buffer_lru;
    };

    struct PageState {
        PageStatus status = PageStatus::Erased;
        PageOwner owner = PageOwner::Unassigned;
        std::uint64_t lpn = 0;
        std::uint64_t block_epoch = 0;
    };

    struct BlockState {
        BlockRole role = BlockRole::Free;
        std::uint32_t valid_pages = 0;
        std::uint32_t invalid_pages = 0;
        std::uint32_t free_pages = 0;
        std::uint32_t next_page = 0;
        std::uint32_t erase_count = 0;
        // GC may reclaim only fully committed blocks. Allocated-but-not-yet
        // programmed pages and programmed-but-not-yet-published mappings are
        // explicit ownership pins, not invalid/free space.
        std::uint32_t pending_program_pages = 0;
        std::uint32_t pending_mapping_publications = 0;
        // Completion of the latest read/program already issued to this block.
        // A later destructive erase may backfill around other blocks, but not
        // ahead of an earlier access to its own target.
        double issued_media_ready_ns = 0.0;
        // Monotonic incarnation of this physical block. Every page
        // reservation and delayed mapping publication captures it; an erase
        // increments the epoch so stale callbacks cannot resurrect data.
        std::uint64_t epoch = 0;
        bool erase_pending = false;
        // Allocate the 1024-bit validity map only after a block receives live
        // data. Multi-million-block profiles otherwise spent ~512 MiB on
        // bitmaps for erased blocks before the first request.
        std::unique_ptr<std::array<std::uint64_t, 16>> valid_bitmap;

        void set_valid(std::uint32_t page) {
            if (!valid_bitmap) {
                valid_bitmap = std::make_unique<std::array<std::uint64_t, 16>>();
                valid_bitmap->fill(0);
            }
            (*valid_bitmap)[page >> 6] |= 1ull << (page & 63);
        }
        void clear_valid(std::uint32_t page) {
            if (valid_bitmap) {
                (*valid_bitmap)[page >> 6] &= ~(1ull << (page & 63));
            }
        }
        [[nodiscard]] bool is_valid(std::uint32_t page) const {
            return valid_bitmap && (((*valid_bitmap)[page >> 6] >> (page & 63)) & 1);
        }
        void set_valid_range(std::uint32_t first_page, std::uint32_t count) {
            if (count == 0 || first_page + count > 1024) {
                throw std::runtime_error("invalid HBF compact valid-page range");
            }
            if (!valid_bitmap) {
                valid_bitmap = std::make_unique<std::array<std::uint64_t, 16>>();
                valid_bitmap->fill(0);
            }
            const auto end = first_page + count;
            auto page = first_page;
            while (page < end) {
                const auto word = page >> 6;
                const auto word_end = std::min<std::uint32_t>(
                    end,
                    (word + 1) << 6);
                const auto width = word_end - page;
                const auto low = page & 63;
                const auto mask = width == 64 ?
                    std::numeric_limits<std::uint64_t>::max() :
                    ((std::uint64_t{1} << width) - 1) << low;
                (*valid_bitmap)[word] |= mask;
                page = word_end;
            }
        }
    };

    struct CompactLogicalImage {
        struct VpnRange {
            // Mapping pages are stack-local: entry i covers the i-th page
            // assigned to this stack, not one contiguous run of global LPNs.
            std::uint64_t first_entry = 0;
            std::uint64_t page_count = 0;
            // Number of earlier compact-image data pages allocated on this
            // VPN's stack. Together with the in-range LPN offset, this recovers
            // the ordinary round-robin data-allocation plane and page ordinal.
            std::uint64_t stack_page_offset = 0;
        };

        struct DataBlockLocation {
            std::size_t plane = 0;
            std::uint64_t block_ordinal = 0;
        };

        bool mutable_image = false;
        std::uint64_t first_lpn = 0;
        std::uint64_t page_count = 0;
        std::uint64_t first_vpn = 0;
        // Consecutive directory slots covering all stacks in every touched
        // mapping group. A partial first/last group can leave inactive slots.
        std::uint64_t vpn_slot_count = 0;
        std::uint64_t mapping_page_count = 0;
        // Per physical plane, data blocks in the order that the normal FTL
        // allocator assigned them. Page ordinal within a plane selects one
        // block/page without materializing an L2P entry.
        std::vector<std::vector<std::uint64_t>> data_blocks_by_plane;
        // One optional mapping-page PPN per directory slot. This is small
        // (one entry per 2 MiB with the published 512-entry mapping page and
        // 4 KiB data pages); inactive boundary slots remain empty.
        std::vector<std::optional<std::uint64_t>> mapping_ppns;
        // One compact descriptor per directory slot. Boundary slots may cover
        // only a suffix or prefix of their stack-local mapping page.
        std::vector<VpnRange> vpn_ranges;
        // Sparse reverse indexes let GC recover the logical owner of a compact
        // valid page without expanding one entry per data page.
        std::unordered_map<std::uint64_t, DataBlockLocation>
            data_block_locations;
        std::vector<std::vector<std::uint64_t>> vpn_offsets_by_stack;
        std::unordered_map<std::uint64_t, std::uint64_t>
            mapping_vpn_by_ppn;
        // Live compact pages remain represented only by block bitmaps and
        // these per-block counts. Retired pages become explicit invalid
        // PageState tombstones until their block is erased.
        std::unordered_map<std::uint64_t, std::uint32_t>
            live_data_pages_by_block;
        std::unordered_map<std::uint64_t, std::uint32_t>
            live_mapping_pages_by_block;
        std::unordered_set<std::uint64_t> retired_lpns;
        std::unordered_set<std::uint64_t> retired_mapping_vpns;
    };

    struct CompactPageIdentity {
        std::uint64_t logical_key = 0;
        PageOwner owner = PageOwner::Unassigned;
    };

    struct GcHeadroom {
        // Pages available to relocation: every whole free block (including
        // the foreground reserve) plus unused pages in GC's active blocks.
        std::uint64_t relocation_pages = 0;
        // Pages the requested foreground role can allocate without borrowing
        // GC's per-plane reserve. Active capacity belongs only to that role.
        std::uint64_t foreground_pages = 0;
        std::uint64_t reserve_target_pages = 0;
    };

    struct ScheduledTransfer {
        double start_ns = 0.0;
        double finish_ns = 0.0;
        double wait_ns = 0.0;
    };

    struct SenseRoundCandidate {
        double start_ns = 0.0;
        bool joins_existing_round = false;
    };


    struct WriteBufferEntry {
        std::uint64_t lpn = 0;
        std::vector<DirtyRange> ranges;
        std::list<std::uint64_t>::iterator iterator;
        // Merged data is readable only after the SRAM stage completes.
        double ready_ns = 0.0;
    };

    struct InflightBufferedWrite {
        std::uint64_t generation = 0;
        std::vector<DirtyRange> ranges;
        double ready_ns = 0.0;
        double commit_ns = 0.0;
        std::uint64_t target_ppn = 0;
        std::uint64_t target_block_epoch = 0;
    };

    struct PendingMappingUpdate {
        std::uint64_t sequence = 0;
        std::uint64_t program_commit_sequence = 0;
        double commit_ns = 0.0;
        std::uint64_t new_ppn = 0;
        std::uint64_t block_epoch = 0;
        // A destructive raw erase can retire the target before this mapping
        // publishes. Keep the record as an ordering tombstone until both the
        // mapping callback and that erase have completed.
        double destructive_ready_ns = 0.0;
    };

    struct PendingCommit {
        std::size_t stack = 0;
        std::function<void()> action;
    };

    struct PendingPhysicalProgram {
        double commit_ns = 0.0;
        std::uint64_t commit_sequence = 0;
    };

    struct PendingPhysicalErase {
        double finish_ns = 0.0;
        std::uint64_t commit_sequence = 0;
    };

    HbfConfig config_;
    AddressHeatmap* address_heatmap_ = nullptr;
    std::uint32_t subarrays_per_plane_ = 0;
    std::vector<ChannelState> channels_;
    std::vector<DieState> dies_;
    std::vector<PlaneState> planes_;
    std::vector<LogicDieState> logic_dies_;
    std::vector<BlockState> blocks_;
    std::unordered_map<std::uint64_t, std::uint64_t> lpn_to_ppn_;
    std::unordered_map<std::uint64_t, std::uint64_t> mapping_vpn_to_ppn_;
    std::unordered_map<std::uint64_t, PageState> programmed_pages_;
    std::optional<CompactLogicalImage> compact_logical_image_;
    std::unordered_set<std::uint64_t> dirty_mapping_vpns_;
    std::unordered_map<
        std::uint64_t,
        std::set<std::pair<double, std::uint64_t>>> pending_dirty_mapping_events_;
    std::unordered_map<std::uint64_t, std::vector<PendingMappingUpdate>> pending_lpn_updates_;
    std::unordered_map<std::uint64_t, std::vector<PendingMappingUpdate>> pending_vpn_updates_;
    // Raw physical programs reserve their page immediately (like the FTL
    // allocator) but become valid only at media completion. This table keeps
    // the in-flight ownership explicit until that commit fires.
    std::unordered_map<std::uint64_t, PendingPhysicalProgram>
        pending_physical_programs_;
    std::unordered_map<std::size_t, PendingPhysicalErase>
        pending_physical_erases_;
    // A targeted dependency may materialize one object's future commit
    // without advancing unrelated blocks/planes. These object-scoped
    // watermarks prevent a later call from using that state in its past.
    std::vector<double> materialized_ready_by_block_;
    std::unordered_map<std::uint64_t, double> materialized_ready_by_lpn_;
    std::unordered_map<std::uint64_t, double> materialized_ready_by_vpn_;
    std::unordered_map<std::uint64_t, double> materialized_ready_by_ppn_;
    std::unordered_map<std::uint64_t, std::vector<InflightBufferedWrite>> inflight_buffered_writes_;
    // Write buffer is logic-die SRAM: one independent buffer per stack.
    // write_buffer_pages / write_buffer_flush_threshold_pages apply PER die.
    std::vector<std::list<std::uint64_t>> write_buffer_lru_by_stack_;
    std::vector<std::unordered_map<std::uint64_t, WriteBufferEntry>> write_buffer_by_stack_;
    // SRAM slot lifetime: a flushed entry's slot stays occupied until its
    // logical mapping publishes (release instants below, one per in-flight
    // generation). The old generation remains readable until that same
    // deadline; releasing at media program completion created two owners for
    // one SRAM slot and a false buffer hit.
    std::vector<std::multiset<double>> write_buffer_slot_release_by_stack_;
    // One completion frontier per allocated page-read credit. Once all
    // credits are allocated, the earliest finishing transaction releases the
    // next credit. Top-level requests are required to arrive in timestamp
    // order, so this completion-order window is deterministic and exact.
    std::vector<std::multiset<double>> page_read_credit_release_by_stack_;
    // Shared-nothing FTL partition: each stack owns its planes, free pages,
    // allocation cursor, and GC; nothing crosses a stack boundary.
    // Data, Mapping, and GC have independent placement streams. Maintenance
    // fallback must not perturb the next foreground stripe decision.
    std::vector<std::size_t> next_data_allocation_plane_per_stack_;
    std::vector<std::size_t> next_mapping_allocation_plane_per_stack_;
    std::vector<std::size_t> next_gc_allocation_plane_per_stack_;
    std::uint64_t total_pages_ = 0;
    std::uint64_t free_pages_ = 0;
    std::vector<std::uint64_t> free_pages_per_stack_;
    std::uint64_t resident_mapping_pages_per_stack_ = 0;
    std::uint64_t resident_mapping_bytes_per_stack_ = 0;
    std::map<std::pair<double, std::uint64_t>, PendingCommit> pending_commits_;
    std::uint64_t next_commit_sequence_ = 0;
    std::uint64_t next_buffer_generation_ = 0;
    std::uint64_t next_cache_touch_sequence_ = 0;
    std::optional<double> last_issue_arrival_ns_;
    // Lower bound on the ready time of every reservation created from the
    // current or any future top-level issue/drain call. Exact calendars may
    // reclaim idle intervals only before this causal watermark.
    double reservation_causal_watermark_ns_ = 0.0;
    // Some foreground operations (notably GC at allocation exhaustion) must
    // advance completed state before accepting more work. Later nominal
    // arrivals queue behind this watermark instead of observing future state.
    std::vector<double> causal_state_ready_by_stack_;
    // State time already made causal for the active top-level operation.
    // Temporal read-buffer state may be materialized through this instant,
    // never through arbitrary future work scheduled inside the operation.
    std::vector<double> state_observation_by_stack_;
    // GC relocation is non-reentrant per stack so a mapping-checkpoint flush
    // cannot recursively select the same half-moved victim.
    std::vector<bool> gc_active_by_stack_;
    double background_finish_ns_ = 0.0;
    mutable HbfStats stats_;

    [[nodiscard]] std::list<std::uint64_t>& write_buffer_lru(std::size_t stack);
    [[nodiscard]] std::unordered_map<std::uint64_t, WriteBufferEntry>& write_buffer(std::size_t stack);
    [[nodiscard]] SenseRoundCandidate preview_sense_round(
        PlaneState& plane,
        std::size_t subarray_index,
        double ready_ns);
    [[nodiscard]] ScheduledTransfer commit_sense_round(
        PlaneState& plane,
        std::size_t subarray_index,
        double ready_ns,
        const SenseRoundCandidate& candidate);
    [[nodiscard]] double preview_full_plane_window(
        PlaneState& plane,
        double earliest_ns,
        double duration_ns);
    void record_full_plane_window(
        PlaneState& plane,
        double begin_ns,
        double end_ns);
    void record_plane_media_busy(PlaneState& plane, double begin_ns, double end_ns);
    // Read-buffer state lives on the logic die of the stack that owns the
    // PPN (derived internally): keying by the request's ingress die would
    // strand entries that the owning block's erase can never purge.
    [[nodiscard]] bool read_buffer_contains(std::uint64_t ppn, double at_ns);
    void read_buffer_insert(std::uint64_t ppn, double ready_ns);
    void enforce_read_buffer_capacity(LogicDieState& logic_die, double at_ns);
    void rebalance_read_buffer_ready_index(LogicDieState& logic_die);
    void index_read_buffer_entry(
        LogicDieState& logic_die,
        std::uint64_t ppn);
    void erase_read_buffer_entry(
        LogicDieState& logic_die,
        std::uint64_t ppn);
    void touch_read_buffer_entry(
        LogicDieState& logic_die,
        std::uint64_t ppn,
        double at_ns);
    [[nodiscard]] bool read_buffer_capacity_exceeded_at(
        const LogicDieState& logic_die,
        double at_ns) const;
    void read_buffer_purge_page(std::uint64_t ppn);
    void read_buffer_purge_block(std::size_t block_index);
    [[nodiscard]] double serve_read_from_read_buffer(
        std::uint64_t ppn,
        std::uint64_t bytes,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        bool external_egress);
    [[nodiscard]] std::size_t stack_for_lpn(std::uint64_t lpn) const;
    [[nodiscard]] std::size_t stack_for_vpn(std::uint64_t mapping_vpn) const;
    [[nodiscard]] std::size_t planes_per_stack() const;
    [[nodiscard]] std::size_t stack_of_plane(std::size_t plane) const;
    [[nodiscard]] std::size_t stack_of_block(std::size_t block_index) const;
    [[nodiscard]] std::size_t mapping_plane_for_vpn(std::uint64_t mapping_vpn) const;
    [[nodiscard]] std::size_t stack_index(const HbfAddress& addr) const;
    [[nodiscard]] std::size_t channel_index(const HbfAddress& addr) const;
    [[nodiscard]] std::size_t die_index(const HbfAddress& addr) const;
    [[nodiscard]] std::size_t plane_index(const HbfAddress& addr) const;
    [[nodiscard]] std::size_t block_index(const HbfAddress& addr) const;
    [[nodiscard]] std::size_t block_plane_index(std::size_t block_index) const;
    void reserve_static_physical_blocks(
        const std::vector<std::size_t>& block_indices);
    [[nodiscard]] std::size_t subarray_index(const HbfAddress& addr) const;
    [[nodiscard]] std::size_t media_lane_index(const HbfAddress& addr) const;
    [[nodiscard]] std::size_t page_buffer_bank_index(const HbfAddress& addr) const;
    [[nodiscard]] std::uint64_t page_count_for(const HbfAddress& addr, std::uint64_t bytes) const;
    [[nodiscard]] HbfAddress decode_ppn(std::uint64_t ppn) const;
    [[nodiscard]] std::uint64_t encode_ppn(const HbfAddress& addr) const;
    [[nodiscard]] HbfAddress logical_page_address(std::uint64_t logical_byte_addr) const;
    [[nodiscard]] std::uint64_t mapping_vpn_for_lpn(std::uint64_t lpn) const;
    [[nodiscard]] std::optional<std::uint64_t> compact_lpn_ppn(
        std::uint64_t lpn) const;
    [[nodiscard]] std::optional<std::uint64_t> compact_mapping_ppn(
        std::uint64_t mapping_vpn) const;
    [[nodiscard]] std::optional<std::uint64_t> compact_lpn_for_ppn(
        std::uint64_t ppn) const;
    [[nodiscard]] std::optional<CompactPageIdentity>
        compact_page_identity(std::uint64_t ppn) const;
    void retire_compact_page(
        std::uint64_t logical_key,
        PageOwner owner);
    void prepopulate_compact_logical_page_range(
        std::uint64_t first_lpn,
        std::uint64_t page_count,
        bool mutable_image);
    [[nodiscard]] std::uint64_t logical_mapping_entry_count() const;
    [[nodiscard]] std::size_t source_index(TransactionSource source) const;
    [[nodiscard]] std::string source_name(TransactionSource source) const;
    [[nodiscard]] HeatmapTrafficSource resolve_heatmap_source(
        TransactionSource source,
        HeatmapTrafficSource attribution) const;
    [[nodiscard]] std::string kind_name(TransactionKind kind) const;
    // Bytes a raw flash-side page operation moves (data + OOB).
    [[nodiscard]] std::uint64_t page_wire_bytes() const {
        return config_.page_size_bytes + config_.oob_bytes_per_page;
    }
    [[nodiscard]] std::string role_name(BlockRole role) const;
    void flush_all_dirty_mapping_pages(
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    void mark_mapping_page_dirty(std::uint64_t mapping_vpn, double at_ns);
    void access_resident_mapping(
        std::uint64_t lpn,
        TransactionSource source,
        MappingAccessKind kind,
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    void flush_dirty_mapping_page(
        std::uint64_t mapping_vpn,
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    [[nodiscard]] std::optional<std::uint64_t> lookup_lpn(
        std::uint64_t lpn,
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    [[nodiscard]] std::optional<std::uint64_t> visible_lpn_at(
        std::uint64_t lpn,
        double at_ns) const;
    void wait_for_prior_lpn_commit(
        std::uint64_t lpn,
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    void wait_for_pending_vpn_erase(
        std::uint64_t mapping_vpn,
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    void wait_for_pending_block_erase(
        std::uint64_t ppn,
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    void tag_pending_mapping_updates_for_erase(
        std::size_t block_index,
        std::uint64_t retired_block_epoch,
        double ready_ns);
    void retire_mapping_update_tombstones(
        std::size_t block_index,
        double ready_ns);
    [[nodiscard]] std::optional<std::uint64_t> visible_mapping_vpn_at(
        std::uint64_t mapping_vpn,
        double at_ns) const;
    void schedule_lpn_mapping_commit(
        std::uint64_t lpn,
        std::uint64_t new_ppn,
        double commit_ns,
        std::uint64_t program_commit_sequence);
    void schedule_vpn_mapping_commit(
        std::uint64_t mapping_vpn,
        std::uint64_t new_ppn,
        double commit_ns,
        std::uint64_t program_commit_sequence);
    void schedule_physical_program_commit(
        std::uint64_t ppn,
        double commit_ns);
    void reserve_physical_program_range(
        std::uint64_t first_ppn,
        std::uint64_t pages);
    [[nodiscard]] std::uint64_t schedule_commit(
        std::size_t stack,
        double at_ns,
        std::function<void()> action);
    void apply_selected_commits_through(
        const std::unordered_set<std::uint64_t>& sequences,
        double at_ns);
    void apply_commits_through(double at_ns);
    void apply_stack_commits_through(std::size_t stack, double at_ns);
    void apply_all_commits();
    [[nodiscard]] bool advance_to_next_commit(double& at_ns, std::size_t stack);
    [[nodiscard]] bool write_buffer_covers(
        const WriteBufferEntry& entry,
        std::uint64_t begin,
        std::uint64_t end) const;
    [[nodiscard]] bool dirty_ranges_cover(
        const std::vector<DirtyRange>& ranges,
        std::uint64_t begin,
        std::uint64_t end) const;
    [[nodiscard]] bool write_buffer_full_page(const WriteBufferEntry& entry) const;
    [[nodiscard]] std::uint64_t write_buffer_covered_bytes(const WriteBufferEntry& entry) const;
    [[nodiscard]] double serve_read_from_write_buffer(
        std::uint64_t lpn,
        std::uint64_t begin,
        std::uint64_t end,
        const std::vector<DirtyRange>& ranges,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    [[nodiscard]] double admit_foreground_page_read(
        std::size_t stack,
        double offered_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    void complete_foreground_page_read(
        std::size_t stack,
        double finish_ns);
    [[nodiscard]] double schedule_external_write_ingress(
        std::size_t stack,
        std::uint64_t bytes,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        const std::string& detail,
        const std::string& sram_span_name);
    [[nodiscard]] double schedule_external_request_command(
        std::size_t stack,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        const std::string& detail);
    [[nodiscard]] double schedule_sram_transfer(
        std::size_t stack,
        std::uint64_t bytes,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        const std::string& name,
        const std::string& detail);
    [[nodiscard]] double schedule_external_read_egress(
        std::size_t stack,
        std::uint64_t bytes,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        const std::string& name,
        const std::string& detail);
    // Write-buffer staging and flush act on the LPN's owning stack (derived
    // internally, like the read buffer above).
    [[nodiscard]] double stage_write_buffer_range(
        std::uint64_t lpn,
        std::uint64_t begin,
        std::uint64_t end,
        HeatmapTrafficSource heatmap_source,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    void flush_write_buffer_entry(
        std::uint64_t lpn,
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    void flush_all_write_buffer_entries(
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    [[nodiscard]] std::uint64_t allocate_free_page(
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        BlockRole role,
        std::size_t stack,
        std::optional<std::size_t> preferred_plane = std::nullopt);
    [[nodiscard]] std::size_t choose_allocation_plane(
        std::size_t stack,
        BlockRole role);
    [[nodiscard]] std::optional<std::size_t> preview_allocation_plane(
        std::size_t stack,
        BlockRole role,
        std::optional<std::size_t> preferred_plane = std::nullopt) const;
    [[nodiscard]] std::size_t allocate_block_from_plane(std::size_t plane_index, BlockRole role);
    [[nodiscard]] std::optional<std::size_t>& active_block_ref(PlaneState& plane, BlockRole role);
    [[nodiscard]] std::uint64_t allocate_page_from_block(std::size_t block_index);
    [[nodiscard]] std::uint64_t allocate_compact_pages_on_plane(
        std::size_t plane_index,
        BlockRole role,
        std::uint32_t page_count,
        std::vector<std::uint64_t>* assigned_blocks = nullptr,
        std::unordered_map<std::uint64_t, std::uint32_t>*
            live_pages_by_block = nullptr);
    [[nodiscard]] std::optional<std::size_t> choose_gc_victim(std::size_t stack) const;
    [[nodiscard]] std::uint64_t gc_relocation_capacity(std::size_t stack) const;
    [[nodiscard]] GcHeadroom gc_headroom(
        std::size_t stack,
        BlockRole allocation_role) const;
    void maybe_run_gc(
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        std::uint64_t required_pages,
        std::size_t stack,
        BlockRole allocation_role,
        std::optional<std::size_t> preferred_plane = std::nullopt);
    void relocate_and_erase_block(
        std::size_t block_index,
        double& at_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans);
    void invalidate_ppn(std::uint64_t ppn);
    void mark_programmed(
        std::uint64_t ppn,
        std::uint64_t lpn,
        PageOwner owner = PageOwner::Logical);
    [[nodiscard]] std::uint64_t schedule_media_program_commit(
        std::uint64_t ppn,
        std::uint64_t lpn,
        PageOwner owner,
        double commit_ns);
    void reset_erased_block(std::size_t block_index);
    void refresh_accounting_stats() const;
    [[nodiscard]] ScheduledTransfer reserve(
        double earliest_ns,
        double duration_ns,
        ResourceTimeline& timeline);
    [[nodiscard]] ScheduledTransfer schedule_flash_transaction(
        const HbfAddress& addr,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        TransactionSource source,
        TransactionKind kind);
    [[nodiscard]] double schedule_command_path(
        const HbfAddress& addr,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        TransactionSource source);
    [[nodiscard]] double schedule_ecc(
        const HbfAddress& addr,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        TransactionSource source,
        bool decode);
    [[nodiscard]] double schedule_read_page(
        std::uint64_t ppn,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        TransactionSource source,
        HeatmapTrafficSource heatmap_attribution,
        ReadPayloadRoute route,
        std::uint64_t external_payload_bytes,
        double* decoded_ready_ns = nullptr);
    [[nodiscard]] double schedule_program_page(
        std::uint64_t ppn,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        TransactionSource source,
        HeatmapTrafficSource heatmap_attribution);
    [[nodiscard]] double schedule_erase_block(
        std::size_t block_index,
        double earliest_ns,
        Breakdown& breakdown,
        std::vector<TraceSpan>* spans,
        TransactionSource source,
        HeatmapTrafficSource heatmap_attribution);
    void refresh_parallel_stats() const;
};

} // namespace hbfsim::physical::hbf
