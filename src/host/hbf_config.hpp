#pragma once
#include "physical/hbf/hbf_config.hpp"

namespace hbfsim::host {

// FullResident is the conventional baseline: every page-level L2P entry is
// provisioned in host DRAM partitions. Cached keeps the authoritative
// mapping pages on HBF media and uses the same DRAM as a bounded page cache.
// Static physical HBF transactions bypass either mode and provide the extent
// path used for explicitly immutable objects.
enum class MappingMode {
    FullResident,
    Cached,
    // No FTL metadata or implicit logical access. The OCP channel frontend
    // owns host-address decoding; raw programs still obey NAND page order.
    RawPhysical,
};

[[nodiscard]] const char* to_string(MappingMode mode);
[[nodiscard]] MappingMode parse_mapping_mode(std::string_view value);

// Volatile representations of the same persisted page-L2P. Extent caches
// compress clean translation pages into affine runs; writes promote them to
// dense pages. This is not a different NAND mapping-page format.
enum class MappingCacheLayout { Page, Entry, Extent };
[[nodiscard]] const char* to_string(MappingCacheLayout layout);
[[nodiscard]] MappingCacheLayout parse_mapping_cache_layout(std::string_view value);

// Host is the implemented OCP-facing organization. DeviceLocal is a mapping-
// only architectural control: it places the SAME mapping policy and budget
// beside the flash and keeps metadata payload off HBIO. It is not an OCP FTL.
enum class MappingPlacement { Host, DeviceLocal };

// Page uses the resident/page-cache/entry-cache translation engine. Other
// organizations own placement and reclamation, over the same physical media.
enum class MappingOrganization { Page, Block, BlockLog, Extent, ObjectSegment };
[[nodiscard]] const char* to_string(MappingOrganization organization);
[[nodiscard]] MappingOrganization parse_mapping_organization(std::string_view value);

// Replacement order of the HBF device DRAM's clean read cache.
enum class DeviceDramPolicy { Lru, Fifo };
[[nodiscard]] const char* to_string(DeviceDramPolicy policy);
[[nodiscard]] DeviceDramPolicy parse_device_dram_policy(std::string_view value);

struct HbfHostConfig {
    MappingOrganization mapping_organization = MappingOrganization::Page;
    std::uint32_t mapping_log_blocks = 4;
    // Structural organizations map, allocate, program and reclaim superblocks:
    // one block at the same offset on this many planes, with consecutive pages
    // striped across them. 1 keeps single-plane blocks. It must divide the
    // plane count; the page FTL stripes pages itself and requires 1.
    std::uint32_t mapping_superblock_planes = 1;
    // FullResident keeps the complete L2P table in controller DRAM. Cached
    // uses a bounded, stack-partitioned LRU cache with the selected volatile
    // layout. Its misses read persistent mapping pages from HBF media.
    MappingMode mapping_mode = MappingMode::FullResident;
    MappingCacheLayout mapping_cache_layout = MappingCacheLayout::Page;
    MappingPlacement mapping_placement = MappingPlacement::Host;
    // Compute is distinct from the pipelined metadata memory port. Zero worker
    // count derives one worker per HBF stack. Host workers are a shared pool;
    // the device-local control partitions the same total count across stacks.
    std::uint32_t mapping_compute_workers = 0;
    // False is a research ablation: pin host workers to stacks so resource
    // pooling is not confused with the execution-placement effect.
    bool mapping_host_compute_shared = true;
    double mapping_control_compute_ns = 5.0;
    double mapping_internal_compute_ns = 0.0;
    // Assumed controller-attached DRAM for the device-local counterfactual.
    // 51.2 GB/s is the aggregate payload ceiling of a 64-bit 6400-MT/s
    // interface; 64-byte transactions include overfetch of short entries.
    // Latency/issue use ctrl_dram_*. This is not a calibrated DRAM device.
    double local_mapping_bandwidth_GBps = 51.2;
    std::uint64_t local_mapping_burst_bytes = 64;
    // Optional read-only L1 over a complete host-HBM L2P. The total SRAM
    // budget is separate from ctrl_dram_bytes and split equally by stack.
    // Each 32-byte record includes the LPN tag, PPN and replacement metadata.
    // Updates always reach HBM and invalidate L1; no dirty SRAM state exists.
    // These are research-probe parameters, not calibrated SRAM timings.
    std::uint64_t mapping_sram_bytes = 0;
    double mapping_sram_latency_ns = 5.0;
    double mapping_sram_issue_ns = 1.0;
    // Explicit hardware assumptions. Zero tag bytes keeps the existing
    // payload-only budget; experiments comparing layouts provision real tags.
    std::uint64_t mapping_cache_tag_bytes = 0;
    double mapping_codec_ns_per_entry = 0.0;
    std::uint64_t mapping_scratch_pages = 0;
    // Accesses are pipelined: issue_ns occupies the single modeled port while
    // latency_ns is the overlappable response latency.
    double ctrl_dram_latency_ns = 100.0;
    double ctrl_dram_issue_ns = 1.0;
    double mapping_update_ns = 25.0;
    // Host/xPU software assumptions, not OCP timing requirements. GC copies
    // cross HBIO in both directions and stage in reserved HBM. Buffer data
    // transfers share the system HBM pseudo-channel buses. The access latency
    // and issue interval above are controller assumptions, not JEDEC timings.
    double host_gc_decision_ns = 50.0;
    // Equal, NAND-block-aligned zones within each modeled host channel.
    std::uint32_t zone_size_blocks = 1;
    double host_zone_remap_ns = 100.0;
    std::uint32_t host_zone_wear_gap = 1;
    double free_page_allocation_ns = 10.0;
    // HBM reservation, divided evenly across HBF stacks and deducted from the
    // system's application HBM capacity. Mapping
    // metadata and the optional data write buffer share this one pool. Zero
    // derives exactly enough capacity for FullResident plus the configured
    // write buffer. Cached requires an explicit budget for the complete
    // mapping-page directory, the configured write buffer, and at least one
    // mapping-cache page per stack. With HBF device DRAM enabled the write
    // buffer lives there instead and this pool holds no write buffer.
    std::uint64_t ctrl_dram_bytes = 0;
    // Optional exact physical-capacity ratio used to derive ctrl_dram_bytes.
    // A value D allocates the largest whole-page, equal per-stack budget no
    // larger than raw HBF capacity / D. Zero disables ratio derivation. A
    // resolved config may contain both this provenance field and the derived
    // ctrl_dram_bytes; callers must not supply an unrelated explicit byte
    // budget at the same time.
    std::uint64_t ctrl_dram_capacity_denominator = 0;
    // Number of L2P entries represented by one persistent mapping checkpoint
    // page. It also determines the resident entry width:
    // page_size_bytes / mapping_entries_per_page.
    std::uint64_t mapping_entries_per_page = 512;
    // Cached mapping retains one controller-resident directory entry per
    // persistent mapping page. The conservative 64-bit entry names its PPN or
    // an unmapped sentinel, is charged inside ctrl_dram_bytes, and consumes
    // one ordinary controller-DRAM pipeline access on each cache miss.
    std::uint64_t mapping_directory_entry_bytes = 8;
    // Page-granular, per-stack data cache in controller DRAM. The capacity is
    // charged against ctrl_dram_bytes whenever coalescing is enabled; its
    // accesses use the controller latency/issue resources plus shared HBM data
    // channels. The channel-level buffer model does not claim DRAM row locality.
    // With HBF device DRAM enabled (below) the buffer lives there instead and
    // uses the device DRAM port.
    bool write_coalescing_enabled = false;
    bool write_buffer_completion_requires_flush = false;
    std::uint64_t write_buffer_pages = 1024;
    std::uint64_t write_buffer_flush_threshold_pages = 0;
    // On-device DRAM of every HBF stack, as in a NAND SSD: not part of HBM
    // and not part of ctrl_dram_bytes. A denominator N > 0 gives each stack
    // floor(raw flash pages per stack / N) pages. The DRAM holds the write
    // buffer (write_buffer_pages per stack when coalescing is enabled; a
    // write completes when it enters the DRAM unless completion requires the
    // flush) and uses the rest as a clean, physically addressed read cache
    // of NAND pages. Its port charges hit latency plus bytes / bandwidth.
    // Zero disables it: the write buffer then lives in the HBM reservation.
    std::uint64_t device_dram_capacity_denominator = 0;
    double device_dram_hit_latency_ns = 90.0;
    double device_dram_bandwidth_GBps = 204.8;
    DeviceDramPolicy device_dram_policy = DeviceDramPolicy::Lru;
    // Snapshot precondition (default off, OCP semantics): free blocks of the
    // installed image that were never programmed are already erased, as in a
    // controller's pre-erased free pool, so their first page-zero program
    // needs no erase and no P/E cycle. Reclaimed blocks still auto-erase.
    // Lifetime studies keep it off.
    bool initial_free_blocks_erased = false;
    // Logical capacity exposed to the host (whole pages). LPNs at or beyond
    // it fail closed in every request and prepopulation. 0 derives the
    // largest value the FTL can serve without ever wedging: per stack,
    //   logical_pages + mapping_pages(logical_pages)
    //       <= (usable_blocks - open_blocks) * pages_per_block - 1
    // where usable blocks are the managed blocks (static/raw extents
    // excluded) beyond the per-plane GC-only reserve, and the open blocks
    // are those that can still hold free pages when the foreground stalls:
    // one GC write frontier per stack, one Mapping (or Data) frontier per
    // plane, and one pinned wear-leveling destination when static wear
    // leveling is enabled. Every other block is then closed, and the
    // strict inequality leaves at least one invalid page in a closed block:
    // the GC victim. See HbfController::derive_logical_capacity_pages.
    std::uint64_t logical_capacity_bytes = 0;
    bool auto_gc_enabled = true;
    // Preventive GC starts (one paced relocation step per host page write)
    // once the pages the foreground can still allocate drop to this value.
    // 0 selects pages_per_block.
    std::uint64_t gc_low_watermark_pages = 0;
    // The foreground blocks on GC when fewer than required + this many
    // pages remain allocatable to it. 0 blocks only when nothing is left.
    std::uint64_t gc_hard_watermark_pages = 0;
    // GC-only reserve, pooled per stack: this many blocks per plane (times
    // the stack's planes with managed blocks) that garbage collection keeps
    // free. It is excluded from the derived logical capacity, and
    // preventive GC starts once the stack's pool (whole free blocks plus
    // the GC frontier's free pages) minus the reserve drops to the low
    // watermark. Under sustained pressure the foreground may borrow from
    // it, but never below the floor one worst-case victim needs
    // (pages_per_block - 1 live pages, twice that under cached mapping
    // where each relocated page may evict a dirty translation page, plus
    // one block for the wear-leveling cold frontier when static wear
    // leveling is enabled): the foreground opens a block only while the
    // pool minus that block stays at or above the floor, so a victim always
    // has somewhere to go. Automatic GC requires the reserve to be at least
    // that floor.
    std::uint64_t gc_reserved_free_blocks_per_plane = 2;
    // Live pages relocated per host page write while a victim is collected
    // preventively. The pace rises automatically as the foreground runway
    // shrinks so the victim is erased before the foreground blocks; hard
    // pressure relocates whatever the blocked allocation still needs.
    std::uint32_t gc_relocation_pages_per_host_write = 1;
    // Victim score = invalid_pages - weight * (erase_count - stack minimum
    // erase count): the weight is the number of invalid pages one erase
    // above the stack's least-worn block is worth, so 0.05 means a block
    // needs 20 more erases than the coldest block before GC prefers a
    // candidate with one invalid page less.
    double gc_wear_leveling_weight = 0.0;
    // Static wear leveling. Greedy garbage collection never selects a block
    // whose pages stay valid, and a block that keeps a few live pages behind
    // an ample supply of fully invalid victims is never chosen either, so
    // long-lived data (pinned weights, a resident prefix catalog, leftovers
    // of an initial image) keeps its blocks at their original erase count
    // while the mutable KV pool absorbs every P/E cycle. After a physical
    // page-zero autoerase, the controller compares the block's actual erase
    // count with the stack's coldest closed block; when the gap reaches this
    // value the cold block's live pages are relocated into a worn block (the
    // plane's open GC block, or else the most worn free block, which then
    // becomes the open GC block). Host reclaim returns the cold block to the
    // free pool; its next page-zero program pays the physical erase. Copying
    // crosses the HBF interface in both directions, charged like GC.
    // 0 disables static wear leveling.
    std::uint32_t static_wear_leveling_erase_gap = 0;
    // The cold-block search runs at most once per this many erases of the
    // collected plane, bounding the scan cost on multi-million-block devices.
    std::uint32_t static_wear_leveling_interval_erases = 1;
    // Late start: no migration before the collected block's post-erase count
    // reaches this value, so a young device is never leveled prematurely.
    // 0 means the gap alone decides.
    std::uint32_t static_wear_leveling_start_erases = 0;
    // Hysteresis: once activated at erase_gap, keep leveling until the
    // eligible wear gap is below this value. 0 uses erase_gap (no hysteresis).
    std::uint32_t static_wear_leveling_stop_gap = 0;
    // Minimum intervening erases in this stack before a recently migrated
    // source or destination block may become a new cold source. Runtime-only
    // controller state; restored devices conservatively cool down all blocks.
    std::uint64_t static_wear_leveling_cooldown_erases = 0;
    // Direct WL copy bytes / accepted logical host bytes. A per-stack token
    // bucket starts empty, holds at most one block, and reserves a whole
    // source's live payload at admission. 0 means unlimited. Mapping and GC
    // traffic are deliberately separate, not covered by this direct-copy cap.
    double static_wear_leveling_max_write_fraction = 0.0;
};

struct HbfConfig {
    physical::hbf::HbfDeviceConfig device;
    HbfHostConfig host;
};

} // namespace hbfsim::host
