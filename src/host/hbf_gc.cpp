#include "host/hbf_controller.hpp"
#include <cmath>

namespace hbfsim::host {
using namespace hbfsim::physical;
using namespace physical;

namespace {
std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs * rhs;
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs + rhs;
}

std::uint64_t checked_ceil_div(
    std::uint64_t numerator,
    std::uint64_t denominator,
    const char* name) {
    if (denominator == 0) {
        throw std::runtime_error(std::string(name) + " has zero denominator");
    }
    return numerator == 0 ? 0 : 1 + (numerator - 1) / denominator;
}

std::uint64_t metadata_vpn(std::uint64_t lpn) {
    return lpn & ~(std::uint64_t{1} << 63);
}

bool is_metadata_lpn(std::uint64_t lpn) {
    return (lpn & (std::uint64_t{1} << 63)) != 0;
}
}

std::uint64_t HbfController::gc_reserve_requirement_pages() const {
    if (!config_.host.auto_gc_enabled || config_.host.mapping_mode == MappingMode::RawPhysical) {
        return 0;
    }
    // A reclaimable victim keeps at most pages_per_block - 1 live pages;
    // under cached mapping each relocated page may evict one dirty
    // translation page; a wear-leveling migration pins one cold block.
    const auto live = static_cast<std::uint64_t>(config_.device.pages_per_block) - 1;
    auto required = config_.host.mapping_mode == MappingMode::Cached ?
        checked_add(live, config_.device.pages_per_block, "HBF GC reserve requirement") :
        live;
    if (config_.host.static_wear_leveling_erase_gap != 0) {
        required = checked_add(
            required, config_.device.pages_per_block, "HBF GC reserve requirement");
    }
    return required;
}

std::uint64_t HbfController::gc_reserve_pages(std::size_t stack) const {
    return checked_mul(
        checked_mul(
            static_cast<std::uint64_t>(config_.host.gc_reserved_free_blocks_per_plane),
            static_cast<std::uint64_t>(active_planes_in_stack(stack)),
            "HBF GC reserve blocks per stack"),
        config_.device.pages_per_block,
        "HBF GC reserve pages per stack");
}

std::uint64_t HbfController::dirty_cached_mapping_pages(std::size_t stack) const {
    if (config_.host.mapping_mode != MappingMode::Cached) {
        return 0;
    }
    return dirty_mapping_pages_by_stack_.at(stack);
}

std::uint64_t HbfController::induced_translation_writebacks(std::size_t block_index,
    std::uint64_t dirty_cached_mapping_pages) const {
    const auto& block = blocks_.at(block_index);
    const auto live_pages = static_cast<std::uint64_t>(block.valid_pages);
    if (config_.host.mapping_mode != MappingMode::Cached ||
        block.role == BlockRole::Mapping) {
        return 0;
    }
    // Relocating a data page updates its translation through the mapping
    // cache. A miss on a full cache evicts the LRU line, which costs one
    // checkpoint program only when that line is dirty. Lines the relocation
    // itself dirties are most-recently-used and can be evicted only after
    // mapping_cache_pages_per_stack_ later misses, so at most
    //   dirty_now + max(0, live - cache_pages)
    // dirty evictions, and never more than one per relocated page, can be
    // induced by the pages that are live now.
    const auto overflow = live_pages > mapping_cache_pages_per_stack_ ?
        live_pages - mapping_cache_pages_per_stack_ : 0;
    return std::min(
        live_pages,
        checked_add(
            dirty_cached_mapping_pages,
            overflow,
            "HBF GC induced translation writebacks"));
}

std::uint64_t HbfController::mapping_allocation_pool_demand(std::size_t stack,
    std::uint64_t pages) const {
    if (pages == 0) return 0;
    return mapping_pool_demand(pages, mapping_frontier_free_pages(stack));
}

std::uint64_t HbfController::mapping_frontier_free_pages(std::size_t stack) const {
    std::uint64_t available = 0;
    const auto pps = planes_per_stack();
    for (std::size_t plane_index = stack * pps;
         plane_index < (stack + 1) * pps;
         ++plane_index) {
        if (const auto active = planes_.at(plane_index).active_mapping_block) {
            available += blocks_.at(*active).free_pages;
        }
    }
    return available;
}

std::uint64_t HbfController::mapping_pool_demand(std::uint64_t pages, std::uint64_t frontier_free_pages) const {
    pages -= std::min(pages, frontier_free_pages);
    return checked_mul(
        checked_ceil_div(pages, config_.device.pages_per_block,
            "HBF mapping allocation blocks"),
        config_.device.pages_per_block,
        "HBF mapping allocation pool demand");
}

bool HbfController::mapping_allocation_preserves_relocation(std::size_t stack) const {
    const auto& victim = gc_victim_by_stack_.at(stack);
    if (!victim) {
        return true;
    }
    auto pool = gc_relocation_capacity(stack);
    std::uint64_t mapping_pages = 0;
    const auto pps = planes_per_stack();
    for (std::size_t plane_index = stack * pps;
         plane_index < (stack + 1) * pps;
         ++plane_index) {
        if (const auto active = planes_.at(plane_index).active_mapping_block) {
            mapping_pages += blocks_.at(*active).free_pages;
        }
    }
    if (mapping_pages == 0) {
        if (pool < config_.device.pages_per_block) {
            return false;
        }
        pool -= config_.device.pages_per_block;
        mapping_pages = config_.device.pages_per_block;
    }
    mapping_pages--;
    const auto& block = blocks_.at(victim->block);
    std::uint64_t remaining = 0;
    for (auto page = victim->scan_page; page < config_.device.pages_per_block; ++page) {
        remaining += block.is_valid(page) ? 1 : 0;
    }
    const bool metadata = block.role == BlockRole::Mapping;
    const auto data_pages = metadata ? 0 : remaining;
    const auto checkpoints = metadata || config_.host.mapping_mode == MappingMode::Cached ?
        remaining : 0;
    const auto uncovered = checkpoints > mapping_pages ?
        checkpoints - mapping_pages : 0;
    const auto mapping_pool = checked_mul(
        checked_ceil_div(uncovered, config_.device.pages_per_block,
            "HBF active relocation mapping blocks"),
        config_.device.pages_per_block,
        "HBF active relocation mapping pool");
    return pool >= checked_add(
        data_pages, mapping_pool, "HBF active relocation allocation floor");
}

std::uint64_t HbfController::worst_gc_victim_demand(std::size_t stack) const {
    // A reclaimable victim keeps at most pages_per_block - 1 live pages.
    const auto live_pages =
        static_cast<std::uint64_t>(config_.device.pages_per_block) - 1;
    if (config_.host.mapping_mode != MappingMode::Cached) {
        return live_pages;
    }
    const auto overflow = live_pages > mapping_cache_pages_per_stack_ ?
        live_pages - mapping_cache_pages_per_stack_ : 0;
    return checked_add(
        live_pages,
        mapping_allocation_pool_demand(stack,
            std::min(
                live_pages,
                checked_add(
                    dirty_cached_mapping_pages(stack),
                    overflow,
                    "HBF worst GC induced writebacks"))),
        "HBF worst GC victim demand");
}

std::uint64_t HbfController::gc_relocation_capacity(std::size_t stack) const {
    auto capacity = gc_headroom(stack, BlockRole::Data).relocation_pages;
    // An active wear-leveling migration relocates cold data into the pinned
    // cold block, but the translation writebacks it induces under cached
    // mapping still draw on the GC pool.
    if (const auto& migration = wear_leveling_by_stack_.at(stack)) {
        const auto owed = mapping_allocation_pool_demand(stack,
            induced_translation_writebacks(migration->block, dirty_cached_mapping_pages(stack)));
        capacity = capacity > owed ? capacity - owed : 0;
    }
    return capacity;
}

void HbfController::remove_managed_block_wear(std::size_t index) {
    const auto& block = blocks_.at(index);
    if (block.role == BlockRole::StaticReadOnly || block.role == BlockRole::RawPhysical) return;
    auto& histogram = managed_wear_by_stack_.at(stack_of_block(index));
    const auto it = histogram.find(block.erase_count);
    if (it == histogram.end() || it->second == 0)
        throw std::runtime_error("HBF managed wear index lost a block");
    if (--it->second == 0) histogram.erase(it);
}

void HbfController::add_managed_block_wear(std::size_t index) {
    const auto& block = blocks_.at(index);
    if (block.role != BlockRole::StaticReadOnly && block.role != BlockRole::RawPhysical)
        ++managed_wear_by_stack_.at(stack_of_block(index))[block.erase_count];
}

void HbfController::mark_gc_headroom_dirty(std::size_t plane) {
    auto& stack = gc_headroom_by_stack_.at(plane / planes_per_stack());
    if (!stack.initialized) return;
    auto& cached = gc_headroom_by_plane_.at(plane);
    if (!cached.dirty) {
        stack.dirty_planes.push_back(plane);
        cached.dirty = true;
    }
}

void HbfController::reset_gc_headroom_cache() {
    for (auto& stack : gc_headroom_by_stack_) {
        stack.initialized = false;
        stack.total = {};
        stack.dirty_planes.clear();
    }
    for (auto& plane : gc_headroom_by_plane_) plane.dirty = false;
}

HbfController::GcHeadroomContribution HbfController::scan_gc_plane_headroom(
    std::size_t plane_index) const {
    const auto& plane = planes_.at(plane_index);
    const auto validate_active = [&](const std::optional<std::size_t>& active,
                                     BlockRole expected_role,
                                     const char* name) -> std::uint64_t {
        if (!active) return 0;
        const auto& block = blocks_.at(*active);
        if (block_plane_index(*active) != plane_index ||
            block.role != expected_role || block.erase_pending || block.free_pages == 0) {
            throw std::runtime_error(std::string("HBF invalid active ") + name +
                " block while computing GC headroom");
        }
        return block.free_pages;
    };
    if ((plane.active_data_block && plane.active_mapping_block &&
            *plane.active_data_block == *plane.active_mapping_block) ||
        (plane.active_data_block && plane.active_gc_block &&
            *plane.active_data_block == *plane.active_gc_block) ||
        (plane.active_mapping_block && plane.active_gc_block &&
            *plane.active_mapping_block == *plane.active_gc_block)) {
        throw std::runtime_error("HBF active Data/Mapping/GC block roles alias each other");
    }
    return {
        validate_active(plane.active_data_block, BlockRole::Data, "Data"),
        validate_active(plane.active_mapping_block, BlockRole::Mapping, "Mapping"),
        validate_active(plane.active_gc_block, BlockRole::GC, "GC"),
        static_cast<std::uint64_t>(plane.free_blocks.size()),
    };
}

const HbfController::GcHeadroomContribution& HbfController::gc_headroom_totals(
    std::size_t stack_index) const {
    auto& stack = gc_headroom_by_stack_.at(stack_index);
    const auto replace = [](GcHeadroomContribution total,
                            const GcHeadroomContribution& old,
                            const GcHeadroomContribution& current) {
        const auto adjust = [](std::uint64_t sum, std::uint64_t previous,
                               std::uint64_t next, const char* name) {
            if (previous > sum)
                throw std::runtime_error("HBF GC headroom cache contribution underflow");
            return checked_add(sum - previous, next, name);
        };
        total.data_pages = adjust(total.data_pages, old.data_pages, current.data_pages,
            "HBF foreground active-block headroom");
        total.mapping_pages = adjust(total.mapping_pages, old.mapping_pages, current.mapping_pages,
            "HBF foreground active-block headroom");
        total.gc_pages = adjust(total.gc_pages, old.gc_pages, current.gc_pages,
            "HBF active-GC relocation headroom");
        total.free_blocks = adjust(total.free_blocks, old.free_blocks, current.free_blocks,
            "HBF stack free blocks");
        return total;
    };
    if (!stack.initialized) {
        GcHeadroomContribution total;
        const auto pps = planes_per_stack();
        for (auto plane = stack_index * pps; plane < (stack_index + 1) * pps; ++plane) {
            const auto current = scan_gc_plane_headroom(plane);
            total = replace(total, {}, current);
            gc_headroom_by_plane_.at(plane) = {current, false};
        }
        stack.total = total;
        stack.initialized = true;
    }
    while (!stack.dirty_planes.empty()) {
        const auto plane = stack.dirty_planes.back();
        auto& cached = gc_headroom_by_plane_.at(plane);
        const auto current = scan_gc_plane_headroom(plane);
        // Validate and calculate all fields before publishing any cache
        // update, so a failed validation leaves this plane queued for retry.
        const auto total = replace(stack.total, cached.value, current);
        stack.total = total;
        cached = {current, false};
        stack.dirty_planes.pop_back();
    }
    return stack.total;
}

HbfController::GcHeadroom HbfController::gc_headroom(std::size_t stack,
    BlockRole allocation_role) const {
    if (allocation_role != BlockRole::Data &&
        allocation_role != BlockRole::Mapping) {
        throw std::runtime_error(
            "HBF GC headroom requires a Data or Mapping allocation role");
    }

    const auto& totals = gc_headroom_totals(stack);
    GcHeadroom headroom;
    headroom.foreground_pages = allocation_role == BlockRole::Data ?
        totals.data_pages : totals.mapping_pages;
    headroom.relocation_pages = totals.gc_pages;
    const auto free_blocks = totals.free_blocks;
    std::uint64_t returning_blocks = 0;
    headroom.relocation_pages = checked_add(
        headroom.relocation_pages,
        checked_mul(
            free_blocks,
            config_.device.pages_per_block,
            "HBF whole-free-block GC headroom"),
        "HBF total GC relocation headroom");
    // Blocks whose relocation reclaim is in flight return to the pool.
    for (const auto& [block_index, erase] : pending_block_transitions_) {
        if (erase.garbage_collection && stack_of_block(block_index) == stack) {
            returning_blocks++;
        }
    }
    // Whole blocks the foreground may open right now: those above the
    // floor one worst-case victim needs. Pages above the configured
    // reserve (pooled across the stack's planes) drive preventive GC.
    const auto floor = gc_reserve_requirement_pages();
    const auto reserve = gc_reserve_pages(stack);
    const auto spare_blocks = [&](std::uint64_t pool, std::uint64_t blocks) {
        if (pool < floor) {
            return std::uint64_t{0};
        }
        return std::min(blocks, (pool - floor) / config_.device.pages_per_block);
    };
    const auto above_reserve = [&](std::uint64_t pool) {
        return pool > reserve ? pool - reserve : 0;
    };
    const auto own_frontier_pages = headroom.foreground_pages;
    const auto spare_now = spare_blocks(headroom.relocation_pages, free_blocks);
    headroom.foreground_pages = checked_add(
        own_frontier_pages,
        checked_mul(
            spare_now,
            config_.device.pages_per_block,
            "HBF foreground whole-free-block headroom"),
        "HBF total foreground GC headroom");
    headroom.preventive_pages = checked_add(
        own_frontier_pages,
        above_reserve(headroom.relocation_pages),
        "HBF preventive GC headroom");
    const auto pool_after = checked_add(
        headroom.relocation_pages,
        checked_mul(
            returning_blocks,
            config_.device.pages_per_block,
            "HBF returning GC pool pages"),
        "HBF GC pool after in-flight reclaims");
    const auto spare_after = spare_blocks(
        pool_after,
        checked_add(free_blocks, returning_blocks, "HBF returning free blocks"));
    headroom.returning_pages = checked_mul(
        spare_after - spare_now,
        config_.device.pages_per_block,
        "HBF returning foreground headroom");
    headroom.returning_preventive_pages =
        above_reserve(pool_after) - above_reserve(headroom.relocation_pages);
    return headroom;
}

void HbfController::mark_gc_candidate_dirty(std::size_t index) {
    auto& block = blocks_.at(index);
    if (block.gc_index_dirty || gc_candidates_by_stack_.at(stack_of_block(index)).empty()) return;
    if (block.gc_bucket == std::numeric_limits<std::uint32_t>::max() &&
        (block.invalid_pages == 0 || block.free_pages != 0)) return;
    gc_dirty_blocks_by_stack_.at(stack_of_block(index)).push_back(index);
    block.gc_index_dirty = true;
}

void HbfController::refresh_gc_candidates(std::size_t stack) {
    auto& buckets = gc_candidates_by_stack_.at(stack);
    auto& dirty = gc_dirty_blocks_by_stack_.at(stack);
    for (const auto index : dirty) {
        auto& block = blocks_.at(index);
        if (block.gc_bucket != std::numeric_limits<std::uint32_t>::max()) {
            buckets.at(block.gc_bucket).erase({block.gc_erase_key, index});
            block.gc_bucket = std::numeric_limits<std::uint32_t>::max();
        }
        if ((block.role == BlockRole::Data || block.role == BlockRole::Mapping ||
             block.role == BlockRole::GC) && block.free_pages == 0 && block.invalid_pages != 0) {
            block.gc_bucket = block.valid_pages +
                (block.role == BlockRole::Mapping ? config_.device.pages_per_block : 0);
            block.gc_erase_key = config_.host.gc_wear_leveling_weight == 0 ? 0 : block.erase_count;
            buckets.at(block.gc_bucket).insert({block.gc_erase_key, index});
        }
        block.gc_index_dirty = false;
    }
    dirty.clear();
}

std::optional<std::size_t> HbfController::choose_gc_victim(std::size_t stack) {
    // Preserve the invalid-page / relocation / wear score. Among equal
    // scores, rotate independently across planes. A lowest-id tie break
    // reclaims whole runs on one plane; tying rotation to the page allocator
    // also locks to one plane when block and stripe sizes are commensurate.
    // Selection is a read-only preview. start_relocation advances the cursor.
    refresh_gc_candidates(stack);
    const auto relocation_capacity = gc_relocation_capacity(stack);
    const auto mapping_free = mapping_frontier_free_pages(stack);
    const auto dirty = dirty_cached_mapping_pages(stack);
    const auto minimum_erase_count = stack_minimum_erase_count(stack);
    const auto& active_victim = gc_victim_by_stack_.at(stack);
    const auto& migration = wear_leveling_by_stack_.at(stack);
    const auto cold_block = cold_block_by_stack_.at(stack);
    const auto count = planes_per_stack() * config_.device.blocks_per_plane;
    const auto first = stack * count;
    const auto pivot = first + next_gc_allocation_plane_per_stack_.at(stack) * config_.device.blocks_per_plane;
    const auto rank = [&](std::size_t i) { return (i + count - pivot) % count; };
    std::optional<std::size_t> best;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const auto& bucket : gc_candidates_by_stack_.at(stack)) {
        if (bucket.empty()) continue;
        const auto representative = bucket.begin()->second;
        const auto& block0 = blocks_[representative];
        const auto induced = induced_translation_writebacks(representative, dirty);
        const auto demand = block0.role == BlockRole::Mapping ?
            mapping_pool_demand(block0.valid_pages, mapping_free) :
            checked_add(block0.valid_pages, mapping_pool_demand(induced, mapping_free),
                "HBF GC victim relocation demand");
        if (demand > relocation_capacity) continue;
        std::optional<double> bucket_score;
        // Search each equal-wear group in cyclic plane order, without
        // scanning every equally eligible block. Later wear levels matter
        // only while their scores still tie (including floating-point ties).
        for (auto level = bucket.begin(); level != bucket.end();) {
            const auto end = bucket.upper_bound({level->first, std::numeric_limits<std::size_t>::max()});
            auto pivot_it = bucket.lower_bound({level->first, pivot});
            if (pivot_it == end) pivot_it = level;
            auto it = pivot_it;
            const auto& base = blocks_[it->second];
            const double score = double(config_.device.pages_per_block - base.valid_pages) - double(induced) -
                config_.host.gc_wear_leveling_weight * double(base.erase_count - minimum_erase_count);
            if (bucket_score && score < *bucket_score) break;
            do {
                const auto i = it->second;
                const auto& block = blocks_[i];
                const auto& plane = planes_.at(block_plane_index(i));
                if (!(block.erase_pending || pending_block_transitions_.contains(i) ||
                      block.pending_program_pages != 0 || block.pending_mapping_publications != 0 ||
                      plane.active_data_block == i || plane.active_mapping_block == i ||
                      plane.active_gc_block == i || cold_block == i ||
                      (active_victim && active_victim->block == i) || (migration && migration->block == i))) {
                    bucket_score = score;
                    if (!best || score > best_score || (score == best_score && rank(i) < rank(*best))) {
                        best = i; best_score = score;
                    }
                    break;
                }
                if (++it == end) it = level;
            } while (it != pivot_it);
            level = end;
        }
    }
    return best;
}

std::optional<HbfController::ActiveRelocation> HbfController::start_relocation(std::size_t block_index,
    RelocationPurpose purpose,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto& block = blocks_.at(block_index);
    if (pending_block_transitions_.contains(block_index) || block.erase_pending ||
        block.pending_program_pages != 0 ||
        block.pending_mapping_publications != 0 ||
        block.free_pages != 0 ||
        checked_add(
            static_cast<std::uint64_t>(block.valid_pages),
            static_cast<std::uint64_t>(block.invalid_pages),
            "HBF relocation source page accounting") != config_.device.pages_per_block) {
        throw std::runtime_error(
            "HBF relocation source is not a closed, fully committed block");
    }
    const bool wear_leveling = purpose == RelocationPurpose::StaticWearLeveling;
    const auto stack = stack_of_block(block_index);
    if (!wear_leveling) {
        // GC victims determine which planes regain allocatable blocks.
        // Wear-leveling moves must not consume this independent GC turn.
        next_gc_allocation_plane_per_stack_.at(stack) =
            (block_plane_index(block_index) + 1) % planes_per_stack();
    }
    const double start_ns = std::max(at_ns, control_ready_ns_.at(stack));
    breakdown.scheduler_queue_wait_ns += start_ns - at_ns;
    breakdown.maintenance_ns += config_.host.host_gc_decision_ns;
    stats_.host_gc_control_ns += config_.host.host_gc_decision_ns;
    const double select_done = start_ns + config_.host.host_gc_decision_ns;
    control_ready_ns_[stack] = select_done;
    if (spans != nullptr) {
        add_trace_span(
            spans,
            wear_leveling ? "static_wear_leveling_select" : "gc_victim_select",
            "maintenance",
            "host/gc",
            at_ns,
            select_done,
            true,
            "block" + std::to_string(block_index));
    }
    at_ns = select_done;
    return ActiveRelocation{
        .block = block_index,
        .purpose = purpose,
        .invalid_pages_at_start = block.invalid_pages,
        .selection_ready_ns = select_done,
        .chain_ready_ns = select_done,
    };
}

bool HbfController::relocation_scan_complete(const ActiveRelocation& relocation) const {
    const auto& block = blocks_.at(relocation.block);
    for (std::uint32_t page = relocation.scan_page;
         page < config_.device.pages_per_block;
         ++page) {
        if (block.is_valid(page)) {
            return false;
        }
    }
    return true;
}

std::optional<std::size_t> HbfController::ensure_cold_block(std::size_t stack,
    std::uint32_t minimum_erase_count) {
    auto& cold = cold_block_by_stack_.at(stack);
    if (cold && blocks_.at(*cold).free_pages != 0) {
        return cold;
    }
    cold.reset();
    // The cold write frontier is opened on the most worn free block of the
    // stack once the GC pool holds the floor, which is sized for the cold
    // block (gc_reserve_requirement_pages). It must lead the migrating
    // block by at least half the gap; otherwise the migration would only
    // create another cold block.
    if (gc_headroom(stack, BlockRole::Data).relocation_pages <
        gc_reserve_requirement_pages()) {
        return std::nullopt;
    }
    const auto pps = planes_per_stack();
    std::optional<std::size_t> destination;
    for (std::size_t plane_index = stack * pps;
         plane_index < (stack + 1) * pps;
         ++plane_index) {
        const auto& plane = planes_.at(plane_index);
        for (const auto block_index : plane.free_blocks) {
            const auto& block = blocks_.at(block_index);
            if (block.erase_pending || pending_block_transitions_.contains(block_index) ||
                static_cast<std::uint64_t>(block.erase_count) + 1 < minimum_erase_count) {
                continue;
            }
            if (!destination ||
                block.erase_count > blocks_.at(*destination).erase_count) {
                destination = block_index;
            }
        }
    }
    if (!destination) {
        return std::nullopt;
    }
    const auto plane_index = block_plane_index(*destination);
    auto& plane = planes_.at(plane_index);
    const auto position = std::find(
        plane.free_blocks.begin(), plane.free_blocks.end(), *destination);
    if (position == plane.free_blocks.end()) {
        throw std::runtime_error(
            "HBF cold-block destination left the free pool");
    }
    mark_gc_headroom_dirty(plane_index);
    plane.free_blocks.erase(position);
    auto& block = blocks_.at(*destination);
    if (block.role != BlockRole::Free || block.free_pages != config_.device.pages_per_block ||
        block.valid_pages != 0 || block.invalid_pages != 0 || block.next_page != 0 ||
        block.pending_program_pages != 0 || block.pending_mapping_publications != 0) {
        throw std::runtime_error(
            "HBF cold-block destination is not an erased block");
    }
    set_block_role(*destination, BlockRole::GC);
    cold = destination;
    return cold;
}

std::uint32_t HbfController::relocate_live_pages(ActiveRelocation& relocation,
    double at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    std::uint32_t max_pages) {
    const auto victim_block = relocation.block;
    const auto victim_stack = stack_of_block(victim_block);
    const bool wear_leveling =
        relocation.purpose == RelocationPurpose::StaticWearLeveling;
    at_ns = std::max(at_ns, relocation.selection_ready_ns);
    const auto block_begin =
        static_cast<std::uint64_t>(victim_block) * config_.device.pages_per_block;
    const auto first_sequence = next_commit_sequence_;
    std::uint32_t relocated = 0;
    auto page = relocation.scan_page;
    for (; page < config_.device.pages_per_block && relocated < max_pages; ++page) {
        const auto& victim = blocks_.at(victim_block);
        if (!victim.is_valid(page)) {
            continue;
        }
        const auto old_ppn = block_begin + page;
        std::uint64_t lpn = 0;
        PageOwner owner = PageOwner::Unassigned;
        if (const auto materialized = programmed_pages_.find(old_ppn);
            materialized != programmed_pages_.end()) {
            if (materialized->second.status != PageStatus::Valid ||
                (materialized->second.owner != PageOwner::Logical &&
                 materialized->second.owner != PageOwner::Mapping)) {
                throw std::runtime_error(
                    "HBF relocation found invalid materialized victim ownership");
            }
            lpn = materialized->second.lpn;
            owner = materialized->second.owner;
        } else if (const auto compact = compact_page_identity(old_ppn)) {
            lpn = compact->logical_key;
            owner = compact->owner;
        } else {
            throw std::runtime_error(
                "HBF relocation valid bitmap page has no logical owner");
        }
        const bool metadata_page = owner == PageOwner::Mapping;
        if (metadata_page != is_metadata_lpn(lpn)) {
            throw std::runtime_error(
                "HBF relocation logical key disagrees with page ownership");
        }

        std::uint64_t pool_need = wear_leveling || metadata_page ? 0 : 1;
        std::uint64_t mapping_programs = !wear_leveling && metadata_page ? 1 : 0;
        if (config_.host.mapping_mode == MappingMode::Cached && !metadata_page) {
            const auto& cache = mapping_cache_by_stack_.at(victim_stack);
            const auto& lru = mapping_cache_lru_by_stack_.at(victim_stack);
            const auto key = mapping_cache_key(lpn);
            const auto existing = cache.find(key);
            const auto required_bytes = config_.host.mapping_cache_tag_bytes +
                (config_.host.mapping_cache_layout == MappingCacheLayout::Entry ?
                    8 : config_.device.page_size_bytes);
            const auto growth = existing == cache.end() ? required_bytes :
                (config_.host.mapping_cache_layout == MappingCacheLayout::Extent &&
                 !existing->second.dense ? required_bytes - existing->second.charged_bytes : 0);
            auto pending_bytes = mapping_cache_bytes_by_stack_.at(victim_stack) + growth;
            // Admission reserves checkpoint space only for dirty records
            // actually displaced by this insertion or in-memory expansion.
            // An entry-cache hit requires no frame growth or checkpoint.
            for (auto it = lru.rbegin();
                 it != lru.rend() && pending_bytes > stats_.mapping_cache_capacity_bytes_per_stack;
                 ++it) {
                const auto& candidate = cache.at(*it);
                if (*it == key || candidate.pins != 0) continue;
                pending_bytes -= candidate.charged_bytes;
                if (mapping_entry_dirty(candidate)) ++mapping_programs;
            }
        }

        pool_need = checked_add(
            pool_need,
            mapping_allocation_pool_demand(victim_stack, mapping_programs),
            "HBF relocation page admission");
        if (pool_need != 0 &&
            gc_headroom(victim_stack, BlockRole::Data).relocation_pages <
                pool_need) {
            break;
        }
        std::optional<std::size_t> cold_destination;
        if (wear_leveling) {
            auto& frontier = cold_block_by_stack_[victim_stack];
            if ((!frontier || blocks_[*frontier].free_pages == 0) &&
                relocation.cold_destination_spare) {
                frontier = relocation.cold_destination_spare;
                relocation.cold_destination_spare.reset();
            }
            cold_destination = ensure_cold_block(victim_stack, relocation.cold_destination_minimum_erases);
            if (!cold_destination) {
                break;
            }
        }

        const auto buffer_ready = std::max(at_ns, copy_ready_ns_.at(victim_stack));
        breakdown.scheduler_queue_wait_ns += buffer_ready - at_ns;
        const double read_done = schedule_read_page(
            old_ppn,
            buffer_ready,
            breakdown,
            spans,
            TransactionSource::Host,
            wear_leveling ? HeatmapTrafficSource::Maintenance : HeatmapTrafficSource::GarbageCollection,
            ReadPayloadRoute::External,
            config_.device.page_size_bytes);
        // The finite copy slot resides in the shared HBM reservation. The
        // common host program path pays its source read exactly once.
        double alloc_ready = host_memory_transfer(victim_stack, config_.device.page_size_bytes,
            Op::Write, read_done, breakdown, spans);
        stats_.host_gc_read_bytes += config_.device.page_size_bytes;
        stats_.host_gc_write_bytes += config_.device.page_size_bytes;
        const auto new_ppn = cold_destination ?
            allocate_page_from_pinned_block(*cold_destination, alloc_ready, breakdown, spans) :
            allocate_free_page(
                alloc_ready,
                breakdown,
                spans,
                metadata_page ? BlockRole::Mapping : BlockRole::GC,
                victim_stack,
                block_plane_index(victim_block));
        const double program_done = schedule_program_page(
            new_ppn,
            alloc_ready,
            breakdown,
            spans,
            TransactionSource::Host,
            wear_leveling ? HeatmapTrafficSource::Maintenance : HeatmapTrafficSource::GarbageCollection,
            &copy_ready_ns_.at(victim_stack));
        const auto program_commit_sequence = schedule_media_program_commit(
            new_ppn,
            lpn,
            owner,
            program_done);
        double mapping_ready = program_done;
        if (!metadata_page) {
            // Relocating a data page updates the same authoritative L2P entry
            // as a user write. In cached mode, the corresponding checkpoint
            // page may need cache admission before it becomes dirty.
            state_observation_by_stack_.at(victim_stack) = std::max(
                state_observation_by_stack_.at(victim_stack), mapping_ready);
            access_mapping(
                lpn,
                TransactionSource::Host,
                MappingAccessKind::Update,
                mapping_ready,
                breakdown,
                spans);
            state_observation_by_stack_.at(victim_stack) = std::max(
                state_observation_by_stack_.at(victim_stack), mapping_ready);
        } else {
            const double mapping_done =
                mapping_ready + config_.host.mapping_update_ns;
            breakdown.translation_ns += config_.host.mapping_update_ns;
            if (spans != nullptr) {
                add_trace_span(
                    spans,
                    "gc_mapping_checkpoint_relocate",
                    "translation",
                    "logic/mapping_table",
                    mapping_ready,
                    mapping_done,
                    true,
                    "vpn" + std::to_string(metadata_vpn(lpn)));
            }
            mapping_ready = mapping_done;
        }
        const double mapping_done = mapping_ready;
        if (metadata_page) {
            schedule_vpn_mapping_commit(
                metadata_vpn(lpn),
                new_ppn,
                mapping_done,
                program_commit_sequence,
                old_ppn);
        } else {
            schedule_lpn_mapping_commit(
                lpn,
                new_ppn,
                mapping_done,
                program_commit_sequence,
                old_ppn);
            mark_mapping_page_dirty(mapping_vpn_for_lpn(lpn), mapping_done);
        }
        relocation.destination_ppns.push_back(new_ppn);
        relocation.chain_ready_ns = std::max(
            relocation.chain_ready_ns, mapping_done);
        relocation.relocated_pages++;
        relocated++;
        stats_.physical_read_bytes += config_.device.page_size_bytes;
        stats_.physical_write_bytes += config_.device.page_size_bytes;
        stats_.page_reads++;
        stats_.page_programs++;
        // GC must not launder a recently migrated block's cooldown by moving
        // its pages into another physical block.
        const auto moved_at = wear_leveling ?
            std::optional<std::uint64_t>(wear_leveling_erase_clock(victim_stack)) :
            wear_leveling_last_move_by_block_[victim_block];
        auto& destination_moved =
            wear_leveling_last_move_by_block_.at(new_ppn / config_.device.pages_per_block);
        if (moved_at && (!destination_moved || *destination_moved < *moved_at))
            destination_moved = moved_at;
        if (wear_leveling) {
            stats_.static_wear_leveling_relocation_payload_bytes +=
                config_.device.page_size_bytes;
            stats_.static_wear_leveling_relocations++;
        } else {
            stats_.gc_relocation_payload_bytes += config_.device.page_size_bytes;
            stats_.gc_relocations++;
            if (metadata_page) {
                stats_.gc_mapping_relocations++;
            } else {
                stats_.gc_data_relocations++;
            }
        }
    }
    relocation.scan_page = page;
    // Every commit scheduled by this step (program completions, conditional
    // publications, induced checkpoints) is a dependency of reclaim.
    for (auto sequence = first_sequence; sequence < next_commit_sequence_; ++sequence) {
        relocation.dependency_sequences.push_back(sequence);
    }
    return relocated;
}

double HbfController::schedule_relocation_reclaim(ActiveRelocation relocation,
    double at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto victim_block = relocation.block;
    const auto victim_stack = stack_of_block(victim_block);
    const bool wear_leveling =
        relocation.purpose == RelocationPurpose::StaticWearLeveling;
    if (!relocation_scan_complete(relocation)) {
        throw std::runtime_error(
            "HBF relocation reclaim was scheduled with live source pages");
    }
    if (pending_block_transitions_.contains(victim_block)) {
        throw std::runtime_error(
            "HBF relocation source already has an in-flight erase");
    }
    // Host ownership is released only after the last publication of this
    // chain, so its commit finds no live page, and every previously issued
    // read from the source has completed.
    // Reclaim only retires host ownership. The next page-zero program pays
    // the physical erase, including its P/E count and completion latency.
    const double finish_ns = std::max({at_ns, relocation.chain_ready_ns,
        blocks_.at(victim_block).issued_media_ready_ns});
    (void)breakdown;
    (void)spans;
    PendingBlockTransition pending{
        .finish_ns = finish_ns,
        .garbage_collection = true,
        .static_wear_leveling = wear_leveling,
        .dependency_sequences = std::move(relocation.dependency_sequences),
        .destination_ppns = std::move(relocation.destination_ppns),
    };
    const auto [pending_transition, inserted] = pending_block_transitions_.emplace(
        victim_block, std::move(pending));
    if (!inserted) {
        throw std::runtime_error(
            "HBF relocation source already has an in-flight erase");
    }
    pending_transition->second.commit_sequence = schedule_commit(
        victim_stack, finish_ns, [this, victim_block]() {
        const auto pending = pending_block_transitions_.find(victim_block);
        if (pending == pending_block_transitions_.end()) {
            throw std::runtime_error(
                "HBF GC reclaim lost its in-flight reservation");
        }
        const auto finish_ns = pending->second.finish_ns;
        const auto block_begin = static_cast<std::uint64_t>(
            victim_block) * config_.device.pages_per_block;
        for (std::uint32_t page = 0;
             page < config_.device.pages_per_block;
             ++page) {
            if (blocks_.at(victim_block).is_valid(page)) {
                throw std::runtime_error(
                    "HBF GC reclaim reached commit with a live source page");
            }
            const auto ppn = block_begin + page;
            if (const auto state = programmed_pages_.find(ppn);
                state != programmed_pages_.end() &&
                state->second.status == PageStatus::Valid) {
                throw std::runtime_error(
                    "HBF GC source page table remained live at reclaim commit");
            }
        }
        release_invalid_block(victim_block);
        materialized_ready_by_block_.at(victim_block) = std::max(
            materialized_ready_by_block_.at(victim_block), finish_ns);
        pending_block_transitions_.erase(pending);
    });
    background_finish_ns_ = std::max(background_finish_ns_, finish_ns);
    const auto reclaimed = static_cast<std::uint64_t>(config_.device.pages_per_block) -
        relocation.relocated_pages;
    if (wear_leveling) {
        // Concurrent host invalidations may make the pre-reserved second
        // destination unnecessary. Return only a wholly untouched block;
        // no program/erase work or wear credit is manufactured here.
        if (relocation.cold_destination_spare) {
            const auto spare = *relocation.cold_destination_spare;
            auto& block = blocks_.at(spare);
            if (block.next_page != 0 || block.valid_pages != 0 ||
                block.pending_program_pages != 0 || block.free_pages != config_.device.pages_per_block)
                throw std::runtime_error("HBF unused WL spare is not erased");
            set_block_role(spare, BlockRole::Free);
            planes_.at(block_plane_index(spare)).free_blocks.push_front(spare);
            relocation.cold_destination_spare.reset();
        }
        stats_.static_wear_leveling_runs++;
        if (wear_leveling_source_runs_by_block_[victim_block]++ == 0) {
            stats_.static_wear_leveling_unique_source_blocks++;
        } else {
            stats_.static_wear_leveling_repeat_source_runs++;
        }
        stats_.static_wear_leveling_reclaimed_invalid_pages = checked_add(
            stats_.static_wear_leveling_reclaimed_invalid_pages,
            reclaimed,
            "HBF static wear-leveling reclaimed invalid-page accounting");
    } else {
        stats_.gc_runs++;
        stats_.gc_reclaimed_invalid_pages = checked_add(
            stats_.gc_reclaimed_invalid_pages,
            reclaimed,
            "HBF GC reclaimed invalid-page accounting");
    }
    return finish_ns;
}

std::optional<std::size_t> HbfController::pending_gc_reclaim(std::size_t stack) const {
    // Earliest-finishing relocation reclaim of the stack (GC victim or a
    // fully migrated wear-leveling source): the cheapest capacity a blocked
    // allocation can wait for, since its relocation work is already done.
    std::optional<std::size_t> earliest;
    for (const auto& [block_index, erase] : pending_block_transitions_) {
        if (!erase.garbage_collection ||
            stack_of_block(block_index) != stack) {
            continue;
        }
        if (!earliest || erase.finish_ns <
                pending_block_transitions_.at(*earliest).finish_ns) {
            earliest = block_index;
        }
    }
    return earliest;
}

void HbfController::maybe_run_gc(double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    std::uint64_t required_pages,
    std::size_t stack,
    BlockRole allocation_role,
    std::optional<std::size_t> preferred_plane) {
    (void)preferred_plane;
    if (!config_.host.auto_gc_enabled || required_pages == 0) {
        return;
    }
    if (required_pages != 1) {
        throw std::runtime_error(
            "HBF GC allocation preview currently requires one page");
    }
    if (gc_active_by_stack_.at(stack)) {
        // A checkpoint evicted inside a relocation step allocates from the
        // GC pool; it never collects recursively.
        return;
    }
    gc_active_by_stack_[stack] = true;
    struct GcActiveGuard {
        std::vector<bool>& active;
        std::size_t stack;
        ~GcActiveGuard() { active[stack] = false; }
    } active_guard{gc_active_by_stack_, stack};

    const auto low_watermark = config_.host.gc_low_watermark_pages == 0 ?
        static_cast<std::uint64_t>(config_.device.pages_per_block) :
        config_.host.gc_low_watermark_pages;
    const auto soft_threshold = checked_add(
        required_pages, low_watermark, "HBF GC soft threshold");
    const auto hard_threshold = checked_add(
        required_pages, config_.host.gc_hard_watermark_pages, "HBF GC hard threshold");
    const auto allocation_ready = [&]() {
        return gc_headroom(stack, allocation_role).foreground_pages >= hard_threshold &&
            (allocation_role != BlockRole::Mapping ||
             mapping_allocation_preserves_relocation(stack));
    };
    auto& victim = gc_victim_by_stack_.at(stack);
    const auto finish_victim = [&](double cursor_ns) {
        (void)schedule_relocation_reclaim(*victim, cursor_ns, breakdown, spans);
        victim.reset();
    };

    auto headroom = gc_headroom(stack, allocation_role);
    if (allocation_ready()) {
        if (allocation_role == BlockRole::Mapping) {
            return;
        }
        // Pages already promised by in-flight relocation reclaims count: a
        // victim whose erase is scheduled has done its relocation work, and
        // starting the next one before it lands would spend reserve blocks
        // twice over. Preventive GC keeps the pool above the configured
        // reserve; the runway it paces against is what the foreground can
        // still write before it blocks on the floor.
        const auto credited = checked_add(
            headroom.preventive_pages,
            headroom.returning_preventive_pages,
            "HBF GC credited preventive headroom");
        const auto runway_pages = checked_add(
            headroom.foreground_pages,
            headroom.returning_pages,
            "HBF GC credited runway");
        if (credited > soft_threshold) {
            // No reclaim pressure: only a wear-leveling migration advances.
            // Its cold frontier may take the plane block this allocation
            // was counting on, so the foreground is rechecked below.
            auto& pending = pending_wear_check_by_stack_.at(stack);
            if (pending && !blocks_.at(*pending).erase_pending &&
                !pending_block_transitions_.contains(*pending)) {
                const auto hot_block = *pending;
                pending.reset();
                maybe_start_static_wear_leveling(at_ns, breakdown, spans, stack,
                    block_plane_index(hot_block), blocks_.at(hot_block).erase_count);
            }
            advance_static_wear_leveling(at_ns, breakdown, spans, stack);
            headroom = gc_headroom(stack, allocation_role);
            if (headroom.foreground_pages >= hard_threshold) {
                return;
            }
        } else {
            // Soft pressure: one paced step at the caller's cursor. The
            // pace rises as the credited runway shrinks so the victim is
            // erased before the foreground blocks.
            double cursor_ns = at_ns;
            if (!victim) {
                const auto selected = choose_gc_victim(stack);
                if (!selected) {
                    return;
                }
                victim = start_relocation(*selected,
                    RelocationPurpose::GarbageCollection,
                    cursor_ns,
                    breakdown,
                    spans);
            }
            const auto runway = runway_pages - hard_threshold;
            const auto remaining = static_cast<std::uint64_t>(
                blocks_.at(victim->block).valid_pages);
            const auto needed_pace = runway == 0 ?
                remaining : checked_ceil_div(remaining, runway, "HBF GC pace");
            const auto pace = std::max<std::uint64_t>(
                config_.host.gc_relocation_pages_per_host_write,
                std::min<std::uint64_t>(needed_pace, config_.device.pages_per_block));
            (void)relocate_live_pages(*victim, cursor_ns, breakdown, spans,
                static_cast<std::uint32_t>(pace));
            if (relocation_scan_complete(*victim)) {
                finish_victim(cursor_ns);
            }
            // The step's GC frontier may have opened the block this
            // allocation was counting on (relocation prefers the victim's
            // plane, where the erase returns it); if so the foreground is
            // blocked until that erase lands and is handled below.
            if (gc_headroom(stack, allocation_role).foreground_pages >= hard_threshold) {
                return;
            }
        }
    }

    // Hard pressure: the allocation cannot proceed. Collect until it can,
    // materializing only this stack's reclaim chain. Each iteration first
    // banks any relocation reclaim already in flight (its capacity costs no
    // further relocation work), then relocates as much of the current
    // victim as the GC pool holds, and schedules its reclaim once no live
    // page remains.
    stats_.gc_user_blocked_runs++;
    const double blocked_begin_ns = at_ns;
    const auto wait_for_reclaim = [&](std::size_t pending_victim, const char* detail) {
        const double wait_begin_ns = at_ns;
        std::unordered_set<std::uint64_t> sequences;
        add_pending_block_transition_commits(pending_victim, sequences);
        at_ns = std::max(
            at_ns, pending_block_transitions_.at(pending_victim).finish_ns);
        apply_selected_commits_through(sequences, at_ns);
        if (at_ns > wait_begin_ns) {
            breakdown.scheduler_queue_wait_ns += at_ns - wait_begin_ns;
            if (spans != nullptr) {
                add_trace_span(
                    spans,
                    "gc/foreground_block",
                    "maintenance",
                    "host/gc",
                    wait_begin_ns,
                    at_ns,
                    true,
                    detail);
            }
        }
    };
    const auto stuck = [&](const std::string& reason) {
        return std::runtime_error(
            "HBF GC cannot make progress for role " + role_name(allocation_role) +
            ": " + reason + " (relocation_capacity=" +
            std::to_string(gc_relocation_capacity(stack)) +
            " pages, no relocation reclaim in flight, no commit pending); raise "
            "hbf-gc-reserved-free-blocks-per-plane or lower "
            "hbf-logical-capacity-bytes");
    };
    const auto iteration_limit = checked_add(
        checked_mul(
            static_cast<std::uint64_t>(planes_per_stack()) * config_.device.blocks_per_plane,
            8,
            "HBF GC iteration guard"),
        256,
        "HBF GC iteration guard");
    for (std::uint64_t iteration = 0;; ++iteration) {
        if (allocation_ready()) {
            break;
        }
        if (iteration > iteration_limit) {
            throw std::runtime_error(
                "HBF GC exceeded its iteration guard without freeing a "
                "foreground page for role " + role_name(allocation_role) +
                " (relocation_capacity=" +
                std::to_string(gc_relocation_capacity(stack)) + " pages)");
        }
        if (const auto pending = pending_gc_reclaim(stack)) {
            wait_for_reclaim(*pending, "wait for in-flight relocation reclaim");
            continue;
        }
        if (!victim) {
            const auto selected = choose_gc_victim(stack);
            if (!selected) {
                // A closed block may still be committing; a pending commit
                // can also retire a live page or a raw erase.
                const double wait_begin_ns = at_ns;
                if (!advance_to_next_commit(at_ns, stack)) {
                    throw stuck(
                        "no closed block with an invalid page fits the GC-only "
                        "reserve");
                }
                if (at_ns > wait_begin_ns) {
                    breakdown.scheduler_queue_wait_ns += at_ns - wait_begin_ns;
                    if (spans != nullptr) {
                        add_trace_span(
                            spans,
                            "gc/foreground_block",
                            "maintenance",
                            "host/gc",
                            wait_begin_ns,
                            at_ns,
                            true,
                            "wait for reclaimable committed state");
                    }
                }
                continue;
            }
            victim = start_relocation(*selected,
                RelocationPurpose::GarbageCollection,
                at_ns,
                breakdown,
                spans);
        }
        const auto relocated = relocate_live_pages(*victim, at_ns, breakdown, spans, config_.device.pages_per_block);
        if (relocation_scan_complete(*victim)) {
            finish_victim(at_ns);
            continue;
        }
        if (relocated == 0) {
            // Stalled on GC-pool capacity mid-victim with no relocation
            // reclaim in flight: only a pending commit (a raw erase, or a
            // publication that retires a stale copy) can still help.
            const double wait_begin_ns = at_ns;
            if (!advance_to_next_commit(at_ns, stack)) {
                throw stuck(
                    "the GC-only reserve cannot hold victim " +
                     std::to_string(victim->block) + " at page " +
                     std::to_string(victim->scan_page) + " with " +
                     std::to_string(blocks_.at(victim->block).valid_pages) +
                     " live pages");
            }
            if (at_ns > wait_begin_ns) {
                breakdown.scheduler_queue_wait_ns += at_ns - wait_begin_ns;
                if (spans != nullptr) {
                    add_trace_span(
                        spans,
                        "gc/foreground_block",
                        "maintenance",
                        "host/gc",
                        wait_begin_ns,
                        at_ns,
                        true,
                        "wait for GC pool capacity");
                }
            }
        }
    }
    if (at_ns > blocked_begin_ns && spans != nullptr) {
        add_trace_span(
            spans,
            "gc/foreground_block",
            "maintenance",
            "host/gc",
            blocked_begin_ns,
            at_ns,
            true,
            "foreground hard-watermark GC");
    }
}

void HbfController::complete_active_relocations(double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // A drain finishes every relocation in flight so the conservation
    // identities (relocations + reclaimed == runs x pages_per_block) close
    // within the process and the persistent image never carries a
    // half-collected victim.
    //
    // Every stack starts its remaining relocations at the common drain
    // boundary. A stack owns its GC control resources, its copy slot and its
    // reclaim/capacity dependencies, so there is no cross-stack reason for
    // one stack's leftover copies to postpone another's. Contention that is
    // genuinely shared (host memory, the mapping path) is still charged by
    // the resource calendars those transfers reserve, which observe every
    // stack's work regardless of the order this loop visits them. The drain
    // ends at the latest per-stack completion.
    const double drain_begin_ns = at_ns;
    double drain_finish_ns = at_ns;
    for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
        double stack_ns = drain_begin_ns;
        for (auto* slot : {&gc_victim_by_stack_.at(stack),
                           &wear_leveling_by_stack_.at(stack)}) {
            if (!*slot) {
                continue;
            }
            gc_active_by_stack_[stack] = true;
            struct GcActiveGuard {
                std::vector<bool>& active;
                std::size_t stack;
                ~GcActiveGuard() { active[stack] = false; }
            } active_guard{gc_active_by_stack_, stack};
            auto& relocation = **slot;
            for (;;) {
                (void)relocate_live_pages(relocation, stack_ns, breakdown, spans, config_.device.pages_per_block);
                if (relocation_scan_complete(relocation)) {
                    break;
                }
                if (const auto pending = pending_gc_reclaim(stack)) {
                    std::unordered_set<std::uint64_t> sequences;
                    add_pending_block_transition_commits(*pending, sequences);
                    stack_ns = std::max(
                        stack_ns, pending_block_transitions_.at(*pending).finish_ns);
                    apply_selected_commits_through(sequences, stack_ns);
                } else if (!advance_to_next_commit(stack_ns, stack)) {
                    throw std::runtime_error(
                        "HBF drain cannot complete an active relocation: the "
                        "GC-only reserve cannot hold its remaining pages; purpose=" +
                        std::string(relocation.purpose == RelocationPurpose::StaticWearLeveling ? "WL" : "GC") +
                        " block=" + std::to_string(relocation.block) +
                        " scan=" + std::to_string(relocation.scan_page) +
                        " live=" + std::to_string(blocks_[relocation.block].valid_pages) +
                        " cold_free=" + std::to_string(cold_block_by_stack_[stack] ?
                            blocks_[*cold_block_by_stack_[stack]].free_pages : 0) +
                        " pool=" + std::to_string(gc_relocation_capacity(stack)));
                }
            }
            const double finish_ns = schedule_relocation_reclaim(relocation, stack_ns, breakdown, spans);
            slot->reset();
            stack_ns = std::max(stack_ns, finish_ns);
        }
        drain_finish_ns = std::max(drain_finish_ns, stack_ns);
    }
    at_ns = drain_finish_ns;
}

std::uint64_t HbfController::allocate_page_from_pinned_block(std::size_t block_index,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // Same controller cost as the free-page allocator, but the destination
    // is the stack's cold write frontier rather than the FIFO free pool.
    breakdown.address_mapping_ns += config_.host.free_page_allocation_ns;
    if (spans != nullptr) {
        add_trace_span(
            spans,
            "free_page_alloc",
            "translation",
            "host/free_page_allocator",
            at_ns,
            at_ns + config_.host.free_page_allocation_ns,
            true,
            "static_wear_leveling");
    }
    at_ns += config_.host.free_page_allocation_ns;
    return allocate_page_from_block(block_index);
}

std::uint64_t HbfController::wear_leveling_erase_clock(std::size_t stack) const {
    std::uint64_t total = 0;
    const auto pps = planes_per_stack();
    for (auto p = stack * pps; p < (stack + 1) * pps; ++p) {
        total += planes_[p].erase_count;
    }
    return total;
}

void HbfController::maybe_start_static_wear_leveling(double at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    std::size_t stack,
    std::size_t hot_plane,
    std::uint32_t hot_erase_count) {
    const auto gap = config_.host.static_wear_leveling_erase_gap;
    if (gap == 0) {
        return;
    }
    auto& plane = planes_.at(hot_plane);
    // Rate limit: one cold-block search per interval of this plane's erases.
    if (plane.erase_count < checked_add(
            plane.static_wear_leveling_checked_erase_count,
            config_.host.static_wear_leveling_interval_erases,
            "HBF static wear-leveling check interval")) {
        return;
    }
    plane.static_wear_leveling_checked_erase_count = plane.erase_count;
    stats_.static_wear_leveling_checks++;
    if (wear_leveling_by_stack_.at(stack)) {
        return;
    }
    const auto stop_gap = config_.host.static_wear_leveling_stop_gap == 0 ? gap :
        config_.host.static_wear_leveling_stop_gap;
    const auto required_gap = wear_leveling_engaged_by_stack_[stack] ? stop_gap : gap;
    if (hot_erase_count < required_gap) {
        wear_leveling_engaged_by_stack_[stack] = false;
        return;
    }
    if (hot_erase_count < config_.host.static_wear_leveling_start_erases) {
        return;
    }

    // Coldest closed block in the stack that still carries live pages. A
    // fully invalid block is ordinary GC work; static read-only and raw
    // extents are outside the FTL and cannot move. Ties prefer the block
    // with fewer live pages because it is cheaper to migrate.
    const auto pps = planes_per_stack();
    const auto block_begin = stack * pps * config_.device.blocks_per_plane;
    const auto block_end = (stack + 1) * pps * config_.device.blocks_per_plane;
    const auto& victim = gc_victim_by_stack_.at(stack);
    const auto cold_frontier = cold_block_by_stack_.at(stack);
    const auto erase_clock = wear_leveling_erase_clock(stack);
    std::optional<std::size_t> cold;
    for (std::size_t i = block_begin; i < block_end; ++i) {
        const auto& block = blocks_[i];
        if ((block.role != BlockRole::Data && block.role != BlockRole::Mapping &&
             block.role != BlockRole::GC) ||
            block.erase_pending || block.pending_program_pages != 0 ||
            block.pending_mapping_publications != 0 ||
            block.free_pages != 0 || block.valid_pages == 0 ||
            checked_add(
                static_cast<std::uint64_t>(block.valid_pages),
                static_cast<std::uint64_t>(block.invalid_pages),
                "HBF static wear-leveling candidate pages") !=
                config_.device.pages_per_block ||
            pending_block_transitions_.contains(i) ||
            (victim && victim->block == i) || cold_frontier == i) {
            continue;
        }
        if (cold) {
            const auto& best = blocks_[*cold];
            if (best.erase_count < block.erase_count ||
                (best.erase_count == block.erase_count &&
                 best.valid_pages <= block.valid_pages)) {
                continue;
            }
        }
        const auto& owner_plane = planes_.at(block_plane_index(i));
        if (owner_plane.active_data_block == i ||
            owner_plane.active_mapping_block == i ||
            owner_plane.active_gc_block == i) {
            continue;
        }
        const auto moved = wear_leveling_last_move_by_block_[i];
        if (moved && erase_clock - *moved < config_.host.static_wear_leveling_cooldown_erases) {
            stats_.static_wear_leveling_cooldown_exclusions++;
            continue;
        }
        cold = i;
    }
    if (!cold) {
        return;
    }
    const auto cold_count = blocks_[*cold].erase_count;
    if (checked_add(
            static_cast<std::uint64_t>(cold_count),
            static_cast<std::uint64_t>(required_gap),
            "HBF static wear-leveling gap") > hot_erase_count) {
        wear_leveling_engaged_by_stack_[stack] = false;
        return;
    }
    wear_leveling_engaged_by_stack_[stack] = true;
    const auto copy_bytes = static_cast<double>(blocks_[*cold].valid_pages) *
        config_.device.page_size_bytes;
    if (config_.host.static_wear_leveling_max_write_fraction > 0.0 &&
        wear_leveling_credit_by_stack_[stack] + 1e-6 < copy_bytes) {
        stats_.static_wear_leveling_budget_deferrals++;
        return;
    }
    // Capacity: cold data lands in the cold frontier (one whole block when
    // none is open yet), and its induced translation writebacks draw on the
    // GC pool, which must still hold the worst reclaimable victim afterwards
    // or GC could wedge behind the migration.
    const auto induced = mapping_allocation_pool_demand(stack,
        induced_translation_writebacks(*cold, dirty_cached_mapping_pages(stack)));
    const auto& open_cold = cold_block_by_stack_.at(stack);
    const std::uint64_t cold_block_pages =
        open_cold && blocks_.at(*open_cold).free_pages >= blocks_[*cold].valid_pages ?
        0 : config_.device.pages_per_block;
    if (gc_relocation_capacity(stack) <
        checked_add(
            checked_add(induced, cold_block_pages,
                "HBF static wear-leveling capacity check"),
            worst_gc_victim_demand(stack),
            "HBF static wear-leveling capacity check")) {
        return;
    }
    const auto minimum_destination_count = checked_add(
        static_cast<std::uint64_t>(cold_count),
        (static_cast<std::uint64_t>(required_gap) + 1) / 2,
        "HBF static wear-leveling destination floor");
    if (minimum_destination_count > std::numeric_limits<std::uint32_t>::max()) {
        return;
    }
    if (!ensure_cold_block(stack, static_cast<std::uint32_t>(minimum_destination_count))) {
        return;
    }
    std::optional<std::size_t> spare;
    if (blocks_[*cold_block_by_stack_[stack]].free_pages < blocks_[*cold].valid_pages) {
        const auto previous = cold_block_by_stack_[stack];
        cold_block_by_stack_[stack].reset();
        spare = ensure_cold_block(stack, static_cast<std::uint32_t>(minimum_destination_count));
        cold_block_by_stack_[stack] = previous;
        if (!spare) return;
    }
    double cursor_ns = at_ns;
    auto migration = start_relocation(*cold,
        RelocationPurpose::StaticWearLeveling,
        cursor_ns,
        breakdown,
        spans);
    migration->cold_destination_minimum_erases =
        static_cast<std::uint32_t>(minimum_destination_count);
    migration->cold_destination_spare = spare;
    if (config_.host.static_wear_leveling_max_write_fraction > 0.0) {
        wear_leveling_credit_by_stack_[stack] =
            std::max(0.0, wear_leveling_credit_by_stack_[stack] - copy_bytes);
    }
    wear_leveling_last_move_by_block_[*cold] = erase_clock;
    wear_leveling_by_stack_.at(stack) = std::move(migration);
}

void HbfController::advance_static_wear_leveling(double at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    std::size_t stack) {
    auto& migration = wear_leveling_by_stack_.at(stack);
    if (!migration) {
        return;
    }
    // Background work at the caller's cursor, paced like preventive GC and
    // never inside the hard-pressure loop.
    (void)relocate_live_pages(*migration, at_ns, breakdown, spans,
        config_.host.gc_relocation_pages_per_host_write);
    if (relocation_scan_complete(*migration)) {
        (void)schedule_relocation_reclaim(*migration, at_ns, breakdown, spans);
        migration.reset();
    }
}

std::uint64_t HbfController::stack_minimum_erase_count(std::size_t stack) const {
    const auto& histogram = managed_wear_by_stack_.at(stack);
    return histogram.empty() ? 0 : histogram.begin()->first;
}
} // namespace hbfsim::host
