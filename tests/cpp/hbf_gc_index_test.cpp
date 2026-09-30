#include "host/hbf_controller.hpp"
#include "../../verification/probes/hbf_with_hbm.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace hbfsim::host {
using namespace hbfsim::physical;

// The exhaustive policy exists only as an independent test oracle, never
// as a selectable production path. Synthetic selector states do not issue
// media requests; the replay below also checks actual mutation hooks.
struct HbfGcIndexTestAccess {
    using Role = HbfController::BlockRole;

    static HbfController::GcHeadroom headroom_oracle(
        const HbfController& d, std::size_t stack, Role role) {
        HbfController::GcHeadroom result;
        std::uint64_t free = 0, returning = 0;
        const auto pps = d.planes_per_stack();
        for (auto p = stack * pps; p < (stack + 1) * pps; ++p) {
            const auto& plane = d.planes_[p];
            const auto pages = [&](const auto& active, Role expected) -> std::uint64_t {
                if (!active) return 0;
                const auto& block = d.blocks_.at(*active);
                if (d.block_plane_index(*active) != p || block.role != expected ||
                    block.erase_pending || !block.free_pages)
                    throw std::runtime_error("oracle found invalid active block");
                return block.free_pages;
            };
            const auto data = pages(plane.active_data_block, Role::Data);
            const auto mapping = pages(plane.active_mapping_block, Role::Mapping);
            result.foreground_pages += role == Role::Data ? data : mapping;
            result.relocation_pages += pages(plane.active_gc_block, Role::GC);
            free += plane.free_blocks.size();
        }
        const auto ppb = d.config_.device.pages_per_block;
        result.relocation_pages += free * ppb;
        for (const auto& [block, transition] : d.pending_block_transitions_)
            if (transition.garbage_collection && d.stack_of_block(block) == stack) ++returning;
        const auto floor = d.gc_reserve_requirement_pages();
        const auto reserve = d.gc_reserve_pages(stack);
        const auto spare = [&](std::uint64_t pool, std::uint64_t blocks) {
            return pool < floor ? std::uint64_t{0} : std::min(blocks, (pool - floor) / ppb);
        };
        const auto above = [&](std::uint64_t pool) { return pool > reserve ? pool - reserve : 0; };
        const auto own = result.foreground_pages;
        const auto now = spare(result.relocation_pages, free);
        const auto after = result.relocation_pages + returning * ppb;
        result.foreground_pages += now * ppb;
        result.preventive_pages = own + above(result.relocation_pages);
        result.returning_pages = (spare(after, free + returning) - now) * ppb;
        result.returning_preventive_pages = above(after) - above(result.relocation_pages);
        return result;
    }

    static void check_headroom(HbfController& d) {
        const auto equal = [](const auto& a, const auto& b) {
            return a.relocation_pages == b.relocation_pages &&
                a.foreground_pages == b.foreground_pages && a.preventive_pages == b.preventive_pages &&
                a.returning_pages == b.returning_pages &&
                a.returning_preventive_pages == b.returning_preventive_pages;
        };
        for (std::size_t stack = 0; stack < d.config_.device.stacks; ++stack) {
            for (const auto role : {Role::Data, Role::Mapping}) {
                const auto expected = headroom_oracle(d, stack, role);
                if (!equal(expected, d.gc_headroom(stack, role)) ||
                    !equal(expected, d.gc_headroom(stack, role)))
                    throw std::runtime_error("cached GC headroom differs from full plane oracle");
            }
            if (!d.gc_headroom_by_stack_[stack].initialized ||
                !d.gc_headroom_by_stack_[stack].dirty_planes.empty())
                throw std::runtime_error("headroom query did not refresh its dirty planes");
        }
    }

    static std::uint64_t minimum(const HbfController& d, std::size_t stack) {
        auto result = std::numeric_limits<std::uint64_t>::max();
        const auto n = d.planes_per_stack() * d.config_.device.blocks_per_plane;
        for (auto i = stack * n; i < (stack + 1) * n; ++i) {
            const auto& b = d.blocks_[i];
            if (b.role != Role::StaticReadOnly && b.role != Role::RawPhysical)
                result = std::min(result, std::uint64_t{b.erase_count});
        }
        return result == std::numeric_limits<std::uint64_t>::max() ? 0 : result;
    }

    static std::uint64_t induced(const HbfController& d, std::size_t i) {
        const auto& b = d.blocks_[i];
        if (d.config_.host.mapping_mode != MappingMode::Cached || b.role == Role::Mapping)
            return 0;
        const auto overflow = b.valid_pages > d.mapping_cache_pages_per_stack_ ?
            b.valid_pages - d.mapping_cache_pages_per_stack_ : 0;
        return std::min<std::uint64_t>(b.valid_pages,
            d.dirty_mapping_pages_by_stack_[d.stack_of_block(i)] + overflow);
    }

    static std::uint64_t pool_demand(
        const HbfController& d, std::size_t stack, std::uint64_t pages) {
        const auto pps = d.planes_per_stack();
        for (auto p = stack * pps; p < (stack + 1) * pps; ++p)
            if (const auto active = d.planes_[p].active_mapping_block)
                pages -= std::min<std::uint64_t>(pages, d.blocks_[*active].free_pages);
        const auto ppb = d.config_.device.pages_per_block;
        return ((pages + ppb - 1) / ppb) * ppb;
    }

    static std::optional<std::size_t> exhaustive(const HbfController& d, std::size_t stack) {
        const auto min_wear = minimum(d, stack);
        auto capacity = d.gc_headroom(stack, Role::Data).relocation_pages;
        const auto& migration = d.wear_leveling_by_stack_[stack];
        if (migration) {
            const auto owed = pool_demand(d, stack, induced(d, migration->block));
            capacity = capacity > owed ? capacity - owed : 0;
        }
        const auto& victim = d.gc_victim_by_stack_[stack];
        const auto cold = d.cold_block_by_stack_[stack];
        const auto n = d.planes_per_stack() * d.config_.device.blocks_per_plane;
        const auto pivot = stack * n + d.next_gc_allocation_plane_per_stack_[stack] *
            d.config_.device.blocks_per_plane;
        const auto rank = [&](std::size_t i) { return (i + n - pivot) % n; };
        std::optional<std::size_t> best;
        double best_score = -std::numeric_limits<double>::infinity();
        for (auto i = stack * n; i < (stack + 1) * n; ++i) {
            const auto& b = d.blocks_[i];
            const auto& p = d.planes_[d.block_plane_index(i)];
            if (b.role == Role::Free || b.role == Role::StaticReadOnly ||
                b.role == Role::RawPhysical || b.erase_pending ||
                d.pending_block_transitions_.contains(i) || b.pending_program_pages ||
                b.pending_mapping_publications || b.free_pages || !b.invalid_pages ||
                p.active_data_block == i || p.active_mapping_block == i ||
                p.active_gc_block == i || cold == i ||
                (victim && victim->block == i) || (migration && migration->block == i))
                continue;
            const auto checkpoints = induced(d, i);
            const auto demand = b.role == Role::Mapping ? pool_demand(d, stack, b.valid_pages) :
                b.valid_pages + pool_demand(d, stack, checkpoints);
            if (demand > capacity) continue;
            const double score = static_cast<double>(b.invalid_pages) -
                static_cast<double>(checkpoints) - d.config_.host.gc_wear_leveling_weight *
                    static_cast<double>(b.erase_count - min_wear);
            if (!best || score > best_score || (score == best_score && rank(i) < rank(*best))) {
                best = i;
                best_score = score;
            }
        }
        return best;
    }

    static void check(HbfController& d) {
        check_headroom(d);
        for (std::size_t stack = 0; stack < d.config_.device.stacks; ++stack) {
            if (minimum(d, stack) != d.stack_minimum_erase_count(stack))
                throw std::runtime_error("incremental managed-wear minimum differs");
            const auto expected = exhaustive(d, stack);
            if (expected != d.choose_gc_victim(stack) || expected != d.choose_gc_victim(stack))
                throw std::runtime_error("indexed GC differs from exhaustive policy");
        }
    }

    static HbfConfig config(bool cached, double weight, bool full = false) {
        HbfConfig c;
        c.device.stacks = full ? 1 : 2;
        c.device.channels_per_stack = 1;
        c.device.dies_per_channel = 1;
        c.device.planes_per_die = full ? 16 : 4;
        c.device.blocks_per_plane = full ? 32768 : 32;
        c.device.pages_per_block = full ? 256 : 16;
        c.device.page_size_bytes = 4096;
        c.host.mapping_entries_per_page = full ? 512 : 16;

        c.host.write_coalescing_enabled = false;
        c.host.gc_wear_leveling_weight = weight;
        c.host.mapping_mode = cached ? MappingMode::Cached : MappingMode::FullResident;
        if (cached) {
            const auto pages = std::uint64_t{c.device.planes_per_die} * c.device.blocks_per_plane * c.device.pages_per_block;
            c.host.ctrl_dram_bytes = pages / c.host.mapping_entries_per_page *
                c.host.mapping_directory_entry_bytes + 8 * c.device.page_size_bytes;
        }
        return c;
    }

    static void change(HbfController& d, std::size_t i, std::mt19937_64& rng) {
        auto& b = d.blocks_[i];
        d.remove_managed_block_wear(i);
        b.role = static_cast<Role>(1 + rng() % 5);
        b.free_pages = 0;
        b.next_page = d.config_.device.pages_per_block;
        b.valid_pages = rng() % d.config_.device.pages_per_block;
        b.invalid_pages = d.config_.device.pages_per_block - b.valid_pages;
        b.pending_program_pages = 0;
        if (rng() % 8 == 0) {
            b.pending_program_pages = 1;
            --b.invalid_pages;
        }
        b.pending_mapping_publications = rng() % 8 == 0 ? 1 : 0;
        b.erase_pending = rng() % 8 == 0;
        b.erase_count = rng() % 128;
        d.add_managed_block_wear(i);
        d.mark_gc_candidate_dirty(i);
        // This selector-only fixture edits private state directly. Production
        // mutations go through the allocator/role hooks checked by replay.
        d.mark_gc_headroom_dirty(d.block_plane_index(i));
    }

    static void populate(HbfController& d, std::mt19937_64& rng) {
        const auto bpp = d.config_.device.blocks_per_plane;
        for (std::size_t p = 0; p < d.planes_.size(); ++p) {
            auto& plane = d.planes_[p];
            plane.free_blocks.clear();
            for (std::size_t b = 0; b < bpp - 2; ++b) change(d, p * bpp + b, rng);
            // A partially used mapping frontier plus one free GC block.
            const auto i = p * bpp + bpp - 2;
            auto& frontier = d.blocks_[i];
            frontier.role = Role::Mapping;
            frontier.free_pages = d.config_.device.pages_per_block / 2;
            frontier.valid_pages = frontier.free_pages;
            frontier.next_page = frontier.valid_pages;
            plane.active_mapping_block = i;
            plane.free_blocks.push_back(i + 1);
            d.mark_gc_headroom_dirty(p);
        }
    }

    static void randomized() {
        for (const auto cached : {false, true})
            for (const double weight : {0.0, 0.05, 1.0, 17.0, 1e-18, 1e308}) {
                HbfController d(config(cached, weight));
                std::mt19937_64 rng(9173);
                populate(d, rng);
                for (unsigned step = 0; step < 1500; ++step) {
                    const auto plane = rng() % d.planes_.size();
                    const auto i = plane * d.config_.device.blocks_per_plane +
                        rng() % (d.config_.device.blocks_per_plane - 2);
                    change(d, i, rng);
                    const auto stack = d.stack_of_block(i);
                    d.dirty_mapping_pages_by_stack_[stack] = rng() % 9;
                    d.next_gc_allocation_plane_per_stack_[stack] = rng() % d.planes_per_stack();
                    // Pins change without changing the candidate's bucket.
                    d.cold_block_by_stack_[stack] = i;
                    d.gc_victim_by_stack_[stack] = HbfController::ActiveRelocation{.block = i};
                    d.wear_leveling_by_stack_[stack] = HbfController::ActiveRelocation{.block = i};
                    d.pending_block_transitions_[i] = {};
                    check(d);
                    d.pending_block_transitions_.erase(i);
                    d.cold_block_by_stack_[stack].reset();
                    d.gc_victim_by_stack_[stack].reset();
                    d.wear_leveling_by_stack_[stack].reset();
                    check(d);
                }
            }
    }

    static void round_robin_ties() {
        HbfController d(config(false, 0.05));
        const auto pps = d.planes_per_stack();
        const auto bpp = d.config_.device.blocks_per_plane;
        for (std::size_t plane = 0; plane < d.planes_.size(); ++plane) {
            // Two equally reclaimable, closed blocks on every plane. This
            // selector fixture deliberately does not issue media operations.
            for (std::size_t offset = 0; offset < 2; ++offset) {
                const auto id = plane * bpp + offset;
                std::erase(d.planes_[plane].free_blocks, id);
                auto& block = d.blocks_[id];
                block.role = Role::Data;
                block.free_pages = 0;
                block.next_page = d.config_.device.pages_per_block;
                block.invalid_pages = d.config_.device.pages_per_block;
                d.mark_gc_candidate_dirty(id);
                d.mark_gc_headroom_dirty(plane);
            }
        }
        for (std::size_t round = 0; round < 2 * pps; ++round) {
            for (std::size_t stack = 0; stack < d.config_.device.stacks; ++stack) {
                // A page-striped writer can return to the same allocation
                // phase at every whole-block reclaim. GC must rotate anyway.
                d.next_data_allocation_plane_per_stack_[stack] = 0;
                const auto chosen = d.choose_gc_victim(stack);
                if (!chosen || chosen != d.choose_gc_victim(stack) ||
                    d.block_plane_index(*chosen) != stack * pps + round % pps)
                    throw std::runtime_error("GC ties repeatedly selected one plane or preview moved its cursor");
                double at = 0;
                Breakdown ignored;
                (void)d.start_relocation(*chosen,
                    HbfController::RelocationPurpose::GarbageCollection, at, ignored, nullptr);
                d.blocks_[*chosen].erase_pending = true;
            }
        }
    }

    static void replay() {
        for (const auto cached : {false, true}) {
            auto c = config(cached, 0.05);
            verification::HbfWithHbm d(c);
            const auto footprint = d.logical_capacity_pages() * 3 / 4;
            d.prepopulate_mutable_logical_page_range(0, footprint);
            std::mt19937_64 rng(1781);
            double at = 0;
            const TraceConfig trace{.mode = TraceMode::Off, .retain_completion_diagnostics = false};
            for (unsigned step = 0; step < 4000; ++step) {
                at = d.issue(PhysicalRequest{.id = "gc-index", .tier = Tier::HBF,
                    .op = Op::Write, .address_space = AddressSpace::Logical,
                    .trace = trace, .arrival_ns = at,
                    .addr = (rng() % footprint) * 4096, .bytes = 4096}).finish_ns;
                check(d);
            }
            (void)d.drain_pending("gc-index-drain", at, trace);
            check(d);
            if (!d.stats().accounting_verified || !d.stats().gc_runs)
                throw std::runtime_error("indexed GC replay did not exercise audited GC");
        }
    }

    static void headroom_boundaries() {
        auto c = config(false, 0.05);
        c.device.blocks_per_plane = 8;
        c.device.pages_per_block = 8;
        HbfController d(c);
        check_headroom(d);
        const auto block = d.allocate_block_from_plane(0, Role::Data);
        d.planes_[0].active_data_block = block;
        if (d.gc_headroom_by_stack_[0].dirty_planes.size() != 1 ||
            !d.gc_headroom_by_stack_[1].dirty_planes.empty())
            throw std::runtime_error("one allocation dirtied unrelated headroom planes");
        for (unsigned page = 0; page < c.device.pages_per_block - 1; ++page) {
            (void)d.allocate_page_from_block(block);
            check_headroom(d);
        }
        const auto reject = [&] {
            bool rejected = false;
            try { (void)d.gc_headroom(0, Role::Data); }
            catch (const std::runtime_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("cached headroom skipped active-block validation");
        };
        d.set_block_role(block, Role::Mapping); reject();
        d.set_block_role(block, Role::Data); check_headroom(d);
        d.blocks_[block].erase_pending = true;
        d.mark_gc_headroom_dirty(0); reject();
        d.blocks_[block].erase_pending = false;
        d.mark_gc_headroom_dirty(0); check_headroom(d);
        d.blocks_[block].free_pages = 0;
        d.mark_gc_headroom_dirty(0); reject();
        d.blocks_[block].free_pages = 1;
        d.mark_gc_headroom_dirty(0); check_headroom(d);
        d.planes_[0].active_mapping_block = block;
        d.mark_gc_headroom_dirty(0); reject();
        d.planes_[0].active_mapping_block.reset();
        d.mark_gc_headroom_dirty(0); check_headroom(d);
        d.planes_[0].active_data_block = c.device.blocks_per_plane;
        d.mark_gc_headroom_dirty(0); reject();
        d.planes_[0].active_data_block = block;
        d.mark_gc_headroom_dirty(0); check_headroom(d);
        (void)d.allocate_page_from_block(block);
        d.planes_[0].active_data_block.reset();
        check_headroom(d);
        bool rejected = false;
        try { (void)d.gc_headroom(0, Role::GC); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("cached headroom accepted an invalid allocation role");

        // Returning capacity is deliberately not cached: adding/removing an
        // in-flight reclaim must change it even with no dirty plane.
        d.pending_block_transitions_[block] = {.garbage_collection = true};
        check_headroom(d);
        if (!d.gc_headroom(0, Role::Data).returning_pages)
            throw std::runtime_error("headroom fixture did not expose returning capacity");
        d.pending_block_transitions_.erase(block);
        check_headroom(d);
        if (d.gc_headroom(0, Role::Data).returning_pages)
            throw std::runtime_error("headroom retained a completed reclaim");
    }

    static void headroom_replay() {
        for (const auto cached : {false, true}) {
            auto c = config(cached, 0.05);
            c.device.blocks_per_plane = 8;
            c.device.pages_per_block = 8;
            c.host.mapping_entries_per_page = 8;
            verification::HbfWithHbm d(c);
            check_headroom(d); // Prime before fencing and compact population.
            d.reserve_static_physical_block_indices({c.device.blocks_per_plane - 1});
            check_headroom(d);
            const auto footprint = d.logical_capacity_pages() * 3 / 4;
            d.prepopulate_mutable_logical_page_range(0, footprint);
            if (!d.compact_logical_image_)
                throw std::runtime_error("headroom replay did not use compact population");
            check_headroom(d);
            std::mt19937_64 rng(88021);
            double at = 0;
            const TraceConfig trace{.mode = TraceMode::Off, .retain_completion_diagnostics = false};
            for (unsigned step = 0; step < 640; ++step) {
                at = d.issue(PhysicalRequest{.id = "headroom", .tier = Tier::HBF,
                    .op = Op::Write, .address_space = AddressSpace::Logical,
                    .trace = trace, .arrival_ns = at,
                    .addr = (rng() % footprint) * 4096, .bytes = 4096}).finish_ns;
                check_headroom(d);
                if (step % 41 == 40) {
                    // Dirty mapping/GC work can extend the administrative
                    // frontier beyond this write's response completion.
                    at = std::max(at, d.stats_.finish_ns);
                    at = d.invalidate_logical_pages(rng() % footprint, 1, at).completion.finish_ns;
                    check_headroom(d);
                }
            }
            at = d.drain_pending("headroom-drain", at, trace).finish_ns;
            check_headroom(d);
            if (!d.stats().accounting_verified || !d.stats().gc_runs || !d.stats().gc_relocations)
                throw std::runtime_error("headroom replay did not exercise audited relocation/reclaim");
            const auto image = d.persistent_image();
            verification::HbfWithHbm restored(c);
            check_headroom(restored); // Restore must discard even a primed fresh cache.
            restored.restore_persistent_image(image);
            check_headroom(restored);
            for (std::size_t stack = 0; stack < c.device.stacks; ++stack)
                for (const auto role : {Role::Data, Role::Mapping}) {
                    const auto before = d.gc_headroom(stack, role), after = restored.gc_headroom(stack, role);
                    if (before.foreground_pages != after.foreground_pages ||
                        before.relocation_pages != after.relocation_pages ||
                        before.preventive_pages != after.preventive_pages)
                        throw std::runtime_error("restore changed GC allocator headroom");
                }
            (void)restored.issue(PhysicalRequest{.id = "headroom-restored-write", .tier = Tier::HBF,
                .op = Op::Write, .address_space = AddressSpace::Logical, .trace = trace,
                .addr = 0, .bytes = 4096});
            check_headroom(restored);
        }
        auto c = config(false, 0.05);
        c.device.blocks_per_plane = 8;
        c.device.pages_per_block = 8;
        verification::HbfWithHbm physical(c);
        check_headroom(physical);
        const auto before = physical.gc_headroom(0, Role::Data).relocation_pages;
        const auto erased = physical.issue(PhysicalRequest{.id = "headroom-free-erase", .tier = Tier::HBF,
            .op = Op::Erase, .address_space = AddressSpace::Physical, .addr = 0, .bytes = 4096});
        check_headroom(physical);
        if (physical.gc_headroom(0, Role::Data).relocation_pages + c.device.pages_per_block != before)
            throw std::runtime_error("pending erase left a free block allocatable");
        (void)physical.drain_pending("headroom-erase-commit", erased.finish_ns);
        check_headroom(physical);
        if (physical.gc_headroom(0, Role::Data).relocation_pages != before)
            throw std::runtime_error("erase completion did not return its free block");
        verification::HbfWithHbm raw(c);
        check_headroom(raw);
        raw.reserve_raw_physical_block_extent(c.device.blocks_per_plane - 1, 1);
        check_headroom(raw);
        const auto programmed = raw.issue(PhysicalRequest{.id = "headroom-raw-write", .tier = Tier::HBF,
            .op = Op::Write, .address_space = AddressSpace::Physical,
            .addr = (c.device.blocks_per_plane - 1) * c.device.pages_per_block * 4096ULL, .bytes = 4096});
        check_headroom(raw);
        (void)raw.drain_pending("headroom-raw-commit", programmed.finish_ns);
        check_headroom(raw);
    }

    static void benchmark() {
        verification::HbfWithHbm d(config(true, 0.05, true));
        std::mt19937_64 rng(9173);
        populate(d, rng);
        check(d); // Build the index before the steady-state comparison.
        constexpr unsigned iterations = 100;
        auto run = [&](bool indexed) {
            std::uint64_t checksum = 0;
            const auto begin = std::chrono::steady_clock::now();
            for (unsigned step = 0; step < iterations; ++step) {
                d.dirty_mapping_pages_by_stack_[0] = step % 9;
                checksum += (indexed ? d.choose_gc_victim(0) : exhaustive(d, 0)).value_or(0);
            }
            const auto seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin).count();
            std::cout << (indexed ? "indexed" : "exhaustive") << " seconds=" << seconds
                      << " selections=" << iterations << " checksum=" << checksum << '\n';
            return checksum;
        };
        if (run(false) != run(true)) throw std::runtime_error("benchmark choices differ");
    }
};
} // namespace hbfsim::host

int main(int argc, char** argv) {
    try {
        using Access = hbfsim::host::HbfGcIndexTestAccess;
        if (argc == 2 && std::string(argv[1]) == "--benchmark") Access::benchmark();
        else if (argc == 2 && std::string(argv[1]) == "--headroom") {
            Access::headroom_boundaries();
            Access::headroom_replay();
            std::cout << "GC headroom cache equivalence passed\n";
        }
        else {
            Access::headroom_boundaries();
            Access::headroom_replay();
            Access::round_robin_ties();
            Access::randomized();
            Access::replay();
            std::cout << "GC index equivalence passed\n";
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
