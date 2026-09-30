#include "host/hbf_controller.hpp"

#include <cmath>
#include <numeric>

namespace hbfsim::host {
using namespace hbfsim::physical;
using namespace physical;


std::uint64_t HbfController::zone_index(std::uint32_t stack, std::uint32_t channel, std::uint64_t zone) const {
    if (mapping_policy_) throw std::invalid_argument("host mapping owns the media; zone administration is unavailable");
    const auto per_channel = blocks_.size() /
        (static_cast<std::uint64_t>(config_.device.stacks) * config_.device.channels_per_stack) /
        config_.host.zone_size_blocks;
    if (stack >= config_.device.stacks || channel >= config_.device.channels_per_stack || zone >= per_channel)
        throw std::runtime_error("HBF zone is outside its host channel");
    return (static_cast<std::uint64_t>(stack) * config_.device.channels_per_stack + channel) *
        per_channel + zone;
}

std::uint64_t HbfController::physical_zone(std::uint64_t zone) const {
    return media_->physical_zone(zone);
}

bool HbfController::zone_available(std::uint64_t physical, bool invalid) const {
    const auto begin = physical * config_.host.zone_size_blocks;
    for (auto b = begin; b < begin + config_.host.zone_size_blocks; ++b) {
        const auto& block = blocks_.at(b);
        if ((block.role != BlockRole::Free && block.role != BlockRole::RawPhysical) ||
            (block.role == BlockRole::Free && logical_capacity_frozen_) ||
            block.erase_pending || block.pending_program_pages || block.pending_mapping_publications ||
            (invalid && block.valid_pages)) return false;
    }
    return true;
}

void HbfController::claim_zone(std::uint64_t physical) {
    const auto begin = physical * config_.host.zone_size_blocks;
    for (auto b = begin; b < begin + config_.host.zone_size_blocks; ++b) {
        auto& block = blocks_.at(b);
        if (block.role != BlockRole::Free) continue;
        auto& free = planes_.at(block_plane_index(b)).free_blocks;
        const auto position = std::find(free.begin(), free.end(), b);
        if (position == free.end()) throw std::runtime_error("HBF zone free block has no pool entry");
        free.erase(position);
        remove_managed_block_wear(b);
        set_block_role(b, BlockRole::RawPhysical);
    }
    zone_managed_ = true;
}

void HbfController::zone_barrier(double at_ns) {
    // Administrative operations are explicit quiescent host barriers. This
    // conservative model fences all issued IO, including reads served by SRAM.
    if (!std::isfinite(at_ns) || at_ns < 0 || at_ns < stats_.finish_ns ||
        (last_issue_arrival_ns_ && at_ns < *last_issue_arrival_ns_))
        throw std::runtime_error("HBF zone management requires a completed IO frontier");
    materialize_committed_state_through(at_ns);
    seed_media_image();
    last_issue_arrival_ns_ = at_ns;
}

PhysicalCompletion HbfController::issue_channel_local(const PhysicalRequest& request) {
    if (mapping_policy_) throw std::invalid_argument("host mapping owns the media; use logical requests");
    const auto page_size = config_.device.page_size_bytes;
    const auto zone_bytes = static_cast<std::uint64_t>(config_.host.zone_size_blocks) *
        config_.device.pages_per_block * page_size;
    const auto capacity = total_pages_ * page_size;
    if (request.tier != Tier::HBF || (request.op != Op::Read && request.op != Op::Write) ||
        request.bytes == 0 || request.addr >= capacity || request.bytes > capacity - request.addr ||
        !std::isfinite(request.arrival_ns) || request.arrival_ns < 0 ||
        (last_issue_arrival_ns_ && request.arrival_ns < *last_issue_arrival_ns_))
        throw std::runtime_error("invalid HBF channel-local IO");
    // One request stays within a zone; the host splits larger transfers at
    // zone boundaries so a remap can never silently make a range contiguous.
    if (request.addr / zone_bytes != (request.addr + request.bytes - 1) / zone_bytes)
        throw std::runtime_error("split HBF channel IO at zone boundaries");
    if (request.op == Op::Write &&
        (request.addr % page_size || request.bytes % page_size))
        throw std::runtime_error("HBF channel writes require complete NAND pages");
    if (request.op == Op::Read &&
        (request.addr % 64 || request.bytes % 64 || request.bytes > page_size ||
         request.addr / page_size != (request.addr + request.bytes - 1) / page_size))
        throw std::runtime_error("HBF channel reads require aligned 64-byte units within one page");
    const auto physical = physical_zone(request.addr / zone_bytes);
    const auto address = physical * zone_bytes + request.addr % zone_bytes;
    materialize_committed_state_through(request.arrival_ns);
    if (!zone_available(physical, false))
        throw std::runtime_error("HBF channel zone overlaps owned media or pending IO");
    if (request.op == Op::Read) {
        for (auto ppn = address / page_size; ppn <= (address + request.bytes - 1) / page_size; ++ppn) {
            if (!blocks_[ppn / config_.device.pages_per_block].is_valid(ppn % config_.device.pages_per_block))
                throw std::runtime_error("HBF read from invalid/erased zone; rewrite data before reading");
        }
    } else {
        // Validate every target before claiming any part of the zone.
        for (auto ppn = address / page_size; ppn < (address + request.bytes) / page_size; ++ppn) {
            const auto& block = blocks_[ppn / config_.device.pages_per_block];
            const auto first = std::max(address / page_size,
                ppn / config_.device.pages_per_block * config_.device.pages_per_block);
            if (first % config_.device.pages_per_block != block.next_page)
                throw std::runtime_error("HBF zone writes must append in NAND page order; reset before reuse");
        }
        claim_zone(physical);
    }
    auto raw = request;
    raw.address_space = AddressSpace::Physical;
    raw.addr = address;
    auto result = issue(raw);
    result.resource_path = "host/channel-zone" + std::to_string(request.addr / zone_bytes) +
        "->physical-zone" + std::to_string(physical) + "/" + result.resource_path;
    return result;
}

std::vector<HbfController::ChannelWriteCompletion> HbfController::issue_channel_write_batch(
    const std::vector<PhysicalRequest>& requests) {
    if (mapping_policy_) throw std::invalid_argument("host mapping owns the media; use logical requests");
    if (requests.empty()) throw std::invalid_argument("HBF fragment batch must contain commands");
    if (mapping_is_local())
        throw std::invalid_argument("device-local mapping accepts logical requests only");
    const auto page_bytes = config_.device.page_size_bytes;
    const auto zone_bytes = std::uint64_t{config_.host.zone_size_blocks} *
        config_.device.pages_per_block * page_bytes;
    const auto capacity = total_pages_ * page_bytes;
    std::vector<std::uint64_t> addresses;
    std::set<std::uint64_t> zones;
    std::set<std::string> ids;
    double previous = std::max(stats_.finish_ns, last_issue_arrival_ns_.value_or(0));
    // Reject malformed batches before reserving transport or claiming media.
    for (const auto& request : requests) {
        if (request.id.empty() || !ids.insert(request.id).second ||
            request.tier != Tier::HBF || request.op != Op::Write ||
            !std::isfinite(request.arrival_ns) || request.arrival_ns < previous ||
            !request.bytes || request.addr >= capacity || request.bytes > capacity - request.addr ||
            request.addr % 64 || request.bytes % 64 ||
            request.bytes > page_bytes - request.addr % page_bytes)
            throw std::invalid_argument("HBF fragment batch requires ordered arrivals and aligned writes within one page at a completed IO frontier");
        previous = request.arrival_ns;
        const auto zone = physical_zone(request.addr / zone_bytes);
        zones.insert(zone);
        addresses.push_back(zone * zone_bytes + request.addr % zone_bytes);
    }
    materialize_committed_state_through(requests.front().arrival_ns);
    for (const auto zone : zones)
        if (!zone_available(zone, false))
            throw std::invalid_argument("HBF fragment batch overlaps media owned by another host path");
    seed_media_image();
    for (const auto zone : zones) claim_zone(zone);

    std::vector<ChannelWriteCompletion> results(requests.size());
    // Process host arrivals, actual HBIO/SRAM receipts, and device timers in
    // time order. Reserving all transfers before executing NAND would let
    // future fragments change resource contention for an earlier program.
    std::map<std::pair<double, std::size_t>, bool> receipts;
    std::size_t next = 0;
    double frontier = requests.front().arrival_ns;
    const auto respond = [&](const physical::hbf::HbfDevice::WriteResponse& response) {
        auto& result = results.at(response.command);
        result.status = response.status;
        result.additional_status = response.additional_status;
        result.completion.finish_ns = response.finish_ns;
        result.completion.note = response.status == 0 ? "nonposted-page-program-complete" :
            response.status == 0x5 ? "write-accumulation-timeout" : "write-command-rejected";
    };
    while (next < requests.size() || !receipts.empty() || media_->pending_write_pages()) {
        const auto never = std::numeric_limits<double>::infinity();
        frontier = std::min({next < requests.size() ? requests[next].arrival_ns : never,
            receipts.empty() ? never : receipts.begin()->first.first, media_->next_write_event_ns()});
        if (!std::isfinite(frontier)) throw std::logic_error("HBF fragment batch lost a device event");
        reservation_causal_watermark_ns_ = frontier;
        for (const auto& response : media_->advance_writes(frontier)) respond(response);
        materialize_committed_state_through(frontier);
        media_->advance_cache(frontier);
        while (next < requests.size() && requests[next].arrival_ns == frontier) {
            const auto& request = requests[next];
            auto& result = results[next];
            auto& out = result.completion;
            out.id = request.id;
            out.tier = Tier::HBF;
            out.op = Op::Write;
            out.arrival_ns = out.start_ns = request.arrival_ns;
            out.logical_bytes = request.bytes;
            out.resource_path = "host/channel-zone" + std::to_string(request.addr / zone_bytes) +
                "/base-die-write-buffer/" + decode(addresses[next]).path();
            auto* spans = trace_spans_enabled(request.trace) ? &out.spans : nullptr;
            const auto channel = media_->channel_for_page(addresses[next] / page_bytes);
            const auto stack = channel / config_.device.channels_per_stack;
            const auto dispatch = reserve(frontier, config_.device.logic_scheduler_issue_ns,
                logic_dies_.at(stack).ingress);
            out.start_ns = dispatch.start_ns;
            out.breakdown.ingress_queue_wait_ns += dispatch.wait_ns;
            out.breakdown.command_ns += config_.device.logic_scheduler_issue_ns;
            if (spans)
                add_trace_span(spans, "logic_scheduler_issue", "logic",
                    "stack" + std::to_string(stack) + "/logic",
                    dispatch.start_ns, dispatch.finish_ns);
            const auto command_done = schedule_external_request_command(channel, dispatch.finish_ns,
                out.breakdown, spans, "write fragment", "user");
            result.received_ns = schedule_external_write_ingress(channel, request.bytes,
                command_done, out.breakdown, spans, "write fragment", "ocp_fragment_ingress");
            receipts.emplace(std::pair{result.received_ns, next}, false);
            ++stats_.program_requests;
            stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, frontier);
            ++next;
        }
        while (!receipts.empty() && receipts.begin()->first.first == frontier) {
            const auto index = receipts.begin()->first.second;
            receipts.erase(receipts.begin());
            const auto& request = requests[index];
            auto& out = results[index].completion;
            const auto ppn = addresses[index] / page_bytes;
            const auto block = ppn / config_.device.pages_per_block;
            // The host zone lifecycle still requires reset before reusing
            // page zero. A timeout has not allocated/advanced this cursor.
            const bool accumulating = media_->buffered_write_read_status(ppn, 0, 64).has_value();
            if (!accumulating && ppn % config_.device.pages_per_block != blocks_[block].next_page) {
                respond({index, 0x6, 0x1, frontier});
                continue;
            }
            const auto admitted = media_->receive_write_fragment(ppn,
                static_cast<std::uint32_t>(addresses[index] % page_bytes),
                static_cast<std::uint32_t>(request.bytes), index, frontier);
            if (admitted.status) {
                respond({index, admitted.status, admitted.additional_status, frontier});
                continue;
            }
            if (!admitted.page_ready) continue;
            reserve_physical_program_range(ppn, 1);
            auto* spans = trace_spans_enabled(request.trace) ? &out.spans : nullptr;
            auto ready = std::max(frontier, blocks_[block].issued_media_ready_ns);
            if (media_->prepare_program(ppn)) {
                ++stats_.auto_erase_requests;
                ready = schedule_erase_block(block, ready, out.breakdown, spans,
                    TransactionSource::User, request.heatmap_source, true);
            }
            const auto done = schedule_buffered_program_page(ppn, ready, out.breakdown,
                spans, TransactionSource::User, request.heatmap_source);
            schedule_physical_program_commit(ppn, done);
            out.physical_bytes = page_bytes;
            stats_.physical_write_bytes += page_bytes;
            stats_.data_program_payload_bytes += page_bytes;
            stats_.raw_physical_program_payload_bytes += page_bytes;
            ++stats_.data_programs;
            ++stats_.raw_physical_programs;
            ++stats_.page_programs;
        }
    }
    for (const auto& result : results) stats_.stage_work += result.completion.breakdown;
    stats_.free_pages = free_pages_;
    stats_.finish_ns = std::max(stats_.finish_ns, frontier);
    last_issue_arrival_ns_ = frontier;
    return results;
}

PhysicalCompletion HbfController::invalidate_zone(std::uint32_t stack, std::uint32_t channel, std::uint64_t zone, double at_ns) {
    const auto physical = physical_zone(zone_index(stack, channel, zone));
    zone_barrier(at_ns);
    if (!zone_available(physical, false))
        throw std::runtime_error("cannot invalidate a zone owned by the logical FTL or static image");
    claim_zone(physical);
    const auto begin = physical * config_.host.zone_size_blocks;
    for (auto b = begin; b < begin + config_.host.zone_size_blocks; ++b) {
        read_buffer_purge_block(b);
        if (blocks_[b].initial_raw_pages) {
            auto& block = blocks_[b];
            stats_.invalidations += block.valid_pages;
            block.invalid_pages += block.valid_pages;
            block.valid_pages = block.initial_raw_pages = 0;
            block.valid_bitmap.reset();
            continue;
        }
        for (std::uint32_t page = 0; page < config_.device.pages_per_block; ++page) {
            if (blocks_[b].is_valid(page)) invalidate_ppn(b * config_.device.pages_per_block + page);
        }
    }
    ++stats_.host_zone_invalidations;
    return PhysicalCompletion{.tier = Tier::HBF, .op = Op::Erase,
        .arrival_ns = at_ns, .start_ns = at_ns, .finish_ns = at_ns,
        .note = "host-invalidated-zone; no media erase or copy"};
}

PhysicalCompletion HbfController::remap_zones(std::uint32_t stack, std::uint32_t channel, std::uint64_t first,
    std::uint64_t second, double at_ns) {
    const auto a = zone_index(stack, channel, first);
    const auto b = zone_index(stack, channel, second);
    if (a == b) throw std::runtime_error("HBF remap needs two distinct zones");
    zone_barrier(at_ns);
    const auto pa = physical_zone(a), pb = physical_zone(b);
    if (!zone_available(pa, true) || !zone_available(pb, true))
        throw std::runtime_error("OCP zone remap requires both zones invalid with no outstanding IO");
    claim_zone(pa);
    claim_zone(pb);
    Breakdown command_work;
    const double command_done = schedule_external_request_command(stack * config_.device.channels_per_stack + channel, at_ns,
        command_work, nullptr, "host zone remap registers", "host");
    const double finish = command_done + config_.host.host_zone_remap_ns;
    media_->execute_zone_remap({.stack = stack, .channel = channel, .first = first, .second = second});
    control_ready_ns_[stack] = std::max(control_ready_ns_[stack], finish);
    last_issue_arrival_ns_ = finish;
    stats_.finish_ns = std::max(stats_.finish_ns, finish);
    ++stats_.host_zone_remaps;
    stats_.host_gc_control_ns += config_.host.host_zone_remap_ns;
    PhysicalCompletion out{.tier = Tier::HBF, .op = Op::Erase,
        .arrival_ns = at_ns, .start_ns = at_ns, .finish_ns = finish,
        .note = "HBF zone-remap opcode 0x8/type 0x00; BUCC.WLS=10b; zero media copies; PEC stays physical"};
    out.breakdown = command_work;
    out.breakdown.maintenance_ns = config_.host.host_zone_remap_ns;
    stats_.stage_work += out.breakdown;
    return out;
}

PhysicalCompletion HbfController::reset_zone(std::uint32_t stack, std::uint32_t channel, std::uint64_t zone, double at_ns) {
    const auto local = zone_index(stack, channel, zone);
    zone_barrier(at_ns);
    const auto physical = physical_zone(local);
    if (!zone_available(physical, true))
        throw std::runtime_error("host must invalidate the complete zone before reset");
    claim_zone(physical);
    // This is host bookkeeping, not an OCP erase or remap command. The next
    // page-zero write causes physical autoerase and is counted exactly once.
    for (auto b = physical * config_.host.zone_size_blocks;
         b < (physical + 1) * config_.host.zone_size_blocks; ++b)
        release_invalid_block(b);
    claim_zone(physical);
    ++stats_.host_zone_resets;
    return PhysicalCompletion{.tier = Tier::HBF, .op = Op::Erase,
        .arrival_ns = at_ns, .start_ns = at_ns, .finish_ns = at_ns,
        .note = "host-zone-reset; no media erase and no automatic remap"};
}

PhysicalCompletion HbfController::recycle_zone(std::uint32_t stack, std::uint32_t channel, std::uint64_t zone, double at_ns) {
    const auto local = zone_index(stack, channel, zone);
    zone_barrier(at_ns);
    const auto physical = physical_zone(local);
    if (!zone_available(physical, true))
        throw std::runtime_error("host must invalidate the complete zone before recycling");
    const auto per_channel = blocks_.size() /
        (static_cast<std::uint64_t>(config_.device.stacks) * config_.device.channels_per_stack) /
        config_.host.zone_size_blocks;
    const auto channel_begin = local / per_channel * per_channel;
    auto cold_local = local;
    auto cold_pec = media_->zone_pec_sum(local);
    const auto hot_pec = cold_pec;
    for (auto candidate = channel_begin; candidate < channel_begin + per_channel; ++candidate) {
        const auto p = physical_zone(candidate);
        if (candidate == local || !zone_available(p, true)) continue;
        const auto pec = media_->zone_pec_sum(candidate);
        if (pec < cold_pec) { cold_local = candidate; cold_pec = pec; }
    }
    PhysicalCompletion out{.tier = Tier::HBF, .op = Op::Erase,
        .arrival_ns = at_ns, .start_ns = at_ns, .finish_ns = at_ns,
        .note = "host-zone-reset"};
    const auto decision_start = std::max(at_ns, control_ready_ns_[stack]);
    out.breakdown.scheduler_queue_wait_ns = decision_start - at_ns;
    out.breakdown.maintenance_ns = config_.host.host_gc_decision_ns;
    at_ns = decision_start + config_.host.host_gc_decision_ns;
    control_ready_ns_[stack] = at_ns;
    stats_.host_gc_control_ns += config_.host.host_gc_decision_ns;
    stats_.stage_work += out.breakdown;
    if (cold_local != local && hot_pec - cold_pec >=
        static_cast<std::uint64_t>(config_.host.host_zone_wear_gap) * config_.host.zone_size_blocks) {
        const auto remap = remap_zones(stack, channel, zone, cold_local - channel_begin, at_ns);
        out.breakdown += remap.breakdown;
        at_ns = remap.finish_ns;
    }
    (void)reset_zone(stack, channel, zone, at_ns);
    stats_.finish_ns = std::max(stats_.finish_ns, at_ns);
    last_issue_arrival_ns_ = at_ns;
    out.finish_ns = at_ns;
    return out;
}

void HbfController::prepopulate_channel_zones(std::uint32_t stack, std::uint32_t channel,
    std::uint64_t first_zone, std::uint64_t count) {
    if (config_.host.mapping_mode != MappingMode::RawPhysical || last_issue_arrival_ns_ ||
        media_image_seeded_ || !count)
        throw std::runtime_error("channel-zone initial population requires raw mode before IO");
    const auto first = zone_index(stack, channel, first_zone);
    if (count > blocks_.size() / config_.host.zone_size_blocks || first_zone + count < first_zone)
        throw std::runtime_error("zone initial population overflows channel");
    (void)zone_index(stack, channel, first_zone + count - 1);
    // Validate the complete range before claiming anything.
    for (auto z = first; z < first + count; ++z) {
        const auto physical = physical_zone(z);
        for (auto b = physical * config_.host.zone_size_blocks;
             b < (physical + 1) * config_.host.zone_size_blocks; ++b)
            if (blocks_[b].role != BlockRole::Free || blocks_[b].next_page)
                throw std::runtime_error("initial zone overlaps existing ownership");
    }
    for (auto z = first; z < first + count; ++z) {
        const auto physical = physical_zone(z);
        claim_zone(physical);
        for (auto b = physical * config_.host.zone_size_blocks;
             b < (physical + 1) * config_.host.zone_size_blocks; ++b) {
            auto& block = blocks_[b];
            block.initial_raw_pages = block.valid_pages = block.next_page = config_.device.pages_per_block;
            block.free_pages = 0;
            block.set_valid_range(0, config_.device.pages_per_block, config_.device.pages_per_block);
            free_pages_ -= config_.device.pages_per_block;
            free_pages_per_stack_[stack] -= config_.device.pages_per_block;
        }
    }
    stats_.free_pages = free_pages_;
}

HbfController::ZoneState HbfController::zone_state(std::uint32_t stack, std::uint32_t channel,
    std::uint64_t zone, double at_ns) {
    const auto local = zone_index(stack, channel, zone);
    zone_barrier(at_ns);
    const auto physical = physical_zone(local);
    ZoneState state{.physical_zone = physical, .pec_sum = media_->zone_pec_sum(local),
        .valid_pages = 0, .invalid_pages = 0, .free_pages = 0};
    for (auto b = physical * config_.host.zone_size_blocks;
         b < (physical + 1) * config_.host.zone_size_blocks; ++b) {
        state.valid_pages += blocks_[b].valid_pages;
        state.invalid_pages += blocks_[b].invalid_pages;
        state.free_pages += blocks_[b].free_pages;
    }
    return state;
}
} // namespace hbfsim::host
