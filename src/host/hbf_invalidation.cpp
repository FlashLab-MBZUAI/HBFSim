#include "host/hbf_controller.hpp"

#include <cmath>

namespace hbfsim::host {

HbfLogicalInvalidationResult HbfController::invalidate_logical_pages(
    std::uint64_t first_lpn, std::uint64_t page_count, double at_ns) {
    if (mapping_policy_) return mapping_policy_->invalidate(*this, first_lpn, page_count, at_ns);
    if (config_.host.mapping_mode == MappingMode::RawPhysical)
        throw std::invalid_argument("HBF logical invalidation requires an FTL mapping mode");
    if (compact_logical_image_ && !compact_logical_image_->mutable_image)
        throw std::invalid_argument("HBF cannot invalidate a compact read-only image");
    const auto capacity = logical_capacity_pages();
    if (page_count == 0 || first_lpn >= capacity || page_count > capacity - first_lpn)
        throw std::invalid_argument("HBF logical invalidation range is outside logical capacity");
    if (!std::isfinite(at_ns) || at_ns < 0 || at_ns < stats_.finish_ns ||
        (last_issue_arrival_ns_ && at_ns < *last_issue_arrival_ns_))
        throw std::invalid_argument("HBF logical invalidation requires a completed IO frontier");

    require_logical_capacity(first_lpn + page_count - 1);
    seed_media_image();
    last_issue_arrival_ns_ = at_ns;
    reservation_causal_watermark_ns_ = at_ns;
    prune_expired_state(at_ns);
    apply_commits_through(at_ns);
    std::fill(state_observation_by_stack_.begin(), state_observation_by_stack_.end(), at_ns);

    HbfLogicalInvalidationResult result;
    result.first_lpn = first_lpn;
    result.page_count = page_count;
    auto& out = result.completion;
    out.tier = Tier::HBF;
    out.op = Op::Erase;
    out.arrival_ns = out.start_ns = at_ns;
    out.resource_path = "host/logical-invalidation";
    out.note = "logical-deallocation; mapping updates charged; invalid media awaits GC";
    const auto physical_before = stats_.physical_read_bytes + stats_.physical_write_bytes;
    std::vector<double> ready_by_stack(config_.device.stacks, at_ns);
    record_mutated_lpn_range(first_lpn, page_count);

    for (auto lpn = first_lpn; lpn < first_lpn + page_count; ++lpn) {
        // Stacks keep their original independent cursors. Reclaim only behind
        // the earliest one: choosing the latest would destroy legal backfill.
        advance_administrative_frontier(
            *std::min_element(ready_by_stack.begin(), ready_by_stack.end()));
        const auto stack = stack_for_lpn(lpn);
        auto& ready = ready_by_stack[stack];
        apply_stack_commits_through(stack, ready);
        state_observation_by_stack_[stack] = ready;

        // These bytes have never been issued to NAND. A free must drop them
        // before a later pressure flush, even if an older mapped version exists.
        auto& buffer = write_buffer_by_stack_[stack];
        if (const auto entry = buffer.find(lpn); entry != buffer.end()) {
            ++result.discarded_buffer_pages;
            result.discarded_buffer_bytes += write_buffer_covered_bytes(entry->second);
            write_buffer_lru_by_stack_[stack].erase(entry->second.iterator);
            buffer.erase(entry);
        }
        if (!lpn_to_ppn_.contains(lpn) && !compact_lpn_ppn(lpn)) {
            ++result.unmapped_pages;
            continue;
        }

        access_mapping(lpn, TransactionSource::User, MappingAccessKind::Update,
            ready, out.breakdown, nullptr);
        // A dirty-cache eviction can have moved this LPN. Retire the mapping
        // current at update completion, never a stale pre-access PPN snapshot.
        apply_stack_commits_through(stack, ready);
        if (const auto mapping = lpn_to_ppn_.find(lpn); mapping != lpn_to_ppn_.end()) {
            read_buffer_purge_page(mapping->second);
            invalidate_ppn(mapping->second);
            lpn_to_ppn_.erase(mapping);
        } else if (const auto ppn = compact_lpn_ppn(lpn)) {
            read_buffer_purge_page(*ppn);
            retire_compact_page(lpn, PageOwner::Logical);
        } else {
            throw std::runtime_error("HBF invalidation lost a live logical mapping");
        }
        ++result.invalidated_pages;
        mark_mapping_page_dirty(mapping_vpn_for_lpn(lpn), ready);
        state_observation_by_stack_[stack] = ready;
        // An already-issued GC copy may publish later. Its expected-old-PPN
        // check rejects the now-deallocated source and retires the stale copy.
    }

    out.finish_ns = std::max(background_finish_ns_,
        *std::max_element(ready_by_stack.begin(), ready_by_stack.end()));
    if (!pending_commits_.empty())
        out.finish_ns = std::max(out.finish_ns, pending_commits_.rbegin()->first.first);
    apply_commits_through(out.finish_ns);
    advance_administrative_frontier(out.finish_ns);
    // This is an explicit administrative barrier, not a persistence drain:
    // unissued buffers, dirty map pages and unfinished paced GC stay live.
    for (std::size_t stack = 0; stack < config_.device.stacks; ++stack) {
        causal_state_ready_by_stack_[stack] = std::max(causal_state_ready_by_stack_[stack], out.finish_ns);
        state_observation_by_stack_[stack] = std::max(state_observation_by_stack_[stack], out.finish_ns);
    }
    last_issue_arrival_ns_ = out.finish_ns;
    out.physical_bytes = stats_.physical_read_bytes + stats_.physical_write_bytes - physical_before;
    stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, at_ns);
    stats_.finish_ns = std::max(stats_.finish_ns, out.finish_ns);
    stats_.free_pages = free_pages_;
    stats_.stage_work += out.breakdown;
    return result;
}

} // namespace hbfsim::host
