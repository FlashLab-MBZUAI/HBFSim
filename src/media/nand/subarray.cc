#include "src/media/nand/subarray.hh"

#include <algorithm>

#include "src/base/simulation.hh"

namespace obelisk {

Subarray::Subarray(uint32_t id, const SubarrayConfig& cfg)
    : name_("Subarray." + std::to_string(id)), id_(id), cfg_(cfg) {}

bool Subarray::buffer_hit(uint64_t page_id) {
    return std::find(cached_pages_.begin(), cached_pages_.end(), page_id) != cached_pages_.end();
}

void Subarray::touch_buffer(uint64_t page_id) {
    auto it = std::find(cached_pages_.begin(), cached_pages_.end(), page_id);
    if (it != cached_pages_.end()) cached_pages_.erase(it);
    cached_pages_.push_back(page_id);
    while (cached_pages_.size() > cfg_.cache_buffer_pages) cached_pages_.erase(cached_pages_.begin());
}

Cycle Subarray::latency_for(const Packet& pkt) const {
    switch (pkt.type) {
        case PacketType::READ:
        case PacketType::READ_KV:
        case PacketType::READ_NON_TEMPORAL:
        case PacketType::PREFETCH:
        case PacketType::GC_READ:
            return cfg_.tR_cycles;
        case PacketType::WRITE:
        case PacketType::GC_WRITE:
            return cfg_.tPROG_cycles;
    }
    return cfg_.tR_cycles;
}

bool Subarray::accept(std::unique_ptr<Packet>& pkt) {
    if (queue_.size() >= cfg_.queue_depth) return false;
    Cycle now = sim_ ? sim_->current_cycle() : 0;
    queue_.push_back({std::move(pkt), now});
    return true;
}

void Subarray::tick(Cycle current_cycle) {
    // Complete current if its deadline has passed.
    if (busy_ && current_cycle >= busy_until_) {
        current_->timing.subarray_time += (busy_until_ - current_start_cycle_);
        completed_.push(std::move(current_));
        busy_ = false;
        current_start_cycle_ = 0;
        busy_until_ = 0;
    }

    // Start next if idle.
    if (!busy_ && !queue_.empty()) {
        auto& head = queue_.front();
        // Record queuing (time from enqueue to service start).
        head.pkt->timing.subarray_queuing += (current_cycle - head.enqueue_cycle);

        Cycle lat = latency_for(*head.pkt);
        if (head.pkt->is_read() && buffer_hit(head.pkt->phys.page_id)) {
            lat = cfg_.cache_hit_read_cycles;
            ++buf_hits_;
        } else if (head.pkt->is_read()) {
            ++buf_misses_;
            touch_buffer(head.pkt->phys.page_id);
        } else {
            touch_buffer(head.pkt->phys.page_id);
        }

        current_ = std::move(head.pkt);
        queue_.pop_front();
        busy_ = true;
        current_start_cycle_ = current_cycle;
        busy_until_ = current_cycle + lat;
    }
}

std::optional<std::unique_ptr<Packet>> Subarray::dequeue() {
    if (completed_.empty()) return std::nullopt;
    auto pkt = std::move(completed_.front());
    completed_.pop();
    return pkt;
}

bool Subarray::has_pending() const { return busy_ || !queue_.empty() || !completed_.empty(); }

}  // namespace obelisk
