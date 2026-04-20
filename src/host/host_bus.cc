#include "src/host/host_bus.hh"

#include <spdlog/spdlog.h>

#include "src/base/simulation.hh"

namespace obelisk {

HostBus::HostBus(const HostBusConfig& cfg) : cfg_(cfg) {}

Cycle HostBus::compute_transfer_cycles(uint32_t size) const {
    // cycles = size * freq_mhz / (bandwidth_gbps * 1000)
    if (cfg_.bandwidth_gbps <= 0.0) return 0;
    double cycles = static_cast<double>(size) * static_cast<double>(cfg_.freq_mhz) / (cfg_.bandwidth_gbps * 1000.0);
    return cycles < 1.0 ? 1 : static_cast<Cycle>(cycles);
}

Cycle HostBus::forward_service_cycles(uint32_t size) const {
    return cfg_.cpu_compute_cycles + cfg_.base_latency_cycles + compute_transfer_cycles(size);
}

Cycle HostBus::return_service_cycles(uint32_t size) const {
    return cfg_.base_latency_cycles + compute_transfer_cycles(size);
}

bool HostBus::accept(std::unique_ptr<Packet>& pkt) {
    if (forward_queue_.size() >= cfg_.queue_depth) return false;
    Cycle now = sim_ ? sim_->current_cycle() : 0;
    Cycle start = std::max(now, forward_bus_busy_until_);
    Cycle xfer = compute_transfer_cycles(pkt->size);
    Cycle ready = start + forward_service_cycles(pkt->size);
    pkt->timing.host_bus_queuing += (start > now) ? (start - now) : 0;
    pkt->timing.host_bus_time += forward_service_cycles(pkt->size);
    forward_bus_busy_until_ = start + xfer;  // bus serializes on transfer window
    forward_queue_.push({std::move(pkt), ready});
    return true;
}

void HostBus::tick(Cycle current_cycle) {
    // Drain forward queue if ready and downstream accepts.
    while (!forward_queue_.empty() && forward_queue_.front().ready_cycle <= current_cycle) {
        if (!downstream_) break;
        auto& head = forward_queue_.front();
        if (downstream_->accept(head.pkt)) {
            forward_queue_.pop();
        } else {
            break;  // downstream busy; retry next tick
        }
    }

    // Pull completions from downstream → process return → expose via dequeue().
    if (downstream_) {
        while (auto maybe = downstream_->dequeue()) {
            auto pkt = std::move(*maybe);
            Cycle now = current_cycle;
            Cycle start = std::max(now, return_bus_busy_until_);
            Cycle xfer = compute_transfer_cycles(pkt->size);
            Cycle ready = start + return_service_cycles(pkt->size);
            pkt->timing.host_bus_queuing += (start > now) ? (start - now) : 0;
            pkt->timing.host_bus_time += return_service_cycles(pkt->size);
            return_bus_busy_until_ = start + xfer;
            return_queue_.push({std::move(pkt), ready});
        }
    }
}

std::optional<std::unique_ptr<Packet>> HostBus::dequeue() {
    if (return_queue_.empty()) return std::nullopt;
    Cycle now = sim_ ? sim_->current_cycle() : 0;
    if (return_queue_.front().ready_cycle > now) return std::nullopt;
    auto pkt = std::move(return_queue_.front().pkt);
    return_queue_.pop();
    return pkt;
}

bool HostBus::has_pending() const { return !forward_queue_.empty() || !return_queue_.empty(); }

}  // namespace obelisk
