#include "src/controller/hbf_controller.hh"

#include <spdlog/spdlog.h>

#include "src/base/simulation.hh"

namespace obelisk {

HBFController::HBFController(const HBFControllerConfig& cfg) : name_(cfg.name), cfg_(cfg) {
    scheduler_ = make_scheduler(cfg.scheduler_type);
}

bool HBFController::accept(std::unique_ptr<Packet>& pkt) {
    Cycle now = sim_ ? sim_->current_cycle() : 0;
    if (pkt->is_write()) {
        if (write_queue_.size() >= cfg_.write_queue_depth) return false;
        entry_cycles_[pkt->id] = now;
        write_queue_.push_back(std::move(pkt));
    } else {
        if (read_queue_.size() >= cfg_.read_queue_depth) return false;
        entry_cycles_[pkt->id] = now;
        read_queue_.push_back(std::move(pkt));
    }
    return true;
}

void HBFController::tick(Cycle current_cycle) {
    // 1. Pull completions from downstream, add controller return latency.
    if (downstream_) {
        while (auto maybe = downstream_->dequeue()) {
            auto pkt = std::move(*maybe);
            pkt->timing.controller_time += cfg_.controller_latency_cycles;
            return_queue_.push({std::move(pkt), current_cycle + cfg_.controller_latency_cycles});
        }
    }

    // 2. Dispatch from queues via scheduler.
    if (downstream_) {
        SchedulerContext ctx{read_queue_, write_queue_, current_cycle, cfg_.enable_read_priority, nullptr};
        while (true) {
            auto pkt = scheduler_->select_next(ctx);
            if (!pkt) break;

            auto it = entry_cycles_.find(pkt->id);
            Cycle entry = (it != entry_cycles_.end()) ? it->second : current_cycle;
            if (it != entry_cycles_.end()) entry_cycles_.erase(it);

            pkt->timing.controller_queuing += (current_cycle - entry);
            pkt->timing.controller_time += cfg_.controller_latency_cycles;

            if (!downstream_->accept(pkt)) {
                // Downstream rejected; push back at queue front (preserve ordering).
                entry_cycles_[pkt->id] = entry;
                if (pkt->is_write())
                    write_queue_.push_front(std::move(pkt));
                else
                    read_queue_.push_front(std::move(pkt));
                break;
            }
        }
    }
}

std::optional<std::unique_ptr<Packet>> HBFController::dequeue() {
    if (return_queue_.empty()) return std::nullopt;
    Cycle now = sim_ ? sim_->current_cycle() : 0;
    if (return_queue_.front().ready_cycle > now) return std::nullopt;
    auto pkt = std::move(return_queue_.front().pkt);
    return_queue_.pop();
    return pkt;
}

bool HBFController::has_pending() const {
    return !read_queue_.empty() || !write_queue_.empty() || !return_queue_.empty();
}

}  // namespace obelisk
