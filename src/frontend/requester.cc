#include "src/frontend/requester.hh"

#include <spdlog/spdlog.h>

#include "src/base/simulation.hh"

namespace obelisk {

Requester::Requester(const RequesterConfig& cfg) : cfg_(cfg) {
    trace_ = std::make_unique<TraceReader>(cfg.trace_file);
}

bool Requester::accept(std::unique_ptr<Packet>& pkt) {
    // Fallback completion push path (not used in pull model).
    pkt->timing.t_arrive = sim_ ? sim_->current_cycle() : pkt->timing.t_arrive;
    if (sim_) sim_->log_completed(*pkt);
    if (in_flight_ > 0) --in_flight_;
    ++total_completed_;
    pkt.reset();
    return true;
}

void Requester::tick(Cycle current_cycle) {
    if (!downstream_) return;

    // 1. Pull completions from downstream.
    while (auto maybe = downstream_->dequeue()) {
        auto pkt = std::move(*maybe);
        pkt->timing.t_arrive = current_cycle;
        if (sim_) sim_->log_completed(*pkt);
        if (in_flight_ > 0) --in_flight_;
        ++total_completed_;
        spdlog::trace("Requester: complete id={} total_time={}", pkt->id, pkt->timing.total_time());
    }

    // 2. Issue new requests.
    uint32_t issued_this_cycle = 0;
    while (issued_this_cycle < cfg_.issue_rate_per_cyc && in_flight_ < cfg_.max_in_flight) {
        if (!held_record_.has_value()) {
            if (trace_drained_) break;
            auto rec = trace_->next();
            if (!rec.has_value()) {
                trace_drained_ = true;
                break;
            }
            held_record_ = std::move(rec);
        }

        const TraceRecord& rec = *held_record_;
        // Gate on timestamp if absolute ts is specified (>=0).
        if (rec.timestamp >= 0 && static_cast<Cycle>(rec.timestamp) > current_cycle) break;

        auto pkt = std::make_unique<Packet>();
        pkt->id = next_id_++;
        pkt->type = rec.type;
        pkt->addr = rec.addr;
        pkt->size = rec.size;
        pkt->layer_id = rec.layer_id;
        pkt->op_name = rec.op_name;
        pkt->timing.t_send = current_cycle;

        if (!downstream_->accept(pkt)) {
            // Downstream busy — keep held_record_ for next cycle.
            break;
        }
        held_record_.reset();
        ++in_flight_;
        ++total_issued_;
        ++issued_this_cycle;
        if (sim_) sim_->stats().record_issue();
    }
}

bool Requester::has_pending() const {
    return in_flight_ > 0 || !trace_drained_ || held_record_.has_value();
}

}  // namespace obelisk
