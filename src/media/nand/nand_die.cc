#include "src/media/nand/nand_die.hh"

namespace obelisk {

NANDDie::NANDDie(const NANDDieConfig& cfg) : name_("NANDDie." + std::to_string(cfg.die_id)), cfg_(cfg) {
    subarrays_.reserve(cfg.num_subarrays);
    for (uint32_t i = 0; i < cfg.num_subarrays; ++i) {
        subarrays_.push_back(std::make_unique<Subarray>(i, cfg.subarray_cfg));
    }
}

bool NANDDie::accept(std::unique_ptr<Packet>& pkt) {
    uint32_t sa = pkt->phys.subarray_id;
    if (subarrays_.empty()) return false;
    sa = sa % subarrays_.size();
    return subarrays_[sa]->accept(pkt);
}

void NANDDie::tick(Cycle current_cycle) {
    for (auto& sa : subarrays_) {
        sa->set_simulation(sim_);
        sa->tick(current_cycle);
        while (auto maybe = sa->dequeue()) completed_.push(std::move(*maybe));
    }
}

std::optional<std::unique_ptr<Packet>> NANDDie::dequeue() {
    if (completed_.empty()) return std::nullopt;
    auto pkt = std::move(completed_.front());
    completed_.pop();
    return pkt;
}

bool NANDDie::has_pending() const {
    if (!completed_.empty()) return true;
    for (auto& sa : subarrays_) {
        if (sa->has_pending()) return true;
    }
    return false;
}

}  // namespace obelisk
