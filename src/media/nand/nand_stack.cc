#include "src/media/nand/nand_stack.hh"

namespace obelisk {

NANDStack::NANDStack(const NANDStackConfig& cfg) : name_(cfg.name), cfg_(cfg) {
    dies_.reserve(cfg.num_dies_per_stack);
    for (uint32_t i = 0; i < cfg.num_dies_per_stack; ++i) {
        NANDDieConfig dcfg;
        dcfg.die_id = i;
        dcfg.num_subarrays = cfg.num_subarrays_per_die;
        dcfg.subarray_cfg = cfg.subarray_cfg;
        dies_.push_back(std::make_unique<NANDDie>(dcfg));
    }
}

NANDDie* NANDStack::get_die(uint32_t die_id) {
    if (die_id >= dies_.size()) return nullptr;
    return dies_[die_id].get();
}

bool NANDStack::accept(std::unique_ptr<Packet>& pkt) {
    uint32_t die = pkt->phys.die_id;
    if (dies_.empty()) return false;
    die = die % dies_.size();
    return dies_[die]->accept(pkt);
}

void NANDStack::tick(Cycle current_cycle) {
    // Just tick the dies — do NOT drain their completed queues. The LogicDie
    // above us pulls from each NANDDie::dequeue() directly. If we drained
    // here we would steal packets from LogicDie.
    for (auto& d : dies_) {
        d->set_simulation(sim_);
        d->tick(current_cycle);
    }
}

std::optional<std::unique_ptr<Packet>> NANDStack::dequeue() {
    // Intentionally returns nothing in the current wiring: LogicDie pulls
    // from NANDDie directly.
    return std::nullopt;
}

bool NANDStack::has_pending() const {
    for (auto& d : dies_) {
        if (d->has_pending()) return true;
    }
    return false;
}

double NANDStack::peak_bandwidth_gbps() const {
    // Rough: pages/sec per subarray * num_subarrays_per_die * num_dies
    // pages/sec per subarray = freq_hz / tR_cycles
    // We don't have freq here; assume 1 GHz.
    double page_bytes = cfg_.subarray_cfg.page_size_bytes;
    double page_per_sec = 1e9 / static_cast<double>(cfg_.subarray_cfg.tR_cycles);
    double bw_per_subarray = page_bytes * page_per_sec;  // bytes/sec
    uint64_t total_subarrays =
        static_cast<uint64_t>(cfg_.num_dies_per_stack) * cfg_.num_subarrays_per_die;
    return bw_per_subarray * static_cast<double>(total_subarrays) / 1e9;  // GB/s
}

}  // namespace obelisk
