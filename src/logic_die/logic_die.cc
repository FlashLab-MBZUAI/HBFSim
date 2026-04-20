#include "src/logic_die/logic_die.hh"

#include <spdlog/spdlog.h>

#include "src/base/simulation.hh"

namespace obelisk {

LogicDie::LogicDie(const LogicDieConfig& cfg) : name_(cfg.name), cfg_(cfg) {
    channel_mux_ = std::make_unique<ChannelMux>(cfg.num_channels, cfg.command_translation_cycles);
    dispatcher_ = std::make_unique<SubarrayDispatcher>(cfg.max_concurrent_subarray_cmds);
    tsv_ = std::make_unique<TSVModel>(TSVConfig{cfg.tsv_bandwidth_gbps, cfg.logic_die_freq_mhz});
    nand_dies_.assign(cfg.num_dies_per_stack, nullptr);
}

void LogicDie::connect_nand_die(uint32_t die_id, IModule* die) {
    if (die_id >= nand_dies_.size()) nand_dies_.resize(die_id + 1, nullptr);
    nand_dies_[die_id] = die;
}

bool LogicDie::accept(std::unique_ptr<Packet>& pkt) {
    if (!dispatcher_->can_admit()) return false;
    Cycle now = sim_ ? sim_->current_cycle() : 0;
    uint32_t ch = pkt->phys.channel_id % cfg_.num_channels;
    Cycle grant = channel_mux_->acquire(ch, now);
    pkt->timing.logic_die_queuing += (grant > now) ? (grant - now - cfg_.command_translation_cycles) : 0;
    pkt->timing.logic_die_time += cfg_.command_translation_cycles;
    dispatcher_->admit();
    forward_queue_.push({std::move(pkt), grant, now});
    return true;
}

void LogicDie::tick(Cycle current_cycle) {
    // 1. Forward ready packets to their NAND die.
    while (!forward_queue_.empty() && forward_queue_.front().ready_cycle <= current_cycle) {
        auto& head = forward_queue_.front();
        uint32_t die_id = head.pkt->phys.die_id;
        if (die_id >= nand_dies_.size() || nand_dies_[die_id] == nullptr) {
            spdlog::warn("LogicDie: no NAND die at die_id={}, dropping", die_id);
            dispatcher_->retire();
            forward_queue_.pop();
            continue;
        }
        if (nand_dies_[die_id]->accept(head.pkt)) {
            forward_queue_.pop();
        } else {
            break;  // downstream full
        }
    }

    // 2. Pull completions from all NAND dies; add TSV latency on return.
    for (auto* die : nand_dies_) {
        if (!die) continue;
        while (auto maybe = die->dequeue()) {
            auto pkt = std::move(*maybe);
            dispatcher_->retire();
            Cycle tsv_end = tsv_->schedule(current_cycle, pkt->size);
            pkt->timing.tsv_time += (tsv_end - current_cycle);
            return_queue_.push({std::move(pkt), tsv_end});
        }
    }
}

std::optional<std::unique_ptr<Packet>> LogicDie::dequeue() {
    if (return_queue_.empty()) return std::nullopt;
    Cycle now = sim_ ? sim_->current_cycle() : 0;
    if (return_queue_.front().ready_cycle > now) return std::nullopt;
    auto pkt = std::move(return_queue_.front().pkt);
    return_queue_.pop();
    return pkt;
}

bool LogicDie::has_pending() const { return !forward_queue_.empty() || !return_queue_.empty(); }

}  // namespace obelisk
