// HBM model.
//
// This is a mock: fixed-latency per-bank plus a bandwidth-based transfer
// window, sufficient for M1 demo. A real study should swap in DRAMsim3 (see
// third_party/DRAMsim3 and design spec Section 5.8). The callback-style
// completion of DRAMsim3 maps to the `completed_` queue below.
//
// TODO(M2): integrate DRAMsim3. Replace the latency model with
// MemorySystem.AddTransaction + ClockTick + CompletionCallback.

#include "src/media/hbm/hbm_interface.hh"

#include <algorithm>

#include "src/base/simulation.hh"

namespace obelisk {

HBMInterface::HBMInterface(const HBMConfig& cfg) : name_(cfg.name), cfg_(cfg) {
    bank_busy_until_.assign(cfg.num_banks, 0);
}

Cycle HBMInterface::transfer_cycles(uint32_t size) const {
    if (cfg_.bandwidth_gbps <= 0.0) return 0;
    double c = static_cast<double>(size) * static_cast<double>(cfg_.freq_mhz) / (cfg_.bandwidth_gbps * 1000.0);
    return c < 1.0 ? 1 : static_cast<Cycle>(c);
}

bool HBMInterface::accept(std::unique_ptr<Packet>& pkt) {
    if (bank_busy_until_.empty()) return false;
    Cycle now = sim_ ? sim_->current_cycle() : 0;
    uint32_t bank = pkt->phys.bank_id % bank_busy_until_.size();
    Cycle bank_start = std::max(now, bank_busy_until_[bank]);
    Cycle access_latency = cfg_.tRCD_cycles + cfg_.tCL_cycles + cfg_.tRP_cycles;
    Cycle xfer = transfer_cycles(pkt->size);
    Cycle ready = bank_start + access_latency + xfer;
    bank_busy_until_[bank] = bank_start + access_latency;  // bank is free after command window; transfer on bus

    pkt->timing.bank_queuing += (bank_start > now) ? (bank_start - now) : 0;
    pkt->timing.bank_time += access_latency + xfer;
    in_flight_.push({std::move(pkt), ready});
    return true;
}

void HBMInterface::tick(Cycle current_cycle) {
    while (!in_flight_.empty() && in_flight_.front().ready_cycle <= current_cycle) {
        completed_.push(std::move(in_flight_.front().pkt));
        in_flight_.pop();
    }
}

std::optional<std::unique_ptr<Packet>> HBMInterface::dequeue() {
    if (completed_.empty()) return std::nullopt;
    auto pkt = std::move(completed_.front());
    completed_.pop();
    return pkt;
}

bool HBMInterface::has_pending() const { return !in_flight_.empty() || !completed_.empty(); }

}  // namespace obelisk
