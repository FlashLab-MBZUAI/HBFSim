#pragma once

#include <cstdint>

#include "obelisk/types.hh"

namespace obelisk {

struct TSVConfig {
    uint32_t bandwidth_gbps = 200;
    uint32_t freq_mhz = 1000;
};

// Models TSV (through-silicon via) as a bandwidth pipe. Serializes transfers.
class TSVModel {
public:
    explicit TSVModel(const TSVConfig& cfg) : cfg_(cfg) {}

    // Register a transfer that starts at `arrive_cycle` for `size_bytes`.
    // Returns the cycle at which the transfer completes.
    Cycle schedule(Cycle arrive_cycle, uint32_t size_bytes);

    Cycle bus_busy_until() const { return bus_busy_until_; }

private:
    TSVConfig cfg_;
    Cycle bus_busy_until_ = 0;

    Cycle transfer_cycles(uint32_t size_bytes) const;
};

}  // namespace obelisk
