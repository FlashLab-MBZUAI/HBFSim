#include "src/logic_die/tsv_model.hh"

#include <algorithm>

namespace obelisk {

Cycle TSVModel::transfer_cycles(uint32_t size_bytes) const {
    if (cfg_.bandwidth_gbps == 0) return 0;
    // cycles = size * freq_mhz / (bw_gbps * 1000)
    double c = static_cast<double>(size_bytes) * static_cast<double>(cfg_.freq_mhz) /
               (static_cast<double>(cfg_.bandwidth_gbps) * 1000.0);
    return c < 1.0 ? 1 : static_cast<Cycle>(c);
}

Cycle TSVModel::schedule(Cycle arrive_cycle, uint32_t size_bytes) {
    Cycle start = std::max(arrive_cycle, bus_busy_until_);
    Cycle end = start + transfer_cycles(size_bytes);
    bus_busy_until_ = end;
    return end;
}

}  // namespace obelisk
