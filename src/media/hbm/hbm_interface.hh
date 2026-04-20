#pragma once

#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "obelisk/imedia.hh"

namespace obelisk {

struct HBMConfig {
    // Number of HBM stacks * banks — used to serialize access per bank.
    uint32_t num_banks = 64;
    // Latency parameters for the mock. A real build should plug in DRAMsim3.
    uint32_t tCL_cycles = 14;        // column latency
    uint32_t tRCD_cycles = 14;       // row-to-column
    uint32_t tRP_cycles = 14;        // precharge
    double bandwidth_gbps = 1024.0;  // per-stack peak (HBM3E ~1 TB/s)
    uint32_t freq_mhz = 1000;
    uint64_t capacity_bytes = 96ULL * 1024 * 1024 * 1024;  // 96 GB
    uint32_t access_granularity = 64;
    std::string name = "HBM";

    // DRAMsim3 integration is a TODO. The fields below are placeholders for
    // when we wire the real library in.
    std::string dramsim3_config_file;
    std::string output_dir;
};

// Mock HBM interface. Fixed-latency model per bank, with bandwidth-based
// transfer time. Kept simple so the project builds without pulling in
// DRAMsim3. See src/media/hbm/hbm_interface.cc for the DRAMsim3 TODO.
class HBMInterface : public IMedia {
public:
    explicit HBMInterface(const HBMConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(Cycle current_cycle) override;
    bool accept(std::unique_ptr<Packet>& pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    MediaType media_type() const override { return MediaType::HBM; }
    uint64_t capacity_bytes() const override { return cfg_.capacity_bytes; }
    uint32_t min_access_granularity() const override { return cfg_.access_granularity; }
    double peak_bandwidth_gbps() const override { return cfg_.bandwidth_gbps; }

private:
    struct InFlight {
        std::unique_ptr<Packet> pkt;
        Cycle ready_cycle = 0;
    };

    std::string name_;
    HBMConfig cfg_;
    std::vector<Cycle> bank_busy_until_;
    std::queue<InFlight> in_flight_;
    std::queue<std::unique_ptr<Packet>> completed_;

    Cycle transfer_cycles(uint32_t size) const;
};

}  // namespace obelisk
