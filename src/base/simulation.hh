#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "obelisk/imodule.hh"
#include "src/base/stats.hh"

namespace obelisk {

struct SimulationConfig {
    uint64_t max_cycles = 100'000'000;
    uint64_t warmup_cycles = 0;
    uint32_t freq_mhz = 1000;
    std::string output_csv_path;
    std::string output_stats_json_path;
    uint32_t log_level = 1;  // 0=off, 1=info, 2=debug, 3=trace
};

class Simulation {
public:
    explicit Simulation(const SimulationConfig& cfg);
    ~Simulation();

    // Register in tick order. Earlier-registered modules tick first.
    void register_module(std::shared_ptr<IModule> m);

    void run();

    Cycle current_cycle() const { return cycle_; }

    Stats& stats() { return stats_; }

    // Record a completed packet to CSV + stats.
    void log_completed(const Packet& pkt);

    const SimulationConfig& config() const { return cfg_; }

private:
    SimulationConfig cfg_;
    Cycle cycle_ = 0;
    std::vector<std::shared_ptr<IModule>> modules_;
    Stats stats_;
    std::ofstream csv_out_;
    bool csv_header_written_ = false;

    void write_csv_header();
    bool any_pending() const;
};

}  // namespace obelisk
