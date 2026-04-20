#pragma once

#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "obelisk/imedia.hh"
#include "src/media/nand/nand_die.hh"

namespace obelisk {

struct NANDStackConfig {
    uint32_t num_dies_per_stack = 16;
    uint32_t num_subarrays_per_die = 32;
    uint64_t capacity_per_die_bytes = 32ULL * 1024 * 1024 * 1024;
    SubarrayConfig subarray_cfg;
    std::string name = "NANDStack";
};

// Entire HBF NAND stack (16 dies × 32 subarrays typical). Routes by die_id.
class NANDStack : public IMedia {
public:
    explicit NANDStack(const NANDStackConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(Cycle current_cycle) override;
    bool accept(std::unique_ptr<Packet>& pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    MediaType media_type() const override { return MediaType::HBF; }
    uint64_t capacity_bytes() const override {
        return static_cast<uint64_t>(cfg_.num_dies_per_stack) * cfg_.capacity_per_die_bytes;
    }
    uint32_t min_access_granularity() const override { return cfg_.subarray_cfg.page_size_bytes; }
    double peak_bandwidth_gbps() const override;

    NANDDie* get_die(uint32_t die_id);

private:
    std::string name_;
    NANDStackConfig cfg_;
    std::vector<std::unique_ptr<NANDDie>> dies_;
    std::queue<std::unique_ptr<Packet>> completed_;
};

}  // namespace obelisk
