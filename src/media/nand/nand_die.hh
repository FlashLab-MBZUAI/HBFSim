#pragma once

#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "obelisk/imodule.hh"
#include "src/media/nand/subarray.hh"

namespace obelisk {

struct NANDDieConfig {
    uint32_t die_id = 0;
    uint32_t num_subarrays = 32;
    SubarrayConfig subarray_cfg;
};

// Groups `num_subarrays` Subarrays. Routes packets by phys.subarray_id.
class NANDDie : public IModule {
public:
    explicit NANDDie(const NANDDieConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(Cycle current_cycle) override;
    bool accept(std::unique_ptr<Packet>& pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    uint32_t die_id() const { return cfg_.die_id; }
    size_t num_subarrays() const { return subarrays_.size(); }

    const Subarray& subarray(uint32_t idx) const { return *subarrays_[idx]; }

private:
    std::string name_;
    NANDDieConfig cfg_;
    std::vector<std::unique_ptr<Subarray>> subarrays_;
    std::queue<std::unique_ptr<Packet>> completed_;
};

}  // namespace obelisk
