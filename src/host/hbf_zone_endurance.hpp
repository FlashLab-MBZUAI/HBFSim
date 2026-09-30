#pragma once
#include "physical/hbf/hbf_device.hpp"
#include <set>
#include <vector>

namespace hbfsim::host {
// Exact wear-only execution of a restricted lifecycle: invalidate a complete
// zone, optionally swap with a colder invalid zone, reset, rewrite every page.
// No partial invalidation, live-page relocation, timing or metadata write is
// inferred. Unsupported lifecycles must use the timed controller instead.
// It shares the real base-die map and PEC implementation with HbfController.
class HbfZoneEndurance {
public:
    HbfZoneEndurance(physical::hbf::HbfDeviceConfig config, std::uint32_t zone_blocks,
        std::uint32_t wear_gap);
    void prepopulate(std::uint64_t first_zone, std::uint64_t count);
    void rewrite(std::uint64_t zone, bool remap);
    [[nodiscard]] std::vector<std::uint32_t> wear() const;
    [[nodiscard]] const auto& mapping() const { return media_.zone_mapping(); }
    std::uint64_t programs = 0, erases = 0, remaps = 0, invalidations = 0, resets = 0;
private:
    physical::hbf::HbfDevice media_;
    std::uint32_t zone_blocks_, pages_per_block_, channels_per_stack_, wear_gap_;
    std::uint64_t zones_per_channel_, blocks_;
    std::vector<bool> valid_;
    std::vector<std::set<std::pair<std::uint64_t,std::uint64_t>>> invalid_;
};
}
