#pragma once

#include <cstdint>

namespace obelisk {

// Tracks how many in-flight sub-array commands exist. Caps at
// max_concurrent so logic die doesn't admit more than the hardware can
// track.
class SubarrayDispatcher {
public:
    explicit SubarrayDispatcher(uint32_t max_concurrent) : max_concurrent_(max_concurrent) {}

    bool can_admit() const { return in_flight_ < max_concurrent_; }
    void admit() { ++in_flight_; }
    void retire() {
        if (in_flight_ > 0) --in_flight_;
    }
    uint32_t in_flight() const { return in_flight_; }

private:
    uint32_t max_concurrent_;
    uint32_t in_flight_ = 0;
};

}  // namespace obelisk
