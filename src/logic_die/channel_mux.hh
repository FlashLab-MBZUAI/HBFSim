#pragma once

#include <cstdint>
#include <vector>

#include "obelisk/types.hh"

namespace obelisk {

// Tracks per-channel bus availability on the logic die. Each channel is
// serialized: only one command per channel at a time.
class ChannelMux {
public:
    explicit ChannelMux(uint32_t num_channels, Cycle cmd_cycles = 2);

    // Acquire channel `ch` starting at `arrive`. Returns the cycle at which
    // the command has been issued and the channel can take the next request.
    Cycle acquire(uint32_t ch, Cycle arrive);

private:
    std::vector<Cycle> channel_busy_until_;
    Cycle cmd_cycles_;
};

}  // namespace obelisk
