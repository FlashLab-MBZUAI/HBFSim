#include "src/logic_die/channel_mux.hh"

#include <algorithm>

namespace obelisk {

ChannelMux::ChannelMux(uint32_t num_channels, Cycle cmd_cycles)
    : channel_busy_until_(num_channels, 0), cmd_cycles_(cmd_cycles) {}

Cycle ChannelMux::acquire(uint32_t ch, Cycle arrive) {
    if (ch >= channel_busy_until_.size()) ch = ch % channel_busy_until_.size();
    Cycle start = std::max(arrive, channel_busy_until_[ch]);
    Cycle end = start + cmd_cycles_;
    channel_busy_until_[ch] = end;
    return end;
}

}  // namespace obelisk
