#pragma once

#include <cstdint>
#include <limits>

namespace obelisk {

using Cycle = uint64_t;
using Addr = uint64_t;
using PacketId = uint64_t;

inline constexpr Cycle kInvalidCycle = std::numeric_limits<Cycle>::max();
inline constexpr PacketId kInvalidPacketId = std::numeric_limits<PacketId>::max();

}  // namespace obelisk
