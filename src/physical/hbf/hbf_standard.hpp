#pragma once
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace hbfsim::physical::hbf {
inline constexpr std::string_view kStandard = "OCP-HBF-0.7.0-2026-08-03";
struct SpeedGrade {
    unsigned number, maximum_channels, maximum_dies;
    double lane_rate_GTs, payload_GBps_per_channel;
};
// p.16 Tables 2/4, with the 75% AXI efficiency applied exactly once.
// pp.32/35 instead say 94/188 GB/s: retain this discrepancy in provenance.
inline SpeedGrade speed_grade(unsigned number) {
    switch (number) {
    case 1: return {1, 8, 8, 8, 48};
    case 2: return {2, 16, 16, 16, 96};
    case 3: return {3, 16, 16, 32, 192};
    default: throw std::invalid_argument("HBF speed grade must be 1, 2 or 3");
    }
}
inline constexpr std::uint64_t kPageBytes = 4096;
inline constexpr std::uint64_t kBurstAlignment = 64;
inline constexpr std::uint32_t kCachedPagesPerBank = 2;
}
