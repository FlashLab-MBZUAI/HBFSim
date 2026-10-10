#include "physical/simulation_session.hpp"
#include "physical/resource_calendar.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

// Count requested heap bytes, not RSS: allocator caches and OS paging should
// not hide retained simulator state. This executable is single-threaded.
namespace census {
// Default operator new alignment can exceed max_align_t (Apple arm64: 16 vs 8).
// Advancing past the header must preserve the allocator's promised alignment.
struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__) Header { std::size_t bytes; };
std::size_t live = 0, peak = 0;
}
void* operator new(std::size_t bytes) {
    auto* p = static_cast<unsigned char*>(std::malloc(sizeof(census::Header) + bytes));
    if (!p) throw std::bad_alloc();
    std::memcpy(p, &bytes, sizeof(bytes));
    census::live += bytes;
    census::peak = std::max(census::peak, census::live);
    return p + sizeof(census::Header);
}
void operator delete(void* p) noexcept {
    if (!p) return;
    auto* h = static_cast<unsigned char*>(p) - sizeof(census::Header);
    std::size_t bytes;
    std::memcpy(&bytes, h, sizeof(bytes));
    census::live -= bytes;
    std::free(h);
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

using namespace hbfsim::physical;
void require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}

void allocation_alignment_is_valid() {
    const auto base = census::live;
    void* allocation = ::operator new(1);
    const bool aligned = reinterpret_cast<std::uintptr_t>(allocation)
        % __STDCPP_DEFAULT_NEW_ALIGNMENT__ == 0;
    const bool counted = census::live == base + 1;
    ::operator delete(allocation);
    require(aligned, "heap census violates default operator new alignment");
    require(counted && census::live == base, "heap census lost requested-byte accounting");
}

template<class Calendar> void calendar_reclaims() {
    const auto base = census::live;
    Calendar calendar;
    for (unsigned i = 0; i < 100000; ++i) (void)calendar.reserve(i * 4, 1);
    const auto work = calendar.reserved_work_ns;
    calendar.prune_before(399960); // Keep ten live gaps while shrinking.
    require(census::live - base < 256 * 1024, "sparse calendar retains its high-water arena");
    require(calendar.reserve(399962, 1).start_ns == 399962, "reclamation changed legal backfill");
    calendar.prune_before(400000);
    require(calendar.gaps_after(400000).empty(), "expired gaps survived reclamation");
    require(calendar.reserved_work_ns == work + 1, "reclamation lost work accounting");
    require(calendar.reserve(400004, 2).start_ns == 400004, "calendar cannot resume after reclamation");
}

void cache_history_is_bounded() {
    hbf::HbfDeviceConfig config;
    config.channels_per_stack = config.dies_per_channel = config.planes_per_die = 1;
    config.blocks_per_plane = 4096;
    hbf::HbfDevice device(config);
    const auto base = census::live;
    for (unsigned i = 0; i < 1000000; ++i) {
        device.advance_cache(i);
        device.cache_fill(i, i + 2);
    }
    require(census::live - base < 64 * 1024, "two-page cache retains cumulative candidates");
    require(device.cache_hit(999998, 1000000), "candidate reclamation lost a pending fill");
    device.advance_cache(1000002);
    require(device.cache_hit(999999, 1000002), "candidate reclamation lost a resident page");
}

void invalidation_is_bounded(std::uint32_t stacks) {
    constexpr std::uint64_t pages = 1000000;
    SimulationSessionConfig config;
    config.hbf.device.stacks = stacks;
    config.hbf.device.channels_per_stack = config.hbf.device.dies_per_channel = 1;
    config.hbf.device.planes_per_die = 4;
    config.hbf.device.blocks_per_plane = (pages + 1023) / 1024 + 32;
    config.hbf.host.mapping_mode = hbfsim::host::MappingMode::Cached;
    config.hbf.host.ctrl_dram_bytes = 64ull << 20;
    config.initial_hbf_logical_pages = pages;
    config.hbm.device.stacks = 1;
    config.hbm.device.channels_per_stack = 2;
    config.trace.mode = TraceMode::Off;
    config.trace.retain_completion_diagnostics = false;
    SimulationSession session(config);
    const auto base = census::live;
    census::peak = base;
    const auto result = session.invalidate_hbf_pages("release", 0, pages);
    require(result.invalidation.invalidated_pages == pages, "large invalidation lost pages");
    require(census::peak - base < 32 * pages, "invalidation retains per-access temporal history");
    const auto invalidation_peak = census::peak - base;
    const auto checkpoint = session.checkpoint_pending("persist");
    require(checkpoint.persistence.quiescence.quiescent(), "checkpoint did not drain");
    const auto before_image = census::live;
    census::peak = before_image;
    const auto image = session.persistent_hbf_image();
    require(image.compact_image->retired_lpns.size() == pages, "checkpoint lost retirement bits");
    require(census::peak - before_image < 4 * pages, "checkpoint expands retired pages into integers");
    std::printf("stacks=%u invalidation_peak_extra=%zu checkpoint_peak_extra=%zu finish_ns=%.17g\n",
        stacks, invalidation_peak, census::peak - before_image, result.invalidation.completion.finish_ns);
}

int main() {
    try {
        allocation_alignment_is_valid();
        calendar_reclaims<ResourceTimeline>();
        cache_history_is_bounded();
        invalidation_is_bounded(1);
        invalidation_is_bounded(4);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
