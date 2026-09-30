#include "host/hbf_zone_endurance.hpp"

namespace hbfsim::host {
HbfZoneEndurance::HbfZoneEndurance(physical::hbf::HbfDeviceConfig config,
    std::uint32_t zone_blocks, std::uint32_t wear_gap)
    : media_(config), zone_blocks_(zone_blocks), pages_per_block_(config.pages_per_block),
      channels_per_stack_(config.channels_per_stack), wear_gap_(wear_gap) {
    media_.configure_host_zones(zone_blocks);
    blocks_ = std::uint64_t{config.stacks} * config.channels_per_stack * config.dies_per_channel *
        config.planes_per_die * config.blocks_per_plane;
    const auto channels = std::uint64_t{config.stacks} * config.channels_per_stack;
    zones_per_channel_ = blocks_ / zone_blocks / channels;
    valid_.resize(blocks_ / zone_blocks, false);
    invalid_.resize(channels);
    for (std::uint64_t zone = 0; zone < valid_.size(); ++zone)
        invalid_[zone / zones_per_channel_].emplace(0, zone);
}
void HbfZoneEndurance::prepopulate(std::uint64_t first, std::uint64_t count) {
    if (programs || !count || first >= valid_.size() || count > valid_.size() - first)
        throw std::runtime_error("invalid initial zone population");
    for (auto z = first; z < first + count; ++z)
        if (valid_[z]) throw std::runtime_error("initial zone population overlaps");
    for (auto z = first; z < first + count; ++z) {
        valid_[z] = true;
        invalid_[z / zones_per_channel_].erase({0, z});
    }
}
void HbfZoneEndurance::rewrite(std::uint64_t zone, bool remap) {
    if (zone >= valid_.size()) throw std::runtime_error("zone rewrite outside device");
    const auto channel = zone / zones_per_channel_;
    auto& pool = invalid_[channel];
    // The workload explicitly expires this complete object before rewriting.
    valid_[zone] = false;
    ++invalidations;
    auto hot_pec = media_.zone_pec_sum(zone);
    pool.emplace(hot_pec, zone);
    const auto [cold_pec, cold_zone] = *pool.begin();
    if (remap && cold_pec < hot_pec &&
        hot_pec - cold_pec >= std::uint64_t{wear_gap_} * zone_blocks_) {
        pool.erase({hot_pec, zone});
        pool.erase({cold_pec, cold_zone});
        media_.execute_zone_remap({.stack=static_cast<std::uint32_t>(channel / channels_per_stack_),
            .channel=static_cast<std::uint32_t>(channel % channels_per_stack_),
            .first=zone % zones_per_channel_, .second=cold_zone % zones_per_channel_});
        pool.emplace(hot_pec, cold_zone);
        pool.emplace(cold_pec, zone);
        hot_pec = cold_pec;
        ++remaps;
    }
    ++resets; // No physical erase at reset or remap.
    pool.erase({hot_pec, zone});
    const auto physical = media_.physical_zone(zone);
    // Whole-zone rewrite: each block gets page 0 exactly once and therefore
    // exactly one autoerase, followed by pages_per_block ordered programs.
    for (auto b = physical * zone_blocks_; b < (physical + 1) * zone_blocks_; ++b)
        (void)media_.record_erase(b);
    erases += zone_blocks_;
    programs += std::uint64_t{zone_blocks_} * pages_per_block_;
    valid_[zone] = true;
}
std::vector<std::uint32_t> HbfZoneEndurance::wear() const {
    std::vector<std::uint32_t> values(blocks_);
    for (std::uint64_t b=0; b<blocks_; ++b) values[b] = media_.block_pec(b);
    return values;
}
}
