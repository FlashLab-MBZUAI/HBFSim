#include "host/hbf_controller.hpp"

#include <iomanip>

namespace hbfsim::host {
using namespace hbfsim::physical;
void HbfController::write_wear_snapshot_json(std::ostream& out) const {
    const auto channels = static_cast<std::uint64_t>(config_.device.stacks) * config_.device.channels_per_stack;
    const auto blocks_per_channel = blocks_.size() / channels;
    // At most 128 cells per channel; cells never cross a channel boundary.
    const auto zones_per_channel = blocks_per_channel / config_.host.zone_size_blocks;
    const auto zones_per_bin = std::max<std::uint64_t>(1, (zones_per_channel + 127) / 128);
    const auto blocks_per_bin = zones_per_bin * config_.host.zone_size_blocks;
    out << std::setprecision(17)
        << "{\"schema\":\"hbfsim.hbf_wear.v1\",\"management\":\"host\","
        << "\"specification\":\"OCP HBF 0.7.0\","
        << "\"frontend\":\"" << (zone_managed_ ? "channel-local zones" : "host page FTL") << "\","
        << "\"zone_mapping_owner\":\"HBF base die\",\"wear_leveling_selection\":\"host-controlled (10b)\","
        << "\"coordinate_order\":\"stack/channel/die/plane/block\","
        << "\"pec_scope\":\"physical lifetime including restored history\","
        << "\"workload_scope\":\"erase commands issued in this run including terminal drain\","
        << "\"stacks\":" << config_.device.stacks
        << ",\"channels_per_stack\":" << config_.device.channels_per_stack
        << ",\"dies_per_channel\":" << config_.device.dies_per_channel
        << ",\"planes_per_die\":" << config_.device.planes_per_die
        << ",\"blocks_per_plane\":" << config_.device.blocks_per_plane
        << ",\"pages_per_block\":" << config_.device.pages_per_block
        << ",\"page_size_bytes\":" << config_.device.page_size_bytes
        << ",\"zone_size_blocks\":" << config_.host.zone_size_blocks
        << ",\"zones_per_channel\":" << zones_per_channel
        << ",\"blocks\":" << blocks_.size()
        << ",\"block_population\":\"all_physical_blocks_including_static_read_only\""
        << ",\"block_index\":\"zero_based_physical_block_index_in_coordinate_order\""
        << ",\"host_gc_read_bytes\":" << stats_.host_gc_read_bytes
        << ",\"host_hbm_reserved_bytes\":" << stats_.host_hbm_reserved_bytes
        << ",\"host_hbm_read_bytes\":" << stats_.host_hbm_read_bytes
        << ",\"host_hbm_write_bytes\":" << stats_.host_hbm_write_bytes
        << ",\"host_hbm_busy_ns\":" << stats_.host_hbm_busy_ns
        << ",\"host_hbm_queue_wait_ns\":" << stats_.host_hbm_queue_wait_ns
        << ",\"host_gc_write_bytes\":" << stats_.host_gc_write_bytes
        << ",\"host_gc_control_ns\":" << stats_.host_gc_control_ns
        << ",\"host_zone_remaps\":" << stats_.host_zone_remaps
        << ",\"host_zone_invalidations\":" << stats_.host_zone_invalidations
        << ",\"host_zone_resets\":" << stats_.host_zone_resets
        << ",\"gc_runs\":" << stats_.gc_runs
        << ",\"gc_relocations\":" << stats_.gc_relocations
        << ",\"host_static_wl_relocations\":" << stats_.static_wear_leveling_relocations
        << ",\"preconditioned_erased_blocks\":" << stats_.preconditioned_erased_blocks
        << ",\"finish_ns\":" << stats_.finish_ns << ",\"initial_block_erase_counts\":[";
    // Array positions are stable physical identities, including static blocks.
    // A fresh controller has an implicit all-zero initial wear population.
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        out << (b ? "," : "")
            << (initial_erase_counts_.empty() ? 0 : initial_erase_counts_.at(b));
    }
    out << "],\"block_erase_counts\":[";
    for (std::size_t b = 0; b < blocks_.size(); ++b) {
        out << (b ? "," : "") << blocks_[b].erase_count;
    }
    out << "],\"bins\":[";
    bool comma = false;
    for (std::uint64_t channel = 0; channel < channels; ++channel) {
        const auto channel_begin = channel * blocks_per_channel;
        for (std::uint64_t offset = 0; offset < blocks_per_channel; offset += blocks_per_bin) {
            const auto begin = channel_begin + offset;
            const auto end = std::min(begin + blocks_per_bin, channel_begin + blocks_per_channel);
            std::uint64_t pec = 0, initial = 0, valid = 0, invalid = 0, free = 0, pending = 0;
            std::uint64_t statics = 0, raw = 0;
            auto minimum = std::numeric_limits<std::uint32_t>::max();
            std::uint32_t maximum = 0;
            for (auto b = begin; b < end; ++b) {
                const auto& block = blocks_[b];
                pec += block.erase_count;
                if (!initial_erase_counts_.empty()) initial += initial_erase_counts_.at(b);
                minimum = std::min(minimum, block.erase_count);
                maximum = std::max(maximum, block.erase_count);
                if (block.role == BlockRole::StaticReadOnly) {
                    ++statics; valid += config_.device.pages_per_block;
                } else {
                    valid += block.valid_pages;
                    invalid += block.invalid_pages;
                    free += block.free_pages;
                    pending += block.pending_program_pages;
                }
                if (block.role == BlockRole::RawPhysical) ++raw;
            }
            out << (comma ? "," : "") << "{\"stack\":" << channel / config_.device.channels_per_stack
                << ",\"channel\":" << channel % config_.device.channels_per_stack
                << ",\"block_begin\":" << begin << ",\"block_end\":" << end
                << ",\"physical_zone_begin\":" << offset / config_.host.zone_size_blocks
                << ",\"physical_zone_end\":" << (end - channel_begin) / config_.host.zone_size_blocks
                << ",\"pec_sum\":" << pec << ",\"pec_min\":" << minimum
                << ",\"pec_max\":" << maximum << ",\"initial_pec_sum\":" << initial
                << ",\"workload_erases\":" << pec - initial
                << ",\"valid_pages\":" << valid << ",\"invalid_pages\":" << invalid
                << ",\"free_pages\":" << free << ",\"pending_pages\":" << pending
                << ",\"static_blocks\":" << statics << ",\"raw_blocks\":" << raw << '}';
            comma = true;
        }
    }
    out << "],\"zone_mapping_default\":\"identity\",\"zone_remapping\":[";
    comma = false;
    for (const auto& [local, physical] : media_->zone_mapping()) {
        const auto channel = local / zones_per_channel;
        out << (comma ? "," : "") << "{\"stack\":" << channel / config_.device.channels_per_stack
            << ",\"channel\":" << channel % config_.device.channels_per_stack
            << ",\"local_zone\":" << local % zones_per_channel
            << ",\"physical_zone\":" << physical % zones_per_channel << '}';
        comma = true;
    }
    out << "]}";
}
} // namespace hbfsim::host
