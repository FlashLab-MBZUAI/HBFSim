#include "src/memory_system/address_mapper.hh"

#include <stdexcept>

namespace obelisk {

AddressMappingPolicy parse_mapping_policy(const std::string& s) {
    if (s == "row_interleave") return AddressMappingPolicy::ROW_INTERLEAVE;
    if (s == "channel_interleave_first" || s == "channel_first") return AddressMappingPolicy::CHANNEL_INTERLEAVE_FIRST;
    if (s == "page_contiguous" || s == "page_contig") return AddressMappingPolicy::PAGE_CONTIGUOUS;
    if (s == "custom") return AddressMappingPolicy::CUSTOM;
    throw std::runtime_error("unknown address mapping policy: " + s);
}

const char* to_string(AddressMappingPolicy p) {
    switch (p) {
        case AddressMappingPolicy::ROW_INTERLEAVE: return "row_interleave";
        case AddressMappingPolicy::CHANNEL_INTERLEAVE_FIRST: return "channel_interleave_first";
        case AddressMappingPolicy::PAGE_CONTIGUOUS: return "page_contiguous";
        case AddressMappingPolicy::CUSTOM: return "custom";
    }
    return "unknown";
}

AddressMapper::AddressMapper(const AddressMapperConfig& cfg) : cfg_(cfg) {}

MediaType AddressMapper::which_media(Addr addr) const {
    if (addr >= cfg_.hbm_base_addr && addr < cfg_.hbm_base_addr + cfg_.hbm_size_bytes) return MediaType::HBM;
    if (addr >= cfg_.hbf_base_addr && addr < cfg_.hbf_base_addr + cfg_.hbf_size_bytes) return MediaType::HBF;
    // Default: treat out-of-range as HBF (fail-open for trace generators that
    // don't set addresses correctly).
    return MediaType::HBF;
}

void AddressMapper::map_hbf(Packet& pkt) const {
    Addr rel = (pkt.addr >= cfg_.hbf_base_addr) ? (pkt.addr - cfg_.hbf_base_addr) : pkt.addr;
    uint64_t page_id = rel / cfg_.page_size_bytes;
    pkt.phys.page_id = page_id;
    pkt.phys.stack_id = 0;

    const uint32_t C = cfg_.num_channels ? cfg_.num_channels : 1;
    const uint32_t D = cfg_.num_dies ? cfg_.num_dies : 1;
    const uint32_t S = cfg_.num_subarrays ? cfg_.num_subarrays : 1;

    switch (cfg_.policy) {
        case AddressMappingPolicy::CHANNEL_INTERLEAVE_FIRST:
        case AddressMappingPolicy::ROW_INTERLEAVE:
            pkt.phys.channel_id = page_id % C;
            pkt.phys.die_id = (page_id / C) % D;
            pkt.phys.subarray_id = (page_id / C / D) % S;
            break;
        case AddressMappingPolicy::PAGE_CONTIGUOUS:
            pkt.phys.subarray_id = page_id % S;
            pkt.phys.die_id = (page_id / S) % D;
            pkt.phys.channel_id = (page_id / S / D) % C;
            break;
        case AddressMappingPolicy::CUSTOM:
            // M2: read mapping from TOML.
            pkt.phys.channel_id = page_id % C;
            pkt.phys.die_id = (page_id / C) % D;
            pkt.phys.subarray_id = (page_id / C / D) % S;
            break;
    }

    pkt.phys.row = static_cast<uint32_t>(rel % cfg_.page_size_bytes);  // within-page offset
    pkt.phys.column = 0;
}

void AddressMapper::map_hbm(Packet& pkt) const {
    Addr rel = (pkt.addr >= cfg_.hbm_base_addr) ? (pkt.addr - cfg_.hbm_base_addr) : pkt.addr;
    const uint32_t B = cfg_.num_hbm_banks ? cfg_.num_hbm_banks : 1;
    const uint32_t C = cfg_.num_channels ? cfg_.num_channels : 1;
    // HBM granularity is smaller (64 B typical). We still compute a
    // "page_id" for stats purposes.
    uint64_t blk = rel / 64;
    pkt.phys.page_id = blk;
    pkt.phys.stack_id = 0;
    pkt.phys.channel_id = blk % C;
    pkt.phys.bank_id = (blk / C) % B;
    pkt.phys.die_id = 0;
    pkt.phys.subarray_id = 0;
    pkt.phys.row = static_cast<uint32_t>(rel & 0xFFFFULL);
    pkt.phys.column = 0;
}

void AddressMapper::map(Packet& pkt) const {
    pkt.target_media = which_media(pkt.addr);
    if (pkt.target_media == MediaType::HBF)
        map_hbf(pkt);
    else
        map_hbm(pkt);
}

}  // namespace obelisk
