#pragma once

#include <cstdint>
#include <string>

#include "obelisk/packet.hh"

namespace obelisk {

enum class AddressMappingPolicy {
    ROW_INTERLEAVE,
    CHANNEL_INTERLEAVE_FIRST,
    PAGE_CONTIGUOUS,
    CUSTOM,
};

AddressMappingPolicy parse_mapping_policy(const std::string& s);
const char* to_string(AddressMappingPolicy p);

struct AddressMapperConfig {
    AddressMappingPolicy policy = AddressMappingPolicy::CHANNEL_INTERLEAVE_FIRST;
    uint32_t num_stacks = 1;
    uint32_t num_channels = 8;
    uint32_t num_dies = 16;
    uint32_t num_subarrays = 32;
    uint32_t num_hbm_banks = 64;
    uint32_t page_size_bytes = 16384;

    Addr hbm_base_addr = 0x0000000000000000;
    uint64_t hbm_size_bytes = 96ULL * 1024 * 1024 * 1024;

    Addr hbf_base_addr = 0x0000001800000000;
    uint64_t hbf_size_bytes = 512ULL * 1024 * 1024 * 1024;
};

class AddressMapper {
public:
    explicit AddressMapper(const AddressMapperConfig& cfg);

    void map(Packet& pkt) const;

    MediaType which_media(Addr addr) const;

    const AddressMapperConfig& config() const { return cfg_; }

private:
    AddressMapperConfig cfg_;

    void map_hbf(Packet& pkt) const;
    void map_hbm(Packet& pkt) const;
};

}  // namespace obelisk
