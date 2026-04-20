#include <catch2/catch_test_macros.hpp>

#include "src/memory_system/address_mapper.hh"

using namespace obelisk;

TEST_CASE("AddressMapper: routes HBM vs HBF by range", "[address_mapper]") {
    AddressMapperConfig cfg;
    cfg.hbm_base_addr = 0x0;
    cfg.hbm_size_bytes = 0x40000000;  // 1 GB
    cfg.hbf_base_addr = 0x1800000000;
    cfg.hbf_size_bytes = 0x100000000;  // 4 GB
    AddressMapper mapper(cfg);

    REQUIRE(mapper.which_media(0x0) == MediaType::HBM);
    REQUIRE(mapper.which_media(0x20000000) == MediaType::HBM);
    REQUIRE(mapper.which_media(0x1800000000) == MediaType::HBF);
    REQUIRE(mapper.which_media(0x1800004000) == MediaType::HBF);
}

TEST_CASE("AddressMapper: channel_interleave_first stripes consecutive pages", "[address_mapper]") {
    AddressMapperConfig cfg;
    cfg.policy = AddressMappingPolicy::CHANNEL_INTERLEAVE_FIRST;
    cfg.num_channels = 4;
    cfg.num_dies = 4;
    cfg.num_subarrays = 8;
    cfg.page_size_bytes = 16384;
    cfg.hbm_base_addr = 0x0;
    cfg.hbm_size_bytes = 0;
    cfg.hbf_base_addr = 0x0;
    cfg.hbf_size_bytes = 1ULL << 32;
    AddressMapper mapper(cfg);

    std::array<uint32_t, 4> channels{};
    for (int i = 0; i < 4; ++i) {
        Packet p;
        p.addr = static_cast<uint64_t>(i) * cfg.page_size_bytes;
        mapper.map(p);
        channels[i] = p.phys.channel_id;
    }
    // Consecutive pages should hit all 4 channels.
    std::sort(channels.begin(), channels.end());
    REQUIRE(channels[0] == 0);
    REQUIRE(channels[1] == 1);
    REQUIRE(channels[2] == 2);
    REQUIRE(channels[3] == 3);
}

TEST_CASE("AddressMapper: page_contiguous keeps pages in same subarray", "[address_mapper]") {
    AddressMapperConfig cfg;
    cfg.policy = AddressMappingPolicy::PAGE_CONTIGUOUS;
    cfg.num_channels = 4;
    cfg.num_dies = 4;
    cfg.num_subarrays = 8;
    cfg.page_size_bytes = 16384;
    cfg.hbm_base_addr = 0x0;
    cfg.hbm_size_bytes = 0;
    cfg.hbf_base_addr = 0x0;
    cfg.hbf_size_bytes = 1ULL << 32;
    AddressMapper mapper(cfg);

    Packet p0, p1;
    p0.addr = 0;
    p1.addr = cfg.page_size_bytes;  // next page
    mapper.map(p0);
    mapper.map(p1);
    // With page_contiguous, consecutive pages step subarray_id.
    REQUIRE(p0.phys.subarray_id != p1.phys.subarray_id);
    REQUIRE(p0.phys.channel_id == p1.phys.channel_id);
    REQUIRE(p0.phys.die_id == p1.phys.die_id);
}
