#include "physical/address_heatmap.hpp"
#include "policies/policy_common.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using hbfsim::physical::AddressDomain;
using hbfsim::physical::AddressBoundary;
using hbfsim::physical::AddressHeatmap;
using hbfsim::physical::AddressHeatmapConfig;
using hbfsim::physical::AddressRegion;
using hbfsim::physical::AddressRegionKind;
using hbfsim::physical::AddressTrafficCounters;
using hbfsim::physical::AddressTrafficRecord;
using hbfsim::physical::HeatmapTrafficSource;
using hbfsim::physical::TrafficDirection;
using hbfsim::physical::kAddressDomainCount;
using hbfsim::physical::kDefaultAddressHeatmapBins;
using hbfsim::physical::kHeatmapTrafficSourceCount;
using hbfsim::physical::kMaxAddressHeatmapBins;
using hbfsim::physical::write_address_heatmap_json;
using hbfsim::policy::MemoryRequest;
using hbfsim::policy::SemanticKind;
using hbfsim::policy::make_composition_address_heatmap_config;
using hbfsim::policy::record_workload_address_traffic;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

AddressHeatmapConfig config_with(std::size_t bins, std::uint64_t size_bytes) {
    AddressHeatmapConfig config;
    config.bin_count = bins;
    for (auto& domain : config.domains) {
        domain.size_bytes = size_bytes;
    }
    return config;
}

std::uint64_t sum_read_bytes(const hbfsim::physical::AddressDomainHeatmap& domain) {
    std::uint64_t total = 0;
    for (const auto& bin : domain.bins) {
        total += bin.total.read_bytes;
    }
    return total;
}

std::uint64_t sum_write_bytes(const hbfsim::physical::AddressDomainHeatmap& domain) {
    std::uint64_t total = 0;
    for (const auto& bin : domain.bins) {
        total += bin.total.write_bytes;
    }
    return total;
}

std::uint64_t sum_erase_bytes(const hbfsim::physical::AddressDomainHeatmap& domain) {
    std::uint64_t total = 0;
    for (const auto& bin : domain.bins) {
        total += bin.total.erase_bytes;
    }
    return total;
}

void test_default_and_config_validation() {
    auto config = config_with(kDefaultAddressHeatmapBins, 4096);
    AddressHeatmap heatmap(config);
    for (std::size_t domain = 0; domain < kAddressDomainCount; ++domain) {
        const auto& state = heatmap.domain(static_cast<AddressDomain>(domain));
        require(
            state.bins.size() == kDefaultAddressHeatmapBins,
            "default bin count was not applied to every domain");
        require(state.bins.front().begin == 0, "first bin must begin at zero");
        require(state.bins.back().end == 4096, "last bin must end at domain size");
    }

    require_throws<std::invalid_argument>(
        [] { AddressHeatmap invalid(config_with(0, 4096)); },
        "zero bins must be rejected");
    require_throws<std::invalid_argument>(
        [] {
            AddressHeatmap invalid(config_with(kMaxAddressHeatmapBins + 1, 1U << 20));
        },
        "bin count above the hard limit must be rejected");
    require_throws<std::invalid_argument>(
        [] { AddressHeatmap invalid(config_with(16, 15)); },
        "domains smaller than the bin count must be rejected");

    auto duplicate = config_with(4, 64);
    duplicate.domains[3].domain = AddressDomain::HbfLogical;
    require_throws<std::invalid_argument>(
        [&] { AddressHeatmap invalid(duplicate); },
        "duplicate domains must be rejected");

    auto invalid_region = config_with(4, 64);
    invalid_region.domains[0].regions.push_back(AddressRegion{
        .name = "outside",
        .begin = 63,
        .end = 65,
    });
    require_throws<std::invalid_argument>(
        [&] { AddressHeatmap invalid(invalid_region); },
        "out-of-domain region must be rejected");
}

void test_contiguous_access_batch_matches_individual_records() {
    const auto config = config_with(13, 1024);
    AddressHeatmap individual(config);
    AddressHeatmap batched(config);
    const AddressTrafficRecord first{
        .domain = AddressDomain::HbmPhysical,
        .direction = TrafficDirection::Read,
        .source = HeatmapTrafficSource::Direct,
        .address = 13,
        .bytes = 64,
    };
    for (std::uint64_t access = 0; access < 10; ++access) {
        auto record = first;
        record.address += access * first.bytes;
        individual.record(record);
    }
    batched.record_contiguous_accesses(first, 10);
    std::ostringstream individual_json;
    std::ostringstream batched_json;
    individual.write_json(individual_json);
    batched.write_json(batched_json);
    require(
        individual_json.str() == batched_json.str(),
        "contiguous heatmap batching changed byte/access/source accounting");
}

void test_exact_overlap_access_semantics_and_sources() {
    AddressHeatmap heatmap(config_with(3, 10));
    heatmap.record(AddressTrafficRecord{
        .domain = AddressDomain::WorkloadLogical,
        .direction = TrafficDirection::Read,
        .source = HeatmapTrafficSource::Workload,
        .address = 2,
        .bytes = 6,
    });
    heatmap.record(AddressTrafficRecord{
        .domain = AddressDomain::WorkloadLogical,
        .direction = TrafficDirection::Write,
        .source = HeatmapTrafficSource::CooperativeBuffer,
        .address = 0,
        .bytes = 10,
    });
    heatmap.record(AddressTrafficRecord{
        .domain = AddressDomain::WorkloadLogical,
        .direction = TrafficDirection::Erase,
        .source = HeatmapTrafficSource::Mapping,
        .address = 5,
        .bytes = 2,
    });
    heatmap.record(AddressTrafficRecord{
        .domain = AddressDomain::WorkloadLogical,
        .direction = TrafficDirection::Read,
        .source = HeatmapTrafficSource::Prepopulate,
        .address = 9,
        .bytes = 1,
    });

    const auto& domain = heatmap.domain(AddressDomain::WorkloadLogical);
    require(
        domain.bins[0].begin == 0 && domain.bins[0].end == 3 &&
            domain.bins[1].begin == 3 && domain.bins[1].end == 6 &&
            domain.bins[2].begin == 6 && domain.bins[2].end == 10,
        "non-divisible domain boundaries are not the canonical integer partition");
    require(
        domain.bins[0].total.read_bytes == 1 &&
            domain.bins[1].total.read_bytes == 3 &&
            domain.bins[2].total.read_bytes == 3,
        "cross-bin read overlap was not distributed exactly");
    require(
        domain.bins[0].total.write_bytes == 3 &&
            domain.bins[1].total.write_bytes == 3 &&
            domain.bins[2].total.write_bytes == 4,
        "whole-domain write overlap was not distributed exactly");
    require(
        domain.bins[0].total.erase_bytes == 0 &&
            domain.bins[1].total.erase_bytes == 1 &&
            domain.bins[2].total.erase_bytes == 1,
        "erase traffic must remain separate from writes");
    require(
        domain.total.read_bytes == 7 && domain.total.write_bytes == 10 &&
            domain.total.erase_bytes == 2,
        "domain traffic totals are wrong");
    require(
        sum_read_bytes(domain) == domain.total.read_bytes &&
            sum_write_bytes(domain) == domain.total.write_bytes &&
            sum_erase_bytes(domain) == domain.total.erase_bytes,
        "bin bytes do not conserve domain bytes");

    require(
        domain.total.read_accesses == 2 && domain.total.write_accesses == 1 &&
            domain.total.erase_accesses == 1,
        "domain accesses must count accepted records once");
    require(
        domain.bins[0].total.read_accesses == 1 &&
            domain.bins[1].total.read_accesses == 1 &&
            domain.bins[2].total.read_accesses == 2,
        "bin accesses must count each touching record once");

    const auto cooperative =
        static_cast<std::size_t>(HeatmapTrafficSource::CooperativeBuffer);
    const auto mapping = static_cast<std::size_t>(HeatmapTrafficSource::Mapping);
    const auto prepopulate =
        static_cast<std::size_t>(HeatmapTrafficSource::Prepopulate);
    require(
        domain.by_source[cooperative].write_bytes == 10 &&
            domain.by_source[cooperative].write_accesses == 1,
        "cooperative-buffer source attribution was lost");
    require(
        domain.by_source[mapping].erase_bytes == 2 &&
            domain.by_source[mapping].erase_accesses == 1,
        "mapping erase source attribution was lost");
    require(
        domain.by_source[prepopulate].read_bytes == 1 &&
            domain.bins[2].by_source[prepopulate].read_accesses == 1,
        "prepopulate source attribution was lost");
}

void test_top_address_and_strong_overflow_guarantee() {
    constexpr auto top = std::numeric_limits<std::uint64_t>::max();
    AddressHeatmap heatmap(config_with(4, top));
    heatmap.record(AddressTrafficRecord{
        .domain = AddressDomain::HbfPhysical,
        .direction = TrafficDirection::Erase,
        .source = HeatmapTrafficSource::GarbageCollection,
        .address = top - 10,
        .bytes = 10,
    });
    heatmap.record(AddressTrafficRecord{
        .domain = AddressDomain::HbfPhysical,
        .direction = TrafficDirection::Write,
        .source = HeatmapTrafficSource::Maintenance,
        .address = top - 1,
        .bytes = 1,
    });
    const auto& top_domain = heatmap.domain(AddressDomain::HbfPhysical);
    require(
        top_domain.bins.back().total.erase_bytes == 10 &&
            top_domain.bins.back().total.write_bytes == 1,
        "traffic ending at the largest representable exclusive limit was lost");
    require_throws<std::out_of_range>(
        [&] {
            heatmap.record(AddressTrafficRecord{
                .domain = AddressDomain::HbfPhysical,
                .direction = TrafficDirection::Read,
                .source = HeatmapTrafficSource::Direct,
                .address = top - 1,
                .bytes = 2,
            });
        },
        "top-address overflow must be rejected before address addition");
    require(
        top_domain.total.erase_bytes == 10 && top_domain.total.write_bytes == 1 &&
            top_domain.total.read_bytes == 0,
        "rejected top-address traffic partially changed counters");

    AddressHeatmap overflow(config_with(1, top));
    overflow.record(AddressTrafficRecord{
        .domain = AddressDomain::HbmPhysical,
        .direction = TrafficDirection::Read,
        .source = HeatmapTrafficSource::Direct,
        .address = 0,
        .bytes = top,
    });
    require_throws<std::overflow_error>(
        [&] {
            overflow.record(AddressTrafficRecord{
                .domain = AddressDomain::HbmPhysical,
                .direction = TrafficDirection::Read,
                .source = HeatmapTrafficSource::Direct,
                .address = 0,
                .bytes = 1,
            });
        },
        "counter overflow must be rejected");
    const auto& overflow_domain = overflow.domain(AddressDomain::HbmPhysical);
    require(
        overflow_domain.total.read_bytes == top &&
            overflow_domain.total.read_accesses == 1 &&
            overflow_domain.bins[0].total.read_bytes == top &&
            overflow_domain.bins[0].total.read_accesses == 1,
        "counter overflow violated the strong update guarantee");
}

void test_full_uint64_address_space_endpoint() {
    constexpr auto top = std::numeric_limits<std::uint64_t>::max();
    auto config = config_with(4, top);
    for (auto& domain : config.domains) {
        domain.size_bytes = AddressBoundary::full_address_space_end();
    }
    AddressHeatmap heatmap(config);
    heatmap.record(AddressTrafficRecord{
        .domain = AddressDomain::WorkloadLogical,
        .direction = TrafficDirection::Read,
        .source = HeatmapTrafficSource::Workload,
        .address = top,
        .bytes = 1,
    });
    const auto& workload = heatmap.domain(AddressDomain::WorkloadLogical);
    require(
        workload.size_bytes.is_full_address_space_end() &&
            workload.bins.back().end.is_full_address_space_end() &&
            workload.bins.back().total.read_bytes == 1,
        "the highest uint64 address was not retained in the final bin");

    std::ostringstream json;
    heatmap.write_json(json);
    require(
        json.str().find("\"size_bytes\":18446744073709551616") !=
            std::string::npos &&
            json.str().find("\"end\":18446744073709551616") !=
                std::string::npos,
        "the full uint64 address-space endpoint was not serialized exactly");

    // Exercise a non-power-of-two partition and a request ending at 2^64.
    // This guards the portable quotient/remainder implementation against the
    // off-by-one that arises when treating 2^64 as UINT64_MAX.
    auto thirds_config = config_with(3, top);
    for (auto& domain : thirds_config.domains) {
        domain.size_bytes = AddressBoundary::full_address_space_end();
    }
    AddressHeatmap thirds(std::move(thirds_config));
    const auto& empty_thirds = thirds.domain(AddressDomain::WorkloadLogical);
    require(
        empty_thirds.bins[0].begin == 0 &&
            empty_thirds.bins[0].end == 6148914691236517205ULL &&
            empty_thirds.bins[1].begin == 6148914691236517205ULL &&
            empty_thirds.bins[1].end == 12297829382473034410ULL &&
            empty_thirds.bins[2].begin == 12297829382473034410ULL &&
            empty_thirds.bins[2].end.is_full_address_space_end(),
        "full-space thirds do not use exact canonical boundaries");
    thirds.record(AddressTrafficRecord{
        .domain = AddressDomain::WorkloadLogical,
        .direction = TrafficDirection::Read,
        .source = HeatmapTrafficSource::Workload,
        .address = 1,
        .bytes = top,
    });
    const auto& populated_thirds = thirds.domain(AddressDomain::WorkloadLogical);
    require(
        populated_thirds.total.read_bytes == top &&
            sum_read_bytes(populated_thirds) == top &&
            populated_thirds.bins[0].total.read_bytes ==
                6148914691236517204ULL &&
            populated_thirds.bins[1].total.read_bytes ==
                6148914691236517205ULL &&
            populated_thirds.bins[2].total.read_bytes ==
                6148914691236517206ULL,
        "a request ending at 2^64 did not conserve bytes across thirds");

    // The default composition heatmap must derive the same extent from a legal
    // request instead of rejecting its exclusive end at 2^64.
    hbfsim::physical::hbm::HbmConfig hbm;
    hbfsim::host::HbfConfig hbf;
    const std::vector<MemoryRequest> requests{{
        .id = "top-byte",
        .op = hbfsim::physical::Op::Read,
        .addr = top,
        .bytes = 1,
        .arrival_ns = 0.0,
        .index = 0,
        .kind = SemanticKind::Metadata,
    }};
    auto composition_config = make_composition_address_heatmap_config(
        hbm, hbf, requests, kDefaultAddressHeatmapBins);
    AddressHeatmap composition_heatmap(std::move(composition_config));
    record_workload_address_traffic(composition_heatmap, requests);
    const auto& composition_workload = composition_heatmap.domain(
        AddressDomain::WorkloadLogical);
    require(
        composition_workload.size_bytes.is_full_address_space_end() &&
            composition_workload.total.read_bytes == 1,
        "default composition heatmap rejected the highest uint64 byte");
}

void test_fixed_memory_and_deterministic_json() {
    auto config = config_with(64, 4096);
    config.domains[0].regions = {
        AddressRegion{
            .name = "staging \"window\"",
            .kind = AddressRegionKind::LayerBuffer,
            .begin = 2048,
            .end = 4096,
        },
        AddressRegion{
            .name = "weights",
            .kind = AddressRegionKind::StaticData,
            .begin = 0,
            .end = 1024,
        },
    };
    AddressHeatmap heatmap(config);
    const auto& before = heatmap.domain(AddressDomain::WorkloadLogical).bins;
    const auto* const storage = before.data();
    const auto capacity = before.capacity();
    for (std::uint64_t i = 0; i < 10000; ++i) {
        heatmap.record(AddressTrafficRecord{
            .domain = AddressDomain::WorkloadLogical,
            .direction = i % 2 == 0 ?
                TrafficDirection::Read : TrafficDirection::Write,
            .source = HeatmapTrafficSource::Workload,
            .address = (i * 17) % 4096,
            .bytes = 1,
        });
    }
    const auto& after = heatmap.domain(AddressDomain::WorkloadLogical).bins;
    require(
        after.data() == storage && after.capacity() == capacity &&
            after.size() == 64,
        "recording traffic changed fixed heatmap storage");

    std::ostringstream first;
    std::ostringstream second;
    heatmap.write_json(first);
    heatmap.write_json(second);
    require(first.str() == second.str(), "heatmap JSON is not deterministic");
    const auto& json = first.str();
    require(
        json.find("\"schema\":\"hbfsim.address_heatmap.v1\"") !=
            std::string::npos,
        "heatmap JSON schema is missing");
    require(
        json.find("\"mapping\"") != std::string::npos &&
            json.find("\"prepopulate\"") != std::string::npos &&
            json.find("\"cooperative_buffer\"") != std::string::npos,
        "structured source vocabulary is incomplete in JSON");
    require(
        json.find("\"erase_bytes\"") != std::string::npos &&
            json.find("\"erase_accesses\"") != std::string::npos,
        "erase traffic is missing from JSON");
    require(
        json.find("staging \\\"window\\\"") != std::string::npos,
        "region names are not JSON escaped");
    require(
        json.find("weights") < json.find("staging \\\"window\\\""),
        "region ordering is not canonical by address");

    const auto snapshot = heatmap.snapshot();
    require(snapshot.bin_count == 64, "snapshot lost the configured bin count");
    require(
        snapshot.domains[0].bins.size() == 64,
        "snapshot lost fixed-size bin storage");
    std::ostringstream retained_json;
    write_address_heatmap_json(retained_json, snapshot);
    require(
        retained_json.str() == json,
        "free snapshot writer diverges from the live accumulator writer");

    auto corrupt = snapshot;
    ++corrupt.domains[0].bins[0].total.read_bytes;
    std::ostringstream rejected_json;
    require_throws<std::invalid_argument>(
        [&] { write_address_heatmap_json(rejected_json, corrupt); },
        "snapshot writer must reject non-conserving retained data");
    require(
        rejected_json.str().empty(),
        "snapshot validation must complete before JSON output begins");
}

void test_invalid_records_and_vocabularies() {
    AddressHeatmap heatmap(config_with(4, 64));
    require_throws<std::invalid_argument>(
        [&] {
            heatmap.record(AddressTrafficRecord{
                .domain = AddressDomain::HbmPhysical,
                .bytes = 0,
            });
        },
        "zero-byte record must be rejected");
    require_throws<std::invalid_argument>(
        [&] {
            heatmap.record(AddressTrafficRecord{
                .domain = AddressDomain::HbmPhysical,
                .source = static_cast<HeatmapTrafficSource>(
                    kHeatmapTrafficSourceCount),
                .bytes = 1,
            });
        },
        "invalid traffic source enum must be rejected");
    require_throws<std::invalid_argument>(
        [&] {
            heatmap.record(AddressTrafficRecord{
                .domain = AddressDomain::HbmPhysical,
                .direction = static_cast<TrafficDirection>(255),
                .bytes = 1,
            });
        },
        "invalid traffic direction enum must be rejected");
    require_throws<std::invalid_argument>(
        [&] { (void)heatmap.domain(static_cast<AddressDomain>(255)); },
        "invalid domain enum must be rejected");
}

void emit_interoperability_fixture(const std::string& path) {
    auto config = config_with(4, 8192);
    config.domains[static_cast<std::size_t>(AddressDomain::WorkloadLogical)]
        .size_bytes = AddressBoundary::full_address_space_end();
    for (std::size_t domain = 0; domain < kAddressDomainCount; ++domain) {
        config.domains[domain].regions.push_back(AddressRegion{
            .name = "fixture-region-" + std::to_string(domain),
            .kind = static_cast<AddressRegionKind>(
                domain % (static_cast<std::size_t>(AddressRegionKind::Other) + 1)),
            .begin = 0,
            .end = 2048,
        });
    }
    AddressHeatmap heatmap(config);
    for (std::size_t domain = 0; domain < kAddressDomainCount; ++domain) {
        const auto address_domain = static_cast<AddressDomain>(domain);
        heatmap.record(AddressTrafficRecord{
            .domain = address_domain,
            .direction = TrafficDirection::Read,
            .source = HeatmapTrafficSource::Workload,
            .address = 1024,
            .bytes = 4096,
        });
        heatmap.record(AddressTrafficRecord{
            .domain = address_domain,
            .direction = TrafficDirection::Write,
            .source = HeatmapTrafficSource::CooperativeBuffer,
            .address = 4096,
            .bytes = 1024,
        });
        heatmap.record(AddressTrafficRecord{
            .domain = address_domain,
            .direction = TrafficDirection::Erase,
            .source = HeatmapTrafficSource::GarbageCollection,
            .address = 6144,
            .bytes = 2048,
        });
    }
    heatmap.record(AddressTrafficRecord{
        .domain = AddressDomain::WorkloadLogical,
        .direction = TrafficDirection::Read,
        .source = HeatmapTrafficSource::Workload,
        .address = std::numeric_limits<std::uint64_t>::max(),
        .bytes = 1,
    });
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("could not open heatmap fixture output: " + path);
    }
    heatmap.write_json(output);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--emit-json") {
            emit_interoperability_fixture(argv[2]);
            return 0;
        }
        if (argc != 1) {
            throw std::runtime_error(
                "usage: address_heatmap_test [--emit-json OUTPUT.json]");
        }
        test_default_and_config_validation();
        test_contiguous_access_batch_matches_individual_records();
        test_exact_overlap_access_semantics_and_sources();
        test_top_address_and_strong_overflow_guarantee();
        test_full_uint64_address_space_endpoint();
        test_fixed_memory_and_deterministic_json();
        test_invalid_records_and_vocabularies();
        std::cout << "address_heatmap_test: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "address_heatmap_test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
