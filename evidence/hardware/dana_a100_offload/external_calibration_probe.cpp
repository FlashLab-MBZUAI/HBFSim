#include "physical/external/external_backing_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using hbfsim::physical::AddressSpace;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::physical::external::ExternalBackingConfig;
using hbfsim::physical::external::ExternalBackingDevice;
using hbfsim::physical::external::parse_external_backing_kind;
using hbfsim::physical::external::to_string;

namespace {

struct Options {
    ExternalBackingConfig config;
    Op op = Op::Read;
    std::uint64_t block_bytes = 0;
    std::uint64_t operations = 0;
    std::uint32_t lanes = 0;
    bool open_loop = false;
};

[[noreturn]] void usage(const std::string& message = {}) {
    if (!message.empty()) {
        std::cerr << "error: " << message << "\n";
    }
    std::cerr
        << "usage: external_calibration_probe [config options] "
           "--op read|write --block-bytes N --lanes N --operations N "
           "--schedule open-loop|closed-loop\n"
        << "config options: --kind KIND --page-bytes N --segment-bytes N "
           "--media-channels N --media-read-queues N "
           "--media-write-queues N --max-outstanding N "
           "--controller-issue-ns X --controller-processing-ns X "
           "--media-read-latency-ns X --media-write-latency-ns X "
           "--media-read-bandwidth-gbps X "
           "--media-write-bandwidth-gbps X --m2s-bandwidth-gbps X "
           "--s2m-bandwidth-gbps X --one-way-propagation-ns X "
           "--command-bytes N --completion-bytes N\n";
    std::exit(message.empty() ? 0 : 2);
}

std::uint64_t parse_u64(const std::string& value, const std::string& name) {
    std::size_t used = 0;
    std::uint64_t parsed = 0;
    try {
        parsed = std::stoull(value, &used);
    } catch (const std::exception&) {
        usage(name + " must be an unsigned integer");
    }
    if (used != value.size()) {
        usage(name + " must be an unsigned integer");
    }
    return parsed;
}

std::uint32_t parse_u32(const std::string& value, const std::string& name) {
    const auto parsed = parse_u64(value, name);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        usage(name + " exceeds uint32 range");
    }
    return static_cast<std::uint32_t>(parsed);
}

double parse_double(const std::string& value, const std::string& name) {
    std::size_t used = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(value, &used);
    } catch (const std::exception&) {
        usage(name + " must be numeric");
    }
    if (used != value.size() || !std::isfinite(parsed)) {
        usage(name + " must be finite numeric data");
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    // The replay never allocates backing capacity. One TiB prevents the
    // evidence matrix from accidentally depending on a profile's study-size
    // capacity override.
    options.config.capacity_bytes = 1ull << 40;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help") {
            usage();
        }
        if (index + 1 >= argc) {
            usage("missing value for " + key);
        }
        const std::string value = argv[++index];
        if (key == "--kind") {
            options.config.kind = parse_external_backing_kind(value);
        } else if (key == "--page-bytes") {
            options.config.page_size_bytes = parse_u64(value, key);
        } else if (key == "--segment-bytes") {
            options.config.request_segment_bytes = parse_u64(value, key);
        } else if (key == "--media-channels") {
            options.config.media_channels = parse_u32(value, key);
        } else if (key == "--media-read-queues") {
            options.config.media_read_queues = parse_u32(value, key);
        } else if (key == "--media-write-queues") {
            options.config.media_write_queues = parse_u32(value, key);
        } else if (key == "--max-outstanding") {
            options.config.max_outstanding_requests = parse_u32(value, key);
        } else if (key == "--controller-issue-ns") {
            options.config.controller_issue_ns = parse_double(value, key);
        } else if (key == "--controller-processing-ns") {
            options.config.controller_processing_ns = parse_double(value, key);
        } else if (key == "--media-read-latency-ns") {
            options.config.media_read_latency_ns = parse_double(value, key);
        } else if (key == "--media-write-latency-ns") {
            options.config.media_write_latency_ns = parse_double(value, key);
        } else if (key == "--media-read-bandwidth-gbps") {
            options.config.media_read_bandwidth_GBps = parse_double(value, key);
        } else if (key == "--media-write-bandwidth-gbps") {
            options.config.media_write_bandwidth_GBps = parse_double(value, key);
        } else if (key == "--m2s-bandwidth-gbps") {
            options.config.m2s_bandwidth_GBps = parse_double(value, key);
        } else if (key == "--s2m-bandwidth-gbps") {
            options.config.s2m_bandwidth_GBps = parse_double(value, key);
        } else if (key == "--one-way-propagation-ns") {
            options.config.one_way_propagation_ns = parse_double(value, key);
        } else if (key == "--command-bytes") {
            options.config.command_bytes = parse_u32(value, key);
        } else if (key == "--completion-bytes") {
            options.config.completion_bytes = parse_u32(value, key);
        } else if (key == "--op") {
            if (value == "read") {
                options.op = Op::Read;
            } else if (value == "write") {
                options.op = Op::Write;
            } else {
                usage("--op must be read or write");
            }
        } else if (key == "--block-bytes") {
            options.block_bytes = parse_u64(value, key);
        } else if (key == "--lanes") {
            options.lanes = parse_u32(value, key);
        } else if (key == "--operations") {
            options.operations = parse_u64(value, key);
        } else if (key == "--schedule") {
            if (value == "open-loop") {
                options.open_loop = true;
            } else if (value == "closed-loop") {
                options.open_loop = false;
            } else {
                usage("--schedule must be open-loop or closed-loop");
            }
        } else {
            usage("unknown option " + key);
        }
    }
    if (options.block_bytes == 0 || options.operations == 0 ||
        options.lanes == 0) {
        usage("block bytes, operations, and lanes must be positive");
    }
    if (options.lanes > options.operations) {
        usage("lanes cannot exceed operations");
    }
    const auto max_address = options.operations >
            std::numeric_limits<std::uint64_t>::max() / options.block_bytes ?
        std::numeric_limits<std::uint64_t>::max() :
        options.operations * options.block_bytes;
    if (max_address > options.config.capacity_bytes) {
        usage("replay address range exceeds one TiB");
    }
    return options;
}

double percentile(std::vector<double> values, double quantile) {
    std::sort(values.begin(), values.end());
    const auto position = (values.size() - 1) * quantile;
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    if (lower == upper) {
        return values[lower];
    }
    const auto fraction = position - static_cast<double>(lower);
    return values[lower] + fraction * (values[upper] - values[lower]);
}

void run(const Options& options) {
    ExternalBackingDevice device(options.config);
    std::vector<double> lane_ready(options.lanes, 0.0);
    std::vector<double> latencies;
    latencies.reserve(options.operations);
    double finish_ns = 0.0;

    for (std::uint64_t index = 0; index < options.operations; ++index) {
        auto lane = std::size_t{0};
        if (!options.open_loop) {
            lane = static_cast<std::size_t>(std::min_element(
                lane_ready.begin(), lane_ready.end()) - lane_ready.begin());
        } else {
            lane = static_cast<std::size_t>(index % options.lanes);
        }
        const auto arrival_ns = options.open_loop ? 0.0 : lane_ready[lane];
        const auto completion = device.issue_contiguous_range(PhysicalRequest{
            .id = "calibration-" + std::to_string(index),
            .tier = Tier::External,
            .op = options.op,
            .address_space = AddressSpace::Logical,
            .arrival_ns = arrival_ns,
            .addr = index * options.block_bytes,
            .bytes = options.block_bytes,
        });
        lane_ready[lane] = completion.finish_ns;
        finish_ns = std::max(finish_ns, completion.finish_ns);
        latencies.push_back(completion.finish_ns - arrival_ns);
    }

    const auto total_bytes = options.operations * options.block_bytes;
    const auto& stats = device.stats();
    std::cout << std::setprecision(17)
              << "{\n"
              << "  \"schema\": \"hbfsim.external_calibration_replay.v1\",\n"
              << "  \"implementation\": \"ExternalBackingDevice::issue_contiguous_range\",\n"
              << "  \"kind\": \"" << to_string(options.config.kind) << "\",\n"
              << "  \"config\": {\"capacity_bytes\": "
              << options.config.capacity_bytes
              << ", \"page_size_bytes\": "
              << options.config.page_size_bytes
              << ", \"request_segment_bytes\": "
              << options.config.request_segment_bytes
              << ", \"media_channels\": "
              << options.config.media_channels
              << ", \"media_read_queues\": "
              << options.config.media_read_queues
              << ", \"media_write_queues\": "
              << options.config.media_write_queues
              << ", \"max_outstanding_requests\": "
              << options.config.max_outstanding_requests << "},\n"
              << "  \"op\": \"" << (options.op == Op::Read ? "read" : "write") << "\",\n"
              << "  \"schedule\": \"" << (options.open_loop ? "open-loop" : "closed-loop") << "\",\n"
              << "  \"block_bytes\": " << options.block_bytes << ",\n"
              << "  \"lanes\": " << options.lanes << ",\n"
              << "  \"operations\": " << options.operations << ",\n"
              << "  \"bytes\": " << total_bytes << ",\n"
              << "  \"wall_time_ns\": " << finish_ns << ",\n"
              << "  \"GB_per_s\": "
              << static_cast<double>(total_bytes) / finish_ns << ",\n"
              << "  \"latency_ns\": {\"p50\": "
              << percentile(latencies, 0.50) << ", \"min\": "
              << *std::min_element(latencies.begin(), latencies.end())
              << ", \"max\": "
              << *std::max_element(latencies.begin(), latencies.end())
              << "},\n"
              << "  \"stats\": {\"read_requests\": "
              << stats.read_requests << ", \"write_requests\": "
              << stats.write_requests << ", \"page_run_requests\": "
              << stats.page_run_requests << ", \"page_run_segments\": "
              << stats.page_run_segments << ", \"page_run_pages\": "
              << stats.page_run_pages << ", \"media_channels\": "
              << stats.media_channels
              << ", \"media_channels_per_queue\": "
              << stats.media_channels_per_queue
              << ", \"media_read_queues\": "
              << stats.media_read_queues
              << ", \"media_write_queues\": "
              << stats.media_write_queues
              << ", \"active_media_resources\": "
              << stats.active_media_resources
              << ", \"max_device_outstanding\": "
              << stats.max_device_outstanding << "}\n"
              << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        run(parse_options(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "external calibration replay failed: "
                  << error.what() << "\n";
        return 1;
    }
}
