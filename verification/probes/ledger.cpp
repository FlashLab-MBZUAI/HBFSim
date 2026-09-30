#include "host/hbf_controller.hpp"
#include "physical/hbm/hbm_device.hpp"
#include "physical/external/external_backing_device.hpp"
#include "policies/reference/direct_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

using hbfsim::physical::AddressSpace;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalCompletion;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::physical::TraceMode;
using hbfsim::physical::TraceSpan;
using hbfsim::host::HbfConfig;
using hbfsim::host::HbfController;
using hbfsim::host::HbfAuditSnapshot;
using hbfsim::physical::hbm::HbmConfig;
using hbfsim::physical::hbm::HbmDevice;
using hbfsim::physical::external::ExternalBackingConfig;
using hbfsim::physical::external::ExternalBackingDevice;
using hbfsim::policy::DirectRunKnobs;
using hbfsim::policy::MemoryRequest;
using hbfsim::policy::flat_policy;
using hbfsim::policy::run_direct_policy;

namespace {

struct RequestSpec {
    std::string id;
    double arrival_ns = 0.0;
    Op op = Op::Read;
    bool drain = false;
    AddressSpace address_space = AddressSpace::Logical;
    std::string address_space_name = "logical";
    std::uint64_t addr = 0;
    std::uint64_t bytes = 0;
};

struct Options {
    std::string case_id;
    std::string model;
    HbmConfig hbm;
    HbfConfig hbf;
    ExternalBackingConfig external;
    std::uint64_t hybrid_read_boundary = 0;
    std::size_t hybrid_max_outstanding_requests = 0;
    std::vector<std::uint64_t> inspect_addresses;
    std::vector<std::uint64_t> prepopulate_lpns;
    std::vector<RequestSpec> requests;
};

[[nodiscard]] std::string json_escape(std::string_view input) {
    std::ostringstream out;
    for (const char ch : input) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                out << "\\u"
                    << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<unsigned int>(
                           static_cast<unsigned char>(ch))
                    << std::dec << std::setfill(' ');
            } else {
                out << ch;
            }
            break;
        }
    }
    return out.str();
}

template <typename T>
[[nodiscard]] T parse_unsigned(const std::string& value, const char* label) {
    static_assert(std::is_unsigned_v<T>);
    if (value.empty() ||
        !std::all_of(value.begin(), value.end(), [](const char ch) {
            return ch >= '0' && ch <= '9';
        })) {
        throw std::runtime_error(
            std::string("invalid ") + label + ": " + value);
    }
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size() ||
        parsed > static_cast<unsigned long long>(
            std::numeric_limits<T>::max())) {
        throw std::runtime_error(
            std::string("invalid ") + label + ": " + value);
    }
    return static_cast<T>(parsed);
}

[[nodiscard]] double parse_double(
    const std::string& value,
    const char* label) {
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::runtime_error(
            std::string("invalid ") + label + ": " + value);
    }
    return parsed;
}

[[nodiscard]] bool parse_bool(
    const std::string& value,
    const char* label) {
    if (value == "true") {
        return true;
    }
    if (value == "false") {
        return false;
    }
    throw std::runtime_error(
        std::string("invalid ") + label + ": " + value);
}

[[nodiscard]] Op parse_op(const std::string& value) {
    if (value == "read") {
        return Op::Read;
    }
    if (value == "write") {
        return Op::Write;
    }
    throw std::runtime_error("unsupported validation request op: " + value);
}

[[nodiscard]] AddressSpace parse_address_space(const std::string& value) {
    if (value == "logical") {
        return AddressSpace::Logical;
    }
    if (value == "physical") {
        return AddressSpace::Physical;
    }
    throw std::runtime_error(
        "unsupported validation address space: " + value);
}

void apply_hbm_config(
    HbmConfig& config,
    const std::string& assignment) {
    const auto separator = assignment.find('=');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == assignment.size()) {
        throw std::runtime_error(
            "--config expects key=value, got: " + assignment);
    }
    const auto key = assignment.substr(0, separator);
    const auto value = assignment.substr(separator + 1);

#define HBFSIM_SET_U64(group, field)                                                   \
    if (key == #field) {                                                       \
        config.group.field = parse_unsigned<std::uint64_t>(value, #field);           \
        return;                                                                \
    }
#define HBFSIM_SET_U32(group, field)                                                   \
    if (key == #field) {                                                       \
        config.group.field = parse_unsigned<std::uint32_t>(value, #field);           \
        return;                                                                \
    }
#define HBFSIM_SET_DOUBLE(group, field)                                                \
    if (key == #field) {                                                       \
        config.group.field = parse_double(value, #field);                            \
        return;                                                                \
    }
#define HBFSIM_SET_BOOL(group, field)                                                  \
    if (key == #field) {                                                       \
        config.group.field = parse_bool(value, #field);                              \
        return;                                                                \
    }

HBFSIM_SET_U64(device, capacity_bytes)
HBFSIM_SET_U32(device, stacks)
HBFSIM_SET_U32(device, channels_per_stack)
HBFSIM_SET_U32(device, pseudo_channels_per_channel)
HBFSIM_SET_U32(device, bank_groups_per_pseudo_channel)
HBFSIM_SET_U32(device, banks_per_group)
HBFSIM_SET_U64(device, channel_row_size_bytes)
HBFSIM_SET_U32(device, channel_width_bits)
HBFSIM_SET_U32(device, burst_length)
HBFSIM_SET_DOUBLE(device, pin_rate_Gbps)
HBFSIM_SET_U32(device, data_rate_per_command_clock)
HBFSIM_SET_DOUBLE(controller, address_mapping_ns)
HBFSIM_SET_DOUBLE(timing, read_latency_ns)
HBFSIM_SET_DOUBLE(timing, write_latency_ns)
HBFSIM_SET_DOUBLE(timing, read_to_write_ns)
HBFSIM_SET_DOUBLE(timing, write_to_read_ns)
HBFSIM_SET_DOUBLE(timing, bandwidth_efficiency)
HBFSIM_SET_U32(controller, queue_depth)
HBFSIM_SET_U64(controller, interleave_bytes)
HBFSIM_SET_U64(controller, service_quantum_bytes)
HBFSIM_SET_U32(controller, service_group_channels)

#undef HBFSIM_SET_U64
#undef HBFSIM_SET_U32
#undef HBFSIM_SET_DOUBLE
#undef HBFSIM_SET_BOOL

    throw std::runtime_error("unknown HBM validation config key: " + key);
}

void apply_hbf_config(
    HbfConfig& config,
    const std::string& assignment) {
    const auto separator = assignment.find('=');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == assignment.size()) {
        throw std::runtime_error(
            "--config expects key=value, got: " + assignment);
    }
    const auto key = assignment.substr(0, separator);
    const auto value = assignment.substr(separator + 1);

#define HBFSIM_SET_HBF_U64(group, field)                                               \
    if (key == #field) {                                                       \
        config.group.field = parse_unsigned<std::uint64_t>(value, #field);           \
        return;                                                                \
    }
#define HBFSIM_SET_HBF_U32(group, field)                                               \
    if (key == #field) {                                                       \
        config.group.field = parse_unsigned<std::uint32_t>(value, #field);           \
        return;                                                                \
    }
#define HBFSIM_SET_HBF_DOUBLE(group, field)                                            \
    if (key == #field) {                                                       \
        config.group.field = parse_double(value, #field);                            \
        return;                                                                \
    }
#define HBFSIM_SET_HBF_BOOL(group, field)                                              \
    if (key == #field) {                                                       \
        config.group.field = parse_bool(value, #field);                              \
        return;                                                                \
    }

HBFSIM_SET_HBF_DOUBLE(host, host_gc_decision_ns)
HBFSIM_SET_HBF_U32(device, speed_grade)
    HBFSIM_SET_HBF_U32(device, stacks)
HBFSIM_SET_HBF_U32(device, channels_per_stack)
HBFSIM_SET_HBF_U32(device, dies_per_channel)
HBFSIM_SET_HBF_U32(device, planes_per_die)
HBFSIM_SET_HBF_U32(device, blocks_per_plane)
HBFSIM_SET_HBF_U32(device, pages_per_block)
HBFSIM_SET_HBF_U64(device, page_size_bytes)
HBFSIM_SET_HBF_U64(device, oob_bytes_per_page)
HBFSIM_SET_HBF_U32(device, media_lanes_per_plane)

HBFSIM_SET_HBF_U32(device, page_buffer_banks_per_plane)
HBFSIM_SET_HBF_DOUBLE(device, t_read_page_ns)
HBFSIM_SET_HBF_DOUBLE(device, t_program_page_ns)
HBFSIM_SET_HBF_DOUBLE(device, t_erase_block_ns)
HBFSIM_SET_HBF_DOUBLE(device, ecc_decode_latency_ns)
HBFSIM_SET_HBF_DOUBLE(device, ecc_encode_latency_ns)
HBFSIM_SET_HBF_DOUBLE(device, ecc_decode_raw_bandwidth_GBps_per_die)
HBFSIM_SET_HBF_DOUBLE(device, ecc_encode_raw_bandwidth_GBps_per_die)
HBFSIM_SET_HBF_DOUBLE(device, channel_bandwidth_GBps)

HBFSIM_SET_HBF_DOUBLE(device, tsv_bandwidth_GBps)
HBFSIM_SET_HBF_DOUBLE(device, media_lane_bandwidth_GBps)
HBFSIM_SET_HBF_DOUBLE(device, logic_sram_bandwidth_GBps)
HBFSIM_SET_HBF_DOUBLE(device, page_buffer_bandwidth_GBps)
HBFSIM_SET_HBF_U64(device, command_address_bytes)
HBFSIM_SET_HBF_DOUBLE(device, logic_scheduler_issue_ns)
HBFSIM_SET_HBF_DOUBLE(device, address_generation_ns)
HBFSIM_SET_HBF_DOUBLE(host, ctrl_dram_latency_ns)
HBFSIM_SET_HBF_DOUBLE(host, ctrl_dram_issue_ns)
HBFSIM_SET_HBF_DOUBLE(host, mapping_update_ns)
HBFSIM_SET_HBF_DOUBLE(host, mapping_control_compute_ns)
HBFSIM_SET_HBF_DOUBLE(host, free_page_allocation_ns)
HBFSIM_SET_HBF_DOUBLE(device, flash_tsu_issue_ns)
HBFSIM_SET_HBF_U64(host, ctrl_dram_bytes)
HBFSIM_SET_HBF_U64(host, mapping_entries_per_page)


HBFSIM_SET_HBF_U64(device, page_read_queue_depth_per_stack)
HBFSIM_SET_HBF_BOOL(host, write_coalescing_enabled)
HBFSIM_SET_HBF_BOOL(host, write_buffer_completion_requires_flush)
HBFSIM_SET_HBF_U64(host, write_buffer_pages)
HBFSIM_SET_HBF_U64(host, write_buffer_flush_threshold_pages)
HBFSIM_SET_HBF_BOOL(host, auto_gc_enabled)
HBFSIM_SET_HBF_U64(host, gc_low_watermark_pages)
HBFSIM_SET_HBF_U64(host, gc_hard_watermark_pages)
HBFSIM_SET_HBF_U64(host, gc_reserved_free_blocks_per_plane)
HBFSIM_SET_HBF_DOUBLE(host, gc_wear_leveling_weight)

#undef HBFSIM_SET_HBF_U64
#undef HBFSIM_SET_HBF_U32
#undef HBFSIM_SET_HBF_DOUBLE
#undef HBFSIM_SET_HBF_BOOL

    throw std::runtime_error("unknown HBF validation config key: " + key);
}

void apply_hybrid_config(
    Options& options,
    const std::string& assignment) {
    if (assignment.starts_with("hbm.")) {
        apply_hbm_config(options.hbm, assignment.substr(4));
        return;
    }
    if (assignment.starts_with("hbf.")) {
        apply_hbf_config(options.hbf, assignment.substr(4));
        return;
    }
    const auto separator = assignment.find('=');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == assignment.size()) {
        throw std::runtime_error(
            "--config expects key=value, got: " + assignment);
    }
    const auto key = assignment.substr(0, separator);
    const auto value = assignment.substr(separator + 1);
    if (key == "policy.kind") {
        if (value != "flat") {
            throw std::runtime_error(
                "hybrid validation requires policy.kind=flat");
        }
        return;
    }
    if (key == "policy.read_boundary") {
        options.hybrid_read_boundary =
            parse_unsigned<std::uint64_t>(value, "hybrid read boundary");
        return;
    }
    if (key == "knobs.max_outstanding_requests") {
        options.hybrid_max_outstanding_requests =
            parse_unsigned<std::size_t>(
                value, "hybrid outstanding request limit");
        return;
    }
    throw std::runtime_error(
        "unknown hybrid validation config key: " + key);
}

void apply_external_config(
    ExternalBackingConfig& config,
    const std::string& assignment) {
    const auto separator = assignment.find('=');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == assignment.size()) {
        throw std::runtime_error(
            "--config expects key=value, got: " + assignment);
    }
    const auto key = assignment.substr(0, separator);
    const auto value = assignment.substr(separator + 1);
    if (key == "kind") {
        config.kind =
            hbfsim::physical::external::parse_external_backing_kind(value);
        return;
    }
#define HBFSIM_SET_EXTERNAL_U64(field)                                         \
    if (key == #field) {                                                       \
        config.field = parse_unsigned<std::uint64_t>(value, #field);            \
        return;                                                                 \
    }
#define HBFSIM_SET_EXTERNAL_U32(field)                                         \
    if (key == #field) {                                                       \
        config.field = parse_unsigned<std::uint32_t>(value, #field);            \
        return;                                                                 \
    }
#define HBFSIM_SET_EXTERNAL_DOUBLE(field)                                      \
    if (key == #field) {                                                       \
        config.field = parse_double(value, #field);                             \
        return;                                                                 \
    }
    HBFSIM_SET_EXTERNAL_U64(capacity_bytes)
    HBFSIM_SET_EXTERNAL_U64(page_size_bytes)
    HBFSIM_SET_EXTERNAL_U64(request_segment_bytes)
    HBFSIM_SET_EXTERNAL_U32(media_channels)
    HBFSIM_SET_EXTERNAL_U32(media_read_queues)
    HBFSIM_SET_EXTERNAL_U32(media_write_queues)
    HBFSIM_SET_EXTERNAL_U32(max_outstanding_requests)
    HBFSIM_SET_EXTERNAL_DOUBLE(controller_issue_ns)
    HBFSIM_SET_EXTERNAL_DOUBLE(controller_processing_ns)
    HBFSIM_SET_EXTERNAL_DOUBLE(media_read_latency_ns)
    HBFSIM_SET_EXTERNAL_DOUBLE(media_write_latency_ns)
    HBFSIM_SET_EXTERNAL_DOUBLE(media_read_bandwidth_GBps)
    HBFSIM_SET_EXTERNAL_DOUBLE(media_write_bandwidth_GBps)
    HBFSIM_SET_EXTERNAL_DOUBLE(m2s_bandwidth_GBps)
    HBFSIM_SET_EXTERNAL_DOUBLE(s2m_bandwidth_GBps)
    HBFSIM_SET_EXTERNAL_DOUBLE(one_way_propagation_ns)
    HBFSIM_SET_EXTERNAL_U32(command_bytes)
    HBFSIM_SET_EXTERNAL_U32(completion_bytes)
#undef HBFSIM_SET_EXTERNAL_DOUBLE
#undef HBFSIM_SET_EXTERNAL_U32
#undef HBFSIM_SET_EXTERNAL_U64
    throw std::runtime_error(
        "unknown external validation config key: " + key);
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto need = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(
                    std::string(option) + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--case-id") {
            options.case_id = need("--case-id");
        } else if (arg == "--model") {
            options.model = need("--model");
        } else if (arg == "--config") {
            const auto assignment = need("--config");
            if (options.model == "hbm") {
                apply_hbm_config(options.hbm, assignment);
            } else if (options.model == "hbf") {
                if (assignment.starts_with("hbm.")) {
                    apply_hbm_config(options.hbm, assignment.substr(4));
                } else {
                    apply_hbf_config(options.hbf, assignment);
                }
            } else if (options.model == "hybrid") {
                apply_hybrid_config(options, assignment);
            } else if (options.model == "external") {
                apply_external_config(options.external, assignment);
            } else {
                throw std::runtime_error(
                    "--model must precede --config and be hbm, hbf, "
                    "hybrid, or external");
            }
        } else if (arg == "--inspect-address") {
            options.inspect_addresses.push_back(
                parse_unsigned<std::uint64_t>(
                    need("--inspect-address"),
                    "inspection address"));
        } else if (arg == "--prepopulate-lpn") {
            options.prepopulate_lpns.push_back(
                parse_unsigned<std::uint64_t>(
                    need("--prepopulate-lpn"),
                    "prepopulation LPN"));
        } else if (arg == "--request") {
            if (i + 6 >= argc) {
                throw std::runtime_error(
                    "--request expects: id arrival_ns op "
                    "address_space addr bytes");
            }
            RequestSpec request;
            request.id = argv[++i];
            request.arrival_ns = parse_double(argv[++i], "request arrival");
            const std::string action = argv[++i];
            request.drain = action == "drain";
            if (!request.drain) {
                request.op = parse_op(action);
            }
            request.address_space_name = argv[++i];
            if (request.drain) {
                if (request.address_space_name != "internal") {
                    throw std::runtime_error(
                        "validation drain address space must be internal");
                }
            } else {
                request.address_space =
                    parse_address_space(request.address_space_name);
            }
            request.addr =
                parse_unsigned<std::uint64_t>(argv[++i], "request address");
            request.bytes =
                parse_unsigned<std::uint64_t>(argv[++i], "request bytes");
            if (request.drain && (request.addr != 0 || request.bytes != 0)) {
                throw std::runtime_error(
                    "validation drain requires address=0 and bytes=0");
            }
            options.requests.push_back(std::move(request));
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (options.case_id.empty()) {
        throw std::runtime_error("--case-id is required");
    }
    if (
        options.model != "hbm"
        && options.model != "hbf"
        && options.model != "hybrid"
        && options.model != "external"
    ) {
        throw std::runtime_error(
            "ledger_probe supports --model hbm, hbf, hybrid, or external");
    }
    return options;
}

class LedgerWriter {
public:
    explicit LedgerWriter(std::string case_id)
        : case_id_(std::move(case_id)) {}

    template <typename Emit>
    void record(
        std::string_view kind,
        std::string_view id,
        const std::optional<std::string>& parent_id,
        Emit emit_fields) {
        std::cout << "{\"schema\":{\"name\":\"hbfsim.verification.ledger\","
                     "\"version\":1},\"record_index\":"
                  << record_index_++
                  << ",\"kind\":\"" << json_escape(kind)
                  << "\",\"id\":\"" << json_escape(id)
                  << "\",\"parent_id\":";
        if (parent_id) {
            std::cout << "\"" << json_escape(*parent_id) << "\"";
        } else {
            std::cout << "null";
        }
        std::cout << ",\"case_id\":\"" << json_escape(case_id_) << "\"";
        emit_fields();
        std::cout << "}\n";
    }

private:
    std::string case_id_;
    std::uint64_t record_index_ = 0;
};

class Sha256 {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        if (size > std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
            throw std::runtime_error("validation state is too large to hash");
        }
        total_bytes_ += static_cast<std::uint64_t>(size);
        while (size != 0) {
            const auto take = std::min(size, buffer_.size() - buffered_);
            std::copy_n(
                data,
                take,
                buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_));
            buffered_ += take;
            data += take;
            size -= take;
            if (buffered_ == buffer_.size()) {
                transform(buffer_.data());
                buffered_ = 0;
            }
        }
    }

    [[nodiscard]] std::string finish_hex() {
        if (total_bytes_ > std::numeric_limits<std::uint64_t>::max() / 8) {
            throw std::runtime_error(
                "validation state is too large for SHA-256 length encoding");
        }
        const auto bit_length = total_bytes_ * 8;
        buffer_[buffered_++] = 0x80;
        if (buffered_ > 56) {
            std::fill(
                buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
                buffer_.end(),
                0);
            transform(buffer_.data());
            buffered_ = 0;
        }
        std::fill(
            buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
            buffer_.begin() + 56,
            0);
        for (std::size_t i = 0; i < 8; ++i) {
            buffer_[63 - i] =
                static_cast<std::uint8_t>(bit_length >> (i * 8));
        }
        transform(buffer_.data());

        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto word : state_) {
            out << std::setw(8) << word;
        }
        return out.str();
    }

private:
    static constexpr std::array<std::uint32_t, 64> round_constants_ = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    static std::uint32_t rotate_right(
        std::uint32_t value,
        unsigned amount) {
        return (value >> amount) | (value << (32 - amount));
    }

    void transform(const std::uint8_t* block) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            const auto offset = i * 4;
            words[i] =
                (static_cast<std::uint32_t>(block[offset]) << 24) |
                (static_cast<std::uint32_t>(block[offset + 1]) << 16) |
                (static_cast<std::uint32_t>(block[offset + 2]) << 8) |
                static_cast<std::uint32_t>(block[offset + 3]);
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            const auto s0 =
                rotate_right(words[i - 15], 7) ^
                rotate_right(words[i - 15], 18) ^
                (words[i - 15] >> 3);
            const auto s1 =
                rotate_right(words[i - 2], 17) ^
                rotate_right(words[i - 2], 19) ^
                (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t i = 0; i < words.size(); ++i) {
            const auto s1 =
                rotate_right(e, 6) ^
                rotate_right(e, 11) ^
                rotate_right(e, 25);
            const auto choose = (e & f) ^ ((~e) & g);
            const auto temp1 =
                h + s1 + choose + round_constants_[i] + words[i];
            const auto s0 =
                rotate_right(a, 2) ^
                rotate_right(a, 13) ^
                rotate_right(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_ = {
        0x6a09e667u,
        0xbb67ae85u,
        0x3c6ef372u,
        0xa54ff53au,
        0x510e527fu,
        0x9b05688cu,
        0x1f83d9abu,
        0x5be0cd19u,
    };
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffered_ = 0;
    std::uint64_t total_bytes_ = 0;
};

[[nodiscard]] std::string sha256_text(std::string_view value) {
    Sha256 hash;
    hash.update(
        reinterpret_cast<const std::uint8_t*>(value.data()),
        value.size());
    return hash.finish_hex();
}

void append_audit_vector(
    std::ostringstream& out,
    std::string_view name,
    const std::vector<std::uint64_t>& values) {
    out << name;
    for (const auto value : values) {
        out << '\t' << value;
    }
    out << '\n';
}

[[nodiscard]] std::string canonical_audit_state(
    const HbfAuditSnapshot& snapshot) {
    std::ostringstream out;
    out << "hbfsim.hbf.audit.v1\n"
        << "quiescent\t" << (snapshot.quiescent() ? 1 : 0) << '\n'
        << "free_pages\t" << snapshot.free_pages << '\n';
    append_audit_vector(
        out, "free_pages_per_stack", snapshot.free_pages_per_stack);
    append_audit_vector(
        out, "data_allocation_cursors", snapshot.data_allocation_cursors);
    append_audit_vector(
        out,
        "mapping_allocation_cursors",
        snapshot.mapping_allocation_cursors);
    append_audit_vector(
        out, "gc_allocation_cursors", snapshot.gc_allocation_cursors);
    append_audit_vector(
        out, "dirty_mapping_vpns", snapshot.dirty_mapping_vpns);
    out << "pending"
        << '\t' << snapshot.pending_dirty_mapping_events
        << '\t' << snapshot.pending_lpn_updates
        << '\t' << snapshot.pending_vpn_updates
        << '\t' << snapshot.pending_commits
        << '\t' << snapshot.write_buffer_entries
        << '\t' << snapshot.inflight_buffered_generations
        << '\t' << snapshot.pending_physical_programs
        << '\t' << snapshot.pending_block_transitions
        << '\n';
    for (const auto& mapping : snapshot.logical_mappings) {
        out << "l2p\t" << mapping.key << '\t' << mapping.ppn << '\n';
    }
    for (const auto& mapping : snapshot.mapping_pages) {
        out << "vpn\t" << mapping.key << '\t' << mapping.ppn << '\n';
    }
    for (const auto& page : snapshot.materialized_pages) {
        out << "page"
            << '\t' << page.ppn
            << '\t' << page.status
            << '\t' << page.owner
            << '\t' << page.logical_key
            << '\t' << page.block_epoch
            << '\n';
    }
    for (const auto& block : snapshot.blocks) {
        out << "block"
            << '\t' << block.block
            << '\t' << block.role
            << '\t' << block.valid_pages
            << '\t' << block.invalid_pages
            << '\t' << block.free_pages
            << '\t' << block.next_page
            << '\t' << block.erase_count
            << '\t' << block.pending_program_pages
            << '\t' << block.pending_mapping_publications
            << '\t' << block.epoch
            << '\t' << (block.erase_pending ? 1 : 0)
            << '\n';
    }
    return out.str();
}

void write_u64_array(
    std::ostream& out,
    const std::vector<std::uint64_t>& values) {
    out << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out << ",";
        }
        out << values[index];
    }
    out << "]";
}

void write_audit_attributes(
    std::ostream& out,
    const HbfAuditSnapshot& snapshot) {
    out << "{\"quiescent\":"
        << (snapshot.quiescent() ? "true" : "false")
        << ",\"free_pages\":" << snapshot.free_pages
        << ",\"free_pages_per_stack\":";
    write_u64_array(out, snapshot.free_pages_per_stack);
    out << ",\"allocation_cursors\":{\"data\":";
    write_u64_array(out, snapshot.data_allocation_cursors);
    out << ",\"mapping\":";
    write_u64_array(out, snapshot.mapping_allocation_cursors);
    out << ",\"gc\":";
    write_u64_array(out, snapshot.gc_allocation_cursors);
    out << "},\"logical_mappings\":[";
    for (std::size_t index = 0;
         index < snapshot.logical_mappings.size();
         ++index) {
        if (index != 0) {
            out << ",";
        }
        const auto& mapping = snapshot.logical_mappings[index];
        out << "{\"lpn\":" << mapping.key
            << ",\"ppn\":" << mapping.ppn << "}";
    }
    out << "],\"mapping_pages\":[";
    for (std::size_t index = 0;
         index < snapshot.mapping_pages.size();
         ++index) {
        if (index != 0) {
            out << ",";
        }
        const auto& mapping = snapshot.mapping_pages[index];
        out << "{\"vpn\":" << mapping.key
            << ",\"ppn\":" << mapping.ppn << "}";
    }
    out << "],\"pages\":[";
    for (std::size_t index = 0;
         index < snapshot.materialized_pages.size();
         ++index) {
        if (index != 0) {
            out << ",";
        }
        const auto& page = snapshot.materialized_pages[index];
        out << "{\"ppn\":" << page.ppn
            << ",\"status\":\"" << json_escape(page.status)
            << "\",\"owner\":\"" << json_escape(page.owner)
            << "\",\"logical_key\":" << page.logical_key
            << ",\"block_epoch\":" << page.block_epoch
            << "}";
    }
    out << "],\"blocks\":[";
    for (std::size_t index = 0; index < snapshot.blocks.size(); ++index) {
        if (index != 0) {
            out << ",";
        }
        const auto& block = snapshot.blocks[index];
        out << "{\"block\":" << block.block
            << ",\"role\":\"" << json_escape(block.role)
            << "\",\"valid_pages\":" << block.valid_pages
            << ",\"invalid_pages\":" << block.invalid_pages
            << ",\"free_pages\":" << block.free_pages
            << ",\"next_page\":" << block.next_page
            << ",\"erase_count\":" << block.erase_count
            << ",\"pending_program_pages\":"
            << block.pending_program_pages
            << ",\"pending_mapping_publications\":"
            << block.pending_mapping_publications
            << ",\"epoch\":" << block.epoch
            << ",\"erase_pending\":"
            << (block.erase_pending ? "true" : "false")
            << "}";
    }
    out << "],\"dirty_mapping_vpns\":";
    write_u64_array(out, snapshot.dirty_mapping_vpns);
    out << ",\"pending\":{"
        << "\"dirty_mapping_events\":"
        << snapshot.pending_dirty_mapping_events
        << ",\"lpn_updates\":" << snapshot.pending_lpn_updates
        << ",\"vpn_updates\":" << snapshot.pending_vpn_updates
        << ",\"commits\":" << snapshot.pending_commits
        << ",\"write_buffer_entries\":" << snapshot.write_buffer_entries
        << ",\"inflight_buffered_generations\":"
        << snapshot.inflight_buffered_generations
        << ",\"physical_programs\":" << snapshot.pending_physical_programs
        << ",\"block_transitions\":" << snapshot.pending_block_transitions
        << "}}";
}

void write_hbm_config(std::ostream& out, const HbmConfig& config) {
    out << std::setprecision(17);
    out << "{"
        << "\"capacity_bytes\":" << config.device.capacity_bytes
        << ",\"stacks\":" << config.device.stacks
        << ",\"channels_per_stack\":" << config.device.channels_per_stack
        << ",\"pseudo_channels_per_channel\":"
        << config.device.pseudo_channels_per_channel
        << ",\"bank_groups_per_pseudo_channel\":"
        << config.device.bank_groups_per_pseudo_channel
        << ",\"banks_per_group\":" << config.device.banks_per_group
        << ",\"channel_row_size_bytes\":" << config.device.channel_row_size_bytes
        << ",\"channel_width_bits\":" << config.device.channel_width_bits
        << ",\"burst_length\":" << config.device.burst_length
        << ",\"pin_rate_Gbps\":" << config.device.pin_rate_Gbps
        << ",\"data_rate_per_command_clock\":"
        << config.device.data_rate_per_command_clock
        << ",\"address_mapping_ns\":" << config.controller.address_mapping_ns
        << ",\"read_latency_ns\":" << config.timing.read_latency_ns
        << ",\"write_latency_ns\":" << config.timing.write_latency_ns
        << ",\"read_to_write_ns\":" << config.timing.read_to_write_ns
        << ",\"write_to_read_ns\":" << config.timing.write_to_read_ns
        << ",\"bandwidth_efficiency\":" << config.timing.bandwidth_efficiency
        << ",\"queue_depth\":" << config.controller.queue_depth
        << ",\"interleave_bytes\":" << config.effective_interleave_bytes()
        << ",\"service_quantum_bytes\":" << config.controller.service_quantum_bytes
        << ",\"service_group_channels\":" << config.controller.service_group_channels
        << ",\"timing_model\":\"" << config.timing_model() << "\""
        << ",\"derived\":{"
        << "\"pseudo_channel_width_bits\":"
        << config.pseudo_channel_width_bits()
        << ",\"row_size_bytes\":" << config.row_size_bytes()
        << ",\"burst_bytes\":" << config.burst_bytes()
        << ",\"channel_bandwidth_GBps\":"
        << config.channel_bandwidth_GBps()
        << ",\"pseudo_channel_bandwidth_GBps\":"
        << config.pseudo_channel_bandwidth_GBps()
        << ",\"command_clock_period_ns\":"
        << config.command_clock_period_ns()
        << ",\"burst_duration_ns\":" << config.burst_duration_ns()
        << ",\"stripe_bytes\":" << config.effective_interleave_bytes() *
            config.device.stacks * config.device.channels_per_stack *
            config.device.pseudo_channels_per_channel
        << "}}";
}

void write_hbf_config(std::ostream& out, const HbfConfig& config) {
    const auto resident =
        hbfsim::host::derive_resident_mapping_capacity(config);
    const auto total_channels =
        static_cast<std::uint64_t>(config.device.stacks) *
        config.device.channels_per_stack;
    const auto total_dies = total_channels * config.device.dies_per_channel;
    const auto total_planes = total_dies * config.device.planes_per_die;
    const auto total_blocks = total_planes * config.device.blocks_per_plane;
    const auto total_pages = total_blocks * config.device.pages_per_block;
    const auto effective_subarrays =
        1u;
    out << std::setprecision(17);
    out << "{"
        << "\"speed_grade\":" << config.device.speed_grade
        << ",\"host_gc_decision_ns\":" << config.host.host_gc_decision_ns
        << ",\"stacks\":" << config.device.stacks
        << ",\"channels_per_stack\":" << config.device.channels_per_stack
        << ",\"dies_per_channel\":" << config.device.dies_per_channel
        << ",\"planes_per_die\":" << config.device.planes_per_die
        << ",\"blocks_per_plane\":" << config.device.blocks_per_plane
        << ",\"pages_per_block\":" << config.device.pages_per_block
        << ",\"page_size_bytes\":" << config.device.page_size_bytes
        << ",\"oob_bytes_per_page\":" << config.device.oob_bytes_per_page
        << ",\"media_lanes_per_plane\":"
        << config.device.media_lanes_per_plane
        << ",\"page_buffer_banks_per_plane\":"
        << config.device.page_buffer_banks_per_plane
        << ",\"t_read_page_ns\":" << config.device.t_read_page_ns
        << ",\"t_program_page_ns\":" << config.device.t_program_page_ns
        << ",\"t_erase_block_ns\":" << config.device.t_erase_block_ns
        << ",\"ecc_decode_latency_ns\":" << config.device.ecc_decode_latency_ns
        << ",\"ecc_encode_latency_ns\":" << config.device.ecc_encode_latency_ns
        << ",\"ecc_decode_raw_bandwidth_GBps_per_die\":"
        << config.device.ecc_decode_raw_bandwidth_GBps_per_die
        << ",\"ecc_encode_raw_bandwidth_GBps_per_die\":"
        << config.device.ecc_encode_raw_bandwidth_GBps_per_die
        << ",\"channel_bandwidth_GBps\":"
        << config.device.channel_bandwidth_GBps
        << ",\"hb_io_bandwidth_GBps\":" << config.device.hb_io_bandwidth_GBps()
        << ",\"tsv_bandwidth_GBps\":" << config.device.tsv_bandwidth_GBps
        << ",\"media_lane_bandwidth_GBps\":"
        << config.device.media_lane_bandwidth_GBps
        << ",\"logic_sram_bandwidth_GBps\":"
        << config.device.logic_sram_bandwidth_GBps
        << ",\"page_buffer_bandwidth_GBps\":"
        << config.device.page_buffer_bandwidth_GBps
        << ",\"command_address_bytes\":" << config.device.command_address_bytes
        << ",\"logic_scheduler_issue_ns\":"
        << config.device.logic_scheduler_issue_ns
        << ",\"address_generation_ns\":" << config.device.address_generation_ns
        << ",\"ctrl_dram_latency_ns\":" << config.host.ctrl_dram_latency_ns
        << ",\"ctrl_dram_issue_ns\":" << config.host.ctrl_dram_issue_ns
        << ",\"mapping_update_ns\":" << config.host.mapping_update_ns
        << ",\"mapping_control_compute_ns\":"
        << config.host.mapping_control_compute_ns
        << ",\"free_page_allocation_ns\":"
        << config.host.free_page_allocation_ns
        << ",\"flash_tsu_issue_ns\":" << config.device.flash_tsu_issue_ns
        << ",\"ctrl_dram_bytes\":" << config.host.ctrl_dram_bytes
        << ",\"mapping_entries_per_page\":"
        << config.host.mapping_entries_per_page
        << ",\"page_read_queue_depth_per_stack\":"
        << config.device.page_read_queue_depth_per_stack
        << ",\"write_coalescing_enabled\":"
        << (config.host.write_coalescing_enabled ? "true" : "false")
        << ",\"write_buffer_completion_requires_flush\":"
        << (config.host.write_buffer_completion_requires_flush ? "true" : "false")
        << ",\"write_buffer_pages\":" << config.host.write_buffer_pages
        << ",\"write_buffer_flush_threshold_pages\":"
        << config.host.write_buffer_flush_threshold_pages
        << ",\"auto_gc_enabled\":"
        << (config.host.auto_gc_enabled ? "true" : "false")
        << ",\"gc_low_watermark_pages\":"
        << config.host.gc_low_watermark_pages
        << ",\"gc_hard_watermark_pages\":"
        << config.host.gc_hard_watermark_pages
        << ",\"gc_reserved_free_blocks_per_plane\":"
        << config.host.gc_reserved_free_blocks_per_plane
        << ",\"gc_wear_leveling_weight\":"
        << config.host.gc_wear_leveling_weight
        << ",\"derived\":{"
        << "\"effective_subarrays_per_plane\":" << effective_subarrays
        << ",\"page_wire_bytes\":"
        << config.device.page_size_bytes + config.device.oob_bytes_per_page
        << ",\"total_channels\":" << total_channels
        << ",\"total_dies\":" << total_dies
        << ",\"total_planes\":" << total_planes
        << ",\"total_pages\":" << total_pages
        << ",\"capacity_bytes\":" << total_pages * config.device.page_size_bytes
        << ",\"resident_mapping_pages_per_stack\":"
        << resident.pages_per_stack
        << ",\"resident_mapping_bytes_per_stack\":"
        << resident.bytes_per_stack
        << ",\"resident_mapping_total_bytes\":" << resident.total_bytes
        << "}}";
}

void write_external_config(
    std::ostream& out,
    const ExternalBackingConfig& config) {
    out << std::setprecision(17)
        << "{\"kind\":\""
        << hbfsim::physical::external::to_string(config.kind)
        << "\",\"capacity_bytes\":" << config.capacity_bytes
        << ",\"page_size_bytes\":" << config.page_size_bytes
        << ",\"request_segment_bytes\":"
        << config.request_segment_bytes
        << ",\"media_channels\":" << config.media_channels
        << ",\"media_read_queues\":" << config.media_read_queues
        << ",\"media_write_queues\":" << config.media_write_queues
        << ",\"max_outstanding_requests\":"
        << config.max_outstanding_requests
        << ",\"controller_issue_ns\":" << config.controller_issue_ns
        << ",\"controller_processing_ns\":"
        << config.controller_processing_ns
        << ",\"media_read_latency_ns\":"
        << config.media_read_latency_ns
        << ",\"media_write_latency_ns\":"
        << config.media_write_latency_ns
        << ",\"media_read_bandwidth_GBps\":"
        << config.media_read_bandwidth_GBps
        << ",\"media_write_bandwidth_GBps\":"
        << config.media_write_bandwidth_GBps
        << ",\"m2s_bandwidth_GBps\":" << config.m2s_bandwidth_GBps
        << ",\"s2m_bandwidth_GBps\":" << config.s2m_bandwidth_GBps
        << ",\"one_way_propagation_ns\":"
        << config.one_way_propagation_ns
        << ",\"command_bytes\":" << config.command_bytes
        << ",\"completion_bytes\":" << config.completion_bytes
        << "}";
}

[[nodiscard]] bool canonical_span(const TraceSpan& span) {
    // Queue spans are derived explanations of gaps between authoritative
    // reservations. The canonical ledger compares typed work reservations and
    // completions; wait localization is reconstructed from adjacent events.
    return span.category != "queue";
}

[[nodiscard]] std::vector<TraceSpan> canonical_spans(
    const PhysicalCompletion& completion) {
    std::vector<TraceSpan> spans;
    for (const auto& span : completion.spans) {
        if (canonical_span(span)) {
            spans.push_back(span);
        }
    }
    std::stable_sort(
        spans.begin(),
        spans.end(),
        [](const TraceSpan& lhs, const TraceSpan& rhs) {
            return std::tie(
                       lhs.start_ns,
                       lhs.end_ns,
                       lhs.category,
                       lhs.name,
                       lhs.entity) <
                std::tie(
                       rhs.start_ns,
                       rhs.end_ns,
                       rhs.category,
                       rhs.name,
                       rhs.entity);
        });
    return spans;
}

void run_hbm(const Options& options) {
    HbmDevice device(options.hbm);
    LedgerWriter ledger(options.case_id);
    ledger.record("header", "ledger", std::nullopt, [&]() {
        std::cout << ",\"model\":\"hbm\",\"producer\":\"production\"";
    });

    std::vector<std::tuple<
        std::uint64_t,
        hbfsim::physical::hbm::HbmAddress,
        std::uint64_t>> mappings;
    mappings.reserve(options.inspect_addresses.size());
    for (const auto address : options.inspect_addresses) {
        const auto decoded = device.decode(address);
        mappings.emplace_back(address, decoded, device.encode(decoded));
    }

    std::vector<PhysicalCompletion> completions;
    completions.reserve(options.requests.size());
    for (const auto& spec : options.requests) {
        if (spec.drain) {
            throw std::runtime_error(
                "HBM validation does not support drain actions");
        }
        const std::string request_record_id = "request/" + spec.id;
        ledger.record("request", request_record_id, std::nullopt, [&]() {
            std::cout << std::setprecision(17)
                      << ",\"request_id\":\"" << json_escape(spec.id)
                      << "\",\"model\":\"hbm\",\"action\":\""
                      << hbfsim::physical::to_string(spec.op)
                      << "\",\"arrival_ns\":" << spec.arrival_ns
                      << ",\"address_space\":\""
                      << json_escape(spec.address_space_name)
                      << "\",\"addr\":" << spec.addr
                      << ",\"bytes\":" << spec.bytes;
        });

        PhysicalRequest request{
            .id = spec.id,
            .tier = Tier::HBM,
            .op = spec.op,
            .address_space = spec.address_space,
            .trace = {.mode = TraceMode::Full},
            .arrival_ns = spec.arrival_ns,
            .addr = spec.addr,
            .bytes = spec.bytes,
        };
        auto completion = device.issue(request);
        const auto spans = canonical_spans(completion);
        for (std::size_t i = 0; i < spans.size(); ++i) {
            const auto& span = spans[i];
            std::ostringstream event_id;
            event_id << request_record_id << "/event/"
                     << std::setw(4) << std::setfill('0') << i;
            ledger.record(
                "event",
                event_id.str(),
                request_record_id,
                [&]() {
                    std::cout << std::setprecision(17)
                              << ",\"request_id\":\""
                              << json_escape(spec.id)
                              << "\",\"model\":\"hbm\",\"resource\":\""
                              << json_escape(span.entity)
                              << "\",\"category\":\""
                              << json_escape(span.category)
                              << "\",\"action\":\""
                              << json_escape(span.name)
                              << "\",\"start_ns\":" << span.start_ns
                              << ",\"finish_ns\":" << span.end_ns
                              << ",\"critical\":"
                              << (span.critical ? "true" : "false")
                              << ",\"diagnostic_detail\":\""
                              << json_escape(span.detail) << "\"";
                    if (span.physical_bytes) std::cout << ",\"physical_bytes\":" << span.physical_bytes;
                });
        }

        ledger.record(
            "completion",
            request_record_id + "/completion",
            request_record_id,
            [&]() {
                std::cout << std::setprecision(17)
                          << ",\"request_id\":\"" << json_escape(spec.id)
                          << "\",\"model\":\"hbm\",\"action\":\"complete\""
                          << ",\"arrival_ns\":" << completion.arrival_ns
                          << ",\"start_ns\":" << completion.start_ns
                          << ",\"finish_ns\":" << completion.finish_ns
                          << ",\"logical_bytes\":" << completion.logical_bytes
                          << ",\"physical_bytes\":" << completion.physical_bytes
                          << ",\"resource\":\""
                          << json_escape(completion.resource_path)
                          << "\",\"result\":\""
                          << json_escape(completion.note) << "\"";
            });
        completions.push_back(std::move(completion));
    }

    const auto& stats = device.stats();
    ledger.record("summary", "summary", std::nullopt, [&]() {
        std::cout << std::setprecision(17)
                  << ",\"model\":\"hbm\",\"action\":\"final\""
                  << ",\"config\":";
        write_hbm_config(std::cout, device.config());
        std::cout << ",\"address_observations\":[";
        for (std::size_t i = 0; i < mappings.size(); ++i) {
            if (i != 0) {
                std::cout << ",";
            }
            const auto& [address, decoded, encoded] = mappings[i];
            std::cout << "{\"address\":" << address
                      << ",\"stack\":" << decoded.stack
                      << ",\"channel\":" << decoded.channel
                      << ",\"pseudo_channel\":" << decoded.pseudo_channel
                      << ",\"bank_group\":" << decoded.bank_group
                      << ",\"bank\":" << decoded.bank
                      << ",\"row\":" << decoded.row
                      << ",\"offset\":" << decoded.offset
                      << ",\"round_trip_address\":" << encoded << "}";
        }
        std::cout << "]"
                  << ",\"counters\":{"
                  << "\"read_bytes\":" << stats.read_bytes
                  << ",\"write_bytes\":" << stats.write_bytes
                  << ",\"bus_busy_ns\":" << stats.bus_busy_ns
                  << ",\"finish_ns\":" << stats.finish_ns
                  << ",\"pseudo_channels\":" << stats.pseudo_channels
                  << ",\"active_pseudo_channels\":"
                  << stats.active_pseudo_channels
                  << ",\"max_pseudo_channel_accesses\":"
                  << stats.max_pseudo_channel_accesses
                  << ",\"max_queue_occupancy\":"
                  << stats.max_queue_occupancy
                  << "}";
    });
}

void run_external(const Options& options) {
    if (!options.prepopulate_lpns.empty()) {
        throw std::runtime_error(
            "external validation does not accept prepopulation inputs");
    }
    ExternalBackingDevice device(options.external);
    LedgerWriter ledger(options.case_id);
    ledger.record("header", "ledger", std::nullopt, [&]() {
        std::cout
            << ",\"model\":\"external\",\"producer\":\"production\"";
    });

    for (const auto& spec : options.requests) {
        if (spec.drain || spec.address_space != AddressSpace::Logical) {
            throw std::runtime_error(
                "external validation requires logical read/write requests");
        }
        const std::string request_record_id = "request/" + spec.id;
        ledger.record("request", request_record_id, std::nullopt, [&]() {
            std::cout << std::setprecision(17)
                      << ",\"request_id\":\"" << json_escape(spec.id)
                      << "\",\"model\":\"external\",\"action\":\""
                      << hbfsim::physical::to_string(spec.op)
                      << "\",\"arrival_ns\":" << spec.arrival_ns
                      << ",\"address_space\":\"logical\""
                      << ",\"addr\":" << spec.addr
                      << ",\"bytes\":" << spec.bytes;
        });

        PhysicalRequest request{
            .id = spec.id,
            .tier = Tier::External,
            .op = spec.op,
            .address_space = AddressSpace::Logical,
            .trace = {.mode = TraceMode::Full},
            .arrival_ns = spec.arrival_ns,
            .addr = spec.addr,
            .bytes = spec.bytes,
        };
        const auto bytes_remaining_in_page =
            options.external.page_size_bytes -
            request.addr % options.external.page_size_bytes;
        const auto completion = request.bytes > bytes_remaining_in_page ?
            device.issue_contiguous_range(request) : device.issue(request);
        const auto spans = canonical_spans(completion);
        for (std::size_t index = 0; index < spans.size(); ++index) {
            const auto& span = spans[index];
            std::ostringstream event_id;
            event_id << request_record_id << "/event/"
                     << std::setw(4) << std::setfill('0') << index;
            ledger.record(
                "event",
                event_id.str(),
                request_record_id,
                [&]() {
                    std::cout << std::setprecision(17)
                              << ",\"request_id\":\""
                              << json_escape(spec.id)
                              << "\",\"model\":\"external\",\"resource\":\""
                              << json_escape(span.entity)
                              << "\",\"category\":\""
                              << json_escape(span.category)
                              << "\",\"action\":\""
                              << json_escape(span.name)
                              << "\",\"start_ns\":" << span.start_ns
                              << ",\"finish_ns\":" << span.end_ns
                              << ",\"critical\":"
                              << (span.critical ? "true" : "false")
                              << ",\"diagnostic_detail\":\""
                              << json_escape(span.detail) << "\"";
                    if (span.physical_bytes) std::cout << ",\"physical_bytes\":" << span.physical_bytes;
                });
        }
        ledger.record(
            "completion",
            request_record_id + "/completion",
            request_record_id,
            [&]() {
                std::cout << std::setprecision(17)
                          << ",\"request_id\":\"" << json_escape(spec.id)
                          << "\",\"model\":\"external\""
                             ",\"action\":\"complete\""
                          << ",\"arrival_ns\":" << completion.arrival_ns
                          << ",\"start_ns\":" << completion.start_ns
                          << ",\"finish_ns\":" << completion.finish_ns
                          << ",\"logical_bytes\":"
                          << completion.logical_bytes
                          << ",\"physical_bytes\":"
                          << completion.physical_bytes
                          << ",\"resource\":\""
                          << json_escape(completion.resource_path)
                          << "\",\"result\":\""
                          << json_escape(completion.note) << "\"";
            });
    }

    const auto& config = device.config();
    const auto& stats = device.stats();
    ledger.record("summary", "summary", std::nullopt, [&]() {
        std::cout << std::setprecision(17)
                  << ",\"model\":\"external\",\"action\":\"final\""
                  << ",\"config\":";
        write_external_config(std::cout, config);
        std::cout << ",\"address_observations\":[";
        for (std::size_t index = 0;
             index < options.inspect_addresses.size();
             ++index) {
            if (index != 0) {
                std::cout << ",";
            }
            const auto address = options.inspect_addresses[index];
            const auto channel =
                (address / config.request_segment_bytes) %
                config.media_channels;
            std::cout << "{\"address\":" << address
                      << ",\"media_channel\":" << channel << "}";
        }
        std::cout
            << "],\"counters\":{"
            << "\"read_requests\":" << stats.read_requests
            << ",\"write_requests\":" << stats.write_requests
            << ",\"read_bytes\":" << stats.read_bytes
            << ",\"write_bytes\":" << stats.write_bytes
            << ",\"page_run_requests\":" << stats.page_run_requests
            << ",\"page_run_segments\":" << stats.page_run_segments
            << ",\"page_run_pages\":" << stats.page_run_pages
            << ",\"media_channels\":" << stats.media_channels
            << ",\"media_channels_per_queue\":"
            << stats.media_channels_per_queue
            << ",\"media_read_queues\":" << stats.media_read_queues
            << ",\"media_write_queues\":" << stats.media_write_queues
            << ",\"active_media_resources\":"
            << stats.active_media_resources
            << ",\"max_device_outstanding\":"
            << stats.max_device_outstanding
            << ",\"outstanding_wait_ns\":" << stats.outstanding_wait_ns
            << ",\"controller_queue_wait_ns\":"
            << stats.controller_queue_wait_ns
            << ",\"controller_issue_busy_ns\":"
            << stats.controller_issue_busy_ns
            << ",\"controller_processing_work_ns\":"
            << stats.controller_processing_work_ns
            << ",\"media_queue_wait_ns\":" << stats.media_queue_wait_ns
            << ",\"media_read_latency_work_ns\":"
            << stats.media_read_latency_work_ns
            << ",\"media_write_latency_work_ns\":"
            << stats.media_write_latency_work_ns
            << ",\"media_read_busy_ns\":" << stats.media_read_busy_ns
            << ",\"media_write_busy_ns\":" << stats.media_write_busy_ns
            << ",\"m2s_payload_bytes\":" << stats.m2s_payload_bytes
            << ",\"m2s_protocol_bytes\":" << stats.m2s_protocol_bytes
            << ",\"m2s_wire_bytes\":" << stats.m2s_wire_bytes
            << ",\"s2m_payload_bytes\":" << stats.s2m_payload_bytes
            << ",\"s2m_protocol_bytes\":" << stats.s2m_protocol_bytes
            << ",\"s2m_wire_bytes\":" << stats.s2m_wire_bytes
            << ",\"m2s_queue_wait_ns\":" << stats.m2s_queue_wait_ns
            << ",\"s2m_queue_wait_ns\":" << stats.s2m_queue_wait_ns
            << ",\"m2s_busy_ns\":" << stats.m2s_busy_ns
            << ",\"s2m_busy_ns\":" << stats.s2m_busy_ns
            << ",\"transport_propagation_work_ns\":"
            << stats.transport_propagation_work_ns
            << ",\"finish_ns\":" << stats.finish_ns
            << "}";
    });
}

void run_hbf(const Options& options) {
    HbmDevice buffer_hbm(options.hbm);
    HbfController device(options.hbf);
    device.attach_hbm_buffer(buffer_hbm);
    device.prepopulate_logical_pages(options.prepopulate_lpns);
    LedgerWriter ledger(options.case_id);
    ledger.record("header", "ledger", std::nullopt, [&]() {
        std::cout << ",\"model\":\"hbf\",\"producer\":\"production\"";
    });

    std::vector<std::tuple<
        std::uint64_t,
        hbfsim::host::HbfAddress,
        std::uint64_t>> mappings;
    mappings.reserve(options.inspect_addresses.size());
    for (const auto address : options.inspect_addresses) {
        const auto decoded = device.decode(address);
        mappings.emplace_back(address, decoded, device.encode(decoded));
    }

    for (const auto& spec : options.requests) {
        const std::string request_record_id = "request/" + spec.id;
        ledger.record("request", request_record_id, std::nullopt, [&]() {
            std::cout << std::setprecision(17)
                      << ",\"request_id\":\"" << json_escape(spec.id)
                      << "\",\"model\":\"hbf\",\"action\":\""
                      << (spec.drain ?
                             "drain" :
                             std::string(hbfsim::physical::to_string(spec.op)))
                      << "\",\"arrival_ns\":" << spec.arrival_ns
                      << ",\"address_space\":\""
                      << json_escape(spec.address_space_name)
                      << "\",\"addr\":" << spec.addr
                      << ",\"bytes\":" << spec.bytes;
        });

        PhysicalCompletion completion;
        std::optional<HbfAuditSnapshot> audit;
        if (spec.drain) {
            completion = device.drain_pending(
                spec.id,
                spec.arrival_ns,
                {.mode = TraceMode::Full});
            audit = device.audit_snapshot();
            if (!audit->quiescent()) {
                throw std::runtime_error(
                    "HBF drain returned with non-quiescent audit state");
            }
        } else {
            PhysicalRequest request{
                .id = spec.id,
                .tier = Tier::HBF,
                .op = spec.op,
                .address_space = spec.address_space,
                .trace = {.mode = TraceMode::Full},
                .arrival_ns = spec.arrival_ns,
                .addr = spec.addr,
                .bytes = spec.bytes,
            };
            completion = device.issue(request);
        }
        const auto spans = canonical_spans(completion);
        for (std::size_t i = 0; i < spans.size(); ++i) {
            const auto& span = spans[i];
            std::ostringstream event_id;
            event_id << request_record_id << "/event/"
                     << std::setw(4) << std::setfill('0') << i;
            ledger.record(
                "event",
                event_id.str(),
                request_record_id,
                [&]() {
                    std::cout << std::setprecision(17)
                              << ",\"request_id\":\""
                              << json_escape(spec.id)
                              << "\",\"model\":\"hbf\",\"resource\":\""
                              << json_escape(span.entity)
                              << "\",\"category\":\""
                              << json_escape(span.category)
                              << "\",\"action\":\""
                              << json_escape(span.name)
                              << "\",\"start_ns\":" << span.start_ns
                              << ",\"finish_ns\":" << span.end_ns
                              << ",\"critical\":"
                              << (span.critical ? "true" : "false")
                              << ",\"diagnostic_detail\":\""
                              << json_escape(span.detail) << "\"";
                    if (span.physical_bytes) std::cout << ",\"physical_bytes\":" << span.physical_bytes;
                });
        }

        if (audit) {
            const auto canonical_state = canonical_audit_state(*audit);
            const auto state_hash = sha256_text(canonical_state);
            ledger.record(
                "state",
                request_record_id + "/state/quiescent",
                request_record_id,
                [&]() {
                    std::cout << std::setprecision(17)
                              << ",\"request_id\":\""
                              << json_escape(spec.id)
                              << "\",\"model\":\"hbf\""
                              << ",\"resource\":\"hbf/ftl\""
                              << ",\"action\":\"quiescent_checkpoint\""
                              << ",\"time_ns\":" << completion.finish_ns
                              << ",\"attributes\":";
                    write_audit_attributes(std::cout, *audit);
                    std::cout << ",\"state_hash\":\"sha256:"
                              << state_hash << "\"";
                });
        }

        ledger.record(
            "completion",
            request_record_id + "/completion",
            request_record_id,
            [&]() {
                std::cout << std::setprecision(17)
                          << ",\"request_id\":\"" << json_escape(spec.id)
                          << "\",\"model\":\"hbf\",\"action\":\"complete\""
                          << ",\"arrival_ns\":" << completion.arrival_ns
                          << ",\"start_ns\":" << completion.start_ns
                          << ",\"finish_ns\":" << completion.finish_ns
                          << ",\"logical_bytes\":" << completion.logical_bytes
                          << ",\"physical_bytes\":" << completion.physical_bytes
                          << ",\"resource\":\""
                          << json_escape(completion.resource_path)
                          << "\",\"result\":\""
                          << json_escape(completion.note) << "\"";
            });
    }

    const auto& stats = device.stats();
    ledger.record("summary", "summary", std::nullopt, [&]() {
        std::cout << std::setprecision(17)
                  << ",\"model\":\"hbf\",\"action\":\"final\""
                  << ",\"config\":";
        write_hbf_config(std::cout, device.config());
        std::cout << ",\"address_observations\":[";
        for (std::size_t i = 0; i < mappings.size(); ++i) {
            if (i != 0) {
                std::cout << ",";
            }
            const auto& [address, decoded, encoded] = mappings[i];
            std::cout << "{\"address\":" << address
                      << ",\"stack\":" << decoded.stack
                      << ",\"channel\":" << decoded.channel
                      << ",\"die\":" << decoded.die
                      << ",\"plane\":" << decoded.plane
                      << ",\"block\":" << decoded.block
                      << ",\"page\":" << decoded.page
                      << ",\"offset\":" << decoded.offset
                      << ",\"round_trip_address\":" << encoded << "}";
        }
        std::cout << "]"
                  << ",\"counters\":{"
                  << "\"read_requests\":" << stats.read_requests
                  << ",\"program_requests\":" << stats.program_requests
                  << ",\"logical_read_bytes\":" << stats.logical_read_bytes
                  << ",\"logical_write_bytes\":" << stats.logical_write_bytes
                  << ",\"physical_read_bytes\":" << stats.physical_read_bytes
                  << ",\"physical_write_bytes\":" << stats.physical_write_bytes
                  << ",\"data_program_payload_bytes\":"
                  << stats.data_program_payload_bytes
                  << ",\"mapping_program_payload_bytes\":"
                  << stats.mapping_program_payload_bytes
                  << ",\"gc_relocation_payload_bytes\":"
                  << stats.gc_relocation_payload_bytes
                  << ",\"page_reads\":" << stats.page_reads
                  << ",\"data_programs\":" << stats.data_programs
                  << ",\"page_programs\":" << stats.page_programs
                  << ",\"mapping_page_programs\":"
                  << stats.mapping_page_programs
                  << ",\"invalidations\":" << stats.invalidations
                  << ",\"write_buffer_hits\":" << stats.write_buffer_hits
                  << ",\"write_buffer_misses\":" << stats.write_buffer_misses
                  << ",\"write_buffer_flushes\":" << stats.write_buffer_flushes
                  << ",\"write_buffer_merged_bytes\":"
                  << stats.write_buffer_merged_bytes
                  << ",\"write_buffer_slot_wait_ops\":"
                  << stats.write_buffer_slot_wait_ops
                  << ",\"write_buffer_slot_wait_ns\":"
                  << stats.write_buffer_slot_wait_ns
                  << ",\"block_erases\":" << stats.block_erases
                  << ",\"gc_runs\":" << stats.gc_runs
                  << ",\"gc_relocations\":" << stats.gc_relocations
                  << ",\"gc_data_relocations\":"
                  << stats.gc_data_relocations
                  << ",\"gc_mapping_relocations\":"
                  << stats.gc_mapping_relocations
                  << ",\"gc_reclaimed_invalid_pages\":"
                  << stats.gc_reclaimed_invalid_pages
                  << ",\"gc_user_blocked_runs\":"
                  << stats.gc_user_blocked_runs
                  << ",\"mapping_lookup_ops\":" << stats.mapping_lookup_ops
                  << ",\"mapping_user_lookup_ops\":"
                  << stats.mapping_user_lookup_ops
                  << ",\"mapping_gc_lookup_ops\":"
                  << stats.mapping_gc_lookup_ops
                  << ",\"mapping_update_ops\":" << stats.mapping_update_ops
                  << ",\"mapping_user_update_ops\":"
                  << stats.mapping_user_update_ops
                  << ",\"mapping_gc_update_ops\":"
                  << stats.mapping_gc_update_ops
                  << ",\"mapping_dram_issue_busy_ns\":"
                  << stats.mapping_dram_issue_busy_ns
                  << ",\"mapping_compute_work_ns\":"
                  << stats.mapping_compute_work_ns
                  << ",\"mapping_compute_ops\":" << stats.mapping_compute_ops
                  << ",\"page_read_admission_events\":"
                  << stats.page_read_admission_events
                  << ",\"flash_scheduler_enqueues\":"
                  << stats.flash_scheduler_enqueues
                  << ",\"flash_scheduler_issues\":"
                  << stats.flash_scheduler_issues
                  << ",\"ecc_decode_ops\":" << stats.ecc_decode_ops
                  << ",\"ecc_decode_codeword_bytes\":"
                  << stats.ecc_decode_codeword_bytes
                  << ",\"ecc_encode_ops\":" << stats.ecc_encode_ops
                  << ",\"ecc_encode_codeword_bytes\":"
                  << stats.ecc_encode_codeword_bytes
                  << ",\"ecc_issue_busy_ns\":" << stats.ecc_issue_busy_ns
                  << ",\"media_busy_ns\":" << stats.media_busy_ns
                  << ",\"channel_command_busy_ns\":"
                  << stats.channel_command_busy_ns
                  << ",\"channel_data_busy_ns\":"
                  << stats.channel_data_busy_ns
                  << ",\"tsv_busy_ns\":" << stats.tsv_busy_ns
                  << ",\"sram_busy_ns\":" << stats.sram_busy_ns
                  << ",\"hb_io_command_busy_ns\":"
                  << stats.hb_io_command_busy_ns
                  << ",\"hb_io_data_busy_ns\":"
                  << stats.hb_io_data_busy_ns
                  << ",\"finish_ns\":" << stats.finish_ns
                  << ",\"total_pages\":" << stats.total_pages
                  << ",\"free_pages\":" << stats.free_pages
                  << ",\"valid_pages\":" << stats.valid_pages
                  << ",\"invalid_pages\":" << stats.invalid_pages
                  << ",\"pending_program_pages\":"
                  << stats.pending_program_pages
                  << ",\"pending_mapping_publications\":"
                  << stats.pending_mapping_publications
                  << ",\"accounting_verified\":"
                  << (stats.accounting_verified ? "true" : "false")
                  << ",\"active_planes\":" << stats.active_planes
                  << ",\"active_media_lanes\":"
                  << stats.active_media_lanes
                  << ",\"active_subarrays\":" << stats.active_subarrays
                  << ",\"active_page_buffer_banks\":"
                  << stats.active_page_buffer_banks
                  << ",\"active_channels\":" << stats.active_channels
                  << ",\"active_dies\":" << stats.active_dies
                  << "}";
    });
}

void write_double_array(
    std::ostream& out,
    const std::vector<double>& values) {
    out << std::setprecision(17) << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            out << ",";
        }
        out << values[index];
    }
    out << "]";
}

[[nodiscard]] std::string hybrid_tier_model(Tier tier) {
    if (tier == Tier::HBM) {
        return "hbm";
    }
    if (tier == Tier::HBF) {
        return "hbf";
    }
    throw std::runtime_error(
        "hybrid validation encountered an unsupported completion tier");
}

[[nodiscard]] std::string hybrid_parent_resource(
    const std::vector<const PhysicalCompletion*>& children) {
    bool has_hbm = false;
    bool has_hbf = false;
    for (const auto* child : children) {
        has_hbm = has_hbm || child->tier == Tier::HBM;
        has_hbf = has_hbf || child->tier == Tier::HBF;
    }
    std::string tiers;
    if (has_hbf) {
        tiers = "hbf";
    }
    if (has_hbm) {
        if (!tiers.empty()) {
            tiers += "+";
        }
        tiers += "hbm";
    }
    return "hybrid/" + tiers + "/" +
        std::to_string(children.size()) + "-child";
}

void run_hybrid(const Options& options) {
    if (!options.inspect_addresses.empty() ||
        !options.prepopulate_lpns.empty()) {
        throw std::runtime_error(
            "hybrid validation does not accept inspection or explicit "
            "prepopulation inputs");
    }
    std::vector<MemoryRequest> requests;
    requests.reserve(options.requests.size());
    for (std::size_t index = 0; index < options.requests.size(); ++index) {
        const auto& spec = options.requests[index];
        if (spec.drain || spec.op != Op::Read ||
            spec.address_space != AddressSpace::Logical) {
            throw std::runtime_error(
                "hybrid validation v1 requires logical reads");
        }
        requests.push_back(MemoryRequest{
            .id = spec.id,
            .op = spec.op,
            .addr = spec.addr,
            .bytes = spec.bytes,
            .arrival_ns = spec.arrival_ns,
            .index = index,
        });
    }
    DirectRunKnobs knobs;
    knobs.max_outstanding_requests =
        options.hybrid_max_outstanding_requests;
    knobs.trace.mode = TraceMode::Full;
    knobs.retain_completions = true;
    const auto policy = flat_policy(options.hybrid_read_boundary);
    auto hbf_config = options.hbf;
    if (hbf_config.host.ctrl_dram_bytes == 0) {
        hbf_config.host.ctrl_dram_bytes =
            hbfsim::host::derive_resident_mapping_capacity(
                hbf_config).total_bytes;
    }
    const auto result = run_direct_policy(
        policy,
        options.hbm,
        hbf_config,
        requests,
        knobs);

    std::vector<std::vector<const PhysicalCompletion*>> children(
        requests.size());
    for (const auto& completion : result.completions) {
        bool assigned = false;
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const auto token = "/op" + std::to_string(index) + "/";
            if (completion.id.find(token) == std::string::npos) {
                continue;
            }
            children[index].push_back(&completion);
            assigned = true;
            break;
        }
        if (!assigned) {
            throw std::runtime_error(
                "hybrid validation retained an unowned completion: " +
                completion.id);
        }
    }

    LedgerWriter ledger(options.case_id);
    ledger.record("header", "ledger", std::nullopt, [&]() {
        std::cout << ",\"model\":\"hybrid\","
                     "\"producer\":\"production\"";
    });
    std::uint64_t physical_bytes = 0;
    for (std::size_t index = 0; index < options.requests.size(); ++index) {
        const auto& spec = options.requests[index];
        const auto& parent_children = children[index];
        if (parent_children.empty()) {
            throw std::runtime_error(
                "hybrid validation parent has no physical completion");
        }
        const std::string request_record_id = "request/" + spec.id;
        ledger.record("request", request_record_id, std::nullopt, [&]() {
            std::cout << std::setprecision(17)
                      << ",\"request_id\":\"" << json_escape(spec.id)
                      << "\",\"model\":\"hybrid\",\"action\":\"read\""
                      << ",\"arrival_ns\":" << spec.arrival_ns
                      << ",\"address_space\":\"logical\""
                      << ",\"addr\":" << spec.addr
                      << ",\"bytes\":" << spec.bytes;
        });

        struct TaggedSpan {
            std::string model;
            TraceSpan span;
        };
        std::vector<TaggedSpan> spans;
        for (const auto* child : parent_children) {
            for (auto span : canonical_spans(*child)) {
                spans.push_back(TaggedSpan{
                    .model = hybrid_tier_model(child->tier),
                    .span = std::move(span),
                });
            }
        }
        std::stable_sort(
            spans.begin(),
            spans.end(),
            [](const TaggedSpan& lhs, const TaggedSpan& rhs) {
                return std::tie(
                           lhs.span.start_ns,
                           lhs.span.end_ns,
                           lhs.span.category,
                           lhs.span.name,
                           lhs.span.entity,
                           lhs.model) <
                    std::tie(
                           rhs.span.start_ns,
                           rhs.span.end_ns,
                           rhs.span.category,
                           rhs.span.name,
                           rhs.span.entity,
                           rhs.model);
            });
        for (std::size_t span_index = 0;
             span_index < spans.size();
             ++span_index) {
            const auto& tagged = spans[span_index];
            std::ostringstream event_id;
            event_id << request_record_id << "/event/"
                     << std::setw(4) << std::setfill('0') << span_index;
            ledger.record(
                "event",
                event_id.str(),
                request_record_id,
                [&]() {
                    std::cout << std::setprecision(17)
                              << ",\"request_id\":\""
                              << json_escape(spec.id)
                              << "\",\"model\":\"" << tagged.model
                              << "\",\"resource\":\""
                              << json_escape(tagged.span.entity)
                              << "\",\"category\":\""
                              << json_escape(tagged.span.category)
                              << "\",\"action\":\""
                              << json_escape(tagged.span.name)
                              << "\",\"start_ns\":"
                              << tagged.span.start_ns
                              << ",\"finish_ns\":"
                              << tagged.span.end_ns
                              << ",\"critical\":"
                              << (tagged.span.critical ? "true" : "false")
                              << ",\"diagnostic_detail\":\""
                              << json_escape(tagged.span.detail) << "\"";
                    if (tagged.span.physical_bytes) std::cout << ",\"physical_bytes\":" << tagged.span.physical_bytes;
                });
        }

        double parent_start_ns = std::numeric_limits<double>::infinity();
        double parent_finish_ns = 0.0;
        std::uint64_t parent_physical_bytes = 0;
        bool has_hbm = false;
        bool has_hbf = false;
        for (const auto* child : parent_children) {
            parent_start_ns = std::min(
                parent_start_ns, child->start_ns);
            parent_finish_ns = std::max(
                parent_finish_ns, child->finish_ns);
            if (
                child->physical_bytes
                > std::numeric_limits<std::uint64_t>::max()
                    - parent_physical_bytes
            ) {
                throw std::runtime_error(
                    "hybrid validation physical-byte sum overflow");
            }
            parent_physical_bytes += child->physical_bytes;
            has_hbm = has_hbm || child->tier == Tier::HBM;
            has_hbf = has_hbf || child->tier == Tier::HBF;
        }
        if (
            parent_physical_bytes
            > std::numeric_limits<std::uint64_t>::max() - physical_bytes
        ) {
            throw std::runtime_error(
                "hybrid validation total physical-byte sum overflow");
        }
        physical_bytes += parent_physical_bytes;
        ledger.record(
            "completion",
            request_record_id + "/completion",
            request_record_id,
            [&]() {
                std::cout << std::setprecision(17)
                          << ",\"request_id\":\"" << json_escape(spec.id)
                          << "\",\"model\":\"hybrid\","
                             "\"action\":\"complete\""
                          << ",\"arrival_ns\":" << spec.arrival_ns
                          << ",\"start_ns\":" << parent_start_ns
                          << ",\"finish_ns\":" << parent_finish_ns
                          << ",\"logical_bytes\":" << spec.bytes
                          << ",\"physical_bytes\":"
                          << parent_physical_bytes
                          << ",\"resource\":\""
                          << hybrid_parent_resource(parent_children)
                          << "\",\"result\":\""
                          << (
                                 has_hbm && has_hbf ?
                                     "split-tier-request" :
                                     "single-tier-request"
                             )
                          << "\"";
            });
    }

    ledger.record("summary", "summary", std::nullopt, [&]() {
        const auto& hbm_stats = result.hbm_stats;
        const auto& hbf_stats = result.hbf_stats;
        std::cout << std::setprecision(17)
                  << ",\"model\":\"hybrid\",\"action\":\"final\""
                  << ",\"config\":{\"hbm\":";
        write_hbm_config(std::cout, options.hbm);
        std::cout << ",\"hbf\":";
        write_hbf_config(std::cout, hbf_config);
        std::cout << ",\"policy\":{\"kind\":\"flat\","
                     "\"read_boundary\":"
                  << options.hybrid_read_boundary
                  << "},\"knobs\":{\"max_outstanding_requests\":"
                  << options.hybrid_max_outstanding_requests
                  << "}},\"address_observations\":[]"
                  << ",\"counters\":{"
                  << "\"ops\":" << result.ops
                  << ",\"reads\":" << result.reads
                  << ",\"writes\":" << result.writes
                  << ",\"logical_bytes\":" << result.logical_bytes
                  << ",\"physical_bytes\":" << physical_bytes
                  << ",\"child_completions\":"
                  << result.completions.size()
                  << ",\"hbm_user_accesses\":"
                  << result.hbm_user_accesses
                  << ",\"hbf_user_accesses\":"
                  << result.hbf_user_accesses
                  << ",\"hbf_direct_user_ops\":"
                  << result.hbf_direct_user_ops
                  << ",\"hbf_static_read_bytes\":"
                  << result.hbf_static_read_bytes
                  << ",\"finish_ns\":" << result.finish_ns
                  << ",\"user_finish_ns\":" << result.user_finish_ns
                  << ",\"first_offered_arrival_ns\":"
                  << result.first_offered_arrival_ns
                  << ",\"last_offered_arrival_ns\":"
                  << result.last_offered_arrival_ns
                  << ",\"service_latencies_ns\":";
        write_double_array(std::cout, result.service_latencies_ns);
        std::cout << ",\"offered_latencies_ns\":";
        write_double_array(std::cout, result.offered_latencies_ns);
        std::cout << ",\"source_latencies_ns\":";
        write_double_array(std::cout, result.source_latencies_ns);
        std::cout
            << ",\"front_end_admission_waited_ops\":"
            << result.front_end_admission_waited_ops
            << ",\"front_end_admission_wait_work_ns\":"
            << result.front_end_admission_wait_work_ns
            << ",\"front_end_admission_max_wait_ns\":"
            << result.front_end_admission_max_wait_ns
            << ",\"phase_barriers\":" << result.phase_barriers
            << ",\"phase_dependency_waited_ops\":"
            << result.phase_dependency_waited_ops
            << ",\"phase_dependency_wait_work_ns\":"
            << result.phase_dependency_wait_work_ns
            << ",\"phase_dependency_max_wait_ns\":"
            << result.phase_dependency_max_wait_ns
            << ",\"warnings\":" << result.warnings.size()
            << ",\"hbm\":{"
            << "\"read_bytes\":" << hbm_stats.read_bytes
            << ",\"write_bytes\":" << hbm_stats.write_bytes
            << ",\"bus_busy_ns\":" << hbm_stats.bus_busy_ns
            << ",\"finish_ns\":" << hbm_stats.finish_ns
            << ",\"pseudo_channels\":" << hbm_stats.pseudo_channels
            << ",\"active_pseudo_channels\":"
            << hbm_stats.active_pseudo_channels
            << ",\"max_pseudo_channel_accesses\":"
            << hbm_stats.max_pseudo_channel_accesses
            << ",\"max_queue_occupancy\":"
            << hbm_stats.max_queue_occupancy
            << "},\"hbf\":{"
            << "\"read_requests\":" << hbf_stats.read_requests
            << ",\"program_requests\":" << hbf_stats.program_requests
            << ",\"logical_read_bytes\":"
            << hbf_stats.logical_read_bytes
            << ",\"logical_write_bytes\":"
            << hbf_stats.logical_write_bytes
            << ",\"physical_read_bytes\":"
            << hbf_stats.physical_read_bytes
            << ",\"physical_write_bytes\":"
            << hbf_stats.physical_write_bytes
            << ",\"page_reads\":" << hbf_stats.page_reads
            << ",\"page_programs\":" << hbf_stats.page_programs
            << ",\"mapping_lookup_ops\":"
            << hbf_stats.mapping_lookup_ops
            << ",\"mapping_user_lookup_ops\":"
            << hbf_stats.mapping_user_lookup_ops
            << ",\"mapping_gc_lookup_ops\":"
            << hbf_stats.mapping_gc_lookup_ops
            << ",\"finish_ns\":" << hbf_stats.finish_ns
            << ",\"total_pages\":" << hbf_stats.total_pages
            << ",\"free_pages\":" << hbf_stats.free_pages
            << ",\"valid_pages\":" << hbf_stats.valid_pages
            << ",\"invalid_pages\":" << hbf_stats.invalid_pages
            << ",\"accounting_verified\":"
            << (hbf_stats.accounting_verified ? "true" : "false")
            << ",\"active_planes\":" << hbf_stats.active_planes
            << ",\"active_channels\":" << hbf_stats.active_channels
            << ",\"active_dies\":" << hbf_stats.active_dies
            << "}}";
    });
}

void print_usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0
        << " --case-id ID --model hbm|hbf|hybrid|external"
        << " [--config key=value]..."
        << " [--inspect-address addr]..."
        << " [--prepopulate-lpn lpn]..."
        << " [--request id arrival_ns "
           "read|write|drain logical|physical|internal addr bytes]..."
        << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.model == "hbm") {
            run_hbm(options);
        } else if (options.model == "hbf") {
            run_hbf(options);
        } else if (options.model == "external") {
            run_external(options);
        } else {
            run_hybrid(options);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        print_usage(argv[0]);
        std::cerr << "ledger_probe error: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
