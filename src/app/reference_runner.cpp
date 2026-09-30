#include "physical/external/external_backing_device.hpp"
#include "host/hbf_controller.hpp"
#include "app/hbf_wear_report.hpp"
#include "physical/hbm/hbm_device.hpp"
#include "policies/reference/behavioral_tiering_policy.hpp"
#include "policies/policy_common.hpp"
#include "policies/reference/direct_policy.hpp"
#include "policies/reference/layer_streaming_policy.hpp"
#include "app/sha256.hpp"
#include "app/source_provenance.hpp"
#include "app/reference_runner.hpp"
#include "app/system_config.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <functional>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef HBFSIM_VERSION
#define HBFSIM_VERSION "unknown"
#endif
#ifndef HBFSIM_BUILD_TYPE
#define HBFSIM_BUILD_TYPE "unknown"
#endif
#ifndef HBFSIM_COMPILER_ID
#define HBFSIM_COMPILER_ID "unknown"
#endif
#ifndef HBFSIM_COMPILER_VERSION
#define HBFSIM_COMPILER_VERSION "unknown"
#endif

using hbfsim::physical::AddressSpace;
using hbfsim::physical::AddressDomain;
using hbfsim::physical::AddressHeatmapSnapshot;
using hbfsim::physical::Breakdown;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalCompletion;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::physical::TraceConfig;
using hbfsim::physical::TraceMode;
using hbfsim::physical::external::ExternalBackingConfig;
using hbfsim::physical::external::ExternalBackingKind;
using hbfsim::physical::external::ExternalBackingStats;
using hbfsim::host::HbfConfig;
using hbfsim::host::HbfController;
using hbfsim::host::MappingMode;
using hbfsim::physical::hbm::HbmConfig;
using hbfsim::physical::hbm::HbmDevice;
using hbfsim::physical::BaseDieLinkConfig;
using hbfsim::physical::BaseDieLinkStats;
using hbfsim::policy::BehavioralAdmissionPolicy;
using hbfsim::policy::BehavioralTieringConfig;
using hbfsim::policy::BehavioralTieringStats;
using hbfsim::policy::PolicyRunResult;
using hbfsim::policy::DirectPolicy;
using hbfsim::policy::DirectRunKnobs;
using hbfsim::policy::MemoryRequest;
using hbfsim::policy::LayerStreamingPolicy;
using hbfsim::policy::LayerStreamingConfig;
using hbfsim::policy::LayerResidencyContract;
using hbfsim::policy::BackingTier;
using hbfsim::policy::LayerStreamingStats;
using hbfsim::policy::SemanticKind;
using hbfsim::policy::SequentialReadWorkload;
using hbfsim::policy::all_hbf_policy;
using hbfsim::policy::all_hbm_policy;
using hbfsim::policy::flat_policy;
using hbfsim::app::sha256_file;
using hbfsim::app::sha256_text;
using hbfsim::app::SystemConfig;
using hbfsim::app::SystemConfigBuilder;
using hbfsim::policy::read_only_direct_policy;
using hbfsim::policy::run_direct_policy;
using hbfsim::policy::run_behavioral_tiering_policy;

namespace {

struct TraceOp {
    std::uint64_t addr = 0;
    Op op = Op::Read;
    std::uint64_t bytes = 64;
    double arrival_ns = 0.0;
    std::size_t line_no = 0;
    SemanticKind kind = SemanticKind::Unknown;
    std::string label;
    std::optional<std::uint64_t> phase;
    std::optional<std::uint64_t> layer;
    std::optional<double> compute_ns;
};

struct Options {
    std::optional<std::string> trace_path;
    // Publication overlays bind the exact trace bytes they were derived
    // from. CLI trace overrides remain possible only when content-identical.
    std::optional<std::string> expected_trace_sha256;
    std::optional<std::uint64_t> expected_trace_file_bytes;
    // Optional full population description for prefix/scaling runs. It is
    // never executed; only initial read-before-write pages seed HBF.
    std::optional<std::string> initial_image_trace_path;
    // File-free fixed-size sequential reads. This is mutually exclusive with
    // --trace and expands lazily into exactly the same MemoryRequest stream.
    std::optional<std::uint64_t> synthetic_sequential_read_bytes;
    std::uint64_t synthetic_sequential_read_base = 0;
    std::optional<std::string> generate_semantic_llm_path;
    std::optional<std::string> chrome_trace_path;
    std::optional<std::string> summary_csv_path;
    std::optional<std::string> summary_json_path;
    std::optional<std::string> config_out_path;
    bool trace_census_only = false;
    std::uint64_t line_size = 64;
    double interarrival_ns = 1.0;
    std::size_t max_ops = 0;
    // FLAT's HBM/HBF address boundary. It follows hbm-capacity-bytes unless
    // set explicitly.
    std::uint64_t flat_hbm_bytes = 64ull * 1024ull * 1024ull;
    bool hbm_capacity_explicit = false;
    bool flat_hbm_bytes_explicit = false;
    // Maximum capacity of one ping-pong layer buffer. Runtime sizing uses
    // the actual maximum per-layer data footprint and may allocate less.
    std::uint64_t layer_buffer_bytes = 32ull * 1024ull * 1024ull * 1024ull;
    // Complete Frontier/object-map population. When enabled, capacity and
    // hot/cold placement come from these independently exported values; the
    // selected trace window contributes traffic only.
    bool explicit_residency_contract = false;
    std::optional<std::uint64_t> residency_page_size_bytes;
    std::optional<std::uint64_t> residency_unique_footprint_bytes;
    std::optional<std::uint64_t> residency_immutable_weight_bytes;
    std::optional<std::uint64_t> residency_immutable_weight_pages;
    std::optional<std::uint64_t> residency_static_weight_pages;
    std::optional<std::uint64_t> residency_runtime_overhead_bytes;
    std::optional<std::uint64_t> residency_block_table_bytes;
    std::optional<std::uint64_t> residency_active_buffer_bytes;
    std::optional<std::uint64_t> residency_kv_region_begin;
    std::optional<std::uint64_t> residency_kv_block_stride_bytes;
    std::optional<std::uint64_t> residency_logical_kv_blocks;
    std::optional<std::uint64_t> residency_hot_kv_blocks;
    // Online behavior-only HBM residency. Zero owns the complete configured
    // HBM capacity. The bounded history is metadata, not data capacity.
    std::uint64_t behavioral_hbm_bytes = 0;
    std::uint32_t behavioral_promotion_threshold = 2;
    std::uint64_t behavioral_history_pages = 1'048'576;
    // Completion-order page-transaction credits. Layer streaming applies the
    // same numeric bound independently to foreground HBM and backing traffic.
    std::size_t max_outstanding_requests = 0;  // 0 = open loop
    std::size_t max_hbm_outstanding_requests = 0;
    std::size_t max_hbf_outstanding_requests = 0;
    // Fixed-size spatial observability, independent of trace mode and
    // bounded by configuration rather than operation count. Off by default:
    // a 1024-bin snapshot is roughly 2 MB of summary JSON per scenario.
    std::size_t address_heatmap_bins = 0;
    std::size_t llm_tokens = 32;
    std::size_t llm_layers = 8;
    std::uint64_t llm_weight_base = 0x1000'0000ull;
    std::uint64_t llm_kv_base = 0x4000'0000ull;
    std::uint64_t llm_scratch_base = 0x7000'0000ull;
    std::uint64_t static_direct_hbm_bytes = 0;
    // Empty = run all scenarios. A per-case selection lets a config be used
    // at capacities its auxiliary scenarios could not survive.
    std::vector<std::string> scenarios;
    // Cooperative HBM/HBF write staging region (0 = off): HBF-bound writes
    // in the direct compositions land in HBM first and destage to the FTL
    // in the background once the pending dirty slots exceed the watermark
    // fraction of the region (see DirectRunKnobs).
    std::uint64_t hbf_hbm_write_buffer_bytes = 0;
    double hbf_hbm_write_buffer_destage_watermark = 0.5;
    TraceMode trace_mode = TraceMode::Off;
};

// Everything one run needs: the runner's own workload/policy options plus
// the engine-owned hardware description, resolved by the same
// SystemConfigBuilder the core `hbfsim` uses. Hardware keys are therefore
// parsed, validated, and defaulted in exactly one place.
struct RunConfig {
    Options options;
    SystemConfig system;
};

// Parse-time state: hardware keys accumulate in the builder and are resolved
// once every config file and CLI override has been applied.
struct RunBuilder {
    Options options;
    SystemConfigBuilder system;
};

constexpr std::uint64_t KiB = 1024ull;
constexpr std::uint64_t MiB = KiB * 1024ull;
constexpr std::uint64_t GiB = MiB * 1024ull;

struct RunProvenance {
    std::vector<std::string> command;
    std::string working_directory;
    std::string executable_path;
    std::string executable_sha256;
    std::uint64_t executable_file_bytes = 0;
    std::string trace_sha256;
    std::uint64_t trace_file_bytes = 0;
    std::string initial_image_trace_sha256;
    std::uint64_t initial_image_trace_file_bytes = 0;
};

struct LatencyDistribution {
    std::uint64_t count = 0;
    double sum_work_ns = 0.0;
    double average_ns = 0.0;
    double p50_ns = 0.0;
    double p95_ns = 0.0;
    double max_ns = 0.0;
};

// Every composition issues its HBM page transactions synchronously, one at a
// time, through HbmDevice::issue (all-hbm, flat, direct-read, both behavioral
// tiers, and the layer streamer alike). The summary records the discipline so
// a reader can tell which HBM path produced a number.
constexpr const char* kHbmIssueMode = "synchronous-per-transaction";

// What the arrival model offered, per scenario, against each present tier's
// peak payload rate. Rates are undefined (nullopt) when the whole stream
// arrives at one instant.
struct OfferedLoad {
    bool open_loop = false;
    double arrival_span_ns = 0.0;
    std::optional<double> rate_GBps;
    std::optional<double> read_rate_GBps;
    std::optional<double> write_rate_GBps;
    std::optional<double> hbm_peak_GBps;
    std::optional<double> hbf_peak_GBps;
    std::optional<double> external_read_peak_GBps;
    std::optional<double> external_write_peak_GBps;
    std::optional<double> hbm_ratio;
    std::optional<double> hbf_ratio;
    std::optional<double> external_ratio;
};

struct ScenarioResult {
    std::string name;
    std::vector<double> service_latencies_ns;
    std::vector<double> offered_latencies_ns;
    std::vector<double> source_latencies_ns;
    std::optional<LatencyDistribution> service_latency_distribution;
    std::optional<LatencyDistribution> offered_latency_distribution;
    std::optional<LatencyDistribution> source_latency_distribution;
    std::vector<PhysicalCompletion> completions;
    std::uint64_t ops = 0;
    std::uint64_t reads = 0;
    std::uint64_t writes = 0;
    std::uint64_t logical_bytes = 0;
    // Offered read/write split of the workload (identical for every scenario).
    std::uint64_t logical_read_bytes = 0;
    std::uint64_t logical_write_bytes = 0;
    std::uint64_t hbm_accesses = 0;
    std::uint64_t hbf_accesses = 0;
    std::uint64_t external_accesses = 0;
    std::uint64_t hbm_user_accesses = 0;
    std::uint64_t hbf_user_accesses = 0;
    std::uint64_t hbf_direct_user_ops = 0;
    std::uint64_t hbm_background_accesses = 0;
    std::uint64_t hbf_background_accesses = 0;
    std::uint64_t external_background_accesses = 0;
    std::optional<LayerStreamingStats> layer_streaming_stats;
    std::optional<BehavioralTieringStats> behavioral_tiering_stats;
    std::uint64_t background_hbf_writes = 0;
    std::uint64_t hbf_static_read_bytes = 0;
    std::uint64_t hbf_logical_read_bytes = 0;
    std::uint64_t hbf_backing_write_bytes = 0;
    std::uint64_t external_backing_read_bytes = 0;
    std::uint64_t external_backing_write_bytes = 0;
    std::uint64_t hbm_foreground_bytes = 0;
    std::uint64_t hbm_streaming_write_bytes = 0;
    std::uint64_t hbm_write_buffer_user_write_bytes = 0;
    std::uint64_t hbm_write_buffer_destaged_bytes = 0;
    std::uint64_t hbm_write_buffer_drain_destaged_bytes = 0;
    std::uint64_t hbm_write_buffer_peak_bytes = 0;
    std::uint64_t hbm_write_buffer_full_waits = 0;
    double hbm_write_buffer_wait_ns = 0.0;
    bool cooperative_write_controller_present = false;
    BaseDieLinkStats base_die_link_stats;
    std::uint64_t phase_barriers = 0;
    std::uint64_t phase_dependency_waited_ops = 0;
    double phase_dependency_wait_work_ns = 0.0;
    double phase_dependency_max_wait_ns = 0.0;
    std::uint64_t front_end_admission_waited_ops = 0;
    double front_end_admission_wait_work_ns = 0.0;
    double front_end_admission_max_wait_ns = 0.0;
    double finish_ns = 0.0;
    // Absolute completion of the last USER op. finish_ns additionally covers
    // write-buffer/mapping drain and late background fills. Reporting derives
    // both elapsed spans from first_arrival_ns and never divides by an
    // absolute timestamp.
    double user_finish_ns = 0.0;
    // Trace time origin and arrival extent. Absolute completion timestamps
    // remain useful for event correlation, while throughput is calculated
    // only from elapsed intervals derived from this origin.
    double first_arrival_ns = 0.0;
    double last_arrival_ns = 0.0;
    bool owns_arrival_frontier = false;
    bool has_hbm = false;
    bool has_hbf = false;
    bool has_external_backing = false;
    hbfsim::physical::hbm::HbmStats hbm_stats;
    hbfsim::host::HbfStats hbf_stats;
    ExternalBackingStats external_backing_stats;
    std::string hbf_wear_snapshot;
    std::optional<AddressHeatmapSnapshot> address_heatmap;
    std::optional<OfferedLoad> offered_load;
    std::vector<std::string> warnings;
};

constexpr const char* kAllHbmScenario = "all-hbm";
constexpr const char* kAllHbfScenario = "all-hbf";
constexpr const char* kFlatScenario = "flat";
constexpr const char* kStaticDirectScenario = "direct-read";
constexpr const char* kLayerStreamingScenario = "hbf-streaming";
constexpr const char* kExternalLayerStreamingScenario =
    "external-streaming";
constexpr const char* kDemandFillScenario = "demand-fill";
constexpr const char* kBehavioralTieringScenario =
    "reuse-filtered";

bool scenario_selected(const Options& options, std::string_view name) {
    return options.scenarios.empty() || std::any_of(
        options.scenarios.begin(),
        options.scenarios.end(),
        [name](const std::string& candidate) { return candidate == name; });
}

std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string upper(std::string value) {
    for (auto& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool parse_op_token(const std::string& token, Op& op) {
    const auto t = upper(token);
    if (t == "R" || t == "READ" || t == "LD" || t == "LOAD") {
        op = Op::Read;
        return true;
    }
    if (t == "W" || t == "WRITE" || t == "ST" || t == "STORE") {
        op = Op::Write;
        return true;
    }
    return false;
}

std::string normalize_token(std::string value) {
    value = trim(std::move(value));
    for (auto& ch : value) {
        if (ch == '-' || ch == '.') {
            ch = '_';
        } else {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
    }
    return value;
}

std::optional<SemanticKind> parse_semantic_kind_token(std::string token) {
    const auto eq = token.find('=');
    if (eq != std::string::npos) {
        token = token.substr(eq + 1);
    }
    const auto value = normalize_token(std::move(token));
    if (value.empty() || value == "unknown" || value == "none") {
        return SemanticKind::Unknown;
    }
    if (value == "model" || value == "weights" || value == "model_weight" ||
        value == "model_weights" || value == "weight") {
        return SemanticKind::ModelWeights;
    }
    if (value == "shared" || value == "shared_context" || value == "shared_kv" ||
        value == "prompt_context" || value == "prefill_context") {
        return SemanticKind::SharedContext;
    }
    if (value == "generated" || value == "generated_context" || value == "generated_kv" ||
        value == "decode_context" || value == "kv_cache") {
        return SemanticKind::GeneratedContext;
    }
    if (value == "scratch" || value == "activation" || value == "activations" ||
        value == "temp" || value == "temporary") {
        return SemanticKind::Scratch;
    }
    if (value == "metadata" || value == "meta" || value == "page_table" ||
        value == "mapping") {
        return SemanticKind::Metadata;
    }
    return std::nullopt;
}

bool try_parse_u64(std::string_view token, std::uint64_t& out) {
    if (token.empty() || token.front() == '-') {
        return false;
    }

    // Match the accepted stoull spelling without paying exception costs for
    // ordinary non-numeric semantic tokens. Decimal remains the default so a
    // zero-padded token is never interpreted as octal.
    if (token.front() == '+') {
        token.remove_prefix(1);
        if (token.empty()) {
            return false;
        }
    }
    int base = 10;
    if (token.size() > 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X')) {
        token.remove_prefix(2);
        base = 16;
        if (token.empty()) {
            return false;
        }
    }

    std::uint64_t value = 0;
    const auto result = std::from_chars(
        token.data(), token.data() + token.size(), value, base);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
        return false;
    }
    out = value;
    return true;
}

bool parse_bool(std::string value) {
    value = upper(trim(std::move(value)));
    if (value == "1" || value == "TRUE" || value == "YES" || value == "ON") {
        return true;
    }
    if (value == "0" || value == "FALSE" || value == "NO" || value == "OFF") {
        return false;
    }
    throw std::runtime_error("invalid boolean value: " + value);
}

bool is_lower_hex_sha256(const std::string& value) {
    return value.size() == 64 &&
        std::all_of(
            value.begin(),
            value.end(),
            [](char ch) {
                return (ch >= '0' && ch <= '9') ||
                    (ch >= 'a' && ch <= 'f');
            });
}

double parse_double(const std::string& token, const char* name) {
    try {
        std::size_t used = 0;
        const auto value = std::stod(token, &used);
        if (used == token.size() && std::isfinite(value)) {
            return value;
        }
    } catch (const std::exception&) {
    }
    throw std::runtime_error(std::string(name) + ": invalid number: " + token);
}

std::uint64_t parse_u64(const std::string& token) {
    if (token.empty() || token.front() == '-') {
        throw std::runtime_error("invalid non-negative integer token: " + token);
    }
    std::uint64_t value = 0;
    if (!try_parse_u64(token, value)) {
        throw std::runtime_error("invalid integer token: " + token);
    }
    return value;
}

std::uint32_t parse_u32(const std::string& token, const char* name) {
    const auto value = parse_u64(token);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds uint32_t range");
    }
    return static_cast<std::uint32_t>(value);
}

std::size_t parse_size(const std::string& token, const char* name) {
    const auto value = parse_u64(token);
    if (value > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(std::string(name) + " exceeds size_t range");
    }
    return static_cast<std::size_t>(value);
}

std::uint64_t checked_mul_u64(std::uint64_t lhs, std::uint64_t rhs, const std::string& name) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(name + " overflows uint64_t");
    }
    return lhs * rhs;
}

std::uint64_t checked_add_u64(std::uint64_t lhs, std::uint64_t rhs, const std::string& name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(name + " overflows uint64_t");
    }
    return lhs + rhs;
}

SequentialReadWorkload make_synthetic_sequential_workload(
    const Options& options) {
    if (!options.synthetic_sequential_read_bytes) {
        throw std::runtime_error("synthetic sequential workload is not configured");
    }
    const auto request_count =
        *options.synthetic_sequential_read_bytes / options.line_size;
    if (request_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "synthetic sequential request count exceeds size_t range");
    }
    return SequentialReadWorkload{
        .base_addr = options.synthetic_sequential_read_base,
        .request_bytes = options.line_size,
        .request_count = static_cast<std::size_t>(request_count),
        .first_arrival_ns = 0.0,
        .interarrival_ns = options.interarrival_ns,
    };
}

std::string synthetic_sequential_descriptor(
    const Options& options,
    const SequentialReadWorkload& workload) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "hbfsim.synthetic-sequential-read.v1\n"
        << "base_addr=" << workload.base_addr << '\n'
        << "request_bytes=" << workload.request_bytes << '\n'
        << "request_count=" << workload.request_count << '\n'
        << "total_bytes=" << *options.synthetic_sequential_read_bytes << '\n'
        << "first_arrival_ns=" << workload.first_arrival_ns << '\n'
        << "interarrival_ns=" << workload.interarrival_ns << '\n';
    return out.str();
}

std::string format_bytes(std::uint64_t bytes) {
    std::ostringstream out;
    out << bytes << "B";
    if (bytes % GiB == 0) {
        out << " (" << (bytes / GiB) << "GiB)";
    } else if (bytes % MiB == 0) {
        out << " (" << (bytes / MiB) << "MiB)";
    } else if (bytes % KiB == 0) {
        out << " (" << (bytes / KiB) << "KiB)";
    }
    return out.str();
}

TraceMode parse_trace_mode_value(const std::string& value);

std::optional<TraceOp> parse_trace_line(
    std::string line,
    std::size_t line_no,
    std::uint64_t line_size,
    double arrival_ns,
    std::set<std::string>& unknown_labels) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
        line = line.substr(0, hash);
    }
    line = trim(line);
    if (line.empty()) {
        return std::nullopt;
    }

    std::istringstream in(line);
    std::vector<std::string> tokens;
    for (std::string token; in >> token;) {
        tokens.push_back(token);
    }
    if (tokens.size() < 2) {
        throw std::runtime_error("trace line " + std::to_string(line_no) +
            " must be Ramulator-compatible: <addr> <R|W>");
    }

    Op op = Op::Read;
    std::uint64_t addr = 0;
    std::size_t op_index = 0;
    std::size_t addr_index = 0;
    if (parse_op_token(tokens[1], op)) {
        addr = parse_u64(tokens[0]);
        addr_index = 0;
        op_index = 1;
    } else if (parse_op_token(tokens[0], op)) {
        addr = parse_u64(tokens[1]);
        addr_index = 1;
        op_index = 0;
    } else {
        throw std::runtime_error("trace line " + std::to_string(line_no) +
            " has no R/W operation token");
    }

    auto bytes = line_size;
    auto kind = SemanticKind::Unknown;
    auto arrival = arrival_ns;
    std::string label;
    std::optional<std::uint64_t> phase;
    std::optional<std::uint64_t> layer;
    std::optional<double> compute_ns;
    bool saw_bytes = false;
    bool saw_phase = false;
    bool saw_layer = false;
    bool saw_compute = false;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i == addr_index || i == op_index) {
            continue;
        }
        auto token = tokens[i];
        const auto eq = token.find('=');
        const auto key = eq == std::string::npos ? std::string{} :
            normalize_token(token.substr(0, eq));
        const auto value = eq == std::string::npos ? token : token.substr(eq + 1);
        // A byte count is either `bytes=`/`size=` or one bare integer. A
        // token that is meant as a count but does not parse (bytes=abc,
        // size=x, -64) and a second bare integer are hard errors: silently
        // demoting them to labels would run the op at --line-size.
        const bool numeric_intent = !value.empty() &&
            (std::isdigit(static_cast<unsigned char>(value.front())) ||
             value.front() == '-' || value.front() == '+');
        if (key == "bytes" || key == "size" || (key.empty() && numeric_intent)) {
            std::uint64_t parsed_bytes = 0;
            if (!try_parse_u64(value, parsed_bytes)) {
                throw std::runtime_error("trace line " + std::to_string(line_no) +
                    " has invalid byte count: " + token);
            }
            if (saw_bytes) {
                throw std::runtime_error("trace line " + std::to_string(line_no) +
                    " has more than one byte count: " + token);
            }
            bytes = parsed_bytes;
            saw_bytes = true;
            continue;
        }
        if (key == "at" || key == "arrival") {
            // Trace-provided arrival time in ns (e.g. from ASTRA-sim 3.0
            // memory-channel export). Overrides the synthesized
            // interarrival-ns arrival for this op only.
            bool ok = false;
            try {
                std::size_t used = 0;
                const double parsed = std::stod(value, &used);
                if (used == value.size() && std::isfinite(parsed) && parsed >= 0.0) {
                    arrival = parsed;
                    ok = true;
                }
            } catch (const std::exception&) {
            }
            if (!ok) {
                throw std::runtime_error("trace line " + std::to_string(line_no) +
                    " has invalid arrival time: " + value);
            }
            continue;
        }
        if (key == "phase") {
            if (saw_phase) {
                throw std::runtime_error(
                    "trace line " + std::to_string(line_no) +
                    " has duplicate phase field");
            }
            std::uint64_t parsed_phase = 0;
            if (!try_parse_u64(value, parsed_phase)) {
                throw std::runtime_error("trace line " + std::to_string(line_no) +
                    " has invalid phase id: " + value);
            }
            phase = parsed_phase;
            saw_phase = true;
            continue;
        }
        if (key == "layer") {
            if (saw_layer) {
                throw std::runtime_error(
                    "trace line " + std::to_string(line_no) +
                    " has duplicate layer field");
            }
            std::uint64_t parsed_layer = 0;
            if (!try_parse_u64(value, parsed_layer)) {
                throw std::runtime_error("trace line " + std::to_string(line_no) +
                    " has invalid layer id: " + value);
            }
            layer = parsed_layer;
            saw_layer = true;
            continue;
        }
        if (key == "compute_ns" || key == "compute") {
            if (saw_compute) {
                throw std::runtime_error(
                    "trace line " + std::to_string(line_no) +
                    " has duplicate compute_ns field");
            }
            bool ok = false;
            try {
                std::size_t used = 0;
                const double parsed = std::stod(value, &used);
                if (used == value.size() && std::isfinite(parsed) &&
                    parsed >= 0.0) {
                    compute_ns = parsed;
                    ok = true;
                }
            } catch (const std::exception&) {
            }
            if (!ok) {
                throw std::runtime_error(
                    "trace line " + std::to_string(line_no) +
                    " has invalid compute_ns: " + value);
            }
            saw_compute = true;
            continue;
        }
        if (key == "label" || key == "region" || key == "name") {
            label = value;
            continue;
        }
        if (key.empty() || key == "kind" || key == "semantic" || key == "type") {
            if (const auto parsed_kind = parse_semantic_kind_token(value)) {
                kind = *parsed_kind;
                if (label.empty()) {
                    label = value;
                }
                continue;
            }
            if (!key.empty()) {
                // An explicit kind= that does not parse is a typo, and silently
                // treating it as a label would change routing in the semantic
                // scenarios.
                throw std::runtime_error("trace line " + std::to_string(line_no) +
                    " has unknown semantic kind: " + value);
            }
        }
        if (label.empty()) {
            label = token;
            unknown_labels.insert(token);
        }
    }
    if (bytes == 0) {
        throw std::runtime_error("trace line " + std::to_string(line_no) +
            " has zero-byte memory operation");
    }
    if (bytes - 1 > std::numeric_limits<std::uint64_t>::max() - addr) {
        throw std::runtime_error("trace line " + std::to_string(line_no) +
            " address range overflows uint64_t");
    }

    return TraceOp{
        .addr = addr,
        .op = op,
        .bytes = bytes,
        .arrival_ns = arrival,
        .line_no = line_no,
        .kind = kind,
        .label = std::move(label),
        .phase = phase,
        .layer = layer,
        .compute_ns = compute_ns,
    };
}

void apply_option(Options& options, const std::string& raw_key, const std::optional<std::string>& value) {
    const auto key = raw_key.starts_with("--") ? raw_key.substr(2) : raw_key;
    auto need = [&]() -> const std::string& {
        if (!value) {
            throw std::runtime_error("option requires a value: " + raw_key);
        }
        return *value;
    };

    if (key == "trace") {
        options.trace_path = need();
    } else if (key == "expected-trace-sha256") {
        options.expected_trace_sha256 = need();
    } else if (key == "expected-trace-bytes") {
        options.expected_trace_file_bytes = parse_u64(need());
    } else if (key == "initial-image-trace") {
        options.initial_image_trace_path = need();
    } else if (key == "synthetic-sequential-read-bytes") {
        options.synthetic_sequential_read_bytes = parse_u64(need());
    } else if (key == "synthetic-sequential-read-base") {
        options.synthetic_sequential_read_base = parse_u64(need());
    } else if (key == "generate-semantic-llm") {
        options.generate_semantic_llm_path = need();
    } else if (key == "chrome-trace") {
        options.chrome_trace_path = need();
    } else if (key == "summary-csv") {
        options.summary_csv_path = need();
    } else if (key == "summary-json") {
        options.summary_json_path = need();
    } else if (key == "config-out") {
        options.config_out_path = need();
    } else if (key == "trace-census-only") {
        options.trace_census_only = parse_bool(need());
    } else if (key == "line-size") {
        options.line_size = parse_u64(need());
    } else if (key == "interarrival-ns") {
        options.interarrival_ns = parse_double(need(), "--interarrival-ns");
    } else if (key == "max-ops") {
        options.max_ops = parse_size(need(), "--max-ops");
    } else if (key == "flat-hbm-bytes") {
        options.flat_hbm_bytes = parse_u64(need());
        options.flat_hbm_bytes_explicit = true;
    } else if (key == "layer-buffer-bytes") {
        options.layer_buffer_bytes = parse_u64(need());
    } else if (key == "explicit-residency-contract") {
        options.explicit_residency_contract = parse_bool(need());
    } else if (key == "residency-page-size-bytes") {
        options.residency_page_size_bytes = parse_u64(need());
    } else if (key == "residency-unique-footprint-bytes") {
        options.residency_unique_footprint_bytes = parse_u64(need());
    } else if (key == "residency-immutable-weight-bytes") {
        options.residency_immutable_weight_bytes = parse_u64(need());
    } else if (key == "residency-immutable-weight-pages") {
        options.residency_immutable_weight_pages = parse_u64(need());
    } else if (key == "residency-static-weight-pages") {
        options.residency_static_weight_pages = parse_u64(need());
    } else if (key == "residency-runtime-overhead-bytes") {
        options.residency_runtime_overhead_bytes = parse_u64(need());
    } else if (key == "residency-block-table-bytes") {
        options.residency_block_table_bytes = parse_u64(need());
    } else if (key == "residency-active-buffer-bytes") {
        options.residency_active_buffer_bytes = parse_u64(need());
    } else if (key == "residency-kv-region-begin") {
        options.residency_kv_region_begin = parse_u64(need());
    } else if (key == "residency-kv-block-stride-bytes") {
        options.residency_kv_block_stride_bytes = parse_u64(need());
    } else if (key == "residency-logical-kv-blocks") {
        options.residency_logical_kv_blocks = parse_u64(need());
    } else if (key == "residency-hot-kv-blocks") {
        options.residency_hot_kv_blocks = parse_u64(need());
    } else if (key == "behavioral-hbm-bytes") {
        options.behavioral_hbm_bytes = parse_u64(need());
    } else if (key == "behavioral-promotion-threshold") {
        options.behavioral_promotion_threshold =
            parse_u32(need(), "--behavioral-promotion-threshold");
    } else if (key == "behavioral-history-pages") {
        options.behavioral_history_pages = parse_u64(need());
    } else if (key == "max-outstanding-requests") {
        options.max_outstanding_requests =
            parse_size(need(), "--max-outstanding-requests");
    } else if (key == "max-hbm-outstanding-requests") {
        options.max_hbm_outstanding_requests =
            parse_size(need(), "--max-hbm-outstanding-requests");
    } else if (key == "max-hbf-outstanding-requests") {
        options.max_hbf_outstanding_requests =
            parse_size(need(), "--max-hbf-outstanding-requests");
    } else if (key == "address-heatmap-bins") {
        options.address_heatmap_bins =
            parse_size(need(), "--address-heatmap-bins");
    } else if (key == "llm-tokens") {
        options.llm_tokens = parse_size(need(), "--llm-tokens");
    } else if (key == "llm-layers") {
        options.llm_layers = parse_size(need(), "--llm-layers");
    } else if (key == "llm-weight-base") {
        options.llm_weight_base = parse_u64(need());
    } else if (key == "llm-kv-base") {
        options.llm_kv_base = parse_u64(need());
    } else if (key == "llm-scratch-base") {
        options.llm_scratch_base = parse_u64(need());
    } else if (key == "static-direct-hbm-bytes") {
        options.static_direct_hbm_bytes = parse_u64(need());
    } else if (key == "scenarios") {
        options.scenarios.clear();
        std::string list = need();
        std::size_t start = 0;
        while (start <= list.size()) {
            const auto comma = list.find(',', start);
            const auto token = list.substr(
                start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!token.empty()) {
                options.scenarios.push_back(token);
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    } else if (key == "hbf-hbm-write-buffer-bytes") {
        options.hbf_hbm_write_buffer_bytes = parse_u64(need());
    } else if (key == "hbf-hbm-write-buffer-destage-watermark") {
        options.hbf_hbm_write_buffer_destage_watermark = parse_double(
            need(), "--hbf-hbm-write-buffer-destage-watermark");
    } else if (key == "trace-mode") {
        options.trace_mode = parse_trace_mode_value(need());
    } else {
        throw std::runtime_error("unknown option: " + raw_key);
    }
}

// One entry point for config-file records and CLI overrides. Hardware keys
// go to the engine's builder (the same parser the core `hbfsim` uses); the
// runner keeps only its workload and policy table above.
void apply_key(RunBuilder& run, const std::string& raw_key, const std::string& value) {
    const auto key = raw_key.starts_with("--") ? raw_key.substr(2) : raw_key;
    if (SystemConfigBuilder::owns_key(key)) {
        if (key == "hbm-capacity-bytes") {
            run.options.hbm_capacity_explicit = true;
        }
        try {
            run.system.apply(key, value);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                (raw_key.starts_with("--") ? raw_key : "--" + key) + ": " +
                error.what());
        }
        return;
    }
    apply_option(run.options, raw_key, value);
}

void load_config_file(RunBuilder& run, const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open config file: " + path);
    }
    std::string line;
    std::size_t line_no = 0;
    std::set<std::string> seen_keys;
    while (std::getline(in, line)) {
        ++line_no;
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error(
                "config line " + std::to_string(line_no) + " must be key=value");
        }
        const auto key = trim(line.substr(0, eq));
        const auto value = trim(line.substr(eq + 1));
        if (key.empty()) {
            throw std::runtime_error("config line " + std::to_string(line_no) + " has empty key");
        }
        if (key == "config") {
            throw std::runtime_error(
                "config files cannot include other config files (line " +
                std::to_string(line_no) + ")");
        }
        if (!seen_keys.insert(key).second) {
            throw std::runtime_error(
                "duplicate config key " + key + " at line " +
                std::to_string(line_no));
        }
        try {
            apply_key(run, key, value);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                path + ":" + std::to_string(line_no) + ": " + error.what());
        }
    }
}

std::vector<TraceOp> load_ramulator_trace(const Options& options) {
    if (!options.trace_path) {
        throw std::runtime_error(
            "--trace is required unless a trace generator is used without simulation");
    }
    std::ifstream in(*options.trace_path);
    if (!in) {
        throw std::runtime_error("cannot open trace: " + *options.trace_path);
    }

    std::vector<TraceOp> ops;
    std::string line;
    std::size_t line_no = 0;
    std::set<std::string> unknown_labels;
    while (std::getline(in, line)) {
        ++line_no;
        const double arrival = static_cast<double>(ops.size()) * options.interarrival_ns;
        auto op = parse_trace_line(line, line_no, options.line_size, arrival, unknown_labels);
        if (!op) {
            continue;
        }
        ops.push_back(*op);
        if (options.max_ops != 0 && ops.size() >= options.max_ops) {
            break;
        }
    }
    if (ops.empty()) {
        throw std::runtime_error("trace contains no memory operations");
    }
    if (!unknown_labels.empty()) {
        std::cerr << "warning: trace labels matching no semantic kind (kept as labels, kind=unknown):";
        for (const auto& token : unknown_labels) {
            std::cerr << ' ' << token;
        }
        std::cerr << '\n';
    }
    return ops;
}

void ensure_parent_dir(const std::string& output_path) {
    const std::filesystem::path path(output_path);
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void finish_output(
    std::ofstream& out,
    const std::string& output_path,
    const char* description) {
    out.flush();
    if (!out) {
        throw std::runtime_error(
            std::string("failed while writing ") + description + ": " + output_path);
    }
}

void emit_semantic_op(
    std::ostream& out,
    std::uint64_t addr,
    Op op,
    std::uint64_t bytes,
    SemanticKind kind,
    std::uint64_t layer) {
    out << "0x" << std::hex << addr << std::dec << ' '
        << (op == Op::Read ? 'R' : 'W') << ' '
        << bytes << ' '
        << hbfsim::policy::to_string(kind)
        << " layer=" << layer
        << '\n';
}

void generate_semantic_llm_trace(const Options& options) {
    const auto& output_path = *options.generate_semantic_llm_path;
    if (output_path.empty()) {
        throw std::runtime_error("internal error: missing generate path");
    }

    constexpr std::uint64_t layer_stride = 0x0100'0000ull;
    constexpr std::uint64_t token_stride = 0x0000'4000ull;
    constexpr std::uint64_t scratch_layer_stride = 0x0002'0000ull;
    constexpr std::uint64_t weight_lines = 128;
    constexpr std::uint64_t kv_lines = 32;
    constexpr std::uint64_t scratch_lines = 16;
    const auto last_layer = static_cast<std::uint64_t>(options.llm_layers - 1);
    const auto last_token = static_cast<std::uint64_t>(options.llm_tokens - 1);
    const auto checked_region_end = [&] (
        std::uint64_t base,
        std::uint64_t layer_offset,
        std::uint64_t token_offset,
        std::uint64_t lines,
        const char* name) {
        auto end = checked_add_u64(base, layer_offset, name);
        end = checked_add_u64(end, token_offset, name);
        const auto line_bytes = checked_mul_u64(lines, options.line_size, name);
        return checked_add_u64(end, line_bytes - 1, name);
    };
    (void)checked_region_end(
        options.llm_weight_base,
        checked_mul_u64(last_layer, layer_stride, "generated weight address range"),
        0,
        weight_lines,
        "generated weight address range");
    (void)checked_region_end(
        options.llm_kv_base,
        checked_mul_u64(last_layer, layer_stride, "generated KV address range"),
        checked_mul_u64(last_token, token_stride, "generated KV address range"),
        kv_lines,
        "generated KV address range");
    (void)checked_region_end(
        options.llm_scratch_base,
        checked_add_u64(
            checked_mul_u64(
                std::min<std::uint64_t>(last_token, 3),
                layer_stride,
                "generated scratch address range"),
            checked_mul_u64(
                last_layer,
                scratch_layer_stride,
                "generated scratch address range"),
            "generated scratch address range"),
        0,
        scratch_lines,
        "generated scratch address range");

    ensure_parent_dir(output_path);
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot write generated trace: " + output_path);
    }

    out << "# HBFSim generated semantic LLM-like trace\n";
    out << "# llm_tokens=" << options.llm_tokens << '\n';
    out << "# llm_layers=" << options.llm_layers << '\n';
    out << "# llm_weight_base=0x" << std::hex << options.llm_weight_base << std::dec << '\n';
    out << "# llm_kv_base=0x" << std::hex << options.llm_kv_base << std::dec << '\n';
    out << "# llm_scratch_base=0x" << std::hex << options.llm_scratch_base << std::dec << '\n';

    for (std::size_t token = 0; token < options.llm_tokens; ++token) {
        for (std::size_t layer = 0; layer < options.llm_layers; ++layer) {
            const auto execution_layer = checked_add_u64(
                checked_mul_u64(token, options.llm_layers, "generated layer id"),
                layer,
                "generated layer id");
            const auto layer_weight = options.llm_weight_base + layer * layer_stride;
            for (std::size_t line = 0; line < weight_lines; ++line) {
                emit_semantic_op(
                    out,
                    layer_weight + line * options.line_size,
                    Op::Read,
                    options.line_size,
                    SemanticKind::ModelWeights,
                    execution_layer);
            }

            const auto layer_kv = options.llm_kv_base + layer * layer_stride;
            const auto history = std::min<std::size_t>(token, 32);
            for (std::size_t prev = token - history; prev < token; ++prev) {
                for (std::size_t line = 0; line < kv_lines; ++line) {
                    const auto addr = layer_kv + prev * token_stride + line * options.line_size;
                    emit_semantic_op(
                        out,
                        addr,
                        Op::Read,
                        options.line_size,
                        SemanticKind::SharedContext,
                        execution_layer);
                }
            }
            for (std::size_t line = 0; line < kv_lines; ++line) {
                const auto addr = layer_kv + token * token_stride + line * options.line_size;
                emit_semantic_op(
                    out,
                    addr,
                    Op::Write,
                    options.line_size,
                    SemanticKind::GeneratedContext,
                    execution_layer);
            }

            const auto scratch = options.llm_scratch_base +
                (token % 4) * layer_stride + layer * scratch_layer_stride;
            for (std::size_t line = 0; line < scratch_lines; ++line) {
                const auto addr = scratch + line * options.line_size;
                emit_semantic_op(
                    out, addr, Op::Write, options.line_size,
                    SemanticKind::Scratch, execution_layer);
                emit_semantic_op(
                    out, addr, Op::Read, options.line_size,
                    SemanticKind::Scratch, execution_layer);
            }
        }
    }
    finish_output(out, output_path, "generated trace");
}

TraceMode effective_trace_mode(const Options& options) {
    if (options.trace_mode != TraceMode::Off) {
        return options.trace_mode;
    }
    return options.chrome_trace_path ? TraceMode::Full : TraceMode::Off;
}

std::uint64_t hbf_capacity_bytes(const HbfConfig& hbf) {
    auto value = static_cast<std::uint64_t>(hbf.device.stacks);
    value = checked_mul_u64(value, hbf.device.channels_per_stack, "HBF capacity");
    value = checked_mul_u64(value, hbf.device.dies_per_channel, "HBF capacity");
    value = checked_mul_u64(value, hbf.device.planes_per_die, "HBF capacity");
    value = checked_mul_u64(value, hbf.device.blocks_per_plane, "HBF capacity");
    value = checked_mul_u64(value, hbf.device.pages_per_block, "HBF capacity");
    value = checked_mul_u64(value, hbf.device.page_size_bytes, "HBF capacity");
    return value;
}

std::uint64_t hbm_cooperative_region_base_addr(
    const Options& options,
    const SystemConfig& system) {
    if (options.hbf_hbm_write_buffer_bytes == 0) return system.hbm.device.capacity_bytes;
    const auto burst = system.hbm.burst_bytes();
    const auto reserved = checked_add_u64(system.hbf.host.ctrl_dram_bytes,
        burst - 1, "controller HBM reservation") / burst * burst;
    if (reserved >= system.hbm.device.capacity_bytes ||
        options.hbf_hbm_write_buffer_bytes > system.hbm.device.capacity_bytes - reserved) {
        throw std::runtime_error("cooperative write region exceeds HBM capacity");
    }
    const auto page = system.hbf.device.page_size_bytes;
    return (system.hbm.device.capacity_bytes - reserved -
        options.hbf_hbm_write_buffer_bytes) / page * page;
}

// Peak payload rates the offered stream is compared against: every HBM
// channel's derived interface rate, the decoded-payload HBIO of every HBF
// stack, and (per direction) the configured external wire rates, where reads
// return over S2M and writes travel over M2S.
double hbm_peak_rate_GBps(const HbmConfig& hbm) {
    return hbm.channel_bandwidth_GBps() * hbm.device.channels_per_stack * hbm.device.stacks;
}

double hbf_peak_rate_GBps(const HbfConfig& hbf) {
    return hbf.device.hb_io_bandwidth_GBps() * hbf.device.stacks;
}

bool open_loop_arrivals(const Options& options) {
    return options.max_outstanding_requests == 0 &&
        options.max_hbm_outstanding_requests == 0 &&
        options.max_hbf_outstanding_requests == 0;
}

// The offered load is a property of the arrival model, measured per scenario
// over that scenario's offered-arrival span (phase barriers can stretch it).
// It says what the workload presented, not what any device delivered, and
// it is undefined when every request arrives at one instant.
void attach_offered_load(
    ScenarioResult& result,
    const Options& options,
    const SystemConfig& system) {
    OfferedLoad load;
    load.open_loop = open_loop_arrivals(options);
    load.arrival_span_ns = result.last_arrival_ns - result.first_arrival_ns;
    const auto rate = [&](std::uint64_t bytes) -> std::optional<double> {
        if (!(load.arrival_span_ns > 0.0)) {
            return std::nullopt;
        }
        return static_cast<double>(bytes) / load.arrival_span_ns;
    };
    load.rate_GBps = rate(result.logical_bytes);
    load.read_rate_GBps = rate(result.logical_read_bytes);
    load.write_rate_GBps = rate(result.logical_write_bytes);
    const auto ratio = [](std::optional<double> value, double peak)
        -> std::optional<double> {
        if (!value || !(peak > 0.0)) {
            return std::nullopt;
        }
        return *value / peak;
    };
    if (result.has_hbm) {
        load.hbm_peak_GBps = hbm_peak_rate_GBps(system.hbm);
        load.hbm_ratio = ratio(load.rate_GBps, *load.hbm_peak_GBps);
    }
    if (result.has_hbf) {
        load.hbf_peak_GBps = hbf_peak_rate_GBps(system.hbf);
        load.hbf_ratio = ratio(load.rate_GBps, *load.hbf_peak_GBps);
    }
    if (result.has_external_backing) {
        load.external_read_peak_GBps = system.external.s2m_bandwidth_GBps;
        load.external_write_peak_GBps = system.external.m2s_bandwidth_GBps;
        const auto read_ratio =
            ratio(load.read_rate_GBps, *load.external_read_peak_GBps);
        const auto write_ratio =
            ratio(load.write_rate_GBps, *load.external_write_peak_GBps);
        if (read_ratio || write_ratio) {
            load.external_ratio = std::max(
                read_ratio.value_or(0.0), write_ratio.value_or(0.0));
        }
    }
    result.offered_load = load;
}

std::uint64_t trace_unique_line_bytes(
    const std::vector<TraceOp>& ops,
    std::uint64_t line_size) {
    if (line_size == 0) {
        throw std::runtime_error("trace line size must be positive");
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> intervals;
    intervals.reserve(ops.size());
    for (const auto& op : ops) {
        const auto first = op.addr / line_size;
        const auto last = checked_add_u64(
            op.addr,
            op.bytes - 1,
            "trace unique-line range") / line_size;
        intervals.emplace_back(first, last);
    }
    if (intervals.empty()) {
        return 0;
    }
    std::sort(intervals.begin(), intervals.end());
    std::uint64_t unique_lines = 0;
    auto current_first = intervals.front().first;
    auto current_last = intervals.front().second;
    const auto add_interval = [&unique_lines](
                                  std::uint64_t first,
                                  std::uint64_t last) {
        const auto span_minus_one = last - first;
        const auto span = checked_add_u64(
            span_minus_one, 1, "trace unique-line interval length");
        unique_lines = checked_add_u64(
            unique_lines, span, "trace unique-line count");
    };
    for (std::size_t index = 1; index < intervals.size(); ++index) {
        const auto [next_first, next_last] = intervals[index];
        const bool adjacent = current_last !=
                std::numeric_limits<std::uint64_t>::max() &&
            next_first == current_last + 1;
        if (next_first <= current_last || adjacent) {
            current_last = std::max(current_last, next_last);
            continue;
        }
        add_interval(current_first, current_last);
        current_first = next_first;
        current_last = next_last;
    }
    add_interval(current_first, current_last);
    return checked_mul_u64(
        unique_lines, line_size, "trace unique-line footprint");
}

// Adapt a library-side composition run into the tool's per-scenario record.
ScenarioResult from_direct_run(PolicyRunResult&& run, std::string name) {
    ScenarioResult result;
    result.name = std::move(name);
    result.has_hbm = run.has_hbm;
    result.has_hbf = run.has_hbf;
    result.service_latencies_ns = std::move(run.service_latencies_ns);
    result.offered_latencies_ns = std::move(run.offered_latencies_ns);
    result.source_latencies_ns = std::move(run.source_latencies_ns);
    result.completions = std::move(run.completions);
    result.warnings = std::move(run.warnings);
    result.ops = run.ops;
    result.reads = run.reads;
    result.writes = run.writes;
    result.logical_bytes = run.logical_bytes;
    result.hbm_user_accesses = run.hbm_user_accesses;
    result.hbf_user_accesses = run.hbf_user_accesses;
    result.hbf_direct_user_ops = run.hbf_direct_user_ops;
    result.hbm_background_accesses = run.hbm_background_accesses;
    result.hbf_background_accesses = run.hbf_background_accesses;
    result.hbm_accesses = checked_add_u64(
        run.hbm_user_accesses,
        run.hbm_background_accesses,
        "scenario HBM access count");
    result.hbf_accesses = checked_add_u64(
        run.hbf_user_accesses,
        run.hbf_background_accesses,
        "scenario HBF access count");
    result.hbf_static_read_bytes = run.hbf_static_read_bytes;
    result.hbm_write_buffer_user_write_bytes =
        run.hbm_write_buffer_user_write_bytes;
    result.hbm_write_buffer_destaged_bytes = run.hbm_write_buffer_destaged_bytes;
    result.hbm_write_buffer_drain_destaged_bytes =
        run.hbm_write_buffer_drain_destaged_bytes;
    result.hbm_write_buffer_peak_bytes = run.hbm_write_buffer_peak_bytes;
    result.hbm_write_buffer_full_waits = run.hbm_write_buffer_full_waits;
    result.hbm_write_buffer_wait_ns = run.hbm_write_buffer_wait_ns;
    result.cooperative_write_controller_present =
        run.cooperative_write_controller_present;
    result.base_die_link_stats = run.base_die_link_stats;
    result.phase_barriers = run.phase_barriers;
    result.phase_dependency_waited_ops =
        run.phase_dependency_waited_ops;
    result.phase_dependency_wait_work_ns =
        run.phase_dependency_wait_work_ns;
    result.phase_dependency_max_wait_ns =
        run.phase_dependency_max_wait_ns;
    result.front_end_admission_waited_ops = run.front_end_admission_waited_ops;
    result.front_end_admission_wait_work_ns = run.front_end_admission_wait_work_ns;
    result.front_end_admission_max_wait_ns = run.front_end_admission_max_wait_ns;
    result.finish_ns = run.finish_ns;
    result.user_finish_ns = run.user_finish_ns;
    result.first_arrival_ns = run.first_offered_arrival_ns;
    result.last_arrival_ns = run.last_offered_arrival_ns;
    result.owns_arrival_frontier = true;
    result.hbm_stats = run.hbm_stats;
    result.hbf_stats = run.hbf_stats;
    result.hbf_logical_read_bytes = run.has_hbf ?
        run.hbf_stats.logical_read_bytes :
        0;
    result.hbf_wear_snapshot = std::move(run.hbf_wear_snapshot);
    result.address_heatmap = std::move(run.address_heatmap);
    return result;
}

DirectRunKnobs make_direct_run_knobs(
    const Options& options,
    const SystemConfig& system,
    const std::vector<MemoryRequest>* initial_image_requests = nullptr) {
    return DirectRunKnobs{
        .max_outstanding_requests = options.max_outstanding_requests,
        .max_hbm_outstanding_requests =
            options.max_hbm_outstanding_requests,
        .max_hbf_outstanding_requests =
            options.max_hbf_outstanding_requests,
        .hbm_write_buffer_bytes = options.hbf_hbm_write_buffer_bytes,
        .hbm_write_buffer_destage_watermark =
            options.hbf_hbm_write_buffer_destage_watermark,
        .base_die_link = system.base_die_link,
        .trace = {
            .mode = effective_trace_mode(options),
            .retain_completion_diagnostics =
                options.chrome_trace_path.has_value(),
        },
        .retain_completions = options.chrome_trace_path.has_value(),
        .address_heatmap_bins = options.address_heatmap_bins,
        .initial_image_requests = initial_image_requests,
    };
}

template <typename RequestSource>
ScenarioResult run_direct(
    const DirectPolicy& policy,
    std::string scenario_name,
    const RequestSource& requests,
    const Options& options,
    const SystemConfig& system,
    const std::vector<MemoryRequest>* initial_image_requests = nullptr) {
    auto run = run_direct_policy(
        policy,
        system.hbm,
        system.hbf,
        requests,
        make_direct_run_knobs(options, system, initial_image_requests));
    return from_direct_run(std::move(run), std::move(scenario_name));
}

BehavioralTieringConfig make_behavioral_tiering_config(
    const Options& options,
    const SystemConfig& system,
    BehavioralAdmissionPolicy policy,
    const std::vector<MemoryRequest>* initial_image_requests) {
    const auto burst = system.hbm.burst_bytes();
    const auto reserved = checked_add_u64(system.hbf.host.ctrl_dram_bytes,
        burst - 1, "behavioral controller HBM reservation") / burst * burst;
    if (reserved >= system.hbm.device.capacity_bytes)
        throw std::runtime_error("behavioral-tiering HBM cannot contain its controller buffers");
    const auto page = system.hbf.device.page_size_bytes;
    const auto application_bytes = (system.hbm.device.capacity_bytes - reserved) / page * page;
    return BehavioralTieringConfig{
        .hbm = system.hbm,
        .hbf = system.hbf,
        .base_die_link = system.base_die_link,
        .admission_policy = policy,
        .hbm_tier_bytes = options.behavioral_hbm_bytes == 0 ?
            application_bytes :
            options.behavioral_hbm_bytes,
        .promotion_threshold =
            options.behavioral_promotion_threshold,
        .history_capacity_pages =
            options.behavioral_history_pages,
        .max_outstanding_requests =
            options.max_outstanding_requests,
        .address_heatmap_bins = options.address_heatmap_bins,
        .trace = {
            .mode = effective_trace_mode(options),
            .retain_completion_diagnostics =
                options.chrome_trace_path.has_value(),
        },
        .retain_completions =
            options.chrome_trace_path.has_value(),
        .retain_decisions = false,
        .initial_image_requests = initial_image_requests,
    };
}

ScenarioResult run_behavioral_tiering(
    const std::vector<MemoryRequest>& requests,
    const Options& options,
    const SystemConfig& system,
    BehavioralAdmissionPolicy policy,
    std::string scenario_name,
    const std::vector<MemoryRequest>* initial_image_requests) {
    auto run = run_behavioral_tiering_policy(
        make_behavioral_tiering_config(
            options, system, policy, initial_image_requests),
        requests);
    auto result = from_direct_run(
        std::move(run.composition), std::move(scenario_name));
    result.background_hbf_writes =
        run.placement.dirty_writeback_pages;
    result.hbf_backing_write_bytes =
        run.placement.dirty_writeback_bytes;
    result.hbm_foreground_bytes =
        run.placement.hbm_foreground_bytes;
    result.behavioral_tiering_stats = std::move(run.placement);
    return result;
}

std::vector<MemoryRequest> make_memory_requests(const std::vector<TraceOp>& ops) {
    std::vector<MemoryRequest> requests;
    requests.reserve(ops.size());
    for (std::size_t i = 0; i < ops.size(); ++i) {
        requests.push_back(MemoryRequest{
            .id = "op" + std::to_string(i),
            .op = ops[i].op,
            .addr = ops[i].addr,
            .bytes = ops[i].bytes,
            .arrival_ns = ops[i].arrival_ns,
            .index = i,
            .kind = ops[i].kind,
            .label = ops[i].label,
            .phase = ops[i].phase,
            .layer = ops[i].layer,
            .compute_ns = ops[i].compute_ns,
        });
    }
    return requests;
}

LayerStreamingConfig make_layer_streaming_config(
    const Options& options,
    const SystemConfig& system,
    BackingTier backing) {
    LayerStreamingConfig cfg;
    cfg.hbm = system.hbm;
    cfg.backing = backing;
    cfg.hbf = system.hbf;
    cfg.external_backing = system.external;
    cfg.base_die_link = system.base_die_link;
    cfg.layer_buffer_bytes = options.layer_buffer_bytes;
    if (options.explicit_residency_contract) {
        cfg.residency_contract = LayerResidencyContract{
            .page_size_bytes =
                *options.residency_page_size_bytes,
            .unique_resident_footprint_bytes =
                *options.residency_unique_footprint_bytes,
            .immutable_weight_bytes =
                *options.residency_immutable_weight_bytes,
            .immutable_weight_pages =
                *options.residency_immutable_weight_pages,
            .static_weight_resident_pages =
                *options.residency_static_weight_pages,
            .runtime_overhead_bytes =
                *options.residency_runtime_overhead_bytes,
            .block_table_bytes =
                *options.residency_block_table_bytes,
            .active_buffer_bytes_per_slot =
                *options.residency_active_buffer_bytes,
            .kv_region_begin =
                *options.residency_kv_region_begin,
            .kv_block_stride_bytes =
                *options.residency_kv_block_stride_bytes,
            .logical_kv_blocks =
                *options.residency_logical_kv_blocks,
            .hot_kv_blocks =
                *options.residency_hot_kv_blocks,
        };
    }
    cfg.max_outstanding_requests = options.max_outstanding_requests;
    cfg.address_heatmap_bins = options.address_heatmap_bins;
    cfg.trace = {
        .mode = effective_trace_mode(options),
        .retain_completion_diagnostics =
            options.chrome_trace_path.has_value(),
    };
    return cfg;
}

ScenarioResult run_layer_streaming(
    const std::vector<MemoryRequest>& requests,
    const Options& options,
    const SystemConfig& system,
    BackingTier backing) {
    ScenarioResult result;
    result.name = backing == BackingTier::Hbf ?
        kLayerStreamingScenario : kExternalLayerStreamingScenario;
    result.has_hbm = true;
    result.has_hbf = backing == BackingTier::Hbf;
    result.has_external_backing = backing == BackingTier::External;

    LayerStreamingPolicy streamer(
        make_layer_streaming_config(options, system, backing));
    auto hybrid = streamer.run(requests);

    result.service_latencies_ns =
        std::move(hybrid.service_latencies_ns);
    result.offered_latencies_ns =
        std::move(hybrid.offered_latencies_ns);
    result.source_latencies_ns =
        std::move(hybrid.source_latencies_ns);
    result.completions = std::move(hybrid.completions);
    result.warnings = std::move(hybrid.warnings);
    result.ops = hybrid.ops;
    result.reads = hybrid.reads;
    result.writes = hybrid.writes;
    result.logical_bytes = hybrid.logical_bytes;
    result.hbm_user_accesses = hybrid.hbm_user_accesses;
    result.hbf_user_accesses = hybrid.hbf_user_accesses;
    result.hbm_background_accesses = hybrid.hbm_background_accesses;
    result.hbf_background_accesses = hybrid.hbf_background_accesses;
    result.external_background_accesses =
        hybrid.external_background_accesses;
    result.hbm_accesses = checked_add_u64(
        result.hbm_user_accesses,
        result.hbm_background_accesses,
        "layer-streaming HBM access count");
    result.hbf_accesses = checked_add_u64(
        result.hbf_user_accesses,
        result.hbf_background_accesses,
        "layer-streaming HBF access count");
    result.external_accesses = hybrid.external_background_accesses;
    result.layer_streaming_stats = hybrid.streaming_stats;
    result.background_hbf_writes = hybrid.background_hbf_writes;
    result.hbf_static_read_bytes = hybrid.hbf_static_read_bytes;
    result.hbf_logical_read_bytes = hybrid.hbf_logical_read_bytes;
    result.hbf_backing_write_bytes = hybrid.hbf_backing_write_bytes;
    result.external_backing_read_bytes =
        hybrid.external_backing_read_bytes;
    result.external_backing_write_bytes =
        hybrid.external_backing_write_bytes;
    result.hbm_foreground_bytes = hybrid.hbm_foreground_bytes;
    result.hbm_streaming_write_bytes = hybrid.hbm_streaming_write_bytes;
    result.base_die_link_stats = hybrid.base_die_link_stats;
    result.front_end_admission_waited_ops = hybrid.front_end_admission_waited_ops;
    result.front_end_admission_wait_work_ns = hybrid.front_end_admission_wait_work_ns;
    result.front_end_admission_max_wait_ns = hybrid.front_end_admission_max_wait_ns;
    result.first_arrival_ns = hybrid.first_offered_arrival_ns;
    result.last_arrival_ns = hybrid.last_offered_arrival_ns;
    result.phase_barriers = hybrid.phase_barriers;
    result.phase_dependency_waited_ops =
        hybrid.phase_dependency_waited_ops;
    result.phase_dependency_wait_work_ns =
        hybrid.phase_dependency_wait_work_ns;
    result.phase_dependency_max_wait_ns =
        hybrid.phase_dependency_max_wait_ns;
    result.owns_arrival_frontier = true;
    result.finish_ns = hybrid.finish_ns;
    result.user_finish_ns = hybrid.user_finish_ns;
    result.hbm_stats = hybrid.hbm_stats;
    result.hbf_stats = hybrid.hbf_stats;
    result.external_backing_stats = hybrid.external_backing_stats;
    result.hbf_wear_snapshot = std::move(hybrid.hbf_wear_snapshot);
    result.address_heatmap = std::move(hybrid.address_heatmap);
    return result;
}

double percentile_sorted(const std::vector<double>& values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    const auto pos = (p / 100.0) * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<std::size_t>(std::floor(pos));
    const auto hi = static_cast<std::size_t>(std::ceil(pos));
    if (lo == hi) {
        return values[lo];
    }
    const auto frac = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

std::string fixed(double value, int precision = 2) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

struct WallClockSnapshot {
    double trace_origin_ns = 0.0;
    double last_offered_arrival_ns = 0.0;
    double last_user_completion_ns = 0.0;
    double quiescent_finish_ns = 0.0;
    double offered_arrival_span_ns = 0.0;
    double post_offer_user_completion_tail_ns = 0.0;
    double user_completion_span_ns = 0.0;
    double drain_tail_ns = 0.0;
    double makespan_ns = 0.0;
};

struct LatencyWorkSnapshot {
    std::uint64_t user_count = 0;
    double service_to_user_completion_sum_work_ns = 0.0;
    double offered_to_user_completion_sum_work_ns = 0.0;
    double source_to_user_completion_sum_work_ns = 0.0;
    // Primary online latency distribution: offered arrival through parent
    // completion, including front-end transaction-credit wait.
    double average_ns = 0.0;
    double p50_ns = 0.0;
    double p95_ns = 0.0;
    double max_ns = 0.0;
    double service_average_ns = 0.0;
    double service_p50_ns = 0.0;
    double service_p95_ns = 0.0;
    double service_max_ns = 0.0;
    double source_average_ns = 0.0;
    double source_p50_ns = 0.0;
    double source_p95_ns = 0.0;
    double source_max_ns = 0.0;
    std::uint64_t phase_barriers = 0;
    std::uint64_t phase_dependency_waited_ops = 0;
    double phase_dependency_wait_work_ns = 0.0;
    double phase_dependency_max_wait_ns = 0.0;
    std::uint64_t front_end_admission_waited_ops = 0;
    double front_end_admission_wait_work_ns = 0.0;
    double front_end_admission_max_wait_ns = 0.0;
};

struct BaseDieLinkStageWorkSnapshot {
    double read_queue_wait_work_ns = 0.0;
    double write_queue_wait_work_ns = 0.0;
    double read_serialization_work_ns = 0.0;
    double write_serialization_work_ns = 0.0;
    double read_fixed_latency_work_ns = 0.0;
    double write_fixed_latency_work_ns = 0.0;
};

struct LayerStreamingControllerStageWorkSnapshot {
    bool present = false;
    double backing_admission_wait_work_ns = 0.0;
    double backing_admission_max_wait_ns = 0.0;
    double user_wait_work_ns = 0.0;
    double user_max_wait_ns = 0.0;
    double exposed_prefetch_ns = 0.0;
    double hidden_prefetch_ns = 0.0;
    double buffer_reuse_wait_work_ns = 0.0;
};

struct CooperativeWriteControllerStageWorkSnapshot {
    bool present = false;
    std::uint64_t full_waited_ops = 0;
    double full_wait_work_ns = 0.0;
};

struct ResourceBusyMetric {
    double busy_ns = 0.0;
    std::uint64_t resource_count = 0;
    double active_span_ns = 0.0;
    double capacity_time_ns = 0.0;
    double utilization = 0.0;
};

struct ResourceBusySnapshot {
    ResourceBusyMetric hbm_data_bus;
    ResourceBusyMetric external_controller;
    ResourceBusyMetric external_media;
    ResourceBusyMetric external_link_m2s;
    ResourceBusyMetric external_link_s2m;
    ResourceBusyMetric hbf_logic_ingress;
    ResourceBusyMetric hbf_mapping_dram_issue;
    ResourceBusyMetric hbf_plane_media;
    ResourceBusyMetric hbf_media_lane;
    ResourceBusyMetric hbf_subarray;
    ResourceBusyMetric hbf_page_buffer_bank;
    ResourceBusyMetric hbf_flash_source_queue;
    ResourceBusyMetric hbf_channel_command;
    ResourceBusyMetric hbf_channel_data;
    ResourceBusyMetric hbf_tsv;
    ResourceBusyMetric hbf_sram;
    ResourceBusyMetric hbf_hbio_command;
    ResourceBusyMetric hbf_hbio_data;
    ResourceBusyMetric hbf_sequencer;
    ResourceBusyMetric hbf_ecc_issue;
    ResourceBusyMetric base_die_link_read;
    ResourceBusyMetric base_die_link_write;
};

struct TimeBreakdownSnapshot {
    WallClockSnapshot wall_clock;
    LatencyWorkSnapshot latency_work;
    Breakdown hbm_stage_work;
    Breakdown hbf_stage_work;
    Breakdown external_backing_stage_work;
    BaseDieLinkStageWorkSnapshot base_die_link_stage_work;
    LayerStreamingControllerStageWorkSnapshot
        layer_streaming_controller_stage_work;
    CooperativeWriteControllerStageWorkSnapshot
        cooperative_write_controller_stage_work;
    ResourceBusySnapshot resource_busy;
};

double sum_latency_work_ns(const std::vector<double>& values) {
    long double total = 0.0L;
    for (const auto value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("cannot sum a non-finite or negative latency");
        }
        total += static_cast<long double>(value);
    }
    if (total > static_cast<long double>(std::numeric_limits<double>::max())) {
        throw std::runtime_error("latency work exceeds finite double range");
    }
    return static_cast<double>(total);
}

LatencyDistribution finalize_latency_samples(
    std::vector<double>& samples,
    std::uint64_t expected_count,
    bool release_samples,
    std::string_view name) {
    if (samples.size() != expected_count) {
        throw std::runtime_error(
            "cannot finalize " + std::string(name) +
            " latency distribution with a count mismatch");
    }
    LatencyDistribution distribution;
    distribution.count = samples.size();
    // Accumulate once in extended precision, then derive the mean from the
    // reported sum.  Keeping two independent floating-point reductions made
    // sum/count diverge from the emitted average on long traces.
    distribution.sum_work_ns = sum_latency_work_ns(samples);
    distribution.average_ns = distribution.count == 0 ? 0.0 :
        distribution.sum_work_ns / static_cast<double>(distribution.count);
    std::sort(samples.begin(), samples.end());
    distribution.p50_ns = percentile_sorted(samples, 50.0);
    distribution.p95_ns = percentile_sorted(samples, 95.0);
    distribution.max_ns = percentile_sorted(samples, 100.0);
    if (release_samples) {
        std::vector<double>().swap(samples);
    }
    return distribution;
}

void finalize_latency_distributions(
    ScenarioResult& result,
    bool release_samples) {
    result.service_latency_distribution = finalize_latency_samples(
        result.service_latencies_ns,
        result.ops,
        release_samples,
        "service");
    result.offered_latency_distribution = finalize_latency_samples(
        result.offered_latencies_ns,
        result.ops,
        release_samples,
        "offered");
    result.source_latency_distribution = finalize_latency_samples(
        result.source_latencies_ns,
        result.ops,
        release_samples,
        "source");

    // The three distributions are the canonical per-parent observations.
    // Controller counters accumulate the same waits online, but subtracting
    // large, nearby timestamps per request and then reducing them in a
    // different order leaves a rounding bound that grows with the number of
    // additions. Preserve the online reductions as an independent audit,
    // using the standard gamma_n bound for sequential floating-point sums,
    // then publish the adjacent-distribution residuals so the reported
    // latency partition is exactly self-consistent.
    const auto reduction_tolerance = [count = result.ops](
        double lhs,
        double rhs,
        double reference_total,
        double reference_prefix) {
        const long double epsilon =
            std::numeric_limits<double>::epsilon();
        const long double n_epsilon =
            static_cast<long double>(std::max<std::uint64_t>(count, 1)) *
            epsilon;
        if (!(n_epsilon < 0.5L)) {
            throw std::runtime_error(
                "latency reduction is too large for a bounded double audit");
        }
        const long double gamma_n = n_epsilon / (1.0L - n_epsilon);
        const long double scale = std::max({
            1.0L,
            std::abs(static_cast<long double>(lhs)),
            std::abs(static_cast<long double>(rhs)),
            std::abs(static_cast<long double>(reference_total)),
            std::abs(static_cast<long double>(reference_prefix)),
        });
        // Covers the online sum, both independently reduced distributions,
        // and their element-wise timestamp subtractions. The resulting
        // tolerance remains O(n*epsilon): for this workload it is
        // parts-per-billion, while any missing nanosecond per request is
        // still rejected.
        return static_cast<double>(
            (8.0L * gamma_n + 64.0L * epsilon) * scale);
    };
    const auto canonical_residual = [&reduction_tolerance](
        double total,
        double prefix,
        std::string_view name) {
        double residual = total - prefix;
        const double tolerance =
            reduction_tolerance(residual, 0.0, total, prefix);
        if (residual < 0.0 && std::abs(residual) <= tolerance) {
            residual = 0.0;
        }
        if (!std::isfinite(residual) || residual < 0.0) {
            throw std::runtime_error(
                std::string(name) +
                " latency residual is non-finite or negative");
        }
        return residual;
    };
    const auto audit_online_reduction = [&reduction_tolerance](
        double online,
        double canonical,
        double reference_total,
        double reference_prefix,
        std::string_view name) {
        const double tolerance = reduction_tolerance(
            online,
            canonical,
            reference_total,
            reference_prefix);
        if (!std::isfinite(online) ||
            std::abs(online - canonical) > tolerance) {
            throw std::runtime_error(
                std::string(name) +
                " online wait reduction diverges from latency samples "
                "(difference=" +
                std::to_string(std::abs(online - canonical)) +
                " ns-work, tolerance=" +
                std::to_string(tolerance) + " ns-work)");
        }
    };
    const double canonical_admission_wait = canonical_residual(
        result.offered_latency_distribution->sum_work_ns,
        result.service_latency_distribution->sum_work_ns,
        "front-end admission");
    const double canonical_phase_wait = canonical_residual(
        result.source_latency_distribution->sum_work_ns,
        result.offered_latency_distribution->sum_work_ns,
        "phase dependency");
    audit_online_reduction(
        result.front_end_admission_wait_work_ns,
        canonical_admission_wait,
        result.offered_latency_distribution->sum_work_ns,
        result.service_latency_distribution->sum_work_ns,
        "front-end admission");
    audit_online_reduction(
        result.phase_dependency_wait_work_ns,
        canonical_phase_wait,
        result.source_latency_distribution->sum_work_ns,
        result.offered_latency_distribution->sum_work_ns,
        "phase dependency");
    result.front_end_admission_wait_work_ns =
        canonical_admission_wait;
    result.phase_dependency_wait_work_ns = canonical_phase_wait;
}

const LatencyDistribution& latency_distribution(const ScenarioResult& result) {
    if (!result.offered_latency_distribution) {
        throw std::runtime_error(
            "scenario offered latency distribution was not finalized before "
            "reporting");
    }
    return *result.offered_latency_distribution;
}

const LatencyDistribution& service_latency_distribution(
    const ScenarioResult& result) {
    if (!result.service_latency_distribution) {
        throw std::runtime_error(
            "scenario service latency distribution was not finalized before "
            "reporting");
    }
    return *result.service_latency_distribution;
}

const LatencyDistribution& source_latency_distribution(
    const ScenarioResult& result) {
    if (!result.source_latency_distribution) {
        throw std::runtime_error(
            "scenario source latency distribution was not finalized before "
            "reporting");
    }
    return *result.source_latency_distribution;
}

ResourceBusyMetric resource_busy_metric(
    double busy_ns,
    std::uint64_t resource_count,
    double active_span_ns) {
    if (!std::isfinite(busy_ns) || busy_ns < 0.0 ||
        !std::isfinite(active_span_ns) || active_span_ns < 0.0) {
        throw std::runtime_error("resource busy accounting is non-finite or negative");
    }
    const long double capacity = static_cast<long double>(resource_count) *
        static_cast<long double>(active_span_ns);
    if (capacity > static_cast<long double>(std::numeric_limits<double>::max())) {
        throw std::runtime_error("resource capacity-time exceeds finite double range");
    }
    ResourceBusyMetric metric;
    metric.busy_ns = busy_ns;
    metric.resource_count = resource_count;
    metric.active_span_ns = active_span_ns;
    metric.capacity_time_ns = static_cast<double>(capacity);
    metric.utilization = metric.capacity_time_ns <= 0.0 ? 0.0 :
        metric.busy_ns / metric.capacity_time_ns;
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, metric.busy_ns, metric.capacity_time_ns});
    if (metric.busy_ns > metric.capacity_time_ns + tolerance) {
        throw std::runtime_error(
            "resource busy work exceeds the reported capacity-time");
    }
    return metric;
}

bool time_close(double lhs, double rhs) {
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    // Latency work is reduced along independent paths: per-operation samples
    // are summed once for each distribution, while controller wait counters
    // are accumulated online. At 100M+ operations, their mathematically
    // equivalent totals can differ by hundreds of double ULPs solely from
    // addition order. 4096 ULP remains a sub-parts-per-trillion check and
    // still rejects any material missing stage.
    return std::abs(lhs - rhs) <=
        4096.0 * std::numeric_limits<double>::epsilon() * scale;
}

TimeBreakdownSnapshot make_time_breakdown(const ScenarioResult& result) {
    const auto& offered = latency_distribution(result);
    const auto& service = service_latency_distribution(result);
    const auto& source = source_latency_distribution(result);
    if (offered.count != result.ops ||
        service.count != result.ops ||
        source.count != result.ops) {
        throw std::runtime_error(
            "time breakdown latency counts do not match operation count");
    }
    if (!std::isfinite(result.first_arrival_ns) ||
        !std::isfinite(result.last_arrival_ns) ||
        !std::isfinite(result.user_finish_ns) ||
        !std::isfinite(result.finish_ns) ||
        result.last_arrival_ns < result.first_arrival_ns ||
        result.user_finish_ns < result.last_arrival_ns ||
        result.finish_ns < result.user_finish_ns) {
        throw std::runtime_error(
            "time breakdown wall-clock frontiers violate causal order");
    }
    TimeBreakdownSnapshot out;
    auto& wall = out.wall_clock;
    wall.trace_origin_ns = result.first_arrival_ns;
    wall.last_offered_arrival_ns = result.last_arrival_ns;
    wall.last_user_completion_ns = result.user_finish_ns;
    wall.quiescent_finish_ns = result.finish_ns;
    wall.offered_arrival_span_ns =
        result.last_arrival_ns - result.first_arrival_ns;
    wall.post_offer_user_completion_tail_ns =
        result.user_finish_ns - result.last_arrival_ns;
    wall.user_completion_span_ns =
        result.user_finish_ns - result.first_arrival_ns;
    wall.drain_tail_ns = result.finish_ns - result.user_finish_ns;
    wall.makespan_ns = result.finish_ns - result.first_arrival_ns;
    if (!time_close(
            wall.offered_arrival_span_ns +
                wall.post_offer_user_completion_tail_ns,
            wall.user_completion_span_ns) ||
        !time_close(
            wall.user_completion_span_ns + wall.drain_tail_ns,
            wall.makespan_ns)) {
        throw std::runtime_error(
            "time breakdown wall-clock spans do not conserve makespan");
    }

    auto& latency = out.latency_work;
    latency.user_count = offered.count;
    latency.service_to_user_completion_sum_work_ns =
        service.sum_work_ns;
    latency.phase_barriers = result.phase_barriers;
    latency.phase_dependency_waited_ops =
        result.phase_dependency_waited_ops;
    latency.phase_dependency_wait_work_ns =
        result.phase_dependency_wait_work_ns;
    latency.phase_dependency_max_wait_ns =
        result.phase_dependency_max_wait_ns;
    latency.front_end_admission_waited_ops = result.front_end_admission_waited_ops;
    latency.front_end_admission_wait_work_ns =
        result.front_end_admission_wait_work_ns;
    latency.front_end_admission_max_wait_ns =
        result.front_end_admission_max_wait_ns;
    latency.offered_to_user_completion_sum_work_ns =
        offered.sum_work_ns;
    latency.source_to_user_completion_sum_work_ns =
        source.sum_work_ns;
    latency.average_ns = offered.average_ns;
    latency.p50_ns = offered.p50_ns;
    latency.p95_ns = offered.p95_ns;
    latency.max_ns = offered.max_ns;
    latency.service_average_ns = service.average_ns;
    latency.service_p50_ns = service.p50_ns;
    latency.service_p95_ns = service.p95_ns;
    latency.service_max_ns = service.max_ns;
    latency.source_average_ns = source.average_ns;
    latency.source_p50_ns = source.p50_ns;
    latency.source_p95_ns = source.p95_ns;
    latency.source_max_ns = source.max_ns;

    if (!time_close(
            latency.service_to_user_completion_sum_work_ns +
                latency.front_end_admission_wait_work_ns,
            latency.offered_to_user_completion_sum_work_ns) ||
        !time_close(
            latency.offered_to_user_completion_sum_work_ns +
                latency.phase_dependency_wait_work_ns,
            latency.source_to_user_completion_sum_work_ns)) {
        throw std::runtime_error(
            "latency distributions do not conserve service, front-end, and "
            "phase wait work");
    }

    if (result.has_hbm) {
        out.hbm_stage_work = result.hbm_stats.stage_work;
    }
    if (result.has_hbf) {
        out.hbf_stage_work = result.hbf_stats.stage_work;
    }
    if (result.has_external_backing) {
        out.external_backing_stage_work = result.external_backing_stats.stage_work;
    }
    const auto& link = result.base_die_link_stats;
    out.base_die_link_stage_work = BaseDieLinkStageWorkSnapshot{
        .read_queue_wait_work_ns = link.read_queue_wait_ns,
        .write_queue_wait_work_ns = link.write_queue_wait_ns,
        .read_serialization_work_ns = link.read_busy_ns,
        .write_serialization_work_ns = link.write_busy_ns,
        .read_fixed_latency_work_ns = link.read_fixed_latency_work_ns,
        .write_fixed_latency_work_ns = link.write_fixed_latency_work_ns,
    };
    auto& streaming_work = out.layer_streaming_controller_stage_work;
    if (result.layer_streaming_stats) {
        const auto& stats = *result.layer_streaming_stats;
        streaming_work.present = true;
        streaming_work.backing_admission_wait_work_ns =
            stats.backing_admission_wait_work_ns;
        streaming_work.backing_admission_max_wait_ns =
            stats.backing_admission_max_wait_ns;
        streaming_work.user_wait_work_ns = stats.user_wait_work_ns;
        streaming_work.user_max_wait_ns = stats.user_max_wait_ns;
        streaming_work.exposed_prefetch_ns = stats.exposed_prefetch_ns;
        streaming_work.hidden_prefetch_ns = stats.hidden_prefetch_ns;
        streaming_work.buffer_reuse_wait_work_ns =
            stats.buffer_reuse_wait_work_ns;
    }
    out.cooperative_write_controller_stage_work =
        CooperativeWriteControllerStageWorkSnapshot{
            .present = result.cooperative_write_controller_present,
            .full_waited_ops = result.hbm_write_buffer_full_waits,
            .full_wait_work_ns = result.hbm_write_buffer_wait_ns,
        };

    auto& busy = out.resource_busy;
    if (result.has_hbm) {
        const auto span = result.hbm_stats.active_span_ns();
        busy.hbm_data_bus = resource_busy_metric(
            result.hbm_stats.bus_busy_ns, result.hbm_stats.pseudo_channels, span);
    }
    if (result.has_hbf) {
        const auto& stats = result.hbf_stats;
        const auto span = stats.active_span_ns();
        busy.hbf_logic_ingress = resource_busy_metric(
            stats.logic_ingress_busy_ns, stats.logic_ingress_resources, span);
        busy.hbf_mapping_dram_issue = resource_busy_metric(
            stats.mapping_dram_issue_busy_ns,
            stats.mapping_dram_resources,
            span);
        busy.hbf_plane_media = resource_busy_metric(stats.media_busy_ns, stats.planes, span);
        busy.hbf_media_lane = resource_busy_metric(
            stats.read_lane_busy_ns, stats.media_lanes, span);
        busy.hbf_subarray = resource_busy_metric(
            stats.subarray_read_busy_ns, stats.subarrays, span);
        busy.hbf_page_buffer_bank = resource_busy_metric(
            stats.page_buffer_bank_busy_ns, stats.page_buffer_banks, span);
        busy.hbf_flash_source_queue = resource_busy_metric(
            stats.flash_source_queue_busy_ns,
            stats.flash_source_queue_resources,
            span);
        busy.hbf_channel_command = resource_busy_metric(
            stats.channel_command_busy_ns, stats.channel_command_resources, span);
        busy.hbf_channel_data = resource_busy_metric(
            stats.channel_data_busy_ns, stats.channel_data_resources, span);
        busy.hbf_tsv = resource_busy_metric(
            stats.tsv_busy_ns, stats.tsv_resources, span);
        busy.hbf_sram = resource_busy_metric(
            stats.sram_busy_ns, stats.sram_resources, span);
        busy.hbf_hbio_command = resource_busy_metric(
            stats.hb_io_command_busy_ns, stats.hbio_command_resources, span);
        busy.hbf_hbio_data = resource_busy_metric(
            stats.hb_io_data_busy_ns, stats.hbio_data_resources, span);
        busy.hbf_sequencer = resource_busy_metric(
            stats.sequencer_busy_ns, stats.dies, span);
        busy.hbf_ecc_issue = resource_busy_metric(
            stats.ecc_issue_busy_ns, stats.dies, span);
    }
    if (result.has_external_backing) {
        const auto& stats = result.external_backing_stats;
        busy.external_controller = resource_busy_metric(
            stats.controller_issue_busy_ns,
            1,
            stats.controller_active_span_ns());
        busy.external_media = resource_busy_metric(
            stats.media_read_busy_ns + stats.media_write_busy_ns +
                stats.cache_flush_busy_ns + stats.cache_prefetch_busy_ns,
            stats.media_channels,
            stats.media_active_span_ns());
        busy.external_link_m2s = resource_busy_metric(
            stats.m2s_busy_ns,
            1,
            stats.m2s_active_span_ns());
        busy.external_link_s2m = resource_busy_metric(
            stats.s2m_busy_ns,
            1,
            stats.s2m_active_span_ns());
    }
    const auto link_span = link.active_span_ns();
    busy.base_die_link_read = resource_busy_metric(
        link.read_busy_ns, link.links, link_span);
    busy.base_die_link_write = resource_busy_metric(
        link.write_busy_ns, link.links, link_span);
    return out;
}

double user_completion_throughput_GBps(const ScenarioResult& result) {
    const auto elapsed_ns = result.user_finish_ns - result.first_arrival_ns;
    return elapsed_ns <= 0.0 ? 0.0 :
        static_cast<double>(result.logical_bytes) / elapsed_ns;
}

double makespan_throughput_GBps(const ScenarioResult& result) {
    const auto elapsed_ns = result.finish_ns - result.first_arrival_ns;
    return elapsed_ns <= 0.0 ? 0.0 :
        static_cast<double>(result.logical_bytes) / elapsed_ns;
}

void print_run_config(
    const Options& options,
    const SystemConfig& system,
    std::size_t op_count,
    std::uint64_t trace_footprint_bytes) {
    const auto& hbm = system.hbm;
    const auto& hbf = system.hbf;
    const auto& external = system.external;
    const auto hbf_bytes = hbf_capacity_bytes(hbf);
    const auto ratio = static_cast<long double>(hbf_bytes) /
        static_cast<long double>(hbm.device.capacity_bytes);
    const auto hbf_codeword_bytes = hbf.device.page_size_bytes + hbf.device.oob_bytes_per_page;
    const double ecc_decode_ii_ns = hbfsim::physical::transfer_time_ns(
        hbf_codeword_bytes, hbf.device.ecc_decode_raw_bandwidth_GBps_per_die);
    const double ecc_encode_ii_ns = hbfsim::physical::transfer_time_ns(
        hbf_codeword_bytes, hbf.device.ecc_encode_raw_bandwidth_GBps_per_die);
    std::cout << "config:"
              << " ops=" << op_count
              << " line_size=" << options.line_size
              << " trace_unique_line_bytes=" << trace_footprint_bytes
              << " hbm_capacity=" << format_bytes(hbm.device.capacity_bytes)
              << " hbf_capacity=" << format_bytes(hbf_bytes)
              << " hbf_to_hbm=" << fixed(static_cast<double>(ratio), 3) << "x"
              << " flat_hbm_bytes=" << options.flat_hbm_bytes
              << " cooperative_write_region_base="
              << hbm_cooperative_region_base_addr(options, system)
              << " cooperative_write_region_bytes="
              << options.hbf_hbm_write_buffer_bytes
              << " cooperative_write_destage_watermark="
              << fixed(options.hbf_hbm_write_buffer_destage_watermark, 3)
              << " layer_buffer_bytes=" << options.layer_buffer_bytes
              << " explicit_residency_contract="
              << (options.explicit_residency_contract ? "true" : "false")
              << " behavioral_hbm_bytes="
              << (options.behavioral_hbm_bytes == 0 ?
                  hbm.device.capacity_bytes :
                  options.behavioral_hbm_bytes)
              << " behavioral_promotion_threshold="
              << options.behavioral_promotion_threshold
              << " behavioral_history_pages="
              << options.behavioral_history_pages
              << " max_outstanding_requests="
              << options.max_outstanding_requests
              << " max_hbm_outstanding_requests="
              << options.max_hbm_outstanding_requests
              << " max_hbf_outstanding_requests="
              << options.max_hbf_outstanding_requests
              << " arrival_model="
              << (open_loop_arrivals(options) ? "open-loop" : "closed-loop")
              << " hbm_issue_mode=" << kHbmIssueMode
              << " address_heatmap_bins=" << options.address_heatmap_bins
              << " base_die_link_read_bw="
              << fixed(system.base_die_link.read_bandwidth_GBps)
              << " base_die_link_write_bw="
              << fixed(system.base_die_link.write_bandwidth_GBps)
              << " base_die_link_latency_ns="
              << fixed(system.base_die_link.latency_ns)
              << " external_backing_kind="
              << hbfsim::physical::external::to_string(external.kind)
              << " external_backing_capacity="
              << format_bytes(external.capacity_bytes)
              << " external_backing_page_size="
              << external.page_size_bytes
              << " external_backing_request_segment_bytes="
              << external.request_segment_bytes
              << " external_backing_media_channels="
              << external.media_channels
              << " external_backing_media_read_queues="
              << external.media_read_queues
              << " external_backing_media_write_queues="
              << external.media_write_queues
              << " external_backing_max_outstanding_requests="
              << external.max_outstanding_requests
              << " external_backing_controller_issue_ns="
              << fixed(external.controller_issue_ns)
              << " external_backing_controller_processing_ns="
              << fixed(external.controller_processing_ns)
              << " external_backing_media_read_latency_ns="
              << fixed(external.media_read_latency_ns)
              << " external_backing_media_write_latency_ns="
              << fixed(external.media_write_latency_ns)
              << " external_backing_media_read_bw="
              << fixed(external.media_read_bandwidth_GBps)
              << " external_backing_media_write_bw="
              << fixed(external.media_write_bandwidth_GBps)
              << " external_backing_m2s_bw="
              << fixed(external.m2s_bandwidth_GBps)
              << " external_backing_s2m_bw="
              << fixed(external.s2m_bandwidth_GBps)
              << " external_backing_one_way_propagation_ns="
              << fixed(external.one_way_propagation_ns)
              << " external_backing_command_bytes="
              << external.command_bytes
              << " external_backing_completion_bytes="
              << external.completion_bytes;
    if (options.generate_semantic_llm_path) {
        std::cout << " llm_weight_base=0x" << std::hex << options.llm_weight_base
                  << " llm_kv_base=0x" << options.llm_kv_base
                  << " llm_scratch_base=0x" << options.llm_scratch_base << std::dec;
    }
    std::cout << " hbm_stacks=" << hbm.device.stacks
              << " hbm_channels=" << hbm.device.channels_per_stack
              << " hbm_pseudo_channels=" << hbm.device.pseudo_channels_per_channel
              << " hbm_bank_groups_per_pseudo_channel="
              << hbm.device.bank_groups_per_pseudo_channel
              << " hbm_address_mapping_scheme=" << hbm.address_mapping_scheme()
              << " hbm_channel_width_bits=" << hbm.device.channel_width_bits
              << " hbm_pseudo_channel_width_bits="
              << hbm.pseudo_channel_width_bits()
              << " hbm_pin_rate_Gbps=" << fixed(hbm.device.pin_rate_Gbps, 3)
              << " hbm_command_clock_MHz=" << fixed(hbm.command_clock_MHz(), 3)
              << " hbm_tck_ns=" << fixed(hbm.command_clock_period_ns(), 6)
              << " hbm_burst_length=" << hbm.device.burst_length
              << " hbm_burst_bytes=" << hbm.burst_bytes()
              << " hbm_burst_ns=" << fixed(hbm.burst_duration_ns(), 6)
              << " hbm_channel_bw_GBps="
              << fixed(hbm.channel_bandwidth_GBps(), 3)
              << " hbm_stack_peak_bw_GBps="
              << fixed(hbm.channel_bandwidth_GBps() * hbm.device.channels_per_stack, 3)
              << " hbm_system_peak_bw_GBps="
              << fixed(hbm_peak_rate_GBps(hbm), 3)
              << " hbm_timing_model=" << hbm.timing_model()
              << " hbf_processor_interconnect="
              << system.hbf_processor_interconnect
              << " hbf_placement_mapping_scheme="
              << hbfsim::host::kPlacementMappingScheme
              << " hbf_stacks=" << hbf.device.stacks
              << " hbf_channels=" << hbf.device.channels_per_stack
              << " hbf_page_size=" << hbf.device.page_size_bytes
              << " hbf_oob_bytes=" << hbf.device.oob_bytes_per_page
              << " hbf_media_lanes_per_plane=" << hbf.device.media_lanes_per_plane
              << " hbf_subarrays_per_plane=" << 1u
              << " hbf_page_buffer_banks_per_plane="
              << hbf.device.page_buffer_banks_per_plane
              << " hbf_read_ns=" << fixed(hbf.device.t_read_page_ns)
              << " hbf_program_ns=" << fixed(hbf.device.t_program_page_ns)
              << " hbf_ecc_decode_latency_ns="
              << fixed(hbf.device.ecc_decode_latency_ns)
              << " hbf_ecc_encode_latency_ns="
              << fixed(hbf.device.ecc_encode_latency_ns)
              << " hbf_ecc_decode_raw_bw_per_die="
              << fixed(hbf.device.ecc_decode_raw_bandwidth_GBps_per_die, 6)
              << " hbf_ecc_encode_raw_bw_per_die="
              << fixed(hbf.device.ecc_encode_raw_bandwidth_GBps_per_die, 6)
              << " hbf_ecc_codeword_size_bytes=" << hbf_codeword_bytes
              << " hbf_ecc_decode_ii_ns=" << fixed(ecc_decode_ii_ns, 6)
              << " hbf_ecc_encode_ii_ns=" << fixed(ecc_encode_ii_ns, 6)
              << " hbf_ecc_issue_topology=shared-per-die"
              << " hbf_hbio_bw=" << fixed(hbf.device.hb_io_bandwidth_GBps())
              << " hbf_system_peak_bw_GBps="
              << fixed(hbf_peak_rate_GBps(hbf), 3)
              << " hbf_media_lane_bw=" << fixed(hbf.device.media_lane_bandwidth_GBps)
              << " hbf_logic_sram_bw=" << fixed(hbf.device.logic_sram_bandwidth_GBps)
              << " hbf_page_buffer_bw=" << fixed(hbf.device.page_buffer_bandwidth_GBps)
              << " hbf_write_coalescing="
              << (hbf.host.write_coalescing_enabled ? "yes" : "no")
              << " hbf_write_buffer_pages=" << hbf.host.write_buffer_pages
              << " hbf_write_buffer_flush_threshold="
              << hbf.host.write_buffer_flush_threshold_pages
              << " hbf_gc_reserved_blocks_per_plane="
              << hbf.host.gc_reserved_free_blocks_per_plane
              << " hbf_gc_wear_weight=" << fixed(hbf.host.gc_wear_leveling_weight)
              << " hbf_static_wl_gap=" << hbf.host.static_wear_leveling_erase_gap
              << " hbf_static_wl_interval="
              << hbf.host.static_wear_leveling_interval_erases
              << " hbf_static_wl_start="
              << hbf.host.static_wear_leveling_start_erases
              << " hbf_static_wl_stop_gap="
              << hbf.host.static_wear_leveling_stop_gap
              << " hbf_static_wl_cooldown_erases="
              << hbf.host.static_wear_leveling_cooldown_erases
              << " hbf_static_wl_max_write_fraction="
              << hbf.host.static_wear_leveling_max_write_fraction
              << '\n';
}

// ---- Comparison-table schema -----------------------------------------
// ONE list defines the result table: header, console cell, CSV cell. Both
// surfaces render the same columns in the same order by construction, so
// they cannot drift apart again. Where the two spellings legitimately
// differ today (percent-scaled+"%" on the console vs raw fraction in the
// CSV; "n/a" vs empty for a missing device), the difference is declared
// per column HERE — not implied by two hand-maintained emitters 300 lines
// apart. Adding a column is one entry.
std::string csv_escape(const std::string& value);

struct ResultCell {
    std::string console;
    std::string csv;
};

ResultCell cell_same(std::string value) {
    return {value, value};
}

ResultCell cell_count(std::uint64_t value) {
    return cell_same(std::to_string(value));
}

ResultCell cell_bool(bool value) {
    return cell_same(value ? "true" : "false");
}

ResultCell cell_fixed(double value) {
    return cell_same(fixed(value));
}

ResultCell cell_percent(double fraction) {
    return {fixed(fraction * 100.0) + "%", fixed(fraction)};
}

ResultCell cell_missing() {
    return {"n/a", ""};
}

ResultCell cell_optional_fixed(std::optional<double> value, int precision = 3) {
    return value ? cell_same(fixed(*value, precision)) : cell_missing();
}

ResultCell cell_resource_utilization(
    double busy_ns,
    std::uint64_t resource_count,
    double active_span_ns) {
    return cell_percent(
        resource_busy_metric(busy_ns, resource_count, active_span_ns).utilization);
}

struct ResultColumn {
    const char* header;
    std::function<ResultCell(const ScenarioResult&)> cell;
};

// Gate a cell on device presence (console "n/a", CSV empty when absent).
template <typename Fn>
std::function<ResultCell(const ScenarioResult&)> hbm_cell(Fn fn) {
    return [fn](const ScenarioResult& r) {
        return r.has_hbm ? fn(r) : cell_missing();
    };
}

template <typename Fn>
std::function<ResultCell(const ScenarioResult&)> hbf_cell(Fn fn) {
    return [fn](const ScenarioResult& r) {
        return r.has_hbf ? fn(r) : cell_missing();
    };
}

template <typename Fn>
std::function<ResultCell(const ScenarioResult&)> external_backing_cell(Fn fn) {
    return [fn](const ScenarioResult& r) {
        return r.has_external_backing ? fn(r) : cell_missing();
    };
}

template <typename Fn>
std::function<ResultCell(const ScenarioResult&)> layer_streaming_cell(Fn fn) {
    return [fn](const ScenarioResult& r) {
        return r.layer_streaming_stats ?
            fn(*r.layer_streaming_stats) : cell_missing();
    };
}

template <typename Fn>
std::function<ResultCell(const ScenarioResult&)> behavioral_tiering_cell(
    Fn fn) {
    return [fn](const ScenarioResult& r) {
        return r.behavioral_tiering_stats ?
            fn(*r.behavioral_tiering_stats) : cell_missing();
    };
}

std::string ratio_text(std::uint64_t active, std::uint64_t total) {
    return std::to_string(active) + "/" + std::to_string(total);
}

const char* layer_streaming_mode_name(const LayerStreamingStats& stats) {
    (void)stats;
    return "layer_streaming";
}

const char* layer_streaming_residency_policy_name(
    const LayerStreamingStats& stats) {
    return stats.explicit_residency_contract ?
        "capacity-aware-static-weight-prefix-v2" :
        "trace-derived-first-touch";
}

const std::vector<ResultColumn>& result_columns() {
    using R = const ScenarioResult&;
    static const std::vector<ResultColumn> columns = {
        {"scenario", [](R r) { return ResultCell{r.name, csv_escape(r.name)}; }},
        {"summary_schema_version", [](R) { return cell_count(19); }},
        {"ops", [](R r) { return cell_count(r.ops); }},
        {"reads", [](R r) { return cell_count(r.reads); }},
        {"writes", [](R r) { return cell_count(r.writes); }},
        {"hbm_accesses", [](R r) { return cell_count(r.hbm_accesses); }},
        {"hbf_accesses", [](R r) { return cell_count(r.hbf_accesses); }},
        {"external_accesses", [](R r) { return cell_count(r.external_accesses); }},
        {"hbm_user_accesses", [](R r) { return cell_count(r.hbm_user_accesses); }},
        {"hbf_user_accesses", [](R r) { return cell_count(r.hbf_user_accesses); }},
        {"hbf_direct_user_ops", [](R r) { return cell_count(r.hbf_direct_user_ops); }},
        {"hbm_background_accesses", [](R r) { return cell_count(r.hbm_background_accesses); }},
        {"hbf_background_accesses", [](R r) { return cell_count(r.hbf_background_accesses); }},
        {"external_background_accesses", [](R r) {
            return cell_count(r.external_background_accesses);
        }},
        {"behavioral_policy", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_same(
                hbfsim::policy::to_string(
                    s.admission_policy));
        })},
        {"behavioral_semantic_inputs_consumed", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_same(
                s.semantic_inputs_consumed ? "true" : "false");
        })},
        {"behavioral_hbm_tier_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_tier_pages);
        })},
        {"behavioral_hbm_tier_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_tier_bytes);
        })},
        {"behavioral_promotion_threshold", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.promotion_threshold);
        })},
        {"behavioral_history_capacity_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.history_capacity_pages);
        })},
        {"behavioral_page_observations", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.page_observations);
        })},
        {"behavioral_hbm_hits", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_hits);
        })},
        {"behavioral_hbf_bypasses", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbf_bypasses);
        })},
        {"behavioral_cold_misses", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.cold_misses);
        })},
        {"behavioral_history_hits", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.history_hits);
        })},
        {"behavioral_promotions", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.promotions);
        })},
        {"behavioral_promotions_with_backing_fill",
         behavioral_tiering_cell([](const BehavioralTieringStats& s) {
            return cell_count(s.promotions_with_backing_fill);
        })},
        {"behavioral_promotions_without_backing_fill",
         behavioral_tiering_cell([](const BehavioralTieringStats& s) {
            return cell_count(s.promotions_without_backing_fill);
        })},
        {"behavioral_clean_evictions", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.clean_evictions);
        })},
        {"behavioral_dirty_evictions", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.dirty_evictions);
        })},
        {"behavioral_history_evictions", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.history_evictions);
        })},
        {"behavioral_hbm_foreground_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_foreground_bytes);
        })},
        {"behavioral_hbf_bypass_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbf_bypass_bytes);
        })},
        {"behavioral_backing_fill_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.backing_fill_pages);
        })},
        {"behavioral_backing_fill_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.backing_fill_bytes);
        })},
        {"behavioral_hbm_install_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_install_pages);
        })},
        {"behavioral_hbm_install_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.hbm_install_bytes);
        })},
        {"behavioral_dirty_writeback_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.dirty_writeback_pages);
        })},
        {"behavioral_dirty_writeback_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.dirty_writeback_bytes);
        })},
        {"behavioral_drain_writeback_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.drain_writeback_pages);
        })},
        {"behavioral_drain_writeback_bytes", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.drain_writeback_bytes);
        })},
        {"behavioral_peak_resident_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.peak_resident_pages);
        })},
        {"behavioral_final_resident_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.final_resident_pages);
        })},
        {"behavioral_final_dirty_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.final_dirty_pages);
        })},
        {"behavioral_peak_history_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.peak_history_pages);
        })},
        {"behavioral_peak_installing_pages", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.peak_installing_pages);
        })},
        {"behavioral_capacity_stalled_promotions",
         behavioral_tiering_cell([](const BehavioralTieringStats& s) {
            return cell_count(s.capacity_stalled_promotions);
        })},
        {"behavioral_capacity_stall_work_ns", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_fixed(s.capacity_stall_work_ns);
        })},
        {"behavioral_transition_waited_transactions",
         behavioral_tiering_cell([](const BehavioralTieringStats& s) {
            return cell_count(s.transition_waited_transactions);
        })},
        {"behavioral_transition_wait_work_ns", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_fixed(s.transition_wait_work_ns);
        })},
        {"behavioral_transition_max_wait_ns", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_fixed(s.transition_max_wait_ns);
        })},
        {"behavioral_decision_fingerprint", behavioral_tiering_cell([](
            const BehavioralTieringStats& s) {
            return cell_count(s.decision_fingerprint);
        })},
        {"layer_streaming_mode", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(layer_streaming_mode_name(s));
        })},
        {"layer_streaming_backing", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(hbfsim::policy::to_string(s.backing));
        })},
        {"layer_streaming_residency_policy", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(layer_streaming_residency_policy_name(s));
        })},
        {"layer_streaming_compact_resident_mapping", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(s.compact_resident_mapping ? "true" : "false");
        })},
        {"layer_streaming_semantic_inputs_consumed", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(s.semantic_inputs_consumed ? "true" : "false");
        })},
        {"layer_streaming_explicit_residency_contract", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_same(s.explicit_residency_contract ? "true" : "false");
        })},
        {"layer_streaming_address_footprint_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.address_footprint_bytes);
        })},
        {"layer_streaming_unique_resident_footprint_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unique_resident_footprint_pages);
        })},
        {"layer_streaming_unique_resident_footprint_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unique_resident_footprint_bytes);
        })},
        {"layer_streaming_capacity_pressure_basis_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.capacity_pressure_basis_bytes);
        })},
        {"layer_streaming_footprint_page_rounding_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.footprint_page_rounding_bytes);
        })},
        {"layer_streaming_hbm_capacity_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hbm_capacity_bytes);
        })},
        {"layer_streaming_hbm_capacity_pressure", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.hbm_capacity_pressure);
        })},
        {"layer_streaming_layers", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.layers);
        })},
        {"layer_streaming_explicit_layer_requests", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.explicit_layer_requests);
        })},
        {"layer_streaming_explicit_compute_layers", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.explicit_compute_layers);
        })},
        {"layer_streaming_compute_work_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.compute_work_ns);
        })},
        {"layer_streaming_hbm_only_resident_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hbm_only_resident_pages);
        })},
        {"layer_streaming_hbm_only_resident_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hbm_only_resident_bytes);
        })},
        {"layer_streaming_hot_kv_candidate_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hot_kv_candidate_pages);
        })},
        {"layer_streaming_hot_kv_candidate_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hot_kv_candidate_bytes);
        })},
        {"layer_streaming_hot_kv_resident_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hot_kv_resident_pages);
        })},
        {"layer_streaming_hot_kv_resident_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hot_kv_resident_bytes);
        })},
        {"layer_streaming_data_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.data_pages);
        })},
        {"layer_streaming_data_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.data_bytes);
        })},
        {"layer_streaming_model_weight_resident_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.model_weight_resident_pages);
        })},
        {"layer_streaming_model_weight_resident_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.model_weight_resident_bytes);
        })},
        {"layer_streaming_model_weight_backing_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.model_weight_backing_pages);
        })},
        {"layer_streaming_model_weight_backing_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.model_weight_backing_bytes);
        })},
        {"layer_streaming_cold_kv_backing_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.cold_kv_backing_pages);
        })},
        {"layer_streaming_cold_kv_backing_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.cold_kv_backing_bytes);
        })},
        {"layer_streaming_unknown_backing_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unknown_backing_pages);
        })},
        {"layer_streaming_unknown_backing_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unknown_backing_bytes);
        })},
        {"layer_streaming_backing_unique_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.backing_unique_pages);
        })},
        {"layer_streaming_backing_unique_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.backing_unique_bytes);
        })},
        {"layer_streaming_resident_physical_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.resident_physical_pages);
        })},
        {"layer_streaming_resident_physical_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.resident_physical_bytes);
        })},
        {"layer_streaming_effective_layer_buffer_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.effective_layer_buffer_pages);
        })},
        {"layer_streaming_effective_layer_buffer_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.effective_layer_buffer_bytes);
        })},
        {"layer_streaming_unused_hbm_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unused_hbm_pages);
        })},
        {"layer_streaming_unused_hbm_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.unused_hbm_bytes);
        })},
        {"layer_streaming_streamed_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.streamed_pages);
        })},
        {"layer_streaming_streamed_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.streamed_bytes);
        })},
        {"layer_streaming_foreground_resident_page_accesses", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.foreground_resident_page_accesses);
        })},
        {"layer_streaming_foreground_buffer_page_accesses", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.foreground_buffer_page_accesses);
        })},
        {"layer_streaming_dirty_pages_written_back", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.dirty_pages_written_back);
        })},
        {"layer_streaming_writeback_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.writeback_bytes);
        })},
        {"layer_streaming_max_layer_data_pages", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.max_layer_data_pages);
        })},
        {"layer_streaming_max_layer_data_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.max_layer_data_bytes);
        })},
        {"layer_streaming_immutable_weight_logical_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.immutable_weight_logical_bytes);
        })},
        {"layer_streaming_runtime_overhead_logical_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.runtime_overhead_logical_bytes);
        })},
        {"layer_streaming_block_table_logical_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.block_table_logical_bytes);
        })},
        {"layer_streaming_active_buffer_logical_bytes_per_slot", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.active_buffer_logical_bytes_per_slot);
        })},
        {"layer_streaming_residency_page_size_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.residency_page_size_bytes);
        })},
        {"layer_streaming_kv_block_stride_bytes", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.kv_block_stride_bytes);
        })},
        {"layer_streaming_logical_kv_blocks", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.logical_kv_blocks);
        })},
        {"layer_streaming_hot_kv_blocks", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.hot_kv_blocks);
        })},
        {"layer_streaming_cold_kv_blocks", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.cold_kv_blocks);
        })},
        {"layer_streaming_backing_request_credit_limit", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.backing_request_credit_limit);
        })},
        {"layer_streaming_backing_max_inflight_requests", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.backing_max_inflight_requests);
        })},
        {"layer_streaming_backing_admission_waited_requests", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.backing_admission_waited_requests);
        })},
        {"layer_streaming_backing_admission_wait_work_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.backing_admission_wait_work_ns);
        })},
        {"layer_streaming_backing_admission_max_wait_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.backing_admission_max_wait_ns);
        })},
        {"layer_streaming_user_waited_ops", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_count(s.user_waited_ops);
        })},
        {"layer_streaming_user_wait_work_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.user_wait_work_ns);
        })},
        {"layer_streaming_user_max_wait_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.user_max_wait_ns);
        })},
        {"layer_streaming_exposed_prefetch_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.exposed_prefetch_ns);
        })},
        {"layer_streaming_hidden_prefetch_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.hidden_prefetch_ns);
        })},
        {"layer_streaming_buffer_reuse_wait_work_ns", layer_streaming_cell([](const LayerStreamingStats& s) {
            return cell_fixed(s.buffer_reuse_wait_work_ns);
        })},
        {"background_hbf_writes", [](R r) { return cell_count(r.background_hbf_writes); }},
        {"hbf_static_read_bytes", [](R r) { return cell_count(r.hbf_static_read_bytes); }},
        {"hbf_logical_read_bytes", [](R r) { return cell_count(r.hbf_logical_read_bytes); }},
        {"hbf_backing_write_bytes", [](R r) { return cell_count(r.hbf_backing_write_bytes); }},
        {"external_backing_read_bytes", [](R r) {
            return cell_count(r.external_backing_read_bytes);
        }},
        {"external_backing_write_bytes", [](R r) {
            return cell_count(r.external_backing_write_bytes);
        }},
        {"hbm_foreground_bytes", [](R r) { return cell_count(r.hbm_foreground_bytes); }},
        {"hbm_streaming_write_bytes", [](R r) { return cell_count(r.hbm_streaming_write_bytes); }},
        {"base_die_link_read_bytes", [](R r) {
            return cell_count(r.base_die_link_stats.read_bytes);
        }},
        {"base_die_link_write_bytes", [](R r) {
            return cell_count(r.base_die_link_stats.write_bytes);
        }},
        {"base_die_link_read_queue_wait_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.read_queue_wait_ns);
        }},
        {"base_die_link_write_queue_wait_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.write_queue_wait_ns);
        }},
        {"base_die_link_read_serialization_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.read_busy_ns);
        }},
        {"base_die_link_write_serialization_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.write_busy_ns);
        }},
        {"base_die_link_read_fixed_latency_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.read_fixed_latency_work_ns);
        }},
        {"base_die_link_write_fixed_latency_work_ns", [](R r) {
            return cell_fixed(r.base_die_link_stats.write_fixed_latency_work_ns);
        }},
        {"external_backing_kind", external_backing_cell([](R r) {
            return cell_same(
                hbfsim::physical::external::to_string(
                    r.external_backing_stats.kind));
        })},
        {"external_read_requests", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.read_requests);
        })},
        {"external_write_requests", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.write_requests);
        })},
        {"external_read_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.read_bytes);
        })},
        {"external_write_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.write_bytes);
        })},
        {"external_range_requests", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.page_run_requests);
        })},
        {"external_transport_requests", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.page_run_segments);
        })},
        {"external_address_pages", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.page_run_pages);
        })},
        {"external_outstanding_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.outstanding_wait_ns);
        })},
        {"external_controller_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.controller_queue_wait_ns);
        })},
        {"external_controller_processing_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.controller_processing_work_ns);
        })},
        {"external_media_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.media_queue_wait_ns);
        })},
        {"external_link_m2s_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.m2s_queue_wait_ns);
        })},
        {"external_link_s2m_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.s2m_queue_wait_ns);
        })},
        {"external_media_read_latency_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.media_read_latency_work_ns);
        })},
        {"external_media_write_latency_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.media_write_latency_work_ns);
        })},
        {"external_transport_propagation_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.transport_propagation_work_ns);
        })},
        {"external_m2s_payload_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.m2s_payload_bytes);
        })},
        {"external_m2s_protocol_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.m2s_protocol_bytes);
        })},
        {"external_m2s_wire_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.m2s_wire_bytes);
        })},
        {"external_s2m_payload_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.s2m_payload_bytes);
        })},
        {"external_s2m_protocol_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.s2m_protocol_bytes);
        })},
        {"external_s2m_wire_bytes", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.s2m_wire_bytes);
        })},
        {"external_media_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.media_read_busy_ns +
                r.external_backing_stats.media_write_busy_ns);
        })},
        {"external_media_read_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.media_read_busy_ns);
        })},
        {"external_media_write_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.media_write_busy_ns);
        })},
        {"external_media_resource_count", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.media_channels);
        })},
        {"external_media_channels_per_queue", external_backing_cell([](R r) {
            return cell_count(
                r.external_backing_stats.media_channels_per_queue);
        })},
        {"external_media_read_queues", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.media_read_queues);
        })},
        {"external_media_write_queues", external_backing_cell([](R r) {
            return cell_count(r.external_backing_stats.media_write_queues);
        })},
        {"external_media_active_span_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.media_active_span_ns());
        })},
        {"external_media_capacity_time_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.media_active_span_ns() *
                static_cast<double>(
                    r.external_backing_stats.media_channels));
        })},
        {"external_media_utilization", external_backing_cell([](R r) {
            return cell_percent(r.external_backing_stats.media_utilization());
        })},
        {"external_controller_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.controller_issue_busy_ns);
        })},
        {"external_controller_resource_count", external_backing_cell([](R) {
            return cell_count(1);
        })},
        {"external_controller_active_span_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.controller_active_span_ns());
        })},
        {"external_controller_capacity_time_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.controller_active_span_ns());
        })},
        {"external_controller_utilization", external_backing_cell([](R r) {
            return cell_percent(
                r.external_backing_stats.controller_utilization());
        })},
        {"external_link_m2s_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.m2s_busy_ns);
        })},
        {"external_link_m2s_resource_count", external_backing_cell([](R) {
            return cell_count(1);
        })},
        {"external_link_m2s_active_span_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.m2s_active_span_ns());
        })},
        {"external_link_m2s_capacity_time_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.m2s_active_span_ns());
        })},
        {"external_link_m2s_utilization", external_backing_cell([](R r) {
            return cell_percent(r.external_backing_stats.m2s_utilization());
        })},
        {"external_link_s2m_busy_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.s2m_busy_ns);
        })},
        {"external_link_s2m_resource_count", external_backing_cell([](R) {
            return cell_count(1);
        })},
        {"external_link_s2m_active_span_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.s2m_active_span_ns());
        })},
        {"external_link_s2m_capacity_time_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.s2m_active_span_ns());
        })},
        {"external_link_s2m_utilization", external_backing_cell([](R r) {
            return cell_percent(r.external_backing_stats.s2m_utilization());
        })},
        {"external_backing_ingress_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.stage_work.ingress_queue_wait_ns);
        })},
        {"external_backing_scheduler_queue_wait_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.stage_work.scheduler_queue_wait_ns);
        })},
        {"external_backing_command_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.stage_work.command_ns);
        })},
        {"external_backing_array_read_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.stage_work.array_read_ns);
        })},
        {"external_backing_array_program_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.stage_work.array_program_ns);
        })},
        {"external_backing_channel_transfer_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.stage_work.channel_transfer_ns);
        })},
        {"external_backing_hb_io_transfer_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.stage_work.hb_io_transfer_ns);
        })},
        {"external_backing_transport_latency_work_ns", external_backing_cell([](R r) {
            return cell_fixed(
                r.external_backing_stats.stage_work.transport_latency_ns);
        })},
        {"external_backing_total_overlapping_work_ns", external_backing_cell([](R r) {
            return cell_fixed(r.external_backing_stats.stage_work.total_work_ns());
        })},
        {"hbm_write_buffer_user_write_bytes", [](R r) {
            return cell_count(r.hbm_write_buffer_user_write_bytes);
        }},
        {"hbm_write_buffer_destaged_bytes", [](R r) {
            return cell_count(r.hbm_write_buffer_destaged_bytes);
        }},
        {"hbm_write_buffer_drain_destaged_bytes", [](R r) {
            return cell_count(r.hbm_write_buffer_drain_destaged_bytes);
        }},
        {"hbm_write_buffer_peak_bytes", [](R r) {
            return cell_count(r.hbm_write_buffer_peak_bytes);
        }},
        {"hbm_write_buffer_full_waits", [](R r) {
            return cell_count(r.hbm_write_buffer_full_waits);
        }},
        {"cooperative_write_controller_full_waited_ops", [](R r) {
            return r.cooperative_write_controller_present ?
                cell_count(r.hbm_write_buffer_full_waits) : cell_missing();
        }},
        {"cooperative_write_controller_full_wait_work_ns", [](R r) {
            return r.cooperative_write_controller_present ?
                cell_fixed(r.hbm_write_buffer_wait_ns) : cell_missing();
        }},
        {"hbf_end_to_end_media_bytes_per_cooperative_user_write_byte", [](R r) {
            if (!r.has_hbf || r.hbm_write_buffer_user_write_bytes == 0) {
                return cell_missing();
            }
            return cell_same(fixed(
                static_cast<double>(r.hbf_stats.physical_write_bytes) /
                static_cast<double>(r.hbm_write_buffer_user_write_bytes),
                6));
        }},
        {"base_die_link_read_busy_ns", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_fixed(r.base_die_link_stats.read_busy_ns);
        }},
        {"base_die_link_read_resource_count", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_count(r.base_die_link_stats.links);
        }},
        {"base_die_link_read_active_span_ns", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_fixed(r.base_die_link_stats.active_span_ns());
        }},
        {"base_die_link_read_capacity_time_ns", [](R r) {
            if (r.base_die_link_stats.links == 0) return cell_missing();
            return cell_fixed(resource_busy_metric(
                r.base_die_link_stats.read_busy_ns,
                r.base_die_link_stats.links,
                r.base_die_link_stats.active_span_ns()).capacity_time_ns);
        }},
        {"base_die_link_read_utilization", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_resource_utilization(
                    r.base_die_link_stats.read_busy_ns,
                    r.base_die_link_stats.links,
                    r.base_die_link_stats.active_span_ns());
        }},
        {"base_die_link_write_busy_ns", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_fixed(r.base_die_link_stats.write_busy_ns);
        }},
        {"base_die_link_write_resource_count", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_count(r.base_die_link_stats.links);
        }},
        {"base_die_link_write_active_span_ns", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_fixed(r.base_die_link_stats.active_span_ns());
        }},
        {"base_die_link_write_capacity_time_ns", [](R r) {
            if (r.base_die_link_stats.links == 0) return cell_missing();
            return cell_fixed(resource_busy_metric(
                r.base_die_link_stats.write_busy_ns,
                r.base_die_link_stats.links,
                r.base_die_link_stats.active_span_ns()).capacity_time_ns);
        }},
        {"base_die_link_write_utilization", [](R r) {
            return r.base_die_link_stats.links == 0 ? cell_missing() :
                cell_resource_utilization(
                    r.base_die_link_stats.write_busy_ns,
                    r.base_die_link_stats.links,
                    r.base_die_link_stats.active_span_ns());
        }},
        {"hbm_address_mapping_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.address_mapping_ns);
        })},
        {"hbm_scheduler_queue_wait_work_ns", hbm_cell([](R r) {
            return cell_same(fixed(r.hbm_stats.stage_work.scheduler_queue_wait_ns));
        })},
        {"hbm_access_latency_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.command_ns);
        })},
        {"hbm_efficiency_overhead_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.maintenance_ns);
        })},
        {"hbm_channel_transfer_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.channel_transfer_ns);
        })},
        {"hbm_total_overlapping_work_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.stage_work.total_work_ns());
        })},
        {"hbm_data_bus_busy_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.bus_busy_ns);
        })},
        {"hbm_controller_buffer_read_bytes", hbm_cell([](R r) { return cell_count(r.hbm_stats.controller_buffer_read_bytes); })},
        {"hbm_controller_buffer_write_bytes", hbm_cell([](R r) { return cell_count(r.hbm_stats.controller_buffer_write_bytes); })},
        {"hbm_controller_buffer_transfers", hbm_cell([](R r) { return cell_count(r.hbm_stats.controller_buffer_transfers); })},
        {"hbm_controller_buffer_bus_busy_ns", hbm_cell([](R r) { return cell_fixed(r.hbm_stats.controller_buffer_bus_busy_ns); })},
        {"hbm_data_bus_resource_count", hbm_cell([](R r) {
            return cell_count(r.hbm_stats.pseudo_channels);
        })},
        {"hbm_data_bus_active_span_ns", hbm_cell([](R r) {
            return cell_fixed(r.hbm_stats.active_span_ns());
        })},
        {"hbm_data_bus_capacity_time_ns", hbm_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbm_stats.bus_busy_ns,
                r.hbm_stats.pseudo_channels,
                r.hbm_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbm_data_bus_utilization", hbm_cell([](R r) {
            return cell_resource_utilization(
                r.hbm_stats.bus_busy_ns,
                r.hbm_stats.pseudo_channels,
                r.hbm_stats.active_span_ns());
        })},
        {"hbm_max_queue_occupancy", hbm_cell([](R r) {
            return cell_count(r.hbm_stats.max_queue_occupancy);
        })},
        {"hbm_channel_transfers", hbm_cell([](R r) {
            return cell_count(r.hbm_stats.channel_transfers);
        })},
        {"hbm_service_quanta", hbm_cell([](R r) {
            return cell_count(r.hbm_stats.service_quanta);
        })},
        {"hbm_bus_parallelism", hbm_cell([](R r) { return cell_same(fixed(r.hbm_stats.bus_parallelism(), 2)); })},
        {"hbm_active_pch", hbm_cell([](R r) {
            return cell_same(ratio_text(r.hbm_stats.active_pseudo_channels, r.hbm_stats.pseudo_channels)); })},
        {"hbm_pch_busy_skew", hbm_cell([](R r) { return cell_same(fixed(r.hbm_stats.pseudo_channel_busy_skew(), 2)); })},
        {"hbf_waf", hbf_cell([](R r) {
            return cell_optional_fixed(r.hbf_stats.waf()); })},
        {"hbf_data_program_payload_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.data_program_payload_bytes); })},
        {"hbf_mapping_program_payload_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_program_payload_bytes); })},
        {"hbf_gc_relocation_payload_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_relocation_payload_bytes); })},
        {"hbf_page_reads", hbf_cell([](R r) { return cell_count(r.hbf_stats.page_reads); })},
        {"hbf_data_programs", hbf_cell([](R r) { return cell_count(r.hbf_stats.data_programs); })},
        {"hbf_page_programs", hbf_cell([](R r) { return cell_count(r.hbf_stats.page_programs); })},
        {"hbf_page_read_admission_events", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.page_read_admission_events);
        })},
        {"hbf_page_read_admission_waited_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.page_read_admission_waited_pages);
        })},
        {"hbf_page_read_admission_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.page_read_admission_wait_ns);
        })},
        {"hbf_page_read_admission_max_wait_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.page_read_admission_max_wait_ns);
        })},
        {"hbf_scalar_read_requests", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.scalar_read_requests); })},
        {"hbf_scalar_read_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.scalar_read_pages); })},
        {"hbf_mapping_page_programs", hbf_cell([](R r) { return cell_count(r.hbf_stats.mapping_page_programs); })},
        {"hbf_mapping_table_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_table_bytes); })},
        {"hbf_resident_mapping_table_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.resident_mapping_table_bytes); })},
        {"hbf_resident_mapping_table_bytes_per_stack", hbf_cell([](R r) {
            return cell_count(
                r.hbf_stats.resident_mapping_table_bytes_per_stack); })},
        {"hbf_resident_mapping_pages_per_stack", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.resident_mapping_pages_per_stack); })},
        {"hbf_mapping_directory_entry_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_directory_entry_bytes); })},
        {"hbf_mapping_directory_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_directory_bytes); })},
        {"hbf_mapping_directory_bytes_per_stack", hbf_cell([](R r) {
            return cell_count(
                r.hbf_stats.mapping_directory_bytes_per_stack); })},
        {"hbf_mapping_cache_capacity_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_cache_capacity_bytes); })},
        {"hbf_controller_dram_budget_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.controller_dram_budget_bytes); })},
        {"hbf_write_buffer_capacity_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.write_buffer_capacity_bytes); })},
        {"hbf_mapping_cache_pages_per_stack", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_cache_pages_per_stack); })},
        {"hbf_mapping_cache_hits", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_cache_hits); })},
        {"hbf_mapping_cache_misses", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_cache_misses); })},
        {"hbf_mapping_cache_erased_misses", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_cache_erased_misses); })},
        {"hbf_mapping_cache_coalesced_misses", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_cache_coalesced_misses); })},
        {"hbf_mapping_cache_evictions", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_cache_evictions); })},
        {"hbf_mapping_cache_dirty_evictions", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_cache_dirty_evictions); })},
        {"hbf_mapping_media_reads", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_media_reads); })},
        {"hbf_mapping_media_read_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_media_read_bytes); })},
        {"hbf_mapping_lookup_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_lookup_ops); })},
        {"hbf_mapping_user_lookup_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_user_lookup_ops); })},
        {"hbf_mapping_gc_lookup_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_gc_lookup_ops); })},
        {"hbf_mapping_update_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_update_ops); })},
        {"hbf_mapping_user_update_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_user_update_ops); })},
        {"hbf_mapping_gc_update_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_gc_update_ops); })},
        {"hbf_mapping_dram_wait_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_dram_wait_ops); })},
        {"hbf_mapping_dram_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.mapping_dram_wait_ns); })},
        {"hbf_mapping_dram_wait_max_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.mapping_dram_wait_max_ns); })},
        {"hbf_mapping_dram_issue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.mapping_dram_issue_busy_ns); })},
        {"hbf_mapping_dram_resources", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_dram_resources); })},
        {"hbf_read_requests", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.read_requests); })},
        {"hbf_program_requests", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.program_requests); })},
        {"hbf_erase_requests", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.erase_requests); })},
        {"hbf_invalidations", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.invalidations); })},
        {"hbf_total_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.total_pages); })},
        {"hbf_free_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.free_pages); })},
        {"hbf_valid_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.valid_pages); })},
        {"hbf_invalid_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.invalid_pages); })},
        {"hbf_pending_program_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.pending_program_pages); })},
        {"hbf_pending_mapping_publications", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.pending_mapping_publications); })},
        {"hbf_static_reserved_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.static_reserved_pages); })},
        {"hbf_initial_logical_data_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.initial_logical_data_pages); })},
        {"hbf_initial_mapping_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.initial_mapping_pages); })},
        {"hbf_compact_initial_logical_data_pages", hbf_cell([](R r) {
            return cell_count(
                r.hbf_stats.compact_initial_logical_data_pages); })},
        {"hbf_compact_initial_mapping_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.compact_initial_mapping_pages); })},
        {"hbf_compact_live_logical_data_pages", hbf_cell([](R r) {
            return cell_count(
                r.hbf_stats.compact_live_logical_data_pages); })},
        {"hbf_compact_live_mapping_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.compact_live_mapping_pages); })},
        {"hbf_compact_retired_logical_data_pages", hbf_cell([](R r) {
            return cell_count(
                r.hbf_stats.compact_retired_logical_data_pages); })},
        {"hbf_compact_retired_mapping_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.compact_retired_mapping_pages); })},
        {"hbf_static_unmaterialized_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.static_unmaterialized_pages); })},
        {"hbf_accounting_verified", hbf_cell([](R r) {
            return cell_bool(r.hbf_stats.accounting_verified); })},
        {"hbf_auto_erase_requests", hbf_cell([](R r) { return cell_count(r.hbf_stats.auto_erase_requests); })},
        {"hbf_host_gc_read_bytes", hbf_cell([](R r) { return cell_count(r.hbf_stats.host_gc_read_bytes); })},
        {"hbf_host_hbm_reserved_bytes", hbf_cell([](R r) { return cell_count(r.hbf_stats.host_hbm_reserved_bytes); })},
        {"hbf_host_hbm_read_bytes", hbf_cell([](R r) { return cell_count(r.hbf_stats.host_hbm_read_bytes); })},
        {"hbf_host_hbm_write_bytes", hbf_cell([](R r) { return cell_count(r.hbf_stats.host_hbm_write_bytes); })},
        {"hbf_host_hbm_busy_ns", hbf_cell([](R r) { return cell_fixed(r.hbf_stats.host_hbm_busy_ns); })},
        {"hbf_host_hbm_queue_wait_ns", hbf_cell([](R r) { return cell_fixed(r.hbf_stats.host_hbm_queue_wait_ns); })},
        {"hbf_host_gc_write_bytes", hbf_cell([](R r) { return cell_count(r.hbf_stats.host_gc_write_bytes); })},
        {"hbf_host_gc_buffer_bytes", hbf_cell([](R r) { return cell_count(r.hbf_stats.host_gc_buffer_bytes); })},
        {"hbf_block_erases", hbf_cell([](R r) { return cell_count(r.hbf_stats.block_erases); })},
        {"hbf_gc_runs", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_runs); })},
        {"hbf_gc_relocations", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_relocations); })},
        {"hbf_gc_data_relocations", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_data_relocations); })},
        {"hbf_gc_mapping_relocations", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_mapping_relocations); })},
        {"hbf_gc_reclaimed_invalid_pages", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_reclaimed_invalid_pages); })},
        {"hbf_gc_user_blocked_runs", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.gc_user_blocked_runs); })},
        {"hbf_flash_scheduler_enqueues", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.flash_scheduler_enqueues); })},
        {"hbf_flash_scheduler_issues", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.flash_scheduler_issues); })},
        {"hbf_ingress_queue_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.ingress_queue_wait_ns);
        })},
        {"hbf_scheduler_queue_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.scheduler_queue_wait_ns);
        })},
        {"hbf_ecc_queue_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.ecc_queue_wait_ns);
        })},
        {"hbf_address_mapping_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.address_mapping_ns);
        })},
        {"hbf_translation_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.translation_ns);
        })},
        {"hbf_mapping_dram_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.mapping_dram_ns);
        })},
        {"hbf_write_buffer_dram_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.write_buffer_dram_ns);
        })},
        {"hbf_command_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.command_ns);
        })},
        {"hbf_ecc_decode_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.ecc_decode_ops);
        })},
        {"hbf_ecc_encode_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.ecc_encode_ops);
        })},
        {"hbf_ecc_decode_queue_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_decode_queue_wait_ns);
        })},
        {"hbf_ecc_encode_queue_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_encode_queue_wait_ns);
        })},
        {"hbf_ecc_decode_response_latency_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_decode_latency_work_ns);
        })},
        {"hbf_ecc_encode_response_latency_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_encode_latency_work_ns);
        })},
        {"hbf_ecc_decode_issue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_decode_issue_busy_ns);
        })},
        {"hbf_ecc_encode_issue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_encode_issue_busy_ns);
        })},
        {"hbf_ecc_decode_codeword_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.ecc_decode_codeword_bytes);
        })},
        {"hbf_ecc_encode_codeword_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.ecc_encode_codeword_bytes);
        })},
        {"hbf_ecc_codeword_bytes", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.ecc_codeword_bytes);
        })},
        {"hbf_ecc_issue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.ecc_issue_busy_ns);
        })},
        {"hbf_ecc_issue_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.dies);
        })},
        {"hbf_ecc_issue_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_ecc_issue_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.ecc_issue_busy_ns,
                r.hbf_stats.dies,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_ecc_issue_utilization", hbf_cell([](R r) {
            return cell_percent(r.hbf_stats.ecc_issue_utilization());
        })},
        {"hbf_ecc_issue_parallelism", hbf_cell([](R r) {
            return cell_same(fixed(r.hbf_stats.ecc_issue_parallelism(), 2));
        })},
        {"hbf_ecc_max_inflight_per_die", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.max_ecc_inflight_per_die);
        })},
        {"hbf_write_buffer_slot_waited_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.write_buffer_slot_wait_ops);
        })},
        {"hbf_write_buffer_slot_wait_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.write_buffer_slot_wait_ns);
        })},
        {"hbf_write_buffer_dram_read_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.write_buffer_dram_read_ops);
        })},
        {"hbf_write_buffer_dram_write_ops", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.write_buffer_dram_write_ops);
        })},
        {"hbf_array_read_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.array_read_ns);
        })},
        {"hbf_array_program_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.array_program_ns);
        })},
        {"hbf_array_erase_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.array_erase_ns);
        })},
        {"hbf_media_lane_transfer_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.media_lane_transfer_ns);
        })},
        {"hbf_page_buffer_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.page_buffer_ns);
        })},
        {"hbf_sram_staging_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.sram_staging_ns);
        })},
        {"hbf_channel_transfer_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.channel_transfer_ns);
        })},
        {"hbf_tsv_transfer_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.tsv_transfer_ns);
        })},
        {"hbf_hb_io_transfer_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.hb_io_transfer_ns);
        })},
        {"hbf_transport_latency_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.transport_latency_ns);
        })},
        {"hbf_ecc_response_latency_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.ecc_latency_ns);
        })},
        {"hbf_maintenance_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.maintenance_ns);
        })},
        {"hbf_total_overlapping_work_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.stage_work.total_work_ns());
        })},
        {"hbf_logic_ingress_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.logic_ingress_busy_ns);
        })},
        {"hbf_logic_ingress_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.logic_ingress_resources);
        })},
        {"hbf_logic_ingress_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_logic_ingress_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.logic_ingress_busy_ns,
                r.hbf_stats.logic_ingress_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_logic_ingress_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.logic_ingress_busy_ns,
                r.hbf_stats.logic_ingress_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_mapping_dram_issue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.mapping_dram_issue_busy_ns);
        })},
        {"hbf_mapping_dram_issue_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.mapping_dram_resources);
        })},
        {"hbf_mapping_dram_issue_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_mapping_dram_issue_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.mapping_dram_issue_busy_ns,
                r.hbf_stats.mapping_dram_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_mapping_dram_issue_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.mapping_dram_issue_busy_ns,
                r.hbf_stats.mapping_dram_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_plane_media_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.media_busy_ns);
        })},
        {"hbf_plane_media_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.planes);
        })},
        {"hbf_plane_media_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_plane_media_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.media_busy_ns,
                r.hbf_stats.planes,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_plane_media_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.media_busy_ns,
                r.hbf_stats.planes,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_media_lane_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.read_lane_busy_ns);
        })},
        {"hbf_media_lane_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.media_lanes);
        })},
        {"hbf_media_lane_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_media_lane_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.read_lane_busy_ns,
                r.hbf_stats.media_lanes,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_media_lane_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.read_lane_busy_ns,
                r.hbf_stats.media_lanes,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_subarray_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.subarray_read_busy_ns);
        })},
        {"hbf_subarray_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.subarrays);
        })},
        {"hbf_subarray_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_subarray_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.subarray_read_busy_ns,
                r.hbf_stats.subarrays,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_subarray_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.subarray_read_busy_ns,
                r.hbf_stats.subarrays,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_page_buffer_bank_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.page_buffer_bank_busy_ns);
        })},
        {"hbf_page_buffer_bank_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.page_buffer_banks);
        })},
        {"hbf_page_buffer_bank_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_page_buffer_bank_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.page_buffer_bank_busy_ns,
                r.hbf_stats.page_buffer_banks,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_page_buffer_bank_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.page_buffer_bank_busy_ns,
                r.hbf_stats.page_buffer_banks,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_flash_source_queue_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.flash_source_queue_busy_ns);
        })},
        {"hbf_flash_source_queue_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.flash_source_queue_resources);
        })},
        {"hbf_flash_source_queue_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_flash_source_queue_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.flash_source_queue_busy_ns,
                r.hbf_stats.flash_source_queue_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_flash_source_queue_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.flash_source_queue_busy_ns,
                r.hbf_stats.flash_source_queue_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_channel_command_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.channel_command_busy_ns);
        })},
        {"hbf_channel_command_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.channel_command_resources);
        })},
        {"hbf_channel_command_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_channel_command_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.channel_command_busy_ns,
                r.hbf_stats.channel_command_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_channel_command_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.channel_command_busy_ns,
                r.hbf_stats.channel_command_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_channel_data_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.channel_data_busy_ns);
        })},
        {"hbf_channel_data_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.channel_data_resources);
        })},
        {"hbf_channel_data_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_channel_data_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.channel_data_busy_ns,
                r.hbf_stats.channel_data_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_channel_data_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.channel_data_busy_ns,
                r.hbf_stats.channel_data_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_tsv_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.tsv_busy_ns);
        })},
        {"hbf_tsv_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.tsv_resources);
        })},
        {"hbf_tsv_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_tsv_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.tsv_busy_ns,
                r.hbf_stats.tsv_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_tsv_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.tsv_busy_ns,
                r.hbf_stats.tsv_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_sram_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.sram_busy_ns);
        })},
        {"hbf_sram_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.sram_resources);
        })},
        {"hbf_sram_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_sram_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.sram_busy_ns,
                r.hbf_stats.sram_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_sram_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.sram_busy_ns,
                r.hbf_stats.sram_resources,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_hbio_command_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.hb_io_command_busy_ns);
        })},
        {"hbf_hbio_command_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.hbio_command_resources);
        })},
        {"hbf_hbio_command_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_hbio_command_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.hb_io_command_busy_ns,
                r.hbf_stats.hbio_command_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_hbio_data_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.hb_io_data_busy_ns);
        })},
        {"hbf_hbio_data_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.hbio_data_resources);
        })},
        {"hbf_hbio_data_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_hbio_data_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.hb_io_data_busy_ns,
                r.hbf_stats.hbio_data_resources,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_sequencer_busy_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.sequencer_busy_ns);
        })},
        {"hbf_sequencer_resource_count", hbf_cell([](R r) {
            return cell_count(r.hbf_stats.dies);
        })},
        {"hbf_sequencer_active_span_ns", hbf_cell([](R r) {
            return cell_fixed(r.hbf_stats.active_span_ns());
        })},
        {"hbf_sequencer_capacity_time_ns", hbf_cell([](R r) {
            return cell_fixed(resource_busy_metric(
                r.hbf_stats.sequencer_busy_ns,
                r.hbf_stats.dies,
                r.hbf_stats.active_span_ns()).capacity_time_ns);
        })},
        {"hbf_sequencer_utilization", hbf_cell([](R r) {
            return cell_resource_utilization(
                r.hbf_stats.sequencer_busy_ns,
                r.hbf_stats.dies,
                r.hbf_stats.active_span_ns());
        })},
        {"hbf_media_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.media_parallelism(), 2)); })},
        {"hbf_active_planes", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_planes, r.hbf_stats.planes)); })},
        {"hbf_read_lane_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.read_lane_parallelism(), 2)); })},
        {"hbf_active_media_lanes", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_media_lanes, r.hbf_stats.media_lanes)); })},
        {"hbf_media_lane_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.media_lane_skew(), 2)); })},
        {"hbf_subarray_read_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.subarray_read_parallelism(), 2)); })},
        {"hbf_active_subarrays", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_subarrays, r.hbf_stats.subarrays)); })},
        {"hbf_subarray_busy_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.subarray_busy_skew(), 2)); })},
        {"hbf_subarray_read_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.subarray_read_skew(), 2)); })},
        {"hbf_page_buffer_bank_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.page_buffer_bank_parallelism(), 2)); })},
        {"hbf_active_page_buffer_banks", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_page_buffer_banks, r.hbf_stats.page_buffer_banks)); })},
        {"hbf_page_buffer_bank_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.page_buffer_bank_skew(), 2)); })},
        {"hbf_page_buffer_bank_read_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.page_buffer_bank_read_skew(), 2)); })},
        {"hbf_plane_media_skew", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.plane_media_skew(), 2)); })},
        {"hbf_io_util", hbf_cell([](R r) { return cell_percent(r.hbf_stats.io_utilization()); })},
        {"hbf_hbio_command_utilization", hbf_cell([](R r) {
            return cell_percent(r.hbf_stats.hbio_command_utilization()); })},
        {"hbf_hbio_data_utilization", hbf_cell([](R r) {
            return cell_percent(r.hbf_stats.hbio_data_utilization()); })},
        {"hbf_channel_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.channel_parallelism(), 2)); })},
        {"hbf_active_channels", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_channels, r.hbf_stats.channels)); })},
        {"hbf_sequencer_parallelism", hbf_cell([](R r) { return cell_same(fixed(r.hbf_stats.sequencer_parallelism(), 2)); })},
        {"hbf_active_dies", hbf_cell([](R r) {
            return cell_same(ratio_text(r.hbf_stats.active_dies, r.hbf_stats.dies)); })},
        {"user_latency_average_ns", [](R r) {
            return cell_fixed(latency_distribution(r).average_ns);
        }},
        {"user_latency_p50_ns", [](R r) {
            return cell_fixed(latency_distribution(r).p50_ns);
        }},
        {"user_latency_p95_ns", [](R r) {
            return cell_fixed(latency_distribution(r).p95_ns);
        }},
        {"user_latency_max_ns", [](R r) {
            return cell_fixed(latency_distribution(r).max_ns);
        }},
        {"service_latency_average_ns", [](R r) {
            return cell_fixed(service_latency_distribution(r).average_ns);
        }},
        {"service_latency_p50_ns", [](R r) {
            return cell_fixed(service_latency_distribution(r).p50_ns);
        }},
        {"service_latency_p95_ns", [](R r) {
            return cell_fixed(service_latency_distribution(r).p95_ns);
        }},
        {"service_latency_max_ns", [](R r) {
            return cell_fixed(service_latency_distribution(r).max_ns);
        }},
        {"source_latency_average_ns", [](R r) {
            return cell_fixed(source_latency_distribution(r).average_ns);
        }},
        {"source_latency_p50_ns", [](R r) {
            return cell_fixed(source_latency_distribution(r).p50_ns);
        }},
        {"source_latency_p95_ns", [](R r) {
            return cell_fixed(source_latency_distribution(r).p95_ns);
        }},
        {"source_latency_max_ns", [](R r) {
            return cell_fixed(source_latency_distribution(r).max_ns);
        }},
        {"trace_origin_ns", [](R r) {
            return cell_fixed(r.first_arrival_ns);
        }},
        {"last_offered_arrival_ns", [](R r) {
            return cell_fixed(r.last_arrival_ns);
        }},
        {"last_user_completion_ns", [](R r) {
            return cell_fixed(r.user_finish_ns);
        }},
        {"quiescent_finish_ns", [](R r) {
            return cell_fixed(r.finish_ns);
        }},
        {"offered_arrival_span_ns", [](R r) {
            return cell_fixed(r.last_arrival_ns - r.first_arrival_ns);
        }},
        {"post_offer_user_completion_tail_ns", [](R r) {
            return cell_fixed(r.user_finish_ns - r.last_arrival_ns);
        }},
        {"user_completion_span_ns", [](R r) {
            return cell_fixed(r.user_finish_ns - r.first_arrival_ns);
        }},
        {"drain_tail_ns", [](R r) {
            return cell_fixed(r.finish_ns - r.user_finish_ns);
        }},
        {"makespan_ns", [](R r) {
            return cell_fixed(r.finish_ns - r.first_arrival_ns);
        }},
        {"service_to_user_completion_sum_work_ns", [](R r) {
            return cell_fixed(
                service_latency_distribution(r).sum_work_ns);
        }},
        {"phase_barriers", [](R r) {
            return cell_count(r.phase_barriers);
        }},
        {"phase_dependency_waited_ops", [](R r) {
            return cell_count(r.phase_dependency_waited_ops);
        }},
        {"phase_dependency_wait_work_ns", [](R r) {
            return cell_fixed(r.phase_dependency_wait_work_ns);
        }},
        {"phase_dependency_max_wait_ns", [](R r) {
            return cell_fixed(r.phase_dependency_max_wait_ns);
        }},
        {"front_end_admission_waited_ops", [](R r) {
            return cell_count(r.front_end_admission_waited_ops);
        }},
        {"front_end_admission_wait_work_ns", [](R r) {
            return cell_fixed(r.front_end_admission_wait_work_ns);
        }},
        {"front_end_admission_max_wait_ns", [](R r) {
            return cell_fixed(r.front_end_admission_max_wait_ns);
        }},
        {"offered_to_user_completion_sum_work_ns", [](R r) {
            return cell_fixed(latency_distribution(r).sum_work_ns);
        }},
        {"source_to_user_completion_sum_work_ns", [](R r) {
            return cell_fixed(
                source_latency_distribution(r).sum_work_ns);
        }},
        {"user_completion_throughput_GBps", [](R r) {
            return cell_fixed(user_completion_throughput_GBps(r)); }},
        {"makespan_throughput_GBps", [](R r) {
            return cell_fixed(makespan_throughput_GBps(r)); }},
        {"hbm_issue_mode", [](R) { return cell_same(kHbmIssueMode); }},
        {"arrival_model", [](R r) {
            return r.offered_load ?
                cell_same(r.offered_load->open_loop ? "open-loop" : "closed-loop") :
                cell_missing();
        }},
        {"offered_rate_GBps", [](R r) {
            return r.offered_load ?
                cell_optional_fixed(r.offered_load->rate_GBps) : cell_missing();
        }},
        {"offered_read_rate_GBps", [](R r) {
            return r.offered_load ?
                cell_optional_fixed(r.offered_load->read_rate_GBps) : cell_missing();
        }},
        {"offered_write_rate_GBps", [](R r) {
            return r.offered_load ?
                cell_optional_fixed(r.offered_load->write_rate_GBps) : cell_missing();
        }},
        {"hbm_offered_to_peak_ratio", [](R r) {
            return r.offered_load ?
                cell_optional_fixed(r.offered_load->hbm_ratio) : cell_missing();
        }},
        {"hbf_offered_to_peak_ratio", [](R r) {
            return r.offered_load ?
                cell_optional_fixed(r.offered_load->hbf_ratio) : cell_missing();
        }},
        {"external_offered_to_peak_ratio", [](R r) {
            return r.offered_load ?
                cell_optional_fixed(r.offered_load->external_ratio) : cell_missing();
        }},
        {"warnings", [](R r) { return cell_count(r.warnings.size()); }},
    };
    return columns;
}

void print_result_table(const std::vector<ScenarioResult>& results) {
    const auto& columns = result_columns();
    for (std::size_t i = 0; i < columns.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << columns[i].header;
    }
    std::cout << '\n';
    for (const auto& result : results) {
        for (std::size_t i = 0; i < columns.size(); ++i) {
            std::cout << (i == 0 ? "" : ",") << columns[i].cell(result).console;
        }
        std::cout << '\n';
    }
}

void print_time_breakdown(const ScenarioResult& result) {
    const auto time = make_time_breakdown(result);
    const auto& wall = time.wall_clock;
    const auto& latency = time.latency_work;
    std::cout << "  Time breakdown contract: wall spans are additive; "
                 "stage work may overlap and must not be summed as elapsed time.\n";
    std::cout << "    wall_clock_component,value_ns\n"
              << "    offered_arrival_span," << fixed(wall.offered_arrival_span_ns) << '\n'
              << "    post_offer_user_completion_tail,"
              << fixed(wall.post_offer_user_completion_tail_ns) << '\n'
              << "    user_completion_span," << fixed(wall.user_completion_span_ns) << '\n'
              << "    drain_tail," << fixed(wall.drain_tail_ns) << '\n'
              << "    makespan," << fixed(wall.makespan_ns) << '\n';
    std::cout << "    latency_work_component,value_ns\n"
              << "    service_to_user_completion_sum,"
              << fixed(latency.service_to_user_completion_sum_work_ns) << '\n'
              << "    phase_dependency_wait_sum,"
              << fixed(latency.phase_dependency_wait_work_ns) << '\n'
              << "    front_end_admission_wait_sum,"
              << fixed(latency.front_end_admission_wait_work_ns) << '\n'
              << "    offered_to_user_completion_sum,"
              << fixed(latency.offered_to_user_completion_sum_work_ns) << '\n'
              << "    source_to_user_completion_sum,"
              << fixed(latency.source_to_user_completion_sum_work_ns) << '\n'
              << "    phase_dependency_max,"
              << fixed(latency.phase_dependency_max_wait_ns) << '\n'
              << "    front_end_admission_max,"
              << fixed(latency.front_end_admission_max_wait_ns) << '\n';

    std::cout << "    stage_component,hbm_work_ns,hbf_work_ns,"
                 "external_work_ns\n";
    const auto stage_row = [&](const char* name, double Breakdown::*member) {
        std::cout << "    " << name << ','
                  << fixed(time.hbm_stage_work.*member) << ','
                  << fixed(time.hbf_stage_work.*member) << ','
                  << fixed(time.external_backing_stage_work.*member) << '\n';
    };
    stage_row("ingress_queue_wait", &Breakdown::ingress_queue_wait_ns);
    stage_row("scheduler_queue_wait", &Breakdown::scheduler_queue_wait_ns);
    stage_row("address_mapping", &Breakdown::address_mapping_ns);
    stage_row("translation", &Breakdown::translation_ns);
    stage_row("mapping_dram", &Breakdown::mapping_dram_ns);
    stage_row("write_buffer_dram", &Breakdown::write_buffer_dram_ns);
    stage_row("refresh_stall", &Breakdown::refresh_stall_ns);
    stage_row("precharge", &Breakdown::precharge_ns);
    stage_row("activation", &Breakdown::activation_ns);
    stage_row("command", &Breakdown::command_ns);
    stage_row("array_read", &Breakdown::array_read_ns);
    stage_row("array_program", &Breakdown::array_program_ns);
    stage_row("array_erase", &Breakdown::array_erase_ns);
    stage_row("media_lane_transfer", &Breakdown::media_lane_transfer_ns);
    stage_row("page_buffer", &Breakdown::page_buffer_ns);
    stage_row("sram_staging", &Breakdown::sram_staging_ns);
    stage_row("channel_transfer", &Breakdown::channel_transfer_ns);
    stage_row("tsv_transfer", &Breakdown::tsv_transfer_ns);
    stage_row("hb_io_transfer", &Breakdown::hb_io_transfer_ns);
    stage_row("transport_latency", &Breakdown::transport_latency_ns);
    stage_row("ecc_queue_wait", &Breakdown::ecc_queue_wait_ns);
    stage_row("ecc_response_latency", &Breakdown::ecc_latency_ns);
    stage_row("maintenance", &Breakdown::maintenance_ns);
    if (result.has_hbf) {
        const auto& hbf = result.hbf_stats;
        std::cout << "    hbf_subdiagnostic,decode,encode\n"
                  << "    ecc_queue_wait_work_ns,"
                  << fixed(hbf.ecc_decode_queue_wait_ns) << ','
                  << fixed(hbf.ecc_encode_queue_wait_ns) << '\n'
                  << "    ecc_response_latency_work_ns,"
                  << fixed(hbf.ecc_decode_latency_work_ns) << ','
                  << fixed(hbf.ecc_encode_latency_work_ns) << '\n'
                  << "    ecc_issue_busy_ns,"
                  << fixed(hbf.ecc_decode_issue_busy_ns) << ','
                  << fixed(hbf.ecc_encode_issue_busy_ns) << '\n'
                  << "    ecc_ops," << hbf.ecc_decode_ops << ','
                  << hbf.ecc_encode_ops << '\n'
                  << "    ecc_codeword_bytes,"
                  << hbf.ecc_decode_codeword_bytes << ','
                  << hbf.ecc_encode_codeword_bytes << '\n'
                  << "    hbf_write_buffer_slot_waited_ops,"
                  << hbf.write_buffer_slot_wait_ops << ",n/a\n"
                  << "    hbf_write_buffer_slot_wait_work_ns,"
                  << fixed(hbf.write_buffer_slot_wait_ns) << ",n/a\n";
    }

    const auto& link = time.base_die_link_stage_work;
    std::cout << "    base_die_link_component,work_ns\n"
              << "    read_queue_wait," << fixed(link.read_queue_wait_work_ns) << '\n'
              << "    write_queue_wait," << fixed(link.write_queue_wait_work_ns) << '\n'
              << "    read_serialization," << fixed(link.read_serialization_work_ns) << '\n'
              << "    write_serialization," << fixed(link.write_serialization_work_ns) << '\n'
              << "    read_fixed_latency," << fixed(link.read_fixed_latency_work_ns) << '\n'
              << "    write_fixed_latency," << fixed(link.write_fixed_latency_work_ns) << '\n';

    const auto& controller =
        time.layer_streaming_controller_stage_work;
    if (controller.present) {
        std::cout << "    layer_streaming_controller_component,work_ns\n"
                  << "    backing_admission_wait_work,"
                  << fixed(controller.backing_admission_wait_work_ns) << '\n'
                  << "    backing_admission_max_wait,"
                  << fixed(controller.backing_admission_max_wait_ns) << '\n'
                  << "    user_wait_work,"
                  << fixed(controller.user_wait_work_ns) << '\n'
                  << "    user_max_wait,"
                  << fixed(controller.user_max_wait_ns) << '\n'
                  << "    exposed_prefetch,"
                  << fixed(controller.exposed_prefetch_ns) << '\n'
                  << "    hidden_prefetch,"
                  << fixed(controller.hidden_prefetch_ns) << '\n'
                  << "    buffer_reuse_wait_work,"
                  << fixed(controller.buffer_reuse_wait_work_ns) << '\n';
    }
    const auto& cooperative = time.cooperative_write_controller_stage_work;
    if (cooperative.present) {
        std::cout << "    cooperative_write_controller_component,value\n"
                  << "    full_waited_ops," << cooperative.full_waited_ops << '\n'
                  << "    full_wait_work_ns,"
                  << fixed(cooperative.full_wait_work_ns) << '\n';
    }

    std::cout << "    resource,busy_ns,resource_count,active_span_ns,"
                 "capacity_time_ns,utilization\n";
    const auto resource_row = [](const char* name, const ResourceBusyMetric& metric) {
        std::cout << "    " << name << ',' << fixed(metric.busy_ns) << ','
                  << metric.resource_count << ',' << fixed(metric.active_span_ns)
                  << ',' << fixed(metric.capacity_time_ns) << ','
                  << fixed(metric.utilization, 6) << '\n';
    };
    const auto& resource = time.resource_busy;
    if (result.has_hbm) resource_row("hbm_data_bus", resource.hbm_data_bus);
    if (result.has_hbf) {
        resource_row("hbf_logic_ingress", resource.hbf_logic_ingress);
        resource_row(
            "hbf_mapping_dram_issue",
            resource.hbf_mapping_dram_issue);
        resource_row("hbf_plane_media", resource.hbf_plane_media);
        resource_row("hbf_media_lane", resource.hbf_media_lane);
        resource_row("hbf_subarray", resource.hbf_subarray);
        resource_row("hbf_page_buffer_bank", resource.hbf_page_buffer_bank);
        resource_row("hbf_flash_source_queue", resource.hbf_flash_source_queue);
        resource_row("hbf_channel_command", resource.hbf_channel_command);
        resource_row("hbf_channel_data", resource.hbf_channel_data);
        resource_row("hbf_tsv", resource.hbf_tsv);
        resource_row("hbf_sram", resource.hbf_sram);
        resource_row("hbf_hbio_command", resource.hbf_hbio_command);
        resource_row("hbf_hbio_data", resource.hbf_hbio_data);
        resource_row("hbf_sequencer", resource.hbf_sequencer);
        resource_row("hbf_ecc_issue", resource.hbf_ecc_issue);
    }
    if (result.has_external_backing) {
        resource_row(
            "external_controller", resource.external_controller);
        resource_row("external_media", resource.external_media);
        resource_row("external_link_m2s", resource.external_link_m2s);
        resource_row("external_link_s2m", resource.external_link_s2m);
    }
    if (result.base_die_link_stats.read_transfers != 0) {
        resource_row("base_die_link_read", resource.base_die_link_read);
    }
    if (result.base_die_link_stats.write_transfers != 0) {
        resource_row("base_die_link_write", resource.base_die_link_write);
    }
}

void print_address_heatmap_summary(const ScenarioResult& result) {
    if (!result.address_heatmap) {
        std::cout << "  Address heatmap: disabled (--address-heatmap-bins 0; "
                     "pass 1..8192 to collect one)\n";
        return;
    }
    std::cout << "  Address heatmap: logical traffic is request bytes; HBM physical "
                 "traffic is full bursts; HBF physical traffic is full pages/blocks.\n";
    std::cout << "    domain,size_bytes,read_bytes,write_bytes,erase_bytes,active_bins,"
                 "total_bins,hottest_read_range,hottest_read_bytes,"
                 "hottest_write_range,hottest_write_bytes\n";
    for (const auto& domain : result.address_heatmap->domains) {
        std::size_t active_bins = 0;
        const hbfsim::physical::AddressHeatmapBin* hottest_read = nullptr;
        const hbfsim::physical::AddressHeatmapBin* hottest_write = nullptr;
        for (const auto& bin : domain.bins) {
            if (bin.total.read_bytes != 0 || bin.total.write_bytes != 0 ||
                bin.total.erase_bytes != 0) {
                ++active_bins;
            }
            if (bin.total.read_bytes != 0 && (hottest_read == nullptr ||
                bin.total.read_bytes > hottest_read->total.read_bytes)) {
                hottest_read = &bin;
            }
            if (bin.total.write_bytes != 0 && (hottest_write == nullptr ||
                bin.total.write_bytes > hottest_write->total.write_bytes)) {
                hottest_write = &bin;
            }
        }
        const auto range = [](const hbfsim::physical::AddressHeatmapBin* bin) {
            if (bin == nullptr) {
                return std::string{"n/a"};
            }
            std::ostringstream text;
            text << "[0x" << std::hex << bin->begin << ",0x";
            if (bin->end.is_full_address_space_end()) {
                text << "10000000000000000";
            } else {
                text << bin->end.finite_value();
            }
            text << ')';
            return text.str();
        };
        std::cout << "    " << hbfsim::physical::to_string(domain.domain) << ','
                  << hbfsim::physical::address_boundary_decimal(domain.size_bytes) << ','
                  << domain.total.read_bytes << ','
                  << domain.total.write_bytes << ',' << domain.total.erase_bytes << ','
                  << active_bins << ',' << domain.bins.size() << ','
                  << range(hottest_read) << ','
                  << (hottest_read == nullptr ? 0 : hottest_read->total.read_bytes) << ','
                  << range(hottest_write) << ','
                  << (hottest_write == nullptr ? 0 : hottest_write->total.write_bytes)
                  << '\n';
    }
}

std::string optional_fixed_text(std::optional<double> value, int precision = 3) {
    return value ? fixed(*value, precision) : std::string{"n/a"};
}

void print_offered_load(const ScenarioResult& result) {
    if (!result.offered_load) {
        return;
    }
    const auto& load = *result.offered_load;
    std::cout << "  Offered: rate_GBps=" << optional_fixed_text(load.rate_GBps, 1)
              << " (reads " << optional_fixed_text(load.read_rate_GBps, 1)
              << ", writes " << optional_fixed_text(load.write_rate_GBps, 1)
              << ") arrival_span_ns=" << fixed(load.arrival_span_ns)
              << " arrival_model="
              << (load.open_loop ? "open-loop" : "closed-loop")
              << " hbm_issue_mode=" << kHbmIssueMode
              << " offered_to_peak: hbm="
              << optional_fixed_text(load.hbm_ratio)
              << " hbf=" << optional_fixed_text(load.hbf_ratio)
              << " external=" << optional_fixed_text(load.external_ratio)
              << '\n';
}

void print_device_stats(const ScenarioResult& result) {
    std::cout << "\n[" << result.name << "]\n";
    print_offered_load(result);
    if (result.has_hbm) {
        const auto& s = result.hbm_stats;
        std::cout << "  HBM: reads=" << s.read_bytes
                  << " writes=" << s.write_bytes
                  << " scheduler_queue_wait_work_ns="
                  << fixed(s.stage_work.scheduler_queue_wait_ns)
                  << " max_queue_occupancy=" << s.max_queue_occupancy
                  << " channel_transfers=" << s.channel_transfers
                  << " service_quanta=" << s.service_quanta
                  << " bus_busy_ns=" << fixed(s.bus_busy_ns)
                  << " util=" << fixed(s.utilization() * 100.0) << "%"
                  << " bus_parallelism=" << fixed(s.bus_parallelism(), 2)
                  << " active_pch=" << s.active_pseudo_channels << "/" << s.pseudo_channels
                  << " pch_busy_skew=" << fixed(s.pseudo_channel_busy_skew(), 2)
                  << " finish_ns=" << fixed(s.finish_ns)
                  << '\n';
    }
    if (result.has_hbf) {
        const auto& s = result.hbf_stats;
        std::cout << "  HBF: logical_read=" << s.logical_read_bytes
                  << " physical_read=" << s.physical_read_bytes
                  << " logical_write=" << s.logical_write_bytes
                  << " physical_write=" << s.physical_write_bytes
                  << " page_reads=" << s.page_reads
                  << " data_programs=" << s.data_programs
                  << " page_programs=" << s.page_programs
                  << " block_erases=" << s.block_erases
                  << " mappings=" << s.mapping_entries
                  << " total_pages=" << s.total_pages
                  << " free_pages=" << s.free_pages
                  << " valid_pages=" << s.valid_pages
                  << " invalid_pages=" << s.invalid_pages
                  << " pending_program_pages=" << s.pending_program_pages
                  << " pending_mapping_publications="
                  << s.pending_mapping_publications
                  << " static_unmaterialized_pages="
                  << s.static_unmaterialized_pages
                  << " accounting_verified="
                  << (s.accounting_verified ? "yes" : "no")
                  << " waf="
                  << (s.waf() ? fixed(*s.waf(), 3) : "n/a")
                  << " mapping_table_bytes=" << s.mapping_table_bytes
                  << " resident_mapping_bytes="
                  << s.resident_mapping_table_bytes
                  << " resident_mapping_bytes_per_stack="
                  << s.resident_mapping_table_bytes_per_stack
                  << " resident_mapping_pages_per_stack="
                  << s.resident_mapping_pages_per_stack
                  << " mapping_directory_bytes="
                  << s.mapping_directory_bytes
                  << " mapping_directory_bytes_per_stack="
                  << s.mapping_directory_bytes_per_stack
                  << " mapping_cache_bytes="
                  << s.mapping_cache_capacity_bytes
                  << " mapping_cache_hits=" << s.mapping_cache_hits
                  << " mapping_cache_misses=" << s.mapping_cache_misses
                  << " mapping_cache_coalesced_misses="
                  << s.mapping_cache_coalesced_misses
                  << " mapping_cache_evictions="
                  << s.mapping_cache_evictions
                  << " mapping_media_reads=" << s.mapping_media_reads
                  << " mapping_lookups=" << s.mapping_lookup_ops
                  << " mapping_user_lookups=" << s.mapping_user_lookup_ops
                  << " mapping_gc_lookups=" << s.mapping_gc_lookup_ops
                  << " mapping_updates=" << s.mapping_update_ops
                  << " mapping_user_updates=" << s.mapping_user_update_ops
                  << " mapping_gc_updates=" << s.mapping_gc_update_ops
                  << " mapping_dram_waits=" << s.mapping_dram_wait_ops
                  << " mapping_dram_wait_work_ns="
                  << fixed(s.mapping_dram_wait_ns)
                  << " mapping_dram_wait_max_ns="
                  << fixed(s.mapping_dram_wait_max_ns)
                  << " mapping_dram_issue_busy_ns="
                  << fixed(s.mapping_dram_issue_busy_ns)
                  << " page_read_admission_waited="
                  << s.page_read_admission_waited_pages
                  << " page_read_admission_wait_ns="
                  << fixed(s.page_read_admission_wait_ns)
                  << " page_read_admission_max_wait_ns="
                  << fixed(s.page_read_admission_max_wait_ns)
                  << " scalar_read_requests=" << s.scalar_read_requests
                  << " scalar_read_pages=" << s.scalar_read_pages
                  << " mapping_programs=" << s.mapping_page_programs
                  << " read_buffer_hits=" << s.read_buffer_hits
                  << " write_buffer_hits=" << s.write_buffer_hits
                  << " write_buffer_misses=" << s.write_buffer_misses
                  << " write_buffer_flushes=" << s.write_buffer_flushes
                  << " write_buffer_read_hits=" << s.write_buffer_read_hits
                  << " write_buffer_read_bytes=" << s.write_buffer_read_bytes
                  << " write_buffer_slot_waits=" << s.write_buffer_slot_wait_ops
                  << " write_buffer_slot_wait_work_ns="
                  << fixed(s.write_buffer_slot_wait_ns)
                  << " read_splits=" << s.read_splits
                  << " flash_sched_enq=" << s.flash_scheduler_enqueues
                  << " flash_sched_issue=" << s.flash_scheduler_issues
                  << " gc_runs=" << s.gc_runs
                  << " gc_relocations=" << s.gc_relocations
                  << " gc_data_relocations=" << s.gc_data_relocations
                  << " gc_mapping_relocations=" << s.gc_mapping_relocations
                  << " gc_reclaimed_invalid_pages="
                  << s.gc_reclaimed_invalid_pages
                  << " ingress_queue_wait_work_ns="
                  << fixed(s.stage_work.ingress_queue_wait_ns)
                  << " scheduler_queue_wait_work_ns="
                  << fixed(s.stage_work.scheduler_queue_wait_ns)
                  << " ecc_queue_wait_work_ns="
                  << fixed(s.stage_work.ecc_queue_wait_ns)
                  << " ecc_decode_ops=" << s.ecc_decode_ops
                  << " ecc_encode_ops=" << s.ecc_encode_ops
                  << " ecc_codeword_bytes=" << s.ecc_codeword_bytes
                  << " ecc_issue_busy_ns=" << fixed(s.ecc_issue_busy_ns)
                  << " ecc_issue_util=" << fixed(s.ecc_issue_utilization() * 100.0) << "%"
                  << " ecc_issue_parallelism=" << fixed(s.ecc_issue_parallelism(), 2)
                  << " ecc_active_dies=" << s.active_ecc_dies << "/" << s.dies
                  << " ecc_max_inflight_per_die=" << s.max_ecc_inflight_per_die
                  << " array_read_work_ns=" << fixed(s.stage_work.array_read_ns)
                  << " array_program_work_ns="
                  << fixed(s.stage_work.array_program_ns)
                  << " array_erase_work_ns=" << fixed(s.stage_work.array_erase_ns)
                  << " media_util=" << fixed(s.media_utilization() * 100.0) << "%"
                  << " media_parallelism=" << fixed(s.media_parallelism(), 2)
                  << " read_lane_parallelism=" << fixed(s.read_lane_parallelism(), 2)
                  << " subarray_read_parallelism=" << fixed(s.subarray_read_parallelism(), 2)
                  << " page_buffer_bank_parallelism="
                  << fixed(s.page_buffer_bank_parallelism(), 2)
                  << " active_planes=" << s.active_planes << "/" << s.planes
                  << " active_media_lanes=" << s.active_media_lanes << "/" << s.media_lanes
                  << " active_subarrays=" << s.active_subarrays << "/" << s.subarrays
                  << " active_page_buffer_banks=" << s.active_page_buffer_banks
                  << "/" << s.page_buffer_banks
                  << " plane_media_skew=" << fixed(s.plane_media_skew(), 2)
                  << " media_lane_skew=" << fixed(s.media_lane_skew(), 2)
                  << " subarray_busy_skew=" << fixed(s.subarray_busy_skew(), 2)
                  << " subarray_read_skew=" << fixed(s.subarray_read_skew(), 2)
                  << " page_buffer_bank_skew=" << fixed(s.page_buffer_bank_skew(), 2)
                  << " page_buffer_bank_read_skew="
                  << fixed(s.page_buffer_bank_read_skew(), 2)
                  << " io_util=" << fixed(s.io_utilization() * 100.0) << "%"
                  << " hbio_cmd_util="
                  << fixed(s.hbio_command_utilization() * 100.0) << "%"
                  << " hbio_data_util="
                  << fixed(s.hbio_data_utilization() * 100.0) << "%"
                  << " channel_parallelism=" << fixed(s.channel_parallelism(), 2)
                  << " active_channels=" << s.active_channels << "/" << s.channels
                  << " sequencer_parallelism=" << fixed(s.sequencer_parallelism(), 2)
                  << " active_dies=" << s.active_dies << "/" << s.dies
                  << " finish_ns=" << fixed(s.finish_ns)
                  << '\n';
    }
    if (result.has_external_backing) {
        const auto& s = result.external_backing_stats;
        std::cout << "  External backing: kind="
                  << hbfsim::physical::external::to_string(s.kind)
                  << " reads=" << s.read_requests
                  << " writes=" << s.write_requests
                  << " read_bytes=" << s.read_bytes
                  << " write_bytes=" << s.write_bytes
                  << " ranges=" << s.page_run_requests
                  << " transport_requests=" << s.page_run_segments
                  << " address_pages=" << s.page_run_pages
                  << " media_channels_per_queue="
                  << s.media_channels_per_queue
                  << " read_queues=" << s.media_read_queues
                  << " write_queues=" << s.media_write_queues
                  << " active_media_resources="
                  << s.active_media_resources
                  << "/" << s.media_channels
                  << " max_device_outstanding="
                  << s.max_device_outstanding
                  << " outstanding_wait_work_ns="
                  << fixed(s.outstanding_wait_ns)
                  << " controller_queue_wait_work_ns="
                  << fixed(s.controller_queue_wait_ns)
                  << " controller_util="
                  << fixed(s.controller_utilization() * 100.0) << "%"
                  << " media_queue_wait_work_ns="
                  << fixed(s.media_queue_wait_ns)
                  << " media_read_latency_work_ns="
                  << fixed(s.media_read_latency_work_ns)
                  << " media_write_latency_work_ns="
                  << fixed(s.media_write_latency_work_ns)
                  << " media_read_busy_ns="
                  << fixed(s.media_read_busy_ns)
                  << " media_write_busy_ns="
                  << fixed(s.media_write_busy_ns)
                  << " media_util="
                  << fixed(s.media_utilization() * 100.0) << "%"
                  << " m2s_wire_bytes=" << s.m2s_wire_bytes
                  << " s2m_wire_bytes=" << s.s2m_wire_bytes
                  << " m2s_util="
                  << fixed(s.m2s_utilization() * 100.0) << "%"
                  << " s2m_util="
                  << fixed(s.s2m_utilization() * 100.0) << "%"
                  << " finish_ns=" << fixed(s.finish_ns)
                  << '\n';
    }
    if (result.hbm_write_buffer_user_write_bytes != 0 ||
        result.hbm_write_buffer_peak_bytes != 0) {
        std::cout << "  Cooperative write region: user_write_bytes="
                  << result.hbm_write_buffer_user_write_bytes
                  << " destaged_bytes=" << result.hbm_write_buffer_destaged_bytes
                  << " peak_bytes=" << result.hbm_write_buffer_peak_bytes
                  << " full_waits=" << result.hbm_write_buffer_full_waits
                  << " wait_work_ns=" << fixed(result.hbm_write_buffer_wait_ns);
        if (result.hbm_write_buffer_user_write_bytes != 0 && result.has_hbf) {
            std::cout << " end_to_end_media_bytes_per_user_write_byte="
                      << fixed(
                          static_cast<double>(result.hbf_stats.physical_write_bytes) /
                          static_cast<double>(
                              result.hbm_write_buffer_user_write_bytes),
                          3);
        }
        std::cout << '\n';
    }
    for (const auto& warning : result.warnings) {
        std::cout << "  warning: " << warning << '\n';
    }
    if (result.behavioral_tiering_stats) {
        const auto& s = *result.behavioral_tiering_stats;
        std::cout << "  Behavioral tiering: policy="
                  << hbfsim::policy::to_string(
                         s.admission_policy)
                  << " semantic_inputs_consumed="
                  << (s.semantic_inputs_consumed ? "yes" : "no")
                  << " hbm_tier_pages=" << s.hbm_tier_pages
                  << " history_capacity_pages="
                  << s.history_capacity_pages
                  << " observations=" << s.page_observations
                  << " hbm_hits=" << s.hbm_hits
                  << " hbf_bypasses=" << s.hbf_bypasses
                  << " promotions=" << s.promotions
                  << " clean_evictions=" << s.clean_evictions
                  << " dirty_evictions=" << s.dirty_evictions
                  << " backing_fill_bytes=" << s.backing_fill_bytes
                  << " dirty_writeback_bytes="
                  << s.dirty_writeback_bytes
                  << " transition_waited_transactions="
                  << s.transition_waited_transactions
                  << " transition_wait_work_ns="
                  << fixed(s.transition_wait_work_ns)
                  << " final_resident_pages="
                  << s.final_resident_pages
                  << " final_dirty_pages="
                  << s.final_dirty_pages
                  << " decision_fingerprint="
                  << s.decision_fingerprint
                  << '\n';
    }
    if (result.layer_streaming_stats) {
        const auto& s = *result.layer_streaming_stats;
        std::cout << "  Layer streaming: mode="
                  << layer_streaming_mode_name(s)
                  << " backing="
                  << hbfsim::policy::to_string(s.backing)
                  << " residency_policy="
                  << layer_streaming_residency_policy_name(s)
                  << " compact_resident_mapping="
                  << (s.compact_resident_mapping ? "yes" : "no")
                  << " semantic_inputs_consumed="
                  << (s.semantic_inputs_consumed ? "yes" : "no")
                  << " explicit_residency_contract="
                  << (s.explicit_residency_contract ? "yes" : "no")
                  << " address_footprint_bytes="
                  << s.address_footprint_bytes
                  << " unique_resident_footprint_pages="
                  << s.unique_resident_footprint_pages
                  << " unique_resident_footprint_bytes="
                  << s.unique_resident_footprint_bytes
                  << " capacity_pressure_basis_bytes="
                  << s.capacity_pressure_basis_bytes
                  << " footprint_page_rounding_bytes="
                  << s.footprint_page_rounding_bytes
                  << " hbm_capacity_bytes=" << s.hbm_capacity_bytes
                  << " hbm_capacity_pressure="
                  << fixed(s.hbm_capacity_pressure, 6)
                  << " layers=" << s.layers
                  << " explicit_layer_requests=" << s.explicit_layer_requests
                  << " explicit_compute_layers="
                  << s.explicit_compute_layers
                  << " compute_work_ns=" << fixed(s.compute_work_ns)
                  << " hbm_only_resident_pages="
                  << s.hbm_only_resident_pages
                  << " hbm_only_resident_bytes="
                  << s.hbm_only_resident_bytes
                  << " hot_kv_candidate_pages="
                  << s.hot_kv_candidate_pages
                  << " hot_kv_candidate_bytes="
                  << s.hot_kv_candidate_bytes
                  << " hot_kv_resident_pages="
                  << s.hot_kv_resident_pages
                  << " hot_kv_resident_bytes="
                  << s.hot_kv_resident_bytes
                  << " data_pages=" << s.data_pages
                  << " data_bytes=" << s.data_bytes
                  << " model_weight_resident_pages="
                  << s.model_weight_resident_pages
                  << " model_weight_resident_bytes="
                  << s.model_weight_resident_bytes
                  << " model_weight_backing_pages="
                  << s.model_weight_backing_pages
                  << " model_weight_backing_bytes="
                  << s.model_weight_backing_bytes
                  << " cold_kv_backing_pages="
                  << s.cold_kv_backing_pages
                  << " cold_kv_backing_bytes="
                  << s.cold_kv_backing_bytes
                  << " unknown_backing_pages="
                  << s.unknown_backing_pages
                  << " unknown_backing_bytes="
                  << s.unknown_backing_bytes
                  << " backing_unique_pages="
                  << s.backing_unique_pages
                  << " backing_unique_bytes="
                  << s.backing_unique_bytes
                  << " resident_physical_pages="
                  << s.resident_physical_pages
                  << " resident_physical_bytes="
                  << s.resident_physical_bytes
                  << " effective_layer_buffer_pages="
                  << s.effective_layer_buffer_pages
                  << " effective_layer_buffer_bytes="
                  << s.effective_layer_buffer_bytes
                  << " unused_hbm_pages="
                  << s.unused_hbm_pages
                  << " unused_hbm_bytes="
                  << s.unused_hbm_bytes
                  << " streamed_pages=" << s.streamed_pages
                  << " streamed_bytes=" << s.streamed_bytes
                  << " foreground_resident_page_accesses="
                  << s.foreground_resident_page_accesses
                  << " foreground_buffer_page_accesses="
                  << s.foreground_buffer_page_accesses
                  << " dirty_pages_written_back="
                  << s.dirty_pages_written_back
                  << " writeback_bytes=" << s.writeback_bytes
                  << " max_layer_data_pages="
                  << s.max_layer_data_pages
                  << " max_layer_data_bytes="
                  << s.max_layer_data_bytes
                  << " immutable_weight_logical_bytes="
                  << s.immutable_weight_logical_bytes
                  << " runtime_overhead_logical_bytes="
                  << s.runtime_overhead_logical_bytes
                  << " block_table_logical_bytes="
                  << s.block_table_logical_bytes
                  << " active_buffer_logical_bytes_per_slot="
                  << s.active_buffer_logical_bytes_per_slot
                  << " residency_page_size_bytes="
                  << s.residency_page_size_bytes
                  << " kv_block_stride_bytes="
                  << s.kv_block_stride_bytes
                  << " logical_kv_blocks="
                  << s.logical_kv_blocks
                  << " hot_kv_blocks="
                  << s.hot_kv_blocks
                  << " cold_kv_blocks="
                  << s.cold_kv_blocks
                  << " backing_request_credit_limit="
                  << s.backing_request_credit_limit
                  << " backing_max_inflight_requests="
                  << s.backing_max_inflight_requests
                  << " backing_admission_waited_requests="
                  << s.backing_admission_waited_requests
                  << " backing_admission_wait_work_ns="
                  << fixed(s.backing_admission_wait_work_ns)
                  << " backing_admission_max_wait_ns="
                  << fixed(s.backing_admission_max_wait_ns)
                  << " user_waited_ops=" << s.user_waited_ops
                  << " user_wait_work_ns=" << fixed(s.user_wait_work_ns)
                  << " user_max_wait_ns=" << fixed(s.user_max_wait_ns)
                  << " exposed_prefetch_ns=" << fixed(s.exposed_prefetch_ns)
                  << " hidden_prefetch_ns=" << fixed(s.hidden_prefetch_ns)
                  << " buffer_reuse_wait_work_ns="
                  << fixed(s.buffer_reuse_wait_work_ns)
                  << " background_hbf_writes=" << result.background_hbf_writes
                  << " hbm_user_accesses=" << result.hbm_user_accesses
                  << " hbm_background_accesses=" << result.hbm_background_accesses
                  << " hbf_user_accesses=" << result.hbf_user_accesses
                  << " hbf_background_accesses=" << result.hbf_background_accesses
                  << " hbf_static_read_bytes=" << result.hbf_static_read_bytes
                  << " hbf_logical_read_bytes=" << result.hbf_logical_read_bytes
                  << " hbf_backing_write_bytes=" << result.hbf_backing_write_bytes
                  << " external_background_accesses="
                  << result.external_background_accesses
                  << " external_backing_read_bytes="
                  << result.external_backing_read_bytes
                  << " external_backing_write_bytes="
                  << result.external_backing_write_bytes
                  << " hbm_streaming_write_bytes="
                  << result.hbm_streaming_write_bytes
                  << " base_link_read_bytes="
                  << result.base_die_link_stats.read_bytes
                  << " base_link_write_bytes="
                  << result.base_die_link_stats.write_bytes
                  << " base_link_read_util="
                  << fixed(result.base_die_link_stats.read_utilization() * 100.0)
                  << "%"
                  << " base_link_write_util="
                  << fixed(result.base_die_link_stats.write_utilization() * 100.0)
                  << "%"
                  << '\n';
    }
    print_time_breakdown(result);
    print_address_heatmap_summary(result);
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string out = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            out += "\"\"";
        } else {
            out += ch;
        }
    }
    out += '"';
    return out;
}

// The inverse of SystemConfigBuilder::apply for a resolved SystemConfig,
// written in the builder's own key vocabulary so `--config-out` replays
// through the same parser the core `hbfsim` uses. Keys with an alternative
// spelling (hbf-capacity-*, hbf-ctrl-dram-*) appear in resolved form only;
// conditional blocks (external device cache, HBF<->external direct link) are
// written only when enabled. The writer then proves it covered every
// engine-owned key: a key added to the builder but not here fails loudly
// instead of silently dropping out of replayed configs. A key retired from
// the builder is skipped here automatically.
void write_system_config_keys(std::ostream& out, const SystemConfig& system) {
    std::set<std::string> emitted;
    const auto put = [&](const char* key, const auto& value) {
        if (!SystemConfigBuilder::owns_key(key)) {
            return;
        }
        emitted.insert(key);
        out << key << '=' << value << '\n';
    };
    const auto put_bool = [&](const char* key, bool value) {
        put(key, value ? "true" : "false");
    };

    const auto& hbm = system.hbm;
    put("hbm-capacity-bytes", hbm.device.capacity_bytes);
    put("hbm-stacks", hbm.device.stacks);
    put("hbm-channels", hbm.device.channels_per_stack);
    put("hbm-pseudo-channels", hbm.device.pseudo_channels_per_channel);
    put("hbm-bank-groups-per-pseudo-channel", hbm.device.bank_groups_per_pseudo_channel);
    put("hbm-banks-per-group", hbm.device.banks_per_group);
    put("hbm-channel-row-size-bytes", hbm.device.channel_row_size_bytes);
    put("hbm-channel-width-bits", hbm.device.channel_width_bits);
    put("hbm-burst-length", hbm.device.burst_length);
    put("hbm-pin-rate-gbps", hbm.device.pin_rate_Gbps);
    put("hbm-data-rate-per-command-clock", hbm.device.data_rate_per_command_clock);
    put("hbm-address-mapping-ns", hbm.controller.address_mapping_ns);
    put("hbm-read-latency-ns", hbm.timing.read_latency_ns);
    put("hbm-write-latency-ns", hbm.timing.write_latency_ns);
    put("hbm-read-to-write-ns", hbm.timing.read_to_write_ns);
    put("hbm-write-to-read-ns", hbm.timing.write_to_read_ns);
    put("hbm-bandwidth-efficiency", hbm.timing.bandwidth_efficiency);
    put("hbm-service-quantum-bytes", hbm.controller.service_quantum_bytes);
    put("hbm-service-group-channels", hbm.controller.service_group_channels);
    put("hbm-queue-depth", hbm.controller.queue_depth);
    put("hbm-interleave-bytes", hbm.effective_interleave_bytes());

    const auto& hbf = system.hbf;
    put("hbf-processor-interconnect", system.hbf_processor_interconnect);
    put("hbf-standard", hbfsim::physical::hbf::kStandard);
    put("hbm-standard", hbfsim::physical::hbm::kStandard);
    put("hbf-speed-grade", hbf.device.speed_grade);
    put("hbf-host-gc-decision-ns", hbf.host.host_gc_decision_ns);
    put("hbf-zone-size-blocks", hbf.host.zone_size_blocks);
    put("hbf-host-zone-wear-gap", hbf.host.host_zone_wear_gap);
    put("hbf-host-zone-remap-ns", hbf.host.host_zone_remap_ns);
    put("hbf-stacks", hbf.device.stacks);
    put("hbf-channels", hbf.device.channels_per_stack);
    put("hbf-dies-per-channel", hbf.device.dies_per_channel);
    put("hbf-planes-per-die", hbf.device.planes_per_die);
    put("hbf-blocks-per-plane", hbf.device.blocks_per_plane);
    put("hbf-pages-per-block", hbf.device.pages_per_block);
    put("hbf-page-size", hbf.device.page_size_bytes);
    put("hbf-oob-bytes", hbf.device.oob_bytes_per_page);
    put("hbf-media-lanes-per-plane", hbf.device.media_lanes_per_plane);
    put("hbf-page-buffer-banks-per-plane", hbf.device.page_buffer_banks_per_plane);
    put("hbf-read-ns", hbf.device.t_read_page_ns);
    put("hbf-program-ns", hbf.device.t_program_page_ns);
    put("hbf-erase-ns", hbf.device.t_erase_block_ns);
    put("hbf-ecc-decode-latency-ns", hbf.device.ecc_decode_latency_ns);
    put("hbf-ecc-encode-latency-ns", hbf.device.ecc_encode_latency_ns);
    put("hbf-ecc-decode-raw-bw", hbf.device.ecc_decode_raw_bandwidth_GBps_per_die);
    put("hbf-ecc-encode-raw-bw", hbf.device.ecc_encode_raw_bandwidth_GBps_per_die);
    put("hbf-channel-bw", hbf.device.channel_bandwidth_GBps);
    put("hbf-tsv-bw", hbf.device.tsv_bandwidth_GBps);
    put("hbf-media-lane-bw", hbf.device.media_lane_bandwidth_GBps);
    put("hbf-logic-sram-bw", hbf.device.logic_sram_bandwidth_GBps);
    put("hbf-page-buffer-bw", hbf.device.page_buffer_bandwidth_GBps);
    put("hbf-mapping-mode", hbfsim::host::to_string(hbf.host.mapping_mode));
    put("hbf-mapping-organization", hbfsim::host::to_string(hbf.host.mapping_organization));
    put("hbf-mapping-log-blocks", hbf.host.mapping_log_blocks);
    put("hbf-mapping-superblock-planes", hbf.host.mapping_superblock_planes);
    put("hbf-mapping-cache-layout", hbfsim::host::to_string(hbf.host.mapping_cache_layout));
    put("hbf-mapping-cache-tag-bytes", hbf.host.mapping_cache_tag_bytes);
    put("hbf-mapping-codec-ns-per-entry", hbf.host.mapping_codec_ns_per_entry);
    put("hbf-mapping-scratch-pages", hbf.host.mapping_scratch_pages);
    put("hbf-logical-capacity-bytes", hbf.host.logical_capacity_bytes);
    if (hbf.host.ctrl_dram_capacity_denominator != 0) {
        put("hbf-ctrl-dram-capacity-denominator",
            hbf.host.ctrl_dram_capacity_denominator);
    } else {
        put("hbf-ctrl-dram-bytes", hbf.host.ctrl_dram_bytes);
    }
    put("hbf-ctrl-dram-latency-ns", hbf.host.ctrl_dram_latency_ns);
    put("hbf-ctrl-dram-issue-ns", hbf.host.ctrl_dram_issue_ns);
    put("hbf-flash-tsu-issue-ns", hbf.device.flash_tsu_issue_ns);
    put("hbf-logic-scheduler-issue-ns", hbf.device.logic_scheduler_issue_ns);
    put("hbf-page-read-queue-depth-per-stack",
        hbf.device.page_read_queue_depth_per_stack);
    put("hbf-write-accumulation-timeout-ns", hbf.device.write_accumulation_timeout_ns);
    put("hbf-outstanding-write-pages-per-channel", hbf.device.outstanding_write_pages_per_channel);
    put("hbf-gc-low-watermark-pages", hbf.host.gc_low_watermark_pages);
    put("hbf-gc-hard-watermark-pages", hbf.host.gc_hard_watermark_pages);
    put("hbf-gc-reserved-free-blocks-per-plane",
        hbf.host.gc_reserved_free_blocks_per_plane);
    put("hbf-gc-wear-leveling-weight", hbf.host.gc_wear_leveling_weight);
    put("hbf-static-wear-leveling-erase-gap",
        hbf.host.static_wear_leveling_erase_gap);
    put("hbf-static-wear-leveling-interval-erases",
        hbf.host.static_wear_leveling_interval_erases);
    put("hbf-static-wear-leveling-start-erases",
        hbf.host.static_wear_leveling_start_erases);
    put("hbf-static-wear-leveling-stop-gap", hbf.host.static_wear_leveling_stop_gap);
    put("hbf-static-wear-leveling-cooldown-erases", hbf.host.static_wear_leveling_cooldown_erases);
    put("hbf-static-wear-leveling-max-write-fraction", hbf.host.static_wear_leveling_max_write_fraction);
    put_bool("hbf-write-coalescing", hbf.host.write_coalescing_enabled);
    put_bool("hbf-write-buffer-completion-requires-flush",
        hbf.host.write_buffer_completion_requires_flush);
    put_bool("hbf-initial-free-blocks-erased", hbf.host.initial_free_blocks_erased);
    put("hbf-write-buffer-pages", hbf.host.write_buffer_pages);
    put("hbf-write-buffer-flush-threshold-pages",
        hbf.host.write_buffer_flush_threshold_pages);
    put("hbf-device-dram-capacity-denominator", hbf.host.device_dram_capacity_denominator);
    put("hbf-device-dram-hit-latency-ns", hbf.host.device_dram_hit_latency_ns);
    put("hbf-device-dram-bw", hbf.host.device_dram_bandwidth_GBps);
    put("hbf-device-dram-policy", hbfsim::host::to_string(hbf.host.device_dram_policy));
    put_bool("hbf-thermal-enable", hbf.device.thermal_enabled);
    put("hbf-thermal-ambient-c", hbf.device.thermal_ambient_c);
    put("hbf-thermal-resistance-c-per-w", hbf.device.thermal_resistance_c_per_w);
    put("hbf-thermal-capacitance-j-per-c", hbf.device.thermal_capacitance_j_per_c);
    put("hbf-thermal-throttle-c", hbf.device.thermal_throttle_c);
    put("hbf-thermal-release-c", hbf.device.thermal_release_c);
    put("hbf-thermal-static-power-w", hbf.device.thermal_static_power_w);
    put("hbf-thermal-read-energy-pj-per-bit",
        hbf.device.thermal_read_energy_pj_per_bit);
    put("hbf-thermal-program-energy-pj-per-bit",
        hbf.device.thermal_program_energy_pj_per_bit);
    put("hbf-thermal-erase-energy-uj-per-block",
        hbf.device.thermal_erase_energy_uj_per_block);
    put("hbf-thermal-neighbor-heat-c", hbf.device.thermal_neighbor_heat_c);
    put("hbf-thermal-throttle-power-w", hbf.device.thermal_throttle_power_w);
    put("hbf-thermal-start-state",
        hbf.device.thermal_start_at_ceiling ? "throttle-ceiling" : "idle");

    put("base-die-link-read-bw", system.base_die_link.read_bandwidth_GBps);
    put("base-die-link-write-bw", system.base_die_link.write_bandwidth_GBps);
    put("base-die-link-latency-ns", system.base_die_link.latency_ns);
    put_bool("hbf-external-direct-link-enable",
        system.hbf_external_direct_link.has_value());
    if (const auto& link = system.hbf_external_direct_link) {
        put("hbf-external-direct-link-read-bw", link->read_bandwidth_GBps);
        put("hbf-external-direct-link-write-bw", link->write_bandwidth_GBps);
        put("hbf-external-direct-link-latency-ns", link->latency_ns);
    }

    // The kind precedes its overrides: the builder rejects a kind change
    // after overrides have been applied.
    const auto& external = system.external;
    put("external-backing-kind",
        hbfsim::physical::external::to_string(external.kind));
    put("external-backing-capacity-bytes", external.capacity_bytes);
    put("external-backing-page-size", external.page_size_bytes);
    put("external-backing-request-segment-bytes", external.request_segment_bytes);
    put("external-backing-media-channels", external.media_channels);
    put("external-backing-media-read-queues", external.media_read_queues);
    put("external-backing-media-write-queues", external.media_write_queues);
    put("external-backing-max-outstanding-requests",
        external.max_outstanding_requests);
    put("external-backing-controller-issue-ns", external.controller_issue_ns);
    put("external-backing-controller-processing-ns",
        external.controller_processing_ns);
    put("external-backing-media-read-latency-ns", external.media_read_latency_ns);
    put("external-backing-media-write-latency-ns",
        external.media_write_latency_ns);
    put("external-backing-media-read-bw", external.media_read_bandwidth_GBps);
    put("external-backing-media-write-bw", external.media_write_bandwidth_GBps);
    put("external-backing-m2s-bw", external.m2s_bandwidth_GBps);
    put("external-backing-s2m-bw", external.s2m_bandwidth_GBps);
    put("external-backing-one-way-propagation-ns",
        external.one_way_propagation_ns);
    put("external-backing-command-bytes", external.command_bytes);
    put("external-backing-completion-bytes", external.completion_bytes);
    if (external.device_cache.enabled) {
        put_bool("external-backing-cache-enabled", true);
        put("external-backing-cache-capacity-bytes",
            external.device_cache.capacity_bytes);
        put("external-backing-cache-ways", external.device_cache.ways);
        put("external-backing-cache-policy",
            hbfsim::physical::external::to_string(external.device_cache.policy));
        put("external-backing-cache-prefetch-degree",
            external.device_cache.prefetch_degree);
        put("external-backing-cache-prefetch-stride",
            external.device_cache.prefetch_stride);
        put("external-backing-cache-hit-latency-ns",
            external.device_cache.hit_latency_ns);
        put("external-backing-cache-hit-bw",
            external.device_cache.hit_bandwidth_GBps);
    }

    static const std::set<std::string> alternative_or_conditional{
        "hbf-capacity-bytes", "hbf-capacity-ratio",
        "hbf-ctrl-dram-bytes", "hbf-ctrl-dram-capacity-denominator",
        "hbf-external-direct-link-read-bw",
        "hbf-external-direct-link-write-bw",
        "hbf-external-direct-link-latency-ns",
        "external-backing-cache-enabled",
        "external-backing-cache-capacity-bytes",
        "external-backing-cache-ways",
        "external-backing-cache-policy",
        "external-backing-cache-prefetch-degree",
        "external-backing-cache-prefetch-stride",
        "external-backing-cache-hit-latency-ns",
        "external-backing-cache-hit-bw",
    };
    for (const auto& key : SystemConfigBuilder::owned_key_list()) {
        if (!emitted.contains(key) && !alternative_or_conditional.contains(key)) {
            throw std::runtime_error(
                "resolved-config export does not cover engine key " + key +
                "; add it to write_system_config_keys in reference_runner.cpp");
        }
    }
}

void write_resolved_config(
    const std::string& output_path,
    const Options& options,
    const SystemConfig& system) {
    ensure_parent_dir(output_path);
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open config output: " + output_path);
    }
    // A resolved config is a replay artifact, not presentation text. Preserve
    // every binary64 value exactly enough for parse -> write -> parse round
    // trips, including non-integer ECC raw-bandwidth assumptions.
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "# HBFSim " << HBFSIM_VERSION << " resolved config\n";
    if (options.trace_path) out << "trace=" << *options.trace_path << '\n';
    if (options.expected_trace_sha256) {
        out << "expected-trace-sha256="
            << *options.expected_trace_sha256 << '\n';
    }
    if (options.expected_trace_file_bytes) {
        out << "expected-trace-bytes="
            << *options.expected_trace_file_bytes << '\n';
    }
    if (options.initial_image_trace_path) {
        out << "initial-image-trace="
            << *options.initial_image_trace_path << '\n';
    }
    if (options.synthetic_sequential_read_bytes) {
        out << "synthetic-sequential-read-bytes="
            << *options.synthetic_sequential_read_bytes << '\n';
        out << "synthetic-sequential-read-base="
            << options.synthetic_sequential_read_base << '\n';
    }
    if (options.generate_semantic_llm_path) {
        out << "generate-semantic-llm=" << *options.generate_semantic_llm_path << '\n';
    }
    if (options.chrome_trace_path) out << "chrome-trace=" << *options.chrome_trace_path << '\n';
    if (options.summary_csv_path) out << "summary-csv=" << *options.summary_csv_path << '\n';
    if (options.summary_json_path) out << "summary-json=" << *options.summary_json_path << '\n';
    out << "trace-census-only="
        << (options.trace_census_only ? "true" : "false") << '\n';
    if (!options.scenarios.empty()) {
        out << "scenarios=";
        for (std::size_t i = 0; i < options.scenarios.size(); ++i) {
            if (i != 0) out << ',';
            out << options.scenarios[i];
        }
        out << '\n';
    }
    out << "line-size=" << options.line_size << '\n';
    out << "interarrival-ns=" << options.interarrival_ns << '\n';
    out << "max-ops=" << options.max_ops << '\n';
    out << "flat-hbm-bytes=" << options.flat_hbm_bytes << '\n';
    out << "static-direct-hbm-bytes=" << options.static_direct_hbm_bytes << '\n';
    out << "layer-buffer-bytes=" << options.layer_buffer_bytes << '\n';
    out << "explicit-residency-contract="
        << (options.explicit_residency_contract ? "true" : "false") << '\n';
    if (options.explicit_residency_contract) {
        out << "residency-page-size-bytes="
            << *options.residency_page_size_bytes << '\n';
        out << "residency-unique-footprint-bytes="
            << *options.residency_unique_footprint_bytes << '\n';
        out << "residency-immutable-weight-bytes="
            << *options.residency_immutable_weight_bytes << '\n';
        out << "residency-immutable-weight-pages="
            << *options.residency_immutable_weight_pages << '\n';
        out << "residency-static-weight-pages="
            << *options.residency_static_weight_pages << '\n';
        out << "residency-runtime-overhead-bytes="
            << *options.residency_runtime_overhead_bytes << '\n';
        out << "residency-block-table-bytes="
            << *options.residency_block_table_bytes << '\n';
        out << "residency-active-buffer-bytes="
            << *options.residency_active_buffer_bytes << '\n';
        out << "residency-kv-region-begin="
            << *options.residency_kv_region_begin << '\n';
        out << "residency-kv-block-stride-bytes="
            << *options.residency_kv_block_stride_bytes << '\n';
        out << "residency-logical-kv-blocks="
            << *options.residency_logical_kv_blocks << '\n';
        out << "residency-hot-kv-blocks="
            << *options.residency_hot_kv_blocks << '\n';
    }
    out << "behavioral-hbm-bytes="
        << options.behavioral_hbm_bytes << '\n';
    out << "behavioral-promotion-threshold="
        << options.behavioral_promotion_threshold << '\n';
    out << "behavioral-history-pages="
        << options.behavioral_history_pages << '\n';
    out << "max-outstanding-requests=" << options.max_outstanding_requests << '\n';
    out << "max-hbm-outstanding-requests="
        << options.max_hbm_outstanding_requests << '\n';
    out << "max-hbf-outstanding-requests="
        << options.max_hbf_outstanding_requests << '\n';
    out << "address-heatmap-bins=" << options.address_heatmap_bins << '\n';
    out << "hbf-hbm-write-buffer-bytes=" << options.hbf_hbm_write_buffer_bytes << '\n';
    out << "hbf-hbm-write-buffer-destage-watermark="
        << options.hbf_hbm_write_buffer_destage_watermark << '\n';
    if (options.generate_semantic_llm_path) {
        out << "llm-tokens=" << options.llm_tokens << '\n';
        out << "llm-layers=" << options.llm_layers << '\n';
        out << "llm-weight-base=" << options.llm_weight_base << '\n';
        out << "llm-kv-base=" << options.llm_kv_base << '\n';
        out << "llm-scratch-base=" << options.llm_scratch_base << '\n';
    }
    out << "trace-mode=" << (options.trace_mode == TraceMode::Off ? "off" :
        options.trace_mode == TraceMode::Summary ? "summary" :
        options.trace_mode == TraceMode::Sampled ? "sampled" : "full") << '\n';
    write_system_config_keys(out, system);
    finish_output(out, output_path, "resolved config");
}

void write_summary_csv(const std::string& output_path, const std::vector<ScenarioResult>& results) {
    ensure_parent_dir(output_path);
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open summary CSV output: " + output_path);
    }
    const auto& columns = result_columns();
    for (std::size_t i = 0; i < columns.size(); ++i) {
        out << (i == 0 ? "" : ",") << columns[i].header;
    }
    out << '\n';
    for (const auto& result : results) {
        for (std::size_t i = 0; i < columns.size(); ++i) {
            out << (i == 0 ? "" : ",") << columns[i].cell(result).csv;
        }
        out << '\n';
    }
    finish_output(out, output_path, "summary CSV");
}

enum class SanityVerdict { Pass, PassWithWarnings, Fail };

const char* to_string(SanityVerdict verdict) {
    switch (verdict) {
    case SanityVerdict::Pass:
        return "PASS";
    case SanityVerdict::PassWithWarnings:
        return "PASS_WITH_WARNINGS";
    case SanityVerdict::Fail:
        return "FAIL";
    }
    return "FAIL";
}

// FAIL is reserved for conservation, bounds, and contract violations: the
// numbers cannot be trusted and the process exits 1. A configuration warning
// (a scenario whose selected parameters did not exercise what it exists
// for, e.g. FLAT with a boundary above the whole footprint) keeps the run
// valid but downgrades it to PASS_WITH_WARNINGS. Advisories (open-loop
// saturation, bandwidth ceilings below HBIO) are recorded per scenario and
// echoed on stderr without changing the verdict.
struct SanityReport {
    SanityVerdict verdict = SanityVerdict::Pass;
    std::vector<std::string> notices;
};

SanityReport run_sanity_checks(
    std::vector<ScenarioResult>& results,
    const Options& options,
    const SystemConfig& system,
    std::uint64_t trace_footprint_bytes) {
    bool ok = true;
    bool configuration_warning = false;
    std::vector<std::string> notices;
    const auto& resolved_hbf = system.hbf;
    const auto& resolved_external_backing = system.external;
    auto by_name = [&results](const std::string& name) -> ScenarioResult* {
        for (auto& result : results) {
            if (result.name == name) {
                return &result;
            }
        }
        return nullptr;
    };

    for (auto& result : results) {
        if (options.address_heatmap_bins == 0 && result.address_heatmap) {
            result.warnings.push_back(
                "address heatmap snapshot exists despite being disabled");
            ok = false;
        } else if (options.address_heatmap_bins != 0 &&
                   !result.address_heatmap) {
            result.warnings.push_back("required address heatmap snapshot is missing");
            ok = false;
        } else if (result.address_heatmap) {
            const auto& heatmap = *result.address_heatmap;
            const auto& workload = heatmap.domains[
                static_cast<std::size_t>(AddressDomain::WorkloadLogical)];
            const auto& hbm_physical = heatmap.domains[
                static_cast<std::size_t>(AddressDomain::HbmPhysical)];
            const auto& hbf_physical = heatmap.domains[
                static_cast<std::size_t>(AddressDomain::HbfPhysical)];
            const auto& external_physical = heatmap.domains[
                static_cast<std::size_t>(AddressDomain::ExternalPhysical)];
            if (heatmap.bin_count != options.address_heatmap_bins ||
                workload.domain != AddressDomain::WorkloadLogical ||
                hbm_physical.domain != AddressDomain::HbmPhysical ||
                hbf_physical.domain != AddressDomain::HbfPhysical ||
                external_physical.domain !=
                    AddressDomain::ExternalPhysical) {
                result.warnings.push_back("address heatmap domain/bin contract mismatch");
                ok = false;
            }
            if (workload.total.read_bytes + workload.total.write_bytes !=
                    result.logical_bytes ||
                workload.total.erase_bytes != 0) {
                result.warnings.push_back(
                    "workload heatmap bytes do not conserve logical request bytes");
                ok = false;
            }
            const auto expected_hbm_read = result.has_hbm ? result.hbm_stats.read_bytes : 0;
            const auto expected_hbm_write = result.has_hbm ? result.hbm_stats.write_bytes : 0;
            if (hbm_physical.total.read_bytes != expected_hbm_read ||
                hbm_physical.total.write_bytes != expected_hbm_write ||
                hbm_physical.total.erase_bytes != 0) {
                result.warnings.push_back(
                    "HBM physical heatmap bytes do not conserve HBM statistics");
                ok = false;
            }
            const auto expected_hbf_read =
                result.has_hbf ? result.hbf_stats.physical_read_bytes : 0;
            const auto expected_hbf_write =
                result.has_hbf ? result.hbf_stats.physical_write_bytes : 0;
            const auto expected_hbf_erase = result.has_hbf ? checked_mul_u64(
                result.hbf_stats.block_erases,
                checked_mul_u64(
                    resolved_hbf.device.pages_per_block,
                    resolved_hbf.device.page_size_bytes,
                    "HBF heatmap block bytes"),
                "HBF heatmap erase bytes") : 0;
            if (hbf_physical.total.read_bytes != expected_hbf_read ||
                hbf_physical.total.write_bytes != expected_hbf_write ||
                hbf_physical.total.erase_bytes != expected_hbf_erase) {
                result.warnings.push_back(
                    "HBF physical heatmap bytes do not conserve HBF statistics");
                ok = false;
            }
            const auto expected_external_read = result.has_external_backing ?
                result.external_backing_stats.read_bytes : 0;
            const auto expected_external_write = result.has_external_backing ?
                result.external_backing_stats.write_bytes : 0;
            if (external_physical.total.read_bytes !=
                    expected_external_read ||
                external_physical.total.write_bytes !=
                    expected_external_write ||
                external_physical.total.erase_bytes != 0) {
                result.warnings.push_back(
                    "external physical heatmap bytes do not conserve "
                    "external-backing statistics");
                ok = false;
            }
        }
        if (result.ops == 0 ||
            result.service_latencies_ns.size() != result.ops ||
            result.offered_latencies_ns.size() != result.ops ||
            result.source_latencies_ns.size() != result.ops) {
            result.warnings.push_back("operation accounting mismatch");
            ok = false;
        }
        for (std::size_t i = 0;
             i < result.service_latencies_ns.size() &&
             i < result.offered_latencies_ns.size() &&
             i < result.source_latencies_ns.size();
             ++i) {
            const auto service = result.service_latencies_ns[i];
            const auto offered = result.offered_latencies_ns[i];
            const auto source = result.source_latencies_ns[i];
            if (!std::isfinite(service) || service < 0.0 ||
                !std::isfinite(offered) || offered < service ||
                !std::isfinite(source) || source < offered) {
                result.warnings.push_back(
                    "non-finite, negative, or non-causal latency");
                ok = false;
                break;
            }
        }
        if (!std::isfinite(result.first_arrival_ns) ||
            !std::isfinite(result.last_arrival_ns) ||
            !std::isfinite(result.user_finish_ns) ||
            !std::isfinite(result.finish_ns) ||
            result.first_arrival_ns < 0.0 ||
            result.last_arrival_ns < result.first_arrival_ns ||
            result.user_finish_ns < result.last_arrival_ns ||
            result.finish_ns < result.user_finish_ns) {
            result.warnings.push_back("invalid scenario timeline ordering");
            ok = false;
        }
        if (result.has_hbm && result.hbm_stats.finish_ns > result.finish_ns + 1e-6) {
            result.warnings.push_back("HBM device finish exceeds scenario finish");
            ok = false;
        }
        if (result.has_hbm &&
            result.hbm_stats.max_queue_occupancy > system.hbm.controller.queue_depth) {
            result.warnings.push_back("HBM queue occupancy exceeded configured depth");
            ok = false;
        }
        if (result.has_hbm) {
            const double utilization = result.hbm_stats.utilization();
            const double parallelism = result.hbm_stats.bus_parallelism();
            const double pseudo_channels =
                static_cast<double>(result.hbm_stats.pseudo_channels);
            if (!std::isfinite(utilization) || utilization < 0.0 ||
                utilization > 1.0 + 1e-9) {
                result.warnings.push_back(
                    "HBM bus utilization falls outside the physical [0, 1] bound");
                ok = false;
            }
            if (!std::isfinite(parallelism) || parallelism < 0.0 ||
                parallelism > pseudo_channels + 1e-9) {
                result.warnings.push_back(
                    "HBM bus parallelism exceeds the pseudo-channel count");
                ok = false;
            }
        }
        if (result.has_hbf && result.hbf_stats.finish_ns > result.finish_ns + 1e-6) {
            result.warnings.push_back("HBF device finish exceeds scenario finish");
            ok = false;
        }
        if (result.has_hbf) {
            const double codeword_bytes = static_cast<double>(
                resolved_hbf.device.page_size_bytes + resolved_hbf.device.oob_bytes_per_page);
            const double dies_per_stack = static_cast<double>(
                resolved_hbf.device.channels_per_stack) * resolved_hbf.device.dies_per_channel;
            const double payload_fraction =
                static_cast<double>(resolved_hbf.device.page_size_bytes) / codeword_bytes;
            const double decode_ceiling_GBps = dies_per_stack *
                resolved_hbf.device.ecc_decode_raw_bandwidth_GBps_per_die *
                payload_fraction;
            const double encode_ceiling_GBps = dies_per_stack *
                resolved_hbf.device.ecc_encode_raw_bandwidth_GBps_per_die *
                payload_fraction;
            const double channel_ceiling_GBps =
                static_cast<double>(resolved_hbf.device.channels_per_stack) *
                resolved_hbf.device.channel_bandwidth_GBps * payload_fraction;
            const double tsv_wire_bytes = codeword_bytes +
                static_cast<double>(resolved_hbf.device.command_address_bytes);
            const double tsv_payload_fraction =
                static_cast<double>(resolved_hbf.device.page_size_bytes) / tsv_wire_bytes;
            const double tsv_ceiling_GBps =
                resolved_hbf.device.tsv_bandwidth_GBps * tsv_payload_fraction;
            const double hbio_GBps = resolved_hbf.device.hb_io_bandwidth_GBps();
            if (decode_ceiling_GBps + 1e-9 < hbio_GBps) {
                result.warnings.push_back(
                    "configured ECC decode payload ceiling is below HBF HBIO bandwidth");
            }
            if (encode_ceiling_GBps + 1e-9 < hbio_GBps) {
                result.warnings.push_back(
                    "configured ECC encode payload ceiling is below HBF HBIO bandwidth");
            }
            if (channel_ceiling_GBps + 1e-9 < hbio_GBps) {
                result.warnings.push_back(
                    "aggregate raw channel payload ceiling is below HBF HBIO bandwidth");
            }
            if (tsv_ceiling_GBps + 1e-9 < hbio_GBps) {
                result.warnings.push_back(
                    "shared TSV command+raw payload ceiling is below HBF HBIO bandwidth");
            }
        }
        const auto hbf_served_read_bytes = result.has_hbf ?
            checked_add_u64(
                checked_add_u64(
                    result.hbf_stats.physical_read_bytes,
                    result.hbf_stats.write_buffer_read_bytes,
                    "HBF served read byte count"),
                checked_add_u64(
                    result.hbf_stats.read_buffer_read_bytes,
                    result.hbf_stats.device_dram_read_hit_bytes,
                    "HBF served read byte count"),
                "HBF served read byte count") : 0;
        if (result.has_hbf &&
            hbf_served_read_bytes < result.hbf_stats.logical_read_bytes) {
            result.warnings.push_back(
                "HBF served read bytes below logical read bytes");
            ok = false;
        }
        const auto hbf_host_write_bytes = result.has_hbf ?
            checked_add_u64(
                result.hbf_stats.logical_write_bytes,
                result.hbf_stats.raw_physical_program_payload_bytes,
                "HBF host-write byte count") : 0;
        if (result.has_hbf && !resolved_hbf.host.write_coalescing_enabled &&
            result.hbf_stats.physical_write_bytes < hbf_host_write_bytes) {
            result.warnings.push_back("HBF physical write bytes below host write bytes");
            ok = false;
        }
        if (result.has_external_backing) {
            const auto& stats = result.external_backing_stats;
            const auto expected_media_resources =
                static_cast<std::uint64_t>(std::max(
                    resolved_external_backing.media_read_queues,
                    resolved_external_backing.media_write_queues)) *
                resolved_external_backing.media_channels;
            if (stats.finish_ns > result.finish_ns + 1e-6) {
                result.warnings.push_back(
                    "external-backing finish exceeds scenario finish");
                ok = false;
            }
            if (stats.read_requests + stats.write_requests !=
                    result.external_accesses ||
                stats.page_run_segments >
                    stats.read_requests + stats.write_requests ||
                stats.page_run_requests > stats.page_run_segments ||
                stats.page_run_segments > stats.page_run_pages ||
                stats.page_run_pages < stats.page_run_requests ||
                (stats.page_run_segments != 0 &&
                    stats.page_run_requests == 0) ||
                (stats.page_run_pages != 0 &&
                    stats.page_run_requests == 0) ||
                stats.media_channels_per_queue !=
                    resolved_external_backing.media_channels ||
                stats.media_read_queues !=
                    resolved_external_backing.media_read_queues ||
                stats.media_write_queues !=
                    resolved_external_backing.media_write_queues ||
                stats.media_channels != expected_media_resources ||
                stats.active_media_resources > stats.media_channels ||
                stats.max_device_outstanding >
                    resolved_external_backing.max_outstanding_requests ||
                stats.m2s_payload_bytes > stats.m2s_wire_bytes ||
                stats.m2s_protocol_bytes !=
                    stats.m2s_wire_bytes - stats.m2s_payload_bytes ||
                stats.s2m_payload_bytes > stats.s2m_wire_bytes ||
                stats.s2m_protocol_bytes !=
                    stats.s2m_wire_bytes - stats.s2m_payload_bytes ||
                stats.m2s_payload_bytes != stats.write_bytes ||
                stats.s2m_payload_bytes != stats.read_bytes) {
                result.warnings.push_back(
                    "external-backing request/queue accounting is inconsistent");
                ok = false;
            }
            const std::array<double, 4> utilization{
                stats.controller_utilization(),
                stats.media_utilization(),
                stats.m2s_utilization(),
                stats.s2m_utilization(),
            };
            if (std::any_of(
                    utilization.begin(),
                    utilization.end(),
                    [](double value) {
                        return !std::isfinite(value) || value < 0.0 ||
                            value > 1.0 + 1e-9;
                    })) {
                result.warnings.push_back(
                    "external-backing utilization falls outside [0, 1]");
                ok = false;
            }
        }
        // Offered-load advisory: open-loop arrivals at or above 90% of a
        // present tier's peak make every latency percentile a queueing
        // artifact of the arrival spacing, not a device property.
        if (result.offered_load && result.offered_load->open_loop) {
            const auto& load = *result.offered_load;
            const auto saturated = [&](
                const char* tier,
                std::optional<double> ratio,
                double peak_GBps) {
                if (!ratio || *ratio < 0.9) {
                    return;
                }
                std::ostringstream text;
                text << "open-loop offered rate "
                     << fixed(load.rate_GBps.value_or(0.0), 1)
                     << " GB/s is " << fixed(*ratio * 100.0, 0) << "% of the "
                     << tier << " peak " << fixed(peak_GBps, 1)
                     << " GB/s; latency percentiles reflect the arrival model "
                        "(--interarrival-ns or at=), not device service; use "
                        "--max-outstanding-requests or trace timestamps for "
                        "latency claims";
                result.warnings.push_back(text.str());
                notices.push_back("[" + result.name + "] " + text.str());
            };
            saturated("HBM", load.hbm_ratio, load.hbm_peak_GBps.value_or(0.0));
            saturated("HBF", load.hbf_ratio, load.hbf_peak_GBps.value_or(0.0));
            saturated(
                "external-backing directional",
                load.external_ratio,
                std::max(
                    load.external_read_peak_GBps.value_or(0.0),
                    load.external_write_peak_GBps.value_or(0.0)));
        }
        for (const auto& warning : result.warnings) {
            if (warning.find("unmapped") != std::string::npos ||
                warning.find("accounting mismatch") != std::string::npos) {
                ok = false;
            } else if (warning.find("did not exercise both") !=
                       std::string::npos) {
                configuration_warning = true;
                notices.push_back("[" + result.name + "] " + warning);
            }
        }
    }

    if (auto* all_hbm = by_name(kAllHbmScenario)) {
        if (all_hbm->hbm_user_accesses < all_hbm->ops ||
            all_hbm->hbm_accesses != all_hbm->hbm_user_accesses ||
            all_hbm->hbf_accesses != 0) {
            all_hbm->warnings.push_back("all-hbm routed outside HBM");
            ok = false;
        }
        if (trace_footprint_bytes > system.hbm.device.capacity_bytes) {
            all_hbm->warnings.push_back(
                "trace unique-line footprint exceeds HBM capacity; all-hbm is capacity-infeasible");
            ok = false;
        }
        // Policy identity: a FLAT run that routed NOTHING to HBF saw the
        // exact same op stream over the exact same HBM path as all-hbm, so
        // the two results must be bit-equal (a divergence is a routing or
        // driver bug, not a modeling choice). Structural form of the
        // comparison-table observation that FLAT degenerates to all-hbm
        // when the split covers the footprint.
        if (auto* flat = by_name(kFlatScenario)) {
            if (flat->hbf_accesses == 0 && flat->hbf_user_accesses == 0) {
                if (flat->finish_ns != all_hbm->finish_ns ||
                    flat->hbm_accesses != all_hbm->hbm_accesses ||
                    flat->service_latencies_ns !=
                        all_hbm->service_latencies_ns ||
                    flat->offered_latencies_ns !=
                        all_hbm->offered_latencies_ns ||
                    flat->source_latencies_ns !=
                        all_hbm->source_latencies_ns) {
                    flat->warnings.push_back(
                        "policy identity violated: FLAT with no HBF traffic "
                        "must equal all-hbm bit for bit");
                    ok = false;
                }
            }
        }
    }
    if (auto* all_hbf = by_name(kAllHbfScenario)) {
        if (all_hbf->hbf_user_accesses < all_hbf->ops ||
            all_hbf->hbm_accesses != 0) {
            all_hbf->warnings.push_back("all-hbf routed outside HBF");
            ok = false;
        }
        if (all_hbf->writes > 0 && !resolved_hbf.host.write_coalescing_enabled) {
            const auto waf = all_hbf->hbf_stats.waf();
            if (waf && *waf < 1.0) {
                all_hbf->warnings.push_back(
                    "HBF WAF below 1.0 is physically suspicious");
                ok = false;
            }
        }
    }
    const auto check_behavioral = [&](
        const char* scenario_name,
        BehavioralAdmissionPolicy expected_policy) {
        auto* behavioral = by_name(scenario_name);
        if (behavioral == nullptr) {
            return;
        }
        if (!behavioral->behavioral_tiering_stats) {
            behavioral->warnings.push_back(
                "behavioral tiering result is missing controller statistics");
            ok = false;
            return;
        }
        const auto& s = *behavioral->behavioral_tiering_stats;
        const auto classified = checked_add_u64(
            checked_add_u64(
                s.hbm_hits,
                s.hbf_bypasses,
                "behavioral classified observations"),
            s.promotions,
            "behavioral classified observations");
        const auto promotion_classes = checked_add_u64(
            s.promotions_with_backing_fill,
            s.promotions_without_backing_fill,
            "behavioral promotion classes");
        const auto expected_fill_bytes = checked_mul_u64(
            s.backing_fill_pages,
            resolved_hbf.device.page_size_bytes,
            "behavioral fill bytes");
        const auto expected_install_bytes = checked_mul_u64(
            s.hbm_install_pages,
            resolved_hbf.device.page_size_bytes,
            "behavioral install bytes");
        const auto expected_writeback_bytes = checked_mul_u64(
            s.dirty_writeback_pages,
            resolved_hbf.device.page_size_bytes,
            "behavioral writeback bytes");
        if (s.admission_policy != expected_policy ||
            s.semantic_inputs_consumed ||
            s.page_observations != classified ||
            s.promotions != promotion_classes ||
            s.promotions_with_backing_fill != s.backing_fill_pages ||
            s.backing_fill_pages != s.hbm_install_pages ||
            s.backing_fill_bytes != expected_fill_bytes ||
            s.hbm_install_bytes != expected_install_bytes ||
            s.dirty_writeback_bytes != expected_writeback_bytes ||
            s.peak_resident_pages > s.hbm_tier_pages ||
            s.final_resident_pages > s.hbm_tier_pages ||
            s.final_dirty_pages != 0 ||
            behavioral->hbf_user_accesses != s.hbf_bypasses ||
            behavioral->hbm_user_accesses < s.hbm_hits ||
            behavioral->base_die_link_stats.read_bytes !=
                s.backing_fill_bytes ||
            behavioral->base_die_link_stats.write_bytes !=
                s.dirty_writeback_bytes ||
            behavioral->background_hbf_writes !=
                s.dirty_writeback_pages ||
            behavioral->hbf_backing_write_bytes !=
                s.dirty_writeback_bytes ||
            behavioral->hbm_foreground_bytes !=
                s.hbm_foreground_bytes) {
            behavioral->warnings.push_back(
                "behavioral placement conservation or policy contract failed");
            ok = false;
        }
    };
    check_behavioral(
        kDemandFillScenario,
        BehavioralAdmissionPolicy::AlwaysAdmit);
    check_behavioral(
        kBehavioralTieringScenario,
        BehavioralAdmissionPolicy::ReuseFiltered);

    const auto check_streaming = [&](
        const char* scenario_name,
        BackingTier expected_backing,
        std::uint64_t page_size) {
        auto* streaming = by_name(scenario_name);
        if (streaming == nullptr) {
            return;
        }
        if (!streaming->layer_streaming_stats) {
            streaming->warnings.push_back(
                "layer-streaming result is missing controller statistics");
            ok = false;
        } else {
            const auto& s = *streaming->layer_streaming_stats;
            const auto foreground_page_transactions = checked_add_u64(
                s.foreground_resident_page_accesses,
                s.foreground_buffer_page_accesses,
                "layer-streaming foreground page transactions");
            if (streaming->hbm_user_accesses !=
                    foreground_page_transactions ||
                streaming->hbm_user_accesses < streaming->ops ||
                streaming->hbf_user_accesses != 0 ||
                s.backing != expected_backing) {
                streaming->warnings.push_back(
                    "layer streaming must execute every foreground page "
                    "transaction from HBM with the selected backing tier");
                ok = false;
            }
            const std::array<double, 8> times{
                s.compute_work_ns,
                s.backing_admission_wait_work_ns,
                s.backing_admission_max_wait_ns,
                s.user_wait_work_ns,
                s.user_max_wait_ns,
                s.exposed_prefetch_ns,
                s.hidden_prefetch_ns,
                s.buffer_reuse_wait_work_ns,
            };
            if (std::any_of(times.begin(), times.end(), [](double value) {
                    return !std::isfinite(value) || value < 0.0;
                })) {
                streaming->warnings.push_back(
                    "layer-streaming controller reported invalid time work");
                ok = false;
            }
            if (s.streamed_bytes != checked_mul_u64(
                    s.streamed_pages,
                    page_size,
                    "layer-streaming DMA bytes") ||
                s.writeback_bytes != checked_mul_u64(
                    s.dirty_pages_written_back,
                    page_size,
                    "layer-streaming writeback bytes") ||
                s.hbm_only_resident_bytes != checked_mul_u64(
                    s.hbm_only_resident_pages,
                    page_size,
                    "layer-streaming HBM-only resident bytes") ||
                s.hot_kv_candidate_bytes != checked_mul_u64(
                    s.hot_kv_candidate_pages,
                    page_size,
                    "layer-streaming hot-KV candidate bytes") ||
                s.hot_kv_resident_bytes != checked_mul_u64(
                    s.hot_kv_resident_pages,
                    page_size,
                    "layer-streaming hot-KV resident bytes") ||
                s.data_bytes != checked_mul_u64(
                    s.data_pages,
                    page_size,
                    "layer-streaming data bytes") ||
                s.model_weight_resident_bytes != checked_mul_u64(
                    s.model_weight_resident_pages,
                    page_size,
                    "layer-streaming model-weight resident bytes") ||
                s.model_weight_backing_bytes != checked_mul_u64(
                    s.model_weight_backing_pages,
                    page_size,
                    "layer-streaming model-weight backing bytes") ||
                s.cold_kv_backing_bytes != checked_mul_u64(
                    s.cold_kv_backing_pages,
                    page_size,
                    "layer-streaming cold-KV backing bytes") ||
                s.unknown_backing_bytes != checked_mul_u64(
                    s.unknown_backing_pages,
                    page_size,
                    "layer-streaming unknown backing bytes") ||
                s.backing_unique_bytes != checked_mul_u64(
                    s.backing_unique_pages,
                    page_size,
                    "layer-streaming unique backing bytes") ||
                s.unique_resident_footprint_bytes != checked_mul_u64(
                    s.unique_resident_footprint_pages,
                    page_size,
                    "layer-streaming unique footprint bytes") ||
                s.capacity_pressure_basis_bytes >
                    s.unique_resident_footprint_bytes ||
                s.footprint_page_rounding_bytes !=
                    s.unique_resident_footprint_bytes -
                        s.capacity_pressure_basis_bytes ||
                s.resident_physical_bytes != checked_mul_u64(
                    s.resident_physical_pages,
                    page_size,
                    "layer-streaming resident physical bytes") ||
                s.effective_layer_buffer_bytes != checked_mul_u64(
                    s.effective_layer_buffer_pages,
                    page_size,
                    "layer-streaming effective buffer bytes") ||
                s.max_layer_data_bytes != checked_mul_u64(
                    s.max_layer_data_pages,
                    page_size,
                    "layer-streaming maximum layer data bytes") ||
                s.unused_hbm_bytes != checked_mul_u64(
                    s.unused_hbm_pages,
                    page_size,
                    "layer-streaming unused HBM bytes") ||
                s.hot_kv_resident_pages > s.hot_kv_candidate_pages ||
                s.backing_unique_pages != checked_add_u64(
                    s.model_weight_backing_pages,
                    checked_add_u64(
                        s.cold_kv_backing_pages,
                        s.unknown_backing_pages,
                        "layer-streaming cold-KV and unknown pages"),
                    "layer-streaming backing partition") ||
                s.data_pages != checked_add_u64(
                    checked_add_u64(
                        s.model_weight_resident_pages,
                        s.hot_kv_resident_pages,
                        "layer-streaming resident data pages"),
                    s.backing_unique_pages,
                    "layer-streaming data partition") ||
                s.unique_resident_footprint_pages != checked_add_u64(
                    s.hbm_only_resident_pages,
                    s.data_pages,
                    "layer-streaming unique footprint") ||
                s.resident_physical_pages != checked_add_u64(
                    s.hbm_only_resident_pages,
                    checked_add_u64(
                        s.model_weight_resident_pages,
                        s.hot_kv_resident_pages,
                        "layer-streaming resident model weights and hot KV"),
                    "layer-streaming HBM residency") ||
                (!s.explicit_residency_contract &&
                 s.effective_layer_buffer_pages !=
                    s.max_layer_data_pages) ||
                (s.explicit_residency_contract &&
                 s.effective_layer_buffer_pages <
                    s.max_layer_data_pages) ||
                s.max_layer_data_bytes > options.layer_buffer_bytes) {
                streaming->warnings.push_back(
                    "layer-streaming page/byte/capacity accounting is inconsistent");
                ok = false;
            }
            const auto backing_transactions = checked_add_u64(
                s.streamed_pages,
                s.dirty_pages_written_back,
                "layer-streaming backing transactions");
            if (s.backing_request_credit_limit !=
                    options.max_outstanding_requests ||
                s.backing_max_inflight_requests > backing_transactions ||
                (s.backing_request_credit_limit != 0 &&
                 s.backing_max_inflight_requests >
                    s.backing_request_credit_limit) ||
                s.backing_admission_waited_requests >
                    backing_transactions ||
                ((s.backing_admission_waited_requests == 0) !=
                 (s.backing_admission_wait_work_ns == 0.0)) ||
                ((s.backing_admission_waited_requests == 0) !=
                 (s.backing_admission_max_wait_ns == 0.0)) ||
                s.backing_admission_max_wait_ns >
                    s.backing_admission_wait_work_ns ||
                (s.backing_request_credit_limit == 0 &&
                 s.backing_admission_waited_requests != 0)) {
                streaming->warnings.push_back(
                    "layer-streaming backing credit accounting is "
                    "inconsistent");
                ok = false;
            }
            const auto runtime_hbm_pages = checked_add_u64(
                s.resident_physical_pages,
                checked_mul_u64(
                    s.effective_layer_buffer_pages,
                    2,
                    "two effective layer buffers"),
                "runtime layer-streaming HBM pages");
            const auto expected_capacity_pressure =
                static_cast<double>(
                    static_cast<long double>(
                        s.capacity_pressure_basis_bytes) /
                    static_cast<long double>(s.hbm_capacity_bytes));
            const auto pressure_tolerance =
                std::numeric_limits<double>::epsilon() *
                std::max(1.0, expected_capacity_pressure) * 8.0;
            if (s.hbm_capacity_bytes != system.hbm.device.capacity_bytes ||
                checked_mul_u64(
                    runtime_hbm_pages,
                    page_size,
                    "runtime layer-streaming HBM bytes") >
                    s.hbm_capacity_bytes ||
                s.compact_resident_mapping !=
                    (s.resident_physical_pages != 0) ||
                checked_add_u64(
                    runtime_hbm_pages,
                    s.unused_hbm_pages,
                    "allocated and unused HBM pages") !=
                    (s.hbm_capacity_bytes - streaming->hbf_stats.host_hbm_reserved_bytes) / page_size ||
                !std::isfinite(s.hbm_capacity_pressure) ||
                s.hbm_capacity_pressure < 0.0 ||
                std::abs(
                    s.hbm_capacity_pressure -
                    expected_capacity_pressure) > pressure_tolerance) {
                streaming->warnings.push_back(
                    "layer-streaming hybrid-residency policy is inconsistent "
                    "with its unique footprint and HBM capacity");
                ok = false;
            }
            if (s.explicit_residency_contract !=
                    options.explicit_residency_contract) {
                streaming->warnings.push_back(
                    "layer-streaming residency contract selection drifted "
                    "from the resolved configuration");
                ok = false;
            }
            if (s.explicit_residency_contract &&
                options.explicit_residency_contract) {
                const auto logical_kv_bytes = checked_mul_u64(
                    s.logical_kv_blocks,
                    s.kv_block_stride_bytes,
                    "explicit logical KV bytes");
                const auto exact_population = checked_add_u64(
                    s.immutable_weight_logical_bytes,
                    checked_add_u64(
                        s.runtime_overhead_logical_bytes,
                        checked_add_u64(
                            s.block_table_logical_bytes,
                            logical_kv_bytes,
                            "explicit block table and logical KV"),
                        "explicit overhead, block table, and logical KV"),
                    "explicit unique resident population");
                if (s.capacity_pressure_basis_bytes != exact_population ||
                    s.logical_kv_blocks != checked_add_u64(
                        s.hot_kv_blocks,
                        s.cold_kv_blocks,
                        "explicit hot and cold KV blocks") ||
                    s.immutable_weight_logical_bytes !=
                        *options.residency_immutable_weight_bytes ||
                    s.runtime_overhead_logical_bytes !=
                        *options.residency_runtime_overhead_bytes ||
                    s.block_table_logical_bytes !=
                        *options.residency_block_table_bytes ||
                    s.active_buffer_logical_bytes_per_slot !=
                        *options.residency_active_buffer_bytes ||
                    s.residency_page_size_bytes !=
                        *options.residency_page_size_bytes ||
                    s.kv_block_stride_bytes !=
                        *options.residency_kv_block_stride_bytes ||
                    s.logical_kv_blocks !=
                        *options.residency_logical_kv_blocks ||
                    s.hot_kv_blocks !=
                        *options.residency_hot_kv_blocks ||
                    s.model_weight_resident_pages !=
                        *options.residency_static_weight_pages) {
                    streaming->warnings.push_back(
                        "explicit layer-streaming residency contract is not "
                        "byte-conservative");
                    ok = false;
                }
            } else if (!s.explicit_residency_contract && (
                s.capacity_pressure_basis_bytes !=
                    s.unique_resident_footprint_bytes ||
                s.footprint_page_rounding_bytes != 0 ||
                s.immutable_weight_logical_bytes != 0 ||
                s.runtime_overhead_logical_bytes != 0 ||
                s.block_table_logical_bytes != 0 ||
                s.active_buffer_logical_bytes_per_slot != 0 ||
                s.residency_page_size_bytes != 0 ||
                s.kv_block_stride_bytes != 0 ||
                s.logical_kv_blocks != 0 ||
                s.hot_kv_blocks != 0 ||
                s.cold_kv_blocks != 0)) {
                streaming->warnings.push_back(
                    "trace-derived layer streaming leaked an explicit "
                    "residency contract");
                ok = false;
            }
            if (s.resident_physical_bytes +
                    checked_mul_u64(
                        s.effective_layer_buffer_bytes,
                        2,
                        "two runtime layer buffers") >
                s.hbm_capacity_bytes) {
                streaming->warnings.push_back(
                    "layer-streaming runtime allocation exceeds HBM");
                ok = false;
            }
            if (expected_backing == BackingTier::Hbf) {
                if (s.streamed_bytes != checked_add_u64(
                        streaming->hbf_static_read_bytes,
                        streaming->hbf_logical_read_bytes,
                        "HBF backing read bytes") ||
                    s.writeback_bytes !=
                        streaming->hbf_backing_write_bytes ||
                    streaming->external_accesses != 0) {
                    streaming->warnings.push_back(
                        "HBF layer-streaming backing traffic does not conserve");
                    ok = false;
                }
            } else if (
                s.streamed_bytes != streaming->external_backing_read_bytes ||
                s.writeback_bytes !=
                    streaming->external_backing_write_bytes ||
                streaming->hbf_accesses != 0 ||
                streaming->base_die_link_stats.read_transfers != 0 ||
                streaming->base_die_link_stats.write_transfers != 0) {
                streaming->warnings.push_back(
                    "external layer-streaming backing traffic does not conserve");
                ok = false;
            }
        }
    };
    check_streaming(
        kLayerStreamingScenario,
        BackingTier::Hbf,
        resolved_hbf.device.page_size_bytes);
    check_streaming(
        kExternalLayerStreamingScenario,
        BackingTier::External,
        resolved_external_backing.page_size_bytes);

    if (resolved_hbf.device.page_size_bytes == resolved_external_backing.page_size_bytes) {
        const auto page_size = resolved_hbf.device.page_size_bytes;
        auto* hbf_streaming = by_name(kLayerStreamingScenario);
        auto* external_streaming =
            by_name(kExternalLayerStreamingScenario);
        if (hbf_streaming != nullptr && external_streaming != nullptr &&
            hbf_streaming->layer_streaming_stats &&
            external_streaming->layer_streaming_stats) {
            const auto& hbf_plan =
                *hbf_streaming->layer_streaming_stats;
            const auto& external_plan =
                *external_streaming->layer_streaming_stats;
            const bool same_plan =
                hbf_plan.compact_resident_mapping ==
                    external_plan.compact_resident_mapping &&
                hbf_plan.semantic_inputs_consumed ==
                    external_plan.semantic_inputs_consumed &&
                hbf_plan.explicit_residency_contract ==
                    external_plan.explicit_residency_contract &&
                hbf_plan.address_footprint_bytes ==
                    external_plan.address_footprint_bytes &&
                hbf_plan.unique_resident_footprint_pages ==
                    external_plan.unique_resident_footprint_pages &&
                hbf_plan.unique_resident_footprint_bytes ==
                    external_plan.unique_resident_footprint_bytes &&
                hbf_plan.capacity_pressure_basis_bytes ==
                    external_plan.capacity_pressure_basis_bytes &&
                hbf_plan.footprint_page_rounding_bytes ==
                    external_plan.footprint_page_rounding_bytes &&
                hbf_plan.hbm_capacity_pressure ==
                    external_plan.hbm_capacity_pressure &&
                hbf_plan.layers == external_plan.layers &&
                hbf_plan.explicit_layer_requests ==
                    external_plan.explicit_layer_requests &&
                hbf_plan.explicit_compute_layers ==
                    external_plan.explicit_compute_layers &&
                hbf_plan.compute_work_ns ==
                    external_plan.compute_work_ns &&
                hbf_plan.hbm_only_resident_pages ==
                    external_plan.hbm_only_resident_pages &&
                hbf_plan.hot_kv_candidate_pages ==
                    external_plan.hot_kv_candidate_pages &&
                hbf_plan.hot_kv_resident_pages ==
                    external_plan.hot_kv_resident_pages &&
                hbf_plan.data_pages == external_plan.data_pages &&
                hbf_plan.model_weight_resident_pages ==
                    external_plan.model_weight_resident_pages &&
                hbf_plan.model_weight_backing_pages ==
                    external_plan.model_weight_backing_pages &&
                hbf_plan.cold_kv_backing_pages ==
                    external_plan.cold_kv_backing_pages &&
                hbf_plan.unknown_backing_pages ==
                    external_plan.unknown_backing_pages &&
                hbf_plan.backing_unique_pages ==
                    external_plan.backing_unique_pages &&
                hbf_plan.resident_physical_pages ==
                    external_plan.resident_physical_pages &&
                hbf_plan.effective_layer_buffer_pages ==
                    external_plan.effective_layer_buffer_pages &&
                hbf_plan.unused_hbm_pages +
                    (hbf_plan.hbm_capacity_bytes / page_size -
                     (hbf_plan.hbm_capacity_bytes - hbf_streaming->hbf_stats.host_hbm_reserved_bytes) / page_size) ==
                    external_plan.unused_hbm_pages &&
                hbf_plan.streamed_pages ==
                    external_plan.streamed_pages &&
                hbf_plan.foreground_resident_page_accesses ==
                    external_plan.foreground_resident_page_accesses &&
                hbf_plan.foreground_buffer_page_accesses ==
                    external_plan.foreground_buffer_page_accesses &&
                hbf_plan.dirty_pages_written_back ==
                    external_plan.dirty_pages_written_back &&
                hbf_plan.max_layer_data_pages ==
                    external_plan.max_layer_data_pages &&
                hbf_plan.immutable_weight_logical_bytes ==
                    external_plan.immutable_weight_logical_bytes &&
                hbf_plan.runtime_overhead_logical_bytes ==
                    external_plan.runtime_overhead_logical_bytes &&
                hbf_plan.block_table_logical_bytes ==
                    external_plan.block_table_logical_bytes &&
                hbf_plan.active_buffer_logical_bytes_per_slot ==
                    external_plan.active_buffer_logical_bytes_per_slot &&
                hbf_plan.residency_page_size_bytes ==
                    external_plan.residency_page_size_bytes &&
                hbf_plan.kv_block_stride_bytes ==
                    external_plan.kv_block_stride_bytes &&
                hbf_plan.logical_kv_blocks ==
                    external_plan.logical_kv_blocks &&
                hbf_plan.hot_kv_blocks ==
                    external_plan.hot_kv_blocks &&
                hbf_plan.cold_kv_blocks ==
                    external_plan.cold_kv_blocks &&
                hbf_streaming->hbm_user_accesses ==
                    external_streaming->hbm_user_accesses &&
                hbf_streaming->hbm_background_accesses ==
                    external_streaming->hbm_background_accesses &&
                hbf_streaming->hbm_foreground_bytes ==
                    external_streaming->hbm_foreground_bytes &&
                hbf_streaming->hbm_streaming_write_bytes ==
                    external_streaming->hbm_streaming_write_bytes;
            if (!same_plan) {
                external_streaming->warnings.push_back(
                    "HBF and external layer-streaming controls diverged "
                    "before the backing-device boundary");
                ok = false;
            }
        }
    }

    return SanityReport{
        .verdict = !ok ? SanityVerdict::Fail :
            configuration_warning ? SanityVerdict::PassWithWarnings :
            SanityVerdict::Pass,
        .notices = std::move(notices),
    };
}

std::string json_escape(const std::string& input) {
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
                out << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(static_cast<unsigned char>(ch))
                    << std::dec;
            } else {
                out << ch;
            }
            break;
        }
    }
    return out.str();
}

void write_stage_work_json(
    std::ostream& out,
    const Breakdown& work,
    const std::string& indent,
    const hbfsim::host::HbfStats* hbf_stats = nullptr) {
    out << "{\n"
        << indent << "  \"ingress_queue_wait_work_ns\": "
        << work.ingress_queue_wait_ns << ",\n"
        << indent << "  \"scheduler_queue_wait_work_ns\": "
        << work.scheduler_queue_wait_ns << ",\n"
        << indent << "  \"address_mapping_work_ns\": "
        << work.address_mapping_ns << ",\n"
        << indent << "  \"translation_work_ns\": " << work.translation_ns << ",\n"
        << indent << "  \"mapping_dram_work_ns\": "
        << work.mapping_dram_ns << ",\n"
        << indent << "  \"write_buffer_dram_work_ns\": "
        << work.write_buffer_dram_ns << ",\n"
        << indent << "  \"refresh_stall_work_ns\": "
        << work.refresh_stall_ns << ",\n"
        << indent << "  \"precharge_work_ns\": " << work.precharge_ns << ",\n"
        << indent << "  \"activation_work_ns\": " << work.activation_ns << ",\n"
        << indent << "  \"command_work_ns\": " << work.command_ns << ",\n"
        << indent << "  \"array_read_work_ns\": " << work.array_read_ns << ",\n"
        << indent << "  \"array_program_work_ns\": "
        << work.array_program_ns << ",\n"
        << indent << "  \"array_erase_work_ns\": " << work.array_erase_ns << ",\n"
        << indent << "  \"media_lane_transfer_work_ns\": "
        << work.media_lane_transfer_ns << ",\n"
        << indent << "  \"page_buffer_work_ns\": " << work.page_buffer_ns << ",\n"
        << indent << "  \"sram_staging_work_ns\": "
        << work.sram_staging_ns << ",\n"
        << indent << "  \"channel_transfer_work_ns\": "
        << work.channel_transfer_ns << ",\n"
        << indent << "  \"tsv_transfer_work_ns\": "
        << work.tsv_transfer_ns << ",\n"
        << indent << "  \"hb_io_transfer_work_ns\": "
        << work.hb_io_transfer_ns << ",\n"
        << indent << "  \"transport_latency_work_ns\": "
        << work.transport_latency_ns << ",\n"
        << indent << "  \"ecc_queue_wait_work_ns\": "
        << work.ecc_queue_wait_ns << ",\n"
        << indent << "  \"ecc_response_latency_work_ns\": "
        << work.ecc_latency_ns << ",\n"
        << indent << "  \"maintenance_work_ns\": " << work.maintenance_ns << ",\n"
        << indent << "  \"total_overlapping_work_ns\": "
        << work.total_work_ns();
    if (hbf_stats != nullptr) {
        const auto& stats = *hbf_stats;
        out << ",\n"
            << indent << "  \"write_buffer\": {"
            << "\"capacity_bytes\": " << stats.write_buffer_capacity_bytes
            << ", \"capacity_bytes_per_stack\": "
            << stats.write_buffer_capacity_bytes_per_stack
            << ", \"dram_read_ops\": "
            << stats.write_buffer_dram_read_ops
            << ", \"dram_write_ops\": "
            << stats.write_buffer_dram_write_ops
            << ", \"dram_read_bytes\": "
            << stats.write_buffer_dram_read_bytes
            << ", \"dram_write_bytes\": "
            << stats.write_buffer_dram_write_bytes
            << ", \"dram_waited_ops\": "
            << stats.write_buffer_dram_wait_ops
            << ", \"dram_wait_work_ns\": "
            << stats.write_buffer_dram_wait_ns
            << ", \"dram_issue_busy_ns\": "
            << stats.write_buffer_dram_issue_busy_ns
            << ", "
            << "\"slot_waited_ops\": " << stats.write_buffer_slot_wait_ops
            << ", \"slot_wait_work_ns\": "
            << stats.write_buffer_slot_wait_ns << "},\n"
            << indent << "  \"resident_mapping\": {"
            << "\"table_bytes\": "
            << stats.resident_mapping_table_bytes
            << ", \"table_bytes_per_stack\": "
            << stats.resident_mapping_table_bytes_per_stack
            << ", \"pages_per_stack\": "
            << stats.resident_mapping_pages_per_stack
            << ", \"lookup_ops\": " << stats.mapping_lookup_ops
            << ", \"user_lookup_ops\": " << stats.mapping_user_lookup_ops
            << ", \"gc_lookup_ops\": " << stats.mapping_gc_lookup_ops
            << ", \"update_ops\": " << stats.mapping_update_ops
            << ", \"user_update_ops\": " << stats.mapping_user_update_ops
            << ", \"gc_update_ops\": " << stats.mapping_gc_update_ops
            << ", \"dram_waited_ops\": " << stats.mapping_dram_wait_ops
            << ", \"dram_wait_work_ns\": " << stats.mapping_dram_wait_ns
            << ", \"dram_wait_max_ns\": " << stats.mapping_dram_wait_max_ns
            << ", \"dram_issue_busy_ns\": "
            << stats.mapping_dram_issue_busy_ns
            << ", \"dram_resources\": " << stats.mapping_dram_resources
            << "},\n"
            << indent << "  \"mapping_cache\": {"
            << "\"logical_table_bytes\": " << stats.mapping_table_bytes
            << ", \"directory_entry_bytes\": "
            << stats.mapping_directory_entry_bytes
            << ", \"directory_bytes\": "
            << stats.mapping_directory_bytes
            << ", \"directory_bytes_per_stack\": "
            << stats.mapping_directory_bytes_per_stack
            << ", \"capacity_bytes\": "
            << stats.mapping_cache_capacity_bytes
            << ", \"capacity_bytes_per_stack\": "
            << stats.mapping_cache_capacity_bytes_per_stack
            << ", \"pages_per_stack\": "
            << stats.mapping_cache_pages_per_stack
            << ", \"entries\": " << stats.mapping_cache_entries
            << ", \"peak_entries\": "
            << stats.mapping_cache_peak_entries
            << ", \"hits\": " << stats.mapping_cache_hits
            << ", \"misses\": " << stats.mapping_cache_misses
            << ", \"erased_misses\": "
            << stats.mapping_cache_erased_misses
            << ", \"coalesced_misses\": "
            << stats.mapping_cache_coalesced_misses
            << ", \"evictions\": " << stats.mapping_cache_evictions
            << ", \"dirty_evictions\": "
            << stats.mapping_cache_dirty_evictions
            << ", \"media_reads\": " << stats.mapping_media_reads
            << ", \"media_read_bytes\": "
            << stats.mapping_media_read_bytes << "},\n"
            << indent << "  \"ecc_directional\": {\n"
            << indent << "    \"decode\": {"
            << "\"queue_wait_work_ns\": " << stats.ecc_decode_queue_wait_ns
            << ", \"response_latency_work_ns\": "
            << stats.ecc_decode_latency_work_ns
            << ", \"issue_busy_ns\": " << stats.ecc_decode_issue_busy_ns
            << ", \"ops\": " << stats.ecc_decode_ops
            << ", \"codeword_bytes\": " << stats.ecc_decode_codeword_bytes
            << "},\n"
            << indent << "    \"encode\": {"
            << "\"queue_wait_work_ns\": " << stats.ecc_encode_queue_wait_ns
            << ", \"response_latency_work_ns\": "
            << stats.ecc_encode_latency_work_ns
            << ", \"issue_busy_ns\": " << stats.ecc_encode_issue_busy_ns
            << ", \"ops\": " << stats.ecc_encode_ops
            << ", \"codeword_bytes\": " << stats.ecc_encode_codeword_bytes
            << "}\n"
            << indent << "  }";
    }
    out << '\n' << indent << '}';
}

void write_resource_busy_json(
    std::ostream& out,
    const ResourceBusyMetric& metric) {
    out << "{\"busy_ns\": " << metric.busy_ns
        << ", \"resource_count\": " << metric.resource_count
        << ", \"active_span_ns\": " << metric.active_span_ns
        << ", \"capacity_time_ns\": " << metric.capacity_time_ns
        << ", \"utilization\": " << metric.utilization << '}';
}

void write_time_breakdown_json(
    std::ostream& out,
    const ScenarioResult& result,
    const std::string& indent) {
    const auto time = make_time_breakdown(result);
    const auto& wall = time.wall_clock;
    const auto& latency = time.latency_work;
    out << indent << "\"time_breakdown\": {\n"
        << indent << "  \"contract_version\": 2,\n"
        << indent << "  \"semantics\": {\n"
        << indent << "    \"wall_clock\": \"additive_non_overlapping_spans\",\n"
        << indent << "    \"latency_work\": \"sum_across_user_operations_may_overlap\",\n"
        << indent << "    \"stage_work\": \"hierarchical_device_work_may_overlap_do_not_sum_parent_and_children\",\n"
        << indent << "    \"controller_work\": \"parent_totals_and_attribution_views_may_overlap\",\n"
        << indent << "    \"resource_busy\": \"exclusive_per_resource_then_aggregated\"\n"
        << indent << "  },\n"
        << indent << "  \"wall_clock_ns\": {\n"
        << indent << "    \"trace_origin_ns\": " << wall.trace_origin_ns << ",\n"
        << indent << "    \"last_offered_arrival_ns\": "
        << wall.last_offered_arrival_ns << ",\n"
        << indent << "    \"last_user_completion_ns\": "
        << wall.last_user_completion_ns << ",\n"
        << indent << "    \"quiescent_finish_ns\": "
        << wall.quiescent_finish_ns << ",\n"
        << indent << "    \"offered_arrival_span_ns\": "
        << wall.offered_arrival_span_ns << ",\n"
        << indent << "    \"post_offer_user_completion_tail_ns\": "
        << wall.post_offer_user_completion_tail_ns << ",\n"
        << indent << "    \"user_completion_span_ns\": "
        << wall.user_completion_span_ns << ",\n"
        << indent << "    \"drain_tail_ns\": " << wall.drain_tail_ns << ",\n"
        << indent << "    \"makespan_ns\": " << wall.makespan_ns << "\n"
        << indent << "  },\n"
        << indent << "  \"latency_work\": {\n"
        << indent << "    \"basis\": \"offered_to_user_completion\",\n"
        << indent << "    \"user_count\": " << latency.user_count << ",\n"
        << indent << "    \"service_to_user_completion_sum_work_ns\": "
        << latency.service_to_user_completion_sum_work_ns << ",\n"
        << indent << "    \"phase_barriers\": "
        << latency.phase_barriers << ",\n"
        << indent << "    \"phase_dependency_waited_ops\": "
        << latency.phase_dependency_waited_ops << ",\n"
        << indent << "    \"phase_dependency_wait_work_ns\": "
        << latency.phase_dependency_wait_work_ns << ",\n"
        << indent << "    \"phase_dependency_max_wait_ns\": "
        << latency.phase_dependency_max_wait_ns << ",\n"
        << indent << "    \"front_end_admission_waited_ops\": "
        << latency.front_end_admission_waited_ops << ",\n"
        << indent << "    \"front_end_admission_wait_work_ns\": "
        << latency.front_end_admission_wait_work_ns << ",\n"
        << indent << "    \"front_end_admission_max_wait_ns\": "
        << latency.front_end_admission_max_wait_ns << ",\n"
        << indent << "    \"offered_to_user_completion_sum_work_ns\": "
        << latency.offered_to_user_completion_sum_work_ns << ",\n"
        << indent << "    \"source_to_user_completion_sum_work_ns\": "
        << latency.source_to_user_completion_sum_work_ns << ",\n"
        << indent << "    \"average_ns\": " << latency.average_ns << ",\n"
        << indent << "    \"p50_ns\": " << latency.p50_ns << ",\n"
        << indent << "    \"p95_ns\": " << latency.p95_ns << ",\n"
        << indent << "    \"max_ns\": " << latency.max_ns << ",\n"
        << indent << "    \"service_average_ns\": "
        << latency.service_average_ns << ",\n"
        << indent << "    \"service_p50_ns\": "
        << latency.service_p50_ns << ",\n"
        << indent << "    \"service_p95_ns\": "
        << latency.service_p95_ns << ",\n"
        << indent << "    \"service_max_ns\": "
        << latency.service_max_ns << ",\n"
        << indent << "    \"source_average_ns\": "
        << latency.source_average_ns << ",\n"
        << indent << "    \"source_p50_ns\": "
        << latency.source_p50_ns << ",\n"
        << indent << "    \"source_p95_ns\": "
        << latency.source_p95_ns << ",\n"
        << indent << "    \"source_max_ns\": "
        << latency.source_max_ns << "\n"
        << indent << "  },\n"
        << indent << "  \"stage_work\": {\n"
        << indent << "    \"scope\": \"all_device_work_including_background_and_drain\",\n"
        << indent << "    \"hbm\": ";
    if (result.has_hbm) {
        write_stage_work_json(out, time.hbm_stage_work, indent + "    ");
    } else {
        out << "null";
    }
    out << ",\n" << indent << "    \"hbf\": ";
    if (result.has_hbf) {
        write_stage_work_json(
            out,
            time.hbf_stage_work,
            indent + "    ",
            &result.hbf_stats);
    } else {
        out << "null";
    }
    out << ",\n" << indent << "    \"external_backing\": ";
    if (result.has_external_backing) {
        write_stage_work_json(
            out,
            time.external_backing_stage_work,
            indent + "    ");
    } else {
        out << "null";
    }
    const auto& link = time.base_die_link_stage_work;
    const auto& streaming_controller =
        time.layer_streaming_controller_stage_work;
    const auto& cooperative_controller =
        time.cooperative_write_controller_stage_work;
    out << ",\n"
        << indent << "    \"base_die_link\": {\n"
        << indent << "      \"read_queue_wait_work_ns\": "
        << link.read_queue_wait_work_ns << ",\n"
        << indent << "      \"write_queue_wait_work_ns\": "
        << link.write_queue_wait_work_ns << ",\n"
        << indent << "      \"read_serialization_work_ns\": "
        << link.read_serialization_work_ns << ",\n"
        << indent << "      \"write_serialization_work_ns\": "
        << link.write_serialization_work_ns << ",\n"
        << indent << "      \"read_fixed_latency_work_ns\": "
        << link.read_fixed_latency_work_ns << ",\n"
        << indent << "      \"write_fixed_latency_work_ns\": "
        << link.write_fixed_latency_work_ns << "\n"
        << indent << "    },\n"
        << indent << "    \"layer_streaming_controller\": {\n"
        << indent << "      \"present\": "
        << (streaming_controller.present ? "true" : "false") << ",\n"
        << indent << "      \"backing_admission_wait_work_ns\": "
        << streaming_controller.backing_admission_wait_work_ns << ",\n"
        << indent << "      \"backing_admission_max_wait_ns\": "
        << streaming_controller.backing_admission_max_wait_ns << ",\n"
        << indent << "      \"user_wait_work_ns\": "
        << streaming_controller.user_wait_work_ns << ",\n"
        << indent << "      \"user_max_wait_ns\": "
        << streaming_controller.user_max_wait_ns << ",\n"
        << indent << "      \"exposed_prefetch_ns\": "
        << streaming_controller.exposed_prefetch_ns << ",\n"
        << indent << "      \"hidden_prefetch_ns\": "
        << streaming_controller.hidden_prefetch_ns << ",\n"
        << indent << "      \"buffer_reuse_wait_work_ns\": "
        << streaming_controller.buffer_reuse_wait_work_ns << "\n"
        << indent << "    },\n"
        << indent << "    \"cooperative_write_controller\": {\n"
        << indent << "      \"present\": "
        << (cooperative_controller.present ? "true" : "false") << ",\n"
        << indent << "      \"full_waited_ops\": "
        << cooperative_controller.full_waited_ops << ",\n"
        << indent << "      \"full_wait_work_ns\": "
        << cooperative_controller.full_wait_work_ns << "\n"
        << indent << "    }\n"
        << indent << "  },\n"
        << indent << "  \"resource_busy\": {\n";
    const std::array<std::pair<const char*, const ResourceBusyMetric*>, 22> resources{{
        {"hbm_data_bus", &time.resource_busy.hbm_data_bus},
        {"external_controller", &time.resource_busy.external_controller},
        {"external_media", &time.resource_busy.external_media},
        {"external_link_m2s", &time.resource_busy.external_link_m2s},
        {"external_link_s2m", &time.resource_busy.external_link_s2m},
        {"hbf_logic_ingress", &time.resource_busy.hbf_logic_ingress},
        {"hbf_mapping_dram_issue",
         &time.resource_busy.hbf_mapping_dram_issue},
        {"hbf_plane_media", &time.resource_busy.hbf_plane_media},
        {"hbf_media_lane", &time.resource_busy.hbf_media_lane},
        {"hbf_subarray", &time.resource_busy.hbf_subarray},
        {"hbf_page_buffer_bank", &time.resource_busy.hbf_page_buffer_bank},
        {"hbf_flash_source_queue", &time.resource_busy.hbf_flash_source_queue},
        {"hbf_channel_command", &time.resource_busy.hbf_channel_command},
        {"hbf_channel_data", &time.resource_busy.hbf_channel_data},
        {"hbf_tsv", &time.resource_busy.hbf_tsv},
        {"hbf_sram", &time.resource_busy.hbf_sram},
        {"hbf_hbio_command", &time.resource_busy.hbf_hbio_command},
        {"hbf_hbio_data", &time.resource_busy.hbf_hbio_data},
        {"hbf_sequencer", &time.resource_busy.hbf_sequencer},
        {"hbf_ecc_issue", &time.resource_busy.hbf_ecc_issue},
        {"base_die_link_read", &time.resource_busy.base_die_link_read},
        {"base_die_link_write", &time.resource_busy.base_die_link_write},
    }};
    for (std::size_t i = 0; i < resources.size(); ++i) {
        out << indent << "    \"" << resources[i].first << "\": ";
        write_resource_busy_json(out, *resources[i].second);
        out << (i + 1 == resources.size() ? "\n" : ",\n");
    }
    out << indent << "  }\n";
    out << indent << '}';
}

void write_summary_json(
    const std::string& output_path,
    const Options& options,
    const SystemConfig& system,
    const RunProvenance& provenance,
    std::size_t op_count,
    std::uint64_t trace_footprint_bytes,
    const std::vector<ScenarioResult>& results,
    SanityVerdict sanity) {
    ensure_parent_dir(output_path);
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open summary JSON output: " + output_path);
    }
    const auto& resolved_hbm = system.hbm;
    const auto& resolved_hbf = system.hbf;
    const auto& resolved_external_backing = system.external;
    // Summary provenance must round-trip every configured/modelled double;
    // six fixed decimals can silently turn small initiation intervals into
    // zero or change an ECC bandwidth sweep point.
    out << std::defaultfloat
        << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\n";
    out << "  \"schema\": {\"name\": \"hbfsim.simulation.summary\", "
        << "\"version\": 19},\n";
    out << "  \"simulator\": {\n";
    out << "    \"name\": \"HBFSim\",\n";
    out << "    \"version\": \"" << json_escape(HBFSIM_VERSION) << "\",\n";
    const hbfsim::app::SourceProvenance source;
    out << "    \"git_commit\": \"" << json_escape(source.git_commit) << "\",\n";
    out << "    \"git_dirty\": " << (source.git_dirty ? "true" : "false") << ",\n";
    out << "    \"tree_hash\": \"" << json_escape(source.tree_hash) << "\",\n";
    out << "    \"source_sha256\": \"" << source.source_sha256 << "\",\n";
    out << "    \"provenance_source\": \"" << json_escape(source.provenance_source) << "\",\n";
    out << "    \"git_state_captured_at\": \"build-time\"\n";
    out << "  },\n";
    out << "  \"build\": {\n";
    out << "    \"type\": \"" << json_escape(HBFSIM_BUILD_TYPE) << "\",\n";
    out << "    \"compiler_id\": \"" << json_escape(HBFSIM_COMPILER_ID) << "\",\n";
    out << "    \"compiler_version\": \""
        << json_escape(HBFSIM_COMPILER_VERSION) << "\",\n";
    out << "    \"executable_path\": "
        << (provenance.executable_path.empty() ? "null" :
            "\"" + json_escape(provenance.executable_path) + "\"") << ",\n";
    out << "    \"executable_file_bytes\": "
        << provenance.executable_file_bytes << ",\n";
    out << "    \"executable_digest\": "
        << (provenance.executable_sha256.empty() ? "null" :
            "{\"algorithm\": \"sha256\", \"value\": \"" +
                provenance.executable_sha256 + "\"}") << "\n";
    out << "  },\n";
    out << "  \"invocation\": {\n";
    out << "    \"working_directory\": \""
        << json_escape(provenance.working_directory) << "\",\n";
    out << "    \"argv\": [";
    for (std::size_t i = 0; i < provenance.command.size(); ++i) {
        if (i != 0) out << ", ";
        out << "\"" << json_escape(provenance.command[i]) << "\"";
    }
    out << "]\n";
    out << "  },\n";
    out << "  \"workload\": {\n";
    out << "    \"source\": \""
        << (options.synthetic_sequential_read_bytes ?
            "synthetic-sequential-read-v1" : "trace-file")
        << "\",\n";
    out << "    \"trace_path\": "
        << (options.trace_path ? "\"" + json_escape(*options.trace_path) + "\"" : "null")
        << ",\n";
    out << "    \"trace_file_bytes\": " << provenance.trace_file_bytes << ",\n";
    out << "    \"trace_digest\": {\"algorithm\": \"sha256\", \"value\": \""
        << provenance.trace_sha256 << "\"},\n";
    out << "    \"initial_image_trace_path\": "
        << (options.initial_image_trace_path ?
            "\"" + json_escape(*options.initial_image_trace_path) + "\"" :
            "null")
        << ",\n";
    out << "    \"initial_image_trace_file_bytes\": "
        << provenance.initial_image_trace_file_bytes << ",\n";
    out << "    \"initial_image_trace_digest\": ";
    if (provenance.initial_image_trace_sha256.empty()) {
        out << "null\n";
    } else {
        out << "{\"algorithm\": \"sha256\", \"value\": \""
            << provenance.initial_image_trace_sha256 << "\"}\n";
    }
    out << "  },\n";
    out << "  \"validation\": {\n";
    out << "    \"status\": \"exploratory_unattached\",\n";
    out << "    \"certificate\": null\n";
    out << "  },\n";
    out << "  \"sanity\": \"" << to_string(sanity) << "\",\n";
    out << "  \"config\": {\n";
    out << "    \"ops\": " << op_count << ",\n";
    out << "    \"trace\": " << (options.trace_path ? "\"" + json_escape(*options.trace_path) + "\"" : "null") << ",\n";
    out << "    \"expected_trace_sha256\": "
        << (options.expected_trace_sha256 ?
            "\"" + *options.expected_trace_sha256 + "\"" :
            "null")
        << ",\n";
    out << "    \"expected_trace_bytes\": ";
    if (options.expected_trace_file_bytes) {
        out << *options.expected_trace_file_bytes;
    } else {
        out << "null";
    }
    out << ",\n";
    out << "    \"initial_image_trace\": "
        << (options.initial_image_trace_path ?
            "\"" + json_escape(*options.initial_image_trace_path) + "\"" :
            "null")
        << ",\n";
    out << "    \"synthetic_sequential_read_bytes\": ";
    if (options.synthetic_sequential_read_bytes) {
        out << *options.synthetic_sequential_read_bytes;
    } else {
        out << "null";
    }
    out << ",\n";
    out << "    \"synthetic_sequential_read_base\": "
        << options.synthetic_sequential_read_base << ",\n";
    out << "    \"generate_semantic_llm\": "
        << (options.generate_semantic_llm_path ?
            "\"" + json_escape(*options.generate_semantic_llm_path) + "\"" : "null")
        << ",\n";
    out << "    \"line_size\": " << options.line_size << ",\n";
    out << "    \"trace_unique_line_bytes\": " << trace_footprint_bytes << ",\n";
    out << "    \"interarrival_ns\": " << options.interarrival_ns << ",\n";
    out << "    \"max_ops\": " << options.max_ops << ",\n";
    const auto resolved_trace_mode = effective_trace_mode(options);
    out << "    \"trace_mode\": \""
        << (resolved_trace_mode == TraceMode::Off ? "off" :
            resolved_trace_mode == TraceMode::Summary ? "summary" :
            resolved_trace_mode == TraceMode::Sampled ? "sampled" : "full")
        << "\",\n";
    out << "    \"requested_scenarios\": [";
    for (std::size_t i = 0; i < options.scenarios.size(); ++i) {
        if (i != 0) out << ", ";
        out << "\"" << json_escape(options.scenarios[i]) << "\"";
    }
    out << "],\n";
    out << "    \"hbm_capacity_bytes\": " << resolved_hbm.device.capacity_bytes << ",\n";
    out << "    \"flat_hbm_bytes\": " << options.flat_hbm_bytes << ",\n";
    out << "    \"static_direct_hbm_bytes\": "
        << options.static_direct_hbm_bytes << ",\n";
    out << "    \"hbf_capacity_bytes\": " << hbf_capacity_bytes(resolved_hbf) << ",\n";
    out << "    \"hbf_to_hbm_capacity_ratio\": "
        << (static_cast<long double>(hbf_capacity_bytes(resolved_hbf)) /
            static_cast<long double>(resolved_hbm.device.capacity_bytes)) << ",\n";
    out << "    \"layer_buffer_bytes\": "
        << options.layer_buffer_bytes << ",\n";
    out << "    \"explicit_residency_contract\": "
        << (options.explicit_residency_contract ? "true" : "false")
        << ",\n";
    out << "    \"residency_contract\": ";
    if (options.explicit_residency_contract) {
        out << "{\n";
        out << "      \"page_size_bytes\": "
            << *options.residency_page_size_bytes << ",\n";
        out << "      \"unique_resident_footprint_bytes\": "
            << *options.residency_unique_footprint_bytes << ",\n";
        out << "      \"immutable_weight_bytes\": "
            << *options.residency_immutable_weight_bytes << ",\n";
        out << "      \"immutable_weight_pages\": "
            << *options.residency_immutable_weight_pages << ",\n";
        out << "      \"static_weight_resident_pages\": "
            << *options.residency_static_weight_pages << ",\n";
        out << "      \"runtime_overhead_bytes\": "
            << *options.residency_runtime_overhead_bytes << ",\n";
        out << "      \"block_table_bytes\": "
            << *options.residency_block_table_bytes << ",\n";
        out << "      \"active_buffer_bytes_per_slot\": "
            << *options.residency_active_buffer_bytes << ",\n";
        out << "      \"kv_region_begin\": "
            << *options.residency_kv_region_begin << ",\n";
        out << "      \"kv_block_stride_bytes\": "
            << *options.residency_kv_block_stride_bytes << ",\n";
        out << "      \"logical_kv_blocks\": "
            << *options.residency_logical_kv_blocks << ",\n";
        out << "      \"hot_kv_blocks\": "
            << *options.residency_hot_kv_blocks << "\n";
        out << "    },\n";
    } else {
        out << "null,\n";
    }
    out << "    \"behavioral_hbm_bytes\": "
        << (options.behavioral_hbm_bytes == 0 ?
            resolved_hbm.device.capacity_bytes :
            options.behavioral_hbm_bytes)
        << ",\n";
    out << "    \"behavioral_promotion_threshold\": "
        << options.behavioral_promotion_threshold << ",\n";
    out << "    \"behavioral_history_pages\": "
        << options.behavioral_history_pages << ",\n";
    out << "    \"max_outstanding_requests\": "
        << options.max_outstanding_requests << ",\n";
    out << "    \"max_hbm_outstanding_requests\": "
        << options.max_hbm_outstanding_requests << ",\n";
    out << "    \"max_hbf_outstanding_requests\": "
        << options.max_hbf_outstanding_requests << ",\n";
    out << "    \"arrival_model\": \""
        << (open_loop_arrivals(options) ? "open-loop" : "closed-loop")
        << "\",\n";
    out << "    \"hbm_issue_mode\": \"" << kHbmIssueMode << "\",\n";
    out << "    \"address_heatmap_bins\": "
        << options.address_heatmap_bins << ",\n";
    out << "    \"hbf_hbm_write_buffer_bytes\": "
        << options.hbf_hbm_write_buffer_bytes << ",\n";
    out << "    \"hbf_hbm_write_buffer_destage_watermark\": "
        << options.hbf_hbm_write_buffer_destage_watermark << ",\n";
    out << "    \"hbm_cooperative_region_base_addr\": "
        << hbm_cooperative_region_base_addr(options, system) << ",\n";
    out << "    \"base_die_link\": {\n";
    out << "      \"read_bw_GBps\": " << system.base_die_link.read_bandwidth_GBps << ",\n";
    out << "      \"write_bw_GBps\": " << system.base_die_link.write_bandwidth_GBps << ",\n";
    out << "      \"latency_ns\": " << system.base_die_link.latency_ns << "\n";
    out << "    },\n";
    out << "    \"external_backing\": {\n";
    out << "      \"kind\": \""
        << hbfsim::physical::external::to_string(resolved_external_backing.kind)
        << "\",\n";
    out << "      \"capacity_bytes\": "
        << resolved_external_backing.capacity_bytes << ",\n";
    out << "      \"page_size_bytes\": "
        << resolved_external_backing.page_size_bytes << ",\n";
    out << "      \"request_segment_bytes\": "
        << resolved_external_backing.request_segment_bytes << ",\n";
    out << "      \"media_channels\": "
        << resolved_external_backing.media_channels << ",\n";
    out << "      \"media_read_queues\": "
        << resolved_external_backing.media_read_queues << ",\n";
    out << "      \"media_write_queues\": "
        << resolved_external_backing.media_write_queues << ",\n";
    out << "      \"max_outstanding_requests\": "
        << resolved_external_backing.max_outstanding_requests << ",\n";
    out << "      \"controller_issue_ns\": "
        << resolved_external_backing.controller_issue_ns << ",\n";
    out << "      \"controller_processing_ns\": "
        << resolved_external_backing.controller_processing_ns << ",\n";
    out << "      \"media_read_latency_ns\": "
        << resolved_external_backing.media_read_latency_ns << ",\n";
    out << "      \"media_write_latency_ns\": "
        << resolved_external_backing.media_write_latency_ns << ",\n";
    out << "      \"media_read_bandwidth_GBps\": "
        << resolved_external_backing.media_read_bandwidth_GBps << ",\n";
    out << "      \"media_write_bandwidth_GBps\": "
        << resolved_external_backing.media_write_bandwidth_GBps << ",\n";
    out << "      \"m2s_bandwidth_GBps\": "
        << resolved_external_backing.m2s_bandwidth_GBps << ",\n";
    out << "      \"s2m_bandwidth_GBps\": "
        << resolved_external_backing.s2m_bandwidth_GBps << ",\n";
    out << "      \"one_way_propagation_ns\": "
        << resolved_external_backing.one_way_propagation_ns << ",\n";
    out << "      \"command_bytes\": "
        << resolved_external_backing.command_bytes << ",\n";
    out << "      \"completion_bytes\": "
        << resolved_external_backing.completion_bytes << "\n";
    out << "    },\n";
    out << "    \"llm\": ";
    if (options.generate_semantic_llm_path) {
        out << "{\n";
        out << "      \"tokens\": " << options.llm_tokens << ",\n";
        out << "      \"layers\": " << options.llm_layers << ",\n";
        out << "      \"weight_base\": " << options.llm_weight_base << ",\n";
        out << "      \"kv_base\": " << options.llm_kv_base << ",\n";
        out << "      \"scratch_base\": " << options.llm_scratch_base << "\n";
        out << "    },\n";
    } else {
        out << "null,\n";
    }
    out << "    \"hbm\": {\n";
    out << "      \"capacity_bytes\": " << resolved_hbm.device.capacity_bytes << ",\n";
    out << "      \"stacks\": " << resolved_hbm.device.stacks << ",\n";
    out << "      \"channels\": " << resolved_hbm.device.channels_per_stack << ",\n";
    out << "      \"pseudo_channels\": "
        << resolved_hbm.device.pseudo_channels_per_channel << ",\n";
    out << "      \"address_mapping_scheme\": \""
        << resolved_hbm.address_mapping_scheme() << "\",\n";
    out << "      \"interleave_bytes\": "
        << resolved_hbm.effective_interleave_bytes() << ",\n";
    out << "      \"bank_groups_per_pseudo_channel\": "
        << resolved_hbm.device.bank_groups_per_pseudo_channel << ",\n";
    out << "      \"banks_per_pseudo_channel\": "
        << static_cast<std::uint64_t>(resolved_hbm.device.bank_groups_per_pseudo_channel) *
            resolved_hbm.device.banks_per_group << ",\n";
    out << "      \"banks_per_group\": " << resolved_hbm.device.banks_per_group << ",\n";
    out << "      \"channel_row_size_bytes\": "
        << resolved_hbm.device.channel_row_size_bytes << ",\n";
    out << "      \"row_size_bytes_per_pseudo_channel\": "
        << resolved_hbm.row_size_bytes() << ",\n";
    out << "      \"channel_width_bits\": "
        << resolved_hbm.device.channel_width_bits << ",\n";
    out << "      \"pseudo_channel_width_bits\": "
        << resolved_hbm.pseudo_channel_width_bits() << ",\n";
    out << "      \"pin_rate_Gbps\": " << resolved_hbm.device.pin_rate_Gbps << ",\n";
    out << "      \"data_rate_per_command_clock\": "
        << resolved_hbm.device.data_rate_per_command_clock << ",\n";
    out << "      \"command_clock_MHz\": "
        << resolved_hbm.command_clock_MHz() << ",\n";
    out << "      \"command_clock_period_ns\": "
        << resolved_hbm.command_clock_period_ns() << ",\n";
    out << "      \"burst_length\": " << resolved_hbm.device.burst_length << ",\n";
    out << "      \"burst_bytes\": " << resolved_hbm.burst_bytes() << ",\n";
    out << "      \"burst_duration_ns\": "
        << resolved_hbm.burst_duration_ns() << ",\n";
    out << "      \"pseudo_channel_bw_GBps\": "
        << resolved_hbm.pseudo_channel_bandwidth_GBps() << ",\n";
    out << "      \"channel_bw_GBps\": "
        << resolved_hbm.channel_bandwidth_GBps() << ",\n";
    out << "      \"stack_peak_bw_GBps\": "
        << resolved_hbm.channel_bandwidth_GBps() * resolved_hbm.device.channels_per_stack
        << ",\n";
    out << "      \"system_peak_bw_GBps\": "
        << hbm_peak_rate_GBps(resolved_hbm) << ",\n";
    out << "      \"address_mapping_ns\": "
        << resolved_hbm.controller.address_mapping_ns << ",\n";
    out << "      \"timing_model\": \"" << resolved_hbm.timing_model() << "\",\n";
    out << "      \"read_latency_ns\": " << resolved_hbm.timing.read_latency_ns << ",\n";
    out << "      \"write_latency_ns\": " << resolved_hbm.timing.write_latency_ns << ",\n";
    out << "      \"read_to_write_ns\": " << resolved_hbm.timing.read_to_write_ns << ",\n";
    out << "      \"write_to_read_ns\": " << resolved_hbm.timing.write_to_read_ns << ",\n";
    out << "      \"bandwidth_efficiency\": " << resolved_hbm.timing.bandwidth_efficiency << ",\n";
    out << "      \"service_quantum_bytes\": " << resolved_hbm.controller.service_quantum_bytes << ",\n";
    out << "      \"service_group_channels\": " << resolved_hbm.controller.service_group_channels << ",\n";
    out << "      \"queue_depth\": " << resolved_hbm.controller.queue_depth << "\n";
    out << "    },\n";
    out << "    \"hbf\": {\n";
    out << "      \"capacity_bytes\": " << hbf_capacity_bytes(resolved_hbf) << ",\n";
    out << "      \"processor_interconnect\": \""
        << json_escape(system.hbf_processor_interconnect) << "\",\n";
    out << "      \"processor_interconnect_model_scope\": "
        << "\"technology-label-only; abstract decoded-payload host interface\",\n";
    out << "      \"placement_mapping_scheme\": \""
        << hbfsim::host::kPlacementMappingScheme << "\",\n";
    out << "      \"stacks\": " << resolved_hbf.device.stacks << ",\n";
    out << "      \"channels\": " << resolved_hbf.device.channels_per_stack << ",\n";
    out << "      \"dies_per_channel\": " << resolved_hbf.device.dies_per_channel << ",\n";
    out << "      \"planes_per_die\": " << resolved_hbf.device.planes_per_die << ",\n";
    out << "      \"blocks_per_plane\": " << resolved_hbf.device.blocks_per_plane << ",\n";
    out << "      \"pages_per_block\": " << resolved_hbf.device.pages_per_block << ",\n";
    out << "      \"page_size\": " << resolved_hbf.device.page_size_bytes << ",\n";
    out << "      \"oob_bytes\": " << resolved_hbf.device.oob_bytes_per_page << ",\n";
    out << "      \"media_lanes_per_plane\": "
        << resolved_hbf.device.media_lanes_per_plane << ",\n";
    out << "      \"subarrays_per_plane\": "
        << 1u << ",\n";
    out << "      \"page_buffer_banks_per_plane\": "
        << resolved_hbf.device.page_buffer_banks_per_plane << ",\n";
    out << "      \"read_ns\": " << resolved_hbf.device.t_read_page_ns << ",\n";
    out << "      \"program_ns\": " << resolved_hbf.device.t_program_page_ns << ",\n";
    out << "      \"erase_ns\": " << resolved_hbf.device.t_erase_block_ns << ",\n";
    out << "      \"ecc_decode_latency_ns\": "
        << resolved_hbf.device.ecc_decode_latency_ns << ",\n";
    out << "      \"ecc_encode_latency_ns\": "
        << resolved_hbf.device.ecc_encode_latency_ns << ",\n";
    out << "      \"ecc_decode_raw_bw_GBps_per_die\": "
        << resolved_hbf.device.ecc_decode_raw_bandwidth_GBps_per_die << ",\n";
    out << "      \"ecc_encode_raw_bw_GBps_per_die\": "
        << resolved_hbf.device.ecc_encode_raw_bandwidth_GBps_per_die << ",\n";
    out << "      \"ecc_codeword_size_bytes\": "
        << resolved_hbf.device.page_size_bytes + resolved_hbf.device.oob_bytes_per_page << ",\n";
    out << "      \"ecc_decode_initiation_ns\": "
        << hbfsim::physical::transfer_time_ns(
            resolved_hbf.device.page_size_bytes + resolved_hbf.device.oob_bytes_per_page,
            resolved_hbf.device.ecc_decode_raw_bandwidth_GBps_per_die) << ",\n";
    out << "      \"ecc_encode_initiation_ns\": "
        << hbfsim::physical::transfer_time_ns(
            resolved_hbf.device.page_size_bytes + resolved_hbf.device.oob_bytes_per_page,
            resolved_hbf.device.ecc_encode_raw_bandwidth_GBps_per_die) << ",\n";
    out << "      \"ecc_issue_topology\": \"shared-per-die\",\n";
    out << "      \"bandwidth_semantics\": {"
        << "\"channel\":\"raw-codeword-per-channel\","
        << "\"tsv\":\"shared-command-and-raw-codeword-per-stack\","
        << "\"hbio\":\"decoded-payload-per-channel-per-direction\","
        << "\"media_lane\":\"raw-codeword-per-lane\","
        << "\"page_buffer\":\"raw-codeword-per-bank\","
        << "\"logic_sram\":\"decoded-payload-per-stack\"},\n";
    out << "      \"channel_bw_GBps\": " << resolved_hbf.device.channel_bandwidth_GBps << ",\n";
    out << "      \"hbio_bw_GBps\": " << resolved_hbf.device.hb_io_bandwidth_GBps() << ",\n";
    out << "      \"system_peak_bw_GBps\": "
        << hbf_peak_rate_GBps(resolved_hbf) << ",\n";
    out << "      \"tsv_bw_GBps\": " << resolved_hbf.device.tsv_bandwidth_GBps << ",\n";
    out << "      \"media_lane_bw_GBps\": "
        << resolved_hbf.device.media_lane_bandwidth_GBps << ",\n";
    out << "      \"logic_sram_bw_GBps\": "
        << resolved_hbf.device.logic_sram_bandwidth_GBps << ",\n";
    out << "      \"page_buffer_bw_GBps\": "
        << resolved_hbf.device.page_buffer_bandwidth_GBps << ",\n";
    out << "      \"command_address_bytes\": "
        << resolved_hbf.device.command_address_bytes << ",\n";
    out << "      \"mapping_mode\": \""
        << hbfsim::host::to_string(resolved_hbf.host.mapping_mode)
        << "\",\n";
    out << "      \"mapping_organization\": \""
        << hbfsim::host::to_string(resolved_hbf.host.mapping_organization) << "\",\n";
    out << "      \"mapping_log_blocks\": " << resolved_hbf.host.mapping_log_blocks << ",\n";
    out << "      \"mapping_superblock_planes\": " << resolved_hbf.host.mapping_superblock_planes << ",\n";
    out << "      \"ctrl_dram_bytes\": " << resolved_hbf.host.ctrl_dram_bytes << ",\n";
    out << "      \"ctrl_dram_capacity_denominator\": "
        << resolved_hbf.host.ctrl_dram_capacity_denominator << ",\n";
    out << "      \"ctrl_dram_latency_ns\": "
        << resolved_hbf.host.ctrl_dram_latency_ns << ",\n";
    out << "      \"ctrl_dram_issue_ns\": "
        << resolved_hbf.host.ctrl_dram_issue_ns << ",\n";
    out << "      \"flash_tsu_issue_ns\": "
        << resolved_hbf.device.flash_tsu_issue_ns << ",\n";
    out << "      \"logic_scheduler_issue_ns\": "
        << resolved_hbf.device.logic_scheduler_issue_ns << ",\n";
    out << "      \"address_generation_ns\": "
        << resolved_hbf.device.address_generation_ns << ",\n";
    out << "      \"mapping_update_ns\": "
        << resolved_hbf.host.mapping_update_ns << ",\n";
    out << "      \"free_page_allocation_ns\": "
        << resolved_hbf.host.free_page_allocation_ns << ",\n";
    out << "      \"mapping_entries_per_page\": "
        << resolved_hbf.host.mapping_entries_per_page << ",\n";
    out << "      \"page_read_queue_depth_per_stack\": "
        << resolved_hbf.device.page_read_queue_depth_per_stack << ",\n";
    out << "      \"write_accumulation_timeout_ns\": "
        << resolved_hbf.device.write_accumulation_timeout_ns << ",\n";
    out << "      \"outstanding_write_pages_per_channel\": "
        << resolved_hbf.device.outstanding_write_pages_per_channel << ",\n";
    out << "      \"auto_gc_enabled\": "
        << (resolved_hbf.host.auto_gc_enabled ? "true" : "false") << ",\n";
    out << "      \"gc_low_watermark_pages\": "
        << resolved_hbf.host.gc_low_watermark_pages << ",\n";
    out << "      \"gc_hard_watermark_pages\": "
        << resolved_hbf.host.gc_hard_watermark_pages << ",\n";
    out << "      \"gc_reserved_free_blocks_per_plane\": "
        << resolved_hbf.host.gc_reserved_free_blocks_per_plane << ",\n";
    out << "      \"gc_wear_leveling_weight\": "
        << resolved_hbf.host.gc_wear_leveling_weight << ",\n";
    out << "      \"static_wear_leveling_erase_gap\": "
        << resolved_hbf.host.static_wear_leveling_erase_gap << ",\n";
    out << "      \"static_wear_leveling_interval_erases\": "
        << resolved_hbf.host.static_wear_leveling_interval_erases << ",\n";
    out << "      \"static_wear_leveling_start_erases\": "
        << resolved_hbf.host.static_wear_leveling_start_erases << ",\n";
    out << "      \"static_wear_leveling_stop_gap\": " << resolved_hbf.host.static_wear_leveling_stop_gap << ",\n";
    out << "      \"static_wear_leveling_cooldown_erases\": " << resolved_hbf.host.static_wear_leveling_cooldown_erases << ",\n";
    out << "      \"static_wear_leveling_max_write_fraction\": " << resolved_hbf.host.static_wear_leveling_max_write_fraction << ",\n";
    out << "      \"write_coalescing\": "
        << (resolved_hbf.host.write_coalescing_enabled ? "true" : "false") << ",\n";
    out << "      \"write_buffer_completion_requires_flush\": "
        << (resolved_hbf.host.write_buffer_completion_requires_flush ? "true" : "false")
        << ",\n";
    out << "      \"write_buffer_pages\": " << resolved_hbf.host.write_buffer_pages << ",\n";
    out << "      \"write_buffer_flush_threshold_pages\": "
        << resolved_hbf.host.write_buffer_flush_threshold_pages << ",\n";
    out << "      \"thermal_enabled\": "
        << (resolved_hbf.device.thermal_enabled ? "true" : "false") << ",\n";
    out << "      \"thermal_start_state\": \""
        << (resolved_hbf.device.thermal_start_at_ceiling ? "throttle-ceiling" : "idle")
        << "\",\n";
    out << "      \"thermal_ambient_c\": "
        << resolved_hbf.device.thermal_ambient_c << ",\n";
    out << "      \"thermal_resistance_c_per_w\": "
        << resolved_hbf.device.thermal_resistance_c_per_w << ",\n";
    out << "      \"thermal_capacitance_j_per_c\": "
        << resolved_hbf.device.thermal_capacitance_j_per_c << ",\n";
    out << "      \"thermal_throttle_c\": "
        << resolved_hbf.device.thermal_throttle_c << ",\n";
    out << "      \"thermal_release_c\": "
        << resolved_hbf.device.thermal_release_c << ",\n";
    out << "      \"thermal_static_power_w\": "
        << resolved_hbf.device.thermal_static_power_w << ",\n";
    out << "      \"thermal_read_energy_pj_per_bit\": "
        << resolved_hbf.device.thermal_read_energy_pj_per_bit << ",\n";
    out << "      \"thermal_program_energy_pj_per_bit\": "
        << resolved_hbf.device.thermal_program_energy_pj_per_bit << ",\n";
    out << "      \"thermal_erase_energy_uj_per_block\": "
        << resolved_hbf.device.thermal_erase_energy_uj_per_block << ",\n";
    out << "      \"thermal_neighbor_heat_c\": "
        << resolved_hbf.device.thermal_neighbor_heat_c << ",\n";
    out << "      \"thermal_throttle_power_w\": "
        << resolved_hbf.device.thermal_throttle_power_w << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"scenarios\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        out << "    {\n";
        out << "      \"name\": \"" << json_escape(result.name) << "\",\n";
        out << "      \"ops\": " << result.ops << ",\n";
        out << "      \"reads\": " << result.reads << ",\n";
        out << "      \"writes\": " << result.writes << ",\n";
        out << "      \"logical_bytes\": " << result.logical_bytes << ",\n";
        out << "      \"hbm_accesses\": " << result.hbm_accesses << ",\n";
        out << "      \"hbf_accesses\": " << result.hbf_accesses << ",\n";
        out << "      \"external_accesses\": "
            << result.external_accesses << ",\n";
        out << "      \"hbm_user_accesses\": " << result.hbm_user_accesses << ",\n";
        out << "      \"hbf_user_accesses\": " << result.hbf_user_accesses << ",\n";
        out << "      \"hbf_direct_user_ops\": " << result.hbf_direct_user_ops << ",\n";
        out << "      \"hbm_background_accesses\": " << result.hbm_background_accesses << ",\n";
        out << "      \"hbf_background_accesses\": " << result.hbf_background_accesses << ",\n";
        out << "      \"external_background_accesses\": "
            << result.external_background_accesses << ",\n";
        out << "      \"behavioral_tiering\": ";
        if (result.behavioral_tiering_stats) {
            const auto& s = *result.behavioral_tiering_stats;
            out << "{\n";
            out << "        \"policy\": \""
                << hbfsim::policy::to_string(
                       s.admission_policy)
                << "\",\n";
            out << "        \"semantic_inputs_consumed\": "
                << (s.semantic_inputs_consumed ? "true" : "false")
                << ",\n";
            out << "        \"hbm_tier_pages\": "
                << s.hbm_tier_pages << ",\n";
            out << "        \"hbm_tier_bytes\": "
                << s.hbm_tier_bytes << ",\n";
            out << "        \"promotion_threshold\": "
                << s.promotion_threshold << ",\n";
            out << "        \"history_capacity_pages\": "
                << s.history_capacity_pages << ",\n";
            out << "        \"page_observations\": "
                << s.page_observations << ",\n";
            out << "        \"hbm_hits\": " << s.hbm_hits << ",\n";
            out << "        \"hbf_bypasses\": "
                << s.hbf_bypasses << ",\n";
            out << "        \"cold_misses\": "
                << s.cold_misses << ",\n";
            out << "        \"history_hits\": "
                << s.history_hits << ",\n";
            out << "        \"promotions\": "
                << s.promotions << ",\n";
            out << "        \"promotions_with_backing_fill\": "
                << s.promotions_with_backing_fill << ",\n";
            out << "        \"promotions_without_backing_fill\": "
                << s.promotions_without_backing_fill << ",\n";
            out << "        \"clean_evictions\": "
                << s.clean_evictions << ",\n";
            out << "        \"dirty_evictions\": "
                << s.dirty_evictions << ",\n";
            out << "        \"history_evictions\": "
                << s.history_evictions << ",\n";
            out << "        \"hbm_foreground_bytes\": "
                << s.hbm_foreground_bytes << ",\n";
            out << "        \"hbf_bypass_bytes\": "
                << s.hbf_bypass_bytes << ",\n";
            out << "        \"backing_fill_pages\": "
                << s.backing_fill_pages << ",\n";
            out << "        \"backing_fill_bytes\": "
                << s.backing_fill_bytes << ",\n";
            out << "        \"hbm_install_pages\": "
                << s.hbm_install_pages << ",\n";
            out << "        \"hbm_install_bytes\": "
                << s.hbm_install_bytes << ",\n";
            out << "        \"dirty_writeback_pages\": "
                << s.dirty_writeback_pages << ",\n";
            out << "        \"dirty_writeback_bytes\": "
                << s.dirty_writeback_bytes << ",\n";
            out << "        \"drain_writeback_pages\": "
                << s.drain_writeback_pages << ",\n";
            out << "        \"drain_writeback_bytes\": "
                << s.drain_writeback_bytes << ",\n";
            out << "        \"peak_resident_pages\": "
                << s.peak_resident_pages << ",\n";
            out << "        \"final_resident_pages\": "
                << s.final_resident_pages << ",\n";
            out << "        \"final_dirty_pages\": "
                << s.final_dirty_pages << ",\n";
            out << "        \"peak_history_pages\": "
                << s.peak_history_pages << ",\n";
            out << "        \"peak_installing_pages\": "
                << s.peak_installing_pages << ",\n";
            out << "        \"capacity_stalled_promotions\": "
                << s.capacity_stalled_promotions << ",\n";
            out << "        \"capacity_stall_work_ns\": "
                << s.capacity_stall_work_ns << ",\n";
            out << "        \"transition_waited_transactions\": "
                << s.transition_waited_transactions << ",\n";
            out << "        \"transition_wait_work_ns\": "
                << s.transition_wait_work_ns << ",\n";
            out << "        \"transition_max_wait_ns\": "
                << s.transition_max_wait_ns << ",\n";
            out << "        \"decision_fingerprint\": "
                << s.decision_fingerprint << "\n";
            out << "      },\n";
        } else {
            out << "null,\n";
        }
        out << "      \"layer_streaming\": ";
        if (result.layer_streaming_stats) {
            const auto& s = *result.layer_streaming_stats;
            out << "{\n";
            out << "        \"mode\": \""
                << layer_streaming_mode_name(s) << "\",\n";
            out << "        \"backing\": \""
                << hbfsim::policy::to_string(s.backing)
                << "\",\n";
            out << "        \"residency_policy\": \""
                << layer_streaming_residency_policy_name(s)
                << "\",\n";
            out << "        \"compact_resident_mapping\": "
                << (s.compact_resident_mapping ? "true" : "false")
                << ",\n";
            out << "        \"semantic_inputs_consumed\": "
                << (s.semantic_inputs_consumed ? "true" : "false")
                << ",\n";
            out << "        \"explicit_residency_contract\": "
                << (s.explicit_residency_contract ? "true" : "false")
                << ",\n";
            out << "        \"address_footprint_bytes\": "
                << s.address_footprint_bytes << ",\n";
            out << "        \"unique_resident_footprint_pages\": "
                << s.unique_resident_footprint_pages << ",\n";
            out << "        \"unique_resident_footprint_bytes\": "
                << s.unique_resident_footprint_bytes << ",\n";
            out << "        \"capacity_pressure_basis_bytes\": "
                << s.capacity_pressure_basis_bytes << ",\n";
            out << "        \"footprint_page_rounding_bytes\": "
                << s.footprint_page_rounding_bytes << ",\n";
            out << "        \"hbm_capacity_bytes\": "
                << s.hbm_capacity_bytes << ",\n";
            out << "        \"hbm_capacity_pressure\": "
                << s.hbm_capacity_pressure << ",\n";
            out << "        \"layers\": " << s.layers << ",\n";
            out << "        \"explicit_layer_requests\": "
                << s.explicit_layer_requests << ",\n";
            out << "        \"explicit_compute_layers\": "
                << s.explicit_compute_layers << ",\n";
            out << "        \"compute_work_ns\": "
                << s.compute_work_ns << ",\n";
            out << "        \"hbm_only_resident_pages\": "
                << s.hbm_only_resident_pages << ",\n";
            out << "        \"hbm_only_resident_bytes\": "
                << s.hbm_only_resident_bytes << ",\n";
            out << "        \"hot_kv_candidate_pages\": "
                << s.hot_kv_candidate_pages << ",\n";
            out << "        \"hot_kv_candidate_bytes\": "
                << s.hot_kv_candidate_bytes << ",\n";
            out << "        \"hot_kv_resident_pages\": "
                << s.hot_kv_resident_pages << ",\n";
            out << "        \"hot_kv_resident_bytes\": "
                << s.hot_kv_resident_bytes << ",\n";
            out << "        \"data_pages\": " << s.data_pages << ",\n";
            out << "        \"data_bytes\": " << s.data_bytes << ",\n";
            out << "        \"model_weight_resident_pages\": "
                << s.model_weight_resident_pages << ",\n";
            out << "        \"model_weight_resident_bytes\": "
                << s.model_weight_resident_bytes << ",\n";
            out << "        \"model_weight_backing_pages\": "
                << s.model_weight_backing_pages << ",\n";
            out << "        \"model_weight_backing_bytes\": "
                << s.model_weight_backing_bytes << ",\n";
            out << "        \"cold_kv_backing_pages\": "
                << s.cold_kv_backing_pages << ",\n";
            out << "        \"cold_kv_backing_bytes\": "
                << s.cold_kv_backing_bytes << ",\n";
            out << "        \"unknown_backing_pages\": "
                << s.unknown_backing_pages << ",\n";
            out << "        \"unknown_backing_bytes\": "
                << s.unknown_backing_bytes << ",\n";
            out << "        \"backing_unique_pages\": "
                << s.backing_unique_pages << ",\n";
            out << "        \"backing_unique_bytes\": "
                << s.backing_unique_bytes << ",\n";
            out << "        \"resident_physical_pages\": "
                << s.resident_physical_pages << ",\n";
            out << "        \"resident_physical_bytes\": "
                << s.resident_physical_bytes << ",\n";
            out << "        \"effective_layer_buffer_pages\": "
                << s.effective_layer_buffer_pages << ",\n";
            out << "        \"effective_layer_buffer_bytes\": "
                << s.effective_layer_buffer_bytes << ",\n";
            out << "        \"unused_hbm_pages\": "
                << s.unused_hbm_pages << ",\n";
            out << "        \"unused_hbm_bytes\": "
                << s.unused_hbm_bytes << ",\n";
            out << "        \"streamed_pages\": " << s.streamed_pages << ",\n";
            out << "        \"streamed_bytes\": " << s.streamed_bytes << ",\n";
            out << "        \"foreground_resident_page_accesses\": "
                << s.foreground_resident_page_accesses << ",\n";
            out << "        \"foreground_buffer_page_accesses\": "
                << s.foreground_buffer_page_accesses << ",\n";
            out << "        \"dirty_pages_written_back\": "
                << s.dirty_pages_written_back << ",\n";
            out << "        \"writeback_bytes\": " << s.writeback_bytes << ",\n";
            out << "        \"max_layer_data_pages\": "
                << s.max_layer_data_pages << ",\n";
            out << "        \"max_layer_data_bytes\": "
                << s.max_layer_data_bytes << ",\n";
            out << "        \"immutable_weight_logical_bytes\": "
                << s.immutable_weight_logical_bytes << ",\n";
            out << "        \"runtime_overhead_logical_bytes\": "
                << s.runtime_overhead_logical_bytes << ",\n";
            out << "        \"block_table_logical_bytes\": "
                << s.block_table_logical_bytes << ",\n";
            out << "        \"active_buffer_logical_bytes_per_slot\": "
                << s.active_buffer_logical_bytes_per_slot << ",\n";
            out << "        \"residency_page_size_bytes\": "
                << s.residency_page_size_bytes << ",\n";
            out << "        \"kv_block_stride_bytes\": "
                << s.kv_block_stride_bytes << ",\n";
            out << "        \"logical_kv_blocks\": "
                << s.logical_kv_blocks << ",\n";
            out << "        \"hot_kv_blocks\": "
                << s.hot_kv_blocks << ",\n";
            out << "        \"cold_kv_blocks\": "
                << s.cold_kv_blocks << ",\n";
            out << "        \"backing_request_credit_limit\": "
                << s.backing_request_credit_limit << ",\n";
            out << "        \"backing_max_inflight_requests\": "
                << s.backing_max_inflight_requests << ",\n";
            out << "        \"backing_admission_waited_requests\": "
                << s.backing_admission_waited_requests << ",\n";
            out << "        \"backing_admission_wait_work_ns\": "
                << s.backing_admission_wait_work_ns << ",\n";
            out << "        \"backing_admission_max_wait_ns\": "
                << s.backing_admission_max_wait_ns << ",\n";
            out << "        \"user_waited_ops\": " << s.user_waited_ops << ",\n";
            out << "        \"user_wait_work_ns\": " << s.user_wait_work_ns << ",\n";
            out << "        \"user_max_wait_ns\": " << s.user_max_wait_ns << ",\n";
            out << "        \"exposed_prefetch_ns\": "
                << s.exposed_prefetch_ns << ",\n";
            out << "        \"hidden_prefetch_ns\": "
                << s.hidden_prefetch_ns << ",\n";
            out << "        \"buffer_reuse_wait_work_ns\": "
                << s.buffer_reuse_wait_work_ns << "\n";
            out << "      },\n";
        } else {
            out << "null,\n";
        }
        out << "      \"hybrid_path\": {\n";
        out << "        \"background_hbf_writes\": "
            << result.background_hbf_writes << ",\n";
        out << "        \"hbf_static_read_bytes\": " << result.hbf_static_read_bytes << ",\n";
        out << "        \"hbf_logical_read_bytes\": " << result.hbf_logical_read_bytes << ",\n";
        out << "        \"hbf_backing_write_bytes\": " << result.hbf_backing_write_bytes << ",\n";
        out << "        \"external_backing_read_bytes\": "
            << result.external_backing_read_bytes << ",\n";
        out << "        \"external_backing_write_bytes\": "
            << result.external_backing_write_bytes << ",\n";
        out << "        \"hbm_foreground_bytes\": " << result.hbm_foreground_bytes << ",\n";
        out << "        \"hbm_streaming_write_bytes\": "
            << result.hbm_streaming_write_bytes << ",\n";
        out << "        \"base_die_link_read_bytes\": "
            << result.base_die_link_stats.read_bytes << ",\n";
        out << "        \"base_die_link_write_bytes\": "
            << result.base_die_link_stats.write_bytes << ",\n";
        out << "        \"hbm_write_buffer_user_write_bytes\": "
            << result.hbm_write_buffer_user_write_bytes << ",\n";
        out << "        \"hbm_write_buffer_destaged_bytes\": " << result.hbm_write_buffer_destaged_bytes << ",\n";
        out << "        \"hbm_write_buffer_drain_destaged_bytes\": "
            << result.hbm_write_buffer_drain_destaged_bytes << ",\n";
        out << "        \"hbm_write_buffer_peak_bytes\": " << result.hbm_write_buffer_peak_bytes << ",\n";
        out << "        \"hbm_write_buffer_full_waits\": " << result.hbm_write_buffer_full_waits << ",\n";
        out << "        \"hbm_write_buffer_full_wait_work_ns\": "
            << result.hbm_write_buffer_wait_ns << "\n";
        out << "      },\n";
        write_time_breakdown_json(out, result, "      ");
        out << ",\n";
        out << "      \"user_completion_throughput_GBps\": "
            << user_completion_throughput_GBps(result) << ",\n";
        out << "      \"makespan_throughput_GBps\": "
            << makespan_throughput_GBps(result) << ",\n";
        out << "      \"hbm_issue_mode\": \"" << kHbmIssueMode << "\",\n";
        out << "      \"offered_load\": ";
        if (result.offered_load) {
            const auto& load = *result.offered_load;
            const auto optional_number = [](std::optional<double> value) {
                std::ostringstream text;
                text << std::defaultfloat
                     << std::setprecision(std::numeric_limits<double>::max_digits10);
                if (value) {
                    text << *value;
                } else {
                    text << "null";
                }
                return text.str();
            };
            out << "{\n";
            out << "        \"arrival_model\": \""
                << (load.open_loop ? "open-loop" : "closed-loop") << "\",\n";
            out << "        \"arrival_span_ns\": " << load.arrival_span_ns << ",\n";
            out << "        \"rate_GBps\": " << optional_number(load.rate_GBps) << ",\n";
            out << "        \"read_rate_GBps\": "
                << optional_number(load.read_rate_GBps) << ",\n";
            out << "        \"write_rate_GBps\": "
                << optional_number(load.write_rate_GBps) << ",\n";
            out << "        \"peak_rate_GBps\": {"
                << "\"hbm\": " << optional_number(load.hbm_peak_GBps)
                << ", \"hbf\": " << optional_number(load.hbf_peak_GBps)
                << ", \"external_read\": "
                << optional_number(load.external_read_peak_GBps)
                << ", \"external_write\": "
                << optional_number(load.external_write_peak_GBps) << "},\n";
            out << "        \"offered_to_peak_ratio\": {"
                << "\"hbm\": " << optional_number(load.hbm_ratio)
                << ", \"hbf\": " << optional_number(load.hbf_ratio)
                << ", \"external\": " << optional_number(load.external_ratio)
                << "}\n";
            out << "      },\n";
        } else {
            out << "null,\n";
        }
        out << "      \"hbm_stats\": ";
        if (result.has_hbm) {
            const auto& s = result.hbm_stats;
            out << "{\"read_bytes\":" << s.read_bytes
                << ",\"write_bytes\":" << s.write_bytes
                << ",\"controller_buffer_read_bytes\":" << s.controller_buffer_read_bytes
                << ",\"controller_buffer_write_bytes\":" << s.controller_buffer_write_bytes
                << ",\"controller_buffer_transfers\":" << s.controller_buffer_transfers
                << ",\"controller_buffer_bus_busy_ns\":" << s.controller_buffer_bus_busy_ns
                << ",\"max_queue_occupancy\":" << s.max_queue_occupancy
                << ",\"channel_transfers\":" << s.channel_transfers
                << ",\"service_quanta\":" << s.service_quanta
                << ",\"bus_parallelism\":" << s.bus_parallelism()
                << ",\"pseudo_channels\":" << s.pseudo_channels
                << ",\"active_pseudo_channels\":" << s.active_pseudo_channels
                << ",\"max_pseudo_channel_accesses\":" << s.max_pseudo_channel_accesses
                << ",\"max_pseudo_channel_busy_ns\":" << s.max_pseudo_channel_busy_ns
                << ",\"avg_active_pseudo_channel_busy_ns\":"
                << s.avg_active_pseudo_channel_busy_ns
                << ",\"pseudo_channel_busy_skew\":"
                << s.pseudo_channel_busy_skew() << "},\n";
        } else {
            out << "null,\n";
        }
        out << "      \"hbf_stats\": ";
        if (result.has_hbf) {
            const auto& s = result.hbf_stats;
            out << "{\"read_requests\":" << s.read_requests
                << ",\"program_requests\":" << s.program_requests
                << ",\"erase_requests\":" << s.erase_requests
                << ",\"logical_read_bytes\":" << s.logical_read_bytes
                << ",\"physical_read_bytes\":" << s.physical_read_bytes
                << ",\"logical_write_bytes\":" << s.logical_write_bytes
                << ",\"physical_write_bytes\":" << s.physical_write_bytes
                << ",\"data_program_payload_bytes\":"
                << s.data_program_payload_bytes
                << ",\"raw_physical_program_payload_bytes\":"
                << s.raw_physical_program_payload_bytes
                << ",\"mapping_program_payload_bytes\":"
                << s.mapping_program_payload_bytes
                << ",\"gc_relocation_payload_bytes\":"
                << s.gc_relocation_payload_bytes
                << ",\"static_wear_leveling_relocation_payload_bytes\":"
                << s.static_wear_leveling_relocation_payload_bytes
                << ",\"page_reads\":" << s.page_reads
                << ",\"data_programs\":" << s.data_programs
                << ",\"raw_physical_programs\":"
                << s.raw_physical_programs
                << ",\"page_programs\":" << s.page_programs
                << ",\"auto_erase_requests\":" << s.auto_erase_requests
                << ",\"preconditioned_erased_blocks\":" << s.preconditioned_erased_blocks
                << ",\"host_gc_read_bytes\":" << s.host_gc_read_bytes
        << ",\"host_hbm_reserved_bytes\":" << s.host_hbm_reserved_bytes
        << ",\"host_hbm_read_bytes\":" << s.host_hbm_read_bytes
        << ",\"host_hbm_write_bytes\":" << s.host_hbm_write_bytes
        << ",\"host_hbm_busy_ns\":" << s.host_hbm_busy_ns
        << ",\"host_hbm_queue_wait_ns\":" << s.host_hbm_queue_wait_ns
                << ",\"host_gc_write_bytes\":" << s.host_gc_write_bytes
                << ",\"host_gc_buffer_bytes\":" << s.host_gc_buffer_bytes
                << ",\"block_erases\":" << s.block_erases
                << ",\"writable_blocks\":" << s.writable_blocks
                << ",\"writable_pages\":" << s.writable_pages
                << ",\"writable_payload_bytes\":"
                << s.writable_payload_bytes
                << ",\"worn_blocks\":" << s.worn_blocks
                << ",\"block_erase_count_sum\":"
                << s.block_erase_count_sum
                << ",\"min_block_erase_count\":"
                << s.min_block_erase_count
                << ",\"max_block_erase_count\":"
                << s.max_block_erase_count
                << ",\"block_erase_count_histogram\":";
            {
                out << '{';
                bool first_bin = true;
                for (const auto& [erase_count, blocks] :
                     s.block_erase_count_histogram) {
                    out << (first_bin ? "" : ",") << '"' << erase_count
                        << "\":" << blocks;
                    first_bin = false;
                }
                out << '}';
            }
            out
                << ",\"mean_block_erase_count\":"
                << (s.writable_blocks == 0 ? 0.0 :
                    static_cast<double>(s.block_erase_count_sum) /
                        static_cast<double>(s.writable_blocks))
                << ",\"block_erase_count_stddev\":";
            const auto mean_block_erase_count = s.writable_blocks == 0 ? 0.0 :
                static_cast<double>(s.block_erase_count_sum) /
                    static_cast<double>(s.writable_blocks);
            const auto mean_square_block_erase_count =
                s.writable_blocks == 0 ? 0.0 :
                static_cast<double>(s.block_erase_count_sum_squares /
                    static_cast<long double>(s.writable_blocks));
            out << std::sqrt(std::max(
                       0.0,
                       mean_square_block_erase_count -
                           mean_block_erase_count * mean_block_erase_count))
                << ",\"mapping_entries\":" << s.mapping_entries
                << ",\"invalidations\":" << s.invalidations
                << ",\"total_pages\":" << s.total_pages
                << ",\"free_pages\":" << s.free_pages
                << ",\"valid_pages\":" << s.valid_pages
                << ",\"invalid_pages\":" << s.invalid_pages
                << ",\"pending_program_pages\":"
                << s.pending_program_pages
                << ",\"pending_mapping_publications\":"
                << s.pending_mapping_publications
                << ",\"static_unmaterialized_pages\":"
                << s.static_unmaterialized_pages
                << ",\"accounting_verified\":"
                << (s.accounting_verified ? "true" : "false")
                << ",\"static_reserved_pages\":" << s.static_reserved_pages
                << ",\"raw_reserved_pages\":" << s.raw_reserved_pages
                << ",\"initial_logical_data_pages\":"
                << s.initial_logical_data_pages
                << ",\"initial_mapping_pages\":"
                << s.initial_mapping_pages
                << ",\"compact_initial_logical_data_pages\":"
                << s.compact_initial_logical_data_pages
                << ",\"compact_initial_mapping_pages\":"
                << s.compact_initial_mapping_pages
                << ",\"compact_live_logical_data_pages\":"
                << s.compact_live_logical_data_pages
                << ",\"compact_live_mapping_pages\":"
                << s.compact_live_mapping_pages
                << ",\"compact_retired_logical_data_pages\":"
                << s.compact_retired_logical_data_pages
                << ",\"compact_retired_mapping_pages\":"
                << s.compact_retired_mapping_pages
                << ",\"waf\":";
            if (const auto value = s.waf()) {
                out << *value;
            } else {
                out << "null";
            }
            out << ",\"waf_definition\":"
                << "\"physical_write_bytes/(logical_write_bytes+"
                   "raw_physical_program_payload_bytes)\""
                << ",\"mapping_table_bytes\":"
                << s.mapping_table_bytes
                << ",\"mapping_table_bytes_per_stack\":"
                << s.mapping_table_bytes_per_stack
                << ",\"mapping_table_pages_per_stack\":"
                << s.mapping_table_pages_per_stack
                << ",\"resident_mapping_table_bytes\":"
                << s.resident_mapping_table_bytes
                << ",\"resident_mapping_table_bytes_per_stack\":"
                << s.resident_mapping_table_bytes_per_stack
                << ",\"resident_mapping_pages_per_stack\":"
                << s.resident_mapping_pages_per_stack
                << ",\"mapping_directory_entry_bytes\":"
                << s.mapping_directory_entry_bytes
                << ",\"mapping_directory_bytes\":"
                << s.mapping_directory_bytes
                << ",\"mapping_directory_bytes_per_stack\":"
                << s.mapping_directory_bytes_per_stack
                << ",\"mapping_cache_capacity_bytes\":"
                << s.mapping_cache_capacity_bytes
                << ",\"mapping_cache_capacity_bytes_per_stack\":"
                << s.mapping_cache_capacity_bytes_per_stack
                << ",\"mapping_cache_pages_per_stack\":"
                << s.mapping_cache_pages_per_stack
                << ",\"controller_dram_budget_bytes\":"
                << s.controller_dram_budget_bytes
                << ",\"controller_dram_budget_bytes_per_stack\":"
                << s.controller_dram_budget_bytes_per_stack
                << ",\"write_buffer_capacity_bytes\":"
                << s.write_buffer_capacity_bytes
                << ",\"write_buffer_capacity_bytes_per_stack\":"
                << s.write_buffer_capacity_bytes_per_stack
                << ",\"mapping_cache_entries\":"
                << s.mapping_cache_entries
                << ",\"mapping_cache_peak_entries\":"
                << s.mapping_cache_peak_entries
                << ",\"mapping_cache_hits\":"
                << s.mapping_cache_hits
                << ",\"mapping_cache_misses\":"
                << s.mapping_cache_misses
                << ",\"mapping_cache_erased_misses\":"
                << s.mapping_cache_erased_misses
                << ",\"mapping_cache_coalesced_misses\":"
                << s.mapping_cache_coalesced_misses
                << ",\"mapping_cache_evictions\":"
                << s.mapping_cache_evictions
                << ",\"mapping_cache_dirty_evictions\":"
                << s.mapping_cache_dirty_evictions
                << ",\"mapping_media_reads\":"
                << s.mapping_media_reads
                << ",\"mapping_media_read_bytes\":"
                << s.mapping_media_read_bytes
                << ",\"mapping_lookup_ops\":" << s.mapping_lookup_ops
                << ",\"mapping_user_lookup_ops\":"
                << s.mapping_user_lookup_ops
                << ",\"mapping_gc_lookup_ops\":"
                << s.mapping_gc_lookup_ops
                << ",\"mapping_update_ops\":" << s.mapping_update_ops
                << ",\"mapping_user_update_ops\":"
                << s.mapping_user_update_ops
                << ",\"mapping_gc_update_ops\":"
                << s.mapping_gc_update_ops
                << ",\"mapping_dram_wait_ops\":"
                << s.mapping_dram_wait_ops
                << ",\"mapping_dram_wait_work_ns\":"
                << s.mapping_dram_wait_ns
                << ",\"mapping_dram_wait_max_ns\":"
                << s.mapping_dram_wait_max_ns
                << ",\"mapping_dram_issue_busy_ns\":"
                << s.mapping_dram_issue_busy_ns
                << ",\"mapping_dram_resources\":"
                << s.mapping_dram_resources
                << ",\"mapping_page_programs\":" << s.mapping_page_programs
                << ",\"device_dram_capacity_bytes\":" << s.device_dram_capacity_bytes
                << ",\"device_dram_capacity_bytes_per_stack\":"
                << s.device_dram_capacity_bytes_per_stack
                << ",\"device_dram_read_cache_pages_per_stack\":"
                << s.device_dram_read_cache_pages_per_stack
                << ",\"device_dram_read_hits\":" << s.device_dram_read_hits
                << ",\"device_dram_read_hit_bytes\":" << s.device_dram_read_hit_bytes
                << ",\"device_dram_read_misses\":" << s.device_dram_read_misses
                << ",\"device_dram_fills\":" << s.device_dram_fills
                << ",\"device_dram_fill_bypasses\":" << s.device_dram_fill_bypasses
                << ",\"device_dram_evictions\":" << s.device_dram_evictions
                << ",\"device_dram_read_bytes\":" << s.device_dram_read_bytes
                << ",\"device_dram_write_bytes\":" << s.device_dram_write_bytes
                << ",\"read_buffer_hits\":" << s.read_buffer_hits
                << ",\"read_buffer_misses\":" << s.read_buffer_misses
                << ",\"read_buffer_read_bytes\":" << s.read_buffer_read_bytes
                << ",\"write_buffer_hits\":" << s.write_buffer_hits
                << ",\"write_buffer_misses\":" << s.write_buffer_misses
                << ",\"write_buffer_flushes\":" << s.write_buffer_flushes
                << ",\"write_buffer_merged_bytes\":" << s.write_buffer_merged_bytes
                << ",\"write_buffer_read_hits\":" << s.write_buffer_read_hits
                << ",\"write_buffer_read_bytes\":" << s.write_buffer_read_bytes
                << ",\"write_buffer_dram_read_ops\":"
                << s.write_buffer_dram_read_ops
                << ",\"write_buffer_dram_write_ops\":"
                << s.write_buffer_dram_write_ops
                << ",\"write_buffer_dram_read_bytes\":"
                << s.write_buffer_dram_read_bytes
                << ",\"write_buffer_dram_write_bytes\":"
                << s.write_buffer_dram_write_bytes
                << ",\"write_buffer_dram_wait_ops\":"
                << s.write_buffer_dram_wait_ops
                << ",\"write_buffer_dram_wait_work_ns\":"
                << s.write_buffer_dram_wait_ns
                << ",\"write_buffer_dram_wait_max_ns\":"
                << s.write_buffer_dram_wait_max_ns
                << ",\"write_buffer_dram_issue_busy_ns\":"
                << s.write_buffer_dram_issue_busy_ns
                << ",\"write_buffer_slot_wait_ops\":" << s.write_buffer_slot_wait_ops
                << ",\"write_buffer_slot_wait_work_ns\":"
                << s.write_buffer_slot_wait_ns
                << ",\"read_splits\":" << s.read_splits
                << ",\"read_split_pages\":" << s.read_split_pages
                << ",\"page_read_admission_events\":"
                << s.page_read_admission_events
                << ",\"page_read_admission_waited_pages\":"
                << s.page_read_admission_waited_pages
                << ",\"page_read_admission_wait_work_ns\":"
                << s.page_read_admission_wait_ns
                << ",\"page_read_admission_max_wait_ns\":"
                << s.page_read_admission_max_wait_ns
                << ",\"scalar_read_requests\":"
                << s.scalar_read_requests
                << ",\"scalar_read_pages\":" << s.scalar_read_pages
                << ",\"flash_scheduler_enqueues\":" << s.flash_scheduler_enqueues
                << ",\"flash_scheduler_issues\":" << s.flash_scheduler_issues
                << ",\"gc_runs\":" << s.gc_runs
                << ",\"gc_relocations\":" << s.gc_relocations
                << ",\"gc_data_relocations\":" << s.gc_data_relocations
                << ",\"gc_mapping_relocations\":"
                << s.gc_mapping_relocations
                << ",\"gc_reclaimed_invalid_pages\":"
                << s.gc_reclaimed_invalid_pages
                << ",\"gc_user_blocked_runs\":" << s.gc_user_blocked_runs
                << ",\"static_wear_leveling_checks\":"
                << s.static_wear_leveling_checks
                << ",\"static_wear_leveling_budget_deferrals\":" << s.static_wear_leveling_budget_deferrals
                << ",\"static_wear_leveling_cooldown_exclusions\":" << s.static_wear_leveling_cooldown_exclusions
                << ",\"static_wear_leveling_unique_source_blocks\":" << s.static_wear_leveling_unique_source_blocks
                << ",\"static_wear_leveling_repeat_source_runs\":" << s.static_wear_leveling_repeat_source_runs
                << ",\"static_wear_leveling_runs\":"
                << s.static_wear_leveling_runs
                << ",\"static_wear_leveling_relocations\":"
                << s.static_wear_leveling_relocations
                << ",\"static_wear_leveling_reclaimed_invalid_pages\":"
                << s.static_wear_leveling_reclaimed_invalid_pages
                << ",\"ecc_decode_queue_wait_work_ns\":"
                << s.ecc_decode_queue_wait_ns
                << ",\"ecc_encode_queue_wait_work_ns\":"
                << s.ecc_encode_queue_wait_ns
                << ",\"ecc_decode_latency_work_ns\":"
                << s.ecc_decode_latency_work_ns
                << ",\"ecc_encode_latency_work_ns\":"
                << s.ecc_encode_latency_work_ns
                << ",\"ecc_decode_issue_busy_ns\":" << s.ecc_decode_issue_busy_ns
                << ",\"ecc_encode_issue_busy_ns\":" << s.ecc_encode_issue_busy_ns
                << ",\"ecc_decode_ops\":" << s.ecc_decode_ops
                << ",\"ecc_encode_ops\":" << s.ecc_encode_ops
                << ",\"ecc_codeword_bytes\":" << s.ecc_codeword_bytes
                << ",\"ecc_decode_codeword_bytes\":" << s.ecc_decode_codeword_bytes
                << ",\"ecc_encode_codeword_bytes\":" << s.ecc_encode_codeword_bytes
                << ",\"ecc_issue_parallelism\":" << s.ecc_issue_parallelism()
                << ",\"active_ecc_dies\":" << s.active_ecc_dies
                << ",\"ecc_max_inflight_per_die\":" << s.max_ecc_inflight_per_die
                << ",\"max_ecc_issue_busy_ns\":" << s.max_ecc_issue_busy_ns
                << ",\"avg_active_ecc_issue_busy_ns\":"
                << s.avg_active_ecc_issue_busy_ns
                << ",\"media_parallelism\":" << s.media_parallelism()
                << ",\"read_lane_parallelism\":" << s.read_lane_parallelism()
                << ",\"subarray_read_parallelism\":" << s.subarray_read_parallelism()
                << ",\"page_buffer_bank_parallelism\":"
                << s.page_buffer_bank_parallelism()
                << ",\"channel_parallelism\":" << s.channel_parallelism()
                << ",\"hbio_parallelism\":" << s.hbio_parallelism()
                << ",\"sequencer_parallelism\":" << s.sequencer_parallelism()
                << ",\"channels\":" << s.channels
                << ",\"active_channels\":" << s.active_channels
                << ",\"dies\":" << s.dies
                << ",\"active_dies\":" << s.active_dies
                << ",\"planes\":" << s.planes
                << ",\"active_planes\":" << s.active_planes
                << ",\"media_lanes\":" << s.media_lanes
                << ",\"active_media_lanes\":" << s.active_media_lanes
                << ",\"subarrays\":" << s.subarrays
                << ",\"active_subarrays\":" << s.active_subarrays
                << ",\"page_buffer_banks\":" << s.page_buffer_banks
                << ",\"active_page_buffer_banks\":"
                << s.active_page_buffer_banks
                << ",\"max_media_lane_busy_ns\":" << s.max_media_lane_busy_ns
                << ",\"avg_active_media_lane_busy_ns\":"
                << s.avg_active_media_lane_busy_ns
                << ",\"media_lane_skew\":" << s.media_lane_skew()
                << ",\"max_media_lane_reads\":" << s.max_media_lane_reads
                << ",\"avg_active_media_lane_reads\":"
                << s.avg_active_media_lane_reads
                << ",\"media_lane_read_skew\":" << s.media_lane_read_skew()
                << ",\"max_subarray_busy_ns\":" << s.max_subarray_busy_ns
                << ",\"avg_active_subarray_busy_ns\":"
                << s.avg_active_subarray_busy_ns
                << ",\"subarray_busy_skew\":" << s.subarray_busy_skew()
                << ",\"max_subarray_reads\":" << s.max_subarray_reads
                << ",\"avg_active_subarray_reads\":"
                << s.avg_active_subarray_reads
                << ",\"subarray_read_skew\":" << s.subarray_read_skew()
                << ",\"max_page_buffer_bank_busy_ns\":"
                << s.max_page_buffer_bank_busy_ns
                << ",\"avg_active_page_buffer_bank_busy_ns\":"
                << s.avg_active_page_buffer_bank_busy_ns
                << ",\"page_buffer_bank_skew\":"
                << s.page_buffer_bank_skew()
                << ",\"max_page_buffer_bank_reads\":"
                << s.max_page_buffer_bank_reads
                << ",\"avg_active_page_buffer_bank_reads\":"
                << s.avg_active_page_buffer_bank_reads
                << ",\"page_buffer_bank_read_skew\":"
                << s.page_buffer_bank_read_skew()
                << ",\"max_plane_media_busy_ns\":" << s.max_plane_media_busy_ns
                << ",\"avg_active_plane_media_busy_ns\":"
                << s.avg_active_plane_media_busy_ns
                << ",\"plane_media_skew\":" << s.plane_media_skew()
                << ",\"max_plane_ops\":" << s.max_plane_ops
                << ",\"avg_active_plane_ops\":" << s.avg_active_plane_ops
                << ",\"plane_op_skew\":" << s.plane_op_skew()
                << ",\"max_channel_busy_ns\":" << s.max_channel_busy_ns
                << ",\"avg_active_channel_busy_ns\":"
                << s.avg_active_channel_busy_ns
                << ",\"channel_busy_skew\":" << s.channel_busy_skew()
                << ",\"max_die_transactions\":" << s.max_die_transactions
                << ",\"avg_active_die_transactions\":"
                << s.avg_active_die_transactions
                << ",\"die_transaction_skew\":" << s.die_transaction_skew()
                << "},\n";
        } else {
            out << "null,\n";
        }
        out << "      \"external_backing_stats\": ";
        if (result.has_external_backing) {
            const auto& s = result.external_backing_stats;
            out << "{\"kind\":\""
                << hbfsim::physical::external::to_string(s.kind)
                << "\",\"read_requests\":" << s.read_requests
                << ",\"write_requests\":" << s.write_requests
                << ",\"read_bytes\":" << s.read_bytes
                << ",\"write_bytes\":" << s.write_bytes
                << ",\"page_run_requests\":" << s.page_run_requests
                << ",\"page_run_segments\":" << s.page_run_segments
                << ",\"page_run_pages\":" << s.page_run_pages
                << ",\"media_channels\":" << s.media_channels
                << ",\"media_channels_per_queue\":"
                << s.media_channels_per_queue
                << ",\"media_read_queues\":" << s.media_read_queues
                << ",\"media_write_queues\":" << s.media_write_queues
                << ",\"active_media_resources\":"
                << s.active_media_resources
                << ",\"max_device_outstanding\":"
                << s.max_device_outstanding
                << ",\"outstanding_wait_work_ns\":"
                << s.outstanding_wait_ns
                << ",\"controller_queue_wait_work_ns\":"
                << s.controller_queue_wait_ns
                << ",\"controller_issue_busy_ns\":"
                << s.controller_issue_busy_ns
                << ",\"controller_processing_work_ns\":"
                << s.controller_processing_work_ns
                << ",\"controller_active_span_ns\":"
                << s.controller_active_span_ns()
                << ",\"controller_utilization\":"
                << s.controller_utilization()
                << ",\"media_queue_wait_work_ns\":"
                << s.media_queue_wait_ns
                << ",\"media_read_latency_work_ns\":"
                << s.media_read_latency_work_ns
                << ",\"media_write_latency_work_ns\":"
                << s.media_write_latency_work_ns
                << ",\"media_read_busy_ns\":" << s.media_read_busy_ns
                << ",\"media_write_busy_ns\":" << s.media_write_busy_ns
                << ",\"media_active_span_ns\":"
                << s.media_active_span_ns()
                << ",\"media_utilization\":" << s.media_utilization()
                << ",\"m2s_payload_bytes\":" << s.m2s_payload_bytes
                << ",\"m2s_protocol_bytes\":" << s.m2s_protocol_bytes
                << ",\"m2s_wire_bytes\":" << s.m2s_wire_bytes
                << ",\"s2m_payload_bytes\":" << s.s2m_payload_bytes
                << ",\"s2m_protocol_bytes\":" << s.s2m_protocol_bytes
                << ",\"s2m_wire_bytes\":" << s.s2m_wire_bytes
                << ",\"m2s_queue_wait_work_ns\":"
                << s.m2s_queue_wait_ns
                << ",\"s2m_queue_wait_work_ns\":"
                << s.s2m_queue_wait_ns
                << ",\"m2s_busy_ns\":" << s.m2s_busy_ns
                << ",\"s2m_busy_ns\":" << s.s2m_busy_ns
                << ",\"transport_propagation_work_ns\":"
                << s.transport_propagation_work_ns
                << ",\"m2s_active_span_ns\":"
                << s.m2s_active_span_ns()
                << ",\"s2m_active_span_ns\":"
                << s.s2m_active_span_ns()
                << ",\"m2s_utilization\":"
                << s.m2s_utilization()
                << ",\"s2m_utilization\":"
                << s.s2m_utilization();
            // The device-cache census appears once the cxl-ssd front cache
            // has served traffic; kinds without a cache never emit it.
            if (s.cache_read_hits + s.cache_read_misses +
                    s.cache_write_hits + s.cache_write_misses > 0) {
                out << ",\"device_cache\":{"
                    << "\"read_hits\":" << s.cache_read_hits
                    << ",\"read_misses\":" << s.cache_read_misses
                    << ",\"write_hits\":" << s.cache_write_hits
                    << ",\"write_misses\":" << s.cache_write_misses
                    << ",\"read_for_ownership_segments\":"
                    << s.cache_read_for_ownership_segments
                    << ",\"read_for_ownership_bytes\":"
                    << s.cache_read_for_ownership_bytes
                    << ",\"writeback_segments\":"
                    << s.cache_writeback_segments
                    << ",\"writeback_bytes\":" << s.cache_writeback_bytes
                    << ",\"prefetch_segments\":"
                    << s.cache_prefetch_segments
                    << ",\"prefetch_bytes\":" << s.cache_prefetch_bytes
                    << ",\"latency_work_ns\":" << s.cache_latency_work_ns
                    << ",\"busy_ns\":" << s.cache_busy_ns
                    << ",\"queue_wait_work_ns\":" << s.cache_queue_wait_ns
                    << ",\"flush_busy_ns\":" << s.cache_flush_busy_ns
                    << ",\"prefetch_busy_ns\":"
                    << s.cache_prefetch_busy_ns
                    << ",\"writeback_gate_wait_work_ns\":"
                    << s.cache_writeback_gate_wait_ns
                    << '}';
            }
            out << "},\n";
        } else {
            out << "null,\n";
        }
        out << "      \"hbf_wear\": "
            << (result.hbf_wear_snapshot.empty() ? "null" : result.hbf_wear_snapshot) << ",\n";
        out << "      \"address_heatmap\": ";
        if (result.address_heatmap) {
            hbfsim::physical::write_address_heatmap_json(
                out, *result.address_heatmap);
        } else {
            out << "null";
        }
        out << ",\n";
        out << "      \"warnings\": [";
        for (std::size_t w = 0; w < result.warnings.size(); ++w) {
            if (w != 0) out << ", ";
            out << "\"" << json_escape(result.warnings[w]) << "\"";
        }
        out << "]\n";
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    finish_output(out, output_path, "summary JSON");
}

void write_chrome_trace(
    const std::string& output_path,
    const std::vector<ScenarioResult>& results) {
    ensure_parent_dir(output_path);
    std::map<std::string, int> tids;
    auto tid_for = [&tids](const std::string& entity) -> int {
        const auto key = entity.empty() ? std::string{"unknown"} : entity;
        const auto found = tids.find(key);
        if (found != tids.end()) {
            return found->second;
        }
        const int next = static_cast<int>(tids.size() + 1);
        tids[key] = next;
        return next;
    };

    for (const auto& result : results) {
        for (const auto& row : result.completions) {
            for (const auto& span : row.spans) {
                (void)tid_for(result.name + "/" + span.entity);
            }
        }
    }

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open Chrome trace output: " + output_path);
    }
    out << std::fixed << std::setprecision(6);
    out << "{\n  \"traceEvents\": [\n";
    bool first = true;
    auto comma = [&]() {
        if (!first) {
            out << ",\n";
        }
        first = false;
    };

    for (const auto& [entity, tid] : tids) {
        comma();
        out << "    {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << tid
            << ",\"args\":{\"name\":\"" << json_escape(entity) << "\"}}";
    }
    for (const auto& result : results) {
        for (const auto& row : result.completions) {
            for (const auto& span : row.spans) {
                comma();
                const auto entity = result.name + "/" + (span.entity.empty() ? row.resource_path : span.entity);
                out << "    {\"name\":\"" << json_escape(span.name)
                    << "\",\"cat\":\"" << json_escape(result.name + "/" + span.category)
                    << "\",\"ph\":\"X\",\"ts\":" << (span.start_ns / 1000.0)
                    << ",\"dur\":" << (span.duration_ns() / 1000.0)
                    << ",\"pid\":1,\"tid\":" << tid_for(entity)
                    << ",\"args\":{"
                    << "\"request\":\"" << json_escape(row.id) << "\","
                    << "\"request_start_ns\":" << row.start_ns << ","
                    << "\"request_finish_ns\":" << row.finish_ns << ","
                    << "\"tier\":\"" << hbfsim::physical::to_string(row.tier) << "\","
                    << "\"op\":\"" << hbfsim::physical::to_string(row.op) << "\","
                    << "\"entity\":\"" << json_escape(entity) << "\","
                    << "\"detail\":\"" << json_escape(span.detail) << "\""
                    << "}}";
            }
        }
    }
    out << "\n  ]\n}\n";
    finish_output(out, output_path, "Chrome trace");
}

TraceMode parse_trace_mode_value(const std::string& value) {
    if (value == "off") return TraceMode::Off;
    if (value == "summary") return TraceMode::Summary;
    if (value == "sampled") return TraceMode::Sampled;
    if (value == "full") return TraceMode::Full;
    throw std::runtime_error("unknown trace mode: " + value);
}

void usage(const char* argv0) {
    std::cerr << "usage:\n"
              << "  " << argv0 << " --version\n"
              << "  " << argv0 << " --config configs/systems/server-hbm128-hbf512.cfg\n"
              << "      --config configs/policies/reference/server-hbm128-hbf512.cfg --trace path\n"
              << "      [--scenarios comma-separated-names]\n"
              << "      [--summary-csv path.csv] [--summary-json path.json] [--config-out path.cfg]\n"
              << "      [--chrome-trace path.json] [--max-ops N]\n"
              << "      [--trace-census-only true] (validate/count spans without device expansion)\n"
              << "\n"
              << "Advanced overrides:\n"
              << "  " << argv0 << " [--config path] --trace path [--line-size N] [--interarrival-ns N]\n"
              << "      [--expected-trace-sha256 HEX --expected-trace-bytes N]\n"
              << "      [--initial-image-trace path] (population only; stabilizes prefix placement)\n"
              << "  " << argv0 << " [--config path] --synthetic-sequential-read-bytes N"
              << " [--synthetic-sequential-read-base N]\n"
              << "      (lazy fixed-size reads; N/base must be line-size aligned; select a direct logical scenario)\n"
              << "      (config files apply first in argv order; any CLI option overrides them)\n"
              << "      [--scenarios comma-separated-names] (omitted runs all eight)\n"
              << "        all-hbm, all-hbf, flat, direct-read, demand-fill, reuse-filtered,\n"
              << "        hbf-streaming, external-streaming\n"
              << "      [--hbm-capacity-bytes N] [--flat-hbm-bytes N]\n"
              << "      [--hbf-capacity-bytes N | --hbf-capacity-ratio N]\n"
              << "      [--layer-buffer-bytes N] (upper bound for one layer-streaming buffer)\n"
              << "      [--explicit-residency-contract true] with every residency-* field:\n"
              << "        unique-footprint, immutable-weight bytes/pages, runtime-overhead,\n"
              << "        block-table, active-buffer, KV-region/stride/logical-blocks/hot-blocks;\n"
              << "        requires the full digest-bound trace and rejects --max-ops\n"
              << "      [--behavioral-hbm-bytes N] (0 = complete configured HBM)\n"
              << "      [--behavioral-promotion-threshold N]"
                 " [--behavioral-history-pages N]\n"
              << "      [--max-outstanding-requests W] (per-tier page credits; 0 = open loop)\n"
              << "      [--max-hbm-outstanding-requests W] [--max-hbf-outstanding-requests W]\n"
              << "        (independent read-only direct-composition credit pools; mutually exclusive with shared W)\n"
              << "      [--address-heatmap-bins N] (0 = off, the default; 1..8192 bins per domain)\n"
              << "      [--hbf-hbm-write-buffer-bytes N] (0 = off; cooperative write staging)\n"
              << "      [--hbf-hbm-write-buffer-destage-watermark F] (0..1 of the region; default 0.5)\n"
              << "      [--<engine-key> VALUE] any hardware key of the core hbfsim engine:\n"
              << "        hbm-*, hbf-*, base-die-link-*, external-backing-* (see configs/README.md).\n"
              << "        The runner forwards them to the same SystemConfigBuilder the core\n"
              << "        `hbfsim` uses, so a config file or override means exactly the same\n"
              << "        thing in both programs; --config-out replays through that parser.\n"
              << "      [--chrome-trace path.json] [--trace-mode off|summary|sampled|full]\n"
              << "      [--summary-csv path.csv] [--summary-json path.json] [--config-out path.cfg]\n"
              << "  " << argv0 << " --generate-semantic-llm path [--llm-tokens N] [--llm-layers N]   (smoke-test trace, not a model workload; LLM serving: python3 -m hbserve run)\n"
              << "      [--llm-weight-base N] [--llm-kv-base N] [--llm-scratch-base N]\n"
              << "\n"
              << "Trace input is Ramulator-compatible text: <addr> <R|W>, one memory op per line.\n"
              << "Optional semantic form: <addr> <R|W> [bytes] "
                 "[model_weights|shared_context|generated_context|scratch|metadata] "
                 "[phase=N] [layer=N] [compute_ns=N].\n"
              << "Simulation-session protocol v1 accepts only explicit HBM/HBF/D2D/barrier "
                 "transactions with addresses, sizes, batch-relative issue times, and dependencies; "
                 "semantic trace fields are not part of that execution boundary.\n"
              << "Config files are key=value with the same names as long CLI options without '--'.\n"
              << "Prefer config files for stable HBM/HBF hardware, timing, flash geometry, and policy settings.\n"
              << "Keep run-local choices such as --trace, output paths, --chrome-trace, and --max-ops on the CLI.\n"
              << "Multiple --config files are applied in order; later CLI options override earlier config values.\n";
}

RunConfig parse_args(int argc, char** argv) {
    RunBuilder run;
    // Two passes: config files apply first (in argv order), then every other
    // CLI option (in argv order) — so a CLI option always overrides config
    // values regardless of where --config appears on the command line.
    std::vector<std::pair<std::string, std::string>> cli_options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--version") {
            std::cout << "HBFSim " << HBFSIM_VERSION
                      << " (git " << HBFSIM_GIT_COMMIT
                      << (HBFSIM_GIT_DIRTY ? ", dirty-at-build" : "")
                      << ", source-sha256 " << HBFSIM_SOURCE_SHA256
                      << ")\n";
            std::exit(0);
        }
        if (!arg.starts_with("--")) {
            throw std::runtime_error("unknown argument: " + arg);
        }

        const auto value = need_value(arg.c_str());
        if (arg == "--config") {
            load_config_file(run, value);
        } else {
            cli_options.emplace_back(arg, value);
        }
    }
    for (const auto& [key, value] : cli_options) {
        apply_key(run, key, value);
    }
    auto options = std::move(run.options);

    // Hardware resolves through the engine's builder: derived HBF geometry,
    // controller-DRAM budgets, external-backing profiles, and every physical
    // range check belong to the core simulator. Constructing each device once
    // here is that validation; it fails closed before any trace is read.
    SystemConfig system = run.system.resolve();
    (void)HbmDevice(system.hbm);
    const auto resolved_host = HbfController(system.hbf).config();
    (void)hbfsim::physical::BaseDieLink(system.base_die_link, "validate");
    if (system.hbf_external_direct_link) {
        (void)hbfsim::physical::BaseDieLink(
            *system.hbf_external_direct_link, "validate-direct-lane");
    }
    (void)hbfsim::physical::external::ExternalBackingDevice(system.external);
    if (options.hbm_capacity_explicit && !options.flat_hbm_bytes_explicit) {
        const auto burst = system.hbm.burst_bytes();
        const auto reserved = (resolved_host.host.ctrl_dram_bytes + burst - 1) / burst * burst;
        options.flat_hbm_bytes = system.hbm.device.capacity_bytes > reserved ?
            (system.hbm.device.capacity_bytes - reserved) / system.hbf.device.page_size_bytes *
                system.hbf.device.page_size_bytes : 0;
    }

    if (options.line_size == 0) {
        throw std::runtime_error("--line-size must be positive");
    }
    if (options.trace_path && options.synthetic_sequential_read_bytes) {
        throw std::runtime_error(
            "--trace and --synthetic-sequential-read-bytes are mutually exclusive");
    }
    if (options.expected_trace_sha256.has_value() !=
        options.expected_trace_file_bytes.has_value()) {
        throw std::runtime_error(
            "--expected-trace-sha256 and --expected-trace-bytes must be "
            "provided together");
    }
    if (options.expected_trace_sha256) {
        if (!options.trace_path) {
            throw std::runtime_error(
                "expected trace identity requires --trace");
        }
        if (!is_lower_hex_sha256(*options.expected_trace_sha256)) {
            throw std::runtime_error(
                "--expected-trace-sha256 must be 64 lowercase hex digits");
        }
    }
    if (options.initial_image_trace_path && !options.trace_path) {
        throw std::runtime_error(
            "--initial-image-trace requires an executed --trace workload");
    }
    if (options.synthetic_sequential_read_bytes) {
        if (*options.synthetic_sequential_read_bytes == 0 ||
            *options.synthetic_sequential_read_bytes % options.line_size != 0 ||
            options.synthetic_sequential_read_base % options.line_size != 0) {
            throw std::runtime_error(
                "synthetic sequential byte count/base must be positive and "
                "--line-size aligned");
        }
        if (options.max_ops != 0) {
            throw std::runtime_error(
                "--max-ops is not used with --synthetic-sequential-read-bytes; "
                "set the exact byte count instead");
        }
    } else if (options.synthetic_sequential_read_base != 0) {
        throw std::runtime_error(
            "--synthetic-sequential-read-base requires "
            "--synthetic-sequential-read-bytes");
    }
    if (options.synthetic_sequential_read_bytes &&
        options.generate_semantic_llm_path) {
        throw std::runtime_error(
            "synthetic sequential input cannot be combined with an LLM trace generator");
    }
    if (options.interarrival_ns < 0.0 || !std::isfinite(options.interarrival_ns)) {
        throw std::runtime_error("--interarrival-ns must be non-negative and finite");
    }
    if (options.address_heatmap_bins > hbfsim::physical::kMaxAddressHeatmapBins) {
        throw std::runtime_error(
            "--address-heatmap-bins must be 0 (disabled) or in [1, 8192]");
    }
    const bool any_layer_streaming =
        scenario_selected(options, kLayerStreamingScenario) ||
        scenario_selected(options, kExternalLayerStreamingScenario);
    if (any_layer_streaming) {
        if (options.layer_buffer_bytes == 0) {
            throw std::runtime_error("--layer-buffer-bytes must be positive");
        }
    }
    const std::array<bool, 12> residency_fields{{
        options.residency_page_size_bytes.has_value(),
        options.residency_unique_footprint_bytes.has_value(),
        options.residency_immutable_weight_bytes.has_value(),
        options.residency_immutable_weight_pages.has_value(),
        options.residency_static_weight_pages.has_value(),
        options.residency_runtime_overhead_bytes.has_value(),
        options.residency_block_table_bytes.has_value(),
        options.residency_active_buffer_bytes.has_value(),
        options.residency_kv_region_begin.has_value(),
        options.residency_kv_block_stride_bytes.has_value(),
        options.residency_logical_kv_blocks.has_value(),
        options.residency_hot_kv_blocks.has_value(),
    }};
    const auto residency_field_count = static_cast<std::size_t>(std::count(
        residency_fields.begin(), residency_fields.end(), true));
    if (options.explicit_residency_contract) {
        if (!any_layer_streaming) {
            throw std::runtime_error(
                "--explicit-residency-contract requires a layer-streaming "
                "scenario");
        }
        if (!options.expected_trace_sha256) {
            throw std::runtime_error(
                "--explicit-residency-contract requires a digest-bound trace");
        }
        if (options.max_ops != 0) {
            throw std::runtime_error(
                "--explicit-residency-contract rejects --max-ops truncation");
        }
        if (residency_field_count != residency_fields.size()) {
            throw std::runtime_error(
                "--explicit-residency-contract requires every residency-* "
                "field");
        }
    } else if (residency_field_count != 0) {
        throw std::runtime_error(
            "residency-* fields require --explicit-residency-contract=true");
    }
    if (options.behavioral_history_pages == 0) {
        throw std::runtime_error(
            "--behavioral-history-pages must be positive");
    }
    if (options.behavioral_promotion_threshold < 2) {
        throw std::runtime_error(
            "--behavioral-promotion-threshold must be at least two");
    }
    if (options.behavioral_hbm_bytes > system.hbm.device.capacity_bytes) {
        throw std::runtime_error(
            "--behavioral-hbm-bytes cannot exceed --hbm-capacity-bytes");
    }
    if (options.hbf_hbm_write_buffer_bytes > system.hbm.device.capacity_bytes) {
        throw std::runtime_error(
            "--hbf-hbm-write-buffer-bytes cannot exceed --hbm-capacity-bytes");
    }
    if (!std::isfinite(options.hbf_hbm_write_buffer_destage_watermark) ||
        options.hbf_hbm_write_buffer_destage_watermark < 0.0 ||
        options.hbf_hbm_write_buffer_destage_watermark > 1.0) {
        throw std::runtime_error(
            "--hbf-hbm-write-buffer-destage-watermark must be in [0, 1]");
    }
    if (options.flat_hbm_bytes > system.hbm.device.capacity_bytes) {
        throw std::runtime_error("--flat-hbm-bytes cannot exceed --hbm-capacity-bytes");
    }
    if (options.static_direct_hbm_bytes > system.hbm.device.capacity_bytes) {
        throw std::runtime_error(
            "--static-direct-hbm-bytes cannot exceed --hbm-capacity-bytes");
    }
    if (options.generate_semantic_llm_path &&
        (options.llm_tokens == 0 || options.llm_layers == 0)) {
        throw std::runtime_error("--llm-tokens and --llm-layers must be positive");
    }
    if (options.max_outstanding_requests != 0 &&
        (options.max_hbm_outstanding_requests != 0 ||
         options.max_hbf_outstanding_requests != 0)) {
        throw std::runtime_error(
            "--max-outstanding-requests cannot be combined with tier-specific "
            "outstanding windows");
    }
    return RunConfig{
        .options = std::move(options),
        .system = std::move(system),
    };
}

} // namespace

int hbfsim::app::run_reference(int argc, char** argv) {
    try {
        const auto run = parse_args(argc, argv);
        const auto& options = run.options;
        const auto& system = run.system;
        if (options.config_out_path) {
            write_resolved_config(*options.config_out_path, options, system);
            std::cout << "wrote resolved config: " << *options.config_out_path << '\n';
        }
        if (options.generate_semantic_llm_path) {
            generate_semantic_llm_trace(options);
            std::cout << "generated semantic LLM trace: "
                      << *options.generate_semantic_llm_path << '\n';
            if (!options.trace_path) {
                return 0;
            }
        }

        std::vector<TraceOp> ops;
        std::vector<TraceOp> initial_image_ops;
        std::optional<SequentialReadWorkload> sequential_workload;
        if (options.synthetic_sequential_read_bytes) {
            sequential_workload = make_synthetic_sequential_workload(options);
        } else {
            ops = load_ramulator_trace(options);
            if (options.initial_image_trace_path) {
                auto image_options = options;
                image_options.trace_path = options.initial_image_trace_path;
                image_options.max_ops = 0;
                initial_image_ops = load_ramulator_trace(image_options);
            }
        }
        RunProvenance provenance;
        provenance.command.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; ++i) {
            provenance.command.emplace_back(argv[i]);
        }
        provenance.working_directory = std::filesystem::current_path().string();
        std::error_code executable_error;
        auto executable_path = std::filesystem::absolute(argv[0], executable_error);
        if (!executable_error &&
            std::filesystem::is_regular_file(executable_path, executable_error) &&
            !executable_error) {
            executable_path = std::filesystem::weakly_canonical(
                executable_path,
                executable_error);
            if (!executable_error) {
                provenance.executable_path = executable_path.string();
                const auto executable_bytes =
                    std::filesystem::file_size(executable_path, executable_error);
                if (!executable_error &&
                    executable_bytes <= std::numeric_limits<std::uint64_t>::max()) {
                    provenance.executable_file_bytes =
                        static_cast<std::uint64_t>(executable_bytes);
                    provenance.executable_sha256 =
                        sha256_file(provenance.executable_path);
                }
            }
        }
        if (options.trace_path) {
            provenance.trace_sha256 = sha256_file(*options.trace_path);
            const auto trace_file_bytes =
                std::filesystem::file_size(*options.trace_path);
            if (trace_file_bytes > std::numeric_limits<std::uint64_t>::max()) {
                throw std::runtime_error("trace file size exceeds uint64_t range");
            }
            provenance.trace_file_bytes =
                static_cast<std::uint64_t>(trace_file_bytes);
            if (options.expected_trace_sha256 &&
                (provenance.trace_sha256 !=
                        *options.expected_trace_sha256 ||
                 provenance.trace_file_bytes !=
                        *options.expected_trace_file_bytes)) {
                throw std::runtime_error(
                    "executed trace identity differs from the digest-bound "
                    "residency config");
            }
        } else {
            provenance.trace_sha256 = sha256_text(
                synthetic_sequential_descriptor(options, *sequential_workload));
        }
        if (options.initial_image_trace_path) {
            provenance.initial_image_trace_sha256 =
                sha256_file(*options.initial_image_trace_path);
            const auto image_trace_file_bytes = std::filesystem::file_size(
                *options.initial_image_trace_path);
            if (image_trace_file_bytes >
                std::numeric_limits<std::uint64_t>::max()) {
                throw std::runtime_error(
                    "initial-image trace file size exceeds uint64_t range");
            }
            provenance.initial_image_trace_file_bytes =
                static_cast<std::uint64_t>(image_trace_file_bytes);
        }
        const auto trace_footprint_bytes = sequential_workload ?
            *options.synthetic_sequential_read_bytes :
            trace_unique_line_bytes(ops, options.line_size);
        const auto op_count = sequential_workload ?
            sequential_workload->request_count : ops.size();
        // The offered stream is a workload property shared by every scenario.
        std::uint64_t logical_bytes = 0;
        std::uint64_t read_bytes = 0;
        std::uint64_t write_bytes = 0;
        std::uint64_t reads = 0;
        std::uint64_t writes = 0;
        std::set<std::uint64_t> phases;
        if (sequential_workload) {
            logical_bytes = *options.synthetic_sequential_read_bytes;
            read_bytes = logical_bytes;
            reads = sequential_workload->request_count;
        } else {
            for (const auto& op : ops) {
                logical_bytes = checked_add_u64(
                    logical_bytes, op.bytes, "trace census logical bytes");
                if (op.op == Op::Read) {
                    ++reads;
                    read_bytes = checked_add_u64(
                        read_bytes, op.bytes, "trace census read bytes");
                } else {
                    ++writes;
                    write_bytes = checked_add_u64(
                        write_bytes, op.bytes, "trace census write bytes");
                }
                if (op.phase) {
                    phases.insert(*op.phase);
                }
            }
        }
        if (options.trace_census_only) {
            std::cout
                << "{\"schema\":{\"name\":\"hbfsim.trace_census\",\"version\":1},"
                << "\"operations\":" << op_count << ','
                << "\"reads\":" << reads << ','
                << "\"writes\":" << writes << ','
                << "\"logical_bytes\":" << logical_bytes << ','
                << "\"read_bytes\":" << read_bytes << ','
                << "\"write_bytes\":" << write_bytes << ','
                << "\"unique_line_bytes\":" << trace_footprint_bytes << ','
                << "\"line_size\":" << options.line_size << ','
                << "\"explicit_phases\":" << phases.size() << ','
                << "\"trace_sha256\":\""
                << json_escape(provenance.trace_sha256) << "\"}\n";
            return 0;
        }
        const auto requests = sequential_workload ?
            std::vector<MemoryRequest>{} : make_memory_requests(ops);
        const auto initial_image_requests =
            make_memory_requests(initial_image_ops);
        const auto* initial_image_request_ptr =
            initial_image_requests.empty() ?
            nullptr :
            &initial_image_requests;
        std::vector<ScenarioResult> results;
        // The scenario table IS the scenario set: name validation,
        // selection, and execution all walk this one list, in this order.
        // Adding a composition point is one entry here — never a new
        // branch per concern.
        struct ScenarioSpec {
            const char* name;
            std::function<ScenarioResult()> run;
        };
        const auto run_direct_source = [&] (
            const char* scenario_name,
            const DirectPolicy& policy) {
            if (sequential_workload) {
                return run_direct(
                    policy,
                    scenario_name,
                    *sequential_workload,
                    options,
                    system);
            }
            return run_direct(
                policy,
                scenario_name,
                requests,
                options,
                system,
                initial_image_request_ptr);
        };
        const std::vector<ScenarioSpec> scenario_table = {
            {kAllHbmScenario,
             [&] {
                 return run_direct_source(
                     kAllHbmScenario, all_hbm_policy());
             }},
            {kAllHbfScenario,
             [&] {
                 return run_direct_source(
                     kAllHbfScenario, all_hbf_policy());
             }},
            {kFlatScenario,
             [&] { return run_direct_source(
                       kFlatScenario,
                       flat_policy(options.flat_hbm_bytes)); }},
            {kStaticDirectScenario,
             [&] { return run_direct_source(
                       kStaticDirectScenario,
                       read_only_direct_policy(options.static_direct_hbm_bytes)); }},
            {kDemandFillScenario,
             [&] { return run_behavioral_tiering(
                       requests,
                       options,
                       system,
                       BehavioralAdmissionPolicy::AlwaysAdmit,
                       kDemandFillScenario,
                       initial_image_request_ptr); }},
            {kBehavioralTieringScenario,
             [&] { return run_behavioral_tiering(
                       requests,
                       options,
                       system,
                       BehavioralAdmissionPolicy::ReuseFiltered,
                       kBehavioralTieringScenario,
                       initial_image_request_ptr); }},
            {kLayerStreamingScenario,
             [&] { return run_layer_streaming(
                       requests, options, system, BackingTier::Hbf); }},
            {kExternalLayerStreamingScenario,
             [&] { return run_layer_streaming(
                       requests, options, system, BackingTier::External); }},
        };
        for (const auto& wanted : options.scenarios) {
            const bool known = std::any_of(
                scenario_table.begin(), scenario_table.end(),
                [&wanted](const ScenarioSpec& spec) { return wanted == spec.name; });
            if (!known) {
                throw std::runtime_error("unknown scenario in --scenarios: " + wanted);
            }
        }
        if ((options.max_hbm_outstanding_requests != 0 ||
             options.max_hbf_outstanding_requests != 0) &&
            scenario_selected(options, kLayerStreamingScenario)) {
            throw std::runtime_error(
                "tier-specific outstanding windows are defined only for "
                "read-only direct compositions; exclude " +
                std::string(kLayerStreamingScenario));
        }
        if ((options.max_hbm_outstanding_requests != 0 ||
             options.max_hbf_outstanding_requests != 0) &&
            scenario_selected(options, kExternalLayerStreamingScenario)) {
            throw std::runtime_error(
                "tier-specific outstanding windows are defined only for "
                "read-only direct compositions; exclude " +
                std::string(kExternalLayerStreamingScenario));
        }
        if ((options.max_hbm_outstanding_requests != 0 ||
             options.max_hbf_outstanding_requests != 0) &&
            (scenario_selected(options, kDemandFillScenario) ||
             scenario_selected(options, kBehavioralTieringScenario))) {
            throw std::runtime_error(
                "tier-specific outstanding windows are defined only for "
                "read-only direct compositions; exclude the behavioral "
                "tiering scenarios");
        }
        if (sequential_workload &&
            (scenario_selected(options, kLayerStreamingScenario) ||
             scenario_selected(options, kExternalLayerStreamingScenario) ||
             scenario_selected(options, kDemandFillScenario) ||
             scenario_selected(options, kBehavioralTieringScenario))) {
            throw std::runtime_error(
                "--synthetic-sequential-read-bytes currently supports "
                "direct scenarios only; exclude behavioral and "
                "layer-streaming scenarios");
        }
        for (const auto& spec : scenario_table) {
            if (scenario_selected(options, spec.name)) {
                results.push_back(spec.run());
            }
        }

        double first_arrival_ns = 0.0;
        double last_arrival_ns = 0.0;
        if (sequential_workload) {
            first_arrival_ns = sequential_workload->first_arrival_ns;
            last_arrival_ns = sequential_workload->first_arrival_ns +
                static_cast<double>(sequential_workload->request_count - 1) *
                    sequential_workload->interarrival_ns;
        } else {
            const auto [first_arrival, last_arrival] = std::minmax_element(
                ops.begin(), ops.end(), [](const TraceOp& lhs, const TraceOp& rhs) {
                    return lhs.arrival_ns < rhs.arrival_ns;
                });
            first_arrival_ns = first_arrival->arrival_ns;
            last_arrival_ns = last_arrival->arrival_ns;
        }
        for (auto& result : results) {
            if (!result.owns_arrival_frontier) {
                result.first_arrival_ns = first_arrival_ns;
                result.last_arrival_ns = last_arrival_ns;
            }
            result.logical_read_bytes = read_bytes;
            result.logical_write_bytes = write_bytes;
            attach_offered_load(result, options, system);
        }

        const auto sanity = run_sanity_checks(
            results, options, system, trace_footprint_bytes);
        for (auto& result : results) {
            finalize_latency_distributions(
                result,
                sequential_workload.has_value());
        }
        print_run_config(options, system, op_count, trace_footprint_bytes);
        print_result_table(results);
        for (const auto& result : results) {
            print_device_stats(result);
        }
        std::cout << "\nSANITY: " << to_string(sanity.verdict) << '\n';
        for (const auto& notice : sanity.notices) {
            std::cerr << "warning: " << notice << '\n';
        }

        const auto wear_root = options.summary_json_path ?
            std::filesystem::path(*options.summary_json_path).parent_path() /
                (std::filesystem::path(*options.summary_json_path).stem().string() + "-hbf-wear") :
            hbfsim::app::default_hbf_wear_prefix().parent_path();
        for (const auto& result : results) {
            if (result.hbf_wear_snapshot.empty()) continue;
            const auto prefix = std::filesystem::absolute(wear_root / result.name);
            hbfsim::app::write_hbf_wear_report(prefix, result.hbf_wear_snapshot);
            std::cout << "wrote HBF wear map: " << prefix.string() << ".html\n";
        }
        if (options.summary_csv_path) {
            write_summary_csv(*options.summary_csv_path, results);
            std::cout << "wrote summary CSV: " << *options.summary_csv_path << '\n';
        }
        if (options.summary_json_path) {
            write_summary_json(
                *options.summary_json_path,
                options,
                system,
                provenance,
                op_count,
                trace_footprint_bytes,
                results,
                sanity.verdict);
            std::cout << "wrote summary JSON: " << *options.summary_json_path << '\n';
        }
        if (options.chrome_trace_path) {
            write_chrome_trace(*options.chrome_trace_path, results);
            std::cout << "wrote Chrome trace: " << *options.chrome_trace_path << '\n';
        }
        return sanity.verdict == SanityVerdict::Fail ? 1 : 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
