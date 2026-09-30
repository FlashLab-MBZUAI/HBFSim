#include "host/hbf_controller.hpp"
#include "physical/hbf/hbf_device.hpp"
#include "physical/hbm/hbm_device.hpp"
#include "policies/policy_common.hpp"
#include "policies/reference/direct_policy.hpp"
#include "policies/reference/layer_streaming_policy.hpp"
#include "hbf_with_hbm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using hbfsim::physical::Op;
using hbfsim::physical::AddressSpace;
using hbfsim::physical::Breakdown;
using hbfsim::physical::PhysicalCompletion;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;
using hbfsim::physical::TraceMode;
using hbfsim::physical::TraceSpan;
using hbfsim::physical::fixed;
using hbfsim::physical::print_breakdown;
using hbfsim::physical::print_completion_header;
using hbfsim::physical::print_completion_row;
using hbfsim::physical::print_trace_waterfall;
using hbfsim::host::HbfAddress;
using hbfsim::host::HbfConfig;
using hbfsim::host::HbfController;
using hbfsim::host::HbfStats;
using hbfsim::physical::hbm::HbmAddress;
using hbfsim::physical::hbm::HbmConfig;
using hbfsim::physical::hbm::HbmDevice;
using hbfsim::physical::hbm::HbmStats;
using hbfsim::policy::PolicyRunResult;
using hbfsim::policy::DirectPolicy;
using hbfsim::policy::DirectRunKnobs;
using hbfsim::policy::LayerStreamingPolicy;
using hbfsim::policy::LayerStreamingConfig;
using hbfsim::policy::MemoryRequest;
using hbfsim::policy::SemanticKind;
using hbfsim::policy::all_hbf_policy;
using hbfsim::policy::all_hbm_policy;
using hbfsim::policy::flat_policy;
using hbfsim::policy::read_only_direct_policy;
using hbfsim::policy::run_direct_policy;
using hbfsim::policy::collect_initial_read_lpns;
using hbfsim::policy::hbf_page_capacity;
using hbfsim::policy::map_static_hbf_page_addr;
using hbfsim::policy::unmap_static_hbf_page_addr;

namespace {

struct ProbeOptions {
    std::string probe = "all";
    bool waterfall = false;
    std::size_t waterfall_limit = 120;
    std::optional<std::string> trace_path;
    std::optional<std::string> folded_path;
    TraceMode trace_mode = TraceMode::Off;
    std::uint64_t trace_sample_period = 1;
};

struct HbmStreamResult {
    double finish_ns = 0.0;
    HbmStats stats;
};

struct HbfParallelResult {
    double finish_ns = 0.0;
    HbfStats stats;
};

ProbeOptions g_options;
std::vector<PhysicalCompletion> g_captured_rows;
std::uint64_t g_request_ordinal = 0;

bool trace_output_requested() {
    return g_options.waterfall || g_options.trace_path || g_options.folded_path;
}

TraceMode effective_trace_mode() {
    if (g_options.trace_mode != TraceMode::Off) {
        return g_options.trace_mode;
    }
    return trace_output_requested() ? TraceMode::Full : TraceMode::Off;
}

TraceMode next_request_trace_mode() {
    const auto mode = effective_trace_mode();
    if (mode != TraceMode::Sampled) {
        return mode;
    }
    const auto period = std::max<std::uint64_t>(1, g_options.trace_sample_period);
    const bool selected = (g_request_ordinal % period) == 0;
    ++g_request_ordinal;
    return selected ? TraceMode::Sampled : TraceMode::Off;
}

std::string optional_fixed(std::optional<double> value, int precision = 3) {
    return value ? fixed(*value, precision) : "n/a";
}

PhysicalRequest request(
    std::string id,
    Tier tier,
    Op op,
    double arrival_ns,
    std::uint64_t addr,
    std::uint64_t bytes,
    AddressSpace address_space = AddressSpace::Logical) {
    return PhysicalRequest{
        .id = std::move(id),
        .tier = tier,
        .op = op,
        .address_space = address_space,
        .trace = {.mode = next_request_trace_mode()},
        .arrival_ns = arrival_ns,
        .addr = addr,
        .bytes = bytes,
        .stream_id = 0,
    };
}

void print_rows(const std::vector<PhysicalCompletion>& rows) {
    g_captured_rows.insert(g_captured_rows.end(), rows.begin(), rows.end());
    print_completion_header(std::cout);
    for (const auto& row : rows) {
        print_completion_row(std::cout, row);
    }
    for (const auto& row : rows) {
        print_breakdown(std::cout, row);
    }
    if (g_options.waterfall) {
        for (const auto& row : rows) {
            print_trace_waterfall(std::cout, row, g_options.waterfall_limit);
        }
    }
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
            out << ch;
            break;
        }
    }
    return out.str();
}

void ensure_parent_dir(const std::string& output_path) {
    const std::filesystem::path path(output_path);
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void write_chrome_trace(
    const std::string& output_path,
    const std::vector<PhysicalCompletion>& rows) {
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

    for (const auto& row : rows) {
        for (const auto& span : row.spans) {
            (void)tid_for(span.entity);
        }
    }

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open trace output: " + output_path);
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

    for (const auto& row : rows) {
        for (const auto& span : row.spans) {
            comma();
            const auto entity = span.entity.empty() ? row.resource_path : span.entity;
            out << "    {\"name\":\"" << json_escape(span.name)
                << "\",\"cat\":\"" << json_escape(span.category)
                << "\",\"ph\":\"X\",\"ts\":" << (span.start_ns / 1000.0)
                << ",\"dur\":" << (span.duration_ns() / 1000.0)
                << ",\"pid\":1,\"tid\":" << tid_for(entity)
                << ",\"args\":{"
                << "\"request\":\"" << json_escape(row.id) << "\","
                << "\"tier\":\"" << hbfsim::physical::to_string(row.tier) << "\","
                << "\"op\":\"" << hbfsim::physical::to_string(row.op) << "\","
                << "\"entity\":\"" << json_escape(entity) << "\","
                << "\"detail\":\"" << json_escape(span.detail) << "\","
                << "\"critical\":" << (span.critical ? "true" : "false") << ","
                << "\"start_ns\":" << span.start_ns << ","
                << "\"end_ns\":" << span.end_ns
                << "}}";
        }
    }

    out << "\n  ]\n}\n";
}

void write_folded_trace(
    const std::string& output_path,
    const std::vector<PhysicalCompletion>& rows) {
    ensure_parent_dir(output_path);
    std::map<std::string, double> totals;
    for (const auto& row : rows) {
        for (const auto& span : row.spans) {
            const auto key = hbfsim::physical::to_string(row.tier) + ";" +
                span.category + ";" + span.name;
            totals[key] += span.duration_ns();
        }
    }

    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open folded output: " + output_path);
    }
    for (const auto& [key, total_ns] : totals) {
        // Folded flamegraph weights are integer picoseconds so sub-ns transfers do not vanish.
        out << key << ' ' << static_cast<long long>(std::llround(total_ns * 1000.0)) << '\n';
    }
}

void print_hbm_stats(const HbmDevice& hbm) {
    const auto& s = hbm.stats();
    std::cout << "  HBM stats: read_bytes=" << s.read_bytes
              << " write_bytes=" << s.write_bytes
              << " bus_busy_ns=" << fixed(s.bus_busy_ns)
              << " utilization=" << fixed(s.utilization() * 100.0) << "%"
              << " bus_parallelism=" << fixed(s.bus_parallelism(), 2)
              << " active_pch=" << s.active_pseudo_channels << "/" << s.pseudo_channels
              << " pch_busy_max_ns=" << fixed(s.max_pseudo_channel_busy_ns)
              << " pch_busy_avg_ns=" << fixed(s.avg_active_pseudo_channel_busy_ns)
              << " pch_busy_skew=" << fixed(s.pseudo_channel_busy_skew(), 2)
              << " finish_ns=" << fixed(s.finish_ns)
              << "\n";
}

[[maybe_unused]] void print_hbf_stats(const HbfController& hbf) {
    const auto& s = hbf.stats();
    std::cout << "  HBF stats: logical_read=" << s.logical_read_bytes
              << " physical_read=" << s.physical_read_bytes
              << " logical_write=" << s.logical_write_bytes
              << " physical_write=" << s.physical_write_bytes
              << " page_reads=" << s.page_reads
              << " page_programs=" << s.page_programs
              << " data_programs=" << s.data_programs
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
              << " invalidations=" << s.invalidations
              << " resident_mapping_bytes="
              << s.resident_mapping_table_bytes
              << " resident_mapping_bytes_per_stack="
              << s.resident_mapping_table_bytes_per_stack
              << " resident_mapping_pages_per_stack="
              << s.resident_mapping_pages_per_stack
              << " mapping_lookups=" << s.mapping_lookup_ops
              << " mapping_user_lookups=" << s.mapping_user_lookup_ops
              << " mapping_gc_lookups=" << s.mapping_gc_lookup_ops
              << " mapping_updates=" << s.mapping_update_ops
              << " mapping_user_updates=" << s.mapping_user_update_ops
              << " mapping_gc_updates=" << s.mapping_gc_update_ops
              << " mapping_dram_waits=" << s.mapping_dram_wait_ops
              << " mapping_dram_wait_work_ns=" << s.mapping_dram_wait_ns
              << " mapping_dram_wait_max_ns="
              << s.mapping_dram_wait_max_ns
              << " mapping_dram_issue_busy_ns="
              << s.mapping_dram_issue_busy_ns
              << " mapping_programs=" << s.mapping_page_programs
              << " write_buffer_hits=" << s.write_buffer_hits
              << " write_buffer_misses=" << s.write_buffer_misses
              << " write_buffer_flushes=" << s.write_buffer_flushes
              << " write_buffer_read_hits=" << s.write_buffer_read_hits
              << " write_buffer_read_bytes=" << s.write_buffer_read_bytes
              << " read_splits=" << s.read_splits
              << " read_split_pages=" << s.read_split_pages
              << " page_read_admission_events="
              << s.page_read_admission_events
              << " page_read_admission_waited="
              << s.page_read_admission_waited_pages
              << " page_read_admission_wait_ns="
              << fixed(s.page_read_admission_wait_ns)
              << " flash_sched_enq=" << s.flash_scheduler_enqueues
              << " flash_sched_issue=" << s.flash_scheduler_issues
              << " gc_runs=" << s.gc_runs
              << " gc_relocations=" << s.gc_relocations
              << " gc_data_relocations=" << s.gc_data_relocations
              << " gc_mapping_relocations=" << s.gc_mapping_relocations
              << " gc_reclaimed_invalid_pages="
              << s.gc_reclaimed_invalid_pages
              << " gc_user_blocked=" << s.gc_user_blocked_runs
              << " waf=" << optional_fixed(s.waf())
              << " array_read_ns=" << fixed(s.stage_work.array_read_ns)
              << " array_program_ns=" << fixed(s.stage_work.array_program_ns)
              << " array_erase_ns=" << fixed(s.stage_work.array_erase_ns)
              << " media_busy_ns=" << fixed(s.media_busy_ns)
              << " channel_busy_ns="
              << fixed(s.channel_command_busy_ns + s.channel_data_busy_ns)
              << " hb_io_busy_ns="
              << fixed(s.hb_io_command_busy_ns + s.hb_io_data_busy_ns)
              << " media_util=" << fixed(s.media_utilization() * 100.0) << "%"
              << " io_util=" << fixed(s.io_utilization() * 100.0) << "%"
              << " hbio_cmd_util="
              << fixed(s.hbio_command_utilization() * 100.0) << "%"
              << " hbio_data_util="
              << fixed(s.hbio_data_utilization() * 100.0) << "%"
              << " media_parallelism=" << fixed(s.media_parallelism(), 2)
              << " read_lane_parallelism=" << fixed(s.read_lane_parallelism(), 2)
              << " subarray_read_parallelism=" << fixed(s.subarray_read_parallelism(), 2)
              << " page_buffer_bank_parallelism="
              << fixed(s.page_buffer_bank_parallelism(), 2)
              << " channel_parallelism=" << fixed(s.channel_parallelism(), 2)
              << " hbio_parallelism=" << fixed(s.hbio_parallelism(), 2)
              << " sequencer_parallelism=" << fixed(s.sequencer_parallelism(), 2)
              << " ecc_decode_ops=" << s.ecc_decode_ops
              << " ecc_encode_ops=" << s.ecc_encode_ops
              << " ecc_codeword_bytes=" << s.ecc_codeword_bytes
              << " ecc_queue_wait_ns=" << fixed(s.stage_work.ecc_queue_wait_ns)
              << " ecc_issue_busy_ns=" << fixed(s.ecc_issue_busy_ns)
              << " ecc_issue_util=" << fixed(s.ecc_issue_utilization() * 100.0) << "%"
              << " ecc_issue_parallelism=" << fixed(s.ecc_issue_parallelism(), 2)
              << " active_ecc_dies=" << s.active_ecc_dies << "/" << s.dies
              << " ecc_max_inflight_per_die=" << s.max_ecc_inflight_per_die
              << " active_planes=" << s.active_planes << "/" << s.planes
              << " active_media_lanes=" << s.active_media_lanes << "/" << s.media_lanes
              << " active_subarrays=" << s.active_subarrays << "/" << s.subarrays
              << " active_page_buffer_banks=" << s.active_page_buffer_banks
              << "/" << s.page_buffer_banks
              << " plane_media_max_ns=" << fixed(s.max_plane_media_busy_ns)
              << " plane_media_avg_ns=" << fixed(s.avg_active_plane_media_busy_ns)
              << " plane_media_skew=" << fixed(s.plane_media_skew(), 2)
              << " plane_op_skew=" << fixed(s.plane_op_skew(), 2)
              << " media_lane_skew=" << fixed(s.media_lane_skew(), 2)
              << " media_lane_read_skew=" << fixed(s.media_lane_read_skew(), 2)
              << " subarray_busy_skew=" << fixed(s.subarray_busy_skew(), 2)
              << " subarray_read_skew=" << fixed(s.subarray_read_skew(), 2)
              << " page_buffer_bank_skew=" << fixed(s.page_buffer_bank_skew(), 2)
              << " page_buffer_bank_read_skew="
              << fixed(s.page_buffer_bank_read_skew(), 2)
              << " active_channels=" << s.active_channels << "/" << s.channels
              << " channel_busy_skew=" << fixed(s.channel_busy_skew(), 2)
              << " active_dies=" << s.active_dies << "/" << s.dies
              << " die_transaction_skew=" << fixed(s.die_transaction_skew(), 2)
              << " finish_ns=" << fixed(s.finish_ns)
              << "\n";
}

#include "hbf_contracts.hpp"

void probe_hbm_interface() {
    HbmConfig config;
    for (auto rate : {6.4, 8.0, 9.6}) {
        config.device.pin_rate_Gbps = rate;
        HbmDevice device(config);
        const auto completion = device.issue(request("interface", Tier::HBM, Op::Read, 0, 0, 1<<20));
        const auto physical = completion.physical_bytes;
        const auto expected_work = physical / config.pseudo_channel_bandwidth_GBps();
        if (std::abs(device.stats().bus_busy_ns - expected_work) > 1e-6)
            throw std::runtime_error("HBM physical bytes do not conserve raw interface work");
        if (device.stats().service_busy_ns < device.stats().bus_busy_ns)
            throw std::runtime_error("HBM effective service exceeded raw interface bandwidth");
        print_hbm_stats(device);
    }
}

void probe_hbm_boundaries() {
    HbmConfig config;
    HbmDevice device(config);
    const auto burst = config.burst_bytes();
    auto completion = device.issue(request("boundary", Tier::HBM, Op::Read, 0, burst-1, burst+2));
    if (completion.logical_bytes != burst+2 || completion.physical_bytes != burst*3)
        throw std::runtime_error("HBM unaligned access lost a boundary burst");
    for (std::uint64_t addr=0; addr<65536; ++addr)
        if (device.encode(device.decode(addr)) != addr)
            throw std::runtime_error("HBM address map failed round-trip");
    print_hbm_stats(device);
}

void probe_hbm_channels() {
    HbmConfig config;
    config.device.channels_per_stack = 1;
    config.device.pseudo_channels_per_channel = 1;
    config.timing.read_latency_ns = 0;
    config.timing.bandwidth_efficiency = 1;
    HbmDevice one(config);
    auto serial = one.issue(request("one", Tier::HBM, Op::Read, 0, 0, 1<<20));
    config.device.channels_per_stack = 4;
    HbmDevice four(config);
    auto parallel = four.issue(request("four", Tier::HBM, Op::Read, 0, 0, 1<<20));
    if (serial.finish_ns != parallel.finish_ns*4 || serial.physical_bytes != parallel.physical_bytes)
        throw std::runtime_error("HBM channel scaling lost bandwidth or traffic");
    print_hbm_stats(four);
}

void probe_hbm_turnaround() {
    HbmConfig config;
    config.device.channels_per_stack = config.device.pseudo_channels_per_channel = 1;
    config.timing.read_latency_ns = config.timing.write_latency_ns = 0;
    config.timing.bandwidth_efficiency = 1;
    HbmDevice device(config);
    auto write = device.issue(request("w", Tier::HBM, Op::Write, 0, 0, 4096));
    auto read = device.issue(request("r", Tier::HBM, Op::Read, 0, 4096, 4096));
    const auto turn = config.command_aligned_time_ns(config.timing.write_to_read_ns);
    if (read.finish_ns != 2*write.finish_ns+turn)
        throw std::runtime_error("HBM aggregate direction-change cost is missing");
    print_hbm_stats(device);
}

void probe_composition_kind_blind() {
    std::cout << "\n== composition kind-blind routing probe ==\n";
    HbmConfig hbm_cfg;
    hbm_cfg.device.stacks = 1;
    hbm_cfg.device.channels_per_stack = 2;
    hbm_cfg.device.pseudo_channels_per_channel = 1;
    HbfConfig hbf_cfg;
    hbf_cfg.device.stacks = 1;
    hbf_cfg.device.channels_per_stack = 1;
    hbf_cfg.device.dies_per_channel = 1;
    hbf_cfg.device.planes_per_die = 2;
    hbf_cfg.device.blocks_per_plane = 64;
    hbf_cfg.device.pages_per_block = 64;
    hbf_cfg.device.page_size_bytes = 4096;

    constexpr std::uint64_t kBoundary = 16384;
    struct OpSpec {
        Op op;
        std::uint64_t addr;
        std::uint64_t bytes;
        SemanticKind kind;
    };
    std::vector<OpSpec> specs;
    for (std::uint64_t k = 0; k < 8; ++k) {
        specs.push_back({Op::Read, 512 * k, 512, SemanticKind::Metadata});
        specs.push_back({Op::Read, kBoundary + 4096 * k, 2048, SemanticKind::ModelWeights});
        specs.push_back({Op::Read, kBoundary + 65536 + 512 * k, 512, SemanticKind::SharedContext});
        specs.push_back({Op::Write, kBoundary + 131072 + 512 * k, 512, SemanticKind::GeneratedContext});
        specs.push_back({Op::Write, 8192 + 64 * k, 64, SemanticKind::Scratch});
    }
    // The single case a kind MAY move: a metadata read above the boundary.
    // Read-only-direct keeps it on HBM when labeled; a plain trace serves it
    // straight from the fabric.
    specs.push_back({Op::Read, kBoundary + 262144, 512, SemanticKind::Metadata});

    const auto build = [&specs](bool labeled) {
        std::vector<MemoryRequest> requests;
        requests.reserve(specs.size());
        for (std::size_t i = 0; i < specs.size(); ++i) {
            requests.push_back(MemoryRequest{
                .id = "op" + std::to_string(i),
                .op = specs[i].op,
                .addr = specs[i].addr,
                .bytes = specs[i].bytes,
                .arrival_ns = static_cast<double>(i),
                .index = i,
                .kind = labeled ? specs[i].kind : SemanticKind::Unknown,
                .label = {},
            });
        }
        return requests;
    };
    const auto plain = build(false);
    const auto labeled = build(true);
    const DirectRunKnobs knobs{};
    const auto signature = [](const PolicyRunResult& r) {
        return std::tuple(
            r.hbm_user_accesses, r.hbf_user_accesses, r.hbf_static_read_bytes,
            r.reads, r.writes, r.logical_bytes);
    };

    struct Preset {
        const char* name;
        DirectPolicy policy;
    };
    const std::vector<Preset> blind_presets{
        {"all-hbm", all_hbm_policy()},
        {"all-hbf", all_hbf_policy()},
        {"FLAT", flat_policy(kBoundary)},
    };
    for (const auto& preset : blind_presets) {
        const auto plain_run =
            run_direct_policy(preset.policy, hbm_cfg, hbf_cfg, plain, knobs);
        const auto labeled_run =
            run_direct_policy(preset.policy, hbm_cfg, hbf_cfg, labeled, knobs);
        std::cout << "  " << preset.name << ": hbm/hbf user accesses plain = "
                  << plain_run.hbm_user_accesses << "/" << plain_run.hbf_user_accesses
                  << ", labeled = " << labeled_run.hbm_user_accesses << "/"
                  << labeled_run.hbf_user_accesses << "\n";
        if (signature(plain_run) != signature(labeled_run)) {
            throw std::runtime_error(
                std::string("composition-kind-blind: ") + preset.name +
                " routed the same trace differently once labels were attached");
        }
    }
    std::cout << "  kind_blind_ok=yes\n";

    const auto direct_plain = run_direct_policy(
        read_only_direct_policy(kBoundary), hbm_cfg, hbf_cfg, plain, knobs);
    const auto direct_labeled = run_direct_policy(
        read_only_direct_policy(kBoundary), hbm_cfg, hbf_cfg, labeled, knobs);
    std::cout << "  read-only-direct: hbm/hbf user accesses plain = "
              << direct_plain.hbm_user_accesses << "/" << direct_plain.hbf_user_accesses
              << ", labeled = " << direct_labeled.hbm_user_accesses << "/"
              << direct_labeled.hbf_user_accesses << "\n";
    if (direct_labeled.hbm_user_accesses != direct_plain.hbm_user_accesses + 1 ||
        direct_labeled.hbf_user_accesses + 1 != direct_plain.hbf_user_accesses ||
        direct_labeled.ops != direct_plain.ops ||
        direct_labeled.logical_bytes != direct_plain.logical_bytes) {
        throw std::runtime_error(
            "composition-kind-blind: read-only-direct labels must only refine "
            "reads toward HBM (exactly the one above-boundary metadata read)");
    }
    std::cout << "  kind_refines_direct_reads=yes\n";
    std::cout << "  expectation: plain address-only traces are first-class — "
              << "all-hbm/all-hbf/FLAT route identically with or without "
              << "kinds; read-only-direct uses kinds only to keep non-streaming reads "
              << "on HBM.\n";
}

void probe_composition_static_mapping() {
    std::cout << "\n== composition static-mapping bijection probe ==\n";
    HbfConfig cfg;
    // The deleted hash/modulo mapper collided specifically for source pages
    // 4 and 5 in this 8x8 single-plane geometry. Exhaustively checking the
    // full domain makes that regression deterministic rather than probable.
    cfg.device.stacks = 1;
    cfg.device.channels_per_stack = 1;
    cfg.device.dies_per_channel = 1;
    cfg.device.planes_per_die = 1;
    cfg.device.blocks_per_plane = 8;
    cfg.device.pages_per_block = 8;
    cfg.device.page_size_bytes = 4096;
    cfg.device.media_lanes_per_plane = 8;

    HbfController hbf(cfg);

    const auto capacity = hbf_page_capacity(hbf);
    std::set<std::uint64_t> physical_pages;
    for (std::uint64_t source_page = 0; source_page < capacity; ++source_page) {
        const auto physical_addr = map_static_hbf_page_addr(hbf, source_page);
        if (!physical_pages.insert(physical_addr / cfg.device.page_size_bytes).second) {
            throw std::runtime_error(
                "composition-static-mapping: two source pages alias one physical page");
        }
        if (unmap_static_hbf_page_addr(hbf, physical_addr) != source_page) {
            throw std::runtime_error(
                "composition-static-mapping: inverse did not recover source page");
        }
    }
    if (physical_pages.size() != capacity) {
        throw std::runtime_error(
            "composition-static-mapping: permutation did not cover full capacity");
    }

    bool source_bound_rejected = false;
    try {
        (void)map_static_hbf_page_addr(hbf, capacity);
    } catch (const std::runtime_error&) {
        source_bound_rejected = true;
    }
    bool physical_bound_rejected = false;
    try {
        (void)unmap_static_hbf_page_addr(hbf, capacity * cfg.device.page_size_bytes);
    } catch (const std::runtime_error&) {
        physical_bound_rejected = true;
    }
    if (!source_bound_rejected || !physical_bound_rejected) {
        throw std::runtime_error(
            "composition-static-mapping: out-of-capacity pages must fail closed");
    }
    std::cout << "  pages=" << capacity
              << " unique=" << physical_pages.size() << "\n";
    std::cout << "  static_mapping_bijection_ok=yes\n";
    std::cout << "  static_mapping_bounds_ok=yes\n";

    // A block-major tensor layout can advance by a power-of-two number of
    // former mapping-page spans. The stack-local mapping-group rotation must
    // distribute that exact adversarial stride while consecutive pages remain
    // page-striped across all stacks.
    auto hash_cfg = cfg;
    hash_cfg.device.stacks = 8;
    hash_cfg.host.mapping_entries_per_page = 512;
    HbfController hash_device(hash_cfg);
    std::array<std::uint64_t, 8> stack_counts{};
    for (std::uint64_t index = 0; index < 4096; ++index) {
        const auto mapping_vpn = index * hash_cfg.device.stacks;
        const auto lpn =
            mapping_vpn * hash_cfg.host.mapping_entries_per_page;
        ++stack_counts.at(hash_device.stack_for_logical_page(lpn));
    }
    const auto [minimum, maximum] = std::minmax_element(
        stack_counts.begin(), stack_counts.end());
    if (*minimum == 0 ||
        static_cast<double>(*maximum) / static_cast<double>(*minimum) > 1.25) {
        throw std::runtime_error(
            "composition-static-mapping: strided logical pages alias HBF stacks");
    }
    for (std::uint64_t group = 0; group < 32; ++group) {
        std::array<bool, 8> seen{};
        const auto first_lpn =
            group * hash_cfg.host.mapping_entries_per_page * hash_cfg.device.stacks;
        for (std::uint64_t lane = 0; lane < hash_cfg.device.stacks; ++lane) {
            seen.at(hash_device.stack_for_logical_page(first_lpn + lane)) = true;
        }
        if (!std::all_of(seen.begin(), seen.end(), [](bool value) {
                return value;
            })) {
            throw std::runtime_error(
                "composition-static-mapping: consecutive stripe did not cover "
                "every HBF stack exactly once");
        }
    }
    std::cout << "  strided_mapping_stack_rotation_ok=yes\n";
    std::cout << "  consecutive_page_striping_ok=yes\n";
}

void probe_composition_initial_image() {
    std::cout << "\n== composition initial-image probe ==\n";
    constexpr std::uint64_t kPageSize = 4096;
    const std::vector<MemoryRequest> inference_trace{
        MemoryRequest{.id = "w0", .op = Op::Write, .addr = 0,
            .bytes = kPageSize, .arrival_ns = 0.0, .index = 0},
        MemoryRequest{.id = "r0", .op = Op::Read, .addr = 0,
            .bytes = kPageSize, .arrival_ns = 1.0, .index = 1},
        MemoryRequest{.id = "r1", .op = Op::Read, .addr = kPageSize,
            .bytes = kPageSize, .arrival_ns = 2.0, .index = 2},
        MemoryRequest{.id = "w2", .op = Op::Write, .addr = 2 * kPageSize,
            .bytes = kPageSize, .arrival_ns = 3.0, .index = 3},
        MemoryRequest{.id = "w3-partial", .op = Op::Write,
            .addr = 3 * kPageSize, .bytes = kPageSize / 2,
            .arrival_ns = 4.0, .index = 4},
        MemoryRequest{.id = "r3-untouched", .op = Op::Read,
            .addr = 3 * kPageSize + kPageSize / 2, .bytes = kPageSize / 2,
            .arrival_ns = 5.0, .index = 5},
        MemoryRequest{.id = "w4-partial", .op = Op::Write,
            .addr = 4 * kPageSize, .bytes = kPageSize / 2,
            .arrival_ns = 6.0, .index = 6},
        MemoryRequest{.id = "r4-covered", .op = Op::Read,
            .addr = 4 * kPageSize, .bytes = kPageSize / 2,
            .arrival_ns = 7.0, .index = 7},
    };
    const auto inferred = collect_initial_read_lpns(inference_trace, kPageSize);
    if (inferred != std::vector<std::uint64_t>{1, 3}) {
        throw std::runtime_error(
            "composition-initial-image: byte-range inference lost or invented old data");
    }

    HbmConfig hbm_cfg;
    hbm_cfg.device.stacks = 1;
    hbm_cfg.device.channels_per_stack = 1;
    hbm_cfg.device.pseudo_channels_per_channel = 1;
    HbfConfig hbf_cfg;
    hbf_cfg.device.stacks = 1;
    hbf_cfg.device.channels_per_stack = 1;
    hbf_cfg.device.dies_per_channel = 1;
    hbf_cfg.device.planes_per_die = 2;
    hbf_cfg.device.blocks_per_plane = 16;
    hbf_cfg.device.pages_per_block = 16;
    hbf_cfg.device.page_size_bytes = kPageSize;
    hbf_cfg.host.mapping_entries_per_page = 16;

    const auto make_trace = [](bool read_first) {
        std::vector<MemoryRequest> requests;
        requests.push_back(MemoryRequest{
            .id = read_first ? "read-first" : "write-first",
            .op = read_first ? Op::Read : Op::Write,
            .addr = 0,
            .bytes = kPageSize,
            .arrival_ns = 0.0,
            .index = 0,
        });
        requests.push_back(MemoryRequest{
            .id = read_first ? "write-second" : "read-second",
            .op = read_first ? Op::Write : Op::Read,
            .addr = 0,
            .bytes = kPageSize,
            .arrival_ns = 1.0,
            .index = 1,
        });
        return requests;
    };

    const DirectRunKnobs knobs{};
    const auto direct_write_first = run_direct_policy(
        all_hbf_policy(), hbm_cfg, hbf_cfg, make_trace(false), knobs);
    const auto direct_read_first = run_direct_policy(
        all_hbf_policy(), hbm_cfg, hbf_cfg, make_trace(true), knobs);
    std::cout << "  direct invalidations write-first/read-first="
              << direct_write_first.hbf_stats.invalidations << "/"
              << direct_read_first.hbf_stats.invalidations << "\n";
    if (direct_write_first.hbf_stats.invalidations != 0 ||
        direct_read_first.hbf_stats.invalidations == 0) {
        throw std::runtime_error(
            "composition-initial-image: all-hbf inferred a page from a future read/write");
    }

    const std::vector<MemoryRequest> prefix{
        MemoryRequest{
            .id = "prefix-r8",
            .op = Op::Read,
            .addr = 8 * kPageSize,
            .bytes = kPageSize,
            .arrival_ns = 0.0,
            .index = 0,
        },
    };
    const std::vector<MemoryRequest> full_population{
        MemoryRequest{
            .id = "population-r0",
            .op = Op::Read,
            .addr = 0,
            .bytes = kPageSize,
            .arrival_ns = 0.0,
            .index = 0,
        },
        MemoryRequest{
            .id = "population-r8",
            .op = Op::Read,
            .addr = 8 * kPageSize,
            .bytes = kPageSize,
            .arrival_ns = 1.0,
            .index = 1,
        },
    };
    DirectRunKnobs population_knobs;
    population_knobs.initial_image_requests = &full_population;
    const auto fixed_population = run_direct_policy(
        all_hbf_policy(),
        hbm_cfg,
        hbf_cfg,
        prefix,
        population_knobs);
    if (fixed_population.hbf_stats.mapping_entries != 2 ||
        fixed_population.hbf_stats.physical_read_bytes == 0) {
        throw std::runtime_error(
            "composition-initial-image: explicit population was not used");
    }
    const std::vector<MemoryRequest> incomplete_population{
        full_population.front(),
    };
    DirectRunKnobs incomplete_knobs;
    incomplete_knobs.initial_image_requests = &incomplete_population;
    bool incomplete_rejected = false;
    try {
        (void)run_direct_policy(
            all_hbf_policy(),
            hbm_cfg,
            hbf_cfg,
            prefix,
            incomplete_knobs);
    } catch (const std::runtime_error&) {
        incomplete_rejected = true;
    }
    if (!incomplete_rejected) {
        throw std::runtime_error(
            "composition-initial-image: incomplete explicit population must fail");
    }

    std::cout << "  initial_image_no_future_knowledge_ok=yes\n";
    std::cout << "  fixed_population_prefix_coverage_ok=yes\n";
}

void probe_composition_coop_write() {
    std::cout << "\n== cooperative HBM/HBF write staging probe ==\n";
    HbmConfig hbm_cfg;
    hbm_cfg.device.stacks = 1;
    hbm_cfg.device.channels_per_stack = 2;
    hbm_cfg.device.pseudo_channels_per_channel = 1;
    HbfConfig hbf_cfg;
    hbf_cfg.device.stacks = 1;
    hbf_cfg.device.channels_per_stack = 1;
    hbf_cfg.device.dies_per_channel = 1;
    hbf_cfg.device.planes_per_die = 4;
    hbf_cfg.device.blocks_per_plane = 64;
    hbf_cfg.device.pages_per_block = 64;
    hbf_cfg.device.page_size_bytes = 4096;
    hbf_cfg.host.mapping_entries_per_page = 4096;
    hbf_cfg.host.write_coalescing_enabled = true;
    hbf_cfg.host.write_buffer_pages = 4;
    hbf_cfg.host.write_buffer_flush_threshold_pages = 2;

    std::vector<MemoryRequest> writes;
    for (std::size_t i = 0; i < 16; ++i) {
        writes.push_back(MemoryRequest{
            .id = "w" + std::to_string(i),
            .op = Op::Write,
            .addr = 4096ull * i,
            .bytes = 4096,
            .arrival_ns = static_cast<double>(i),
            .index = i,
            .kind = SemanticKind::Unknown,
            .label = {},
        });
    }
    const auto max_latency = [](const PolicyRunResult& r) {
        double worst = 0.0;
        for (const auto v : r.offered_latencies_ns) {
            worst = std::max(worst, v);
        }
        return worst;
    };
    const double hbm_fast_limit_ns = hbf_cfg.device.t_program_page_ns / 2.0;

    // Region holds the burst: writes complete at HBM speed, destages park
    // and stream out after the last user op — every page still programs
    // (work conserved into the tail).
    DirectRunKnobs burst_knobs;
    // Isolate capacity-triggered destage; watermark policy has its own probe.
    burst_knobs.hbm_write_buffer_destage_watermark = 1.0;
    burst_knobs.hbm_write_buffer_bytes = 1ull << 20;  // 1 MiB >= 16 pages
    const auto burst = run_direct_policy(
        flat_policy(0), hbm_cfg, hbf_cfg, writes, burst_knobs);
    std::cout << "  burst: max write latency = " << fixed(max_latency(burst))
              << " ns, region waits = " << burst.hbm_write_buffer_full_waits
              << ", tail past user finish = "
              << fixed(burst.finish_ns - burst.user_finish_ns)
              << " ns, hbf page_programs = " << burst.hbf_stats.page_programs << "\n";
    if (burst.hbm_write_buffer_full_waits != 0 ||
        max_latency(burst) > hbm_fast_limit_ns) {
        throw std::runtime_error(
            "composition-coop-write: a burst within the region must complete at HBM speed");
    }
    if (burst.hbm_write_buffer_user_write_bytes != 16ull * 4096 ||
        burst.hbm_write_buffer_destaged_bytes != 16ull * 4096 ||
        burst.hbf_stats.logical_write_bytes != 16ull * 4096 ||
        burst.hbf_stats.page_programs < 16 ||
        burst.finish_ns - burst.user_finish_ns < 75000.0) {
        throw std::runtime_error(
            "composition-coop-write: every buffered page must destage and program in the tail");
    }
    std::cout << "  coop_burst_ok=yes\n";

    // Multiple sub-page updates to one logical page share one physical slot.
    // Only initialized dirty byte ranges are readable from and destaged out
    // of that slot; an untouched range retains its original HBF placement.
    auto one_page_partial_knobs = burst_knobs;
    one_page_partial_knobs.hbm_write_buffer_bytes = 4096;
    const std::vector<MemoryRequest> same_page_partial{
        MemoryRequest{.id = "partial-a", .op = Op::Write, .addr = 128,
            .bytes = 128, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::GeneratedContext, .label = {}},
        MemoryRequest{.id = "partial-b", .op = Op::Write, .addr = 512,
            .bytes = 768, .arrival_ns = 1.0, .index = 1,
            .kind = SemanticKind::Scratch, .label = {}},
        MemoryRequest{.id = "partial-overwrite", .op = Op::Write, .addr = 640,
            .bytes = 256, .arrival_ns = 2.0, .index = 2,
            .kind = SemanticKind::Scratch, .label = {}},
        MemoryRequest{.id = "read-dirty", .op = Op::Read, .addr = 128,
            .bytes = 128, .arrival_ns = 3.0, .index = 3,
            .kind = SemanticKind::GeneratedContext, .label = {}},
        MemoryRequest{.id = "read-untouched", .op = Op::Read, .addr = 256,
            .bytes = 64, .arrival_ns = 4.0, .index = 4,
            .kind = SemanticKind::Unknown, .label = {}},
    };
    const auto partial = run_direct_policy(
        flat_policy(0), hbm_cfg, hbf_cfg, same_page_partial,
        one_page_partial_knobs);
    std::cout << "  partial: peak=" << partial.hbm_write_buffer_peak_bytes
              << " waits=" << partial.hbm_write_buffer_full_waits
              << " user=" << partial.hbm_write_buffer_user_write_bytes
              << " destage=" << partial.hbm_write_buffer_destaged_bytes
              << " link=" << partial.base_die_link_stats.write_bytes
              << " hbf=" << partial.hbf_stats.logical_write_bytes
              << " accesses=" << partial.hbm_user_accesses << '/' << partial.hbf_user_accesses
              << " programs=" << partial.hbf_stats.data_programs << '\n';
    if (partial.hbm_write_buffer_peak_bytes != 4096 ||
        partial.hbm_write_buffer_full_waits != 0 ||
        partial.hbm_write_buffer_user_write_bytes != 1152 ||
        partial.hbm_write_buffer_destaged_bytes != 896 ||
        partial.base_die_link_stats.write_bytes != 896 ||
        partial.hbf_stats.logical_write_bytes != 896 ||
        partial.hbm_user_accesses != 4 || partial.hbf_user_accesses != 1 ||
        partial.hbf_stats.data_programs != 1) {
        throw std::runtime_error(
            "composition-coop-write: same-page sub-page writes did not share "
            "one slot with byte-exact dirty ownership");
    }
    std::cout << "  coop_subpage_merge_ok=yes\n";

    // One unaligned request may cross a page boundary. It owns one full slot
    // per touched logical page, while foreground and destage traffic cover
    // only the request's 256 B + 512 B initialized pieces.
    auto two_page_partial_knobs = burst_knobs;
    two_page_partial_knobs.hbm_write_buffer_bytes = 2ull * 4096;
    const std::vector<MemoryRequest> cross_page_partial{
        MemoryRequest{.id = "cross-page-partial", .op = Op::Write,
            .addr = 4096 - 256, .bytes = 768, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Scratch, .label = {}},
        MemoryRequest{.id = "cross-page-reread", .op = Op::Read,
            .addr = 4096 - 256, .bytes = 768, .arrival_ns = 1.0, .index = 1,
            .kind = SemanticKind::Scratch, .label = {}},
    };
    const auto cross_page = run_direct_policy(
        flat_policy(0), hbm_cfg, hbf_cfg, cross_page_partial,
        two_page_partial_knobs);
    if (cross_page.hbm_write_buffer_peak_bytes != 2ull * 4096 ||
        cross_page.hbm_write_buffer_full_waits != 0 ||
        cross_page.hbm_write_buffer_user_write_bytes != 768 ||
        cross_page.hbm_write_buffer_destaged_bytes != 768 ||
        cross_page.base_die_link_stats.write_bytes != 768 ||
        cross_page.hbf_stats.logical_write_bytes != 768 ||
        cross_page.hbm_user_accesses != 4 ||
        cross_page.hbf_stats.data_programs != 2) {
        throw std::runtime_error(
            "composition-coop-write: unaligned cross-page write lost slot or "
            "byte-range accounting");
    }
    std::cout << "  coop_cross_page_subpage_ok=yes\n";

    // ASTRA decode traces commonly emit 1536 B and 2304 B writes. Exercise
    // both sizes with an overlap across two resident pages: user traffic
    // counts every supplied byte, while destage counts their 3584 B union.
    const std::vector<MemoryRequest> astra_partial_sizes{
        MemoryRequest{.id = "astra-1536", .op = Op::Write, .addr = 3072,
            .bytes = 1536, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Scratch, .label = {}},
        MemoryRequest{.id = "astra-2304", .op = Op::Write,
            .addr = 4096 + 256, .bytes = 2304, .arrival_ns = 1.0, .index = 1,
            .kind = SemanticKind::Scratch, .label = {}},
    };
    const auto astra_partial = run_direct_policy(
        flat_policy(0), hbm_cfg, hbf_cfg, astra_partial_sizes,
        two_page_partial_knobs);
    if (astra_partial.hbm_write_buffer_peak_bytes != 2ull * 4096 ||
        astra_partial.hbm_write_buffer_full_waits != 0 ||
        astra_partial.hbm_write_buffer_user_write_bytes != 3840 ||
        astra_partial.hbm_write_buffer_destaged_bytes != 3584 ||
        astra_partial.base_die_link_stats.write_bytes != 3584 ||
        astra_partial.hbf_stats.logical_write_bytes != 3584 ||
        astra_partial.hbf_stats.data_programs != 2) {
        throw std::runtime_error(
            "composition-coop-write: ASTRA-sized sub-page writes lost host or "
            "dirty-union byte accounting");
    }
    std::cout << "  coop_astra_subpage_sizes_ok=yes\n";

    const std::vector<MemoryRequest> one_slot_cross_page{
        MemoryRequest{.id = "astra-2304-one-slot", .op = Op::Write,
            .addr = 4096 - 1024, .bytes = 2304, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Scratch, .label = {}},
    };
    const auto one_slot_cross = run_direct_policy(
        flat_policy(0), hbm_cfg, hbf_cfg, one_slot_cross_page,
        one_page_partial_knobs);
    if (one_slot_cross.hbm_write_buffer_peak_bytes != 4096 ||
        one_slot_cross.hbm_write_buffer_full_waits == 0 ||
        one_slot_cross.hbm_write_buffer_user_write_bytes != 2304 ||
        one_slot_cross.hbm_write_buffer_destaged_bytes != 2304 ||
        one_slot_cross.base_die_link_stats.write_bytes != 2304 ||
        one_slot_cross.hbf_stats.logical_write_bytes != 2304 ||
        one_slot_cross.hbf_stats.data_programs != 2) {
        throw std::runtime_error(
            "composition-coop-write: cross-page 2304 B write did not stream "
            "through a one-page region with causal backpressure");
    }
    std::cout << "  coop_cross_page_one_slot_stream_ok=yes\n";

    // Region of 2 pages: capacity pressure force-drains parked entries at
    // program pace; writes honestly throttle.
    DirectRunKnobs tight_knobs = burst_knobs;
    tight_knobs.hbm_write_buffer_bytes = 2ull * 4096;
    const auto pressed = run_direct_policy(
        flat_policy(0), hbm_cfg, hbf_cfg, writes, tight_knobs);
    std::cout << "  tight region: max write latency = " << fixed(max_latency(pressed))
              << " ns, region waits = " << pressed.hbm_write_buffer_full_waits
              << ", hbf page_programs = " << pressed.hbf_stats.page_programs << "\n";
    if (pressed.hbm_write_buffer_full_waits == 0 || max_latency(pressed) < 75000.0) {
        throw std::runtime_error(
            "composition-coop-write: a full region must throttle writes to program pace");
    }
    if (pressed.hbf_stats.page_programs != burst.hbf_stats.page_programs) {
        throw std::runtime_error(
            "composition-coop-write: program work must be conserved across region sizes");
    }
    std::cout << "  coop_backpressure_ok=yes\n";

    // Cross a logical stack stripe and ensure each page is charged to its
    // owner link instead of billing all bytes to the first page's stack.
    auto striped_hbf_cfg = hbf_cfg;
    striped_hbf_cfg.device.stacks = 2;
    striped_hbf_cfg.host.mapping_entries_per_page = 1;
    const std::vector<MemoryRequest> striped_write{
        MemoryRequest{.id = "striped-write", .op = Op::Write, .addr = 0,
            .bytes = 8192, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Unknown, .label = {}},
    };
    const auto striped_direct = run_direct_policy(
        flat_policy(0),
        hbm_cfg,
        striped_hbf_cfg,
        striped_write,
        burst_knobs);
    if (striped_direct.base_die_link_stats.write_transfers != 2 ||
        striped_direct.base_die_link_stats.write_bytes != 8192 ||
        std::abs(striped_direct.base_die_link_stats.write_fixed_latency_work_ns -
                 2.0 * burst_knobs.base_die_link.latency_ns) > 1e-9) {
        throw std::runtime_error(
            "composition-coop-write: direct destage must split bytes and fixed latency by stack");
    }

    std::cout << "  coop_stack_striping_ok=yes\n";
    std::cout << "  base_die_link_fixed_latency_work_ok=yes\n";

    auto undersized_region = burst_knobs;
    undersized_region.hbm_write_buffer_bytes = 4096;
    const auto oversized_write = run_direct_policy(
        flat_policy(0),
        hbm_cfg,
        striped_hbf_cfg,
        striped_write,
        undersized_region);
    if (oversized_write.hbm_write_buffer_peak_bytes != 4096 ||
        oversized_write.hbm_write_buffer_full_waits == 0 ||
        oversized_write.hbm_write_buffer_destaged_bytes != 8192 ||
        oversized_write.base_die_link_stats.write_bytes != 8192 ||
        oversized_write.hbf_stats.logical_write_bytes != 8192 ||
        oversized_write.hbf_stats.data_programs != 2) {
        throw std::runtime_error(
            "composition-coop-write: a request larger than the region must "
            "stream causally through finite page slots");
    }
    std::cout << "  coop_oversized_write_bounded_ok=yes\n";

    auto small_hbm_cfg = hbm_cfg;
    small_hbm_cfg.device.capacity_bytes = 1ull << 20;
    auto allocated_knobs = burst_knobs;
    allocated_knobs.hbm_write_buffer_bytes = 64ull << 10;
    const std::vector<MemoryRequest> high_address_reuse{
        MemoryRequest{.id = "high-write", .op = Op::Write, .addr = 2ull << 20,
            .bytes = 4096, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Unknown, .label = {}},
        MemoryRequest{.id = "high-read", .op = Op::Read, .addr = 2ull << 20,
            .bytes = 4096, .arrival_ns = 1.0, .index = 1,
            .kind = SemanticKind::Unknown, .label = {}},
    };
    const auto allocated = run_direct_policy(
        flat_policy(0),
        small_hbm_cfg,
        hbf_cfg,
        high_address_reuse,
        allocated_knobs);
    if (allocated.hbm_user_accesses != 2 ||
        allocated.hbm_write_buffer_destaged_bytes != 4096) {
        throw std::runtime_error(
            "composition-coop-write: global HBF tags must use physical HBM allocations");
    }

    // One-slot pressure forces page A to destage before page B can reuse the
    // same physical HBM slot. The following read has the same offered arrival
    // as both writes, but in-order buffer backpressure admits it only after A
    // has entered the authoritative HBF write path. It must never read B
    // through A's stale slot pointer.
    auto one_slot_knobs = burst_knobs;
    one_slot_knobs.hbm_write_buffer_bytes = 4096;
    const std::vector<MemoryRequest> reuse_after_pressure{
        MemoryRequest{.id = "slot-a", .op = Op::Write, .addr = 2ull << 20,
            .bytes = 4096, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Unknown, .label = {}},
        MemoryRequest{.id = "slot-b", .op = Op::Write,
            .addr = (2ull << 20) + 4096, .bytes = 4096,
            .arrival_ns = 0.0, .index = 1,
            .kind = SemanticKind::Unknown, .label = {}},
        MemoryRequest{.id = "read-a-after-reuse", .op = Op::Read,
            .addr = 2ull << 20, .bytes = 4096,
            .arrival_ns = 0.0, .index = 2,
            .kind = SemanticKind::Unknown, .label = {}},
    };
    const auto reuse_after_release = run_direct_policy(
        flat_policy(0),
        small_hbm_cfg,
        hbf_cfg,
        reuse_after_pressure,
        one_slot_knobs);
    std::cout << "  slot reuse: hbm/hbf user = "
              << reuse_after_release.hbm_user_accesses << "/"
              << reuse_after_release.hbf_user_accesses
              << ", read latency = "
              << fixed(reuse_after_release.offered_latencies_ns.back()) << " ns\n";
    if (reuse_after_release.hbm_user_accesses != 2 ||
        reuse_after_release.hbf_user_accesses != 1 ||
        reuse_after_release.hbm_write_buffer_full_waits != 1 ||
        reuse_after_release.hbm_write_buffer_wait_ns <= 0.0 ||
        reuse_after_release.offered_latencies_ns.back() <
            reuse_after_release.hbm_write_buffer_wait_ns) {
        throw std::runtime_error(
            "composition-coop-write: slot reuse lost in-order placement coherence");
    }
    std::cout << "  coop_slot_reuse_coherence_ok=yes\n";

    bool foreground_overlap_rejected = false;
    try {
        (void)run_direct_policy(
            flat_policy(1ull << 20),
            small_hbm_cfg,
            hbf_cfg,
            {MemoryRequest{.id = "reserved-overlap", .op = Op::Read,
                .addr = 980ull << 10, .bytes = 4096, .arrival_ns = 0.0,
                .index = 0, .kind = SemanticKind::Scratch, .label = {}}},
            allocated_knobs);
    } catch (const std::runtime_error&) {
        foreground_overlap_rejected = true;
    }
    if (!foreground_overlap_rejected) {
        throw std::runtime_error(
            "composition-coop-write: ordinary HBM traffic overlapped write region");
    }
    std::cout << "  coop_physical_allocation_ok=yes\n";

    bool subpage_region_rejected = false;
    try {
        auto invalid_region = burst_knobs;
        invalid_region.hbm_write_buffer_bytes = 4095;
        (void)run_direct_policy(
            flat_policy(0), hbm_cfg, hbf_cfg, {writes.front()}, invalid_region);
    } catch (const std::runtime_error&) {
        subpage_region_rejected = true;
    }
    bool unaligned_base_rejected = false;
    try {
        auto unaligned_hbm = hbm_cfg;
        unaligned_hbm.device.capacity_bytes += 1;
        auto one_page_region = burst_knobs;
        one_page_region.hbm_write_buffer_bytes = 4096;
        (void)run_direct_policy(
            flat_policy(0), unaligned_hbm, hbf_cfg, {writes.front()},
            one_page_region);
    } catch (const std::runtime_error&) {
        unaligned_base_rejected = true;
    }
    if (!subpage_region_rejected || !unaligned_base_rejected) {
        throw std::runtime_error(
            "composition-coop-write: invalid slot-region geometry did not fail closed");
    }
    std::cout << "  coop_invalid_region_geometry_rejected=yes\n";
    std::cout << "  expectation: the HBM write region absorbs bursts at HBM speed,"
              << " destages yield to user traffic and drain afterwards"
              << " over the configured HBM->HBF path; capacity overflow honestly"
              << " throttles to the destage/program rate.\n";
}

void probe_composition_reuse_routing() {
    std::cout << "\n== read-after-write routing coherence probe ==\n";
    HbmConfig hbm_cfg;
    hbm_cfg.device.stacks = 1;
    hbm_cfg.device.channels_per_stack = 2;
    hbm_cfg.device.pseudo_channels_per_channel = 1;
    HbfConfig hbf_cfg;
    hbf_cfg.device.stacks = 1;
    hbf_cfg.device.channels_per_stack = 1;
    hbf_cfg.device.dies_per_channel = 1;
    hbf_cfg.device.planes_per_die = 2;
    hbf_cfg.device.blocks_per_plane = 64;
    hbf_cfg.device.pages_per_block = 64;
    hbf_cfg.device.page_size_bytes = 4096;
    hbf_cfg.host.mapping_entries_per_page = 4096;

    const auto make_ops = []() {
        std::vector<MemoryRequest> ops;
        ops.push_back(MemoryRequest{.id = "w", .op = Op::Write, .addr = 16384,
            .bytes = 4096, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Unknown, .label = {}});
        ops.push_back(MemoryRequest{.id = "r-written", .op = Op::Read, .addr = 16384,
            .bytes = 4096, .arrival_ns = 1.0, .index = 1,
            .kind = SemanticKind::Unknown, .label = {}});
        ops.push_back(MemoryRequest{.id = "r-fresh", .op = Op::Read, .addr = 20480,
            .bytes = 4096, .arrival_ns = 2.0, .index = 2,
            .kind = SemanticKind::Unknown, .label = {}});
        return ops;
    };
    const DirectRunKnobs knobs{};

    // A flat parent that straddles the tier boundary must be split for both
    // reads and writes while retaining one user-level op/completion latency.
    const std::vector<MemoryRequest> flat_cross_read{
        MemoryRequest{.id = "flat-cross-read", .op = Op::Read, .addr = 4096,
            .bytes = 8192, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Unknown, .label = {}},
    };
    const auto flat_read = run_direct_policy(
        flat_policy(8192), hbm_cfg, hbf_cfg, flat_cross_read, knobs);
    if (flat_read.ops != 1 || flat_read.hbm_user_accesses != 1 ||
        flat_read.hbf_user_accesses != 1 ||
        flat_read.hbf_stats.logical_read_bytes != 4096 ||
        !flat_read.warnings.empty()) {
        throw std::runtime_error(
            "composition-reuse-routing: flat cross-boundary read was not split exactly");
    }

    const std::vector<MemoryRequest> flat_cross_write{
        MemoryRequest{.id = "flat-cross-write", .op = Op::Write, .addr = 4096,
            .bytes = 8192, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Unknown, .label = {}},
    };
    const auto flat_write = run_direct_policy(
        flat_policy(8192), hbm_cfg, hbf_cfg, flat_cross_write, knobs);
    if (flat_write.ops != 1 || flat_write.hbm_user_accesses != 1 ||
        flat_write.hbf_user_accesses != 1 ||
        flat_write.hbm_stats.write_bytes - flat_write.hbm_stats.controller_buffer_write_bytes != 4096 ||
        flat_write.hbf_stats.logical_write_bytes != 4096 ||
        !flat_write.warnings.empty()) {
        throw std::runtime_error(
            "composition-reuse-routing: flat cross-boundary write was not split exactly");
    }

    bool unaligned_boundary_rejected = false;
    try {
        (void)run_direct_policy(
            flat_policy(8193), hbm_cfg, hbf_cfg, flat_cross_read, knobs);
    } catch (const std::runtime_error&) {
        unaligned_boundary_rejected = true;
    }
    if (!unaligned_boundary_rejected) {
        throw std::runtime_error(
            "composition-reuse-routing: sub-page tier boundary must fail closed");
    }
    std::cout << "  reuse_flat_boundary_split_ok=yes\n";

    // Read-only-direct: written data lives on HBM; only never-written pages
    // may be read from the flash R region.
    const auto ro = run_direct_policy(
        read_only_direct_policy(8192), hbm_cfg, hbf_cfg, make_ops(), knobs);
    std::cout << "  read-only-direct: hbm/hbf user = " << ro.hbm_user_accesses
              << "/" << ro.hbf_user_accesses
              << ", static read bytes = " << ro.hbf_static_read_bytes << "\n";
    if (ro.hbm_user_accesses != 2 || ro.hbf_user_accesses != 1 ||
        ro.hbf_static_read_bytes != 4096) {
        throw std::runtime_error(
            "composition-reuse-routing: read-only-direct must keep written data "
            "on HBM and read only fresh pages from flash");
    }
    std::cout << "  reuse_readonly_ok=yes\n";

    // A bounded completion-order window must neither move admission backwards
    // nor wait for an older HBF request after a younger HBM request returned a
    // credit.
    DirectRunKnobs window_knobs;
    window_knobs.max_outstanding_requests = 2;
    const std::vector<MemoryRequest> window_trace{
        MemoryRequest{.id = "slow-hbf", .op = Op::Read, .addr = 8192,
            .bytes = 64, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Unknown, .label = {}},
        MemoryRequest{.id = "fast-hbm", .op = Op::Read, .addr = 0,
            .bytes = 64, .arrival_ns = 0.0, .index = 1,
            .kind = SemanticKind::Unknown, .label = {}},
        MemoryRequest{.id = "gated-hbm-a", .op = Op::Read, .addr = 64,
            .bytes = 64, .arrival_ns = 0.0, .index = 2,
            .kind = SemanticKind::Unknown, .label = {}},
        MemoryRequest{.id = "gated-hbm-b", .op = Op::Read, .addr = 128,
            .bytes = 64, .arrival_ns = 0.0, .index = 3,
            .kind = SemanticKind::Unknown, .label = {}},
    };
    const auto windowed = run_direct_policy(
        flat_policy(4096), hbm_cfg, hbf_cfg, window_trace, window_knobs);
    if (windowed.offered_latencies_ns.back() > 1000.0) {
        throw std::runtime_error(
            "composition-reuse-routing: closed-loop admission moved backwards in time");
    }
    if (windowed.front_end_admission_waited_ops != 2 ||
        windowed.front_end_admission_wait_work_ns <= 0.0 ||
        windowed.front_end_admission_max_wait_ns <= 0.0 ||
        windowed.front_end_admission_max_wait_ns >
            windowed.front_end_admission_wait_work_ns) {
        throw std::runtime_error(
            "composition-reuse-routing: direct front-end admission work was not conserved");
    }
    std::cout << "  closed_loop_monotonic_admission_ok=yes\n";
    std::cout << "  direct_front_end_admission_wait_ok=yes\n";

    // Parents ready in the same phase share credits round-robin. A long DMA
    // may keep using returned credits, but it cannot consume the first credit
    // of a short peer that was already ready at the same timestamp.
    const std::vector<MemoryRequest> fair_trace{
        MemoryRequest{
            .id = "long-dma",
            .op = Op::Read,
            .addr = 0,
            .bytes = 16 * hbf_cfg.device.page_size_bytes,
            .arrival_ns = 0.0,
            .index = 0,
            .kind = SemanticKind::ModelWeights,
            .label = {},
            .phase = 0,
        },
        MemoryRequest{
            .id = "short-peer",
            .op = Op::Read,
            .addr = 1ull << 20,
            .bytes = hbf_cfg.device.page_size_bytes,
            .arrival_ns = 0.0,
            .index = 1,
            .kind = SemanticKind::SharedContext,
            .label = {},
            .phase = 0,
        },
    };
    const auto fair = run_direct_policy(
        all_hbm_policy(),
        hbm_cfg,
        hbf_cfg,
        fair_trace,
        window_knobs);
    std::cout << "  fairness: long offered/service = "
              << fixed(fair.offered_latencies_ns.at(0)) << "/"
              << fixed(fair.service_latencies_ns.at(0))
              << " ns, short offered/service = "
              << fixed(fair.offered_latencies_ns.at(1)) << "/"
              << fixed(fair.service_latencies_ns.at(1)) << " ns\n";
    if (fair.offered_latencies_ns.size() != 2 ||
        fair.service_latencies_ns.size() != 2 ||
        std::abs(
            fair.offered_latencies_ns.at(1) -
            fair.service_latencies_ns.at(1)) > 1e-9) {
        throw std::runtime_error(
            "composition-reuse-routing: long parent monopolized the "
            "physical-transaction window");
    }
    std::cout << "  parent_round_robin_window_fairness_ok=yes\n";

    // Fairness must not violate same-page program order. In particular, the
    // read of the second page below cannot overtake the corresponding segment
    // of the earlier two-page write and fall through to static HBF.
    const std::vector<MemoryRequest> ordered_overlap_trace{
        MemoryRequest{
            .id = "two-page-write",
            .op = Op::Write,
            .addr = 8192,
            .bytes = 2 * hbf_cfg.device.page_size_bytes,
            .arrival_ns = 0.0,
            .index = 0,
            .kind = SemanticKind::GeneratedContext,
            .label = {},
            .phase = 0,
        },
        MemoryRequest{
            .id = "read-second-written-page",
            .op = Op::Read,
            .addr = 8192 + hbf_cfg.device.page_size_bytes,
            .bytes = hbf_cfg.device.page_size_bytes,
            .arrival_ns = 0.0,
            .index = 1,
            .kind = SemanticKind::SharedContext,
            .label = {},
            .phase = 0,
        },
    };
    const auto ordered_overlap = run_direct_policy(
        read_only_direct_policy(0),
        hbm_cfg,
        hbf_cfg,
        ordered_overlap_trace,
        window_knobs);
    if (ordered_overlap.hbm_user_accesses != 3 ||
        ordered_overlap.hbf_user_accesses != 0 ||
        ordered_overlap.hbf_static_read_bytes != 0 ||
        ordered_overlap.front_end_admission_waited_ops != 1) {
        throw std::runtime_error(
            "composition-reuse-routing: round-robin violated overlapping "
            "write/read order");
    }
    std::cout << "  parent_round_robin_overlap_ordering_ok=yes\n";

    // Window credits belong to page transactions, not trace-record parents.
    // The same byte stream must therefore produce the same physical schedule
    // whether the trace stores one page per record or one 64-page record.
    std::vector<MemoryRequest> page_records;
    for (std::size_t page = 0; page < 64; ++page) {
        page_records.push_back(MemoryRequest{
            .id = "page-" + std::to_string(page),
            .op = Op::Read,
            .addr = page * hbf_cfg.device.page_size_bytes,
            .bytes = hbf_cfg.device.page_size_bytes,
            .arrival_ns = 0.0,
            .index = page,
            .kind = SemanticKind::Unknown,
            .label = {},
        });
    }
    const std::vector<MemoryRequest> large_record{
        MemoryRequest{
            .id = "large",
            .op = Op::Read,
            .addr = 0,
            .bytes = 64 * hbf_cfg.device.page_size_bytes,
            .arrival_ns = 0.0,
            .index = 0,
            .kind = SemanticKind::Unknown,
            .label = {},
        },
    };
    DirectRunKnobs transaction_knobs;
    transaction_knobs.max_outstanding_requests = 8;
    const auto direct_pages = run_direct_policy(
        all_hbf_policy(),
        hbm_cfg,
        hbf_cfg,
        page_records,
        transaction_knobs);
    const auto direct_large = run_direct_policy(
        all_hbf_policy(),
        hbm_cfg,
        hbf_cfg,
        large_record,
        transaction_knobs);
    if (direct_pages.user_finish_ns != direct_large.user_finish_ns ||
        direct_pages.hbf_user_accesses != 64 ||
        direct_large.hbf_user_accesses != 64 ||
        direct_pages.hbf_stats.physical_read_bytes !=
            direct_large.hbf_stats.physical_read_bytes) {
        throw std::runtime_error(
            "composition-reuse-routing: direct window depends on parent "
            "request size");
    }

    LayerStreamingConfig streaming_cfg;
    streaming_cfg.hbm = hbm_cfg;
    streaming_cfg.hbf = hbf_cfg;
    // These unlabeled pages are backing-tier data under hybrid residency, so
    // this synthetic one-layer workload needs a buffer for all 64 pages.
    streaming_cfg.layer_buffer_bytes = 64 * hbf_cfg.device.page_size_bytes;
    streaming_cfg.max_outstanding_requests = 8;
    LayerStreamingPolicy streaming_pages_system(streaming_cfg);
    LayerStreamingPolicy streaming_large_system(streaming_cfg);
    const auto streaming_pages =
        streaming_pages_system.run(page_records);
    const auto streaming_large =
        streaming_large_system.run(large_record);
    if (streaming_pages.user_finish_ns != streaming_large.user_finish_ns ||
        streaming_pages.hbm_user_accesses != 64 ||
        streaming_large.hbm_user_accesses != 64 ||
        streaming_pages.hbm_stats.read_bytes !=
            streaming_large.hbm_stats.read_bytes) {
        throw std::runtime_error(
            "composition-reuse-routing: layer window depends on parent "
            "request size");
    }
    const auto check_backing_window = [](const auto& result) {
        const auto& stats = result.streaming_stats;
        return stats.backing_request_credit_limit == 8 &&
            stats.backing_max_inflight_requests == 8 &&
            stats.backing_admission_waited_requests == 56 &&
            stats.backing_admission_wait_work_ns > 0.0 &&
            stats.backing_admission_max_wait_ns > 0.0 &&
            stats.backing_admission_max_wait_ns <=
                stats.backing_admission_wait_work_ns;
    };
    if (!check_backing_window(streaming_pages) ||
        !check_backing_window(streaming_large)) {
        throw std::runtime_error(
            "composition-reuse-routing: layer backing DMA escaped its "
            "completion-order credit pool");
    }
    std::cout << "  page_transaction_window_record_size_invariant_ok=yes\n";
    std::cout << "  layer_backing_credit_limit_ok=yes\n";

    // With cooperative staging, a page still parked in the HBM write region
    // is served from HBM; only after the HBF write path accepts its destage
    // does it belong to the FTL.
    DirectRunKnobs coop_knobs;
    coop_knobs.hbm_write_buffer_bytes = 1ull << 20;
    const auto parked = run_direct_policy(
        flat_policy(0), hbm_cfg, hbf_cfg, make_ops(), coop_knobs);
    std::cout << "  parked-region: hbm/hbf user = " << parked.hbm_user_accesses
              << "/" << parked.hbf_user_accesses
              << ", static read bytes = " << parked.hbf_static_read_bytes << "\n";
    if (parked.hbm_user_accesses != 2 || parked.hbf_user_accesses != 1 ||
        parked.hbf_static_read_bytes != 0) {
        throw std::runtime_error(
            "composition-reuse-routing: a page parked in the HBM write region "
            "must be read from HBM");
    }
    std::cout << "  reuse_parked_ok=yes\n";

    std::cout << "  expectation: reads of pages written this run route to where"
              << " the data actually lives (page-table knowledge), never to a"
              << " static physical region that never received it.\n";
}

void run_all() {
    probe_hbm_interface();
    probe_hbm_boundaries();
    probe_hbm_channels();
    probe_hbm_turnaround();
    probe_hbf_standard();
    probe_composition_kind_blind();
    probe_composition_static_mapping();
    probe_composition_initial_image();
    probe_composition_coop_write();
    probe_composition_reuse_routing();
}

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " [probe] [--waterfall] [--waterfall-limit N]"
              << " [--trace path.json] [--folded path.folded]"
              << " [--trace-mode off|summary|sampled|full] [--trace-sample N]\n"
              << "  probes: all, hbm-interface, hbm-boundaries, hbm-channels,"
              << " hbm-turnaround,"
              << " hbf-standard, hbf-ecc-pipeline, hbf-bank-pipeline, hbf-host-boundary,"
              << " hbf-program-barriers, hbf-calendar, composition-kind-blind,"
              << " composition-static-mapping, composition-initial-image,"
              << " composition-coop-write, composition-reuse-routing\n";
}

TraceMode parse_trace_mode_value(const std::string& value) {
    if (value == "off") {
        return TraceMode::Off;
    }
    if (value == "summary") {
        return TraceMode::Summary;
    }
    if (value == "sampled") {
        return TraceMode::Sampled;
    }
    if (value == "full") {
        return TraceMode::Full;
    }
    throw std::runtime_error("unknown trace mode: " + value);
}

ProbeOptions parse_args(int argc, char** argv) {
    ProbeOptions options;
    bool probe_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--waterfall") {
            options.waterfall = true;
        } else if (arg == "--waterfall-limit") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--waterfall-limit requires a value");
            }
            options.waterfall_limit = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--trace") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--trace requires a path");
            }
            options.trace_path = argv[++i];
        } else if (arg == "--folded") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--folded requires a path");
            }
            options.folded_path = argv[++i];
        } else if (arg == "--trace-mode") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--trace-mode requires a value");
            }
            options.trace_mode = parse_trace_mode_value(argv[++i]);
        } else if (arg == "--trace-sample") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--trace-sample requires a value");
            }
            options.trace_sample_period = std::max<std::uint64_t>(1, std::stoull(argv[++i]));
            options.trace_mode = TraceMode::Sampled;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (!probe_set) {
            options.probe = arg;
            probe_set = true;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        g_options = parse_args(argc, argv);
        if (g_options.probe == "all") {
            run_all();
        } else if (g_options.probe == "hbm-interface") {
            probe_hbm_interface();
        } else if (g_options.probe == "hbm-boundaries") {
            probe_hbm_boundaries();
        } else if (g_options.probe == "hbm-channels") {
            probe_hbm_channels();
        } else if (g_options.probe == "hbm-turnaround") {
            probe_hbm_turnaround();
        } else if (g_options.probe == "hbf-standard") {
            probe_hbf_standard();
        } else if (g_options.probe == "hbf-ecc-pipeline") {
            probe_hbf_ecc_pipeline();
        } else if (g_options.probe == "hbf-bank-pipeline") {
            probe_hbf_bank_pipeline();
        } else if (g_options.probe == "hbf-host-boundary") {
            probe_hbf_host_boundary();
        } else if (g_options.probe == "hbf-program-barriers") {
            probe_hbf_program_barriers();
        } else if (g_options.probe == "hbf-calendar") {
            probe_hbf_calendar();
        } else if (g_options.probe == "composition-kind-blind") {
            probe_composition_kind_blind();
        } else if (g_options.probe == "composition-static-mapping") {
            probe_composition_static_mapping();
        } else if (g_options.probe == "composition-initial-image") {
            probe_composition_initial_image();
        } else if (g_options.probe == "composition-coop-write") {
            probe_composition_coop_write();
        } else if (g_options.probe == "composition-reuse-routing") {
            probe_composition_reuse_routing();
        } else {
            usage(argv[0]);
            return 2;
        }
        if (g_options.trace_path) {
            write_chrome_trace(*g_options.trace_path, g_captured_rows);
            std::cout << "  wrote Chrome trace: " << *g_options.trace_path << "\n";
        }
        if (g_options.folded_path) {
            write_folded_trace(*g_options.folded_path, g_captured_rows);
            std::cout << "  wrote folded trace: " << *g_options.folded_path << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
