#include "physical/hbf/hbf_device.hpp"
#include "physical/hbm/hbm_device.hpp"
#include "physical/hybrid/composition_common.hpp"
#include "physical/hybrid/direct_composition.hpp"
#include "physical/hybrid/layer_streaming_composition.hpp"

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
using hbfsim::physical::hbf::HbfAddress;
using hbfsim::physical::hbf::HbfConfig;
using hbfsim::physical::hbf::HbfDevice;
using hbfsim::physical::hbf::HbfStats;
using hbfsim::physical::hbm::HbmAddress;
using hbfsim::physical::hbm::HbmConfig;
using hbfsim::physical::hbm::HbmDevice;
using hbfsim::physical::hbm::HbmStats;
using hbfsim::physical::hybrid::CompositionRunResult;
using hbfsim::physical::hybrid::DirectPolicy;
using hbfsim::physical::hybrid::DirectRunKnobs;
using hbfsim::physical::hybrid::LayerStreamingComposition;
using hbfsim::physical::hybrid::LayerStreamingConfig;
using hbfsim::physical::hybrid::MemoryRequest;
using hbfsim::physical::hybrid::SemanticKind;
using hbfsim::physical::hybrid::all_hbf_policy;
using hbfsim::physical::hybrid::all_hbm_policy;
using hbfsim::physical::hybrid::flat_policy;
using hbfsim::physical::hybrid::read_only_direct_policy;
using hbfsim::physical::hybrid::run_direct_composition;
using hbfsim::physical::hybrid::collect_initial_read_lpns;
using hbfsim::physical::hybrid::hbf_page_capacity;
using hbfsim::physical::hybrid::map_static_hbf_page_addr;
using hbfsim::physical::hybrid::unmap_static_hbf_page_addr;

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
              << " row_hits=" << s.row_hits
              << " row_misses=" << s.row_misses
              << " row_conflicts=" << s.row_conflicts
              << " hit_rate=" << fixed(s.row_hit_rate() * 100.0) << "%"
              << " activations=" << s.activations
              << " precharges=" << s.precharges
              << " refresh_count=" << s.refresh_count
              << " refresh_stall_ns=" << fixed(s.stage_work.refresh_stall_ns)
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

void print_hbf_stats(const HbfDevice& hbf) {
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
              << " program_verify_ns=" << fixed(s.stage_work.program_verify_ns)
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

void probe_hbm_interface() {
    std::cout << "\n== HBM interface derivation probe ==\n";
    const auto close = [](double lhs, double rhs, double tolerance = 1e-9) {
        return std::abs(lhs - rhs) <= tolerance;
    };

    HbmConfig cfg;
    cfg.stacks = 1;
    cfg.channels_per_stack = 1;
    cfg.pseudo_channels_per_channel = 2;
    cfg.bank_groups_per_pseudo_channel = 4;
    cfg.banks_per_group = 1;
    cfg.channel_row_size_bytes = 2048;
    cfg.channel_width_bits = 64;
    cfg.burst_length = 8;
    cfg.pin_rate_Gbps = 6.4;
    cfg.data_rate_per_command_clock = 4;
    cfg.tCCD_S_cycles = 2;
    cfg.tCCD_L_cycles = 4;

    if (cfg.pseudo_channel_width_bits() != 32 ||
        cfg.row_size_bytes() != 1024 || cfg.burst_bytes() != 32 ||
        !close(cfg.channel_bandwidth_GBps(), 51.2) ||
        !close(cfg.pseudo_channel_bandwidth_GBps(), 25.6) ||
        !close(cfg.command_clock_MHz(), 1600.0) ||
        !close(cfg.command_clock_period_ns(), 0.625) ||
        !close(cfg.burst_duration_ns(), 1.25) ||
        !close(cfg.tCCD_S_ns(), 1.25) || !close(cfg.tCCD_L_ns(), 2.5)) {
        throw std::runtime_error(
            "hbm-interface: width/rate/BL/clock derivation is inconsistent");
    }
    std::cout << "  pin_rate=" << fixed(cfg.pin_rate_Gbps, 2)
              << " Gb/s channel_width=" << cfg.channel_width_bits
              << "b pch_width=" << cfg.pseudo_channel_width_bits()
              << "b channel_bw=" << fixed(cfg.channel_bandwidth_GBps(), 2)
              << " GB/s pch_bw=" << fixed(cfg.pseudo_channel_bandwidth_GBps(), 2)
              << " GB/s\n";
    std::cout << "  command_clock=" << fixed(cfg.command_clock_MHz(), 2)
              << " MHz tCK=" << fixed(cfg.command_clock_period_ns(), 3)
              << " ns BL=" << cfg.burst_length
              << " burst=" << cfg.burst_bytes() << " B/"
              << fixed(cfg.burst_duration_ns(), 3) << " ns tCCD="
              << fixed(cfg.tCCD_S_ns(), 3) << "/"
              << fixed(cfg.tCCD_L_ns(), 3) << " ns\n";

    HbmConfig mapping_cfg = cfg;
    mapping_cfg.stacks = 2;
    mapping_cfg.channels_per_stack = 2;
    mapping_cfg.banks_per_group = 2;
    HbmDevice mapping(mapping_cfg);
    for (std::uint32_t stack = 0; stack < mapping_cfg.stacks; ++stack) {
        for (std::uint32_t channel = 0;
             channel < mapping_cfg.channels_per_stack; ++channel) {
            for (std::uint32_t pseudo = 0;
                 pseudo < mapping_cfg.pseudo_channels_per_channel; ++pseudo) {
                for (std::uint32_t group = 0;
                     group < mapping_cfg.bank_groups_per_pseudo_channel; ++group) {
                    HbmAddress original{
                        .stack = stack,
                        .channel = channel,
                        .pseudo_channel = pseudo,
                        .bank_group = group,
                        .bank = group % mapping_cfg.banks_per_group,
                        .row = 7,
                        .offset = 3 * mapping_cfg.burst_bytes() + 5,
                    };
                    const auto decoded = mapping.decode(mapping.encode(original));
                    if (decoded.stack != original.stack ||
                        decoded.channel != original.channel ||
                        decoded.pseudo_channel != original.pseudo_channel ||
                        decoded.bank_group != original.bank_group ||
                        decoded.bank != original.bank || decoded.row != original.row ||
                        decoded.offset != original.offset) {
                        throw std::runtime_error(
                            "hbm-interface: interleaved address mapping did not round-trip");
                    }
                }
            }
        }
    }
    // The reversible swizzle must remain exact even when uint64 addition
    // would wrap and the pseudo-channel count is not a power of two.
    HbmConfig boundary_cfg;
    boundary_cfg.capacity_bytes = std::numeric_limits<std::uint64_t>::max();
    boundary_cfg.stacks = 1;
    boundary_cfg.channels_per_stack = 1;
    boundary_cfg.pseudo_channels_per_channel = 3;
    boundary_cfg.bank_groups_per_pseudo_channel = 1;
    boundary_cfg.banks_per_group = 1;
    boundary_cfg.channel_row_size_bytes = 3;
    boundary_cfg.channel_width_bits = 24;
    boundary_cfg.burst_length = 1;
    boundary_cfg.pin_rate_Gbps = 1.0;
    boundary_cfg.data_rate_per_command_clock = 1;
    boundary_cfg.tCCD_S_cycles = 1;
    boundary_cfg.tCCD_L_cycles = 2;
    HbmDevice boundary_mapping(boundary_cfg);
    constexpr auto kMaxAddress = std::numeric_limits<std::uint64_t>::max();
    for (const auto address : std::array<std::uint64_t, 8>{
             0, 1, 2, 999, kMaxAddress - 1000, kMaxAddress - 3,
             kMaxAddress - 2, kMaxAddress - 1}) {
        if (boundary_mapping.encode(boundary_mapping.decode(address)) != address) {
            throw std::runtime_error(
                "hbm-interface: uint64 boundary mapping did not round-trip");
        }
    }
    // After one complete pseudo-channel stripe, the next sequential burst
    // advances the bank group before the bank-within-group or column.
    HbmDevice stripe(cfg);
    const auto stripe_bytes = cfg.burst_bytes() * cfg.pseudo_channels_per_channel;
    const auto first_group = stripe.decode(0);
    const auto next_group = stripe.decode(stripe_bytes);
    if (first_group.bank_group != 0 || next_group.bank_group != 1 ||
        first_group.bank != 0 || next_group.bank != 0 ||
        first_group.offset != 0 || next_group.offset != 0) {
        throw std::runtime_error(
            "hbm-interface: sequential mapping does not rotate bank groups first");
    }
    // Independent golden vectors pin the raw decode order. Round-trip alone
    // could let encode and decode share the same wrong permutation.
    const auto mapping_group_bytes = mapping_cfg.burst_bytes() *
        mapping_cfg.stacks * mapping_cfg.channels_per_stack *
        mapping_cfg.pseudo_channels_per_channel;
    for (std::uint64_t group = 0; group < 8; ++group) {
        const auto decoded = mapping.decode(group * mapping_group_bytes);
        if (decoded.bank_group != group % 4 || decoded.bank != group / 4 ||
            decoded.row != 0 || decoded.offset != 0) {
            throw std::runtime_error(
                "hbm-interface: raw bank-group/bank golden vector changed");
        }
    }
    const auto column_one = mapping.decode(8 * mapping_group_bytes);
    const auto column_last = mapping.decode(31 * 8 * mapping_group_bytes);
    const auto row_one = mapping.decode(32 * 8 * mapping_group_bytes);
    const auto burst_tail = mapping.decode(mapping_cfg.burst_bytes() - 1);
    if (column_one.row != 0 || column_one.offset != mapping_cfg.burst_bytes() ||
        column_last.row != 0 ||
        column_last.offset != mapping_cfg.row_size_bytes() - mapping_cfg.burst_bytes() ||
        row_one.row != 1 || row_one.offset != 0 ||
        burst_tail.offset != mapping_cfg.burst_bytes() - 1) {
        throw std::runtime_error(
            "hbm-interface: column/row/offset golden boundary vector changed");
    }

    // Commands are issued only on the derived command-clock grid, even when
    // the offered request arrives between edges.
    HbmDevice aligned(cfg);
    const auto aligned_addr = aligned.encode(
        HbmAddress{.pseudo_channel = 0, .bank_group = 0, .bank = 0,
                   .row = 1, .offset = 0});
    const auto off_grid = aligned.issue(request(
        "off-grid", Tier::HBM, Op::Read, 0.1,
        aligned_addr, cfg.burst_bytes()));
    if (!close(off_grid.start_ns, cfg.command_clock_period_ns())) {
        throw std::runtime_error(
            "hbm-interface: an off-grid command was not rounded to the next CK edge");
    }
    HbmDevice exact_edge(cfg);
    const auto exact_completion = exact_edge.issue(request(
        "exact-edge", Tier::HBM, Op::Read, cfg.command_clock_period_ns(),
        aligned_addr, cfg.burst_bytes()));
    if (exact_completion.start_ns != cfg.command_clock_period_ns()) {
        throw std::runtime_error(
            "hbm-interface: an exact command edge did not remain stable");
    }
    const double just_after_edge = std::nextafter(
        cfg.command_clock_period_ns(),
        std::numeric_limits<double>::infinity());
    HbmDevice after_edge(cfg);
    const auto after_completion = after_edge.issue(request(
        "after-edge", Tier::HBM, Op::Read, just_after_edge,
        aligned_addr, cfg.burst_bytes()));
    if (after_completion.start_ns < just_after_edge ||
        !close(after_completion.start_ns, 2.0 * cfg.command_clock_period_ns())) {
        throw std::runtime_error(
            "hbm-interface: a post-edge command moved backward or failed to advance");
    }
    HbmConfig refresh_metric_cfg = cfg;
    refresh_metric_cfg.refresh_enabled = true;
    refresh_metric_cfg.address_mapping_ns = 0.1;
    HbmDevice refresh_metric(refresh_metric_cfg);
    const auto refresh_metric_completion = refresh_metric.issue(request(
        "refresh-metric", Tier::HBM, Op::Read, 0.0,
        aligned_addr, cfg.burst_bytes()));
    if (refresh_metric.stats().refresh_count != 0 ||
        refresh_metric_completion.breakdown.refresh_stall_ns != 0.0 ||
        refresh_metric_completion.breakdown.scheduler_queue_wait_ns <= 0.0) {
        throw std::runtime_error(
            "hbm-interface: CK alignment was misattributed to refresh stall");
    }

    // Same-bank-group row hits obey tCCD_L exactly.
    HbmDevice same_group(cfg);
    const auto same_addr = [&same_group](std::uint64_t offset) {
        return same_group.encode(HbmAddress{
            .pseudo_channel = 0, .bank_group = 0, .bank = 0,
            .row = 3, .offset = offset});
    };
    (void)same_group.issue(request(
        "same-prime", Tier::HBM, Op::Read, 0,
        same_addr(0), cfg.burst_bytes()));
    const auto same_a = same_group.issue(request(
        "same-a", Tier::HBM, Op::Read, 0,
        same_addr(cfg.burst_bytes()), cfg.burst_bytes()));
    const auto same_b = same_group.issue(request(
        "same-b", Tier::HBM, Op::Read, 0,
        same_addr(2 * cfg.burst_bytes()), cfg.burst_bytes()));
    if (same_a.note != "row-hit" || same_b.note != "row-hit" ||
        !close(same_b.start_ns - same_a.start_ns, cfg.tCCD_L_ns())) {
        throw std::runtime_error(
            "hbm-interface: same-bank-group row hits violated tCCD_L");
    }

    // Alternating two already-open bank groups fills the DQ bus at tCCD_S.
    HbmDevice alternating(cfg);
    const auto alt_addr = [&alternating](std::uint32_t group, std::uint64_t offset) {
        return alternating.encode(HbmAddress{
            .pseudo_channel = 0, .bank_group = group, .bank = 0,
            .row = 5, .offset = offset});
    };
    (void)alternating.issue(request(
        "alt-prime-0", Tier::HBM, Op::Read, 0,
        alt_addr(0, 0), cfg.burst_bytes()));
    (void)alternating.issue(request(
        "alt-prime-1", Tier::HBM, Op::Read, 0,
        alt_addr(1, 0), cfg.burst_bytes()));
    const auto alt_a = alternating.issue(request(
        "alt-a", Tier::HBM, Op::Read, 0,
        alt_addr(0, cfg.burst_bytes()), cfg.burst_bytes()));
    const auto alt_b = alternating.issue(request(
        "alt-b", Tier::HBM, Op::Read, 0,
        alt_addr(1, cfg.burst_bytes()), cfg.burst_bytes()));
    const auto alt_c = alternating.issue(request(
        "alt-c", Tier::HBM, Op::Read, 0,
        alt_addr(0, 2 * cfg.burst_bytes()), cfg.burst_bytes()));
    if (alt_a.note != "row-hit" || alt_b.note != "row-hit" ||
        alt_c.note != "row-hit" ||
        !close(alt_b.start_ns - alt_a.start_ns, cfg.tCCD_S_ns()) ||
        !close(alt_c.start_ns - alt_b.start_ns, cfg.tCCD_S_ns())) {
        throw std::runtime_error(
            "hbm-interface: alternating bank groups did not sustain tCCD_S");
    }

    // Long trains make clock drift or an occasional extra-cycle insertion
    // visible instead of checking only two adjacent commands.
    HbmDevice same_train(cfg);
    const auto train_same_addr = same_train.encode(HbmAddress{
        .pseudo_channel = 0, .bank_group = 0, .bank = 0, .row = 9, .offset = 0});
    (void)same_train.issue(request(
        "train-same-prime", Tier::HBM, Op::Read, 0,
        train_same_addr, cfg.burst_bytes()));
    double previous_issue_ns = -1.0;
    for (std::uint64_t i = 0; i < 512; ++i) {
        const auto completion = same_train.issue(request(
            "train-same", Tier::HBM, Op::Read, 0,
            train_same_addr, cfg.burst_bytes()));
        if (cfg.command_aligned_time_ns(completion.start_ns) != completion.start_ns ||
            (previous_issue_ns >= 0.0 &&
             !close(completion.start_ns - previous_issue_ns, cfg.tCCD_L_ns()))) {
            throw std::runtime_error(
                "hbm-interface: long same-group train drifted from tCCD_L grid");
        }
        previous_issue_ns = completion.start_ns;
    }
    HbmDevice alternating_train(cfg);
    const auto train_addr = [&alternating_train](std::uint32_t group) {
        return alternating_train.encode(HbmAddress{
            .pseudo_channel = 0, .bank_group = group, .bank = 0,
            .row = 11, .offset = 0});
    };
    (void)alternating_train.issue(request(
        "train-alt-prime-0", Tier::HBM, Op::Read, 0,
        train_addr(0), cfg.burst_bytes()));
    (void)alternating_train.issue(request(
        "train-alt-prime-1", Tier::HBM, Op::Read, 0,
        train_addr(1), cfg.burst_bytes()));
    previous_issue_ns = -1.0;
    for (std::uint64_t i = 0; i < 512; ++i) {
        const auto completion = alternating_train.issue(request(
            "train-alt", Tier::HBM, Op::Read, 0,
            train_addr(static_cast<std::uint32_t>(i % 2)), cfg.burst_bytes()));
        if (cfg.command_aligned_time_ns(completion.start_ns) != completion.start_ns ||
            (previous_issue_ns >= 0.0 &&
             !close(completion.start_ns - previous_issue_ns, cfg.tCCD_S_ns()))) {
            throw std::runtime_error(
                "hbm-interface: long alternating-group train drifted from tCCD_S grid");
        }
        previous_issue_ns = completion.start_ns;
    }

    // End-to-end sequential traffic must be able to approach the derived
    // interface peak. This catches both a reintroduced tBL override and an
    // address map that strands traffic behind tCCD_L.
    HbmConfig saturation_cfg = cfg;
    saturation_cfg.stacks = 8;
    saturation_cfg.channels_per_stack = 32;
    saturation_cfg.bank_groups_per_pseudo_channel = 16;
    saturation_cfg.banks_per_group = 4;
    HbmDevice distribution(saturation_cfg);
    constexpr std::uint64_t kDistributionSamples = 16384;
    for (const auto stride : std::array<std::uint64_t, 4>{
             64, 4096, 65536, 1048576}) {
        std::set<std::uint64_t> pseudo_channels;
        std::set<std::uint64_t> banks;
        for (std::uint64_t i = 0; i < kDistributionSamples; ++i) {
            const auto decoded = distribution.decode(i * stride);
            pseudo_channels.insert(
                (static_cast<std::uint64_t>(decoded.stack) *
                     saturation_cfg.channels_per_stack + decoded.channel) *
                    saturation_cfg.pseudo_channels_per_channel +
                decoded.pseudo_channel);
            banks.insert(
                static_cast<std::uint64_t>(decoded.bank) *
                    saturation_cfg.bank_groups_per_pseudo_channel +
                decoded.bank_group);
        }
        if (pseudo_channels.size() < 3 * 512 / 4 || banks.size() < 48) {
            throw std::runtime_error(
                "hbm-interface: common power-of-two stride aliases HBM resources");
        }
    }
    HbmDevice saturation(saturation_cfg);
    // Long enough to amortize ACT/startup transients across all 512 PCs; this
    // is an end-to-end sustained-bandwidth guard, not a cold-burst latency test.
    constexpr std::uint64_t kSaturationOps = 16384;
    constexpr std::uint64_t kSaturationBytes = 4096;
    for (std::uint64_t i = 0; i < kSaturationOps; ++i) {
        (void)saturation.issue(request(
            "saturation", Tier::HBM, Op::Read, 0,
            i * kSaturationBytes, kSaturationBytes));
    }
    const auto& saturation_stats = saturation.stats();
    const double achieved_GBps =
        static_cast<double>(saturation_stats.read_bytes) /
        saturation_stats.active_span_ns();
    const double peak_GBps = saturation_cfg.stacks *
        saturation_cfg.channels_per_stack * saturation_cfg.channel_bandwidth_GBps();
    if (achieved_GBps < 0.85 * peak_GBps || achieved_GBps > 1.001 * peak_GBps) {
        throw std::runtime_error(
            "hbm-interface: sequential traffic cannot reach the derived interface peak");
    }
    std::cout << "  sequential=" << fixed(achieved_GBps, 2)
              << " GB/s derived_peak=" << fixed(peak_GBps, 2)
              << " GB/s efficiency=" << fixed(achieved_GBps / peak_GBps * 100.0, 2)
              << "%\n";

    // Metamorphic checks: width changes bytes and bandwidth together; rate
    // changes bandwidth and time together; BL changes bytes and duration by
    // the same factor; pseudo-channel partitioning preserves channel peak.
    auto wide = cfg;
    wide.channel_width_bits = 128;
    if (wide.burst_bytes() != 64 ||
        !close(wide.channel_bandwidth_GBps(), 102.4) ||
        !close(wide.burst_duration_ns(), cfg.burst_duration_ns())) {
        throw std::runtime_error("hbm-interface: width metamorphic relation failed");
    }
    auto fast = cfg;
    fast.pin_rate_Gbps = 12.8;
    if (!close(fast.channel_bandwidth_GBps(), 102.4) ||
        !close(fast.burst_duration_ns(), 0.625) ||
        !close(fast.tCCD_S_ns(), 0.625)) {
        throw std::runtime_error("hbm-interface: rate metamorphic relation failed");
    }
    auto long_burst = cfg;
    long_burst.burst_length = 16;
    long_burst.tCCD_S_cycles = 4;
    long_burst.tCCD_L_cycles = 8;
    if (long_burst.burst_bytes() != 64 ||
        !close(long_burst.burst_duration_ns(), 2.5) ||
        !close(long_burst.channel_bandwidth_GBps(), cfg.channel_bandwidth_GBps())) {
        throw std::runtime_error("hbm-interface: BL metamorphic relation failed");
    }
    auto unsplit = cfg;
    unsplit.pseudo_channels_per_channel = 1;
    if (unsplit.burst_bytes() != 64 ||
        !close(unsplit.channel_bandwidth_GBps(), cfg.channel_bandwidth_GBps())) {
        throw std::runtime_error(
            "hbm-interface: pseudo-channel partition changed channel peak");
    }

    int rejected = 0;
    const auto must_reject = [&rejected](HbmConfig invalid) {
        try {
            (void)HbmDevice(invalid);
        } catch (const std::runtime_error&) {
            rejected++;
        }
    };
    auto invalid = cfg;
    invalid.channel_width_bits = 65;
    must_reject(invalid);
    invalid = cfg;
    invalid.pseudo_channels_per_channel = 3;
    must_reject(invalid);
    invalid = cfg;
    invalid.burst_length = 7;
    must_reject(invalid);
    invalid = cfg;
    invalid.channel_row_size_bytes = 2000;
    must_reject(invalid);
    invalid = cfg;
    invalid.tCCD_S_cycles = 1;
    must_reject(invalid);
    invalid = cfg;
    invalid.pin_rate_Gbps = std::numeric_limits<double>::denorm_min();
    must_reject(invalid);
    if (rejected != 6) {
        throw std::runtime_error(
            "hbm-interface: invalid interface profiles did not all fail closed");
    }

    std::cout << "  hbm_interface_derivation_ok=yes\n";
    std::cout << "  hbm_address_interleave_ok=yes\n";
    std::cout << "  hbm_address_golden_vectors_ok=yes\n";
    std::cout << "  hbm_uint64_mapping_ok=yes\n";
    std::cout << "  hbm_command_clock_alignment_ok=yes\n";
    std::cout << "  hbm_tccd_analytical_ok=yes\n";
    std::cout << "  hbm_long_train_clock_stability_ok=yes\n";
    std::cout << "  hbm_stride_distribution_ok=yes\n";
    std::cout << "  hbm_sequential_peak_reachable=yes\n";
    std::cout << "  hbm_interface_metamorphic_ok=yes\n";
    std::cout << "  hbm_invalid_profile_rejected=yes\n";
}

void probe_hbm_row() {
    std::cout << "\n== HBM row-buffer probe ==\n";
    HbmConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.pseudo_channels_per_channel = 1;
    cfg.bank_groups_per_pseudo_channel = 1;
    cfg.banks_per_group = 1;
    HbmDevice hbm(cfg);
    const auto row0 = hbm.encode(HbmAddress{.row = 7, .offset = 0});
    const auto row0_next = hbm.encode(HbmAddress{.row = 7, .offset = 128});
    const auto row1 = hbm.encode(HbmAddress{.row = 8, .offset = 0});

    std::vector<PhysicalCompletion> rows;
    rows.push_back(hbm.issue(request("first-row-miss", Tier::HBM, Op::Read, 0, row0, 128)));
    rows.push_back(hbm.issue(request("same-row-hit", Tier::HBM, Op::Read, 0, row0_next, 128)));
    rows.push_back(hbm.issue(request("new-row-conflict", Tier::HBM, Op::Read, 0, row1, 128)));
    print_rows(rows);
    print_hbm_stats(hbm);
    std::cout << "  expectation: same-row-hit has no ACT/PRE; new-row-conflict pays PRE+ACT.\n";
}

void probe_hbm_boundaries() {
    std::cout << "\n== HBM request-boundary probe ==\n";
    HbmConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.pseudo_channels_per_channel = 2;
    cfg.bank_groups_per_pseudo_channel = 1;
    cfg.banks_per_group = 1;
    cfg.channel_row_size_bytes = 128;
    cfg.capacity_bytes = 128;
    HbmDevice hbm(cfg);

    // [31, 97) touches four 32-B bursts across two independently mapped
    // pseudo-channels: 1 B + 32 B + 32 B + 1 B.
    auto unaligned = request("cross-burst-row-map", Tier::HBM, Op::Read, 0, 31, 66);
    unaligned.trace.mode = TraceMode::Full;
    const auto completion = hbm.pump(hbm.enqueue(unaligned));
    const auto& stats = hbm.stats();
    const auto burst_spans = static_cast<std::size_t>(std::count_if(
        completion.spans.begin(), completion.spans.end(), [](const auto& span) {
            return span.name == "read_burst";
        }));
    print_rows({completion});
    print_hbm_stats(hbm);
    std::cout << "  logical/physical bytes = " << completion.logical_bytes << "/"
              << completion.physical_bytes << ", burst spans = " << burst_spans
              << ", active pch = " << stats.active_pseudo_channels << "\n";
    if (completion.logical_bytes != 66 || completion.physical_bytes != 128 ||
        burst_spans != 4) {
        throw std::runtime_error(
            "hbm-boundaries: an unaligned range must charge every touched burst");
    }
    if (stats.activations != 2 || stats.row_misses != 2 || stats.row_hits != 2 ||
        stats.active_pseudo_channels != 2) {
        throw std::runtime_error(
            "hbm-boundaries: each mapped row must make independent row/channel decisions");
    }
    if (completion.start_ns > completion.finish_ns ||
        completion.finish_ns != stats.finish_ns) {
        throw std::runtime_error(
            "hbm-boundaries: parent completion must cover every internal burst");
    }
    const double expected_transfer_work_ns =
        static_cast<double>(burst_spans) * cfg.burst_duration_ns();
    if (std::abs(stats.stage_work.address_mapping_ns -
                 static_cast<double>(burst_spans) * cfg.address_mapping_ns) > 1e-9 ||
        std::abs(stats.stage_work.channel_transfer_ns -
                 expected_transfer_work_ns) > 1e-9 ||
        std::abs(stats.bus_busy_ns - expected_transfer_work_ns) > 1e-9 ||
        !(stats.stage_work.total_work_ns() >
          completion.breakdown.total_work_ns())) {
        throw std::runtime_error(
            "hbm-boundaries: canonical stage work did not aggregate every burst child");
    }
    HbmDevice no_trace_hbm(cfg);
    auto no_trace_request = unaligned;
    no_trace_request.trace.mode = TraceMode::Off;
    (void)no_trace_hbm.issue(no_trace_request);
    if (!(no_trace_hbm.stats().stage_work == stats.stage_work)) {
        throw std::runtime_error(
            "hbm-boundaries: trace collection changed canonical stage work");
    }
    bool range_rejected = false;
    try {
        (void)hbm.issue(request(
            "out-of-capacity", Tier::HBM, Op::Read, 0, 96, 33));
    } catch (const std::runtime_error&) {
        range_rejected = true;
    }
    if (!range_rejected) {
        throw std::runtime_error("hbm-boundaries: capacity overflow must fail closed");
    }
    std::cout << "  boundary_split_ok=yes\n";
    std::cout << "  hbm_capacity_bounds_ok=yes\n";
    std::cout << "  hbm_stage_work_child_aggregation_ok=yes\n";
    std::cout << "  hbm_stage_work_trace_independent_ok=yes\n";
}

HbmStreamResult run_hbm_stream(std::uint32_t channels) {
    HbmConfig cfg;
    cfg.channels_per_stack = channels;
    cfg.pseudo_channels_per_channel = 1;
    cfg.bank_groups_per_pseudo_channel = 4;
    cfg.banks_per_group = 4;
    HbmDevice hbm(cfg);
    for (std::uint32_t i = 0; i < 256; ++i) {
        HbmAddress addr;
        addr.channel = i % channels;
        addr.bank_group = (i / channels) % cfg.bank_groups_per_pseudo_channel;
        addr.bank = (i / (channels * cfg.bank_groups_per_pseudo_channel)) % cfg.banks_per_group;
        addr.row = i / (channels * cfg.bank_groups_per_pseudo_channel * cfg.banks_per_group);
        const auto physical = hbm.encode(addr);
        (void)hbm.issue(request(
            "stream", Tier::HBM, Op::Read, 0, physical, cfg.row_size_bytes()));
    }
    return HbmStreamResult{
        .finish_ns = hbm.stats().finish_ns,
        .stats = hbm.stats(),
    };
}

void probe_hbm_channels() {
    std::cout << "\n== HBM channel scaling probe ==\n";
    const auto one = run_hbm_stream(1);
    const auto four = run_hbm_stream(4);
    std::cout << "  1 channel finish_ns = " << fixed(one.finish_ns)
              << " bus_parallelism=" << fixed(one.stats.bus_parallelism(), 2)
              << " active_pch=" << one.stats.active_pseudo_channels << "/"
              << one.stats.pseudo_channels << "\n";
    std::cout << "  4 channel finish_ns = " << fixed(four.finish_ns)
              << " bus_parallelism=" << fixed(four.stats.bus_parallelism(), 2)
              << " active_pch=" << four.stats.active_pseudo_channels << "/"
              << four.stats.pseudo_channels << "\n";
    std::cout << "  speedup             = " << fixed(one.finish_ns / four.finish_ns, 2) << "x\n";
    std::cout << "  expectation: more independent channels reduce stream finish time.\n";
}

double run_hbm_refresh(bool refresh) {
    HbmConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.pseudo_channels_per_channel = 1;
    cfg.bank_groups_per_pseudo_channel = 1;
    cfg.banks_per_group = 4;
    cfg.refresh_enabled = refresh;
    cfg.tREFI_ns = 200.0;
    cfg.tRFC_ns = 80.0;
    HbmDevice hbm(cfg);
    for (std::uint32_t i = 0; i < 16; ++i) {
        HbmAddress addr;
        addr.bank = i % cfg.banks_per_group;
        addr.row = i / cfg.banks_per_group;
        (void)hbm.issue(request("refresh-stream", Tier::HBM, Op::Read, i * 10.0, hbm.encode(addr), 2048));
    }
    print_hbm_stats(hbm);
    return hbm.stats().finish_ns;
}

void probe_hbm_refresh() {
    std::cout << "\n== HBM refresh probe ==\n";
    std::cout << "  refresh off:\n";
    const double off = run_hbm_refresh(false);
    std::cout << "  refresh on:\n";
    const double on = run_hbm_refresh(true);
    std::cout << "  off_finish_ns = " << fixed(off) << "\n";
    std::cout << "  on_finish_ns  = " << fixed(on) << "\n";
    if (on <= off) {
        throw std::runtime_error("hbm-refresh: refresh must increase stream finish time");
    }

    // A request after several idle refresh periods must not see the row that
    // was open before those refreshes. This specifically covers the formerly
    // skipped `next_refresh + tRFC <= arrival` path.
    HbmConfig state_cfg;
    state_cfg.channels_per_stack = 1;
    state_cfg.pseudo_channels_per_channel = 1;
    state_cfg.bank_groups_per_pseudo_channel = 1;
    state_cfg.banks_per_group = 1;
    state_cfg.refresh_enabled = true;
    state_cfg.tREFI_ns = 100.0;
    state_cfg.tRFC_ns = 20.0;
    state_cfg.tRREFD_ns = 5.0;
    HbmDevice state(state_cfg);
    const auto row = state.encode(HbmAddress{.row = 9, .offset = 0});
    (void)state.issue(request("open-before-idle-refresh", Tier::HBM, Op::Read, 0, row, 32));
    const auto after = state.issue(request(
        "same-row-after-three-refreshes", Tier::HBM, Op::Read, 350, row, 32));
    const auto& state_stats = state.stats();
    std::cout << "  idle refreshes = " << state_stats.refresh_count
              << ", second note = " << after.note
              << ", activations = " << state_stats.activations << "\n";
    if (state_stats.refresh_count != 3 || state_stats.activations != 2 ||
        state_stats.row_misses != 2 || state_stats.row_hits != 0 ||
        after.note != "row-miss") {
        throw std::runtime_error(
            "hbm-refresh: elapsed all-bank refreshes must close every open row");
    }
    std::cout << "  refresh_closes_idle_rows=yes\n";
    std::cout << "  expectation: refresh adds stalls, advances every elapsed window,"
              << " and all-bank refresh always closes rows.\n";
}

void probe_hbm_turnaround() {
    std::cout << "\n== HBM read/write turnaround probe ==\n";
    HbmConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.pseudo_channels_per_channel = 1;
    cfg.bank_groups_per_pseudo_channel = 1;
    cfg.banks_per_group = 1;
    const HbmAddress base{.row = 3, .offset = 0};
    const HbmAddress same_row_next{.row = 3, .offset = 128};

    std::cout << "  read -> write on an open row:\n";
    HbmDevice read_then_write(cfg);
    const auto rw0 = read_then_write.encode(base);
    const auto rw1 = read_then_write.encode(same_row_next);
    std::vector<PhysicalCompletion> rw_rows;
    rw_rows.push_back(read_then_write.issue(request("read-opens-row", Tier::HBM, Op::Read, 0, rw0, 128)));
    rw_rows.push_back(read_then_write.issue(request("same-row-write-after-read", Tier::HBM, Op::Write, 0, rw1, 128)));
    print_rows(rw_rows);
    print_hbm_stats(read_then_write);

    std::cout << "  write -> read on an open row:\n";
    HbmDevice write_then_read(cfg);
    const auto wr0 = write_then_read.encode(base);
    const auto wr1 = write_then_read.encode(same_row_next);
    std::vector<PhysicalCompletion> wr_rows;
    wr_rows.push_back(write_then_read.issue(request("write-opens-row", Tier::HBM, Op::Write, 0, wr0, 128)));
    wr_rows.push_back(write_then_read.issue(request("same-row-read-after-write", Tier::HBM, Op::Read, 0, wr1, 128)));
    print_rows(wr_rows);
    print_hbm_stats(write_then_read);

    // The transfer duration has one source of truth. Read/write turnaround
    // gates must cover the derived BL8 data burst without a legacy tBL
    // override or burst-extension repair path.
    HbmDevice derived_write_read(cfg);
    const auto derived_wr_addr0 = derived_write_read.encode(
        HbmAddress{.row = 5, .offset = 0});
    const auto derived_wr_addr1 = derived_write_read.encode(
        HbmAddress{.row = 5, .offset = cfg.burst_bytes()});
    const auto derived_write = derived_write_read.issue(request(
        "derived-write", Tier::HBM, Op::Write, 0,
        derived_wr_addr0, cfg.burst_bytes()));
    const auto derived_read = derived_write_read.issue(request(
        "read-after-derived-write", Tier::HBM, Op::Read, 0,
        derived_wr_addr1, cfg.burst_bytes()));
    if (derived_read.start_ns < derived_write.finish_ns) {
        throw std::runtime_error(
            "hbm-turnaround: WR->RD gate released before the derived burst finished");
    }

    HbmDevice derived_read_write(cfg);
    const auto derived_rw_addr0 = derived_read_write.encode(
        HbmAddress{.row = 6, .offset = 0});
    const auto derived_rw_addr1 = derived_read_write.encode(
        HbmAddress{.row = 6, .offset = cfg.burst_bytes()});
    const auto first_read = derived_read_write.issue(request(
        "derived-read", Tier::HBM, Op::Read, 0,
        derived_rw_addr0, cfg.burst_bytes()));
    const auto next_write = derived_read_write.issue(request(
        "write-after-derived-read", Tier::HBM, Op::Write, 0,
        derived_rw_addr1, cfg.burst_bytes()));
    const double read_to_write_floor = first_read.start_ns + cfg.tRTW_ns;
    if (next_write.start_ns + 1e-9 < read_to_write_floor) {
        throw std::runtime_error(
            "hbm-turnaround: RD->WR gate released before tRTW");
    }

    HbmDevice derived_write_conflict(cfg);
    const auto conflict_addr0 = derived_write_conflict.encode(
        HbmAddress{.row = 7, .offset = 0});
    const auto conflict_addr1 = derived_write_conflict.encode(
        HbmAddress{.row = 8, .offset = 0});
    const auto conflict_write = derived_write_conflict.issue(request(
        "derived-write-before-conflict", Tier::HBM, Op::Write, 0,
        conflict_addr0, cfg.burst_bytes()));
    const auto conflict_read = derived_write_conflict.issue(request(
        "conflict-after-derived-write", Tier::HBM, Op::Read, 0,
        conflict_addr1, cfg.burst_bytes()));
    if (conflict_read.start_ns < conflict_write.finish_ns) {
        throw std::runtime_error(
            "hbm-turnaround: PRE gate released before the derived write finished");
    }
    std::cout << "  derived burst WR->RD starts " << fixed(derived_read.start_ns)
              << " ns after write finishes " << fixed(derived_write.finish_ns)
              << " ns; RD->WR starts " << fixed(next_write.start_ns) << " ns\n";
    std::cout << "  derived_burst_gates_ok=yes\n";
    std::cout << "  expectation: second commands are row hits, but wait on tRTW or tWTR_L.\n";
}

void probe_hbf_basic() {
    std::cout << "\n== HBF read/program/erase probe ==\n";
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    HbfDevice hbf(cfg);
    const auto logical_page = std::uint64_t{0};
    const auto physical_block = hbf.encode(HbfAddress{.block = 0, .page = 0});
    std::vector<PhysicalCompletion> rows;
    rows.push_back(hbf.issue(request("logical-page-program", Tier::HBF, Op::Write, 0,
        logical_page, 512)));
    rows.push_back(hbf.issue(request("logical-page-read", Tier::HBF, Op::Read, 0,
        logical_page, 512)));
    rows.push_back(hbf.issue(request("physical-block-erase", Tier::HBF, Op::Erase, 0,
        physical_block, 0, AddressSpace::Physical)));
    hbfsim::physical::Breakdown expected_stage_work;
    for (const auto& row : rows) {
        expected_stage_work += row.breakdown;
    }
    const auto& stats = hbf.stats();
    const auto work_close = [](double lhs, double rhs) {
        return std::abs(lhs - rhs) <=
            1e-9 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
    };
    if (!work_close(
            stats.stage_work.total_work_ns(),
            expected_stage_work.total_work_ns()) ||
        stats.logic_ingress_busy_ns <= 0.0 || stats.tsv_busy_ns <= 0.0 ||
        stats.sram_busy_ns <= 0.0 || stats.flash_source_queue_busy_ns <= 0.0 ||
        stats.channel_command_busy_ns <= 0.0 ||
        stats.channel_data_busy_ns <= 0.0 ||
        stats.logic_ingress_resources != cfg.stacks ||
        stats.tsv_resources != cfg.stacks || stats.sram_resources != cfg.stacks ||
        stats.flash_source_queue_resources != 4 ||
        stats.channel_command_resources != cfg.channels_per_stack * cfg.stacks ||
        stats.channel_data_resources != cfg.channels_per_stack * cfg.stacks) {
        throw std::runtime_error(
            "hbf-basic: canonical stage work/resource capacity accounting diverged");
    }
    HbfDevice traced_hbf(cfg);
    HbfDevice untraced_hbf(cfg);
    auto traced_read = request(
        "stage-work-traced", Tier::HBF, Op::Read, 0.0, 0,
        cfg.page_size_bytes, AddressSpace::Physical);
    traced_read.trace.mode = TraceMode::Full;
    auto untraced_read = traced_read;
    untraced_read.id = "stage-work-untraced";
    untraced_read.trace.mode = TraceMode::Off;
    (void)traced_hbf.issue(traced_read);
    (void)untraced_hbf.issue(untraced_read);
    if (!(traced_hbf.stats().stage_work == untraced_hbf.stats().stage_work) ||
        traced_hbf.stats().tsv_busy_ns != untraced_hbf.stats().tsv_busy_ns ||
        traced_hbf.stats().sram_busy_ns != untraced_hbf.stats().sram_busy_ns) {
        throw std::runtime_error(
            "hbf-basic: trace collection changed canonical time accounting");
    }
    print_rows(rows);
    print_hbf_stats(hbf);
    std::cout << "  hbf_stage_work_exact_once_ok=yes\n";
    std::cout << "  hbf_resource_busy_capacity_ok=yes\n";
    std::cout << "  hbf_time_accounting_trace_independent_ok=yes\n";
    std::cout << "  expectation: program >> read, erase is block-level and mapping is explicit.\n";
}

void probe_hbf_page_read_admission() {
    std::cout << "\n== HBF page-read admission probe ==\n";
    HbfConfig serialized_cfg;
    serialized_cfg.channels_per_stack = 1;
    serialized_cfg.dies_per_channel = 1;
    serialized_cfg.planes_per_die = 1;
    serialized_cfg.read_buffer_pages = 0;
    serialized_cfg.page_read_queue_depth_per_stack = 1;
    HbfDevice serialized(serialized_cfg);
    const auto physical = serialized.encode(
        HbfAddress{.stack = 0, .channel = 0, .die = 0, .plane = 0,
                   .block = 0, .page = 0});
    const auto one_credit = serialized.issue(request(
        "page-read-depth-1",
        Tier::HBF,
        Op::Read,
        0.0,
        physical,
        2 * serialized_cfg.page_size_bytes,
        AddressSpace::Physical));
    const auto& one_stats = serialized.stats();
    if (one_stats.page_read_admission_events != 2 ||
        one_stats.page_read_admission_waited_pages != 1 ||
        one_stats.page_read_admission_wait_ns <= 0.0 ||
        one_stats.page_read_admission_max_wait_ns <= 0.0 ||
        one_credit.breakdown.scheduler_queue_wait_ns <
            one_stats.page_read_admission_wait_ns) {
        throw std::runtime_error(
            "hbf-page-read-admission: one-credit split did not backpressure");
    }

    auto parallel_cfg = serialized_cfg;
    parallel_cfg.page_read_queue_depth_per_stack = 2;
    HbfDevice parallel(parallel_cfg);
    (void)parallel.issue(request(
        "page-read-depth-2",
        Tier::HBF,
        Op::Read,
        0.0,
        physical,
        2 * parallel_cfg.page_size_bytes,
        AddressSpace::Physical));
    const auto& parallel_stats = parallel.stats();
    if (parallel_stats.page_read_admission_events != 2 ||
        parallel_stats.page_read_admission_waited_pages != 0 ||
        parallel_stats.page_read_admission_wait_ns != 0.0) {
        throw std::runtime_error(
            "hbf-page-read-admission: available page credits falsely waited");
    }
    print_hbf_stats(serialized);
    std::cout << "  page_read_credit_backpressure_ok=yes\n";
}

void probe_hbf_planes() {
    std::cout << "\n== HBF plane parallelism probe ==\n";
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 2;
    cfg.hb_io_bandwidth_GBps = 4096.0;
    cfg.tsv_bandwidth_GBps = 4096.0;
    cfg.channel_bandwidth_GBps = 4096.0;

    HbfDevice same(cfg);
    const auto p0 = same.encode(HbfAddress{.plane = 0, .block = 0, .page = 0});
    auto s0 = same.issue(request("same-plane-a", Tier::HBF, Op::Read, 0,
        p0, 2048, AddressSpace::Physical));
    auto s1 = same.issue(request("same-plane-b", Tier::HBF, Op::Read, 0,
        p0, 2048, AddressSpace::Physical));
    print_rows({s0, s1});
    print_hbf_stats(same);

    HbfDevice diff(cfg);
    const auto d0 = diff.encode(HbfAddress{.plane = 0, .block = 0, .page = 0});
    const auto d1 = diff.encode(HbfAddress{.plane = 1, .block = 0, .page = 0});
    auto c0 = diff.issue(request("diff-plane-a", Tier::HBF, Op::Read, 0,
        d0, 2048, AddressSpace::Physical));
    auto c1 = diff.issue(request("diff-plane-b", Tier::HBF, Op::Read, 0,
        d1, 2048, AddressSpace::Physical));
    print_rows({c0, c1});
    print_hbf_stats(diff);
    std::cout << "  expectation: same-plane-b queues behind media; diff-plane-b can overlap media.\n";
}

HbfParallelResult run_hbf_parallel_write_stream(std::uint32_t planes) {
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = planes;
    cfg.blocks_per_plane = 16;
    cfg.pages_per_block = 64;
    cfg.page_size_bytes = 512;
    cfg.hb_io_bandwidth_GBps = 4096.0;
    cfg.tsv_bandwidth_GBps = 4096.0;
    cfg.channel_bandwidth_GBps = 4096.0;
    cfg.logic_sram_bandwidth_GBps = 4096.0;
    cfg.page_buffer_bandwidth_GBps = 4096.0;
    cfg.mapping_entries_per_page = 1024;
    cfg.write_coalescing_enabled = false;
    HbfDevice hbf(cfg);
    for (std::uint32_t i = 0; i < 64; ++i) {
        (void)hbf.issue(request(
            "parallel-write",
            Tier::HBF,
            Op::Write,
            0,
            static_cast<std::uint64_t>(i) * cfg.page_size_bytes,
            cfg.page_size_bytes));
    }
    return HbfParallelResult{
        .finish_ns = hbf.stats().finish_ns,
        .stats = hbf.stats(),
    };
}

void probe_hbf_parallelism() {
    std::cout << "\n== HBF media parallelism probe ==\n";
    const auto one = run_hbf_parallel_write_stream(1);
    const auto four = run_hbf_parallel_write_stream(4);
    std::cout << "  1 plane finish_ns = " << fixed(one.finish_ns)
              << " media_parallelism=" << fixed(one.stats.media_parallelism(), 2)
              << " active_planes=" << one.stats.active_planes << "/"
              << one.stats.planes
              << " plane_media_skew=" << fixed(one.stats.plane_media_skew(), 2)
              << "\n";
    std::cout << "  4 plane finish_ns = " << fixed(four.finish_ns)
              << " media_parallelism=" << fixed(four.stats.media_parallelism(), 2)
              << " active_planes=" << four.stats.active_planes << "/"
              << four.stats.planes
              << " plane_media_skew=" << fixed(four.stats.plane_media_skew(), 2)
              << "\n";
    std::cout << "  speedup           = " << fixed(one.finish_ns / four.finish_ns, 2) << "x\n";
    std::cout << "  expectation: round-robin allocation spreads independent page programs across planes.\n";
}

HbfParallelResult run_hbf_media_lane_read_stream(std::uint32_t lanes) {
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.blocks_per_plane = 1;
    cfg.pages_per_block = 128;
    cfg.page_size_bytes = 512;
    cfg.media_lanes_per_plane = lanes;
    cfg.subarrays_per_plane = 64;
    cfg.page_buffer_banks_per_plane = 64;
    cfg.hb_io_bandwidth_GBps = 4096.0;
    cfg.tsv_bandwidth_GBps = 4096.0;
    cfg.channel_bandwidth_GBps = 4096.0;
    cfg.logic_sram_bandwidth_GBps = 4096.0;
    cfg.page_buffer_bandwidth_GBps = 4096.0;
    cfg.media_lane_bandwidth_GBps = 0.128;
    cfg.t_read_page_ns = 100.0;
    cfg.ecc_decode_latency_ns = 1.0;
    cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    cfg.logic_scheduler_issue_ns = 1.0;
    cfg.flash_tsu_issue_ns = 1.0;
    HbfDevice hbf(cfg);
    for (std::uint32_t i = 0; i < 64; ++i) {
        const auto addr = hbf.encode(HbfAddress{.block = 0, .page = i});
        (void)hbf.issue(request(
            "lane-read",
            Tier::HBF,
            Op::Read,
            0,
            addr,
            cfg.page_size_bytes,
            AddressSpace::Physical));
    }
    return HbfParallelResult{
        .finish_ns = hbf.stats().finish_ns,
        .stats = hbf.stats(),
    };
}

void probe_hbf_media_lanes() {
    std::cout << "\n== HBF media-lane read parallelism probe ==\n";
    const auto one = run_hbf_media_lane_read_stream(1);
    const auto four = run_hbf_media_lane_read_stream(4);
    std::cout << "  1 lane finish_ns = " << fixed(one.finish_ns)
              << " read_lane_parallelism=" << fixed(one.stats.read_lane_parallelism(), 2)
              << " active_media_lanes=" << one.stats.active_media_lanes << "/"
              << one.stats.media_lanes
              << " media_lane_skew=" << fixed(one.stats.media_lane_skew(), 2)
              << "\n";
    std::cout << "  4 lane finish_ns = " << fixed(four.finish_ns)
              << " read_lane_parallelism=" << fixed(four.stats.read_lane_parallelism(), 2)
              << " active_media_lanes=" << four.stats.active_media_lanes << "/"
              << four.stats.media_lanes
              << " media_lane_skew=" << fixed(four.stats.media_lane_skew(), 2)
              << "\n";
    std::cout << "  speedup          = " << fixed(one.finish_ns / four.finish_ns, 2) << "x\n";
    std::cout << "  expectation: same-plane reads sense in independent subarrays, then overlap across internal media lanes.\n";
}

HbfParallelResult run_hbf_page_buffer_bank_read_stream(std::uint32_t banks) {
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.blocks_per_plane = 1;
    cfg.pages_per_block = 128;
    cfg.page_size_bytes = 512;
    cfg.media_lanes_per_plane = 64;
    cfg.subarrays_per_plane = 64;
    cfg.page_buffer_banks_per_plane = banks;
    cfg.hb_io_bandwidth_GBps = 4096.0;
    cfg.tsv_bandwidth_GBps = 4096.0;
    cfg.channel_bandwidth_GBps = 4096.0;
    cfg.logic_sram_bandwidth_GBps = 4096.0;
    cfg.media_lane_bandwidth_GBps = 4096.0;
    cfg.page_buffer_bandwidth_GBps = 0.128;
    cfg.t_read_page_ns = 100.0;
    cfg.ecc_decode_latency_ns = 1.0;
    cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    cfg.logic_scheduler_issue_ns = 1.0;
    cfg.flash_tsu_issue_ns = 1.0;
    HbfDevice hbf(cfg);
    for (std::uint32_t i = 0; i < 64; ++i) {
        const auto addr = hbf.encode(HbfAddress{.block = 0, .page = i});
        (void)hbf.issue(request(
            "page-buffer-bank-read",
            Tier::HBF,
            Op::Read,
            0,
            addr,
            cfg.page_size_bytes,
            AddressSpace::Physical));
    }
    return HbfParallelResult{
        .finish_ns = hbf.stats().finish_ns,
        .stats = hbf.stats(),
    };
}

void probe_hbf_page_buffer_banks() {
    std::cout << "\n== HBF page-buffer bank parallelism probe ==\n";
    const auto one = run_hbf_page_buffer_bank_read_stream(1);
    const auto four = run_hbf_page_buffer_bank_read_stream(4);
    std::cout << "  1 bank finish_ns = " << fixed(one.finish_ns)
              << " page_buffer_bank_parallelism="
              << fixed(one.stats.page_buffer_bank_parallelism(), 2)
              << " active_page_buffer_banks=" << one.stats.active_page_buffer_banks
              << "/" << one.stats.page_buffer_banks
              << " page_buffer_bank_skew="
              << fixed(one.stats.page_buffer_bank_skew(), 2)
              << "\n";
    std::cout << "  4 bank finish_ns = " << fixed(four.finish_ns)
              << " page_buffer_bank_parallelism="
              << fixed(four.stats.page_buffer_bank_parallelism(), 2)
              << " active_page_buffer_banks=" << four.stats.active_page_buffer_banks
              << "/" << four.stats.page_buffer_banks
              << " page_buffer_bank_skew="
              << fixed(four.stats.page_buffer_bank_skew(), 2)
              << "\n";
    std::cout << "  speedup          = " << fixed(one.finish_ns / four.finish_ns, 2) << "x\n";
    std::cout << "  expectation: reads sense and use media lanes in parallel, then overlap across page-buffer banks.\n";
}

void probe_hbf_subarrays() {
    std::cout << "\n== HBF subarray read-conflict probe ==\n";
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.blocks_per_plane = 1;
    cfg.pages_per_block = 32;
    cfg.page_size_bytes = 512;
    cfg.media_lanes_per_plane = 4;
    cfg.subarrays_per_plane = 4;
    cfg.page_buffer_banks_per_plane = 4;
    cfg.hb_io_bandwidth_GBps = 4096.0;
    cfg.tsv_bandwidth_GBps = 4096.0;
    cfg.channel_bandwidth_GBps = 4096.0;
    cfg.logic_sram_bandwidth_GBps = 4096.0;
    cfg.media_lane_bandwidth_GBps = 4096.0;
    cfg.page_buffer_bandwidth_GBps = 4096.0;
    cfg.t_read_page_ns = 4000.0;
    cfg.ecc_decode_latency_ns = 1.0;
    cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    cfg.logic_scheduler_issue_ns = 1.0;
    cfg.flash_tsu_issue_ns = 1.0;

    // Batch-activation model: a quadrant senses in rounds; the
    // parallelism is round WIDTH — reads ready before a round activates
    // share one tR window, at most one per subarray. Eight reads to one
    // subarray need eight rounds; eight reads over four subarrays fit into
    // about three (the first activates alone, the rest batch).
    HbfDevice same8_dev(cfg);
    for (std::uint32_t i = 0; i < 8; ++i) {
        const auto addr = same8_dev.encode(
            HbfAddress{.block = 0, .page = i * cfg.subarrays_per_plane});
        (void)same8_dev.issue(request(
            "same-subarray", Tier::HBF, Op::Read, 0, addr,
            cfg.page_size_bytes, AddressSpace::Physical));
    }
    HbfDevice spread_dev(cfg);
    std::vector<PhysicalCompletion> spread_rows;
    for (std::uint32_t i = 0; i < 8; ++i) {
        const auto addr = spread_dev.encode(HbfAddress{.block = 0, .page = i});
        auto completion = spread_dev.issue(request(
            "spread-subarray-" + std::to_string(i), Tier::HBF, Op::Read, 0, addr,
            cfg.page_size_bytes, AddressSpace::Physical));
        if (i < 2) {
            spread_rows.push_back(std::move(completion));
        }
    }
    print_rows(spread_rows);
    print_hbf_stats(spread_dev);
    const double same8 = same8_dev.stats().finish_ns;
    const double spread8 = spread_dev.stats().finish_ns;
    std::cout << "  batch mode, 8 reads one subarray finish_ns  = " << fixed(same8) << "\n";
    std::cout << "  batch mode, 8 reads four subarrays finish_ns = " << fixed(spread8) << "\n";
    std::cout << "  speedup          = " << fixed(same8 / spread8, 2) << "x\n";
    std::cout << "  expectation: one subarray serializes eight rounds; four subarrays batch into ~3 rounds.\n";

    // Legacy independent-subarray semantics stays guarded (batch off):
    // near-simultaneous reads to different subarrays overlap fully.
    cfg.batch_activation = false;
    HbfDevice same(cfg);
    const auto same_a = same.encode(HbfAddress{.block = 0, .page = 0});
    const auto same_b = same.encode(HbfAddress{.block = 0, .page = 4});
    (void)same.issue(request(
        "same-subarray-a", Tier::HBF, Op::Read, 0, same_a, cfg.page_size_bytes, AddressSpace::Physical));
    (void)same.issue(request(
        "same-subarray-b", Tier::HBF, Op::Read, 0, same_b, cfg.page_size_bytes, AddressSpace::Physical));
    HbfDevice different(cfg);
    const auto diff_a = different.encode(HbfAddress{.block = 0, .page = 0});
    const auto diff_b = different.encode(HbfAddress{.block = 0, .page = 1});
    (void)different.issue(request(
        "different-subarray-a", Tier::HBF, Op::Read, 0, diff_a, cfg.page_size_bytes, AddressSpace::Physical));
    (void)different.issue(request(
        "different-subarray-b", Tier::HBF, Op::Read, 0, diff_b, cfg.page_size_bytes, AddressSpace::Physical));
    std::cout << "  independent sensitivity (batch off): same-subarray pair finish_ns = "
              << fixed(same.stats().finish_ns)
              << ", different-subarray pair finish_ns = "
              << fixed(different.stats().finish_ns) << "\n";
    std::cout << "  independent overlap ratio = "
              << fixed(same.stats().finish_ns / different.stats().finish_ns, 2) << "x\n";
    std::cout << "  expectation: with batch activation off, pages 0 and 1 overlap across subarrays.\n";
}

void probe_hbf_exact_calendar() {
    std::cout << "\n== HBF exact reservation-calendar probe ==\n";

    // The controller-DRAM budget is global and partitioned per stack. Every
    // partition must hold its complete resident mapping table; an explicit
    // An undersized budget fails instead of silently creating a partial table.
    HbfConfig mapping_budget_cfg;
    mapping_budget_cfg.stacks = 2;
    mapping_budget_cfg.channels_per_stack = 1;
    mapping_budget_cfg.dies_per_channel = 1;
    mapping_budget_cfg.planes_per_die = 1;
    mapping_budget_cfg.blocks_per_plane = 4;
    mapping_budget_cfg.pages_per_block = 4;
    mapping_budget_cfg.page_size_bytes = 512;
    mapping_budget_cfg.oob_bytes_per_page = 0;
    mapping_budget_cfg.ctrl_dram_bytes = 512;
    bool undersized_mapping_rejected = false;
    try {
        (void)HbfDevice(mapping_budget_cfg);
    } catch (const std::runtime_error&) {
        undersized_mapping_rejected = true;
    }
    mapping_budget_cfg.ctrl_dram_bytes =
        mapping_budget_cfg.stacks * mapping_budget_cfg.page_size_bytes;
    HbfDevice minimum_mapping(mapping_budget_cfg);
    if (!undersized_mapping_rejected ||
        minimum_mapping.stats().resident_mapping_pages_per_stack != 1 ||
        minimum_mapping.stats().resident_mapping_table_bytes_per_stack !=
            mapping_budget_cfg.page_size_bytes) {
        throw std::runtime_error(
            "hbf-exact-calendar: resident mapping budget did not fail closed");
    }
    std::cout << "  resident_mapping_budget_fail_closed_ok=yes\n";

    // Fill one die's ECC issue calendar with 96 far-future reservations
    // separated by one exactly fitting initiation interval apiece. Reads issued
    // later to the other plane become ready early and must retain access to
    // all 96 gaps. A small downstream pipeline tail is physical; the obsolete
    // 64-gap history lost 32 real intervals and added 68 us beyond the
    // slow-plane frontier.
    HbfConfig gap_cfg;
    gap_cfg.stacks = 1;
    gap_cfg.channels_per_stack = 1;
    gap_cfg.dies_per_channel = 1;
    gap_cfg.planes_per_die = 2;
    gap_cfg.blocks_per_plane = 4;
    gap_cfg.pages_per_block = 1024;
    gap_cfg.page_size_bytes = 512;
    gap_cfg.oob_bytes_per_page = 0;
    gap_cfg.media_lanes_per_plane = 32;
    gap_cfg.subarrays_per_plane = 32;
    gap_cfg.page_buffer_banks_per_plane = 32;
    gap_cfg.t_read_page_ns = 4000.0;
    gap_cfg.ecc_decode_latency_ns = 2000.0;
    gap_cfg.ecc_decode_raw_bandwidth_GBps_per_die = 0.256;
    gap_cfg.logic_scheduler_issue_ns = 1.0;
    gap_cfg.flash_tsu_issue_ns = 1.0;
    gap_cfg.channel_bandwidth_GBps = 1.0e9;
    gap_cfg.hb_io_bandwidth_GBps = 1.0e9;
    gap_cfg.tsv_bandwidth_GBps = 1.0e9;
    gap_cfg.media_lane_bandwidth_GBps = 1.0e9;
    gap_cfg.logic_sram_bandwidth_GBps = 1.0e9;
    gap_cfg.page_buffer_bandwidth_GBps = 1.0e9;

    HbfDevice gap_device(gap_cfg);
    double slow_plane_finish = 0.0;
    double backfill_finish = 0.0;
    for (std::uint32_t i = 0; i < 96; ++i) {
        const auto addr = gap_device.encode(HbfAddress{
            .plane = 0,
            .block = i / 32,
            .page = (i % 32) * gap_cfg.subarrays_per_plane,
        });
        const auto completion = gap_device.issue(request(
            "calendar-slow-" + std::to_string(i),
            Tier::HBF,
            Op::Read,
            0.0,
            addr,
            gap_cfg.page_size_bytes,
            AddressSpace::Physical));
        slow_plane_finish = std::max(slow_plane_finish, completion.finish_ns);
    }
    for (std::uint32_t i = 0; i < 96; ++i) {
        const auto addr = gap_device.encode(HbfAddress{
            .plane = 1,
            .page = i,
        });
        const auto completion = gap_device.issue(request(
            "calendar-backfill-" + std::to_string(i),
            Tier::HBF,
            Op::Read,
            0.0,
            addr,
            gap_cfg.page_size_bytes,
            AddressSpace::Physical));
        backfill_finish = std::max(backfill_finish, completion.finish_ns);
    }
    if (backfill_finish >
        slow_plane_finish + 2.0 * gap_cfg.t_read_page_ns + 1e-6) {
        throw std::runtime_error(
            "hbf-exact-calendar: more than 64 reservations lost reusable idle capacity");
    }
    std::cout << "  slow_plane_finish_ns=" << fixed(slow_plane_finish)
              << " backfill_finish_ns=" << fixed(backfill_finish) << "\n";
    std::cout << "  resource_calendar_backfill_over_64_ok=yes\n";

    // One subarray creates 96 future batch rounds. A later-called sibling
    // subarray is ready early and must retain access to every joinable one.
    // One extra physical round is allowed because the first command becomes
    // ready after the first round has already activated; the obsolete fixed
    // 64-round history added 132 us beyond the original frontier.
    HbfConfig round_cfg = gap_cfg;
    round_cfg.planes_per_die = 1;
    round_cfg.blocks_per_plane = 2;
    round_cfg.media_lanes_per_plane = 2;
    round_cfg.subarrays_per_plane = 2;
    round_cfg.page_buffer_banks_per_plane = 2;
    round_cfg.ecc_decode_latency_ns = 0.001;
    round_cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    round_cfg.logic_scheduler_issue_ns = 0.001;
    round_cfg.flash_tsu_issue_ns = 0.001;

    HbfDevice round_device(round_cfg);
    double first_subarray_finish = 0.0;
    double joined_subarray_finish = 0.0;
    for (std::uint32_t i = 0; i < 96; ++i) {
        const auto addr = round_device.encode(HbfAddress{
            .block = i / 512,
            .page = (i % 512) * 2,
        });
        const auto completion = round_device.issue(request(
            "round-slow-" + std::to_string(i),
            Tier::HBF,
            Op::Read,
            0.0,
            addr,
            round_cfg.page_size_bytes,
            AddressSpace::Physical));
        first_subarray_finish =
            std::max(first_subarray_finish, completion.finish_ns);
    }
    for (std::uint32_t i = 0; i < 96; ++i) {
        const auto addr = round_device.encode(HbfAddress{
            .block = i / 512,
            .page = (i % 512) * 2 + 1,
        });
        const auto completion = round_device.issue(request(
            "round-join-" + std::to_string(i),
            Tier::HBF,
            Op::Read,
            0.0,
            addr,
            round_cfg.page_size_bytes,
            AddressSpace::Physical));
        joined_subarray_finish =
            std::max(joined_subarray_finish, completion.finish_ns);
    }
    if (joined_subarray_finish >
        first_subarray_finish + round_cfg.t_read_page_ns + 1e-6) {
        throw std::runtime_error(
            "hbf-exact-calendar: more than 64 future sense rounds lost legal batch slots");
    }
    std::cout << "  first_subarray_finish_ns=" << fixed(first_subarray_finish)
              << " joined_subarray_finish_ns=" << fixed(joined_subarray_finish) << "\n";
    std::cout << "  sense_round_backfill_over_64_ok=yes\n";

    // A future program reservation must not pull an earlier-ready read behind
    // it merely because the program was scheduled first in host call order.
    // Conversely, the complete sense->lane->page-buffer path is atomic with
    // respect to a full-plane program: a slow lane may not let a read straddle
    // the program window.
    auto plane_backfill_config = []() {
        HbfConfig cfg;
        cfg.stacks = 1;
        cfg.channels_per_stack = 1;
        cfg.dies_per_channel = 1;
        cfg.planes_per_die = 1;
        cfg.blocks_per_plane = 8;
        cfg.pages_per_block = 8;
        cfg.page_size_bytes = 512;
        cfg.oob_bytes_per_page = 0;
        cfg.subarrays_per_plane = 1;
        cfg.media_lanes_per_plane = 1;
        cfg.page_buffer_banks_per_plane = 1;
        cfg.write_buffer_pages = 2;
        cfg.write_buffer_flush_threshold_pages = 1;
        cfg.write_coalescing_enabled = true;
        cfg.t_read_page_ns = 1000.0;
        cfg.t_program_page_ns = 80000.0;
        cfg.t_program_verify_ns = 1.0;
        cfg.channel_bandwidth_GBps = 1.0e9;
        cfg.hb_io_bandwidth_GBps = 1.0e9;
        cfg.tsv_bandwidth_GBps = 1.0e9;
        cfg.media_lane_bandwidth_GBps = 1.0e9;
        cfg.page_buffer_bandwidth_GBps = 1.0e9;
        cfg.logic_sram_bandwidth_GBps = 1.0e9;
        cfg.ecc_decode_latency_ns = 0.001;
        cfg.ecc_encode_latency_ns = 10000.0;
        cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
        cfg.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
        cfg.logic_scheduler_issue_ns = 0.001;
        cfg.flash_tsu_issue_ns = 0.001;
        return cfg;
    };
    auto find_plane_span = [](const PhysicalCompletion& completion, const std::string& name)
        -> const TraceSpan& {
        const auto found = std::find_if(
            completion.spans.begin(), completion.spans.end(),
            [&name](const TraceSpan& span) { return span.name == name; });
        if (found == completion.spans.end()) {
            throw std::runtime_error(
                "hbf-exact-calendar: missing trace span " + name);
        }
        return *found;
    };
    auto run_plane_backfill = [&](HbfConfig cfg) {
        HbfDevice device(cfg);
        device.reserve_static_physical_pages({0});
        auto write = request(
            "plane-future-program", Tier::HBF, Op::Write, 0.0, 0,
            cfg.page_size_bytes, AddressSpace::Logical);
        write.trace.mode = TraceMode::Full;
        const auto write_completion = device.issue(write);
        auto read = request(
            "plane-early-read", Tier::HBF, Op::Read, 0.0, 0,
            cfg.page_size_bytes, AddressSpace::Physical);
        read.trace.mode = TraceMode::Full;
        const auto read_completion = device.issue(read);
        return std::make_pair(write_completion, read_completion);
    };

    const auto fast_plane = run_plane_backfill(plane_backfill_config());
    const auto& fast_program = find_plane_span(fast_plane.first, "user/array_program");
    const auto& fast_read_buffer = find_plane_span(fast_plane.second, "user/page_buffer_out");
    if (fast_read_buffer.end_ns > fast_program.start_ns + 1e-6) {
        throw std::runtime_error(
            "hbf-exact-calendar: future program created phantom read blocking");
    }
    std::cout << "  plane_barrier_backfill_ok=yes\n";

    auto slow_lane_cfg = plane_backfill_config();
    slow_lane_cfg.media_lane_bandwidth_GBps =
        static_cast<double>(slow_lane_cfg.page_size_bytes) / 12000.0;
    const auto slow_plane = run_plane_backfill(slow_lane_cfg);
    const auto& slow_verify = find_plane_span(slow_plane.first, "user/program_verify");
    const auto& slow_read = find_plane_span(slow_plane.second, "user/array_read");
    if (slow_read.start_ns + 1e-6 < slow_verify.end_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: read data path straddled a full-plane program");
    }
    std::cout << "  plane_barrier_no_straddle_ok=yes\n";

    // Full-plane operations use the same exact calendars as reads. A short
    // program/erase can therefore occupy a common early gap before a read
    // whose mapping miss makes its array phase ready much later.
    auto reverse_backfill_cfg = plane_backfill_config();
    reverse_backfill_cfg.write_coalescing_enabled = false;
    reverse_backfill_cfg.ctrl_dram_latency_ns = 10000.0;
    reverse_backfill_cfg.t_program_page_ns = 100.0;
    reverse_backfill_cfg.t_program_verify_ns = 1.0;
    reverse_backfill_cfg.t_erase_block_ns = 100.0;
    reverse_backfill_cfg.ecc_decode_latency_ns = 1.0;
    reverse_backfill_cfg.ecc_encode_latency_ns = 1.0;
    HbfDevice program_backfill_device(reverse_backfill_cfg);
    program_backfill_device.prepopulate_logical_pages({0});
    auto late_read_request = request(
        "plane-late-read", Tier::HBF, Op::Read, 0.0, 0,
        reverse_backfill_cfg.page_size_bytes, AddressSpace::Logical);
    late_read_request.trace.mode = TraceMode::Full;
    const auto late_read = program_backfill_device.issue(late_read_request);
    const auto raw_block2_addr = program_backfill_device.encode(HbfAddress{.block = 2});
    auto early_program_request = request(
        "plane-early-program", Tier::HBF, Op::Write, 0.0,
        raw_block2_addr, reverse_backfill_cfg.page_size_bytes,
        AddressSpace::Physical);
    early_program_request.trace.mode = TraceMode::Full;
    const auto early_program = program_backfill_device.issue(early_program_request);
    if (find_plane_span(early_program, "user/program_verify").end_ns >
        find_plane_span(late_read, "user/array_read").start_ns + 1e-6) {
        throw std::runtime_error(
            "hbf-exact-calendar: program failed to backfill a common pre-read gap");
    }
    std::cout << "  full_plane_program_reverse_backfill_ok=yes\n";

    HbfDevice erase_backfill_device(reverse_backfill_cfg);
    erase_backfill_device.prepopulate_logical_pages({0});
    auto erase_late_read_request = request(
        "plane-erase-late-read", Tier::HBF, Op::Read, 0.0, 0,
        reverse_backfill_cfg.page_size_bytes, AddressSpace::Logical);
    erase_late_read_request.trace.mode = TraceMode::Full;
    const auto erase_late_read = erase_backfill_device.issue(
        erase_late_read_request);
    const auto erase_block2_addr = erase_backfill_device.encode(HbfAddress{.block = 2});
    auto early_erase_request = request(
        "plane-early-erase", Tier::HBF, Op::Erase, 0.0,
        erase_block2_addr, 0, AddressSpace::Physical);
    early_erase_request.trace.mode = TraceMode::Full;
    const auto early_erase = erase_backfill_device.issue(early_erase_request);
    if (find_plane_span(early_erase, "user/block_erase").end_ns >
        find_plane_span(erase_late_read, "user/array_read").start_ns + 1e-6) {
        throw std::runtime_error(
            "hbf-exact-calendar: erase failed to backfill a common pre-read gap");
    }
    std::cout << "  full_plane_erase_reverse_backfill_ok=yes\n";

    // Erasing an already-free raw block must remove it from the allocator
    // until the erase commit. The old path left block0 in free_blocks, so the
    // same-arrival logical write claimed it and the later erase callback
    // destroyed that new reservation. After the commit, the block must be
    // returned exactly once and become reusable at the tail of the pool.
    auto pending_free_erase_cfg = reverse_backfill_cfg;
    // One additional block holds the resident table's persistent checkpoint;
    // it is not part of the data allocator's reuse sequence.
    pending_free_erase_cfg.blocks_per_plane = 5;
    pending_free_erase_cfg.pages_per_block = 2;
    pending_free_erase_cfg.auto_gc_enabled = false;
    HbfDevice pending_free_erase_device(pending_free_erase_cfg);
    const auto pending_block0_addr = pending_free_erase_device.encode(
        HbfAddress{.block = 0});
    const auto pending_free_erase = pending_free_erase_device.issue(request(
        "pending-free-block-erase",
        Tier::HBF,
        Op::Erase,
        0.0,
        pending_block0_addr,
        0,
        AddressSpace::Physical));
    const auto competing_write = pending_free_erase_device.issue(request(
        "pending-free-block-competing-write",
        Tier::HBF,
        Op::Write,
        0.0,
        0,
        pending_free_erase_cfg.page_size_bytes,
        AddressSpace::Logical));
    if (competing_write.resource_path.find("block1/page0") == std::string::npos) {
        throw std::runtime_error(
            "hbf-exact-calendar: pending erase block remained FTL-allocatable");
    }
    double pending_cursor = std::max(
        pending_free_erase.finish_ns, competing_write.finish_ns);
    const auto pending_free_drain = pending_free_erase_device.drain_pending(
        "pending-free-block-drain", pending_cursor);
    pending_cursor = pending_free_drain.finish_ns;
    PhysicalCompletion reused_erased_block;
    for (std::uint64_t lpn = 1; lpn <= 6; ++lpn) {
        reused_erased_block = pending_free_erase_device.issue(request(
            "pending-free-block-followup-" + std::to_string(lpn),
            Tier::HBF,
            Op::Write,
            pending_cursor,
            lpn * pending_free_erase_cfg.page_size_bytes,
            pending_free_erase_cfg.page_size_bytes,
            AddressSpace::Logical));
        pending_cursor = reused_erased_block.finish_ns;
    }
    const auto pending_final_drain = pending_free_erase_device.drain_pending(
        "pending-free-block-final-drain", pending_cursor);
    (void)pending_final_drain;
    if (reused_erased_block.resource_path.find("block0/page0") ==
            std::string::npos ||
        !pending_free_erase_device.stats().accounting_verified) {
        throw std::runtime_error(
            "hbf-exact-calendar: erased free block was not returned exactly once");
    }
    std::cout << "  pending_free_block_erase_ownership_ok=yes\n";

    // Future fills do not occupy the one-page cache until they complete. A
    // later-called A request at the same arrival therefore merges with A's
    // in-flight fill even though B was scheduled between the two calls.
    auto read_buffer_cfg = round_cfg;
    read_buffer_cfg.batch_activation = false;
    read_buffer_cfg.read_buffer_pages = 1;
    HbfDevice read_buffer_device(read_buffer_cfg);
    const auto rb_a1 = read_buffer_device.issue(request(
        "rb-future-a1", Tier::HBF, Op::Read, 0.0, 0,
        read_buffer_cfg.page_size_bytes, AddressSpace::Physical));
    (void)rb_a1;
    const auto rb_b = read_buffer_device.issue(request(
        "rb-future-b", Tier::HBF, Op::Read, 0.0,
        read_buffer_cfg.page_size_bytes,
        read_buffer_cfg.page_size_bytes, AddressSpace::Physical));
    const auto rb_a2 = read_buffer_device.issue(request(
        "rb-future-a2", Tier::HBF, Op::Read, 0.0, 0,
        read_buffer_cfg.page_size_bytes, AddressSpace::Physical));
    const auto rb_b2 = read_buffer_device.issue(request(
        "rb-future-b2", Tier::HBF, Op::Read, rb_b.finish_ns,
        read_buffer_cfg.page_size_bytes,
        read_buffer_cfg.page_size_bytes, AddressSpace::Physical));
    const auto rb_stats = read_buffer_device.stats();
    if (rb_stats.page_reads != 2 || rb_stats.read_buffer_hits != 2 ||
        rb_a2.note != "read-buffer-hit" || rb_b2.note != "read-buffer-hit") {
        throw std::runtime_error(
            "hbf-exact-calendar: future fill evicted an in-flight read-buffer merge target");
    }
    std::cout << "  read_buffer_future_fill_merge_ok=yes\n";

    auto delayed_rb_cfg = plane_backfill_config();
    delayed_rb_cfg.write_coalescing_enabled = false;
    delayed_rb_cfg.read_buffer_pages = 1;
    delayed_rb_cfg.t_read_page_ns = 100.0;
    delayed_rb_cfg.ctrl_dram_latency_ns = 1000.0;
    delayed_rb_cfg.ecc_decode_latency_ns = 1.0;
    HbfDevice delayed_rb_device(delayed_rb_cfg);
    delayed_rb_device.prepopulate_logical_pages({0});
    (void)delayed_rb_device.issue(request(
        "rb-delayed-a-fill", Tier::HBF, Op::Read, 0.0, 0,
        delayed_rb_cfg.page_size_bytes, AddressSpace::Physical));
    (void)delayed_rb_device.issue(request(
        "rb-delayed-b-fill", Tier::HBF, Op::Read, 0.0,
        delayed_rb_cfg.page_size_bytes, delayed_rb_cfg.page_size_bytes,
        AddressSpace::Physical));
    const auto delayed_rb_read = delayed_rb_device.issue(request(
        "rb-delayed-a-logical", Tier::HBF, Op::Read, 0.0, 0,
        delayed_rb_cfg.page_size_bytes, AddressSpace::Logical));
    if (delayed_rb_device.stats().page_reads != 3 ||
        delayed_rb_read.note == "read-buffer-hit") {
        throw std::runtime_error(
            "hbf-exact-calendar: delayed cache lookup claimed an already-evictable fill");
    }
    std::cout << "  read_buffer_delayed_lookup_no_false_hit_ok=yes\n";

    auto partial_rb_cfg = plane_backfill_config();
    partial_rb_cfg.read_buffer_pages = 1;
    partial_rb_cfg.write_coalescing_enabled = true;
    partial_rb_cfg.write_buffer_flush_threshold_pages = 0;
    HbfDevice partial_rb_device(partial_rb_cfg);
    partial_rb_device.prepopulate_logical_pages({0});
    const auto partial_warm = partial_rb_device.issue(request(
        "rb-partial-warm", Tier::HBF, Op::Read, 0.0, 0,
        partial_rb_cfg.page_size_bytes, AddressSpace::Logical));
    const auto partial_write = partial_rb_device.issue(request(
        "rb-partial-write", Tier::HBF, Op::Write, partial_warm.finish_ns,
        0, 64, AddressSpace::Logical));
    const auto partial_read = partial_rb_device.issue(request(
        "rb-partial-overlay", Tier::HBF, Op::Read, partial_write.finish_ns,
        0, 128, AddressSpace::Logical));
    const auto partial_stats = partial_rb_device.stats();
    if (partial_read.note != "mapped-read-with-write-buffer-overlay" ||
        partial_stats.read_buffer_read_bytes != 128 ||
        partial_stats.write_buffer_read_bytes != 64) {
        throw std::runtime_error(
            "hbf-exact-calendar: partial WB overlay charged a full read-buffer page");
    }
    std::cout << "  read_buffer_partial_overlay_exact_bytes_ok=yes\n";

    // A raw physical read of an erased page can cache the erased value. The
    // following program changes the page contents without changing its block
    // epoch, so program completion itself must retire the exact PPN cache line.
    HbfConfig program_cache_cfg;
    program_cache_cfg.stacks = 1;
    program_cache_cfg.channels_per_stack = 1;
    program_cache_cfg.dies_per_channel = 1;
    program_cache_cfg.planes_per_die = 1;
    program_cache_cfg.blocks_per_plane = 2;
    program_cache_cfg.pages_per_block = 4;
    program_cache_cfg.page_size_bytes = 512;
    program_cache_cfg.oob_bytes_per_page = 0;
    program_cache_cfg.read_buffer_pages = 1;
    program_cache_cfg.ecc_decode_latency_ns = 1.0;
    program_cache_cfg.ecc_encode_latency_ns = 1.0;
    program_cache_cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    program_cache_cfg.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    HbfDevice program_cache_device(program_cache_cfg);
    const auto erased_warm = program_cache_device.issue(request(
        "program-cache-erased-warm", Tier::HBF, Op::Read, 0.0, 0,
        program_cache_cfg.page_size_bytes, AddressSpace::Physical));
    const auto raw_program = program_cache_device.issue(request(
        "program-cache-raw-write", Tier::HBF, Op::Write,
        erased_warm.finish_ns, 0, program_cache_cfg.page_size_bytes,
        AddressSpace::Physical));
    const auto programmed_read = program_cache_device.issue(request(
        "program-cache-post-write-read", Tier::HBF, Op::Read,
        raw_program.finish_ns, 0, program_cache_cfg.page_size_bytes,
        AddressSpace::Physical));
    const auto program_cache_stats = program_cache_device.stats();
    if (programmed_read.note == "read-buffer-hit" ||
        program_cache_stats.page_reads != 2 ||
        program_cache_stats.read_buffer_hits != 0 ||
        program_cache_stats.ecc_decode_ops != 2) {
        throw std::runtime_error(
            "hbf-exact-calendar: program left an erased-value read-buffer line readable");
    }
    std::cout << "  program_invalidates_read_buffer_ok=yes\n";

    // Program media completion and logical mapping publication are separate
    // events. If a destructive erase lands between them, the retired block
    // epoch must make the late publication a no-op rather than resurrecting
    // the erased page.
    auto epoch_cfg = plane_backfill_config();
    epoch_cfg.write_coalescing_enabled = false;
    epoch_cfg.mapping_update_ns = 3000000.0;
    epoch_cfg.read_buffer_pages = 1;
    HbfDevice epoch_device(epoch_cfg);
    (void)epoch_device.issue(request(
        "epoch-logical-write", Tier::HBF, Op::Write, 0.0, 0,
        epoch_cfg.page_size_bytes, AddressSpace::Logical));
    (void)epoch_device.issue(request(
        "epoch-physical-erase", Tier::HBF, Op::Erase, 0.0, 0, 0,
        AddressSpace::Physical));
    const auto epoch_drain = epoch_device.drain_pending("epoch-drain", 0.0);
    const auto epoch_read = epoch_device.issue(request(
        "epoch-post-erase-read", Tier::HBF, Op::Read,
        epoch_drain.finish_ns, 0, epoch_cfg.page_size_bytes,
        AddressSpace::Logical));
    if (epoch_read.note != "unmapped-erased-read" ||
        epoch_device.stats().mapping_entries != 0) {
        throw std::runtime_error(
            "hbf-exact-calendar: late mapping publish resurrected an erased block epoch");
    }
    std::cout << "  erase_epoch_no_resurrection_ok=yes\n";

    // An erase also retires decoded SRAM data. A read admitted after the
    // erase may return the erased logical image, but never the old read-buffer
    // line keyed by the reused PPN.
    auto cache_epoch_cfg = plane_backfill_config();
    cache_epoch_cfg.read_buffer_pages = 1;
    HbfDevice cache_epoch_device(cache_epoch_cfg);
    cache_epoch_device.prepopulate_logical_pages({0});
    (void)cache_epoch_device.issue(request(
        "epoch-cache-warm", Tier::HBF, Op::Read, 0.0, 0,
        cache_epoch_cfg.page_size_bytes, AddressSpace::Logical));
    const auto cache_erase = cache_epoch_device.issue(request(
        "epoch-cache-erase", Tier::HBF, Op::Erase, 0.0, 0, 0,
        AddressSpace::Physical));
    const auto cache_read = cache_epoch_device.issue(request(
        "epoch-cache-read", Tier::HBF, Op::Read, 0.0, 0,
        cache_epoch_cfg.page_size_bytes, AddressSpace::Logical));
    if (cache_read.note != "unmapped-erased-read" ||
        cache_read.finish_ns < cache_erase.finish_ns ||
        cache_epoch_device.stats().read_buffer_hits != 0) {
        throw std::runtime_error(
            "hbf-exact-calendar: erase left a readable stale cache epoch");
    }
    std::cout << "  erase_read_buffer_epoch_ok=yes\n";

    // If erase retires a data page before its first LPN publication, the
    // pending mapping becomes an ordering tombstone. A later logical read may
    // return erased data, but only after both the prior write callback and the
    // destructive block operation have completed.
    HbfConfig pending_erase_cfg;
    pending_erase_cfg.stacks = 1;
    pending_erase_cfg.channels_per_stack = 1;
    pending_erase_cfg.dies_per_channel = 1;
    pending_erase_cfg.planes_per_die = 1;
    pending_erase_cfg.blocks_per_plane = 4;
    pending_erase_cfg.pages_per_block = 4;
    pending_erase_cfg.page_size_bytes = 512;
    pending_erase_cfg.oob_bytes_per_page = 0;
    pending_erase_cfg.read_buffer_pages = 0;
    HbfDevice pending_erase_device(pending_erase_cfg);
    const auto pending_write = pending_erase_device.issue(request(
        "pending-erase-write", Tier::HBF, Op::Write, 0.0, 0,
        pending_erase_cfg.page_size_bytes, AddressSpace::Logical));
    const auto pending_erase = pending_erase_device.issue(request(
        "pending-erase-block", Tier::HBF, Op::Erase, 0.0, 0, 0,
        AddressSpace::Physical));
    const auto pending_read = pending_erase_device.issue(request(
        "pending-erase-read", Tier::HBF, Op::Read,
        pending_write.finish_ns, 0, pending_erase_cfg.page_size_bytes,
        AddressSpace::Logical));
    if (pending_read.note != "unmapped-erased-read" ||
        pending_read.finish_ns + 1e-6 < pending_erase.finish_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: retired pending mapping bypassed its block erase");
    }
    std::cout << "  erase_pending_mapping_dependency_ok=yes\n";

    // Mapping checkpoint media is not part of the resident-table read path.
    // After persisting VPN1 on plane 1, erase that checkpoint block while the
    // corresponding data page remains readable from plane 0.
    HbfConfig pending_vpn_cfg;
    pending_vpn_cfg.stacks = 1;
    pending_vpn_cfg.channels_per_stack = 1;
    pending_vpn_cfg.dies_per_channel = 1;
    pending_vpn_cfg.planes_per_die = 2;
    pending_vpn_cfg.blocks_per_plane = 8;
    pending_vpn_cfg.pages_per_block = 4;
    pending_vpn_cfg.page_size_bytes = 512;
    pending_vpn_cfg.oob_bytes_per_page = 0;
    pending_vpn_cfg.mapping_entries_per_page = 1;
    pending_vpn_cfg.read_buffer_pages = 0;
    pending_vpn_cfg.ecc_decode_latency_ns = 1.0;
    pending_vpn_cfg.ecc_encode_latency_ns = 1.0;
    pending_vpn_cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    pending_vpn_cfg.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    HbfDevice pending_vpn_device(pending_vpn_cfg);
    const auto vpn_write = pending_vpn_device.issue(request(
        "pending-vpn-write", Tier::HBF, Op::Write, 0.0,
        pending_vpn_cfg.page_size_bytes,
        pending_vpn_cfg.page_size_bytes, AddressSpace::Logical));
    const auto checkpoint = pending_vpn_device.drain_pending(
        "pending-vpn-checkpoint", vpn_write.finish_ns);
    const auto first_mapping_block_addr = pending_vpn_device.encode(
        HbfAddress{.plane = 1, .block = 0});
    const auto pending_vpn_erase = pending_vpn_device.issue(request(
        "pending-vpn-erase", Tier::HBF, Op::Erase, checkpoint.finish_ns,
        first_mapping_block_addr, 0, AddressSpace::Physical));
    auto pending_vpn_read_request = request(
        "pending-vpn-post-erase-read", Tier::HBF, Op::Read,
        checkpoint.finish_ns, pending_vpn_cfg.page_size_bytes,
        pending_vpn_cfg.page_size_bytes,
        AddressSpace::Logical);
    pending_vpn_read_request.trace.mode = TraceMode::Full;
    const auto pending_vpn_read = pending_vpn_device.issue(
        pending_vpn_read_request);
    const bool traced_pending_vpn_erase_wait = std::any_of(
        pending_vpn_read.spans.begin(),
        pending_vpn_read.spans.end(),
        [](const TraceSpan& span) {
            return span.name == "wait_pending_mapping_block_erase";
        });
    if (pending_vpn_read.finish_ns >= pending_vpn_erase.finish_ns ||
        pending_vpn_device.stats().mapping_page_programs != 1 ||
        traced_pending_vpn_erase_wait) {
        throw std::runtime_error(
            "hbf-exact-calendar: checkpoint erase leaked into resident lookup");
    }
    std::cout << "  resident_mapping_ignores_checkpoint_erase_ok=yes\n";

    // A destructive erase may backfill before unrelated future work, but it
    // may not pass a program already issued to the same block. Use a very long
    // encode latency so an incorrect full-plane backfill is easy to detect.
    HbfConfig program_erase_cfg;
    program_erase_cfg.stacks = 1;
    program_erase_cfg.channels_per_stack = 1;
    program_erase_cfg.dies_per_channel = 1;
    program_erase_cfg.planes_per_die = 1;
    program_erase_cfg.blocks_per_plane = 2;
    program_erase_cfg.pages_per_block = 4;
    program_erase_cfg.page_size_bytes = 512;
    program_erase_cfg.oob_bytes_per_page = 0;
    program_erase_cfg.ecc_decode_latency_ns = 1.0;
    program_erase_cfg.ecc_encode_latency_ns = 10000000.0;
    program_erase_cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    program_erase_cfg.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    HbfDevice program_erase_device(program_erase_cfg);
    auto ordered_program_request = request(
        "ordered-raw-program", Tier::HBF, Op::Write, 0.0, 0,
        program_erase_cfg.page_size_bytes, AddressSpace::Physical);
    ordered_program_request.trace.mode = TraceMode::Full;
    const auto ordered_program = program_erase_device.issue(
        ordered_program_request);
    auto ordered_erase_request = request(
        "ordered-raw-erase", Tier::HBF, Op::Erase, 0.0, 0, 0,
        AddressSpace::Physical);
    ordered_erase_request.trace.mode = TraceMode::Full;
    const auto ordered_erase = program_erase_device.issue(
        ordered_erase_request);
    if (find_plane_span(ordered_erase, "user/block_erase").start_ns + 1e-6 <
        find_plane_span(ordered_program, "user/program_verify").end_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: erase backfilled ahead of a same-block program");
    }
    std::cout << "  erase_waits_target_program_ok=yes\n";

    HbfDevice raw_order_device(program_erase_cfg);
    auto raw_order_write_request = request(
        "raw-order-write", Tier::HBF, Op::Write, 0.0, 0,
        program_erase_cfg.page_size_bytes, AddressSpace::Physical);
    raw_order_write_request.trace.mode = TraceMode::Full;
    const auto raw_order_write = raw_order_device.issue(raw_order_write_request);
    auto raw_order_read_request = request(
        "raw-order-read", Tier::HBF, Op::Read, 1.0, 0,
        program_erase_cfg.page_size_bytes, AddressSpace::Physical);
    raw_order_read_request.trace.mode = TraceMode::Full;
    const auto raw_order_read = raw_order_device.issue(raw_order_read_request);
    if (find_plane_span(raw_order_read, "user/array_read").start_ns + 1e-6 <
        find_plane_span(raw_order_write, "user/program_verify").end_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: raw physical read bypassed its PPN program");
    }
    std::cout << "  raw_read_waits_target_program_ok=yes\n";

    auto read_erase_cfg = program_erase_cfg;
    read_erase_cfg.blocks_per_plane = 4;
    read_erase_cfg.ctrl_dram_latency_ns = 10000000.0;
    HbfDevice read_erase_device(read_erase_cfg);
    read_erase_device.prepopulate_logical_pages({0});
    auto ordered_read_request = request(
        "ordered-logical-read", Tier::HBF, Op::Read, 0.0, 0,
        read_erase_cfg.page_size_bytes, AddressSpace::Logical);
    ordered_read_request.trace.mode = TraceMode::Full;
    const auto ordered_read = read_erase_device.issue(ordered_read_request);
    auto read_following_erase_request = request(
        "ordered-read-following-erase", Tier::HBF, Op::Erase, 1.0, 0, 0,
        AddressSpace::Physical);
    read_following_erase_request.trace.mode = TraceMode::Full;
    const auto read_following_erase = read_erase_device.issue(
        read_following_erase_request);
    if (find_plane_span(read_following_erase, "user/block_erase").start_ns + 1e-6 <
        find_plane_span(ordered_read, "user/page_buffer_out").end_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: erase backfilled ahead of a same-block read");
    }
    std::cout << "  erase_waits_target_read_ok=yes\n";

    // Waiting for an independent block erase must not make a later same-LPN
    // read observe the future mapping before its own program. The LPN-specific
    // dependency still applies that program+mapping chain in order.
    auto state_time_cfg = program_erase_cfg;
    state_time_cfg.blocks_per_plane = 8;
    state_time_cfg.t_erase_block_ns = 20000000.0;
    state_time_cfg.read_buffer_pages = 0;
    HbfDevice state_time_device(state_time_cfg);
    state_time_device.prepopulate_logical_pages({0});
    auto future_mapping_write_request = request(
        "state-time-future-write", Tier::HBF, Op::Write, 0.0, 0,
        state_time_cfg.page_size_bytes, AddressSpace::Logical);
    future_mapping_write_request.trace.mode = TraceMode::Full;
    const auto future_mapping_write = state_time_device.issue(
        future_mapping_write_request);
    const auto independent_block_addr = state_time_device.encode(
        HbfAddress{.block = 7});
    const auto state_time_erase = state_time_device.issue(request(
        "state-time-independent-erase", Tier::HBF, Op::Erase, 0.0,
        independent_block_addr, 0, AddressSpace::Physical));
    (void)state_time_device.issue(request(
        "state-time-erase-dependent-read", Tier::HBF, Op::Read, 0.0,
        independent_block_addr, state_time_cfg.page_size_bytes,
        AddressSpace::Physical));
    auto post_materialization_read_request = request(
        "state-time-logical-read", Tier::HBF, Op::Read, 0.0, 0,
        state_time_cfg.page_size_bytes, AddressSpace::Logical);
    post_materialization_read_request.trace.mode = TraceMode::Full;
    const auto post_materialization_read = state_time_device.issue(
        post_materialization_read_request);
    std::cout << "  targeted_state_path=" << post_materialization_read.resource_path
              << " read_finish_ns=" << fixed(post_materialization_read.finish_ns)
              << " write_verify_end_ns="
              << fixed(find_plane_span(
                    future_mapping_write, "user/program_verify").end_ns)
              << " erase_finish_ns=" << fixed(state_time_erase.finish_ns)
              << "\n";
    if (post_materialization_read.resource_path.find("block0/page1") ==
            std::string::npos ||
        find_plane_span(post_materialization_read, "user/array_read").start_ns + 1e-6 <
            find_plane_span(future_mapping_write, "user/program_verify").end_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: targeted wait exposed a mapping before its program");
    }
    std::cout << "  targeted_state_no_reverse_time_travel_ok=yes\n";

    auto run_plane_order = [&plane_backfill_config](bool dependent_first) {
        auto cfg = plane_backfill_config();
        cfg.planes_per_die = 2;
        cfg.blocks_per_plane = 2;
        cfg.t_read_page_ns = 100.0;
        cfg.t_erase_block_ns = 20000000.0;
        cfg.ecc_decode_latency_ns = 1.0;
        HbfDevice device(cfg);
        const auto plane0 = device.encode(HbfAddress{.plane = 0});
        const auto plane1 = device.encode(HbfAddress{.plane = 1});
        const auto erase = device.issue(request(
            "order-plane0-erase", Tier::HBF, Op::Erase, 0.0,
            plane0, 0, AddressSpace::Physical));
        PhysicalCompletion independent;
        if (dependent_first) {
            (void)device.issue(request(
                "order-plane0-dependent", Tier::HBF, Op::Read, 0.0,
                plane0, cfg.page_size_bytes, AddressSpace::Physical));
            independent = device.issue(request(
                "order-plane1-independent", Tier::HBF, Op::Read, 0.0,
                plane1, cfg.page_size_bytes, AddressSpace::Physical));
        } else {
            independent = device.issue(request(
                "order-plane1-independent", Tier::HBF, Op::Read, 0.0,
                plane1, cfg.page_size_bytes, AddressSpace::Physical));
            (void)device.issue(request(
                "order-plane0-dependent", Tier::HBF, Op::Read, 0.0,
                plane0, cfg.page_size_bytes, AddressSpace::Physical));
        }
        return std::make_pair(erase.finish_ns, independent.finish_ns);
    };
    const auto dependent_first = run_plane_order(true);
    const auto independent_first = run_plane_order(false);
    if (std::abs(dependent_first.second - independent_first.second) > 1e-6 ||
        dependent_first.second >= dependent_first.first ||
        independent_first.second >= independent_first.first) {
        throw std::runtime_error(
            "hbf-exact-calendar: independent plane timing depended on API call order");
    }
    std::cout << "  erase_plane_call_order_invariant_ok=yes\n";

    // Fast-forwarding stack 0 to a long erase completion must neither block
    // stack 1 nor materialize stack 1's future mapping callback.
    auto stack_cfg = plane_backfill_config();
    stack_cfg.stacks = 2;
    stack_cfg.write_coalescing_enabled = false;
    stack_cfg.mapping_update_ns = 1000000.0;
    HbfDevice stack_device(stack_cfg);
    std::uint64_t stack1_lpn = 0;
    const auto stack_capacity = hbf_page_capacity(stack_device);
    while (stack1_lpn < stack_capacity &&
           stack_device.stack_for_logical_page(stack1_lpn) != 1) {
        ++stack1_lpn;
    }
    if (stack1_lpn == stack_capacity) {
        throw std::runtime_error(
            "hbf-exact-calendar: logical placement did not reach stack 1");
    }
    const auto stack1_lpn_addr = stack1_lpn * stack_cfg.page_size_bytes;
    stack_device.prepopulate_logical_pages({stack1_lpn});
    (void)stack_device.issue(request(
        "stack1-future-write", Tier::HBF, Op::Write, 0.0,
        stack1_lpn_addr, stack_cfg.page_size_bytes, AddressSpace::Logical));
    const auto stack0_erase = stack_device.issue(request(
        "stack0-long-erase", Tier::HBF, Op::Erase, 0.0, 0, 0,
        AddressSpace::Physical));
    (void)stack_device.issue(request(
        "stack0-post-erase-barrier", Tier::HBF, Op::Read, 0.0, 0,
        stack_cfg.page_size_bytes, AddressSpace::Physical));
    const auto stack1_physical_addr = stack_device.encode(HbfAddress{.stack = 1});
    const auto stack1_read = stack_device.issue(request(
        "stack1-early-read", Tier::HBF, Op::Read, 0.0,
        stack1_physical_addr, stack_cfg.page_size_bytes, AddressSpace::Physical));
    std::cout << "  stack0_erase_finish_ns=" << fixed(stack0_erase.finish_ns)
              << " stack1_read_finish_ns=" << fixed(stack1_read.finish_ns)
              << " stack1_read_note=" << stack1_read.note << "\n";
    if (stack_device.stats().invalidations != 0 ||
        stack1_read.finish_ns >= stack0_erase.finish_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: one stack's erase advanced or blocked another stack");
    }
    std::cout << "  erase_stack_isolation_ok=yes\n";

    auto plane_erase_cfg = plane_backfill_config();
    plane_erase_cfg.planes_per_die = 2;
    plane_erase_cfg.blocks_per_plane = 2;
    plane_erase_cfg.t_read_page_ns = 100.0;
    plane_erase_cfg.t_erase_block_ns = 10000.0;
    plane_erase_cfg.ecc_decode_latency_ns = 1.0;
    HbfDevice plane_erase_device(plane_erase_cfg);
    const auto plane0_addr = plane_erase_device.encode(HbfAddress{.plane = 0});
    const auto plane1_addr = plane_erase_device.encode(HbfAddress{.plane = 1});
    const auto plane0_erase = plane_erase_device.issue(request(
        "plane0-long-erase", Tier::HBF, Op::Erase, 0.0,
        plane0_addr, 0, AddressSpace::Physical));
    const auto plane1_read = plane_erase_device.issue(request(
        "plane1-independent-read", Tier::HBF, Op::Read, 0.0,
        plane1_addr, plane_erase_cfg.page_size_bytes,
        AddressSpace::Physical));
    if (plane1_read.finish_ns >= plane0_erase.finish_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: erase blocked an independent plane in the same stack");
    }
    std::cout << "  erase_same_stack_plane_isolation_ok=yes\n";

    auto cross_stack_cfg = plane_backfill_config();
    cross_stack_cfg.stacks = 2;
    cross_stack_cfg.mapping_entries_per_page = 1;
    cross_stack_cfg.t_erase_block_ns = 10000.0;
    HbfDevice cross_stack_device(cross_stack_cfg);
    std::uint64_t cross_stack_first_lpn = 0;
    const auto cross_stack_capacity = hbf_page_capacity(cross_stack_device);
    while (cross_stack_first_lpn + 1 < cross_stack_capacity &&
           (cross_stack_device.stack_for_logical_page(
                cross_stack_first_lpn) == 1 ||
            cross_stack_device.stack_for_logical_page(
                cross_stack_first_lpn + 1) != 1)) {
        ++cross_stack_first_lpn;
    }
    if (cross_stack_first_lpn + 1 >= cross_stack_capacity) {
        throw std::runtime_error(
            "hbf-exact-calendar: no adjacent cross-stack page pair");
    }
    const auto cross_stack_second_lpn = cross_stack_first_lpn + 1;
    cross_stack_device.prepopulate_logical_pages({cross_stack_second_lpn});
    const auto stack1_block0_addr = cross_stack_device.encode(HbfAddress{.stack = 1});
    const auto cross_stack_erase = cross_stack_device.issue(request(
        "cross-stack-page-erase", Tier::HBF, Op::Erase, 0.0,
        stack1_block0_addr, 0, AddressSpace::Physical));
    const auto cross_stack_read = cross_stack_device.issue(request(
        "cross-stack-two-page-read", Tier::HBF, Op::Read, 0.0,
        cross_stack_first_lpn * cross_stack_cfg.page_size_bytes,
        2 * cross_stack_cfg.page_size_bytes, AddressSpace::Logical));
    if (cross_stack_read.note != "unmapped-erased-multi-page-read" ||
        cross_stack_read.physical_bytes != 0 ||
        cross_stack_read.finish_ns < cross_stack_erase.finish_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: multi-page request bypassed a non-first stack erase dependency");
    }
    std::cout << "  multipage_cross_stack_erase_dependency_ok=yes\n";

    // Raw physical ownership is a distinct block role: the FTL allocates its
    // logical page elsewhere, and raw programming into a controller-owned
    // data block fails closed.
    auto raw_cfg = plane_backfill_config();
    raw_cfg.write_coalescing_enabled = false;
    HbfDevice raw_device(raw_cfg);
    (void)raw_device.issue(request(
        "raw-owner-program", Tier::HBF, Op::Write, 0.0, 0,
        raw_cfg.page_size_bytes, AddressSpace::Physical));
    raw_device.prepopulate_logical_pages({0});
    const auto raw_drain = raw_device.drain_pending("raw-owner-drain", 0.0);
    const auto logical_elsewhere = raw_device.issue(request(
        "raw-owner-logical-read", Tier::HBF, Op::Read,
        raw_drain.finish_ns, 0, raw_cfg.page_size_bytes,
        AddressSpace::Logical));
    HbfDevice owned_device(raw_cfg);
    owned_device.prepopulate_logical_pages({0});
    bool raw_into_data_rejected = false;
    try {
        (void)owned_device.issue(request(
            "raw-owner-invalid-mix", Tier::HBF, Op::Write, 0.0,
            raw_cfg.page_size_bytes, raw_cfg.page_size_bytes,
            AddressSpace::Physical));
    } catch (const std::runtime_error&) {
        raw_into_data_rejected = true;
    }
    if (logical_elsewhere.resource_path.find("/block1/") == std::string::npos ||
        !raw_into_data_rejected) {
        throw std::runtime_error(
            "hbf-exact-calendar: raw physical and FTL ownership shared a block");
    }
    std::cout << "  raw_physical_ownership_ok=yes\n";

    HbfDevice static_reject_device(raw_cfg);
    HbfDevice static_reject_control(raw_cfg);
    static_reject_device.prepopulate_logical_pages({0});
    static_reject_control.prepopulate_logical_pages({0});
    const auto legal_static_ppn =
        2 * static_cast<std::uint64_t>(raw_cfg.pages_per_block);
    bool mixed_static_rejected = false;
    try {
        static_reject_device.reserve_static_physical_pages({
            legal_static_ppn,
            0,
        });
    } catch (const std::runtime_error&) {
        mixed_static_rejected = true;
    }
    const auto legal_static_addr = legal_static_ppn * raw_cfg.page_size_bytes;
    const auto post_static_reject = static_reject_device.issue(request(
        "static-reject-post-write", Tier::HBF, Op::Write, 0.0,
        legal_static_addr, raw_cfg.page_size_bytes, AddressSpace::Physical));
    const auto static_reject_fresh = static_reject_control.issue(request(
        "static-reject-control-write", Tier::HBF, Op::Write, 0.0,
        legal_static_addr, raw_cfg.page_size_bytes, AddressSpace::Physical));
    if (!mixed_static_rejected ||
        std::abs(post_static_reject.finish_ns - static_reject_fresh.finish_ns) > 1e-6 ||
        static_reject_device.stats().static_reserved_pages !=
            static_reject_control.stats().static_reserved_pages) {
        throw std::runtime_error(
            "hbf-exact-calendar: rejected static reservation partially mutated state");
    }
    std::cout << "  static_reservation_fail_closed_ok=yes\n";

    // The full table is present from time zero. Three simultaneous lookups
    // occupy only their short issue slots; their long response latencies
    // overlap and no mapping-page flash read exists.
    auto mapping_cfg = plane_backfill_config();
    mapping_cfg.write_coalescing_enabled = false;
    mapping_cfg.ctrl_dram_issue_ns = 10.0;
    mapping_cfg.ctrl_dram_latency_ns = 1000.0;
    mapping_cfg.address_generation_ns = 1.0;
    mapping_cfg.read_buffer_pages = 0;
    HbfDevice mapping_device(mapping_cfg);
    mapping_device.prepopulate_logical_pages({0, 1});
    (void)mapping_device.issue(request(
        "resident-map-a1", Tier::HBF, Op::Read, 0.0, 0,
        mapping_cfg.page_size_bytes, AddressSpace::Logical));
    (void)mapping_device.issue(request(
        "resident-map-b", Tier::HBF, Op::Read, 0.0,
        mapping_cfg.page_size_bytes, mapping_cfg.page_size_bytes,
        AddressSpace::Logical));
    (void)mapping_device.issue(request(
        "resident-map-a2", Tier::HBF, Op::Read, 0.0, 0,
        mapping_cfg.page_size_bytes, AddressSpace::Logical));
    const auto& mapping_stats = mapping_device.stats();
    if (mapping_stats.mapping_lookup_ops != 3 ||
        mapping_stats.mapping_user_lookup_ops != 3 ||
        mapping_stats.mapping_gc_lookup_ops != 0 ||
        mapping_stats.mapping_dram_wait_ops != 2 ||
        std::abs(mapping_stats.mapping_dram_issue_busy_ns - 30.0) > 1e-6 ||
        mapping_stats.mapping_page_programs != 0 ||
        mapping_stats.resident_mapping_table_bytes == 0) {
        throw std::runtime_error(
            "hbf-exact-calendar: resident mapping pipeline diverged");
    }
    std::cout << "  resident_mapping_pipeline_ok=yes\n";

    // A dependent physical read can reserve a future ingress issue slot. A
    // later-called multi-page read at the same host arrival may backfill its
    // short dispatch before that slot, but its longer address-generation
    // phase does not fit. The split must use the actual second reservation;
    // advancing from its requested time would launch page work while ingress
    // is physically occupied by the dependent read.
    auto split_ingress_cfg = plane_backfill_config();
    split_ingress_cfg.write_coalescing_enabled = false;
    split_ingress_cfg.read_buffer_pages = 0;
    split_ingress_cfg.address_generation_ns = 1000000.0;
    HbfDevice split_ingress_device(split_ingress_cfg);
    const auto programmed_addr = split_ingress_device.encode(
        HbfAddress{.block = 0, .page = 0});
    auto future_program_request = request(
        "split-ingress-future-program", Tier::HBF, Op::Write, 0.0,
        programmed_addr, split_ingress_cfg.page_size_bytes,
        AddressSpace::Physical);
    future_program_request.trace.mode = TraceMode::Full;
    (void)split_ingress_device.issue(future_program_request);
    auto dependent_read_request = request(
        "split-ingress-dependent-read", Tier::HBF, Op::Read, 0.0,
        programmed_addr, split_ingress_cfg.page_size_bytes,
        AddressSpace::Physical);
    dependent_read_request.trace.mode = TraceMode::Full;
    const auto dependent_read = split_ingress_device.issue(
        dependent_read_request);
    const auto split_addr = split_ingress_device.encode(
        HbfAddress{.block = 2, .page = 0});
    auto split_read_request = request(
        "split-ingress-multipage-read", Tier::HBF, Op::Read, 0.0,
        split_addr, 2 * split_ingress_cfg.page_size_bytes,
        AddressSpace::Physical);
    split_read_request.trace.mode = TraceMode::Full;
    const auto split_read = split_ingress_device.issue(split_read_request);
    const auto& dependent_issue = find_plane_span(
        dependent_read, "logic_scheduler_issue");
    const auto& split_issue = find_plane_span(
        split_read, "logic_scheduler_issue");
    const auto& split_work = find_plane_span(split_read, "read_split");
    const double expected_split_wait = split_work.start_ns - split_issue.end_ns;
    if (expected_split_wait <= 0.0 ||
        split_work.start_ns + 1e-6 < dependent_issue.end_ns ||
        std::abs(
            (split_work.end_ns - split_work.start_ns) -
            split_ingress_cfg.address_generation_ns) > 1e-6 ||
        split_read.breakdown.ingress_queue_wait_ns + 1e-6 <
            expected_split_wait) {
        throw std::runtime_error(
            "hbf-exact-calendar: multi-page split ignored its actual ingress reservation");
    }
    std::cout << "  multipage_split_ingress_reservation_ok=yes\n";

    auto dirty_drain_cfg = plane_backfill_config();
    dirty_drain_cfg.write_coalescing_enabled = false;
    dirty_drain_cfg.mapping_entries_per_page = 1;
    dirty_drain_cfg.mapping_update_ns = 10000.0;
    HbfDevice dirty_drain_device(dirty_drain_cfg);
    const auto dirty_write = dirty_drain_device.issue(request(
        "resident-map-dirty-future-write", Tier::HBF, Op::Write, 0.0, 0,
        dirty_drain_cfg.page_size_bytes, AddressSpace::Logical));
    const auto dirty_drain = dirty_drain_device.drain_pending(
        "resident-map-dirty-future-drain", 0.0);
    if (dirty_drain_device.stats().mapping_page_programs != 1 ||
        dirty_drain.finish_ns < dirty_write.finish_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: drain lost a future resident mapping update");
    }
    std::cout << "  resident_mapping_future_dirty_drain_ok=yes\n";

    // A flushed WB generation is readable until page_done, so its sole SRAM
    // slot cannot be reassigned at the earlier program_done instant.
    auto wb_lifetime_cfg = plane_backfill_config();
    wb_lifetime_cfg.write_buffer_pages = 1;
    wb_lifetime_cfg.write_buffer_flush_threshold_pages = 1;
    wb_lifetime_cfg.write_coalescing_enabled = true;
    wb_lifetime_cfg.mapping_update_ns = 1000000.0;
    HbfDevice wb_lifetime_device(wb_lifetime_cfg);
    auto wb_a_request = request(
        "wb-generation-a", Tier::HBF, Op::Write, 0.0, 0,
        wb_lifetime_cfg.page_size_bytes, AddressSpace::Logical);
    wb_a_request.trace.mode = TraceMode::Full;
    const auto wb_a = wb_lifetime_device.issue(wb_a_request);
    auto wb_b_request = request(
        "wb-generation-b", Tier::HBF, Op::Write, 100000.0,
        wb_lifetime_cfg.page_size_bytes,
        wb_lifetime_cfg.page_size_bytes, AddressSpace::Logical);
    wb_b_request.trace.mode = TraceMode::Full;
    const auto wb_b = wb_lifetime_device.issue(wb_b_request);
    const auto& wb_a_publish = find_plane_span(
        wb_a, "resident_mapping_update");
    const auto& wb_b_payload = find_plane_span(wb_b, "user/data_in_hbio");
    if (wb_b_payload.start_ns + 1e-6 < wb_a_publish.end_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: WB slot was reused while its old generation remained readable");
    }
    std::cout << "  write_buffer_generation_lifetime_ok=yes\n";

    // A write-buffer flush reads controller state through the same finite
    // logic-die SRAM as a media-read payload. Make the flush too long to fit
    // before an already-reserved future read stage: it must queue behind that
    // stage, and aggregate SRAM busy work must exactly equal canonical SRAM
    // stage work.
    auto wb_sram_cfg = plane_backfill_config();
    wb_sram_cfg.write_buffer_pages = 1;
    wb_sram_cfg.write_buffer_flush_threshold_pages = 1;
    wb_sram_cfg.write_coalescing_enabled = true;
    wb_sram_cfg.read_buffer_pages = 0;
    wb_sram_cfg.mapping_update_ns = 2000.0;
    wb_sram_cfg.t_read_page_ns = 1000.0;
    HbfDevice wb_sram_device(wb_sram_cfg);
    wb_sram_device.reserve_static_physical_pages({0});
    auto future_sram_read_request = request(
        "wb-sram-future-read", Tier::HBF, Op::Read, 0.0, 0,
        wb_sram_cfg.page_size_bytes, AddressSpace::Physical);
    future_sram_read_request.trace.mode = TraceMode::Full;
    const auto future_sram_read = wb_sram_device.issue(
        future_sram_read_request);
    auto contending_flush_request = request(
        "wb-sram-contending-flush", Tier::HBF, Op::Write, 0.0, 0,
        wb_sram_cfg.page_size_bytes, AddressSpace::Logical);
    contending_flush_request.trace.mode = TraceMode::Full;
    const auto contending_flush = wb_sram_device.issue(
        contending_flush_request);
    const auto& future_read_sram = find_plane_span(
        future_sram_read, "user/sram_stage_read");
    const auto& flush_sram = find_plane_span(
        contending_flush, "write_buffer_flush");
    const auto& wb_sram_stats = wb_sram_device.stats();
    const double sram_work_tolerance = 1e-9 * std::max({
        1.0,
        std::abs(wb_sram_stats.sram_busy_ns),
        std::abs(wb_sram_stats.stage_work.sram_staging_ns),
    });
    if (flush_sram.start_ns + 1e-6 < future_read_sram.end_ns ||
        contending_flush.breakdown.scheduler_queue_wait_ns <= 0.0 ||
        std::abs(
            wb_sram_stats.sram_busy_ns -
            wb_sram_stats.stage_work.sram_staging_ns) > sram_work_tolerance) {
        throw std::runtime_error(
            "hbf-exact-calendar: write-buffer flush bypassed finite SRAM scheduling");
    }
    std::cout << "  write_buffer_flush_sram_reservation_ok=yes\n";

    // A full block with an allocated-but-not-yet-programmed page or a valid
    // page whose mapping has not published is pinned. GC must wait for those
    // ownership transitions before relocating the complete live set.
    HbfConfig gc_pin_cfg;
    gc_pin_cfg.stacks = 1;
    gc_pin_cfg.channels_per_stack = 1;
    gc_pin_cfg.dies_per_channel = 1;
    gc_pin_cfg.planes_per_die = 1;
    gc_pin_cfg.blocks_per_plane = 3;
    gc_pin_cfg.pages_per_block = 4;
    gc_pin_cfg.page_size_bytes = 512;
    gc_pin_cfg.oob_bytes_per_page = 0;
    gc_pin_cfg.subarrays_per_plane = 1;
    gc_pin_cfg.media_lanes_per_plane = 1;
    gc_pin_cfg.page_buffer_banks_per_plane = 1;
    gc_pin_cfg.write_coalescing_enabled = false;
    gc_pin_cfg.read_buffer_pages = 0;
    gc_pin_cfg.mapping_entries_per_page = 512;
    gc_pin_cfg.ctrl_dram_bytes = 512;
    gc_pin_cfg.t_read_page_ns = 1.0;
    gc_pin_cfg.t_program_page_ns = 100.0;
    gc_pin_cfg.t_program_verify_ns = 1.0;
    gc_pin_cfg.t_erase_block_ns = 100.0;
    gc_pin_cfg.ecc_decode_latency_ns = 1.0;
    gc_pin_cfg.ecc_encode_latency_ns = 1.0;
    gc_pin_cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    gc_pin_cfg.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    gc_pin_cfg.channel_bandwidth_GBps = 1.0e9;
    gc_pin_cfg.hb_io_bandwidth_GBps = 1.0e9;
    gc_pin_cfg.tsv_bandwidth_GBps = 1.0e9;
    gc_pin_cfg.media_lane_bandwidth_GBps = 1.0e9;
    gc_pin_cfg.page_buffer_bandwidth_GBps = 1.0e9;
    gc_pin_cfg.logic_sram_bandwidth_GBps = 1.0e9;
    gc_pin_cfg.logic_scheduler_issue_ns = 0.001;
    gc_pin_cfg.flash_tsu_issue_ns = 0.001;
    gc_pin_cfg.address_generation_ns = 0.001;
    gc_pin_cfg.ctrl_dram_latency_ns = 0.001;
    gc_pin_cfg.ctrl_dram_issue_ns = 0.001;
    gc_pin_cfg.free_page_allocation_ns = 0.001;
    gc_pin_cfg.mapping_update_ns = 10.0;
    gc_pin_cfg.gc_low_watermark_pages = 4;
    // Force the final allocation into hard pressure while the preceding
    // program/publication is still pending. Hard GC may wait for that commit;
    // soft GC deliberately may not manufacture a victim from future state.
    gc_pin_cfg.gc_hard_watermark_pages = 7;
    gc_pin_cfg.gc_reserved_free_blocks_per_plane = 0;
    HbfDevice gc_pin_device(gc_pin_cfg);
    gc_pin_device.prepopulate_logical_pages({0, 1});
    const auto gc_w0 = gc_pin_device.issue(request(
        "gc-pin-w0", Tier::HBF, Op::Write, 0.0, 0,
        gc_pin_cfg.page_size_bytes, AddressSpace::Logical));
    (void)gc_pin_device.issue(request(
        "gc-pin-w2", Tier::HBF, Op::Write, gc_w0.finish_ns,
        2 * gc_pin_cfg.page_size_bytes, gc_pin_cfg.page_size_bytes,
        AddressSpace::Logical));
    (void)gc_pin_device.issue(request(
        "gc-pin-w3", Tier::HBF, Op::Write, gc_w0.finish_ns,
        3 * gc_pin_cfg.page_size_bytes, gc_pin_cfg.page_size_bytes,
        AddressSpace::Logical));
    const auto gc_pin_drain = gc_pin_device.drain_pending(
        "gc-pin-drain", gc_w0.finish_ns);
    const auto gc_pin_read = gc_pin_device.issue(request(
        "gc-pin-read2", Tier::HBF, Op::Read, gc_pin_drain.finish_ns,
        2 * gc_pin_cfg.page_size_bytes, gc_pin_cfg.page_size_bytes,
        AddressSpace::Logical));
    if (gc_pin_device.stats().gc_runs == 0 ||
        gc_pin_read.note == "unmapped-erased-read") {
        throw std::runtime_error(
            "hbf-exact-calendar: GC erased an in-flight program/publication target");
    }
    std::cout << "  gc_inflight_ownership_pin_ok=yes\n";

    // A drain is a top-level causal barrier just like issue(). A rejected
    // backwards drain must not mutate the device, and a later issue must not
    // travel behind a successful drain.
    HbfDevice causal_device(round_cfg);
    (void)causal_device.issue(request(
        "calendar-causal-read",
        Tier::HBF,
        Op::Read,
        10.0,
        0,
        round_cfg.page_size_bytes,
        AddressSpace::Physical));
    bool backwards_drain_rejected = false;
    try {
        (void)causal_device.drain_pending("calendar-backwards-drain", 9.0);
    } catch (const std::runtime_error&) {
        backwards_drain_rejected = true;
    }
    (void)causal_device.drain_pending("calendar-causal-drain", 20.0);
    bool post_drain_issue_rejected = false;
    try {
        (void)causal_device.issue(request(
            "calendar-post-drain-backwards-issue",
            Tier::HBF,
            Op::Read,
            19.0,
            0,
            round_cfg.page_size_bytes,
            AddressSpace::Physical));
    } catch (const std::runtime_error&) {
        post_drain_issue_rejected = true;
    }
    if (!backwards_drain_rejected || !post_drain_issue_rejected) {
        throw std::runtime_error(
            "hbf-exact-calendar: issue/drain causal order did not fail closed");
    }
    std::cout << "  drain_causal_barrier_ok=yes\n";

    auto drain_barrier_cfg = plane_backfill_config();
    drain_barrier_cfg.write_coalescing_enabled = false;
    drain_barrier_cfg.mapping_update_ns = 3000000.0;
    HbfDevice drain_barrier_device(drain_barrier_cfg);
    (void)drain_barrier_device.issue(request(
        "drain-barrier-write", Tier::HBF, Op::Write, 0.0, 0,
        drain_barrier_cfg.page_size_bytes, AddressSpace::Logical));
    const auto future_drain = drain_barrier_device.drain_pending(
        "drain-barrier-drain", 0.0);
    const auto post_drain_read = drain_barrier_device.issue(request(
        "drain-barrier-read", Tier::HBF, Op::Read, 0.0, 0,
        drain_barrier_cfg.page_size_bytes, AddressSpace::Logical));
    if (post_drain_read.start_ns + 1e-6 < future_drain.finish_ns) {
        throw std::runtime_error(
            "hbf-exact-calendar: post-drain issue observed future state before the drain barrier");
    }
    std::cout << "  drain_future_state_barrier_ok=yes\n";
}

void probe_hbf_ecc_pipeline() {
    std::cout << "\n== HBF pipelined ECC probe ==\n";

    auto base_config = []() {
        HbfConfig cfg;
        cfg.stacks = 1;
        cfg.channels_per_stack = 1;
        cfg.dies_per_channel = 1;
        cfg.planes_per_die = 2;
        cfg.blocks_per_plane = 1;
        cfg.pages_per_block = 128;
        cfg.page_size_bytes = 4096;
        cfg.oob_bytes_per_page = 0;
        cfg.media_lanes_per_plane = 128;
        cfg.subarrays_per_plane = 128;
        cfg.page_buffer_banks_per_plane = 128;
        cfg.t_read_page_ns = 0.001;
        cfg.logic_scheduler_issue_ns = 0.001;
        cfg.flash_tsu_issue_ns = 0.001;
        cfg.channel_bandwidth_GBps = 1.0e9;
        cfg.hb_io_bandwidth_GBps = 1.0e9;
        cfg.tsv_bandwidth_GBps = 1.0e9;
        cfg.media_lane_bandwidth_GBps = 1.0e9;
        cfg.logic_sram_bandwidth_GBps = 1.0e9;
        cfg.page_buffer_bandwidth_GBps = 1.0e9;
        cfg.ecc_decode_latency_ns = 500.0;
        cfg.ecc_encode_latency_ns = 500.0;
        cfg.ecc_decode_raw_bandwidth_GBps_per_die = 64.0;
        cfg.ecc_encode_raw_bandwidth_GBps_per_die = 64.0;
        cfg.batch_activation = false;
        cfg.read_buffer_pages = 0;
        return cfg;
    };
    auto traced_request = [](std::string id, Op op, std::uint64_t addr, std::uint64_t bytes) {
        auto value = request(
            std::move(id), Tier::HBF, op, 0.0, addr, bytes,
            AddressSpace::Physical);
        value.trace.mode = TraceMode::Full;
        return value;
    };
    auto find_span = [](const PhysicalCompletion& completion, const std::string& name)
        -> const TraceSpan& {
        const auto found = std::find_if(
            completion.spans.begin(), completion.spans.end(),
            [&name](const TraceSpan& span) { return span.name == name; });
        if (found == completion.spans.end()) {
            throw std::runtime_error("hbf-ecc-pipeline: missing trace span " + name);
        }
        return *found;
    };
    auto close = [](double lhs, double rhs, double tolerance = 1e-6) {
        return std::abs(lhs - rhs) <= tolerance;
    };

    // Latency and throughput are independent: changing only decode latency
    // moves the first completion by the same amount while the issue slot
    // remains one 4096 B / 64 GB/s = 64 ns interval.
    auto short_cfg = base_config();
    HbfDevice short_device(short_cfg);
    const auto short_addr = short_device.encode(HbfAddress{.plane = 0, .page = 0});
    const auto short_read = short_device.issue(traced_request(
        "ecc-first-500", Op::Read, short_addr, short_cfg.page_size_bytes));
    auto long_cfg = short_cfg;
    long_cfg.ecc_decode_latency_ns = 900.0;
    HbfDevice long_device(long_cfg);
    const auto long_addr = long_device.encode(HbfAddress{.plane = 0, .page = 0});
    const auto long_read = long_device.issue(traced_request(
        "ecc-first-900", Op::Read, long_addr, long_cfg.page_size_bytes));
    const auto& short_issue = find_span(short_read, "user/ecc_decode_issue");
    const auto& short_latency = find_span(short_read, "user/ecc_decode_latency");
    const auto& long_latency = find_span(long_read, "user/ecc_decode_latency");
    if (!close(short_issue.duration_ns(), 64.0) ||
        !close(short_latency.duration_ns(), 500.0) ||
        !close(long_latency.duration_ns(), 900.0) ||
        !close(long_read.finish_ns - short_read.finish_ns, 400.0) ||
        !close(short_read.breakdown.ecc_latency_ns, 500.0)) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: response latency was conflated with initiation interval");
    }
    std::cout << "  ecc_first_read_latency_ok=yes\n";

    // Same-die codewords start one II apart while their 500 ns response
    // windows overlap. The old model incorrectly separated them by 500 ns.
    HbfDevice same_die(short_cfg);
    const auto same_a_addr = same_die.encode(HbfAddress{.plane = 0, .page = 0});
    const auto same_b_addr = same_die.encode(HbfAddress{.plane = 1, .page = 0});
    const auto same_a = same_die.issue(traced_request(
        "ecc-same-die-a", Op::Read, same_a_addr, short_cfg.page_size_bytes));
    const auto same_b = same_die.issue(traced_request(
        "ecc-same-die-b", Op::Read, same_b_addr, short_cfg.page_size_bytes));
    const auto& same_a_issue = find_span(same_a, "user/ecc_decode_issue");
    const auto& same_b_issue = find_span(same_b, "user/ecc_decode_issue");
    const auto& same_a_latency = find_span(same_a, "user/ecc_decode_latency");
    const auto& same_b_latency = find_span(same_b, "user/ecc_decode_latency");
    if (!close(same_b_issue.start_ns - same_a_issue.start_ns, 64.0) ||
        same_b_latency.start_ns >= same_a_latency.end_ns) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: same-die decode did not pipeline at the configured II");
    }
    std::cout << "  ecc_same_die_ii_ok=yes\n";

    // Different dies own independent issue ports. Put them on independent
    // channels as well so no shared flash-channel resource clouds the check.
    auto cross_cfg = short_cfg;
    cross_cfg.channels_per_stack = 2;
    cross_cfg.dies_per_channel = 1;
    cross_cfg.planes_per_die = 1;
    HbfDevice cross_die(cross_cfg);
    const auto cross_a_addr = cross_die.encode(HbfAddress{.channel = 0, .page = 0});
    const auto cross_b_addr = cross_die.encode(HbfAddress{.channel = 1, .page = 0});
    const auto cross_a = cross_die.issue(traced_request(
        "ecc-cross-die-a", Op::Read, cross_a_addr, cross_cfg.page_size_bytes));
    const auto cross_b = cross_die.issue(traced_request(
        "ecc-cross-die-b", Op::Read, cross_b_addr, cross_cfg.page_size_bytes));
    const auto& cross_a_issue = find_span(cross_a, "user/ecc_decode_issue");
    const auto& cross_b_issue = find_span(cross_b, "user/ecc_decode_issue");
    const auto& cross_a_latency = find_span(cross_a, "user/ecc_decode_latency");
    const auto& cross_b_latency = find_span(cross_b, "user/ecc_decode_latency");
    if (std::abs(cross_b_issue.start_ns - cross_a_issue.start_ns) >= 64.0 ||
        cross_b.breakdown.ecc_queue_wait_ns != 0.0 ||
        cross_b_latency.start_ns >= cross_a_latency.end_ns ||
        cross_die.stats().active_ecc_dies != 2) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: independent dies incorrectly shared an ECC issue port");
    }
    std::cout << "  ecc_cross_die_parallel_ok=yes\n";

    // Raw bandwidth includes OOB: at 64 GB/s, 4096 B and 4608 B codewords
    // require 64 ns and 72 ns initiation intervals respectively.
    auto run_oob_pair = [&](std::uint64_t oob_bytes) {
        auto cfg = short_cfg;
        cfg.oob_bytes_per_page = oob_bytes;
        HbfDevice device(cfg);
        const auto a_addr = device.encode(HbfAddress{.plane = 0, .page = 0});
        const auto b_addr = device.encode(HbfAddress{.plane = 1, .page = 0});
        const auto a = device.issue(traced_request(
            "ecc-oob-a", Op::Read, a_addr, cfg.page_size_bytes));
        const auto b = device.issue(traced_request(
            "ecc-oob-b", Op::Read, b_addr, cfg.page_size_bytes));
        const double spacing =
            find_span(b, "user/ecc_decode_issue").start_ns -
            find_span(a, "user/ecc_decode_issue").start_ns;
        return std::pair<double, HbfStats>{spacing, device.stats()};
    };
    const auto [plain_spacing, plain_stats] = run_oob_pair(0);
    const auto [oob_spacing, oob_stats] = run_oob_pair(512);
    if (!close(plain_spacing, 64.0) || !close(oob_spacing, 72.0) ||
        plain_stats.ecc_codeword_bytes != 8192 ||
        oob_stats.ecc_codeword_bytes != 9216) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: ECC initiation interval ignored codeword OOB bytes");
    }
    std::cout << "  ecc_oob_codeword_ok=yes\n";

    // The flash-side fabric carries the raw codeword, but decoded SRAM and
    // the external HBIO carry payload only. Lock both byte counts and the
    // causal boundary around ECC so an OOB change cannot silently derate the
    // advertised payload interface or move decode after HBIO again.
    auto path_cfg = short_cfg;
    path_cfg.oob_bytes_per_page = 512;
    path_cfg.channel_bandwidth_GBps = 64.0;
    path_cfg.tsv_bandwidth_GBps = 64.0;
    path_cfg.logic_sram_bandwidth_GBps = 64.0;
    path_cfg.hb_io_bandwidth_GBps = 64.0;
    HbfDevice read_path(path_cfg);
    const auto read_path_addr = read_path.encode(HbfAddress{.plane = 0, .page = 0});
    const auto path_read = read_path.issue(traced_request(
        "ecc-raw-payload-read", Op::Read, read_path_addr, path_cfg.page_size_bytes));
    const auto& read_lane = find_span(path_read, "user/array_to_page_buffer");
    const auto& read_page_buffer = find_span(path_read, "user/page_buffer_out");
    const auto& read_channel = find_span(path_read, "user/data_out_channel");
    const auto& read_tsv = find_span(path_read, "user/data_out_tsv");
    const auto& read_ecc = find_span(path_read, "user/ecc_decode_latency");
    const auto& read_sram = find_span(path_read, "user/sram_stage_read");
    const auto& read_hbio = find_span(path_read, "user/data_out_hbio");

    HbfDevice write_path(path_cfg);
    const auto write_path_addr = write_path.encode(HbfAddress{.plane = 0, .page = 0});
    const auto path_write = write_path.issue(traced_request(
        "ecc-raw-payload-write", Op::Write, write_path_addr, path_cfg.page_size_bytes));
    const auto& write_hbio = find_span(path_write, "user/data_in_hbio");
    const auto& write_sram = find_span(path_write, "user/sram_stage_write");
    const auto& write_ecc = find_span(path_write, "user/ecc_encode_latency");
    const auto& write_tsv = find_span(path_write, "user/data_in_tsv");
    const auto& write_channel = find_span(path_write, "user/data_in_channel");

    if (read_lane.detail.find("4608B raw") == std::string::npos ||
        read_page_buffer.detail.find("4608B raw") == std::string::npos ||
        read_lane.end_ns > read_page_buffer.start_ns ||
        read_page_buffer.end_ns > read_channel.start_ns ||
        !close(read_channel.duration_ns(), 72.0) ||
        !close(read_tsv.duration_ns(), 72.0) ||
        !close(read_sram.duration_ns(), 64.0) ||
        !close(read_hbio.duration_ns(), 64.0) ||
        read_channel.end_ns > read_tsv.start_ns ||
        read_tsv.end_ns > read_ecc.start_ns ||
        read_ecc.end_ns > read_sram.start_ns ||
        read_sram.end_ns > read_hbio.start_ns ||
        !close(write_hbio.duration_ns(), 64.0) ||
        !close(write_sram.duration_ns(), 64.0) ||
        !close(write_tsv.duration_ns(), 72.0) ||
        !close(write_channel.duration_ns(), 72.0) ||
        write_hbio.end_ns > write_sram.start_ns ||
        write_sram.end_ns > write_ecc.start_ns ||
        write_ecc.end_ns > write_tsv.start_ns ||
        write_tsv.end_ns > write_channel.start_ns) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: raw/payload byte semantics or ECC path order diverged");
    }
    std::cout << "  ecc_raw_payload_path_ok=yes\n";

    // GC, RMW, and buffered destage are logic-die-internal data movements.
    // Resident mapping lookup touches only controller DRAM, while only the
    // foreground payload may cross HBIO. Likewise, a 64 B coalesced write
    // crosses HBIO once at admission; its later full-page RMW/program stays
    // internal until raw TSV/channel transfer.
    auto route_cfg = short_cfg;
    route_cfg.blocks_per_plane = 4;
    route_cfg.mapping_entries_per_page = 1;
    route_cfg.read_buffer_pages = 0;
    route_cfg.write_coalescing_enabled = true;
    route_cfg.write_buffer_pages = 4;
    HbfDevice route_device(route_cfg);
    route_device.prepopulate_logical_pages({0});
    auto route_read_request = request(
        "ecc-route-read", Tier::HBF, Op::Read, 0.0, 0,
        route_cfg.page_size_bytes);
    route_read_request.trace.mode = TraceMode::Full;
    const auto route_read = route_device.issue(route_read_request);
    auto route_write_request = request(
        "ecc-route-write-64B", Tier::HBF, Op::Write, route_read.finish_ns,
        0, 64);
    route_write_request.trace.mode = TraceMode::Full;
    const auto route_write = route_device.issue(route_write_request);
    const auto route_drain = route_device.drain_pending(
        "ecc-route-drain", route_write.finish_ns, {.mode = TraceMode::Full});
    auto has_span = [](const PhysicalCompletion& completion, const std::string& name) {
        return std::any_of(
            completion.spans.begin(), completion.spans.end(),
            [&name](const TraceSpan& span) { return span.name == name; });
    };
    auto has_hbio_category = [](const PhysicalCompletion& completion) {
        return std::any_of(
            completion.spans.begin(), completion.spans.end(),
            [](const TraceSpan& span) { return span.category == "hbio"; });
    };
    const auto& admitted_payload = find_span(route_write, "user/data_in_hbio");
    const double expected_write_hbio_ns = hbfsim::physical::transfer_time_ns(
        route_cfg.command_address_bytes + 64, route_cfg.hb_io_bandwidth_GBps);
    if (!has_span(route_read, "resident_mapping_lookup") ||
        has_span(route_read, "mapping/ecc_decode_latency") ||
        has_span(route_read, "mapping/sram_stage_read") ||
        has_span(route_read, "mapping/cmd_addr_hbio") ||
        has_span(route_read, "mapping/data_out_hbio") ||
        !has_span(route_read, "user/data_out_hbio") ||
        admitted_payload.detail.find("64B payload") == std::string::npos ||
        !close(route_write.breakdown.hb_io_transfer_ns, expected_write_hbio_ns) ||
        has_hbio_category(route_drain) ||
        !has_span(route_drain, "user/ecc_decode_latency") ||
        !has_span(route_drain, "partial_page_buffer_overlay") ||
        !has_span(route_drain, "user/ecc_encode_latency") ||
        !has_span(route_drain, "mapping/ecc_encode_latency")) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: internal controller data incorrectly crossed HBIO");
    }
    std::cout << "  ecc_internal_route_ok=yes\n";

    HbfDevice erased_fill_device(route_cfg);
    auto erased_fill_write_request = request(
        "ecc-erased-fill-write-64B", Tier::HBF, Op::Write, 0.0, 0, 64);
    erased_fill_write_request.trace.mode = TraceMode::Full;
    const auto erased_fill_write = erased_fill_device.issue(
        erased_fill_write_request);
    auto erased_fill_read_request = request(
        "ecc-erased-fill-read-128B",
        Tier::HBF,
        Op::Read,
        erased_fill_write.finish_ns,
        0,
        128);
    erased_fill_read_request.trace.mode = TraceMode::Full;
    const auto erased_fill_read = erased_fill_device.issue(
        erased_fill_read_request);
    const auto erased_fill_drain = erased_fill_device.drain_pending(
        "ecc-erased-fill-drain",
        erased_fill_read.finish_ns,
        {.mode = TraceMode::Full});
    const auto& erased_fill_payload = find_span(
        erased_fill_write, "user/data_in_hbio");
    if (erased_fill_payload.detail.find("64B payload") == std::string::npos ||
        !has_span(erased_fill_read, "write_buffer_read_assemble") ||
        !has_span(erased_fill_drain, "partial_page_erased_fill") ||
        has_hbio_category(erased_fill_drain)) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: unmapped partial write lacked explicit erased fill");
    }
    std::cout << "  ecc_partial_erased_fill_ok=yes\n";

    HbfDevice unmapped_read_device(route_cfg);
    auto unmapped_read_request = request(
        "ecc-unmapped-read-128B", Tier::HBF, Op::Read, 0.0, 0, 128,
        AddressSpace::Logical);
    unmapped_read_request.trace.mode = TraceMode::Full;
    const auto unmapped_read = unmapped_read_device.issue(
        unmapped_read_request);
    const auto& unmapped_sram = find_span(
        unmapped_read, "unmapped_read_erased_fill");
    const auto& unmapped_hbio = find_span(
        unmapped_read, "unmapped_read_hbio_out");
    const auto unmapped_stats = unmapped_read_device.stats();
    if (unmapped_read.note != "unmapped-erased-read" ||
        !close(
            unmapped_sram.duration_ns(),
            hbfsim::physical::transfer_time_ns(
                128, route_cfg.logic_sram_bandwidth_GBps)) ||
        !close(
            unmapped_hbio.duration_ns(),
            hbfsim::physical::transfer_time_ns(
                128, route_cfg.hb_io_bandwidth_GBps)) ||
        has_span(unmapped_read, "user/ecc_decode_latency") ||
        unmapped_stats.ecc_decode_ops != 0 ||
        unmapped_stats.physical_read_bytes != 0) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: unmapped read did not return exact erased payload");
    }
    std::cout << "  ecc_unmapped_erased_read_ok=yes\n";

    auto invalid_cfg = path_cfg;
    invalid_cfg.blocks_per_plane = 2;
    HbfDevice rejected_device(invalid_cfg);
    HbfDevice fresh_device(invalid_cfg);
    const auto protected_addr = rejected_device.encode(
        HbfAddress{.plane = 0, .block = 0, .page = 0});
    const auto fresh_protected_addr = fresh_device.encode(
        HbfAddress{.plane = 0, .block = 0, .page = 0});
    rejected_device.reserve_static_physical_pages({
        protected_addr / invalid_cfg.page_size_bytes});
    fresh_device.reserve_static_physical_pages({
        fresh_protected_addr / invalid_cfg.page_size_bytes});
    bool invalid_write_rejected = false;
    try {
        (void)rejected_device.issue(traced_request(
            "ecc-invalid-static-write",
            Op::Write,
            protected_addr,
            invalid_cfg.page_size_bytes));
    } catch (const std::runtime_error&) {
        invalid_write_rejected = true;
    }
    const auto legal_addr = rejected_device.encode(
        HbfAddress{.plane = 0, .block = 1, .page = 0});
    const auto fresh_legal_addr = fresh_device.encode(
        HbfAddress{.plane = 0, .block = 1, .page = 0});
    const auto post_rejection = rejected_device.issue(traced_request(
        "ecc-post-rejection-write",
        Op::Write,
        legal_addr,
        invalid_cfg.page_size_bytes));
    const auto fresh_write = fresh_device.issue(traced_request(
        "ecc-fresh-write",
        Op::Write,
        fresh_legal_addr,
        invalid_cfg.page_size_bytes));
    if (!invalid_write_rejected ||
        !close(post_rejection.finish_ns, fresh_write.finish_ns) ||
        !close(
            post_rejection.breakdown.hb_io_transfer_ns,
            fresh_write.breakdown.hb_io_transfer_ns) ||
        rejected_device.stats().ecc_encode_ops !=
            fresh_device.stats().ecc_encode_ops) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: rejected physical write mutated device timelines");
    }

    HbfDevice duplicate_device(invalid_cfg);
    HbfDevice duplicate_control(invalid_cfg);
    const auto duplicate_addr = duplicate_device.encode(
        HbfAddress{.plane = 0, .block = 0, .page = 0});
    const auto duplicate_control_addr = duplicate_control.encode(
        HbfAddress{.plane = 0, .block = 0, .page = 0});
    const auto first_duplicate_device = duplicate_device.issue(traced_request(
        "ecc-first-physical-program",
        Op::Write,
        duplicate_addr,
        invalid_cfg.page_size_bytes));
    const auto first_duplicate_control = duplicate_control.issue(traced_request(
        "ecc-first-physical-program-control",
        Op::Write,
        duplicate_control_addr,
        invalid_cfg.page_size_bytes));
    bool inflight_duplicate_rejected = false;
    try {
        (void)duplicate_device.issue(traced_request(
            "ecc-inflight-duplicate-program",
            Op::Write,
            duplicate_addr,
            invalid_cfg.page_size_bytes));
    } catch (const std::runtime_error&) {
        inflight_duplicate_rejected = true;
    }
    bool committed_duplicate_rejected = false;
    try {
        auto committed_duplicate = traced_request(
            "ecc-committed-duplicate-program",
            Op::Write,
            duplicate_addr,
            invalid_cfg.page_size_bytes);
        committed_duplicate.arrival_ns = first_duplicate_device.finish_ns;
        (void)duplicate_device.issue(committed_duplicate);
    } catch (const std::runtime_error&) {
        committed_duplicate_rejected = true;
    }
    const auto next_addr = duplicate_device.encode(
        HbfAddress{.plane = 0, .block = 0, .page = 1});
    const auto next_control_addr = duplicate_control.encode(
        HbfAddress{.plane = 0, .block = 0, .page = 1});
    auto next_request = traced_request(
        "ecc-next-sequential-program",
        Op::Write,
        next_addr,
        invalid_cfg.page_size_bytes);
    next_request.arrival_ns = first_duplicate_device.finish_ns;
    auto next_control_request = traced_request(
        "ecc-next-sequential-program-control",
        Op::Write,
        next_control_addr,
        invalid_cfg.page_size_bytes);
    next_control_request.arrival_ns = first_duplicate_control.finish_ns;
    const auto next_after_rejections = duplicate_device.issue(next_request);
    const auto next_control = duplicate_control.issue(next_control_request);

    HbfDevice future_invalid(invalid_cfg);
    const auto future_protected = future_invalid.encode(
        HbfAddress{.plane = 0, .block = 0, .page = 0});
    future_invalid.reserve_static_physical_pages({
        future_protected / invalid_cfg.page_size_bytes});
    bool future_invalid_rejected = false;
    try {
        auto invalid_at_100 = traced_request(
            "ecc-future-invalid",
            Op::Write,
            future_protected,
            invalid_cfg.page_size_bytes);
        invalid_at_100.arrival_ns = 100.0;
        (void)future_invalid.issue(invalid_at_100);
    } catch (const std::runtime_error&) {
        future_invalid_rejected = true;
    }
    bool earlier_after_invalid_rejected = false;
    try {
        const auto earlier_addr = future_invalid.encode(
            HbfAddress{.plane = 0, .block = 1, .page = 0});
        auto earlier = traced_request(
            "ecc-earlier-after-invalid",
            Op::Write,
            earlier_addr,
            invalid_cfg.page_size_bytes);
        earlier.arrival_ns = 50.0;
        (void)future_invalid.issue(earlier);
    } catch (const std::runtime_error&) {
        earlier_after_invalid_rejected = true;
    }
    if (!inflight_duplicate_rejected || !committed_duplicate_rejected ||
        !close(next_after_rejections.finish_ns, next_control.finish_ns) ||
        !future_invalid_rejected || !earlier_after_invalid_rejected) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: physical program reservation was not fail-closed");
    }
    std::cout << "  ecc_invalid_write_no_side_effect_ok=yes\n";

    auto erase_program_cfg = invalid_cfg;
    erase_program_cfg.t_erase_block_ns = 1000.0;
    HbfDevice erase_then_program(erase_program_cfg);
    const auto erase_addr = erase_then_program.encode(
        HbfAddress{.plane = 0, .block = 0, .page = 0});
    const auto physical_erase = erase_then_program.issue(traced_request(
        "ecc-physical-erase",
        Op::Erase,
        erase_addr,
        0));
    const auto program_after_erase = erase_then_program.issue(traced_request(
        "ecc-program-after-inflight-erase",
        Op::Write,
        erase_addr,
        erase_program_cfg.page_size_bytes));
    const auto erase_program_drain = erase_then_program.drain_pending(
        "ecc-erase-program-drain",
        program_after_erase.finish_ns,
        {.mode = TraceMode::Full});
    if (program_after_erase.start_ns < physical_erase.finish_ns ||
        erase_program_drain.finish_ns < program_after_erase.finish_ns ||
        erase_then_program.stats().ecc_encode_ops != 1) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: program did not wait for in-flight physical erase");
    }
    std::cout << "  ecc_erase_then_program_ok=yes\n";

    // Saturation: latency remains 500 ns, but 128 completions advance at
    // 64 ns/codeword and reach the payload ceiling of 64 GB/s.
    auto saturation_cfg = short_cfg;
    saturation_cfg.planes_per_die = 1;
    HbfDevice saturation(saturation_cfg);
    std::vector<double> decode_finishes;
    decode_finishes.reserve(128);
    for (std::uint32_t page = 0; page < 128; ++page) {
        const auto addr = saturation.encode(HbfAddress{.page = page});
        const auto completion = saturation.issue(traced_request(
            "ecc-saturation-" + std::to_string(page),
            Op::Read,
            addr,
            saturation_cfg.page_size_bytes));
        decode_finishes.push_back(
            find_span(completion, "user/ecc_decode_latency").end_ns);
    }
    std::sort(decode_finishes.begin(), decode_finishes.end());
    const auto& saturation_stats = saturation.stats();
    const double payload_GBps =
        static_cast<double>(saturation_cfg.page_size_bytes) / 64.0;
    if (!close(decode_finishes.back() - decode_finishes.front(), 127.0 * 64.0) ||
        !close(payload_GBps, 64.0) ||
        !close(saturation_stats.ecc_issue_busy_ns, 128.0 * 64.0) ||
        saturation_stats.ecc_decode_ops != 128 ||
        saturation_stats.ecc_codeword_bytes != 128ull * 4096ull ||
        saturation_stats.max_ecc_inflight_per_die != 8) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: saturated decode rate did not follow ECC initiation interval");
    }
    std::cout << "  ecc_saturation_ok=yes payload_GBps="
              << fixed(payload_GBps, 3) << "\n";

    // Encode uses the same pipeline semantics and shares the issue port with
    // decode. A read and a write on different planes may overlap in latency,
    // but their issue intervals may never overlap.
    HbfDevice mixed(short_cfg);
    const auto mixed_write_addr = mixed.encode(HbfAddress{.plane = 0, .page = 0});
    const auto mixed_read_addr = mixed.encode(HbfAddress{.plane = 1, .page = 0});
    const auto mixed_write = mixed.issue(traced_request(
        "ecc-mixed-write", Op::Write, mixed_write_addr, short_cfg.page_size_bytes));
    const auto mixed_read = mixed.issue(traced_request(
        "ecc-mixed-read", Op::Read, mixed_read_addr, short_cfg.page_size_bytes));
    const auto& encode_issue = find_span(mixed_write, "user/ecc_encode_issue");
    const auto& decode_issue = find_span(mixed_read, "user/ecc_decode_issue");
    const bool disjoint = encode_issue.end_ns <= decode_issue.start_ns ||
        decode_issue.end_ns <= encode_issue.start_ns;
    if (!disjoint || mixed.stats().ecc_encode_ops != 1 ||
        mixed.stats().ecc_decode_ops != 1) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: shared decode/encode issue-port policy diverged");
    }
    std::cout << "  ecc_shared_read_write_issue_ok=yes\n";
    print_rows({
        short_read, same_a, same_b, cross_a, cross_b,
        path_read, path_write, route_read, route_write, route_drain,
        mixed_write, mixed_read});
    print_hbf_stats(saturation);

    bool zero_bw_rejected = false;
    bool nonfinite_bw_rejected = false;
    bool latency_shorter_than_codeword_rejected = false;
    try {
        auto invalid = short_cfg;
        invalid.ecc_decode_raw_bandwidth_GBps_per_die = 0.0;
        (void)HbfDevice(invalid);
    } catch (const std::runtime_error&) {
        zero_bw_rejected = true;
    }
    try {
        auto invalid = short_cfg;
        invalid.ecc_encode_raw_bandwidth_GBps_per_die =
            std::numeric_limits<double>::quiet_NaN();
        (void)HbfDevice(invalid);
    } catch (const std::runtime_error&) {
        nonfinite_bw_rejected = true;
    }
    try {
        auto invalid = short_cfg;
        invalid.ecc_decode_latency_ns = 1.0;
        (void)HbfDevice(invalid);
    } catch (const std::runtime_error&) {
        latency_shorter_than_codeword_rejected = true;
    }
    if (!zero_bw_rejected || !nonfinite_bw_rejected ||
        !latency_shorter_than_codeword_rejected) {
        throw std::runtime_error(
            "hbf-ecc-pipeline: invalid ECC configurations did not fail closed");
    }
    std::cout << "  ecc_invalid_config_rejected=yes\n";
}

PhysicalCompletion run_hbf_io(double hb_io_gbps) {
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.media_lanes_per_plane = 16;
    cfg.subarrays_per_plane = 16;
    cfg.page_buffer_banks_per_plane = 16;
    cfg.hb_io_bandwidth_GBps = hb_io_gbps;
    cfg.tsv_bandwidth_GBps = 1.0e9;
    cfg.channel_bandwidth_GBps = 1.0e9;
    cfg.media_lane_bandwidth_GBps = 1.0e9;
    cfg.logic_sram_bandwidth_GBps = 1.0e9;
    cfg.page_buffer_bandwidth_GBps = 1.0e9;
    cfg.ecc_decode_latency_ns = 1.0;
    cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    cfg.t_read_page_ns = 100.0;
    cfg.read_buffer_pages = 0;
    HbfDevice hbf(cfg);
    const auto addr = hbf.encode(HbfAddress{.block = 0, .page = 0});
    auto c = hbf.issue(request("large-read", Tier::HBF, Op::Read, 0,
        addr, 1ull << 20, AddressSpace::Physical));
    print_rows({c});
    print_hbf_stats(hbf);
    return c;
}

void probe_hbf_io() {
    std::cout << "\n== HBF HB-IO bottleneck probe ==\n";
    std::cout << "  fast HB IO:\n";
    const auto fast = run_hbf_io(4096.0);
    std::cout << "  slow HB IO:\n";
    const auto slow = run_hbf_io(16.0);
    std::cout << "  fast_finish_ns = " << fixed(fast.finish_ns) << "\n";
    std::cout << "  slow_finish_ns = " << fixed(slow.finish_ns) << "\n";
    if (slow.finish_ns <= fast.finish_ns ||
        slow.breakdown.hb_io_transfer_ns <=
            200.0 * fast.breakdown.hb_io_transfer_ns ||
        slow.breakdown.hb_io_transfer_ns <= slow.breakdown.tsv_transfer_ns) {
        throw std::runtime_error(
            "hbf-io: isolated HBIO bandwidth did not control the slow path");
    }
    std::cout << "  hbio_isolated_bottleneck_ok=yes\n";
    std::cout << "  expectation: only HBIO bandwidth changes; raw TSV/channel/ECC remain non-binding.\n";
}

void probe_hbf_rounding() {
    std::cout << "\n== HBF page rounding probe ==\n";
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.page_size_bytes = 2048;
    HbfDevice hbf(cfg);
    if (hbf.stats().waf()) {
        throw std::runtime_error(
            "hbf-rounding: zero-denominator WAF must be undefined");
    }
    const auto addr = std::uint64_t{128};
    auto c = hbf.issue(request("128B-subpage-write", Tier::HBF, Op::Write, 0, addr, 128));
    auto drain = hbf.drain_pending("drain-dirty-mapping", c.finish_ns, {.mode = next_request_trace_mode()});
    print_rows({c, drain});
    print_hbf_stats(hbf);
    const auto waf = hbf.stats().waf();
    if (!waf || std::abs(*waf - 32.0) > 1e-12 ||
        hbf.stats().data_programs != 1 ||
        hbf.stats().page_programs != 2) {
        throw std::runtime_error(
            "hbf-rounding: WAF accounting diverged");
    }
    std::cout << "  waf_denominator_zero_is_na=yes\n";
    std::cout << "  canonical_waf_ok=yes\n";
    std::cout << "  expectation: 128 logical bytes cause one 2048B data program plus one "
                 "2048B mapping program: WAF=(2048+2048)/128=32.\n";
}

void probe_hbf_resident_mapping() {
    std::cout << "\n== HBF resident mapping probe ==\n";
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.blocks_per_plane = 8;
    cfg.pages_per_block = 4;
    cfg.page_size_bytes = 512;
    cfg.mapping_entries_per_page = 1;
    cfg.read_buffer_pages = 0;
    HbfDevice hbf(cfg);
    hbf.prepopulate_logical_pages({0, 1});

    std::vector<PhysicalCompletion> rows;
    rows.push_back(hbf.issue(request(
        "read-lpn0-resident-map", Tier::HBF, Op::Read, 0, 0, 512)));
    rows.push_back(hbf.issue(request(
        "reread-lpn0-resident-map", Tier::HBF, Op::Read, 0, 0, 512)));
    rows.push_back(hbf.issue(request(
        "read-lpn1-resident-map", Tier::HBF, Op::Read, 0, 512, 512)));
    print_rows(rows);
    print_hbf_stats(hbf);
    const auto& stats = hbf.stats();
    if (stats.mapping_lookup_ops != 3 ||
        stats.mapping_user_lookup_ops != 3 ||
        stats.page_reads != 3 ||
        stats.physical_read_bytes != 3 * cfg.page_size_bytes ||
        stats.mapping_page_programs != 0 ||
        stats.resident_mapping_table_bytes != hbf.config().ctrl_dram_bytes) {
        throw std::runtime_error(
            "hbf-resident-mapping: lookup or capacity accounting diverged");
    }
    std::cout << "  resident_mapping_no_flash_read_ok=yes\n";
    std::cout << "  expectation: all three reads use the pre-resident table; "
                 "only three data pages reach flash.\n";
}

void probe_hbf_gc() {
    std::cout << "\n== HBF mapping invalidation + GC probe ==\n";
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.blocks_per_plane = 4;
    cfg.pages_per_block = 2;
    cfg.page_size_bytes = 512;
    // Mapping pages persist on eviction/drain (not per write), so user data
    // programs alone must reach the watermark; block0 keeps lpn1 valid so the
    // GC victim has one page to relocate.
    cfg.gc_low_watermark_pages = 5;
    HbfDevice hbf(cfg);

    std::vector<PhysicalCompletion> rows;
    rows.push_back(hbf.issue(request("write-lpn0", Tier::HBF, Op::Write, 0, 0, 512)));
    rows.push_back(hbf.issue(request(
        "write-lpn1", Tier::HBF, Op::Write, rows.back().finish_ns, 512, 512)));
    rows.push_back(hbf.issue(request(
        "rewrite-lpn0", Tier::HBF, Op::Write, rows.back().finish_ns, 0, 512)));
    rows.push_back(hbf.issue(request(
        "write-lpn2-gc", Tier::HBF, Op::Write, rows.back().finish_ns, 1024, 512)));
    print_rows(rows);
    print_hbf_stats(hbf);
    std::cout << "  expectation: rewrite invalidates old PPN; low free pages trigger GC relocation + erase.\n";

    // Relocation of data pages is itself a resident L2P update. Start with a
    // clean checkpoint for VPN0, force a three-page relocation, and then
    // checkpoint VPN0 plus the foreground write's VPN1.
    HbfConfig persistence_cfg;
    persistence_cfg.channels_per_stack = 1;
    persistence_cfg.dies_per_channel = 1;
    persistence_cfg.planes_per_die = 1;
    persistence_cfg.blocks_per_plane = 5;
    persistence_cfg.pages_per_block = 4;
    persistence_cfg.page_size_bytes = 512;
    persistence_cfg.oob_bytes_per_page = 0;
    persistence_cfg.mapping_entries_per_page = 4;
    persistence_cfg.read_buffer_pages = 0;
    persistence_cfg.gc_low_watermark_pages = 8;
    persistence_cfg.ecc_decode_latency_ns = 1.0;
    persistence_cfg.ecc_encode_latency_ns = 1.0;
    persistence_cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    persistence_cfg.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    HbfDevice persistence_device(persistence_cfg);
    persistence_device.prepopulate_logical_pages({0, 1, 2, 3});
    const auto persistence_rewrite = persistence_device.issue(request(
        "gc-persistence-rewrite", Tier::HBF, Op::Write, 0.0, 0,
        persistence_cfg.page_size_bytes, AddressSpace::Logical));
    const auto persistence_drain1 = persistence_device.drain_pending(
        "gc-persistence-drain-1",
        persistence_rewrite.finish_ns,
        {.mode = TraceMode::Full});
    const auto mapping_programs_after_drain1 =
        persistence_device.stats().mapping_page_programs;
    auto persistence_trigger_request = request(
        "gc-persistence-trigger", Tier::HBF, Op::Write,
        persistence_drain1.finish_ns,
        4 * persistence_cfg.page_size_bytes,
        persistence_cfg.page_size_bytes,
        AddressSpace::Logical);
    persistence_trigger_request.trace.mode = TraceMode::Full;
    const auto persistence_trigger = persistence_device.issue(
        persistence_trigger_request);
    const auto persistence_drain2 = persistence_device.drain_pending(
        "gc-persistence-drain-2",
        persistence_trigger.finish_ns,
        {.mode = TraceMode::Full});
    const auto persistence_stats = persistence_device.stats();
    const bool traced_gc_resident_update = std::any_of(
        persistence_drain1.spans.begin(),
        persistence_drain1.spans.end(),
        [](const TraceSpan& span) {
            return span.name == "gc_resident_mapping_update";
        }) || std::any_of(
            persistence_trigger.spans.begin(),
            persistence_trigger.spans.end(),
            [](const TraceSpan& span) {
                return span.name == "gc_resident_mapping_update";
            }) || std::any_of(
            persistence_drain2.spans.begin(),
            persistence_drain2.spans.end(),
            [](const TraceSpan& span) {
                return span.name == "gc_resident_mapping_update";
            });
    std::cout << "  persistence_gc_runs=" << persistence_stats.gc_runs
              << " relocations=" << persistence_stats.gc_relocations
              << " mapping_programs=" << persistence_stats.mapping_page_programs
              << " resident_gc_updates="
              << persistence_stats.mapping_gc_update_ops
              << " traced_resident_update="
              << (traced_gc_resident_update ? "yes" : "no")
              << "\n";
    if (persistence_stats.gc_runs != 1 ||
        persistence_stats.gc_relocations != 3 ||
        mapping_programs_after_drain1 != 2 ||
        persistence_stats.mapping_page_programs != 3 ||
        persistence_stats.mapping_gc_update_ops !=
            persistence_stats.gc_data_relocations ||
        !traced_gc_resident_update) {
        throw std::runtime_error(
            "hbf-gc: relocation bypassed resident mapping or lost checkpoint");
    }
    std::cout << "  gc_relocation_mapping_persistence_ok=yes\n";

    // The soft watermark protects whole-block relocation capacity, while an
    // append into an already-active Mapping block consumes none of that
    // capacity. The old page-granular/free-block-only test subtracted one page
    // on every append and then relocated blocks 0, 1, and 6 unnecessarily.
    HbfConfig logical_headroom_cfg;
    logical_headroom_cfg.channels_per_stack = 1;
    logical_headroom_cfg.dies_per_channel = 1;
    logical_headroom_cfg.planes_per_die = 1;
    logical_headroom_cfg.blocks_per_plane = 8;
    logical_headroom_cfg.pages_per_block = 4;
    logical_headroom_cfg.page_size_bytes = 512;
    logical_headroom_cfg.oob_bytes_per_page = 0;
    logical_headroom_cfg.mapping_entries_per_page = 1;
    logical_headroom_cfg.read_buffer_pages = 0;
    logical_headroom_cfg.gc_low_watermark_pages = 7;
    logical_headroom_cfg.ecc_decode_latency_ns = 1.0;
    logical_headroom_cfg.ecc_encode_latency_ns = 1.0;
    logical_headroom_cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    logical_headroom_cfg.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    HbfDevice logical_headroom_device(logical_headroom_cfg);
    logical_headroom_device.prepopulate_logical_pages({0, 1, 2, 3, 4, 5, 6, 7});
    const auto logical_headroom_overwrite = logical_headroom_device.issue(request(
        "gc-logical-headroom-overwrite", Tier::HBF, Op::Write,
        0.0, 0, 2 * logical_headroom_cfg.page_size_bytes,
        AddressSpace::Logical));
    const auto logical_headroom_drain = logical_headroom_device.drain_pending(
        "gc-logical-headroom-drain",
        logical_headroom_overwrite.finish_ns,
        {.mode = TraceMode::Full});
    (void)logical_headroom_drain;
    const auto logical_headroom_stats = logical_headroom_device.stats();
    std::cout << "  logical_headroom_gc_runs=" << logical_headroom_stats.gc_runs
              << " relocations=" << logical_headroom_stats.gc_relocations
              << " mapping_programs=" << logical_headroom_stats.mapping_page_programs
              << "\n";
    if (logical_headroom_stats.gc_runs != 0 ||
        logical_headroom_stats.gc_relocations != 0 ||
        logical_headroom_stats.mapping_page_programs != 2) {
        throw std::runtime_error(
            "hbf-gc: active Mapping capacity falsely consumed whole-block headroom");
    }
    std::cout << "  gc_role_headroom_no_false_trigger_ok=yes\n";

    // Raw-owned blocks are intentionally unavailable to the FTL. This stress
    // leaves one useful data victim plus mapping-checkpoint blocks. Unused
    // pages in the resulting active GC block must count as relocation
    // headroom so the drain converges without repeatedly moving user LPNs.
    HbfConfig convergence_cfg = logical_headroom_cfg;
    HbfDevice convergence_device(convergence_cfg);
    convergence_device.prepopulate_logical_pages({0, 1, 2, 3});
    std::vector<PhysicalCompletion> convergence_issued;
    convergence_issued.push_back(convergence_device.issue(request(
        "gc-convergence-overwrite", Tier::HBF, Op::Write,
        0.0, 0, 2 * convergence_cfg.page_size_bytes,
        AddressSpace::Logical)));
    // Checkpoint persistence owns low-numbered Mapping blocks. Reserve raw
    // capacity from the untouched tail rather than assuming block3 is free.
    for (const auto block : {5u, 6u, 7u}) {
        convergence_issued.push_back(convergence_device.issue(request(
            "gc-convergence-raw-block" + std::to_string(block),
            Tier::HBF,
            Op::Write,
            0.0,
            convergence_device.encode(HbfAddress{.block = block}),
            convergence_cfg.page_size_bytes,
            AddressSpace::Physical)));
    }
    double convergence_drain_arrival = 0.0;
    for (const auto& completion : convergence_issued) {
        convergence_drain_arrival = std::max(
            convergence_drain_arrival, completion.finish_ns);
    }
    const auto convergence_drain = convergence_device.drain_pending(
        "gc-convergence-drain",
        convergence_drain_arrival,
        {.mode = TraceMode::Full});
    const auto convergence_stats = convergence_device.stats();
    const auto victim_count = static_cast<std::size_t>(std::count_if(
        convergence_drain.spans.begin(),
        convergence_drain.spans.end(),
        [](const TraceSpan& span) {
            return span.name == "gc_victim_select" && span.detail == "block0";
        }));
    const auto lpn2_relocations = static_cast<std::size_t>(std::count_if(
        convergence_drain.spans.begin(),
        convergence_drain.spans.end(),
        [](const TraceSpan& span) {
            return span.name == "gc_resident_mapping_update" &&
                span.detail == "lpn2";
        }));
    const auto lpn3_relocations = static_cast<std::size_t>(std::count_if(
        convergence_drain.spans.begin(),
        convergence_drain.spans.end(),
        [](const TraceSpan& span) {
            return span.name == "gc_resident_mapping_update" &&
                span.detail == "lpn3";
        }));
    std::cout << "  convergence_gc_runs=" << convergence_stats.gc_runs
              << " relocations=" << convergence_stats.gc_relocations
              << " data_relocations="
              << convergence_stats.gc_data_relocations
              << " mapping_relocations="
              << convergence_stats.gc_mapping_relocations
              << " mapping_programs=" << convergence_stats.mapping_page_programs
              << " victim_block0=" << victim_count
              << " lpn2_moves=" << lpn2_relocations
              << " lpn3_moves=" << lpn3_relocations << "\n";
    if (convergence_stats.gc_runs != 1 ||
        convergence_stats.gc_relocations != 2 ||
        convergence_stats.gc_data_relocations != 2 ||
        convergence_stats.gc_mapping_relocations != 0 ||
        convergence_stats.mapping_page_programs != 4 ||
        victim_count != 1 || lpn2_relocations != 1 || lpn3_relocations != 1) {
        throw std::runtime_error(
            "hbf-gc: soft watermark failed to converge on active GC headroom");
    }
    std::cout << "  gc_active_relocation_headroom_converges_ok=yes\n";

    // Headroom projection must follow the exact plane that the role's own
    // round-robin cursor will choose. Summing active pages across planes made
    // an active page on plane0 hide the whole-block debit about to occur on
    // plane1; GC then appeared one request late on an unrelated append.
    HbfConfig multiplane_cfg;
    multiplane_cfg.channels_per_stack = 1;
    multiplane_cfg.dies_per_channel = 1;
    multiplane_cfg.planes_per_die = 2;
    multiplane_cfg.blocks_per_plane = 3;
    multiplane_cfg.pages_per_block = 2;
    multiplane_cfg.page_size_bytes = 512;
    multiplane_cfg.oob_bytes_per_page = 0;
    multiplane_cfg.read_buffer_pages = 0;
    multiplane_cfg.mapping_entries_per_page = 512;
    multiplane_cfg.ctrl_dram_bytes = 512;
    multiplane_cfg.gc_reserved_free_blocks_per_plane = 1;
    multiplane_cfg.gc_low_watermark_pages = 1;
    multiplane_cfg.gc_hard_watermark_pages = 1;
    multiplane_cfg.ecc_decode_latency_ns = 1.0;
    multiplane_cfg.ecc_encode_latency_ns = 1.0;
    multiplane_cfg.ecc_decode_raw_bandwidth_GBps_per_die = 1.0e9;
    multiplane_cfg.ecc_encode_raw_bandwidth_GBps_per_die = 1.0e9;
    HbfDevice multiplane_device(multiplane_cfg);
    double multiplane_arrival = 0.0;
    std::vector<PhysicalCompletion> multiplane_rows;
    for (const auto lpn : {0ULL, 1ULL, 0ULL, 2ULL, 3ULL}) {
        multiplane_rows.push_back(multiplane_device.issue(request(
            "gc-multiplane-lpn" + std::to_string(lpn),
            Tier::HBF,
            Op::Write,
            multiplane_arrival,
            lpn * multiplane_cfg.page_size_bytes,
            multiplane_cfg.page_size_bytes,
            AddressSpace::Logical)));
        multiplane_arrival = multiplane_rows.back().finish_ns;
    }
    const auto gc_before_open = multiplane_device.stats().gc_runs;
    auto opening_request = request(
        "gc-multiplane-opening-lpn4",
        Tier::HBF,
        Op::Write,
        multiplane_arrival,
        4 * multiplane_cfg.page_size_bytes,
        multiplane_cfg.page_size_bytes,
        AddressSpace::Logical);
    opening_request.trace.mode = TraceMode::Full;
    const auto opening_completion = multiplane_device.issue(opening_request);
    const auto gc_after_open = multiplane_device.stats().gc_runs;
    const auto append_completion = multiplane_device.issue(request(
        "gc-multiplane-append-lpn5",
        Tier::HBF,
        Op::Write,
        opening_completion.finish_ns,
        5 * multiplane_cfg.page_size_bytes,
        multiplane_cfg.page_size_bytes,
        AddressSpace::Logical));
    const auto multiplane_stats = multiplane_device.stats();
    const bool opened_plane1 =
        opening_completion.resource_path.find("plane1/block1/page0") !=
        std::string::npos;
    const bool appended_plane0 =
        append_completion.resource_path.find("plane0/block1/page1") !=
        std::string::npos;
    const auto opening_victims = static_cast<std::size_t>(std::count_if(
        opening_completion.spans.begin(),
        opening_completion.spans.end(),
        [](const TraceSpan& span) { return span.name == "gc_victim_select"; }));
    std::cout << "  multiplane_gc_before_open=" << gc_before_open
              << " after_open=" << gc_after_open
              << " after_append=" << multiplane_stats.gc_runs
              << " relocations=" << multiplane_stats.gc_relocations
              << " blocked=" << multiplane_stats.gc_user_blocked_runs
              << " opening_victims=" << opening_victims << "\n";
    if (gc_before_open != 0 || gc_after_open != 1 ||
        multiplane_stats.gc_runs != 1 ||
        multiplane_stats.gc_relocations != 1 ||
        multiplane_stats.gc_user_blocked_runs != 0 ||
        opening_victims != 1 || !opened_plane1 || !appended_plane0) {
        throw std::runtime_error(
            "hbf-gc: multi-plane block-opening debit was charged to the wrong request");
    }
    std::cout << "  gc_multiplane_opening_debit_ok=yes\n";
}

void probe_hbf_tsu_scheduler() {
    std::cout << "\n== HBF flash transaction scheduler probe ==\n";
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.blocks_per_plane = 8;
    cfg.pages_per_block = 4;
    cfg.page_size_bytes = 512;
    HbfDevice hbf(cfg);
    const auto p0 = hbf.encode(HbfAddress{.block = 0, .page = 0});

    std::vector<PhysicalCompletion> rows;
    rows.push_back(hbf.issue(request("physical-read-tsu", Tier::HBF, Op::Read, 0,
        p0, 512, AddressSpace::Physical)));
    rows.push_back(hbf.issue(request("physical-program-tsu", Tier::HBF, Op::Write, 0,
        p0, 512, AddressSpace::Physical)));
    rows.push_back(hbf.issue(request("physical-erase-tsu", Tier::HBF, Op::Erase, 0,
        p0, 0, AddressSpace::Physical)));
    print_rows(rows);
    print_hbf_stats(hbf);
    std::cout << "  expectation: read/program/erase all pass through flash_scheduler_issue_* on the die.\n";
}

void probe_hbf_write_coalescer() {
    std::cout << "\n== HBF write coalescer probe ==\n";
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.blocks_per_plane = 16;
    cfg.pages_per_block = 8;
    cfg.page_size_bytes = 512;
    cfg.mapping_entries_per_page = 8;
    cfg.write_coalescing_enabled = true;
    cfg.write_buffer_pages = 4;
    HbfDevice hbf(cfg);

    std::vector<PhysicalCompletion> rows;
    rows.push_back(hbf.issue(request("64B-write-a", Tier::HBF, Op::Write, 0, 0, 64)));
    rows.push_back(hbf.issue(request("64B-write-b-same-lpn", Tier::HBF, Op::Write, 0, 64, 64)));
    rows.push_back(hbf.issue(request("read-buffered-byte-range", Tier::HBF, Op::Read, 0, 0, 64)));
    rows.push_back(hbf.drain_pending("drain-write-buffer", 0, {.mode = next_request_trace_mode()}));
    print_rows(rows);
    print_hbf_stats(hbf);
    std::cout << "  expectation: two small writes stage in SRAM, read hits write buffer, drain programs one data page plus one mapping page.\n";

    const auto heatmap_config = [](const HbfConfig& device_config) {
        hbfsim::physical::AddressHeatmapConfig config;
        config.bin_count = 16;
        const auto capacity = static_cast<std::uint64_t>(device_config.stacks) *
            device_config.channels_per_stack * device_config.dies_per_channel *
            device_config.planes_per_die * device_config.blocks_per_plane *
            device_config.pages_per_block * device_config.page_size_bytes;
        for (auto& domain : config.domains) {
            domain.size_bytes = capacity;
        }
        return config;
    };
    const auto source_slot = [](hbfsim::physical::HeatmapTrafficSource source) {
        return static_cast<std::size_t>(source);
    };

    // A later request may evict an older buffered page, and the final drain
    // flushes whatever remains. Neither trigger owns the deferred media work:
    // the dirty page's persisted provenance does. Include a partial update of
    // a mapped page so both its merge read and replacement program are checked.
    auto deferred_cfg = cfg;
    deferred_cfg.write_buffer_pages = 1;
    deferred_cfg.write_buffer_flush_threshold_pages = 0;
    hbfsim::physical::AddressHeatmap deferred_heatmap(
        heatmap_config(deferred_cfg));
    HbfDevice deferred(deferred_cfg, &deferred_heatmap);
    deferred.prepopulate_logical_pages({0});
    auto direct_partial = request(
        "deferred-direct-partial",
        Tier::HBF,
        Op::Write,
        0,
        0,
        64);
    direct_partial.heatmap_source =
        hbfsim::physical::HeatmapTrafficSource::Direct;
    (void)deferred.issue(direct_partial);
    auto destage_next = request(
        "destage-evicts-direct",
        Tier::HBF,
        Op::Write,
        0,
        deferred_cfg.page_size_bytes,
        deferred_cfg.page_size_bytes);
    destage_next.heatmap_source =
        hbfsim::physical::HeatmapTrafficSource::Destage;
    (void)deferred.issue(destage_next);
    (void)deferred.drain_pending("provenance-drain", 0);

    const auto& deferred_physical = deferred_heatmap.domain(
        hbfsim::physical::AddressDomain::HbfPhysical);
    const auto& deferred_direct = deferred_physical.by_source.at(source_slot(
        hbfsim::physical::HeatmapTrafficSource::Direct));
    const auto& deferred_destage = deferred_physical.by_source.at(source_slot(
        hbfsim::physical::HeatmapTrafficSource::Destage));
    if (deferred_direct.read_bytes != deferred_cfg.page_size_bytes ||
        deferred_direct.write_bytes != deferred_cfg.page_size_bytes ||
        deferred_destage.read_bytes != 0 ||
        deferred_destage.write_bytes != deferred_cfg.page_size_bytes ||
        deferred_physical.total.read_bytes != deferred.stats().physical_read_bytes ||
        deferred_physical.total.write_bytes != deferred.stats().physical_write_bytes) {
        throw std::runtime_error(
            "hbf-coalesce: deferred flush was relabeled by its evictor or drain");
    }

    // Overwrites retain byte-level last-writer ranges. The physical page
    // program is conservatively attributed to the source owning the most
    // dirty bytes, independent of the request that forces eviction.
    hbfsim::physical::AddressHeatmap merged_heatmap(
        heatmap_config(deferred_cfg));
    HbfDevice merged(deferred_cfg, &merged_heatmap);
    auto direct_whole = request(
        "direct-whole-page",
        Tier::HBF,
        Op::Write,
        0,
        0,
        deferred_cfg.page_size_bytes);
    direct_whole.heatmap_source =
        hbfsim::physical::HeatmapTrafficSource::Direct;
    (void)merged.issue(direct_whole);
    auto destage_majority = request(
        "destage-majority-overwrite",
        Tier::HBF,
        Op::Write,
        0,
        192,
        320);
    destage_majority.heatmap_source =
        hbfsim::physical::HeatmapTrafficSource::Destage;
    (void)merged.issue(destage_majority);
    auto streaming_evictor = request(
        "streaming-install-evictor",
        Tier::HBF,
        Op::Write,
        0,
        deferred_cfg.page_size_bytes,
        deferred_cfg.page_size_bytes);
    streaming_evictor.heatmap_source =
        hbfsim::physical::HeatmapTrafficSource::StreamingInstall;
    (void)merged.issue(streaming_evictor);
    (void)merged.drain_pending("merged-provenance-drain", 0);

    const auto& merged_physical = merged_heatmap.domain(
        hbfsim::physical::AddressDomain::HbfPhysical);
    const auto& merged_direct = merged_physical.by_source.at(source_slot(
        hbfsim::physical::HeatmapTrafficSource::Direct));
    const auto& merged_destage = merged_physical.by_source.at(source_slot(
        hbfsim::physical::HeatmapTrafficSource::Destage));
    const auto& merged_streaming = merged_physical.by_source.at(source_slot(
        hbfsim::physical::HeatmapTrafficSource::StreamingInstall));
    if (merged_direct.write_bytes != 0 ||
        merged_destage.write_bytes != deferred_cfg.page_size_bytes ||
        merged_streaming.write_bytes != deferred_cfg.page_size_bytes ||
        merged_physical.total.write_bytes != merged.stats().physical_write_bytes) {
        throw std::runtime_error(
            "hbf-coalesce: merged dirty-range provenance is not last-writer/dominant-source stable");
    }

    hbfsim::physical::AddressHeatmap tied_heatmap(
        heatmap_config(deferred_cfg));
    HbfDevice tied(deferred_cfg, &tied_heatmap);
    (void)tied.issue(direct_whole);
    auto destage_half = request(
        "destage-half-overwrite",
        Tier::HBF,
        Op::Write,
        0,
        deferred_cfg.page_size_bytes / 2,
        deferred_cfg.page_size_bytes / 2);
    destage_half.heatmap_source =
        hbfsim::physical::HeatmapTrafficSource::Destage;
    (void)tied.issue(destage_half);
    (void)tied.issue(streaming_evictor);
    (void)tied.drain_pending("tied-provenance-drain", 0);
    const auto& tied_physical = tied_heatmap.domain(
        hbfsim::physical::AddressDomain::HbfPhysical);
    const auto& tied_direct = tied_physical.by_source.at(source_slot(
        hbfsim::physical::HeatmapTrafficSource::Direct));
    const auto& tied_destage = tied_physical.by_source.at(source_slot(
        hbfsim::physical::HeatmapTrafficSource::Destage));
    const auto& tied_streaming = tied_physical.by_source.at(source_slot(
        hbfsim::physical::HeatmapTrafficSource::StreamingInstall));
    if (tied_direct.write_bytes != deferred_cfg.page_size_bytes ||
        tied_destage.write_bytes != 0 ||
        tied_streaming.write_bytes != deferred_cfg.page_size_bytes ||
        tied_physical.total.write_bytes != tied.stats().physical_write_bytes) {
        throw std::runtime_error(
            "hbf-coalesce: equal-byte source tie did not use stable enum order");
    }
    std::cout << "  deferred_heatmap_provenance_ok=yes\n";
    std::cout << "  merged_heatmap_provenance_ok=yes\n";
    std::cout << "  tied_heatmap_provenance_ok=yes\n";
}

void probe_hbf_write_backpressure() {
    std::cout << "\n== HBF write-buffer slot backpressure probe ==\n";
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.blocks_per_plane = 64;
    cfg.pages_per_block = 64;
    cfg.page_size_bytes = 512;
    // One mapping page covers every LPN in the probe: no mapping-read noise.
    cfg.mapping_entries_per_page = 4096;
    cfg.write_coalescing_enabled = true;
    cfg.write_buffer_pages = 4;
    cfg.write_buffer_flush_threshold_pages = 2;
    HbfDevice hbf(cfg);

    constexpr std::size_t kBurst = 4;  // = write_buffer_pages (SRAM slot budget)
    constexpr std::size_t kTotal = 16;
    std::uint64_t burst_waits = 0;
    double last_finish_ns = 0.0;
    for (std::size_t i = 0; i < kTotal; ++i) {
        const auto completion = hbf.issue(request(
            "write-lpn" + std::to_string(i),
            Tier::HBF,
            Op::Write,
            0,
            static_cast<std::uint64_t>(i) * cfg.page_size_bytes,
            cfg.page_size_bytes));
        last_finish_ns = std::max(last_finish_ns, completion.finish_ns);
        if (i + 1 == kBurst) {
            burst_waits = hbf.stats().write_buffer_slot_wait_ops;
        }
    }
    print_hbf_stats(hbf);
    const auto& s = hbf.stats();
    const double program_ns = cfg.t_program_page_ns + cfg.t_program_verify_ns;
    // Single plane => programs strictly serialize at one per program_ns; the
    // k-th admit past the SRAM budget waits for the k-th program completion,
    // so the last admit sits at (kTotal - kBurst) serialized programs (plus
    // ns-scale issue overheads, minus none: the floor is exact).
    const double admit_floor_ns = static_cast<double>(kTotal - kBurst) * program_ns;
    std::cout << "  burst_slot_waits=" << burst_waits << "\n";
    std::cout << "  sustained_slot_waits=" << s.write_buffer_slot_wait_ops << "\n";
    std::cout << "  last_admit_finish_ns=" << fixed(last_finish_ns)
              << " floor_ns=" << fixed(admit_floor_ns) << "\n";
    if (burst_waits != 0) {
        throw std::runtime_error(
            "hbf-write-backpressure: writes inside the SRAM budget must not wait for slots");
    }
    if (s.write_buffer_slot_wait_ops != kTotal - kBurst) {
        throw std::runtime_error(
            "hbf-write-backpressure: every admit past the budget must wait for a program");
    }
    if (last_finish_ns < admit_floor_ns || last_finish_ns > admit_floor_ns + program_ns) {
        throw std::runtime_error(
            "hbf-write-backpressure: last admit must sit within one program of the floor");
    }
    std::cout << "  admit_floor_ok=yes\n";
    std::cout << "  expectation: 4-slot SRAM absorbs the first four writes at ns scale;"
              << " every later admit waits for the oldest in-flight program"
              << " (single plane, one program per 80 us).\n";

    // A buffer containing only live entries used to insert the new LPN first
    // and evict afterward, transiently exceeding its SRAM capacity. Force that
    // exact state with a one-slot, no-threshold buffer and require the victim's
    // program release to precede the second stage.
    auto live_full_cfg = cfg;
    live_full_cfg.write_buffer_pages = 1;
    live_full_cfg.write_buffer_flush_threshold_pages = 0;
    HbfDevice live_full(live_full_cfg);
    (void)live_full.issue(request(
        "fill-only-live-slot", Tier::HBF, Op::Write, 0, 0, live_full_cfg.page_size_bytes));
    auto blocked_request = request(
        "admit-after-live-eviction",
        Tier::HBF,
        Op::Write,
        0,
        live_full_cfg.page_size_bytes,
        live_full_cfg.page_size_bytes);
    blocked_request.trace.mode = TraceMode::Full;
    const auto blocked = live_full.issue(blocked_request);
    const auto verify = std::find_if(
        blocked.spans.begin(), blocked.spans.end(), [](const auto& span) {
            return span.name.ends_with("/program_verify");
        });
    const auto stage = std::find_if(
        blocked.spans.begin(), blocked.spans.end(), [](const auto& span) {
            return span.name == "write_buffer_stage";
        });
    if (live_full.stats().write_buffer_slot_wait_ops != 1 ||
        live_full.stats().write_buffer_slot_wait_ns <= 0.0) {
        throw std::runtime_error(
            "hbf-write-backpressure: a live-full buffer must block the new LPN admission");
    }
    if (verify == blocked.spans.end() || stage == blocked.spans.end() ||
        stage->start_ns < verify->end_ns) {
        throw std::runtime_error(
            "hbf-write-backpressure: new data became visible before an SRAM slot release");
    }
    std::cout << "  live_full_admission_ok=yes\n";

    // Watermarks are per-stack quantities. Accept the exact usable capacity,
    // reject either watermark above it, including UINT64_MAX, before GC can
    // evaluate required_pages + watermark.
    HbfConfig watermark_cfg;
    watermark_cfg.channels_per_stack = 1;
    watermark_cfg.dies_per_channel = 1;
    watermark_cfg.planes_per_die = 1;
    watermark_cfg.blocks_per_plane = 4;
    watermark_cfg.pages_per_block = 8;
    watermark_cfg.page_size_bytes = 512;
    constexpr std::uint64_t kPerStackPages = 32;
    watermark_cfg.gc_low_watermark_pages = kPerStackPages;
    HbfDevice exact_watermark(watermark_cfg);
    (void)exact_watermark;
    const auto rejects_watermark = [](HbfConfig candidate) {
        try {
            HbfDevice invalid(candidate);
            (void)invalid;
            return false;
        } catch (const std::runtime_error&) {
            return true;
        }
    };
    auto excessive_low = watermark_cfg;
    excessive_low.gc_low_watermark_pages = kPerStackPages + 1;
    auto excessive_hard = watermark_cfg;
    excessive_hard.gc_low_watermark_pages = 0;
    excessive_hard.gc_hard_watermark_pages = std::numeric_limits<std::uint64_t>::max();
    if (!rejects_watermark(excessive_low) || !rejects_watermark(excessive_hard)) {
        throw std::runtime_error(
            "hbf-write-backpressure: out-of-capacity GC watermarks must fail at construction");
    }
    std::cout << "  watermark_bounds_ok=yes\n";

    // The last byte in uint64_t is a legal one-byte logical range. Exercise it
    // through write-buffer segmentation, then reject the first genuinely
    // overflowing inclusive range. This catches page_end wraparound and the
    // resulting underflowed dirty range.
    auto top_cfg = cfg;
    top_cfg.write_buffer_pages = 2;
    top_cfg.write_buffer_flush_threshold_pages = 0;
    HbfDevice top(top_cfg);
    constexpr auto kTop = std::numeric_limits<std::uint64_t>::max();
    (void)top.issue(request("top-byte-write", Tier::HBF, Op::Write, 0, kTop, 1));
    (void)top.issue(request("top-two-byte-write", Tier::HBF, Op::Write, 0, kTop - 1, 2));
    const auto top_read = top.issue(request(
        "top-two-byte-read", Tier::HBF, Op::Read, 0, kTop - 1, 2));
    bool rejected_overflow = false;
    try {
        (void)top.issue(request(
            "overflowing-top-range", Tier::HBF, Op::Write, 0, kTop - 1, 3));
    } catch (const std::runtime_error& ex) {
        rejected_overflow = std::string(ex.what()).find("overflows uint64_t") !=
            std::string::npos;
    }
    if (top_read.note != "write-buffer-read" ||
        top.stats().write_buffer_read_bytes < 2 || !rejected_overflow) {
        throw std::runtime_error(
            "hbf-write-backpressure: top-of-address-space segmentation is not fail-safe");
    }
    std::cout << "  logical_top_range_ok=yes\n";
}

void probe_hbf_mapping_writeback() {
    std::cout << "\n== HBF mapping writeback probe ==\n";
    HbfConfig cfg;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.blocks_per_plane = 16;
    cfg.pages_per_block = 8;
    cfg.page_size_bytes = 512;
    cfg.mapping_entries_per_page = 16;
    HbfDevice hbf(cfg);

    std::vector<PhysicalCompletion> rows;
    rows.push_back(hbf.issue(request("write-lpn0", Tier::HBF, Op::Write, 0, 0, 512)));
    rows.push_back(hbf.issue(request("write-lpn1", Tier::HBF, Op::Write, 0, 512, 512)));
    rows.push_back(hbf.issue(request("write-lpn2", Tier::HBF, Op::Write, 0, 1024, 512)));
    rows.push_back(hbf.drain_pending("drain-dirty-mapping", 0, {.mode = next_request_trace_mode()}));
    print_rows(rows);
    print_hbf_stats(hbf);
    std::cout << "  expectation: three data programs share one dirty mapping page program at drain.\n";
}

void probe_composition_kind_blind() {
    std::cout << "\n== composition kind-blind routing probe ==\n";
    HbmConfig hbm_cfg;
    hbm_cfg.stacks = 1;
    hbm_cfg.channels_per_stack = 2;
    hbm_cfg.pseudo_channels_per_channel = 1;
    HbfConfig hbf_cfg;
    hbf_cfg.stacks = 1;
    hbf_cfg.channels_per_stack = 1;
    hbf_cfg.dies_per_channel = 1;
    hbf_cfg.planes_per_die = 2;
    hbf_cfg.blocks_per_plane = 64;
    hbf_cfg.pages_per_block = 64;
    hbf_cfg.page_size_bytes = 512;

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
    const auto signature = [](const CompositionRunResult& r) {
        return std::tuple(
            r.hbm_user_accesses, r.hbf_user_accesses, r.hbf_static_read_bytes,
            r.reads, r.writes, r.logical_bytes);
    };

    struct Preset {
        const char* name;
        DirectPolicy policy;
    };
    const std::vector<Preset> blind_presets{
        {"all-HBM", all_hbm_policy()},
        {"all-HBF", all_hbf_policy()},
        {"FLAT", flat_policy(kBoundary)},
    };
    for (const auto& preset : blind_presets) {
        const auto plain_run =
            run_direct_composition(preset.policy, hbm_cfg, hbf_cfg, plain, knobs);
        const auto labeled_run =
            run_direct_composition(preset.policy, hbm_cfg, hbf_cfg, labeled, knobs);
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

    const auto direct_plain = run_direct_composition(
        read_only_direct_policy(kBoundary), hbm_cfg, hbf_cfg, plain, knobs);
    const auto direct_labeled = run_direct_composition(
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
              << "all-HBM/all-HBF/FLAT route identically with or without "
              << "kinds; read-only-direct uses kinds only to keep non-streaming reads "
              << "on HBM.\n";
}

void probe_composition_static_mapping() {
    std::cout << "\n== composition static-mapping bijection probe ==\n";
    HbfConfig cfg;
    // The deleted hash/modulo mapper collided specifically for source pages
    // 4 and 5 in this 8x8 single-plane geometry. Exhaustively checking the
    // full domain makes that regression deterministic rather than probable.
    cfg.stacks = 1;
    cfg.channels_per_stack = 1;
    cfg.dies_per_channel = 1;
    cfg.planes_per_die = 1;
    cfg.blocks_per_plane = 8;
    cfg.pages_per_block = 8;
    cfg.page_size_bytes = 512;
    cfg.media_lanes_per_plane = 8;
    cfg.subarrays_per_plane = 8;
    HbfDevice hbf(cfg);

    const auto capacity = hbf_page_capacity(hbf);
    std::set<std::uint64_t> physical_pages;
    for (std::uint64_t source_page = 0; source_page < capacity; ++source_page) {
        const auto physical_addr = map_static_hbf_page_addr(hbf, source_page);
        if (!physical_pages.insert(physical_addr / cfg.page_size_bytes).second) {
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
        (void)unmap_static_hbf_page_addr(hbf, capacity * cfg.page_size_bytes);
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
    hash_cfg.stacks = 8;
    hash_cfg.mapping_entries_per_page = 512;
    HbfDevice hash_device(hash_cfg);
    std::array<std::uint64_t, 8> stack_counts{};
    for (std::uint64_t index = 0; index < 4096; ++index) {
        const auto mapping_vpn = index * hash_cfg.stacks;
        const auto lpn =
            mapping_vpn * hash_cfg.mapping_entries_per_page;
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
            group * hash_cfg.mapping_entries_per_page * hash_cfg.stacks;
        for (std::uint64_t lane = 0; lane < hash_cfg.stacks; ++lane) {
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
    constexpr std::uint64_t kPageSize = 512;
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
    hbm_cfg.stacks = 1;
    hbm_cfg.channels_per_stack = 1;
    hbm_cfg.pseudo_channels_per_channel = 1;
    HbfConfig hbf_cfg;
    hbf_cfg.stacks = 1;
    hbf_cfg.channels_per_stack = 1;
    hbf_cfg.dies_per_channel = 1;
    hbf_cfg.planes_per_die = 2;
    hbf_cfg.blocks_per_plane = 16;
    hbf_cfg.pages_per_block = 16;
    hbf_cfg.page_size_bytes = kPageSize;
    hbf_cfg.mapping_entries_per_page = 16;

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
    const auto direct_write_first = run_direct_composition(
        all_hbf_policy(), hbm_cfg, hbf_cfg, make_trace(false), knobs);
    const auto direct_read_first = run_direct_composition(
        all_hbf_policy(), hbm_cfg, hbf_cfg, make_trace(true), knobs);
    std::cout << "  direct invalidations write-first/read-first="
              << direct_write_first.hbf_stats.invalidations << "/"
              << direct_read_first.hbf_stats.invalidations << "\n";
    if (direct_write_first.hbf_stats.invalidations != 0 ||
        direct_read_first.hbf_stats.invalidations == 0) {
        throw std::runtime_error(
            "composition-initial-image: all-HBF inferred a page from a future read/write");
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
    const auto fixed_population = run_direct_composition(
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
        (void)run_direct_composition(
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
    hbm_cfg.stacks = 1;
    hbm_cfg.channels_per_stack = 2;
    hbm_cfg.pseudo_channels_per_channel = 1;
    HbfConfig hbf_cfg;
    hbf_cfg.stacks = 1;
    hbf_cfg.channels_per_stack = 1;
    hbf_cfg.dies_per_channel = 1;
    hbf_cfg.planes_per_die = 4;
    hbf_cfg.blocks_per_plane = 64;
    hbf_cfg.pages_per_block = 64;
    hbf_cfg.page_size_bytes = 4096;
    hbf_cfg.mapping_entries_per_page = 4096;
    hbf_cfg.write_coalescing_enabled = true;
    hbf_cfg.write_buffer_pages = 4;
    hbf_cfg.write_buffer_flush_threshold_pages = 2;

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
    const auto max_latency = [](const CompositionRunResult& r) {
        double worst = 0.0;
        for (const auto v : r.offered_latencies_ns) {
            worst = std::max(worst, v);
        }
        return worst;
    };
    const double hbm_fast_limit_ns = hbf_cfg.t_program_page_ns / 2.0;

    // Region holds the burst: writes complete at HBM speed, destages park
    // and stream out after the last user op — every page still programs
    // (work conserved into the tail).
    DirectRunKnobs burst_knobs;
    burst_knobs.hbm_write_buffer_bytes = 1ull << 20;  // 1 MiB >= 16 pages
    const auto burst = run_direct_composition(
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
    const auto partial = run_direct_composition(
        flat_policy(0), hbm_cfg, hbf_cfg, same_page_partial,
        one_page_partial_knobs);
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
    const auto cross_page = run_direct_composition(
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
    const auto astra_partial = run_direct_composition(
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
    const auto one_slot_cross = run_direct_composition(
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
    const auto pressed = run_direct_composition(
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

    // Streamed (KAIST layer-wise) smoke: chains leave with production; the
    // op stays HBM-fast and the same pages program.
    DirectRunKnobs streamed_knobs = burst_knobs;
    streamed_knobs.destage_policy = DirectRunKnobs::DestagePolicy::Streamed;
    const auto streamed = run_direct_composition(
        flat_policy(0), hbm_cfg, hbf_cfg, writes, streamed_knobs);
    std::cout << "  streamed: max write latency = " << fixed(max_latency(streamed))
              << " ns, destaged = " << streamed.hbm_write_buffer_destaged_bytes
              << " B, hbf page_programs = " << streamed.hbf_stats.page_programs << "\n";
    if (max_latency(streamed) > hbm_fast_limit_ns ||
        streamed.hbm_write_buffer_destaged_bytes != 16ull * 4096 ||
        streamed.hbf_stats.page_programs != burst.hbf_stats.page_programs) {
        throw std::runtime_error(
            "composition-coop-write: streamed destage must stay HBM-fast and conserve programs");
    }
    std::cout << "  coop_streamed_ok=yes\n";

    // A streamed ready event exactly tied with foreground arrival yields to
    // that foreground. Determine the write's HBM completion, then place an
    // ordinary HBM read at precisely that timestamp; streamed and deferred
    // policies must expose the same read latency.
    const MemoryRequest tie_write{
        .id = "tie-write", .op = Op::Write, .addr = 8192,
        .bytes = 4096, .arrival_ns = 0.0, .index = 0,
        .kind = SemanticKind::Unknown, .label = {},
    };
    auto one_page_deferred = burst_knobs;
    one_page_deferred.hbm_write_buffer_bytes = 4096;
    auto one_page_streamed = one_page_deferred;
    one_page_streamed.destage_policy = DirectRunKnobs::DestagePolicy::Streamed;
    const auto tie_seed = run_direct_composition(
        flat_policy(4096), hbm_cfg, hbf_cfg, {tie_write}, one_page_deferred);
    const std::vector<MemoryRequest> tie_trace{
        tie_write,
        MemoryRequest{.id = "tie-foreground-read", .op = Op::Read, .addr = 0,
            .bytes = 64, .arrival_ns = tie_seed.user_finish_ns, .index = 1,
            .kind = SemanticKind::Unknown, .label = {}},
    };
    const auto tie_deferred = run_direct_composition(
        flat_policy(4096), hbm_cfg, hbf_cfg, tie_trace, one_page_deferred);
    const auto tie_streamed = run_direct_composition(
        flat_policy(4096), hbm_cfg, hbf_cfg, tie_trace, one_page_streamed);
    if (std::abs(tie_deferred.offered_latencies_ns.at(1) -
                 tie_streamed.offered_latencies_ns.at(1)) > 1e-9) {
        throw std::runtime_error(
            "composition-coop-write: same-time streamed destage preempted foreground HBM");
    }
    std::cout << "  coop_streamed_foreground_tie_priority_ok=yes\n";

    // Cross a logical stack stripe and ensure each page is charged to its
    // owner link instead of billing all bytes to the first page's stack.
    auto striped_hbf_cfg = hbf_cfg;
    striped_hbf_cfg.stacks = 2;
    striped_hbf_cfg.mapping_entries_per_page = 1;
    const std::vector<MemoryRequest> striped_write{
        MemoryRequest{.id = "striped-write", .op = Op::Write, .addr = 0,
            .bytes = 8192, .arrival_ns = 0.0, .index = 0,
            .kind = SemanticKind::Unknown, .label = {}},
    };
    const auto striped_direct = run_direct_composition(
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
    const auto oversized_streamed = run_direct_composition(
        flat_policy(0),
        hbm_cfg,
        striped_hbf_cfg,
        striped_write,
        undersized_region);
    if (oversized_streamed.hbm_write_buffer_peak_bytes != 4096 ||
        oversized_streamed.hbm_write_buffer_full_waits == 0 ||
        oversized_streamed.hbm_write_buffer_destaged_bytes != 8192 ||
        oversized_streamed.base_die_link_stats.write_bytes != 8192 ||
        oversized_streamed.hbf_stats.logical_write_bytes != 8192 ||
        oversized_streamed.hbf_stats.data_programs != 2) {
        throw std::runtime_error(
            "composition-coop-write: a request larger than the region must "
            "stream causally through finite page slots");
    }
    std::cout << "  coop_oversized_write_streamed_ok=yes\n";

    auto small_hbm_cfg = hbm_cfg;
    small_hbm_cfg.capacity_bytes = 1ull << 20;
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
    const auto allocated = run_direct_composition(
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
    const auto reuse_after_release = run_direct_composition(
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
        (void)run_direct_composition(
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
        (void)run_direct_composition(
            flat_policy(0), hbm_cfg, hbf_cfg, {writes.front()}, invalid_region);
    } catch (const std::runtime_error&) {
        subpage_region_rejected = true;
    }
    bool unaligned_base_rejected = false;
    try {
        auto unaligned_hbm = hbm_cfg;
        unaligned_hbm.capacity_bytes += 1;
        auto one_page_region = burst_knobs;
        one_page_region.hbm_write_buffer_bytes = 4096;
        (void)run_direct_composition(
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
              << " destages yield to user traffic and stream out afterwards"
              << " over the configured HBM->HBF path; capacity overflow honestly"
              << " throttles to the destage/program rate.\n";
}

void probe_composition_reuse_routing() {
    std::cout << "\n== read-after-write routing coherence probe ==\n";
    HbmConfig hbm_cfg;
    hbm_cfg.stacks = 1;
    hbm_cfg.channels_per_stack = 2;
    hbm_cfg.pseudo_channels_per_channel = 1;
    HbfConfig hbf_cfg;
    hbf_cfg.stacks = 1;
    hbf_cfg.channels_per_stack = 1;
    hbf_cfg.dies_per_channel = 1;
    hbf_cfg.planes_per_die = 2;
    hbf_cfg.blocks_per_plane = 64;
    hbf_cfg.pages_per_block = 64;
    hbf_cfg.page_size_bytes = 4096;
    hbf_cfg.mapping_entries_per_page = 4096;

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
    const auto flat_read = run_direct_composition(
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
    const auto flat_write = run_direct_composition(
        flat_policy(8192), hbm_cfg, hbf_cfg, flat_cross_write, knobs);
    if (flat_write.ops != 1 || flat_write.hbm_user_accesses != 1 ||
        flat_write.hbf_user_accesses != 1 ||
        flat_write.hbm_stats.write_bytes != 4096 ||
        flat_write.hbf_stats.logical_write_bytes != 4096 ||
        !flat_write.warnings.empty()) {
        throw std::runtime_error(
            "composition-reuse-routing: flat cross-boundary write was not split exactly");
    }

    bool unaligned_boundary_rejected = false;
    try {
        (void)run_direct_composition(
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
    const auto ro = run_direct_composition(
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
    const auto windowed = run_direct_composition(
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
            .bytes = 16 * hbf_cfg.page_size_bytes,
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
            .bytes = hbf_cfg.page_size_bytes,
            .arrival_ns = 0.0,
            .index = 1,
            .kind = SemanticKind::SharedContext,
            .label = {},
            .phase = 0,
        },
    };
    const auto fair = run_direct_composition(
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
            .bytes = 2 * hbf_cfg.page_size_bytes,
            .arrival_ns = 0.0,
            .index = 0,
            .kind = SemanticKind::GeneratedContext,
            .label = {},
            .phase = 0,
        },
        MemoryRequest{
            .id = "read-second-written-page",
            .op = Op::Read,
            .addr = 8192 + hbf_cfg.page_size_bytes,
            .bytes = hbf_cfg.page_size_bytes,
            .arrival_ns = 0.0,
            .index = 1,
            .kind = SemanticKind::SharedContext,
            .label = {},
            .phase = 0,
        },
    };
    const auto ordered_overlap = run_direct_composition(
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
            .addr = page * hbf_cfg.page_size_bytes,
            .bytes = hbf_cfg.page_size_bytes,
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
            .bytes = 64 * hbf_cfg.page_size_bytes,
            .arrival_ns = 0.0,
            .index = 0,
            .kind = SemanticKind::Unknown,
            .label = {},
        },
    };
    DirectRunKnobs transaction_knobs;
    transaction_knobs.max_outstanding_requests = 8;
    const auto direct_pages = run_direct_composition(
        all_hbf_policy(),
        hbm_cfg,
        hbf_cfg,
        page_records,
        transaction_knobs);
    const auto direct_large = run_direct_composition(
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
    streaming_cfg.layer_buffer_bytes = 64 * hbf_cfg.page_size_bytes;
    streaming_cfg.max_outstanding_requests = 8;
    LayerStreamingComposition streaming_pages_system(streaming_cfg);
    LayerStreamingComposition streaming_large_system(streaming_cfg);
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
    const auto parked = run_direct_composition(
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

void probe_hbm_frfcfs() {
    std::cout << "\n== HBM FR-FCFS scheduler probe ==\n";

    // A synchronous, diagnostics-free span can execute one representative for
    // each exact pseudo-channel state class instead of allocating every burst
    // in every equivalent controller. Keep ordinary enqueue/pump as an
    // independent oracle: completion frontiers and integer counters must be
    // exact, while additive floating-point work may differ only by summation
    // roundoff.
    HbmConfig isolated_cfg;
    isolated_cfg.stacks = 6;
    isolated_cfg.channels_per_stack = 32;
    isolated_cfg.pseudo_channels_per_channel = 2;
    HbmDevice isolated_fast(isolated_cfg);
    HbmDevice isolated_queued(isolated_cfg);
    const auto diagnostics_free_request = [](
        std::string id,
        Op op,
        double arrival_ns,
        std::uint64_t addr) {
        auto value = request(
            std::move(id), Tier::HBM, op, arrival_ns, addr, 4096);
        value.trace.retain_completion_diagnostics = false;
        return value;
    };
    const auto require_same_completion = [](
        const PhysicalCompletion& fast,
        const PhysicalCompletion& queued,
        std::string_view case_name) {
        if (fast.id != queued.id ||
            fast.tier != queued.tier ||
            fast.op != queued.op ||
            fast.arrival_ns != queued.arrival_ns ||
            fast.start_ns != queued.start_ns ||
            fast.finish_ns != queued.finish_ns ||
            fast.logical_bytes != queued.logical_bytes ||
            fast.physical_bytes != queued.physical_bytes ||
            fast.resource_path != queued.resource_path ||
            fast.note != queued.note ||
            !(fast.breakdown == queued.breakdown) ||
            !fast.spans.empty() ||
            !queued.spans.empty()) {
            throw std::runtime_error(
                "hbm-frfcfs: " + std::string(case_name) +
                " differs from enqueue/pump oracle");
        }
    };
    const std::array<std::pair<Op, std::uint64_t>, 6> isolated_stream{{
        {Op::Write, 0},
        {Op::Read, 4096},
        {Op::Write, 8192},
        {Op::Read, 0},
        {Op::Read, 12288},
        {Op::Write, 4096},
    }};
    for (std::size_t index = 0; index < isolated_stream.size(); ++index) {
        const auto [op, addr] = isolated_stream[index];
        const auto arrival_ns = static_cast<double>(index) * 25.0;
        const auto fast_request = diagnostics_free_request(
            "isolated-" + std::to_string(index), op, arrival_ns, addr);
        const auto queued_request = fast_request;
        const auto fast = isolated_fast.issue(fast_request);
        const auto queued = isolated_queued.pump(
            isolated_queued.enqueue(queued_request));
        require_same_completion(
            fast, queued, "isolated synchronous path");
    }

    // The optimized issue() path must consume exactly the same parent tickets
    // as enqueue()+pump(). Otherwise a later public enqueue would expose the
    // implementation choice and could collide with retained controller state.
    const auto continuation = diagnostics_free_request(
        "ticket-continuation", Op::Read, 175.0, 16384);
    const auto fast_continuation_ticket = isolated_fast.enqueue(continuation);
    const auto queued_continuation_ticket =
        isolated_queued.enqueue(continuation);
    if (fast_continuation_ticket != queued_continuation_ticket) {
        throw std::runtime_error(
            "hbm-frfcfs: isolated synchronous path changed ticket sequence");
    }
    require_same_completion(
        isolated_fast.pump(fast_continuation_ticket),
        isolated_queued.pump(queued_continuation_ticket),
        "post-optimization ticket continuation");

    // Exercise both partial edge groups and a complete middle stripe. The
    // state-class path must reproduce the real bounded FR-FCFS queues even
    // when a pseudo-channel is revisited after the swizzle changes.
    const auto stripe_group_bytes =
        isolated_cfg.burst_bytes() *
        isolated_cfg.stacks *
        isolated_cfg.channels_per_stack *
        isolated_cfg.pseudo_channels_per_channel;
    auto cross_group_request = diagnostics_free_request(
        "cross-stripe-group",
        Op::Write,
        200.0,
        stripe_group_bytes - isolated_cfg.burst_bytes());
    cross_group_request.bytes = stripe_group_bytes;
    const auto fast_cross_group =
        isolated_fast.issue(cross_group_request);
    const auto queued_cross_group = isolated_queued.pump(
        isolated_queued.enqueue(cross_group_request));
    require_same_completion(
        fast_cross_group,
        queued_cross_group,
        "cross-stripe-group state classes");
    if (fast_cross_group.finish_ns == fast_cross_group.arrival_ns) {
        throw std::runtime_error(
            "hbm-frfcfs: cross-stripe-group oracle did no physical work");
    }

    // Cross queue-depth boundaries with a long span, then change direction
    // and continue from a non-integral arrival time. This makes the oracle
    // sensitive to admission order, FR-FCFS bypass, bus turnarounds, class
    // materialization, and floating-point accumulation.
    auto long_span = diagnostics_free_request(
        "long-span", Op::Read, 211.125,
        stripe_group_bytes + 3 * isolated_cfg.burst_bytes());
    long_span.bytes = 1024 * 1024;
    require_same_completion(
        isolated_fast.issue(long_span),
        isolated_queued.pump(isolated_queued.enqueue(long_span)),
        "multi-group queue-depth span");

    auto mixed_span = diagnostics_free_request(
        "mixed-span", Op::Write, 223.375,
        2 * stripe_group_bytes - isolated_cfg.burst_bytes());
    mixed_span.bytes = 17 * stripe_group_bytes +
        5 * isolated_cfg.burst_bytes();
    require_same_completion(
        isolated_fast.issue(mixed_span),
        isolated_queued.pump(isolated_queued.enqueue(mixed_span)),
        "mixed-direction partial-edge span");

    // The exact aggregate must remain identical after both optimized spans
    // and a public queued continuation have changed controller state.
    const auto final_request = diagnostics_free_request(
        "post-fallback-state", Op::Read, 225.0, 0);
    require_same_completion(
        isolated_fast.issue(final_request),
        isolated_queued.pump(isolated_queued.enqueue(final_request)),
        "post-fallback controller state");

    const auto& fast_stats = isolated_fast.stats();
    const auto& queued_stats = isolated_queued.stats();
    const auto near = [](double lhs, double rhs) {
        const auto scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
        return std::abs(lhs - rhs) <= 1e-12 * scale;
    };
    const auto breakdown_near = [&near](const Breakdown& lhs,
                                        const Breakdown& rhs) {
        const std::array<double Breakdown::*, 23> fields{{
            &Breakdown::ingress_queue_wait_ns,
            &Breakdown::scheduler_queue_wait_ns,
            &Breakdown::address_mapping_ns,
            &Breakdown::translation_ns,
            &Breakdown::mapping_dram_ns,
            &Breakdown::refresh_stall_ns,
            &Breakdown::precharge_ns,
            &Breakdown::activation_ns,
            &Breakdown::command_ns,
            &Breakdown::array_read_ns,
            &Breakdown::array_program_ns,
            &Breakdown::array_erase_ns,
            &Breakdown::program_verify_ns,
            &Breakdown::media_lane_transfer_ns,
            &Breakdown::page_buffer_ns,
            &Breakdown::sram_staging_ns,
            &Breakdown::channel_transfer_ns,
            &Breakdown::tsv_transfer_ns,
            &Breakdown::hb_io_transfer_ns,
            &Breakdown::transport_latency_ns,
            &Breakdown::ecc_queue_wait_ns,
            &Breakdown::ecc_latency_ns,
            &Breakdown::maintenance_ns,
        }};
        return std::all_of(
            fields.begin(), fields.end(),
            [&](const auto field) { return near(lhs.*field, rhs.*field); });
    };
    if (fast_stats.read_bytes != queued_stats.read_bytes ||
        fast_stats.write_bytes != queued_stats.write_bytes ||
        fast_stats.row_hits != queued_stats.row_hits ||
        fast_stats.row_misses != queued_stats.row_misses ||
        fast_stats.row_conflicts != queued_stats.row_conflicts ||
        fast_stats.activations != queued_stats.activations ||
        fast_stats.precharges != queued_stats.precharges ||
        fast_stats.refresh_count != queued_stats.refresh_count ||
        !near(fast_stats.bus_busy_ns, queued_stats.bus_busy_ns) ||
        fast_stats.first_arrival_ns != queued_stats.first_arrival_ns ||
        fast_stats.finish_ns != queued_stats.finish_ns ||
        fast_stats.pseudo_channels != queued_stats.pseudo_channels ||
        fast_stats.active_pseudo_channels !=
            queued_stats.active_pseudo_channels ||
        fast_stats.max_pseudo_channel_accesses !=
            queued_stats.max_pseudo_channel_accesses ||
        fast_stats.max_queue_occupancy !=
            queued_stats.max_queue_occupancy ||
        !near(fast_stats.max_pseudo_channel_busy_ns,
              queued_stats.max_pseudo_channel_busy_ns) ||
        !near(fast_stats.avg_active_pseudo_channel_busy_ns,
              queued_stats.avg_active_pseudo_channel_busy_ns) ||
        !breakdown_near(fast_stats.stage_work, queued_stats.stage_work)) {
        throw std::runtime_error(
            "hbm-frfcfs: isolated synchronous cumulative state differs "
            "from enqueue/pump oracle");
    }
    std::cout << "  isolated_synchronous_event_equivalence_oracle_ok=yes\n";
    std::cout << "  isolated_synchronous_ticket_sequence_ok=yes\n";
    std::cout << "  multi_group_state_class_oracle_ok=yes\n";

    HbmConfig cfg;
    cfg.stacks = 1;
    cfg.channels_per_stack = 1;
    cfg.pseudo_channels_per_channel = 1;
    cfg.bank_groups_per_pseudo_channel = 4;
    cfg.banks_per_group = 4;
    const auto row_addr = [](std::uint64_t row, std::uint64_t offset) {
        return HbmAddress{.row = row, .offset = offset};
    };
    // The prime at t=990 opens row0 and its ACT holds the bank's PRE gate
    // until 990 + tRAS = 1022: the conflict arriving at 1000 CANNOT issue
    // before 1022, so a row0 hit arriving at 1001 is genuinely contending
    // (eligible) and FR-FCFS serves it first. Bus work is conserved either
    // way; only the order swaps.
    const double prime_ns = 990.0;

    HbmDevice queued(cfg);
    (void)queued.issue(request("prime-row0", Tier::HBM, Op::Read, prime_ns,
        queued.encode(row_addr(0, 0)), 32));
    const auto conflict_ticket = queued.enqueue(request(
        "conflict-row1", Tier::HBM, Op::Read, 1000, queued.encode(row_addr(1, 0)), 32));
    const auto hit_ticket = queued.enqueue(request(
        "hit-row0", Tier::HBM, Op::Read, 1001, queued.encode(row_addr(0, 32)), 32));
    queued.drain_queues();
    const auto conflict_done = queued.pump(conflict_ticket);
    const auto hit_done = queued.pump(hit_ticket);
    std::cout << "  queued: hit finish_ns = " << fixed(hit_done.finish_ns)
              << ", conflict finish_ns = " << fixed(conflict_done.finish_ns) << "\n";
    if (hit_done.finish_ns >= conflict_done.finish_ns) {
        throw std::runtime_error("hbm-frfcfs: a contending row hit must bypass the blocked conflict");
    }
    std::cout << "  frfcfs_hit_first=yes\n";

    // QD1 (synchronous issue): no queue, no reordering — arrival order holds
    // even though the same tRAS block is present.
    HbmDevice qd1(cfg);
    (void)qd1.issue(request("prime-row0", Tier::HBM, Op::Read, prime_ns,
        qd1.encode(row_addr(0, 0)), 32));
    const auto qd1_conflict = qd1.issue(request(
        "conflict-row1", Tier::HBM, Op::Read, 1000, qd1.encode(row_addr(1, 0)), 32));
    const auto qd1_hit = qd1.issue(request(
        "hit-row0", Tier::HBM, Op::Read, 1001, qd1.encode(row_addr(0, 32)), 32));
    std::cout << "  QD1: conflict finish_ns = " << fixed(qd1_conflict.finish_ns)
              << ", late row0 finish_ns = " << fixed(qd1_hit.finish_ns) << "\n";
    if (qd1_conflict.finish_ns >= qd1_hit.finish_ns) {
        throw std::runtime_error("hbm-frfcfs: QD1 must keep arrival order");
    }
    std::cout << "  qd1_in_order=yes\n";

    // Eligibility: a hit that only arrives AFTER the blocked conflict could
    // issue (> 1022 here) must NOT bypass it — the lazy queue must not be
    // clairvoyant about future arrivals.
    HbmDevice late(cfg);
    (void)late.issue(request("prime-row0", Tier::HBM, Op::Read, prime_ns,
        late.encode(row_addr(0, 0)), 32));
    const auto late_conflict_ticket = late.enqueue(request(
        "conflict-row1", Tier::HBM, Op::Read, 1000, late.encode(row_addr(1, 0)), 32));
    const auto late_hit_ticket = late.enqueue(request(
        "late-row0", Tier::HBM, Op::Read, 1100, late.encode(row_addr(0, 32)), 32));
    late.drain_queues();
    const auto late_conflict = late.pump(late_conflict_ticket);
    const auto late_hit = late.pump(late_hit_ticket);
    std::cout << "  late arrival: conflict finish_ns = " << fixed(late_conflict.finish_ns)
              << ", late row0 finish_ns = " << fixed(late_hit.finish_ns) << "\n";
    if (late_conflict.finish_ns >= late_hit.finish_ns) {
        throw std::runtime_error(
            "hbm-frfcfs: a hit arriving after the conflict could issue must not bypass it");
    }
    std::cout << "  no_clairvoyance=yes\n";

    // Read/write turnaround is part of command readiness. The old conflict's
    // PRE is ready well before the younger row-hit RD because the priming
    // write imposes a long WTR gate. Arrival-only eligibility would pick the
    // hit, then stall on RD and delay work that could already issue.
    auto mixed_cfg = cfg;
    mixed_cfg.address_mapping_ns = 20.0;
    mixed_cfg.tWTR_S_ns = 100.0;
    mixed_cfg.tWTR_L_ns = 100.0;
    HbmDevice mixed(mixed_cfg);
    (void)mixed.issue(request("prime-write-row0", Tier::HBM, Op::Write, 0.0,
        mixed.encode(row_addr(0, 0)), 32));
    const auto mixed_conflict_ticket = mixed.enqueue(request(
        "ready-write-conflict", Tier::HBM, Op::Write, 30.0,
        mixed.encode(row_addr(1, 0)), 32));
    const auto mixed_hit_ticket = mixed.enqueue(request(
        "blocked-read-hit", Tier::HBM, Op::Read, 30.0,
        mixed.encode(row_addr(0, 32)), 32));
    mixed.drain_queues();
    const auto mixed_conflict = mixed.pump(mixed_conflict_ticket);
    const auto mixed_hit = mixed.pump(mixed_hit_ticket);
    std::cout << "  mixed R/W readiness: conflict finish_ns = "
              << fixed(mixed_conflict.finish_ns)
              << ", blocked hit finish_ns = " << fixed(mixed_hit.finish_ns) << "\n";
    if (mixed_conflict.finish_ns >= mixed_hit.finish_ns) {
        throw std::runtime_error(
            "hbm-frfcfs: a not-ready mixed-R/W hit delayed a ready conflict");
    }
    std::cout << "  mixed_rw_readiness_ok=yes\n";

    // Anti-starvation: within the eligibility window (arrivals 1005/1010/1015,
    // all before the conflict's deliberately long tRAS issue floor) a tight
    // 10 ns cap lets only the first two hits bypass; the loose default lets
    // all three.
    const auto run_cap_stream = [&row_addr, prime_ns](HbmConfig stream_cfg)
        -> std::vector<PhysicalCompletion> {
        HbmDevice device(stream_cfg);
        (void)device.issue(request("prime-row0", Tier::HBM, Op::Read, prime_ns,
            device.encode(row_addr(0, 0)), 32));
        std::vector<std::uint64_t> tickets;
        tickets.push_back(device.enqueue(request(
            "conflict-row1", Tier::HBM, Op::Read, 1000,
            device.encode(row_addr(1, 0)), 32)));
        for (std::uint32_t k = 1; k <= 3; ++k) {
            tickets.push_back(device.enqueue(request(
                "hit-row0-" + std::to_string(k), Tier::HBM, Op::Read,
                1000.0 + 5.0 * k, device.encode(row_addr(0, 32ull * k)), 32)));
        }
        device.drain_queues();
        std::vector<PhysicalCompletion> completions;
        completions.reserve(tickets.size());
        for (const auto ticket : tickets) {
            completions.push_back(device.pump(ticket));
        }
        return completions;
    };
    auto cap_cfg = cfg;
    cap_cfg.tRAS_ns = 100.0;
    cap_cfg.tRC_ns = 114.0;
    auto tight_cfg = cap_cfg;
    tight_cfg.frfcfs_cap_ns = 10.0;
    const auto tight = run_cap_stream(tight_cfg);
    const auto loose = run_cap_stream(cap_cfg);  // default cap
    std::cout << "  tight cap (10 ns): in-cap hit finish_ns = " << fixed(tight[1].finish_ns)
              << ", conflict finish_ns = " << fixed(tight[0].finish_ns)
              << ", out-of-cap hit finish_ns = " << fixed(tight[3].finish_ns) << "\n";
    std::cout << "  loose cap (default): conflict finish_ns = " << fixed(loose[0].finish_ns)
              << ", last hit finish_ns = " << fixed(loose[3].finish_ns) << "\n";
    if (!(tight[1].finish_ns < tight[0].finish_ns &&
          tight[0].finish_ns < tight[3].finish_ns)) {
        throw std::runtime_error(
            "hbm-frfcfs: tight cap must serve the conflict before out-of-cap hits");
    }
    if (loose[0].finish_ns <= loose[3].finish_ns) {
        throw std::runtime_error("hbm-frfcfs: loose cap should let every eligible hit bypass");
    }
    std::cout << "  starvation_cap_ok=yes\n";

    // A time cap cannot protect an old conflict when every request has the
    // same timestamp. The count cap is one queue-depth window: with QD=4,
    // exactly four row hits may pass before the conflict is forced to issue.
    auto same_time_cfg = cfg;
    same_time_cfg.queue_depth = 4;
    same_time_cfg.frfcfs_cap_ns = 0.0;
    same_time_cfg.tRAS_ns = 1000.0;
    same_time_cfg.tRC_ns = 1014.0;
    HbmDevice same_time(same_time_cfg);
    (void)same_time.issue(request("same-time-prime", Tier::HBM, Op::Read, 0.0,
        same_time.encode(row_addr(0, 0)), 32));
    const auto same_conflict_ticket = same_time.enqueue(request(
        "same-time-conflict", Tier::HBM, Op::Read, 1.0,
        same_time.encode(row_addr(1, 0)), 32));
    std::vector<std::uint64_t> same_hit_tickets;
    for (std::uint64_t k = 1; k <= 8; ++k) {
        same_hit_tickets.push_back(same_time.enqueue(request(
            "same-time-hit-" + std::to_string(k), Tier::HBM, Op::Read, 1.0,
            same_time.encode(row_addr(0, 32 * k)), 32)));
    }
    same_time.drain_queues();
    const auto same_conflict = same_time.pump(same_conflict_ticket);
    std::vector<PhysicalCompletion> same_hits;
    same_hits.reserve(same_hit_tickets.size());
    for (const auto ticket : same_hit_tickets) {
        same_hits.push_back(same_time.pump(ticket));
    }
    if (!(same_hits[3].finish_ns < same_conflict.finish_ns &&
          same_conflict.finish_ns < same_hits[4].finish_ns)) {
        throw std::runtime_error(
            "hbm-frfcfs: equal-timestamp hits exceeded the bounded bypass count");
    }
    std::cout << "  same-arrival: fourth hit finish_ns = "
              << fixed(same_hits[3].finish_ns)
              << ", conflict finish_ns = " << fixed(same_conflict.finish_ns)
              << ", fifth hit finish_ns = " << fixed(same_hits[4].finish_ns) << "\n";
    std::cout << "  same_arrival_bounded_bypass_ok=yes\n";

    // A large logical request must be generated into bounded queues instead
    // of materializing every child twice. The occupancy metric makes that
    // memory invariant directly testable while byte and child counts protect
    // parent aggregation.
    auto streaming_cfg = cfg;
    streaming_cfg.queue_depth = 2;
    HbmDevice streaming(streaming_cfg);
    constexpr std::uint64_t large_bytes = 1ull << 20;
    const auto large = streaming.issue(request(
        "large-streamed-request", Tier::HBM, Op::Read, 0.0, 0, large_bytes));
    const auto& large_stats = streaming.stats();
    const auto expected_children = large_bytes / streaming_cfg.burst_bytes();
    const auto classified_children = large_stats.row_hits + large_stats.row_misses +
        large_stats.row_conflicts;
    if (large.logical_bytes != large_bytes || large.physical_bytes != large_bytes ||
        large_stats.read_bytes != large_bytes ||
        classified_children != expected_children ||
        large_stats.max_queue_occupancy != streaming_cfg.queue_depth) {
        throw std::runtime_error(
            "hbm-frfcfs: large split request violated bounded streaming/count invariants");
    }
    std::cout << "  large request: children=" << expected_children
              << " max_queue_occupancy=" << large_stats.max_queue_occupancy
              << " physical_bytes=" << large.physical_bytes << "\n";
    std::cout << "  large_request_streaming_ok=yes\n";

    HbmDevice ordered(cfg);
    (void)ordered.enqueue(request(
        "ordered-late", Tier::HBM, Op::Read, 10.0,
        ordered.encode(row_addr(0, 0)), 64));
    bool decreasing_arrival_rejected = false;
    try {
        (void)ordered.enqueue(request(
            "ordered-early", Tier::HBM, Op::Read, 9.0,
            ordered.encode(row_addr(0, 64)), 64));
    } catch (const std::runtime_error&) {
        decreasing_arrival_rejected = true;
    }
    if (!decreasing_arrival_rejected) {
        throw std::runtime_error(
            "hbm-frfcfs: decreasing enqueue arrivals must fail closed");
    }
    std::cout << "  decreasing_arrival_rejected=yes\n";
    std::cout << "  expectation: hits bypass only requests they genuinely contend with"
              << " (their first command is ready before the oldest could issue),"
              << " bounded by time and count caps; large parents stream through the"
              << " finite controller queue, and QD1 keeps arrival order.\n";
}

void probe_hbm_tfaw() {
    std::cout << "\n== HBM tFAW probe ==\n";
    HbmConfig cfg;
    cfg.stacks = 1;
    cfg.channels_per_stack = 1;
    cfg.pseudo_channels_per_channel = 1;
    cfg.bank_groups_per_pseudo_channel = 4;
    cfg.banks_per_group = 4;
    HbmDevice hbm(cfg);
    constexpr std::uint32_t kReads = 64;
    for (std::uint32_t i = 0; i < kReads; ++i) {
        HbmAddress addr;
        addr.bank_group = i % cfg.bank_groups_per_pseudo_channel;
        addr.bank = (i / cfg.bank_groups_per_pseudo_channel) % cfg.banks_per_group;
        addr.row = i / (cfg.bank_groups_per_pseudo_channel * cfg.banks_per_group);
        (void)hbm.issue(request("tfaw-stream", Tier::HBM, Op::Read, 0, hbm.encode(addr), 64));
    }
    const auto& s = hbm.stats();
    // Consecutive activations rotate bank groups, so tRRD_S is the only other
    // ACT-to-ACT constraint and tFAW must set the sustained activation rate.
    const double tfaw_floor_ns = (kReads / 4.0 - 1.0) * cfg.tFAW_ns;
    print_hbm_stats(hbm);
    std::cout << "  tfaw_floor_ns = " << fixed(tfaw_floor_ns) << "\n";
    std::cout << "  expectation: every access activates; at most 4 activations per tFAW"
              << " window, so finish stays above the floor.\n";
    if (s.activations != kReads) {
        throw std::runtime_error("hbm-tfaw: expected one activation per access");
    }
    if (s.finish_ns < tfaw_floor_ns) {
        throw std::runtime_error("hbm-tfaw: tFAW under-enforced (more than 4 ACTs per window)");
    }
    std::cout << "  tfaw_ok=yes\n";
}

void run_all() {
    probe_hbm_interface();
    probe_hbm_row();
    probe_hbm_boundaries();
    probe_hbm_channels();
    probe_hbm_refresh();
    probe_hbm_turnaround();
    probe_hbm_tfaw();
    probe_hbm_frfcfs();
    probe_hbf_basic();
    probe_hbf_page_read_admission();
    probe_hbf_planes();
    probe_hbf_parallelism();
    probe_hbf_media_lanes();
    probe_hbf_page_buffer_banks();
    probe_hbf_subarrays();
    probe_hbf_exact_calendar();
    probe_hbf_ecc_pipeline();
    probe_hbf_io();
    probe_hbf_rounding();
    probe_hbf_resident_mapping();
    probe_hbf_gc();
    probe_hbf_tsu_scheduler();
    probe_hbf_write_coalescer();
    probe_hbf_write_backpressure();
    probe_hbf_mapping_writeback();
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
              << "  probes: all, hbm-interface, hbm-row, hbm-boundaries, hbm-channels, hbm-refresh,"
              << " hbm-turnaround, hbm-tfaw,"
              << " hbm-frfcfs,"
              << " hbf-basic, hbf-page-read-admission, hbf-planes, hbf-io,"
              << " hbf-rounding, hbf-resident-mapping, hbf-gc,"
              << " hbf-parallelism, hbf-media-lanes, hbf-page-buffer-banks,"
              << " hbf-subarrays, hbf-exact-calendar, hbf-ecc-pipeline, hbf-tsu,"
              << " hbf-coalesce, hbf-write-backpressure, hbf-mapping-writeback,"
              << " composition-kind-blind, composition-static-mapping,"
              << " composition-initial-image, composition-coop-write,"
              << " composition-reuse-routing\n";
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
        } else if (g_options.probe == "hbm-row") {
            probe_hbm_row();
        } else if (g_options.probe == "hbm-boundaries") {
            probe_hbm_boundaries();
        } else if (g_options.probe == "hbm-channels") {
            probe_hbm_channels();
        } else if (g_options.probe == "hbm-refresh") {
            probe_hbm_refresh();
        } else if (g_options.probe == "hbm-turnaround") {
            probe_hbm_turnaround();
        } else if (g_options.probe == "hbm-tfaw") {
            probe_hbm_tfaw();
        } else if (g_options.probe == "hbm-frfcfs") {
            probe_hbm_frfcfs();
        } else if (g_options.probe == "hbf-basic") {
            probe_hbf_basic();
        } else if (g_options.probe == "hbf-page-read-admission") {
            probe_hbf_page_read_admission();
        } else if (g_options.probe == "hbf-planes") {
            probe_hbf_planes();
        } else if (g_options.probe == "hbf-parallelism") {
            probe_hbf_parallelism();
        } else if (g_options.probe == "hbf-media-lanes") {
            probe_hbf_media_lanes();
        } else if (g_options.probe == "hbf-page-buffer-banks") {
            probe_hbf_page_buffer_banks();
        } else if (g_options.probe == "hbf-subarrays") {
            probe_hbf_subarrays();
        } else if (g_options.probe == "hbf-exact-calendar") {
            probe_hbf_exact_calendar();
        } else if (g_options.probe == "hbf-ecc-pipeline") {
            probe_hbf_ecc_pipeline();
        } else if (g_options.probe == "hbf-io") {
            probe_hbf_io();
        } else if (g_options.probe == "hbf-rounding") {
            probe_hbf_rounding();
        } else if (g_options.probe == "hbf-resident-mapping") {
            probe_hbf_resident_mapping();
        } else if (g_options.probe == "hbf-gc") {
            probe_hbf_gc();
        } else if (g_options.probe == "hbf-tsu") {
            probe_hbf_tsu_scheduler();
        } else if (g_options.probe == "hbf-coalesce") {
            probe_hbf_write_coalescer();
        } else if (g_options.probe == "hbf-write-backpressure") {
            probe_hbf_write_backpressure();
        } else if (g_options.probe == "hbf-mapping-writeback") {
            probe_hbf_mapping_writeback();
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
