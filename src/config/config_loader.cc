#include "src/config/config_loader.hh"

#include <toml++/toml.h>

#include <cctype>
#include <stdexcept>
#include <string>

namespace obelisk {

uint64_t parse_byte_size(const std::string& s) {
    if (s.empty()) throw std::runtime_error("parse_byte_size: empty");
    if (s.size() > 1 && (s[0] == '0') && (s[1] == 'x' || s[1] == 'X')) {
        return std::stoull(s.substr(2), nullptr, 16);
    }
    size_t i = 0;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '_')) ++i;
    std::string num;
    num.reserve(i);
    for (size_t k = 0; k < i; ++k)
        if (s[k] != '_') num += s[k];
    if (num.empty()) throw std::runtime_error("parse_byte_size: no number in " + s);
    uint64_t base = std::stoull(num);
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    std::string unit;
    for (; i < s.size(); ++i) {
        if (!std::isalpha(static_cast<unsigned char>(s[i]))) break;
        unit += std::toupper(static_cast<unsigned char>(s[i]));
    }
    if (unit.empty() || unit == "B") return base;
    if (unit == "KB" || unit == "KIB" || unit == "K") return base * (1ULL << 10);
    if (unit == "MB" || unit == "MIB" || unit == "M") return base * (1ULL << 20);
    if (unit == "GB" || unit == "GIB" || unit == "G") return base * (1ULL << 30);
    if (unit == "TB" || unit == "TIB" || unit == "T") return base * (1ULL << 40);
    throw std::runtime_error("parse_byte_size: unknown unit in " + s);
}

uint64_t parse_addr(const std::string& s) {
    if (s.size() > 1 && (s[0] == '0') && (s[1] == 'x' || s[1] == 'X')) {
        return std::stoull(s.substr(2), nullptr, 16);
    }
    return std::stoull(s, nullptr, 0);
}

namespace {

// Read a byte-size field that may be either an integer or a human string.
template <typename NodeView>
uint64_t get_bytes(const NodeView& n, uint64_t def) {
    if (auto s = n.template value<std::string>()) return parse_byte_size(*s);
    if (auto v = n.template value<int64_t>()) return static_cast<uint64_t>(*v);
    return def;
}

template <typename NodeView>
Addr get_addr(const NodeView& n, Addr def) {
    if (auto s = n.template value<std::string>()) return parse_addr(*s);
    if (auto v = n.template value<int64_t>()) return static_cast<Addr>(*v);
    return def;
}

}  // namespace

ObeliskConfig load_config(const std::string& path) {
    toml::table root = toml::parse_file(path);
    ObeliskConfig cfg;

    // [simulation]
    cfg.simulation.max_cycles = root["simulation"]["max_cycles"].value_or<int64_t>(100'000'000);
    cfg.simulation.warmup_cycles = root["simulation"]["warmup_cycles"].value_or<int64_t>(0);
    cfg.simulation.freq_mhz = root["simulation"]["freq_mhz"].value_or<int64_t>(1000);
    cfg.simulation.output_csv_path = root["simulation"]["output_csv_path"].value_or<std::string>("");
    cfg.simulation.output_stats_json_path = root["simulation"]["output_stats_json_path"].value_or<std::string>("");
    cfg.simulation.log_level = root["simulation"]["log_level"].value_or<int64_t>(1);

    // [requester]
    cfg.requester.trace_file = root["requester"]["trace_file"].value_or<std::string>("");
    cfg.requester.max_in_flight = root["requester"]["max_in_flight"].value_or<int64_t>(64);
    cfg.requester.issue_rate_per_cyc = root["requester"]["issue_rate_per_cyc"].value_or<int64_t>(1);
    cfg.requester.warm_up = root["requester"]["warm_up"].value_or<bool>(false);
    cfg.requester.warm_up_cycles = root["requester"]["warm_up_cycles"].value_or<int64_t>(0);

    // [host_bus]
    cfg.host_bus.bandwidth_gbps = root["host_bus"]["bandwidth_gbps"].value_or<double>(128.0);
    cfg.host_bus.base_latency_cycles = root["host_bus"]["base_latency_cycles"].value_or<int64_t>(50);
    cfg.host_bus.queue_depth = root["host_bus"]["queue_depth"].value_or<int64_t>(32);
    cfg.host_bus.cpu_compute_cycles = root["host_bus"]["cpu_compute_cycles"].value_or<int64_t>(100);
    cfg.host_bus.freq_mhz = cfg.simulation.freq_mhz;

    // [address_mapping]
    std::string policy_str = root["address_mapping"]["policy"].value_or<std::string>("channel_interleave_first");
    cfg.address_mapper.policy = parse_mapping_policy(policy_str);
    cfg.address_mapper.num_stacks = root["address_mapping"]["num_stacks"].value_or<int64_t>(1);
    cfg.address_mapper.num_channels = root["address_mapping"]["num_channels"].value_or<int64_t>(8);
    cfg.address_mapper.num_dies = root["address_mapping"]["num_dies_hbf"].value_or<int64_t>(16);
    cfg.address_mapper.num_subarrays = root["address_mapping"]["num_subarrays_hbf"].value_or<int64_t>(32);
    cfg.address_mapper.num_hbm_banks = root["address_mapping"]["num_hbm_banks"].value_or<int64_t>(64);
    cfg.address_mapper.page_size_bytes = root["address_mapping"]["page_size_bytes"].value_or<int64_t>(16384);
    cfg.address_mapper.hbm_base_addr = get_addr(root["address_mapping"]["hbm_base_addr"], 0x0ULL);
    cfg.address_mapper.hbm_size_bytes = get_bytes(root["address_mapping"]["hbm_size_bytes"], 96ULL << 30);
    cfg.address_mapper.hbf_base_addr = get_addr(root["address_mapping"]["hbf_base_addr"], 0x1800000000ULL);
    cfg.address_mapper.hbf_size_bytes = get_bytes(root["address_mapping"]["hbf_size_bytes"], 512ULL << 30);

    // [hbf_controller]
    cfg.hbf_controller.read_queue_depth = root["hbf_controller"]["read_queue_depth"].value_or<int64_t>(128);
    cfg.hbf_controller.write_queue_depth = root["hbf_controller"]["write_queue_depth"].value_or<int64_t>(64);
    cfg.hbf_controller.scheduler_type = root["hbf_controller"]["scheduler_type"].value_or<std::string>("frfcfs");
    cfg.hbf_controller.controller_latency_cycles =
        root["hbf_controller"]["controller_latency_cycles"].value_or<int64_t>(10);
    cfg.hbf_controller.enable_read_priority = root["hbf_controller"]["enable_read_priority"].value_or<bool>(true);
    cfg.hbf_controller.max_outstanding_per_subarray =
        root["hbf_controller"]["max_outstanding_per_subarray"].value_or<int64_t>(4);

    // [logic_die]
    cfg.logic_die.num_channels = root["logic_die"]["num_channels"].value_or<int64_t>(8);
    cfg.logic_die.num_dies_per_stack = root["logic_die"]["num_dies_per_stack"].value_or<int64_t>(16);
    cfg.logic_die.num_subarrays_per_die = root["logic_die"]["num_subarrays_per_die"].value_or<int64_t>(32);
    cfg.logic_die.tsv_bandwidth_gbps = root["logic_die"]["tsv_bandwidth_gbps"].value_or<int64_t>(200);
    cfg.logic_die.logic_die_freq_mhz = root["logic_die"]["logic_die_freq_mhz"].value_or<int64_t>(1000);
    cfg.logic_die.command_translation_cycles = root["logic_die"]["command_translation_cycles"].value_or<int64_t>(2);
    cfg.logic_die.max_concurrent_subarray_cmds =
        root["logic_die"]["max_concurrent_subarray_cmds"].value_or<int64_t>(64);

    // [nand_media]
    cfg.nand_stack.num_dies_per_stack =
        root["nand_media"]["num_dies_per_stack"].value_or<int64_t>(cfg.logic_die.num_dies_per_stack);
    cfg.nand_stack.num_subarrays_per_die =
        root["nand_media"]["num_subarrays_per_die"].value_or<int64_t>(cfg.logic_die.num_subarrays_per_die);
    cfg.nand_stack.capacity_per_die_bytes = get_bytes(root["nand_media"]["capacity_per_die_bytes"], 32ULL << 30);
    cfg.nand_stack.subarray_cfg.page_size_bytes = root["nand_media"]["page_size_bytes"].value_or<int64_t>(16384);
    cfg.nand_stack.subarray_cfg.pages_per_block = root["nand_media"]["pages_per_block"].value_or<int64_t>(512);
    cfg.nand_stack.subarray_cfg.blocks_per_subarray = root["nand_media"]["blocks_per_subarray"].value_or<int64_t>(1024);
    cfg.nand_stack.subarray_cfg.tR_cycles = root["nand_media"]["tR_cycles"].value_or<int64_t>(50'000);
    cfg.nand_stack.subarray_cfg.tPROG_cycles = root["nand_media"]["tPROG_cycles"].value_or<int64_t>(500'000);
    cfg.nand_stack.subarray_cfg.tBERS_cycles = root["nand_media"]["tBERS_cycles"].value_or<int64_t>(3'000'000);
    cfg.nand_stack.subarray_cfg.cache_buffer_pages = root["nand_media"]["cache_buffer_pages"].value_or<int64_t>(4);
    cfg.nand_stack.subarray_cfg.cache_hit_read_cycles =
        root["nand_media"]["cache_hit_read_cycles"].value_or<int64_t>(500);
    cfg.nand_stack.subarray_cfg.queue_depth =
        root["nand_media"]["queue_depth"].value_or<int64_t>(cfg.hbf_controller.max_outstanding_per_subarray);

    cfg.has_hbf = root.contains("nand_media") || root.contains("hbf_controller");

    // [hbm_media]
    cfg.has_hbm = root.contains("hbm_media");
    cfg.hbm.num_banks = root["hbm_media"]["num_banks"].value_or<int64_t>(cfg.address_mapper.num_hbm_banks);
    cfg.hbm.tCL_cycles = root["hbm_media"]["tCL_cycles"].value_or<int64_t>(14);
    cfg.hbm.tRCD_cycles = root["hbm_media"]["tRCD_cycles"].value_or<int64_t>(14);
    cfg.hbm.tRP_cycles = root["hbm_media"]["tRP_cycles"].value_or<int64_t>(14);
    cfg.hbm.bandwidth_gbps = root["hbm_media"]["bandwidth_gbps"].value_or<double>(1024.0);
    cfg.hbm.freq_mhz = cfg.simulation.freq_mhz;
    cfg.hbm.capacity_bytes = get_bytes(root["hbm_media"]["capacity_bytes"], cfg.address_mapper.hbm_size_bytes);
    cfg.hbm.access_granularity = root["hbm_media"]["access_granularity"].value_or<int64_t>(64);
    cfg.hbm.dramsim3_config_file = root["hbm_media"]["dramsim3_config_file"].value_or<std::string>("");
    cfg.hbm.output_dir = root["hbm_media"]["output_dir"].value_or<std::string>("");

    return cfg;
}

}  // namespace obelisk
