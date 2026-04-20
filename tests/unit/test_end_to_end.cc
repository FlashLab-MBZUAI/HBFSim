#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>

#include "src/base/simulation.hh"
#include "src/controller/hbf_controller.hh"
#include "src/frontend/requester.hh"
#include "src/host/host_bus.hh"
#include "src/logic_die/logic_die.hh"
#include "src/media/nand/nand_stack.hh"
#include "src/memory_system/memory_system.hh"

using namespace obelisk;

static std::string write_tmp_trace() {
    std::string path = std::tmpnam(nullptr);
    path += ".trace";
    std::ofstream out(path);
    // 8 sequential page reads, first at ts=0, rest back-to-back.
    out << "0 R 0x1800000000 16384 0 read\n";
    for (int i = 1; i < 8; ++i) {
        out << "-1 R 0x" << std::hex << (0x1800000000ULL + i * 16384ULL) << std::dec << " 16384 0 read\n";
    }
    return path;
}

TEST_CASE("End-to-end: small HBF trace completes and produces CSV", "[e2e]") {
    std::filesystem::create_directory("e2e_out");
    auto trace = write_tmp_trace();
    std::string csv = "e2e_out/e2e.csv";
    std::string json = "e2e_out/e2e.json";

    SimulationConfig sim_cfg;
    sim_cfg.max_cycles = 10'000'000;
    sim_cfg.output_csv_path = csv;
    sim_cfg.output_stats_json_path = json;
    sim_cfg.log_level = 0;
    Simulation sim(sim_cfg);

    RequesterConfig rcfg;
    rcfg.trace_file = trace;
    rcfg.max_in_flight = 32;
    rcfg.issue_rate_per_cyc = 2;
    auto requester = std::make_shared<Requester>(rcfg);

    HostBusConfig hbcfg;
    hbcfg.bandwidth_gbps = 128.0;
    hbcfg.base_latency_cycles = 20;
    hbcfg.cpu_compute_cycles = 20;
    hbcfg.freq_mhz = 1000;
    auto host_bus = std::make_shared<HostBus>(hbcfg);

    AddressMapperConfig amcfg;
    amcfg.num_channels = 4;
    amcfg.num_dies = 4;
    amcfg.num_subarrays = 8;
    amcfg.page_size_bytes = 16384;
    amcfg.hbm_base_addr = 0x0;
    amcfg.hbm_size_bytes = 0x40000000;
    amcfg.hbf_base_addr = 0x1800000000;
    amcfg.hbf_size_bytes = 0x40000000;
    auto mapper = std::make_unique<AddressMapper>(amcfg);

    HBFControllerConfig ctrl_cfg;
    auto ctrl = std::make_shared<HBFController>(ctrl_cfg);

    LogicDieConfig ld_cfg;
    ld_cfg.num_channels = 4;
    ld_cfg.num_dies_per_stack = 4;
    ld_cfg.num_subarrays_per_die = 8;
    auto ld = std::make_shared<LogicDie>(ld_cfg);

    NANDStackConfig ns_cfg;
    ns_cfg.num_dies_per_stack = 4;
    ns_cfg.num_subarrays_per_die = 8;
    ns_cfg.subarray_cfg.tR_cycles = 10'000;
    ns_cfg.subarray_cfg.queue_depth = 4;
    auto nand = std::make_shared<NANDStack>(ns_cfg);

    for (uint32_t d = 0; d < ld_cfg.num_dies_per_stack; ++d) {
        ld->connect_nand_die(d, nand->get_die(d));
    }
    ctrl->connect_downstream(ld.get());

    auto mem_sys = std::make_shared<MemorySystem>(std::move(mapper), nullptr, ctrl.get());
    requester->connect_downstream(host_bus.get());
    host_bus->connect_downstream(mem_sys.get());

    sim.register_module(requester);
    sim.register_module(host_bus);
    sim.register_module(mem_sys);
    sim.register_module(ctrl);
    sim.register_module(ld);
    sim.register_module(nand);

    sim.run();

    auto s = sim.stats().compute();
    REQUIRE(s.completed_requests == 8);
    REQUIRE(s.total_requests == 8);
    REQUIRE(s.hbf_read_bytes == 8 * 16384);
    REQUIRE(s.mean_latency_cycles > 10'000);  // at least tR

    // CSV file exists and has 1 header line + 8 data lines.
    std::ifstream cin(csv);
    REQUIRE(cin.good());
    int lines = 0;
    std::string line;
    while (std::getline(cin, line)) ++lines;
    REQUIRE(lines == 9);

    std::remove(trace.c_str());
}
