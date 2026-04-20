#include <spdlog/spdlog.h>

#include <cstdio>
#include <exception>
#include <memory>
#include <string>

#include "src/base/simulation.hh"
#include "src/config/config_loader.hh"
#include "src/controller/hbf_controller.hh"
#include "src/frontend/requester.hh"
#include "src/host/host_bus.hh"
#include "src/logic_die/logic_die.hh"
#include "src/media/hbm/hbm_interface.hh"
#include "src/media/nand/nand_stack.hh"
#include "src/memory_system/memory_system.hh"

using namespace obelisk;

static int run(const std::string& config_path) {
    auto cfg = load_config(config_path);

    Simulation sim(cfg.simulation);

    auto requester = std::make_shared<Requester>(cfg.requester);
    auto host_bus = std::make_shared<HostBus>(cfg.host_bus);

    auto mapper = std::make_unique<AddressMapper>(cfg.address_mapper);

    // HBF stack (controller + logic die + nand).
    std::shared_ptr<HBFController> hbf_controller;
    std::shared_ptr<LogicDie> logic_die;
    std::shared_ptr<NANDStack> nand_stack;
    if (cfg.has_hbf) {
        hbf_controller = std::make_shared<HBFController>(cfg.hbf_controller);
        logic_die = std::make_shared<LogicDie>(cfg.logic_die);
        nand_stack = std::make_shared<NANDStack>(cfg.nand_stack);

        for (uint32_t d = 0; d < cfg.logic_die.num_dies_per_stack; ++d) {
            auto* die = nand_stack->get_die(d);
            if (die) logic_die->connect_nand_die(d, die);
        }

        hbf_controller->connect_downstream(logic_die.get());
    }

    // HBM.
    std::shared_ptr<HBMInterface> hbm;
    if (cfg.has_hbm) {
        hbm = std::make_shared<HBMInterface>(cfg.hbm);
    }

    auto mem_system =
        std::make_shared<MemorySystem>(std::move(mapper), hbm ? hbm.get() : nullptr,
                                       hbf_controller ? hbf_controller.get() : nullptr);

    requester->connect_downstream(host_bus.get());
    host_bus->connect_downstream(mem_system.get());

    // Registration order matters: upstream modules first so they see
    // freshly-dequeued packets same cycle.
    sim.register_module(requester);
    sim.register_module(host_bus);
    sim.register_module(mem_system);
    if (hbf_controller) sim.register_module(hbf_controller);
    if (logic_die) sim.register_module(logic_die);
    if (nand_stack) sim.register_module(nand_stack);
    if (hbm) sim.register_module(hbm);

    sim.run();

    auto s = sim.stats().compute();
    spdlog::info("completed requests: {} / {}", s.completed_requests, s.total_requests);
    spdlog::info("aggregate bandwidth: {:.2f} GB/s (hbm={:.2f} hbf={:.2f})", s.aggregate_gbps(),
                 s.hbm_read_gbps() + s.hbm_write_gbps(), s.hbf_read_gbps() + s.hbf_write_gbps());
    spdlog::info("latency mean={:.1f} p50={} p95={} p99={} max={}", s.mean_latency_cycles, s.p50_latency,
                 s.p95_latency, s.p99_latency, s.max_latency);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <config.toml>\n", argv[0]);
        return 2;
    }
    try {
        return run(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "obelisk: error: %s\n", e.what());
        return 1;
    }
}
