#include "../../verification/probes/hbf_with_hbm.hpp"
#include "host/hbf_controller.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hbfsim::host {
using namespace hbfsim::physical;

// A drain must finish every relocation still in flight. Each HBF stack owns
// its GC control resources, its copy slot and its reclaim dependencies, so
// the leftover relocations of one stack must not push another stack's copies
// into the future: draining two stacks together costs about what draining the
// slower one alone costs, not the sum.
struct HbfGcIndexTestAccess {
    static std::size_t arm(HbfController& device, std::size_t stack, double at_ns) {
        std::size_t armed = 0;
        for (std::size_t block = 0; block < device.blocks_.size(); ++block) {
            const auto& candidate = device.blocks_[block];
            if (device.stack_of_block(block) != stack ||
                candidate.role != HbfController::BlockRole::Data ||
                candidate.free_pages != 0 || candidate.invalid_pages == 0 ||
                candidate.valid_pages == 0) {
                continue;
            }
            Breakdown ignored;
            double cursor = at_ns;
            device.gc_victim_by_stack_[stack] = device.start_relocation(
                block,
                HbfController::RelocationPurpose::GarbageCollection,
                cursor,
                ignored,
                nullptr);
            ++device.stats_.gc_runs;
            ++armed;
            break;
        }
        return armed;
    }
};
}

namespace {
using hbfsim::host::HbfConfig;
using HbfController = hbfsim::verification::HbfWithHbm;
using hbfsim::host::HbfGcIndexTestAccess;
using hbfsim::physical::AddressSpace;
using hbfsim::physical::Op;
using hbfsim::physical::PhysicalRequest;
using hbfsim::physical::Tier;

constexpr std::size_t kStacks = 2;

HbfConfig drain_config() {
    HbfConfig config;
    config.device.stacks = kStacks;
    config.device.channels_per_stack = 1;
    config.device.planes_per_die = 1;
    config.device.blocks_per_plane = 16;
    config.device.pages_per_block = 4;
    config.host.mapping_entries_per_page = 64;
    config.host.auto_gc_enabled = false;
    return config;
}

PhysicalRequest overwrite(std::uint64_t addr, std::uint64_t bytes) {
    PhysicalRequest request;
    request.id = "overwrite";
    request.tier = Tier::HBF;
    request.op = Op::Write;
    request.address_space = AddressSpace::Logical;
    request.arrival_ns = 0.0;
    request.addr = addr;
    request.bytes = bytes;
    return request;
}

// Returns the drain cost, in ns, of finishing relocations armed on `stacks`.
double drain_cost_ns(const std::vector<std::size_t>& stacks) {
    HbfController device(drain_config());
    device.prepopulate_logical_pages({0, 1, 2, 3, 4, 5, 6, 7});
    double at_ns = 0.0;
    for (int i = 0; i < 2; ++i) {
        at_ns = std::max(
            at_ns,
            device.issue(overwrite(
                static_cast<std::uint64_t>(i) * device.config().device.page_size_bytes,
                device.config().device.page_size_bytes)).finish_ns);
    }
    at_ns = device.drain_pending("setup", at_ns).finish_ns;
    std::size_t armed = 0;
    for (const auto stack : stacks) {
        armed += HbfGcIndexTestAccess::arm(device, stack, at_ns);
    }
    if (armed != stacks.size()) {
        throw std::runtime_error(
            "GC drain test could not arm a victim on every requested stack");
    }
    return device.drain_pending("gc-drain", at_ns).finish_ns - at_ns;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("HBF GC drain parallelism: " + message);
    }
}
}

int main() {
    try {
        const auto stack0 = drain_cost_ns({0});
        const auto stack1 = drain_cost_ns({1});
        const auto both = drain_cost_ns({0, 1});
        const auto slowest = std::max(stack0, stack1);
        std::cout << "gc drain stack0=" << stack0 << "ns stack1=" << stack1
                  << "ns both=" << both << "ns\n";
        require(stack0 > 0.0 && stack1 > 0.0, "a single-stack drain must cost time");
        // Sequential chaining across stacks would cost about stack0 + stack1.
        require(
            both < stack0 + stack1 - 0.25 * slowest,
            "draining two stacks together must not cost the sum of both");
        // Shared host-memory and mapping contention is still charged, so the
        // joint drain may exceed the slowest stack by a small margin only.
        require(
            both >= slowest && both <= slowest * 1.05,
            "a joint drain must stay close to the slowest stack alone");
        std::cout << "hbf gc drain parallelism contract ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
