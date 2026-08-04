#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/request.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

namespace {

struct Operation {
    std::uint64_t arrival_cycle = 0;
    int type = Ramulator::Request::Type::Read;
    std::uint64_t address = 0;
    int bytes = 0;
};

std::vector<Operation> load_trace(const char* path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open trace");
    }
    std::vector<Operation> operations;
    std::string line;
    std::uint64_t previous_arrival = 0;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            throw std::runtime_error("trace contains a blank line");
        }
        std::istringstream fields(line);
        std::uint64_t arrival = 0;
        std::uint64_t address = 0;
        int bytes = 0;
        std::string type;
        std::string trailing;
        if (!(fields >> arrival >> type >> address >> bytes) ||
            (fields >> trailing)) {
            throw std::runtime_error(
                "trace line " + std::to_string(line_number) +
                " must contain exactly: arrival_cycle R|W address bytes");
        }
        if (type != "R" && type != "W") {
            throw std::runtime_error(
                "trace line " + std::to_string(line_number) +
                " has an invalid operation");
        }
        if (bytes <= 0) {
            throw std::runtime_error(
                "trace line " + std::to_string(line_number) +
                " has a nonpositive request size");
        }
        if (!operations.empty() && arrival < previous_arrival) {
            throw std::runtime_error(
                "trace arrival cycles must be nondecreasing");
        }
        previous_arrival = arrival;
        operations.push_back({
            arrival,
            type == "R" ? Ramulator::Request::Type::Read
                        : Ramulator::Request::Type::Write,
            address,
            bytes,
        });
    }
    if (operations.empty()) {
        throw std::runtime_error("trace is empty");
    }
    if (operations.front().arrival_cycle != 0) {
        throw std::runtime_error(
            "the differential service epoch requires first arrival cycle 0");
    }
    return operations;
}

int run(const char* config_path, const char* trace_path) {
    const auto config = Ramulator::Config::parse_config_file(config_path);
    auto* frontend = Ramulator::Factory::create_frontend(config);
    auto* memory = Ramulator::Factory::create_memory_system(config);
    if (frontend == nullptr || memory == nullptr) {
        throw std::runtime_error("Ramulator factory returned null");
    }
    frontend->connect_memory_system(memory);
    memory->connect_frontend(frontend);

    const auto operations = load_trace(trace_path);
    const auto transaction_bytes = memory->get_tx_bytes();
    if (transaction_bytes <= 0) {
        throw std::runtime_error("Ramulator reported invalid transaction bytes");
    }
    for (const auto& operation : operations) {
        if (operation.bytes != transaction_bytes ||
            operation.address % transaction_bytes != 0) {
            throw std::runtime_error(
                "every differential request must be one aligned transaction");
        }
    }

    std::size_t next = 0;
    std::size_t completed = 0;
    std::int64_t last_depart_cycle = -1;
    std::uint64_t driver_cycle = 0;
    while (completed != operations.size()) {
        while (next != operations.size() &&
               operations[next].arrival_cycle <= driver_cycle) {
            const auto& operation = operations[next];
            const bool accepted = frontend->receive_external_requests(
                operation.type,
                operation.address,
                0,
                [&](Ramulator::Request& request) {
                    if (request.depart < 0) {
                        throw std::runtime_error(
                            "Ramulator completed a request without depart cycle");
                    }
                    ++completed;
                    last_depart_cycle = std::max(
                        last_depart_cycle, request.depart);
                },
                operation.bytes);
            if (!accepted) {
                break;
            }
            ++next;
        }
        memory->tick();
        ++driver_cycle;
        if (driver_cycle > 1000000) {
            throw std::runtime_error("simulation did not quiesce");
        }
    }
    if (last_depart_cycle < 0) {
        throw std::runtime_error("simulation emitted no completion cycle");
    }

    frontend->update_stats_recursive();
    memory->update_stats_recursive();
    auto stats = memory->collect_stats();
    auto controller = stats["controller"];
    if (controller.is_sequence()) {
        controller = controller.seq().at(0);
    }
    const auto reads = stats["total_num_read_requests"].as<int>();
    const auto writes = stats["total_num_write_requests"].as<int>();
    const auto row_hits =
        controller["row_hits"].as<unsigned long long>();
    const auto row_misses =
        controller["row_misses"].as<unsigned long long>();
    const auto row_conflicts =
        controller["row_conflicts"].as<unsigned long long>();
    if (reads < 0 || writes < 0 ||
        static_cast<std::size_t>(reads + writes) != operations.size() ||
        row_hits + row_misses + row_conflicts != operations.size()) {
        throw std::runtime_error(
            "Ramulator statistics do not cover the submitted requests");
    }

    // ControllerBase::tick_prologue() increments m_clk before it issues the
    // first cycle-0 request. Therefore controller cycle 1 is the explicit
    // request-service epoch for this driver; preserving both raw cycle values
    // avoids hiding the one-cycle coordinate offset in an adapter constant.
    constexpr std::uint64_t request_epoch_cycle = 1;
    std::cout << std::setprecision(17)
              << "{"
              << "\"clock_period_ns\":" << memory->get_tCK() << ","
              << "\"request_epoch_cycle\":" << request_epoch_cycle << ","
              << "\"completion_cycle\":" << last_depart_cycle << ","
              << "\"transaction_bytes\":" << transaction_bytes << ","
              << "\"read_requests\":" << reads << ","
              << "\"write_requests\":" << writes << ","
              << "\"row_hits\":" << row_hits << ","
              << "\"row_misses\":" << row_misses << ","
              << "\"row_conflicts\":" << row_conflicts
              << "}\n";

    frontend->finalize();
    memory->finalize();
    delete frontend;
    delete memory;
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: ramulator2_driver CONFIG TRACE\n";
        return 2;
    }
    try {
        return run(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << "ramulator2_driver error: " << error.what() << '\n';
        return 1;
    }
}
