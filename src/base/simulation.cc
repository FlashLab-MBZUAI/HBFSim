#include "src/base/simulation.hh"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace obelisk {

Simulation::Simulation(const SimulationConfig& cfg) : cfg_(cfg) {
    stats_.set_freq_mhz(cfg.freq_mhz);
    if (!cfg.output_csv_path.empty()) {
        csv_out_.open(cfg.output_csv_path);
        if (!csv_out_) throw std::runtime_error("simulation: cannot open " + cfg.output_csv_path);
        write_csv_header();
    }

    switch (cfg.log_level) {
        case 0: spdlog::set_level(spdlog::level::off); break;
        case 1: spdlog::set_level(spdlog::level::info); break;
        case 2: spdlog::set_level(spdlog::level::debug); break;
        case 3: spdlog::set_level(spdlog::level::trace); break;
        default: spdlog::set_level(spdlog::level::info); break;
    }
}

Simulation::~Simulation() {
    if (csv_out_.is_open()) csv_out_.close();
}

void Simulation::register_module(std::shared_ptr<IModule> m) {
    m->set_simulation(this);
    modules_.push_back(std::move(m));
}

void Simulation::write_csv_header() {
    csv_out_ << Packet::csv_header() << "\n";
    csv_header_written_ = true;
}

void Simulation::log_completed(const Packet& pkt) {
    stats_.record_completion(pkt);
    if (csv_out_.is_open()) {
        csv_out_ << pkt.to_csv_row() << "\n";
    }
}

bool Simulation::any_pending() const {
    for (const auto& m : modules_) {
        if (m->has_pending()) return true;
    }
    return false;
}

void Simulation::run() {
    spdlog::info("Simulation::run start, max_cycles={}, modules={}", cfg_.max_cycles, modules_.size());

    for (cycle_ = 0; cycle_ < cfg_.max_cycles; ++cycle_) {
        for (auto& m : modules_) m->tick(cycle_);

        if (!any_pending()) {
            spdlog::info("all modules idle at cycle {}", cycle_);
            break;
        }

        if ((cycle_ & 0xFFFFF) == 0 && cycle_ != 0) {
            spdlog::debug("cycle={}", cycle_);
        }
    }

    stats_.set_total_cycles(cycle_);

    spdlog::info("Simulation::run done, cycles={}", cycle_);

    if (!cfg_.output_stats_json_path.empty()) {
        stats_.dump_json(cfg_.output_stats_json_path);
        spdlog::info("stats dumped to {}", cfg_.output_stats_json_path);
    }

    if (csv_out_.is_open()) csv_out_.flush();
}

}  // namespace obelisk
