#pragma once

#include <memory>
#include <optional>
#include <queue>
#include <string>

#include "obelisk/imodule.hh"
#include "src/frontend/trace_reader.hh"

namespace obelisk {

struct RequesterConfig {
    std::string trace_file;
    uint32_t max_in_flight = 64;
    uint32_t issue_rate_per_cyc = 1;
    bool warm_up = false;
    Cycle warm_up_cycles = 0;
};

// Injects requests from a trace. Collects completed packets that come back
// up the pipeline, logs them, and decrements in-flight count.
class Requester : public IModule {
public:
    explicit Requester(const RequesterConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(Cycle current_cycle) override;
    bool accept(std::unique_ptr<Packet>& pkt) override;  // completion fallback path
    bool has_pending() const override;

    void connect_downstream(IModule* m) { downstream_ = m; }

    uint64_t total_issued() const { return total_issued_; }
    uint64_t total_completed() const { return total_completed_; }

private:
    std::string name_ = "Requester";
    RequesterConfig cfg_;
    std::unique_ptr<TraceReader> trace_;

    IModule* downstream_ = nullptr;

    // Pending trace record that could not be pushed last tick.
    std::optional<TraceRecord> held_record_;

    uint32_t in_flight_ = 0;
    uint64_t next_id_ = 0;
    uint64_t total_issued_ = 0;
    uint64_t total_completed_ = 0;
    bool trace_drained_ = false;
};

}  // namespace obelisk
