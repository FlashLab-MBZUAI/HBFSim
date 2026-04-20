#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <queue>
#include <string>

#include "obelisk/imodule.hh"
#include "src/controller/scheduler.hh"

namespace obelisk {

struct HBFControllerConfig {
    uint32_t read_queue_depth = 64;
    uint32_t write_queue_depth = 32;
    std::string scheduler_type = "frfcfs";  // fcfs | frfcfs | hbf_aware
    uint32_t controller_latency_cycles = 10;
    bool enable_read_priority = true;
    uint32_t max_outstanding_per_subarray = 4;
    std::string name = "HBFController";
};

class HBFController : public IModule {
public:
    explicit HBFController(const HBFControllerConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(Cycle current_cycle) override;
    bool accept(std::unique_ptr<Packet>& pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    void connect_downstream(IModule* m) { downstream_ = m; }

private:
    struct PendingReturn {
        std::unique_ptr<Packet> pkt;
        Cycle ready_cycle = 0;
    };

    struct QueuedForward {
        std::unique_ptr<Packet> pkt;
        Cycle entry_cycle = 0;
    };

    std::string name_;
    HBFControllerConfig cfg_;
    std::unique_ptr<IScheduler> scheduler_;

    std::deque<std::unique_ptr<Packet>> read_queue_;
    std::deque<std::unique_ptr<Packet>> write_queue_;

    // Entry times (parallel to queues, indexed by packet id).
    std::unordered_map<PacketId, Cycle> entry_cycles_;

    std::queue<PendingReturn> return_queue_;

    IModule* downstream_ = nullptr;
};

}  // namespace obelisk
