#pragma once

#include <memory>
#include <optional>
#include <queue>
#include <string>

#include "obelisk/imodule.hh"

namespace obelisk {

struct HostBusConfig {
    double bandwidth_gbps = 128.0;      // PCIe 6.0 x16
    uint32_t base_latency_cycles = 50;  // bus base latency
    uint32_t queue_depth = 32;
    uint32_t cpu_compute_cycles = 100;  // coarse CPU model
    uint32_t freq_mhz = 1000;           // needed for byte<->cycle conversion
};

// Models a host-side bus (PCIe/CXL). Serializes packet transfers by bandwidth,
// adds base latency and CPU compute time. Supports forward + return paths.
class HostBus : public IModule {
public:
    explicit HostBus(const HostBusConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(Cycle current_cycle) override;
    bool accept(std::unique_ptr<Packet>& pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    void connect_downstream(IModule* m) { downstream_ = m; }

private:
    struct InFlight {
        std::unique_ptr<Packet> pkt;
        Cycle ready_cycle = 0;
    };

    std::string name_ = "HostBus";
    HostBusConfig cfg_;

    // Forward direction (upstream → downstream).
    std::queue<InFlight> forward_queue_;
    Cycle forward_bus_busy_until_ = 0;

    // Return direction (downstream → upstream via dequeue).
    std::queue<InFlight> return_queue_;
    Cycle return_bus_busy_until_ = 0;

    IModule* downstream_ = nullptr;

    Cycle compute_transfer_cycles(uint32_t size) const;
    Cycle forward_service_cycles(uint32_t size) const;
    Cycle return_service_cycles(uint32_t size) const;
};

}  // namespace obelisk
