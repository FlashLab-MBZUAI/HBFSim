#pragma once

#include <memory>
#include <optional>
#include <queue>
#include <string>

#include "obelisk/imodule.hh"
#include "src/memory_system/address_mapper.hh"

namespace obelisk {

// Orchestrates HBM vs HBF routing. Sits between the HostBus and the two
// media subsystems. Fills in physical location via AddressMapper, routes
// to the appropriate downstream, and merges completions.
class MemorySystem : public IModule {
public:
    MemorySystem(std::unique_ptr<AddressMapper> mapper, IModule* hbm_downstream, IModule* hbf_downstream,
                 std::string name = "MemorySystem");

    const std::string& name() const override { return name_; }
    void tick(Cycle current_cycle) override;
    bool accept(std::unique_ptr<Packet>& pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

private:
    std::string name_;
    std::unique_ptr<AddressMapper> mapper_;
    IModule* hbm_ = nullptr;
    IModule* hbf_ = nullptr;
    std::queue<std::unique_ptr<Packet>> completed_;
};

}  // namespace obelisk
