#pragma once

#include <memory>
#include <optional>
#include <string>

#include "obelisk/packet.hh"

namespace obelisk {

class Simulation;

class IModule {
public:
    virtual ~IModule() = default;

    virtual const std::string& name() const = 0;

    virtual void tick(Cycle current_cycle) = 0;

    // Non-blocking accept. Takes the packet by reference so the callee can
    // move out of `pkt` only on success. If accept returns false, `pkt` is
    // left untouched and the caller may retry next tick.
    virtual bool accept(std::unique_ptr<Packet>& pkt) = 0;

    // Dequeue a completed packet (for modules that return responses upstream).
    virtual std::optional<std::unique_ptr<Packet>> dequeue() { return std::nullopt; }

    virtual bool has_pending() const = 0;

    virtual void set_simulation(Simulation* sim) { sim_ = sim; }

protected:
    Simulation* sim_ = nullptr;
};

}  // namespace obelisk
