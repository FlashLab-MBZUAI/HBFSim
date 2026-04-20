#pragma once

#include <deque>
#include <memory>
#include <string>

#include "obelisk/packet.hh"

namespace obelisk {

// Deviation from design spec Section 5.4: we give the scheduler direct refs to
// the controller's read/write deques instead of a candidate vector of
// unique_ptrs, because moving out of std::queue to build a vector then putting
// back is wasteful. See docs/decisions.md.
struct SchedulerContext {
    std::deque<std::unique_ptr<Packet>>& read_queue;
    std::deque<std::unique_ptr<Packet>>& write_queue;
    Cycle current_cycle = 0;
    bool read_priority = true;

    // Optional: for HBF-aware scheduler, busy flags for each subarray
    // keyed as (die_id * num_subarrays + subarray_id). Empty = unknown.
    const std::vector<bool>* subarray_busy = nullptr;
};

class IScheduler {
public:
    virtual ~IScheduler() = default;

    // Pick and remove one packet from the queues. Returns nullptr if no
    // candidate selected this cycle.
    virtual std::unique_ptr<Packet> select_next(SchedulerContext ctx) = 0;

    virtual const std::string& name() const = 0;
};

std::unique_ptr<IScheduler> make_scheduler(const std::string& type);

}  // namespace obelisk
