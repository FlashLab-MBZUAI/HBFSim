#include "src/controller/scheduler.hh"

namespace obelisk {

namespace {

// FR-FCFS: First-Ready, First-Come-First-Served. "Ready" here is interpreted
// as the first packet whose target row already has an activated page in
// recently-accessed memory. Since the controller doesn't directly track
// per-subarray row state, we approximate by: scan queue for a packet whose
// phys.page_id matches the most recently issued packet's page_id.
class SchedulerFRFCFS : public IScheduler {
public:
    std::unique_ptr<Packet> select_next(SchedulerContext ctx) override {
        // Look in read queue first.
        auto pick = [&](std::deque<std::unique_ptr<Packet>>& q) -> std::unique_ptr<Packet> {
            if (q.empty()) return nullptr;
            // Prefer row hit against last_page_id_.
            for (auto it = q.begin(); it != q.end(); ++it) {
                if ((*it)->phys.page_id == last_page_id_ && last_page_id_ != 0) {
                    auto pkt = std::move(*it);
                    q.erase(it);
                    last_page_id_ = pkt->phys.page_id;
                    return pkt;
                }
            }
            // Otherwise FCFS head.
            auto pkt = std::move(q.front());
            q.pop_front();
            last_page_id_ = pkt->phys.page_id;
            return pkt;
        };

        if (ctx.read_priority) {
            if (auto p = pick(ctx.read_queue)) return p;
            if (auto p = pick(ctx.write_queue)) return p;
        } else {
            if (auto p = pick(ctx.read_queue)) return p;
            if (auto p = pick(ctx.write_queue)) return p;
        }
        return nullptr;
    }

    const std::string& name() const override {
        static const std::string n = "FRFCFS";
        return n;
    }

private:
    uint64_t last_page_id_ = 0;
};

}  // namespace

std::unique_ptr<IScheduler> make_scheduler_frfcfs() { return std::make_unique<SchedulerFRFCFS>(); }

}  // namespace obelisk
