#include "src/controller/scheduler.hh"

namespace obelisk {

namespace {

// HBF-aware scheduler (M1 placeholder, paper contribution is M2+).
// Strategy:
//   1. Skip packets whose target subarray is busy (per subarray_busy vector).
//   2. Prefer KV reads (READ_KV) over plain READ (they are latency-sensitive).
//   3. Fall back to FCFS if no packet can dispatch.
class SchedulerHBFAware : public IScheduler {
public:
    std::unique_ptr<Packet> select_next(SchedulerContext ctx) override {
        auto is_busy = [&](const Packet& p) -> bool {
            if (!ctx.subarray_busy || ctx.subarray_busy->empty()) return false;
            size_t idx = p.phys.die_id * 1024 + p.phys.subarray_id;  // loose indexing
            if (idx >= ctx.subarray_busy->size()) return false;
            return (*ctx.subarray_busy)[idx];
        };

        auto pick_first_free = [&](std::deque<std::unique_ptr<Packet>>& q,
                                    PacketType preferred) -> std::unique_ptr<Packet> {
            if (q.empty()) return nullptr;
            // Prefer preferred type.
            for (auto it = q.begin(); it != q.end(); ++it) {
                if ((*it)->type == preferred && !is_busy(**it)) {
                    auto pkt = std::move(*it);
                    q.erase(it);
                    return pkt;
                }
            }
            // Fall back: any non-busy.
            for (auto it = q.begin(); it != q.end(); ++it) {
                if (!is_busy(**it)) {
                    auto pkt = std::move(*it);
                    q.erase(it);
                    return pkt;
                }
            }
            return nullptr;
        };

        if (ctx.read_priority) {
            if (auto p = pick_first_free(ctx.read_queue, PacketType::READ_KV)) return p;
            if (auto p = pick_first_free(ctx.write_queue, PacketType::WRITE)) return p;
        } else {
            if (auto p = pick_first_free(ctx.write_queue, PacketType::WRITE)) return p;
            if (auto p = pick_first_free(ctx.read_queue, PacketType::READ_KV)) return p;
        }
        return nullptr;
    }

    const std::string& name() const override {
        static const std::string n = "HBFAware";
        return n;
    }
};

}  // namespace

std::unique_ptr<IScheduler> make_scheduler_hbf_aware() { return std::make_unique<SchedulerHBFAware>(); }

}  // namespace obelisk
