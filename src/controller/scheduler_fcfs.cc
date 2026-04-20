#include "src/controller/scheduler.hh"

namespace obelisk {

namespace {

class SchedulerFCFS : public IScheduler {
public:
    std::unique_ptr<Packet> select_next(SchedulerContext ctx) override {
        if (ctx.read_priority && !ctx.read_queue.empty()) {
            auto pkt = std::move(ctx.read_queue.front());
            ctx.read_queue.pop_front();
            return pkt;
        }
        if (!ctx.read_queue.empty()) {
            auto pkt = std::move(ctx.read_queue.front());
            ctx.read_queue.pop_front();
            return pkt;
        }
        if (!ctx.write_queue.empty()) {
            auto pkt = std::move(ctx.write_queue.front());
            ctx.write_queue.pop_front();
            return pkt;
        }
        return nullptr;
    }

    const std::string& name() const override {
        static const std::string n = "FCFS";
        return n;
    }
};

}  // namespace

std::unique_ptr<IScheduler> make_scheduler_fcfs() { return std::make_unique<SchedulerFCFS>(); }

}  // namespace obelisk
