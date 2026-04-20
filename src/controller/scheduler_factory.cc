#include <stdexcept>

#include "src/controller/scheduler.hh"

namespace obelisk {

std::unique_ptr<IScheduler> make_scheduler_fcfs();
std::unique_ptr<IScheduler> make_scheduler_frfcfs();
std::unique_ptr<IScheduler> make_scheduler_hbf_aware();

std::unique_ptr<IScheduler> make_scheduler(const std::string& type) {
    if (type == "fcfs") return make_scheduler_fcfs();
    if (type == "frfcfs") return make_scheduler_frfcfs();
    if (type == "hbf_aware") return make_scheduler_hbf_aware();
    throw std::runtime_error("unknown scheduler type: " + type);
}

}  // namespace obelisk
