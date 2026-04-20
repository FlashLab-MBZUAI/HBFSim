#include "src/memory_system/memory_system.hh"

#include <spdlog/spdlog.h>

namespace obelisk {

MemorySystem::MemorySystem(std::unique_ptr<AddressMapper> mapper, IModule* hbm_downstream, IModule* hbf_downstream,
                           std::string name)
    : name_(std::move(name)), mapper_(std::move(mapper)), hbm_(hbm_downstream), hbf_(hbf_downstream) {}

bool MemorySystem::accept(std::unique_ptr<Packet>& pkt) {
    mapper_->map(*pkt);
    IModule* target = (pkt->target_media == MediaType::HBM) ? hbm_ : hbf_;
    if (!target) {
        spdlog::error("MemorySystem: no downstream for media={}", to_string(pkt->target_media));
        return false;
    }
    return target->accept(pkt);
}

void MemorySystem::tick(Cycle current_cycle) {
    if (hbm_) {
        while (auto maybe = hbm_->dequeue()) completed_.push(std::move(*maybe));
    }
    if (hbf_) {
        while (auto maybe = hbf_->dequeue()) completed_.push(std::move(*maybe));
    }
}

std::optional<std::unique_ptr<Packet>> MemorySystem::dequeue() {
    if (completed_.empty()) return std::nullopt;
    auto pkt = std::move(completed_.front());
    completed_.pop();
    return pkt;
}

bool MemorySystem::has_pending() const { return !completed_.empty(); }

}  // namespace obelisk
