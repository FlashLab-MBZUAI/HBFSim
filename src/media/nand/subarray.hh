#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "obelisk/imodule.hh"

namespace obelisk {

struct SubarrayConfig {
    uint32_t page_size_bytes = 16384;
    uint32_t pages_per_block = 512;
    uint32_t blocks_per_subarray = 1024;
    uint32_t tR_cycles = 50'000;       // NAND read latency  (~50us @ 1GHz)
    uint32_t tPROG_cycles = 500'000;   // NAND program       (~500us)
    uint32_t tBERS_cycles = 3'000'000; // NAND block erase   (~3ms)
    uint32_t cache_buffer_pages = 4;
    uint32_t cache_hit_read_cycles = 500;  // when page is in buffer
    uint32_t queue_depth = 4;
};

// One independent NAND sub-array. Serializes one command at a time. Holds a
// small page buffer that reduces read latency on hit.
class Subarray : public IModule {
public:
    Subarray(uint32_t id, const SubarrayConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(Cycle current_cycle) override;
    bool accept(std::unique_ptr<Packet>& pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    uint32_t id() const { return id_; }
    bool is_busy() const { return busy_; }
    uint64_t total_read_hits() const { return buf_hits_; }
    uint64_t total_read_misses() const { return buf_misses_; }

private:
    std::string name_;
    uint32_t id_;
    SubarrayConfig cfg_;

    struct Queued {
        std::unique_ptr<Packet> pkt;
        Cycle enqueue_cycle = 0;
    };
    std::deque<Queued> queue_;

    bool busy_ = false;
    Cycle busy_until_ = 0;
    std::unique_ptr<Packet> current_;
    Cycle current_start_cycle_ = 0;

    std::queue<std::unique_ptr<Packet>> completed_;

    // Recent page ids (LRU).
    std::vector<uint64_t> cached_pages_;
    uint64_t buf_hits_ = 0;
    uint64_t buf_misses_ = 0;

    Cycle latency_for(const Packet& pkt) const;
    bool buffer_hit(uint64_t page_id);
    void touch_buffer(uint64_t page_id);
};

}  // namespace obelisk
