#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "obelisk/packet.hh"

namespace obelisk {

struct AggregateStats {
    uint64_t total_cycles = 0;
    uint64_t total_requests = 0;
    uint64_t completed_requests = 0;

    uint64_t hbm_read_bytes = 0;
    uint64_t hbm_write_bytes = 0;
    uint64_t hbf_read_bytes = 0;
    uint64_t hbf_write_bytes = 0;

    double freq_mhz = 1000.0;

    double hbm_read_gbps() const;
    double hbm_write_gbps() const;
    double hbf_read_gbps() const;
    double hbf_write_gbps() const;
    double aggregate_gbps() const;

    double mean_latency_cycles = 0;
    uint64_t p50_latency = 0;
    uint64_t p95_latency = 0;
    uint64_t p99_latency = 0;
    uint64_t max_latency = 0;
};

class Stats {
public:
    Stats() = default;

    void set_freq_mhz(double f) { freq_mhz_ = f; }

    void record_completion(const Packet& pkt);
    void record_issue() { ++total_issued_; }
    void set_total_cycles(uint64_t c) { total_cycles_ = c; }
    void set_total_issued(uint64_t n) { total_issued_ = n; }

    AggregateStats compute() const;

    // Write JSON summary.
    void dump_json(const std::string& path) const;

private:
    double freq_mhz_ = 1000.0;
    uint64_t total_cycles_ = 0;
    uint64_t total_issued_ = 0;

    uint64_t hbm_read_bytes_ = 0;
    uint64_t hbm_write_bytes_ = 0;
    uint64_t hbf_read_bytes_ = 0;
    uint64_t hbf_write_bytes_ = 0;

    std::vector<uint64_t> latencies_;
};

}  // namespace obelisk
