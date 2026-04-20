#include "src/base/stats.hh"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace obelisk {

static double bytes_to_gbps(uint64_t bytes, uint64_t cycles, double freq_mhz) {
    if (cycles == 0) return 0.0;
    // sim_time_seconds = cycles / (freq_mhz * 1e6)
    double sim_time_s = static_cast<double>(cycles) / (freq_mhz * 1e6);
    if (sim_time_s <= 0.0) return 0.0;
    double gb = static_cast<double>(bytes) / 1e9;
    return gb / sim_time_s;
}

double AggregateStats::hbm_read_gbps() const { return bytes_to_gbps(hbm_read_bytes, total_cycles, freq_mhz); }
double AggregateStats::hbm_write_gbps() const { return bytes_to_gbps(hbm_write_bytes, total_cycles, freq_mhz); }
double AggregateStats::hbf_read_gbps() const { return bytes_to_gbps(hbf_read_bytes, total_cycles, freq_mhz); }
double AggregateStats::hbf_write_gbps() const { return bytes_to_gbps(hbf_write_bytes, total_cycles, freq_mhz); }
double AggregateStats::aggregate_gbps() const {
    return hbm_read_gbps() + hbm_write_gbps() + hbf_read_gbps() + hbf_write_gbps();
}

void Stats::record_completion(const Packet& pkt) {
    latencies_.push_back(pkt.timing.total_time());
    const uint64_t bytes = pkt.size;
    if (pkt.target_media == MediaType::HBM) {
        if (pkt.is_read())
            hbm_read_bytes_ += bytes;
        else if (pkt.is_write())
            hbm_write_bytes_ += bytes;
    } else {
        if (pkt.is_read())
            hbf_read_bytes_ += bytes;
        else if (pkt.is_write())
            hbf_write_bytes_ += bytes;
    }
}

static uint64_t percentile(std::vector<uint64_t>& v, double p) {
    if (v.empty()) return 0;
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

AggregateStats Stats::compute() const {
    AggregateStats s;
    s.total_cycles = total_cycles_;
    s.total_requests = total_issued_;
    s.completed_requests = latencies_.size();
    s.hbm_read_bytes = hbm_read_bytes_;
    s.hbm_write_bytes = hbm_write_bytes_;
    s.hbf_read_bytes = hbf_read_bytes_;
    s.hbf_write_bytes = hbf_write_bytes_;
    s.freq_mhz = freq_mhz_;

    if (!latencies_.empty()) {
        double sum = 0.0;
        uint64_t mx = 0;
        for (uint64_t l : latencies_) {
            sum += static_cast<double>(l);
            mx = std::max(mx, l);
        }
        s.mean_latency_cycles = sum / static_cast<double>(latencies_.size());
        s.max_latency = mx;
        auto copy = latencies_;
        s.p50_latency = percentile(copy, 0.50);
        s.p95_latency = percentile(copy, 0.95);
        s.p99_latency = percentile(copy, 0.99);
    }
    return s;
}

void Stats::dump_json(const std::string& path) const {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("stats: cannot open " + path);
    auto s = compute();
    double sim_time_s = s.total_cycles == 0 ? 0.0 : static_cast<double>(s.total_cycles) / (freq_mhz_ * 1e6);
    out << "{\n";
    out << "  \"simulation\": {\n";
    out << "    \"total_cycles\": " << s.total_cycles << ",\n";
    out << "    \"total_requests\": " << s.total_requests << ",\n";
    out << "    \"completed_requests\": " << s.completed_requests << ",\n";
    out << "    \"sim_time_seconds\": " << sim_time_s << "\n";
    out << "  },\n";
    out << "  \"bandwidth\": {\n";
    out << "    \"hbm_read_gbps\": " << s.hbm_read_gbps() << ",\n";
    out << "    \"hbm_write_gbps\": " << s.hbm_write_gbps() << ",\n";
    out << "    \"hbf_read_gbps\": " << s.hbf_read_gbps() << ",\n";
    out << "    \"hbf_write_gbps\": " << s.hbf_write_gbps() << ",\n";
    out << "    \"aggregate_gbps\": " << s.aggregate_gbps() << "\n";
    out << "  },\n";
    out << "  \"latency\": {\n";
    out << "    \"mean_cycles\": " << s.mean_latency_cycles << ",\n";
    out << "    \"p50_cycles\": " << s.p50_latency << ",\n";
    out << "    \"p95_cycles\": " << s.p95_latency << ",\n";
    out << "    \"p99_cycles\": " << s.p99_latency << ",\n";
    out << "    \"max_cycles\": " << s.max_latency << "\n";
    out << "  }\n";
    out << "}\n";
}

}  // namespace obelisk
