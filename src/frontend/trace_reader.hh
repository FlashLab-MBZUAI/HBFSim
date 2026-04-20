#pragma once

#include <fstream>
#include <optional>
#include <string>

#include "obelisk/packet.hh"

namespace obelisk {

// Ramulator-compatible trace format:
//   <timestamp> <type> <addr> <size> [<layer_id>] [<op_name>]
// timestamp: absolute cycle, or -1 for back-to-back (issue ASAP).
// type: R, W, RKV, RNT.
struct TraceRecord {
    int64_t timestamp = -1;
    PacketType type = PacketType::READ;
    Addr addr = 0;
    uint32_t size = 64;
    uint32_t layer_id = 0;
    std::string op_name;
};

class TraceReader {
public:
    explicit TraceReader(const std::string& path);

    // Returns next record, or nullopt at EOF.
    std::optional<TraceRecord> next();

    bool eof() const { return eof_; }
    uint64_t lines_read() const { return lines_read_; }

private:
    std::ifstream in_;
    bool eof_ = false;
    uint64_t lines_read_ = 0;
    std::string path_;
};

}  // namespace obelisk
