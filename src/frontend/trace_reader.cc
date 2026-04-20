#include "src/frontend/trace_reader.hh"

#include <sstream>
#include <stdexcept>

namespace obelisk {

TraceReader::TraceReader(const std::string& path) : path_(path) {
    in_.open(path);
    if (!in_) throw std::runtime_error("trace_reader: cannot open " + path);
}

std::optional<TraceRecord> TraceReader::next() {
    std::string line;
    while (std::getline(in_, line)) {
        ++lines_read_;
        // Strip leading whitespace + skip blank/comment.
        size_t i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i == line.size()) continue;
        if (line[i] == '#') continue;

        std::istringstream ss(line.substr(i));
        TraceRecord r;
        std::string type_str, addr_str;
        if (!(ss >> r.timestamp >> type_str >> addr_str >> r.size)) {
            throw std::runtime_error("trace_reader: malformed line in " + path_ + ": " + line);
        }
        r.type = parse_packet_type(type_str);

        // Address: hex (0x...) or decimal.
        if (addr_str.size() > 2 && (addr_str.substr(0, 2) == "0x" || addr_str.substr(0, 2) == "0X")) {
            r.addr = std::stoull(addr_str.substr(2), nullptr, 16);
        } else {
            r.addr = std::stoull(addr_str, nullptr, 0);
        }

        // Optional layer_id and op_name.
        std::string rest;
        if (ss >> r.layer_id) {
            std::string op;
            if (ss >> op) r.op_name = op;
        }
        return r;
    }
    eof_ = true;
    return std::nullopt;
}

}  // namespace obelisk
