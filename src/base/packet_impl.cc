#include <sstream>
#include <stdexcept>

#include "obelisk/packet.hh"

namespace obelisk {

const char* to_string(PacketType t) {
    switch (t) {
        case PacketType::READ: return "READ";
        case PacketType::WRITE: return "WRITE";
        case PacketType::READ_KV: return "READ_KV";
        case PacketType::READ_NON_TEMPORAL: return "READ_NT";
        case PacketType::PREFETCH: return "PREFETCH";
        case PacketType::GC_READ: return "GC_READ";
        case PacketType::GC_WRITE: return "GC_WRITE";
    }
    return "UNKNOWN";
}

PacketType parse_packet_type(const std::string& s) {
    if (s == "R" || s == "READ") return PacketType::READ;
    if (s == "W" || s == "WRITE") return PacketType::WRITE;
    if (s == "RKV" || s == "READ_KV") return PacketType::READ_KV;
    if (s == "RNT" || s == "READ_NT" || s == "READ_NON_TEMPORAL") return PacketType::READ_NON_TEMPORAL;
    if (s == "PREFETCH") return PacketType::PREFETCH;
    throw std::runtime_error("unknown packet type: " + s);
}

const char* to_string(MediaType m) {
    switch (m) {
        case MediaType::HBM: return "HBM";
        case MediaType::HBF: return "HBF";
    }
    return "UNKNOWN";
}

std::string Packet::csv_header() {
    return "id,type,target_media,addr,size,layer_id,op_name,"
           "t_send,t_arrive,"
           "host_bus_queuing,host_bus_time,"
           "controller_queuing,controller_time,"
           "logic_die_queuing,logic_die_time,"
           "subarray_queuing,subarray_time,"
           "bank_queuing,bank_time,"
           "tsv_time,total_time,"
           "stack_id,channel_id,die_id,subarray_id,bank_id,page_id";
}

std::string Packet::to_csv_row() const {
    std::ostringstream os;
    os << id << "," << to_string(type) << "," << to_string(target_media) << "," << addr << "," << size << ","
       << layer_id << "," << op_name << "," << timing.t_send << "," << timing.t_arrive << ","
       << timing.host_bus_queuing << "," << timing.host_bus_time << "," << timing.controller_queuing << ","
       << timing.controller_time << "," << timing.logic_die_queuing << "," << timing.logic_die_time << ","
       << timing.subarray_queuing << "," << timing.subarray_time << "," << timing.bank_queuing << ","
       << timing.bank_time << "," << timing.tsv_time << "," << timing.total_time() << "," << phys.stack_id << ","
       << phys.channel_id << "," << phys.die_id << "," << phys.subarray_id << "," << phys.bank_id << ","
       << phys.page_id;
    return os.str();
}

}  // namespace obelisk
