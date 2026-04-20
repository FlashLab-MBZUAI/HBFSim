#pragma once

#include <cstdint>
#include <string>

#include "obelisk/types.hh"

namespace obelisk {

enum class PacketType : uint8_t {
    READ,
    WRITE,
    READ_KV,
    READ_NON_TEMPORAL,
    PREFETCH,
    GC_READ,
    GC_WRITE,
};

const char* to_string(PacketType t);
PacketType parse_packet_type(const std::string& s);

enum class MediaType : uint8_t {
    HBM,
    HBF,
};

const char* to_string(MediaType m);

struct PhysicalLocation {
    uint32_t stack_id = 0;
    uint32_t channel_id = 0;
    uint32_t die_id = 0;
    uint32_t subarray_id = 0;  // HBF
    uint32_t bank_id = 0;      // HBM
    uint32_t row = 0;
    uint32_t column = 0;
    uint64_t page_id = 0;
};

struct TimingStats {
    Cycle t_send = 0;
    Cycle t_arrive = 0;

    Cycle host_bus_queuing = 0;
    Cycle host_bus_time = 0;

    Cycle controller_queuing = 0;
    Cycle controller_time = 0;

    Cycle logic_die_queuing = 0;
    Cycle logic_die_time = 0;

    Cycle subarray_queuing = 0;  // HBF
    Cycle subarray_time = 0;     // HBF

    Cycle bank_queuing = 0;  // HBM
    Cycle bank_time = 0;     // HBM

    Cycle tsv_time = 0;

    Cycle total_time() const { return t_arrive >= t_send ? t_arrive - t_send : 0; }
};

struct Packet {
    PacketId id = kInvalidPacketId;
    PacketType type = PacketType::READ;
    Addr addr = 0;
    uint32_t size = 64;
    MediaType target_media = MediaType::HBF;

    PhysicalLocation phys;
    TimingStats timing;

    uint32_t workload_tag = 0;
    uint32_t layer_id = 0;
    std::string op_name;

    bool is_read() const {
        return type == PacketType::READ || type == PacketType::READ_KV ||
               type == PacketType::READ_NON_TEMPORAL || type == PacketType::PREFETCH ||
               type == PacketType::GC_READ;
    }
    bool is_write() const { return type == PacketType::WRITE || type == PacketType::GC_WRITE; }

    std::string to_csv_row() const;
    static std::string csv_header();
};

}  // namespace obelisk
