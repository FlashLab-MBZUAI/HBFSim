#pragma once

#include <string>

#include "src/base/simulation.hh"
#include "src/controller/hbf_controller.hh"
#include "src/frontend/requester.hh"
#include "src/host/host_bus.hh"
#include "src/logic_die/logic_die.hh"
#include "src/media/hbm/hbm_interface.hh"
#include "src/media/nand/nand_stack.hh"
#include "src/memory_system/address_mapper.hh"

namespace obelisk {

struct ObeliskConfig {
    SimulationConfig simulation;
    RequesterConfig requester;
    HostBusConfig host_bus;
    AddressMapperConfig address_mapper;
    HBFControllerConfig hbf_controller;
    LogicDieConfig logic_die;
    NANDStackConfig nand_stack;
    HBMConfig hbm;
    bool has_hbm = true;
    bool has_hbf = true;
};

ObeliskConfig load_config(const std::string& path);

// Parse strings like "96 GB", "512 MB", "1024", "0x1800000000" into uint64.
uint64_t parse_byte_size(const std::string& s);
uint64_t parse_addr(const std::string& s);

}  // namespace obelisk
