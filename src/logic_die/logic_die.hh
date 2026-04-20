#pragma once

#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "obelisk/imodule.hh"
#include "src/logic_die/channel_mux.hh"
#include "src/logic_die/subarray_dispatcher.hh"
#include "src/logic_die/tsv_model.hh"

namespace obelisk {

struct LogicDieConfig {
    uint32_t num_channels = 8;
    uint32_t num_dies_per_stack = 16;
    uint32_t num_subarrays_per_die = 32;
    uint32_t tsv_bandwidth_gbps = 200;
    uint32_t logic_die_freq_mhz = 1000;
    uint32_t command_translation_cycles = 2;
    uint32_t max_concurrent_subarray_cmds = 64;
    std::string name = "LogicDie";
};

// Logic die sits between the HBF controller and the NAND die stack. Models:
//  - Per-channel command mux (serialize commands by channel)
//  - Admission control on the sub-array dispatcher
//  - TSV bandwidth pipe for the data burst on the return path
class LogicDie : public IModule {
public:
    explicit LogicDie(const LogicDieConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(Cycle current_cycle) override;
    bool accept(std::unique_ptr<Packet>& pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    // Connect a NAND die by id (downstream).
    void connect_nand_die(uint32_t die_id, IModule* die);

private:
    struct Returning {
        std::unique_ptr<Packet> pkt;
        Cycle ready_cycle = 0;
    };

    std::string name_;
    LogicDieConfig cfg_;
    std::unique_ptr<ChannelMux> channel_mux_;
    std::unique_ptr<SubarrayDispatcher> dispatcher_;
    std::unique_ptr<TSVModel> tsv_;
    std::vector<IModule*> nand_dies_;

    // Packets awaiting forward dispatch — in-flight at logic die cmd stage.
    struct ForwardWait {
        std::unique_ptr<Packet> pkt;
        Cycle ready_cycle = 0;  // when cmd translation / channel grant finishes
        Cycle entry_cycle = 0;
    };
    std::queue<ForwardWait> forward_queue_;

    // Completed from NAND + applied TSV latency, waiting for upstream dequeue.
    std::queue<Returning> return_queue_;
};

}  // namespace obelisk
