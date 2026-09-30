#pragma once
#include "physical/hbf/hbf_config.hpp"
#include "physical/hbf/hbf_standard.hpp"
#include "physical/resource_calendar.hpp"
#include <map>
#include <optional>
#include <set>
#include <unordered_map>

namespace hbfsim::physical::hbf {
// Device-owned state: channel-local -> NAND zone mapping, physical PEC,
// transport, NAND program order and two decoded pages per bank. Application
// LPN mapping, data liveness, allocation and GC policy remain with the host.
class HbfDevice {
public:
    explicit HbfDevice(HbfDeviceConfig config);
    struct Transfer { double start_ns, finish_ns, wait_ns; };
    struct Channel {
        ResourceTimeline command, rx, tx;
        double command_work_ns = 0, rx_work_ns = 0, tx_work_ns = 0;
    };
    enum class Direction { Command, HostToDevice, DeviceToHost };
    [[nodiscard]] Transfer transfer(std::size_t channel, Direction direction,
        std::uint64_t bytes, double arrival_ns, double causal_watermark_ns);
    [[nodiscard]] std::size_t channel_for_page(std::uint64_t ppn) const;
    [[nodiscard]] std::size_t bank_for_page(std::uint64_t ppn) const;
    // Untimed image installation is explicit and forbidden after commands.
    void seed_block(std::uint64_t block, std::uint32_t pages, std::uint32_t pec = 0);
    // Untimed image precondition: a never-programmed block already sits in
    // the erased state (erased during installation, like a controller's
    // pre-erased free pool), so its first page-zero program needs no erase.
    // Any program consumes the state; an explicit erase() never sets it.
    void seed_erased_block(std::uint64_t block);
    [[nodiscard]] std::uint64_t preconditioned_erased_blocks() const { return preconditioned_erased_; }
    [[nodiscard]] std::uint64_t unconsumed_erased_blocks() const { return unconsumed_erased_; }
    // OCP v0.7.0 sections 4.6, 5.2.4.6 and 11.4.1. This model advertises
    // host-controlled WL only. Configuration selects BUCC.WLS=10b; it does
    // not implement the optional product-specific base-die WL algorithm.
    void configure_host_zones(std::uint32_t blocks_per_zone);
    struct ZoneRemapCommand {
        std::uint8_t opcode = 0x8;
        std::uint8_t remap_type = 0x00; // Swap
        std::uint32_t stack = 0, channel = 0;
        std::uint64_t first = 0, second = 0;
    };
    void execute_zone_remap(const ZoneRemapCommand& command);
    [[nodiscard]] std::uint64_t physical_zone(std::uint64_t channel_zone) const;
    [[nodiscard]] std::uint64_t zone_pec_sum(std::uint64_t channel_zone) const;
    [[nodiscard]] std::uint32_t block_pec(std::uint64_t block) const;
    // Called once for each scheduled physical erase, including page zero.
    std::uint32_t record_erase(std::uint64_t block);
    [[nodiscard]] const std::map<std::uint64_t, std::uint64_t>& zone_mapping() const { return zone_mapping_; }
    void validate_zone_mapping(const std::map<std::uint64_t, std::uint64_t>& mapping) const;
    void restore_zone_mapping(const std::map<std::uint64_t, std::uint64_t>& mapping);
    // Device-reception events, in time order. A successful receipt is NOT a
    // command completion. The caller schedules NAND only when page_ready is
    // true and supplies its finish through complete_program().
    struct WriteAdmission {
        std::uint8_t status = 0, additional_status = 0;
        bool page_ready = false;
    };
    struct WriteResponse {
        std::uint64_t command;
        std::uint8_t status, additional_status;
        double finish_ns;
    };
    [[nodiscard]] WriteAdmission receive_write_fragment(std::uint64_t ppn,
        std::uint32_t offset, std::uint32_t bytes, std::uint64_t command, double at_ns);
    [[nodiscard]] std::vector<WriteResponse> advance_writes(double at_ns);
    [[nodiscard]] double next_write_event_ns() const;
    [[nodiscard]] std::size_t pending_write_pages() const { return writes_.size(); }
    // nullopt: read NAND; 0: forward received bytes; 0xA: missing bytes.
    // This timing model tracks coverage rather than payload values.
    [[nodiscard]] std::optional<std::uint8_t> buffered_write_read_status(
        std::uint64_t ppn, std::uint32_t offset, std::uint32_t bytes) const;
    // Returns whether this command must perform the page-zero auto erase:
    // every page-zero program erases its block unless it is pre-erased.
    [[nodiscard]] bool prepare_program(std::uint64_t ppn);
    void complete_program(std::uint64_t ppn, double ready_ns);
    [[nodiscard]] double read_ready(std::uint64_t ppn) const;
    void erase(std::uint64_t block);
    [[nodiscard]] bool cache_hit(std::uint64_t ppn, double at_ns);
    void cache_fill(std::uint64_t ppn, double ready_ns);
    void advance_cache(double causal_watermark_ns);
    void cache_purge_page(std::uint64_t ppn);
    void cache_purge_block(std::uint64_t block);

    [[nodiscard]] const std::vector<Channel>& channels() const { return channels_; }
    [[nodiscard]] const HbfDeviceConfig& config() const { return config_; }
private:
    void validate_page(std::uint64_t ppn) const;
    void validate_block(std::uint64_t block) const;
    struct Block { std::uint32_t next_page = 0; };
    struct CacheEvent { std::uint64_t page; int action; /* -1 invalidate, 0 touch, 1 fill */ };
    struct BankCache {
        std::set<std::uint64_t> candidates;
        std::size_t future_consumers = 0;
        std::vector<std::uint64_t> resident; // LRU -> MRU at the causal frontier
        std::map<std::pair<double, std::uint64_t>, CacheEvent> events;
        bool pending = false;
    };
    void track_cache_events(std::size_t index, BankCache& bank);
    static bool apply_cache_event(std::vector<std::uint64_t>& resident, CacheEvent event);
    [[nodiscard]] static std::vector<std::uint64_t> project_cache(const BankCache& bank,
        double at_ns, bool* valid = nullptr);
    HbfDeviceConfig config_;
    std::uint64_t total_pages_ = 0;
    std::uint32_t zone_size_blocks_ = 0;
    std::uint8_t wear_leveling_selection_ = 0;
    std::map<std::uint64_t, std::uint64_t> zone_mapping_;
    std::vector<std::uint32_t> physical_pec_;
    std::vector<Channel> channels_;
    std::unordered_map<std::uint64_t, Block> blocks_;
    // Per-block pre-erased state (empty unless an image seeded one).
    std::vector<std::uint8_t> erased_;
    std::uint64_t preconditioned_erased_ = 0, unconsumed_erased_ = 0;
    std::unordered_map<std::uint64_t, double> program_ready_;
    std::multimap<double, std::uint64_t> program_completions_;
    struct PartialWrite {
        std::uint64_t coverage = 0;
        double event_ns = 0;
        bool programming = false;
        std::vector<std::uint64_t> commands;
    };
    std::unordered_map<std::uint64_t, PartialWrite> writes_;
    std::map<std::pair<double, std::uint64_t>, bool> write_events_;
    std::vector<std::uint32_t> outstanding_writes_;
    double write_clock_ns_ = 0;
    // Temporal records retain future consumers until the causal watermark
    // passes. Logical residency at each timestamp is exactly two per bank.
    std::unordered_map<std::size_t, BankCache> cache_;
    // Banks without temporal events need no work when the frontier advances.
    // Each pending bank appears once, regardless of its event count.
    std::vector<std::size_t> pending_cache_banks_;
    std::uint64_t touch_sequence_ = 0;
    bool started_ = false;
};
}
