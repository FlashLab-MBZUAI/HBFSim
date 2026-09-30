#include "physical/hbf/hbf_device.hpp"
#include <algorithm>
#include <cmath>

namespace hbfsim::physical::hbf {
HbfDevice::HbfDevice(HbfDeviceConfig config) : config_(config) {
    const auto grade = speed_grade(config.speed_grade);
    if (!config.stacks || !config.channels_per_stack ||
        config.channels_per_stack > grade.maximum_channels ||
        !config.dies_per_channel || config.dies_per_channel > 4 ||
        config.channels_per_stack * config.dies_per_channel > grade.maximum_dies ||
        !config.planes_per_die || config.planes_per_die > 16 ||
        config.page_size_bytes != kPageBytes || !config.pages_per_block || !config.blocks_per_plane)
        throw std::invalid_argument("invalid OCP HBF channel/die/bank/page geometry");
    auto capacity = std::uint64_t{config.page_size_bytes};
    for (const auto factor : {config.stacks, config.channels_per_stack, config.dies_per_channel,
             config.planes_per_die, config.blocks_per_plane, config.pages_per_block}) {
        if (capacity > std::numeric_limits<std::uint64_t>::max() / factor)
            throw std::invalid_argument("OCP HBF geometry exceeds the address space");
        capacity *= factor;
    }
    total_pages_ = capacity / config.page_size_bytes;
    physical_pec_.resize(total_pages_ / config.pages_per_block);
    channels_.resize(static_cast<std::size_t>(config.stacks) * config.channels_per_stack);
    if (!std::isfinite(config.write_accumulation_timeout_ns) ||
        config.write_accumulation_timeout_ns <= 0 || !config.outstanding_write_pages_per_channel)
        throw std::invalid_argument("invalid HBF write accumulation timeout or page limit");
    outstanding_writes_.resize(channels_.size());
}
void HbfDevice::validate_page(std::uint64_t ppn) const {
    if (ppn >= total_pages_) throw std::out_of_range("HBF page outside device geometry");
}
void HbfDevice::validate_block(std::uint64_t block) const {
    if (block >= total_pages_ / config_.pages_per_block)
        throw std::out_of_range("HBF block outside device geometry");
}
std::size_t HbfDevice::bank_for_page(std::uint64_t ppn) const {
    validate_page(ppn);
    return ppn / config_.pages_per_block / config_.blocks_per_plane;
}
std::size_t HbfDevice::channel_for_page(std::uint64_t ppn) const {
    return bank_for_page(ppn) / config_.planes_per_die / config_.dies_per_channel;
}
HbfDevice::Transfer HbfDevice::transfer(std::size_t channel, Direction direction,
    std::uint64_t bytes, double arrival, double watermark) {
    if (!std::isfinite(arrival) || arrival < 0 || !bytes ||
        !std::isfinite(watermark) || watermark < 0 || watermark > arrival)
        throw std::invalid_argument("invalid HBF channel transfer");
    auto& c = channels_.at(channel);
    auto& r = direction == Direction::Command ? c.command :
        direction == Direction::HostToDevice ? c.rx : c.tx;
    auto& work = direction == Direction::Command ? c.command_work_ns :
        direction == Direction::HostToDevice ? c.rx_work_ns : c.tx_work_ns;
    const auto duration = bytes / speed_grade(config_.speed_grade).payload_GBps_per_channel;
    r.prune_before(watermark);
    const auto reserved = r.reserve(arrival, duration);
    work += duration; started_ = true;
    return {reserved.start_ns, reserved.finish_ns, reserved.start_ns - arrival};
}
void HbfDevice::seed_block(std::uint64_t block, std::uint32_t pages, std::uint32_t pec) {
    validate_block(block);
    if (started_ || pages > config_.pages_per_block) throw std::logic_error("invalid timed HBF image installation");
    if (pages) blocks_[block].next_page = pages;
    physical_pec_[block] = pec;
}
void HbfDevice::seed_erased_block(std::uint64_t block) {
    validate_block(block);
    const auto found = blocks_.find(block);
    if (started_ || (found != blocks_.end() && found->second.next_page))
        throw std::logic_error("only an unprogrammed block can be seeded erased before HBF commands");
    if (erased_.empty()) erased_.resize(physical_pec_.size());
    if (erased_[block]) return;
    erased_[block] = 1;
    ++preconditioned_erased_; ++unconsumed_erased_;
}
void HbfDevice::configure_host_zones(std::uint32_t blocks_per_zone) {
    const auto channel_blocks = total_pages_ / config_.pages_per_block / channels_.size();
    if (started_ || !zone_mapping_.empty() || !blocks_per_zone || channel_blocks % blocks_per_zone)
        throw std::runtime_error("HBF zone configuration requires equal block-aligned zones before IO");
    zone_size_blocks_ = blocks_per_zone;
    wear_leveling_selection_ = 0b10;
}
std::uint64_t HbfDevice::physical_zone(std::uint64_t zone) const {
    if (!zone_size_blocks_ || zone >= physical_pec_.size() / zone_size_blocks_)
        throw std::runtime_error("HBF channel-local zone outside configured geometry");
    const auto found = zone_mapping_.find(zone);
    return found == zone_mapping_.end() ? zone : found->second;
}
void HbfDevice::validate_zone_mapping(const std::map<std::uint64_t, std::uint64_t>& mapping) const {
    if (!zone_size_blocks_) throw std::runtime_error("HBF zones are not configured");
    const auto count = physical_pec_.size() / zone_size_blocks_;
    const auto per_channel = count / channels_.size();
    std::set<std::uint64_t> destinations;
    for (const auto& [local, physical] : mapping)
        if (local >= count || physical >= count || local == physical ||
            local / per_channel != physical / per_channel ||
            !mapping.contains(physical) || !destinations.insert(physical).second)
            throw std::runtime_error("HBF zone map is not a channel-local permutation");
}
void HbfDevice::restore_zone_mapping(const std::map<std::uint64_t, std::uint64_t>& mapping) {
    if (started_) throw std::runtime_error("HBF zone restore requires an unstarted device");
    validate_zone_mapping(mapping);
    zone_mapping_ = mapping;
}
void HbfDevice::execute_zone_remap(const ZoneRemapCommand& command) {
    if (wear_leveling_selection_ != 0b10 || command.opcode != 0x8 || command.remap_type != 0x00)
        throw std::runtime_error("HBF zone swap requires opcode 0x8, type 0x00 and BUCC.WLS=10b");
    const auto per_channel = physical_pec_.size() / zone_size_blocks_ / channels_.size();
    if (command.stack >= config_.stacks || command.channel >= config_.channels_per_stack ||
        command.first >= per_channel || command.second >= per_channel || command.first == command.second)
        throw std::runtime_error("HBF remap requires distinct zones in the same channel");
    const auto base = (std::uint64_t{command.stack} * config_.channels_per_stack + command.channel) * per_channel;
    const auto a = base + command.first, b = base + command.second;
    const auto pa = physical_zone(a), pb = physical_zone(b);
    const auto set = [&](auto local, auto physical) {
        if (local == physical) zone_mapping_.erase(local);
        else zone_mapping_[local] = physical;
    };
    set(a, pb); set(b, pa);
    // Only the address map changes. The host has checked invalidity and IO
    // quiescence; no NAND erase, copy, PEC reset or data publication occurs.
    started_ = true;
}
std::uint32_t HbfDevice::block_pec(std::uint64_t block) const {
    validate_block(block);
    return physical_pec_[block];
}
std::uint32_t HbfDevice::record_erase(std::uint64_t block) {
    validate_block(block);
    auto& pec = physical_pec_[block];
    if (pec == std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("HBF physical PEC overflow");
    started_ = true;
    return ++pec;
}
std::uint64_t HbfDevice::zone_pec_sum(std::uint64_t zone) const {
    const auto physical = physical_zone(zone);
    std::uint64_t sum = 0;
    for (auto b = physical * zone_size_blocks_; b < (physical + 1) * zone_size_blocks_; ++b)
        sum += physical_pec_[b];
    return sum;
}
namespace {
std::uint64_t fragment_mask(std::uint32_t offset, std::uint32_t bytes) {
    if (offset >= kPageBytes || !bytes || offset % 64 || bytes % 64 || bytes > kPageBytes - offset)
        throw std::invalid_argument("HBF fragments require aligned 64-byte units within one page");
    return bytes == kPageBytes ? ~std::uint64_t{0} :
        ((std::uint64_t{1} << (bytes / 64)) - 1) << (offset / 64);
}
}
HbfDevice::WriteAdmission HbfDevice::receive_write_fragment(std::uint64_t ppn,
    std::uint32_t offset, std::uint32_t bytes, std::uint64_t command, double at) {
    validate_page(ppn);
    const auto mask = fragment_mask(offset, bytes);
    if (!std::isfinite(at) || at < write_clock_ns_ || next_write_event_ns() <= at)
        throw std::invalid_argument("advance HBF write events before receiving a fragment");
    write_clock_ns_ = at;
    started_ = true;
    const auto channel = channel_for_page(ppn);
    auto found = writes_.find(ppn);
    if (found != writes_.end()) {
        if (found->second.programming) return {.status = 0x6, .additional_status = 0x1};
        if (found->second.coverage & mask) return {.status = 0x2};
    } else {
        const auto block_index = ppn / config_.pages_per_block;
        const auto block = blocks_.find(block_index);
        const auto next_page = block == blocks_.end() ? 0 : block->second.next_page;
        const auto page = ppn % config_.pages_per_block;
        if (page && page != next_page) return {.status = 0x6, .additional_status = 0x1};
        // A new page-zero cycle cannot retire in-flight pages of its block.
        if (!page)
            for (const auto& [active, write] : writes_) {
                (void)write;
                if (active / config_.pages_per_block == block_index)
                    return {.status = 0x6, .additional_status = 0x1};
            }
        if (outstanding_writes_[channel] >= config_.outstanding_write_pages_per_channel)
            return {.status = 0x4};
        const auto deadline = at + config_.write_accumulation_timeout_ns;
        if (!std::isfinite(deadline) || deadline <= at)
            throw std::invalid_argument("HBF accumulation timeout exceeds time resolution");
        found = writes_.emplace(ppn, PartialWrite{.event_ns = deadline}).first;
        write_events_.emplace(std::pair{deadline, ppn}, false);
        ++outstanding_writes_[channel];
    }
    auto& write = found->second;
    write.coverage |= mask;
    write.commands.push_back(command);
    const bool full = write.coverage == ~std::uint64_t{0};
    if (full) write_events_.erase({write.event_ns, ppn});
    return {.page_ready = full};
}
double HbfDevice::next_write_event_ns() const {
    return write_events_.empty() ? std::numeric_limits<double>::infinity() :
        write_events_.begin()->first.first;
}
std::vector<HbfDevice::WriteResponse> HbfDevice::advance_writes(double at) {
    if (!std::isfinite(at) || at < write_clock_ns_)
        throw std::invalid_argument("HBF write event time must be finite and nondecreasing");
    std::vector<WriteResponse> responses;
    while (!write_events_.empty() && write_events_.begin()->first.first <= at) {
        const auto event = write_events_.begin();
        const auto [finish, ppn] = event->first;
        const auto& write = writes_.at(ppn);
        for (const auto command : write.commands)
            responses.push_back({command, static_cast<std::uint8_t>(write.programming ? 0 : 0x5), 0, finish});
        --outstanding_writes_.at(channel_for_page(ppn));
        writes_.erase(ppn);
        write_events_.erase(event);
    }
    write_clock_ns_ = at;
    return responses;
}
std::optional<std::uint8_t> HbfDevice::buffered_write_read_status(
    std::uint64_t ppn, std::uint32_t offset, std::uint32_t bytes) const {
    validate_page(ppn);
    const auto mask = fragment_mask(offset, bytes);
    const auto found = writes_.find(ppn);
    if (found == writes_.end()) return std::nullopt;
    return (found->second.coverage & mask) == mask ? 0 : 0xA;
}
bool HbfDevice::prepare_program(std::uint64_t ppn) {
    validate_page(ppn);
    if (const auto pending = writes_.find(ppn); pending != writes_.end() &&
        (pending->second.coverage != ~std::uint64_t{0} || pending->second.programming))
        throw std::logic_error("HBF program requires one complete, unprogrammed 4 KiB buffer");
    started_ = true;
    auto& block = blocks_[ppn / config_.pages_per_block];
    const auto page = ppn % config_.pages_per_block;
    if (page && page != block.next_page) throw std::runtime_error("HBF write-order status 0x6");
    if (!page) { cache_purge_block(ppn / config_.pages_per_block);
        for (std::uint32_t p = 0; p < config_.pages_per_block; ++p) program_ready_.erase(ppn + p); }
    block.next_page = static_cast<std::uint32_t>(page + 1);
    // Any program consumes a pre-erased state.
    bool preerased = false;
    if (!erased_.empty() && erased_[ppn / config_.pages_per_block]) {
        erased_[ppn / config_.pages_per_block] = 0; --unconsumed_erased_;
        preerased = page == 0;
    }
    return page == 0 && !preerased;
}
void HbfDevice::complete_program(std::uint64_t ppn, double ready) {
    validate_page(ppn);
    (void)blocks_.at(ppn / config_.pages_per_block);
    if (!std::isfinite(ready) || ready < 0) throw std::invalid_argument("invalid HBF program completion time");
    if (const auto pending = writes_.find(ppn); pending != writes_.end()) {
        auto& write = pending->second;
        if (write.coverage != ~std::uint64_t{0} || write.programming || ready < write_clock_ns_)
            throw std::logic_error("invalid HBF accumulated-page program completion");
        write.programming = true;
        write.event_ns = ready;
        write_events_.emplace(std::pair{ready, ppn}, true);
    }
    program_ready_[ppn] = ready;
    program_completions_.emplace(ready, ppn);
}
double HbfDevice::read_ready(std::uint64_t ppn) const {
    validate_page(ppn);
    const auto found = blocks_.find(ppn / config_.pages_per_block);
    if (found == blocks_.end() || ppn % config_.pages_per_block >= found->second.next_page)
        throw std::runtime_error("HBF erased-page read status 0x7");
    const auto ready = program_ready_.find(ppn);
    return ready == program_ready_.end() ? 0 : ready->second;
}
void HbfDevice::erase(std::uint64_t block) {
    validate_block(block);
    // A raw erase never substitutes for the page-zero auto erase.
    if (!erased_.empty() && erased_[block]) { erased_[block] = 0; --unconsumed_erased_; }
    blocks_.erase(block);
    cache_purge_block(block);
    for (auto page = block * config_.pages_per_block; page < (block + 1) * config_.pages_per_block; ++page)
        program_ready_.erase(page);
}
bool HbfDevice::apply_cache_event(std::vector<std::uint64_t>& resident, CacheEvent event) {
    const auto at = std::find(resident.begin(), resident.end(), event.page);
    if (event.action < 0) { if (at != resident.end()) resident.erase(at); return true; }
    if (event.action == 0 && at == resident.end()) return false;
    if (at != resident.end()) resident.erase(at);
    resident.push_back(event.page);
    if (resident.size() > kCachedPagesPerBank) resident.erase(resident.begin());
    return true;
}
std::vector<std::uint64_t> HbfDevice::project_cache(const BankCache& bank, double at, bool* valid) {
    auto resident = bank.resident;
    if (valid) *valid = true;
    for (const auto& [key, event] : bank.events) {
        if (key.first > at) break;
        if (!apply_cache_event(resident, event) && valid) *valid = false;
    }
    return resident;
}
bool HbfDevice::cache_hit(std::uint64_t ppn, double at) {
    auto found = cache_.find(bank_for_page(ppn));
    if (found == cache_.end()) return false;
    auto& bank = found->second;
    if (!bank.candidates.contains(ppn)) return false;
    const auto resident = project_cache(bank, at);
    if (std::find(resident.begin(), resident.end(), ppn) == resident.end()) return false;
    const auto key = std::pair{at, touch_sequence_++};
    bank.events.emplace(key, CacheEvent{ppn, false});
    bool valid = true;
    (void)project_cache(bank, std::numeric_limits<double>::infinity(), &valid);
    if (!valid) bank.events.erase(key);
    else {
        ++bank.future_consumers;
        track_cache_events(found->first, bank);
    }
    return valid;
}
void HbfDevice::track_cache_events(std::size_t index, BankCache& bank) {
    if (bank.pending) return;
    pending_cache_banks_.push_back(index);
    bank.pending = true;
}
void HbfDevice::cache_fill(std::uint64_t ppn, double ready) {
    const auto index = bank_for_page(ppn);
    auto& bank = cache_[index];
    const auto key = std::pair{ready, touch_sequence_++};
    bank.events.emplace(key, CacheEvent{ppn, true});
    if (bank.future_consumers && bank.events.rbegin()->first != key) {
        bool valid = true;
        (void)project_cache(bank, std::numeric_limits<double>::infinity(), &valid);
        // A future cache-hit response is already committed. Bypass this fill
        // instead of evicting its data retroactively; residency stays <= 2.
        if (!valid) { bank.events.erase(key); return; }
    }
    bank.candidates.insert(ppn);
    track_cache_events(index, bank);
}
void HbfDevice::advance_cache(double watermark) {
    while (!program_completions_.empty() && program_completions_.begin()->first <= watermark) {
        const auto [ready, ppn] = *program_completions_.begin();
        const auto entry = program_ready_.find(ppn);
        if (entry != program_ready_.end() && entry->second == ready) program_ready_.erase(entry);
        program_completions_.erase(program_completions_.begin());
    }
    for (std::size_t i = 0; i < pending_cache_banks_.size();) {
        auto& bank = cache_.at(pending_cache_banks_[i]);
        auto event = bank.events.begin();
        while (event != bank.events.end() && event->first.first <= watermark) {
            if (!apply_cache_event(bank.resident, event->second))
                throw std::logic_error("HBF cache lost a reserved consumer");
            if (event->second.action == 0) --bank.future_consumers;
            event = bank.events.erase(event);
        }
        // The negative-lookup index covers resident pages and pending events,
        // not every page ever filled. Rebuild only after stale entries dominate
        // so continuous overlap cannot grow the index without bound.
        if (bank.events.empty() ||
            bank.candidates.size() > 2 * (bank.resident.size() + bank.events.size())) {
            bank.candidates = {bank.resident.begin(), bank.resident.end()};
            for (const auto& [key, pending] : bank.events)
                bank.candidates.insert(pending.page);
        }
        if (bank.events.empty()) {
            bank.pending = false;
            pending_cache_banks_[i] = pending_cache_banks_.back();
            pending_cache_banks_.pop_back();
        } else {
            ++i;
        }
    }
}
void HbfDevice::cache_purge_page(std::uint64_t ppn) {
    const auto b = cache_.find(bank_for_page(ppn));
    if (b == cache_.end()) return;
    auto& bank = b->second;
    if (bank.events.empty()) {
        std::erase(bank.resident, ppn);
        bank.candidates.erase(ppn);
        return;
    }
    // Keep prior fills: deleting history can resurrect an evicted page.
    // The host fences block invalidation after its outstanding consumers.
    bank.events.emplace(std::pair{bank.events.rbegin()->first.first, touch_sequence_++},
        CacheEvent{ppn, -1});
}
void HbfDevice::cache_purge_block(std::uint64_t block) {
    const auto b = cache_.find(bank_for_page(block * config_.pages_per_block));
    if (b == cache_.end()) return;
    std::set<std::uint64_t> pages;
    for (const auto page : b->second.resident)
        if (page / config_.pages_per_block == block) pages.insert(page);
    for (const auto& [key, event] : b->second.events) {
        (void)key;
        if (event.page / config_.pages_per_block == block) pages.insert(event.page);
    }
    for (const auto page : pages) cache_purge_page(page);
}
}
