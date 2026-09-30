#include "policies/reference/behavioral_tiering_policy.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace hbfsim::policy {
namespace {

[[nodiscard]] std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs,
                                        const char* label) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(label) + " overflows uint64_t");
    }
    return lhs + rhs;
}

[[nodiscard]] std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs,
                                        const char* label) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(label) + " overflows uint64_t");
    }
    return lhs * rhs;
}

[[nodiscard]] PhysicalRequest
physical_request(std::string id, Tier tier, Op op, double arrival_ns,
                 std::uint64_t addr, std::uint64_t bytes,
                 AddressSpace address_space, TraceConfig trace,
                 HeatmapTrafficSource source) {
    return PhysicalRequest{
        .id = std::move(id),
        .tier = tier,
        .op = op,
        .address_space = address_space,
        .trace = trace,
        .arrival_ns = arrival_ns,
        .addr = addr,
        .bytes = bytes,
        .stream_id = 0,
        .heatmap_source = source,
    };
}

[[nodiscard]] BaseDieLinkStats
aggregate_link_stats(const std::vector<BaseDieLink>& links) {
    BaseDieLinkStats total;
    total.links = links.size();
    for (const auto& link : links) {
        const auto& stats = link.stats();
        total.read_transfers += stats.read_transfers;
        total.write_transfers += stats.write_transfers;
        total.read_bytes += stats.read_bytes;
        total.write_bytes += stats.write_bytes;
        total.read_queue_wait_ns += stats.read_queue_wait_ns;
        total.write_queue_wait_ns += stats.write_queue_wait_ns;
        total.read_busy_ns += stats.read_busy_ns;
        total.write_busy_ns += stats.write_busy_ns;
        total.read_fixed_latency_work_ns += stats.read_fixed_latency_work_ns;
        total.write_fixed_latency_work_ns += stats.write_fixed_latency_work_ns;
        total.first_arrival_ns =
            std::min(total.first_arrival_ns, stats.first_arrival_ns);
        total.finish_ns = std::max(total.finish_ns, stats.finish_ns);
    }
    return total;
}

class RunEngine {
  public:
    RunEngine(const BehavioralTieringConfig& config,
              const std::vector<MemoryRequest>& requests)
        : config_(config), requests_(requests),
          page_size_(config.hbf.device.page_size_bytes) {
        validate_config();
        validate_memory_requests(requests_, "behavioral-tiering");
        if (requests_.empty()) {
            throw std::runtime_error(
                "behavioral-tiering requires at least one request");
        }

        result_.composition.retain_completions = config_.retain_completions;
        result_.composition.has_hbm = true;
        result_.composition.has_hbf = true;
        result_.composition.service_latencies_ns.assign(
            requests_.size(), std::numeric_limits<double>::quiet_NaN());
        result_.composition.offered_latencies_ns.assign(
            requests_.size(), std::numeric_limits<double>::quiet_NaN());
        result_.composition.source_latencies_ns.assign(
            requests_.size(), std::numeric_limits<double>::quiet_NaN());
        parents_.resize(requests_.size());
        if (requests_.front().phase) {
            std::size_t begin = 0;
            while (begin < requests_.size()) {
                const auto phase = requests_[begin].phase;
                std::size_t end = begin + 1;
                while (end < requests_.size() &&
                       requests_[end].phase == phase) {
                    ++end;
                }
                phase_groups_.push_back(PhaseGroup{
                    .begin = begin,
                    .end = end,
                });
                begin = end;
            }
        }

        result_.placement.admission_policy = config_.admission_policy;
        result_.placement.hbm_tier_pages = config_.hbm_tier_bytes / page_size_;
        result_.placement.hbm_tier_bytes = config_.hbm_tier_bytes;
        result_.placement.promotion_threshold = config_.promotion_threshold;
        result_.placement.history_capacity_pages =
            config_.history_capacity_pages;

        if (config_.address_heatmap_bins != 0) {
            address_heatmap_.emplace(make_composition_address_heatmap_config(
                config_.hbm, config_.hbf, requests_,
                config_.address_heatmap_bins));
            record_workload_address_traffic(*address_heatmap_, requests_);
        }
        hbm_.emplace(config_.hbm,
                     address_heatmap_ ? &*address_heatmap_ : nullptr);
        hbf_.emplace(config_.hbf,
                     address_heatmap_ ? &*address_heatmap_ : nullptr);
        hbf_->attach_hbm_buffer(*hbm_);
        if (config_.hbm_tier_bytes > hbm_->application_capacity_bytes()) {
            throw std::runtime_error(
                "behavioral-tiering resident pages overlap the HBF controller's HBM reservation");
        }
        links_.reserve(config_.hbf.device.stacks);
        for (std::uint32_t stack = 0; stack < config_.hbf.device.stacks; ++stack) {
            links_.emplace_back(config_.base_die_link,
                                "behavioral_tiering/base_die_link/stack" +
                                    std::to_string(stack));
        }
        prepare_backing_image();
    }

    [[nodiscard]] BehavioralTieringRunResult run() {
        if (phase_groups_.empty()) {
            for (std::size_t index = 0; index < requests_.size(); ++index) {
                schedule_parent(index, requests_[index].arrival_ns);
            }
        } else {
            schedule_phase_group(0, 0.0);
        }
        while (!events_.empty()) {
            auto event = events_.top();
            events_.pop();
            if (event.time_ns < now_ns_) {
                throw std::runtime_error(
                    "behavioral-tiering event traveled backward in time");
            }
            now_ns_ = event.time_ns;
            if (hbm_) hbm_->advance_buffer_frontier(now_ns_);
            event.action();
        }
        finalize();
        return std::move(result_);
    }

  private:
    enum class EventPriority : std::uint8_t {
        Completion = 0,
        Arrival = 1,
        Dispatch = 2,
    };

    struct Event {
        double time_ns = 0.0;
        EventPriority priority = EventPriority::Completion;
        std::uint64_t sequence = 0;
        std::function<void()> action;
    };

    struct EventLater {
        bool operator()(const Event& lhs, const Event& rhs) const {
            if (lhs.time_ns != rhs.time_ns) {
                return lhs.time_ns > rhs.time_ns;
            }
            if (lhs.priority != rhs.priority) {
                return lhs.priority > rhs.priority;
            }
            return lhs.sequence > rhs.sequence;
        }
    };

    struct Transaction {
        std::size_t request_index = 0;
        std::size_t transaction_index = 0;
        std::uint64_t page = 0;
        std::uint64_t offset = 0;
        std::uint64_t bytes = 0;
        double admitted_ns = std::numeric_limits<double>::quiet_NaN();
        std::optional<double> transition_parked_ns = std::nullopt;
        std::uint64_t decision_observation = 0;
        BehavioralDecisionAction planned_action =
            BehavioralDecisionAction::HbfBypass;
        std::uint32_t history_observations = 0;
        std::optional<std::uint64_t> planned_slot = std::nullopt;
        std::optional<std::uint64_t> planned_victim = std::nullopt;
        bool planned_victim_dirty = false;
        bool needs_backing_fill = false;
        bool complete = false;
    };

    struct Parent {
        std::size_t remaining = 0;
        double source_arrival_ns = std::numeric_limits<double>::quiet_NaN();
        double offered_arrival_ns = std::numeric_limits<double>::quiet_NaN();
        double first_admitted_ns = std::numeric_limits<double>::infinity();
        double finish_ns = 0.0;
        std::deque<std::shared_ptr<Transaction>> pending_transactions;
        bool admission_ready = false;
    };

    struct PhaseGroup {
        std::size_t begin = 0;
        std::size_t end = 0;
    };

    struct Resident {
        std::uint64_t slot = 0;
        bool dirty = false;
        bool busy = false;
        std::deque<std::shared_ptr<Transaction>> waiters{};
    };

    struct Installing {
        std::shared_ptr<Transaction> demand;
        bool needs_backing_fill = false;
        std::uint64_t slot = 0;
        std::optional<std::uint64_t> victim_page;
        bool victim_dirty = false;
        std::optional<double> capacity_stalled_ns = std::nullopt;
    };

    struct Evicting {
        std::uint64_t slot = 0;
    };

    struct Ghost {
        std::uint32_t observations = 0;
        std::list<std::uint64_t>::iterator lru;
    };

    struct PolicyResident {
        std::uint64_t slot = 0;
        bool dirty = false;
        // Position in resident_lru_ (front = least recently used). Every
        // observation moves the page to the back, so victim choice is O(1)
        // instead of a scan over a tier that can hold tens of millions of
        // pages.
        std::list<std::uint64_t>::iterator lru;
    };

    const BehavioralTieringConfig& config_;
    const std::vector<MemoryRequest>& requests_;
    const std::uint64_t page_size_;
    BehavioralTieringRunResult result_;
    std::optional<AddressHeatmap> address_heatmap_;
    std::optional<hbm::HbmDevice> hbm_;
    std::optional<host::HbfController> hbf_;
    std::vector<BaseDieLink> links_;
    std::vector<Parent> parents_;
    std::vector<PhaseGroup> phase_groups_;
    std::size_t active_phase_group_ = 0;
    std::size_t active_phase_parents_remaining_ = 0;

    std::priority_queue<Event, std::vector<Event>, EventLater> events_;
    std::uint64_t next_event_sequence_ = 0;
    double now_ns_ = 0.0;
    std::size_t inflight_transactions_ = 0;
    std::size_t pending_admission_transactions_ = 0;
    std::size_t completed_parents_ = 0;
    std::deque<std::size_t> admission_ready_parents_;
    bool admission_dispatch_scheduled_ = false;

    std::unordered_map<std::uint64_t, Resident> residents_;
    // Physical slot ownership is indexed explicitly. Scanning every resident
    // to validate a fresh slot makes a fitting tier O(promotions^2), which is
    // prohibitive for multi-million-page traces.
    std::unordered_map<std::uint64_t, std::uint64_t> slot_owners_;
    std::unordered_map<std::uint64_t, Installing> installing_;
    std::deque<std::uint64_t> installation_queue_;
    std::unordered_map<std::uint64_t, Evicting> evicting_;
    // Decisions are planned in causal trace-fragment order, independently of
    // physical timing. These per-page queues and eviction edges prevent the
    // round-robin executor from overtaking an earlier promotion/hit while
    // leaving unrelated pages free to overlap.
    std::unordered_map<std::uint64_t, std::deque<std::uint64_t>>
        planned_page_observations_;
    std::unordered_map<std::uint64_t, std::deque<std::size_t>>
        transition_waiters_;
    std::unordered_map<std::uint64_t, PolicyResident> policy_residents_;
    std::list<std::uint64_t> resident_lru_;
    std::uint64_t next_policy_slot_ = 0;

    std::list<std::uint64_t> ghost_lru_;
    std::unordered_map<std::uint64_t, Ghost> ghosts_;
    std::unordered_set<std::uint64_t> policy_backing_known_;

    bool drain_started_ = false;
    bool drain_done_ = false;
    std::uint64_t drain_writebacks_remaining_ = 0;

    void validate_config() const {
        if (page_size_ == 0) {
            throw std::runtime_error(
                "behavioral-tiering HBF page size must be positive");
        }
        if (config_.hbm_tier_bytes == 0 ||
            config_.hbm_tier_bytes > config_.hbm.device.capacity_bytes ||
            config_.hbm_tier_bytes % page_size_ != 0) {
            throw std::runtime_error(
                "behavioral-tiering HBM tier must be a positive, "
                "page-aligned prefix within HBM capacity");
        }
        if (config_.history_capacity_pages == 0 ||
            config_.history_capacity_pages >
                std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error(
                "behavioral-tiering history capacity must fit size_t");
        }
        if (config_.admission_policy ==
                BehavioralAdmissionPolicy::ReuseFiltered &&
            config_.promotion_threshold < 2) {
            throw std::runtime_error(
                "reuse-filtered behavioral admission requires a promotion "
                "threshold of at least two");
        }
        if (config_.hbf.device.stacks == 0) {
            throw std::runtime_error(
                "behavioral-tiering requires at least one HBF stack");
        }
    }

    void prepare_backing_image() {
        const auto& population = config_.initial_image_requests != nullptr
                                     ? *config_.initial_image_requests
                                     : requests_;
        if (config_.initial_image_requests != nullptr) {
            validate_memory_requests(population,
                                     "behavioral-tiering initial image");
            const auto required =
                collect_initial_read_lpns(requests_, page_size_);
            const auto supplied =
                collect_initial_read_lpns(population, page_size_);
            if (!std::includes(supplied.begin(), supplied.end(),
                               required.begin(), required.end())) {
                throw std::runtime_error(
                    "behavioral-tiering initial-image trace does not cover "
                    "every initial read page");
            }
        }
        const auto initial_pages =
            collect_initial_read_lpns(population, page_size_);
        hbf_->prepopulate_logical_pages(initial_pages);
        policy_backing_known_.insert(initial_pages.begin(),
                                     initial_pages.end());
    }

    void schedule(double time_ns, EventPriority priority,
                  std::function<void()> action) {
        if (!std::isfinite(time_ns) || time_ns < now_ns_) {
            throw std::runtime_error(
                "behavioral-tiering scheduled invalid causal time");
        }
        events_.push(Event{
            .time_ns = time_ns,
            .priority = priority,
            .sequence = next_event_sequence_++,
            .action = std::move(action),
        });
    }

    void schedule_parent(std::size_t request_index, double offered_arrival_ns) {
        const auto& request = requests_.at(request_index);
        auto& parent = parents_.at(request_index);
        if (std::isfinite(parent.offered_arrival_ns) ||
            offered_arrival_ns < request.arrival_ns) {
            throw std::runtime_error(
                "behavioral parent was scheduled with an invalid frontier");
        }
        parent.source_arrival_ns = request.arrival_ns;
        parent.offered_arrival_ns = offered_arrival_ns;
        schedule(offered_arrival_ns, EventPriority::Arrival,
                 [this, request_index] { offer_parent(request_index); });
    }

    void schedule_phase_group(std::size_t group_index,
                              double dependency_ready_ns) {
        if (group_index >= phase_groups_.size() ||
            active_phase_parents_remaining_ != 0) {
            throw std::runtime_error(
                "behavioral phase scheduler entered an invalid state");
        }
        active_phase_group_ = group_index;
        const auto& group = phase_groups_[group_index];
        active_phase_parents_remaining_ = group.end - group.begin;
        for (auto index = group.begin; index < group.end; ++index) {
            const auto source_arrival_ns = requests_[index].arrival_ns;
            const auto offered_arrival_ns =
                std::max(source_arrival_ns, dependency_ready_ns);
            const auto wait_ns = offered_arrival_ns - source_arrival_ns;
            if (wait_ns > 0.0) {
                ++result_.composition.phase_dependency_waited_ops;
                result_.composition.phase_dependency_wait_work_ns += wait_ns;
                if (!std::isfinite(
                        result_.composition.phase_dependency_wait_work_ns)) {
                    throw std::runtime_error(
                        "behavioral phase-dependency wait work is not finite");
                }
                result_.composition.phase_dependency_max_wait_ns = std::max(
                    result_.composition.phase_dependency_max_wait_ns,
                    wait_ns);
            }
            schedule_parent(index, offered_arrival_ns);
        }
    }

    [[nodiscard]] std::uint64_t slot_address(std::uint64_t slot,
                                             std::uint64_t offset = 0) const {
        return checked_add(
            checked_mul(slot, page_size_, "behavioral HBM slot address"),
            offset, "behavioral HBM byte address");
    }

    void offer_parent(std::size_t request_index) {
        const auto& request = requests_.at(request_index);
        auto& parent = parents_.at(request_index);
        result_.composition.first_offered_arrival_ns = std::min(
            result_.composition.first_offered_arrival_ns,
            parent.offered_arrival_ns);
        result_.composition.last_offered_arrival_ns = std::max(
            result_.composition.last_offered_arrival_ns,
            parent.offered_arrival_ns);

        auto remaining = request.bytes;
        auto cursor = request.addr;
        std::size_t transaction_index = 0;
        while (remaining != 0) {
            const auto offset = cursor % page_size_;
            const auto bytes = std::min(remaining, page_size_ - offset);
            auto transaction = std::make_shared<Transaction>(Transaction{
                .request_index = request_index,
                .transaction_index = transaction_index++,
                .page = cursor / page_size_,
                .offset = offset,
                .bytes = bytes,
            });
            ++parent.remaining;
            plan_transaction(transaction);
            parent.pending_transactions.push_back(std::move(transaction));
            ++pending_admission_transactions_;
            remaining -= bytes;
            if (remaining != 0) {
                cursor = checked_add(cursor, bytes,
                                     "behavioral next request fragment");
            }
        }
        if (parent.remaining == 0) {
            throw std::runtime_error(
                "behavioral-tiering parent produced no page transaction");
        }
        mark_parent_admission_ready(request_index);
        request_admission_dispatch();
    }

    [[nodiscard]] std::optional<std::uint64_t> transition_blocker(
        const std::shared_ptr<Transaction>& transaction) const {
        if (installing_.contains(transaction->page) ||
            evicting_.contains(transaction->page)) {
            return transaction->page;
        }
        const auto own =
            planned_page_observations_.find(transaction->page);
        if (own == planned_page_observations_.end() ||
            own->second.empty() ||
            own->second.front() > transaction->decision_observation) {
            throw std::runtime_error(
                "behavioral planned page-decision order is inconsistent");
        }
        if (own->second.front() < transaction->decision_observation) {
            return transaction->page;
        }
        if (!transaction->planned_victim) {
            return std::nullopt;
        }
        const auto victim =
            planned_page_observations_.find(*transaction->planned_victim);
        if (victim != planned_page_observations_.end() &&
            !victim->second.empty() &&
            victim->second.front() < transaction->decision_observation) {
            return *transaction->planned_victim;
        }
        return std::nullopt;
    }

    [[nodiscard]] std::uint64_t choose_policy_lru_victim() const {
        if (policy_residents_.empty() || resident_lru_.empty()) {
            throw std::runtime_error(
                "behavioral policy has no resident page to replace");
        }
        // Observations are strictly ordered, so the list front is the unique
        // least-recently-observed resident (the same victim the
        // placement oracle derives from observation counters).
        return resident_lru_.front();
    }

    void plan_transaction(const std::shared_ptr<Transaction>& transaction) {
        const auto& request = requests_.at(transaction->request_index);
        auto resident = policy_residents_.find(transaction->page);
        if (resident != policy_residents_.end()) {
            transaction->planned_action = BehavioralDecisionAction::HbmHit;
            transaction->planned_slot = resident->second.slot;
            ++result_.placement.hbm_hits;
            record_decision(transaction, transaction->planned_action, 0,
                            resident->second.slot);
            resident_lru_.splice(
                resident_lru_.end(), resident_lru_, resident->second.lru);
            if (request.op == Op::Write) {
                resident->second.dirty = true;
            }
            return;
        }

        const auto [promote, observations] =
            observe_nonresident(transaction->page);
        transaction->history_observations = observations;
        if (!promote) {
            transaction->planned_action = BehavioralDecisionAction::HbfBypass;
            ++result_.placement.hbf_bypasses;
            record_decision(transaction, transaction->planned_action,
                            observations);
            if (request.op == Op::Write) {
                policy_backing_known_.insert(transaction->page);
            }
            return;
        }

        transaction->planned_action = BehavioralDecisionAction::Promote;
        ++result_.placement.promotions;
        const auto tier_pages = result_.placement.hbm_tier_pages;
        std::uint64_t slot = 0;
        if (policy_residents_.size() < tier_pages) {
            if (next_policy_slot_ >= tier_pages) {
                throw std::runtime_error(
                    "behavioral policy lost an unused HBM slot");
            }
            slot = next_policy_slot_++;
        } else {
            const auto victim_page = choose_policy_lru_victim();
            const auto victim = policy_residents_.find(victim_page);
            if (victim == policy_residents_.end()) {
                throw std::runtime_error(
                    "behavioral policy lost its LRU victim");
            }
            slot = victim->second.slot;
            transaction->planned_victim = victim_page;
            transaction->planned_victim_dirty = victim->second.dirty;
            seed_evicted_page(victim_page);
            if (victim->second.dirty) {
                ++result_.placement.dirty_evictions;
                policy_backing_known_.insert(victim_page);
            } else {
                ++result_.placement.clean_evictions;
            }
            resident_lru_.erase(victim->second.lru);
            policy_residents_.erase(victim);
        }
        transaction->planned_slot = slot;

        const bool full_page_write = request.op == Op::Write &&
                                     transaction->offset == 0 &&
                                     transaction->bytes == page_size_;
        const bool partial_write_has_backing =
            request.op == Op::Write && !full_page_write &&
            policy_backing_known_.contains(transaction->page);
        transaction->needs_backing_fill =
            request.op == Op::Read || partial_write_has_backing;
        if (transaction->needs_backing_fill) {
            ++result_.placement.promotions_with_backing_fill;
        } else {
            ++result_.placement.promotions_without_backing_fill;
        }

        record_decision(transaction, transaction->planned_action, observations,
                        slot, transaction->planned_victim,
                        transaction->planned_victim_dirty);
        resident_lru_.push_back(transaction->page);
        const auto [_, inserted] = policy_residents_.emplace(
            transaction->page,
            PolicyResident{
                .slot = slot,
                .dirty = request.op == Op::Write,
                .lru = std::prev(resident_lru_.end()),
            });
        if (!inserted) {
            throw std::runtime_error(
                "behavioral policy promoted an existing resident");
        }
        result_.placement.peak_resident_pages =
            std::max(result_.placement.peak_resident_pages,
                     static_cast<std::uint64_t>(policy_residents_.size()));
    }

    void mark_parent_admission_ready(std::size_t request_index) {
        auto& parent = parents_.at(request_index);
        if (parent.pending_transactions.empty() || parent.admission_ready) {
            throw std::runtime_error(
                "behavioral parent entered the admission ring incorrectly");
        }
        parent.admission_ready = true;
        admission_ready_parents_.push_back(request_index);
    }

    void request_admission_dispatch() {
        if (admission_dispatch_scheduled_) {
            return;
        }
        admission_dispatch_scheduled_ = true;
        schedule(now_ns_, EventPriority::Dispatch, [this] {
            admission_dispatch_scheduled_ = false;
            pump_admission();
        });
    }

    void wake_transition_waiters(std::uint64_t page) {
        auto found = transition_waiters_.find(page);
        if (found == transition_waiters_.end()) {
            return;
        }
        auto waiters = std::move(found->second);
        transition_waiters_.erase(found);
        while (!waiters.empty()) {
            const auto request_index = waiters.front();
            waiters.pop_front();
            auto& parent = parents_.at(request_index);
            if (parent.admission_ready ||
                parent.pending_transactions.empty()) {
                throw std::runtime_error(
                    "behavioral transition waiter lost its parent");
            }
            const auto blocker =
                transition_blocker(parent.pending_transactions.front());
            if (blocker) {
                transition_waiters_[*blocker].push_back(request_index);
            } else {
                mark_parent_admission_ready(request_index);
            }
        }
    }

    void resume_after_transition(std::uint64_t page) {
        wake_transition_waiters(page);
        request_admission_dispatch();
    }

    void pump_admission() {
        const auto limit = config_.max_outstanding_requests;
        while (!admission_ready_parents_.empty() &&
               (limit == 0 || inflight_transactions_ < limit)) {
            const auto request_index = admission_ready_parents_.front();
            auto& parent = parents_.at(request_index);
            if (!parent.admission_ready ||
                parent.pending_transactions.empty()) {
                throw std::runtime_error(
                    "behavioral admission ring lost a ready parent");
            }
            auto& front = parent.pending_transactions.front();
            const auto blocker = transition_blocker(front);
            if (blocker) {
                admission_ready_parents_.pop_front();
                parent.admission_ready = false;
                transition_waiters_[*blocker].push_back(request_index);
                if (!front->transition_parked_ns) {
                    front->transition_parked_ns = now_ns_;
                }
                continue;
            }

            admission_ready_parents_.pop_front();
            parent.admission_ready = false;
            auto transaction = std::move(parent.pending_transactions.front());
            parent.pending_transactions.pop_front();
            if (pending_admission_transactions_ == 0) {
                throw std::runtime_error(
                    "behavioral pending-admission count underflow");
            }
            --pending_admission_transactions_;
            if (!parent.pending_transactions.empty()) {
                mark_parent_admission_ready(request_index);
            }
            if (transaction->transition_parked_ns) {
                const auto wait_ns =
                    now_ns_ - *transaction->transition_parked_ns;
                ++result_.placement.transition_waited_transactions;
                result_.placement.transition_wait_work_ns += wait_ns;
                result_.placement.transition_max_wait_ns =
                    std::max(result_.placement.transition_max_wait_ns, wait_ns);
                transaction->transition_parked_ns.reset();
            }
            transaction->admitted_ns = now_ns_;
            ++inflight_transactions_;
            if (!std::isfinite(parent.first_admitted_ns)) {
                parent.first_admitted_ns = now_ns_;
                const auto wait_ns =
                    now_ns_ - parent.offered_arrival_ns;
                if (wait_ns > 0.0) {
                    ++result_.composition.front_end_admission_waited_ops;
                    result_.composition.front_end_admission_wait_work_ns +=
                        wait_ns;
                    result_.composition.front_end_admission_max_wait_ns =
                        std::max(
                            result_.composition.front_end_admission_max_wait_ns,
                            wait_ns);
                }
            }
            route_transaction(std::move(transaction));
        }
    }

    void hash_decision_word(std::uint64_t word) {
        constexpr std::uint64_t kPrime = 1099511628211ull;
        for (unsigned int byte = 0; byte < 8; ++byte) {
            result_.placement.decision_fingerprint ^=
                (word >> (byte * 8)) & 0xffull;
            result_.placement.decision_fingerprint *= kPrime;
        }
    }

    void
    record_decision(const std::shared_ptr<Transaction>& transaction,
                    BehavioralDecisionAction action,
                    std::uint32_t history_observations,
                    std::optional<std::uint64_t> slot = std::nullopt,
                    std::optional<std::uint64_t> evicted_page = std::nullopt,
                    bool evicted_page_dirty = false) {
        const auto observation = ++result_.placement.page_observations;
        transaction->decision_observation = observation;
        hash_decision_word(observation);
        hash_decision_word(transaction->request_index);
        hash_decision_word(transaction->transaction_index);
        hash_decision_word(transaction->page);
        hash_decision_word(
            requests_.at(transaction->request_index).op == Op::Read ? 0 : 1);
        hash_decision_word(static_cast<std::uint64_t>(action));
        hash_decision_word(history_observations);
        hash_decision_word(
            slot.value_or(std::numeric_limits<std::uint64_t>::max()));
        hash_decision_word(
            evicted_page.value_or(std::numeric_limits<std::uint64_t>::max()));
        hash_decision_word(evicted_page_dirty ? 1 : 0);
        auto& page_observations =
            planned_page_observations_[transaction->page];
        if (!page_observations.empty() &&
            page_observations.back() >= observation) {
            throw std::runtime_error(
                "behavioral planned page observations are not increasing");
        }
        page_observations.push_back(observation);
        if (evicted_page) {
            auto& victim_observations =
                planned_page_observations_[*evicted_page];
            if (!victim_observations.empty() &&
                victim_observations.back() >= observation) {
                throw std::runtime_error(
                    "behavioral planned victim observations are not "
                    "increasing");
            }
            victim_observations.push_back(observation);
        }
        if (!config_.retain_decisions) {
            return;
        }
        result_.decisions.push_back(BehavioralPlacementDecision{
            .observation = observation,
            .request_index = transaction->request_index,
            .transaction_index = transaction->transaction_index,
            .logical_page = transaction->page,
            .op = requests_.at(transaction->request_index).op,
            .action = action,
            .history_observations = history_observations,
            .hbm_slot = slot,
            .evicted_page = evicted_page,
            .evicted_page_dirty = evicted_page_dirty,
        });
    }

    void route_transaction(std::shared_ptr<Transaction> transaction) {
        if (transaction->planned_action == BehavioralDecisionAction::HbmHit) {
            const auto resident = residents_.find(transaction->page);
            if (resident == residents_.end() || !transaction->planned_slot ||
                resident->second.slot != *transaction->planned_slot) {
                throw std::runtime_error(
                    "behavioral planned HBM hit lost its resident slot");
            }
            issue_or_queue_resident(std::move(transaction));
            return;
        }
        if (transaction->planned_action ==
            BehavioralDecisionAction::HbfBypass) {
            issue_hbf_bypass(std::move(transaction));
            return;
        }
        if (transaction->planned_action != BehavioralDecisionAction::Promote ||
            !transaction->planned_slot) {
            throw std::runtime_error(
                "behavioral transaction has an invalid planned action");
        }
        const auto page = transaction->page;
        const auto needs_backing_fill = transaction->needs_backing_fill;
        const auto slot = *transaction->planned_slot;
        const auto victim_page = transaction->planned_victim;
        const auto victim_dirty = transaction->planned_victim_dirty;
        const auto [_, inserted] = installing_.emplace(
            page, Installing{
                      .demand = std::move(transaction),
                      .needs_backing_fill = needs_backing_fill,
                      .slot = slot,
                      .victim_page = victim_page,
                      .victim_dirty = victim_dirty,
                  });
        if (!inserted) {
            throw std::runtime_error(
                "behavioral-tiering created duplicate page installation");
        }
        installation_queue_.push_back(page);
        result_.placement.peak_installing_pages =
            std::max(result_.placement.peak_installing_pages,
                     static_cast<std::uint64_t>(installing_.size()));
        pump_installations();
    }

    [[nodiscard]] std::pair<bool, std::uint32_t>
    observe_nonresident(std::uint64_t page) {
        if (config_.admission_policy ==
            BehavioralAdmissionPolicy::AlwaysAdmit) {
            ++result_.placement.cold_misses;
            return {true, 1};
        }

        const auto found = ghosts_.find(page);
        if (found == ghosts_.end()) {
            ++result_.placement.cold_misses;
            insert_or_update_ghost(page, 1);
            return {false, 1};
        }

        ++result_.placement.history_hits;
        auto observations = found->second.observations;
        if (observations != std::numeric_limits<std::uint32_t>::max()) {
            ++observations;
        }
        ghost_lru_.erase(found->second.lru);
        ghosts_.erase(found);
        if (observations >= config_.promotion_threshold) {
            return {true, observations};
        }
        insert_or_update_ghost(page, observations);
        return {false, observations};
    }

    void insert_or_update_ghost(std::uint64_t page,
                                std::uint32_t observations) {
        const auto old = ghosts_.find(page);
        if (old != ghosts_.end()) {
            ghost_lru_.erase(old->second.lru);
            ghosts_.erase(old);
        }
        while (ghosts_.size() >= config_.history_capacity_pages) {
            if (ghost_lru_.empty()) {
                throw std::runtime_error(
                    "behavioral ghost history lost its LRU state");
            }
            const auto victim = ghost_lru_.front();
            ghost_lru_.pop_front();
            if (ghosts_.erase(victim) != 1) {
                throw std::runtime_error(
                    "behavioral ghost LRU referenced an absent page");
            }
            ++result_.placement.history_evictions;
        }
        ghost_lru_.push_back(page);
        const auto iterator = std::prev(ghost_lru_.end());
        ghosts_.emplace(page,
                        Ghost{.observations = observations, .lru = iterator});
        result_.placement.peak_history_pages =
            std::max(result_.placement.peak_history_pages,
                     static_cast<std::uint64_t>(ghosts_.size()));
    }

    void seed_evicted_page(std::uint64_t page) {
        if (config_.admission_policy !=
            BehavioralAdmissionPolicy::ReuseFiltered) {
            return;
        }
        insert_or_update_ghost(page, config_.promotion_threshold - 1);
    }

    void issue_hbf_bypass(std::shared_ptr<Transaction> transaction) {
        const auto& request = requests_.at(transaction->request_index);
        const auto logical_addr =
            checked_add(checked_mul(transaction->page, page_size_,
                                    "behavioral HBF bypass page address"),
                        transaction->offset, "behavioral HBF bypass address");
        auto completion = hbf_->issue(
            physical_request(request.id + "/behavioral/bypass/page" +
                                 std::to_string(transaction->page) + "/off" +
                                 std::to_string(transaction->offset),
                             Tier::HBF, request.op, now_ns_, logical_addr,
                             transaction->bytes, AddressSpace::Logical,
                             config_.trace, HeatmapTrafficSource::Workload));
        const auto finish_ns = completion.finish_ns;
        result_.placement.hbf_bypass_bytes =
            checked_add(result_.placement.hbf_bypass_bytes, transaction->bytes,
                        "behavioral HBF bypass bytes");
        add_user_completion(result_.composition, std::move(completion), false,
                            true);
        schedule(finish_ns, EventPriority::Completion,
                 [this, transaction = std::move(transaction)] {
                     complete_transaction(transaction);
                 });
    }

    void issue_or_queue_resident(std::shared_ptr<Transaction> transaction) {
        const auto found = residents_.find(transaction->page);
        if (found == residents_.end()) {
            throw std::runtime_error(
                "behavioral HBM hit lost resident page state");
        }
        if (found->second.busy) {
            found->second.waiters.push_back(std::move(transaction));
            return;
        }
        issue_resident(std::move(transaction));
    }

    void issue_resident(std::shared_ptr<Transaction> transaction) {
        auto found = residents_.find(transaction->page);
        if (found == residents_.end() || found->second.busy) {
            throw std::runtime_error(
                "behavioral resident issue found invalid page state");
        }
        auto& resident = found->second;
        resident.busy = true;
        const auto& request = requests_.at(transaction->request_index);
        if (request.op == Op::Write) {
            resident.dirty = true;
        }
        auto completion = hbm_->issue(
            physical_request(request.id + "/behavioral/hbm/page" +
                                 std::to_string(transaction->page) + "/off" +
                                 std::to_string(transaction->offset),
                             Tier::HBM, request.op, now_ns_,
                             slot_address(resident.slot, transaction->offset),
                             transaction->bytes, AddressSpace::Physical,
                             config_.trace, HeatmapTrafficSource::Workload));
        const auto finish_ns = completion.finish_ns;
        result_.placement.hbm_foreground_bytes =
            checked_add(result_.placement.hbm_foreground_bytes,
                        transaction->bytes, "behavioral HBM foreground bytes");
        add_user_completion(result_.composition, std::move(completion), true,
                            false);
        schedule(finish_ns, EventPriority::Completion,
                 [this, page = transaction->page,
                  transaction = std::move(transaction)] {
                     complete_resident_transaction(page, transaction);
                 });
    }

    void complete_resident_transaction(
        std::uint64_t page, const std::shared_ptr<Transaction>& transaction) {
        auto found = residents_.find(page);
        if (found == residents_.end() || !found->second.busy) {
            throw std::runtime_error(
                "behavioral resident completion lost busy page");
        }
        found->second.busy = false;
        std::shared_ptr<Transaction> next;
        if (!found->second.waiters.empty()) {
            next = std::move(found->second.waiters.front());
            found->second.waiters.pop_front();
            issue_resident(next);
        }
        complete_transaction(transaction);
        found = residents_.find(page);
        if (found != residents_.end() && !found->second.busy &&
            found->second.waiters.empty()) {
            pump_installations();
        }
    }

    void note_capacity_stall(Installing& installing) {
        if (!installing.capacity_stalled_ns) {
            installing.capacity_stalled_ns = now_ns_;
            ++result_.placement.capacity_stalled_promotions;
        }
    }

    void pump_installations() {
        while (!installation_queue_.empty()) {
            const auto page = installation_queue_.front();
            auto installing = installing_.find(page);
            if (installing == installing_.end()) {
                throw std::runtime_error(
                    "behavioral installation queue lost page state");
            }
            const auto slot = installing->second.slot;
            if (slot >= result_.placement.hbm_tier_pages) {
                throw std::runtime_error(
                    "behavioral planned installation slot exceeds capacity");
            }
            if (!installing->second.victim_page) {
                if (slot_owners_.contains(slot)) {
                    throw std::runtime_error(
                        "behavioral initial installation reused a live slot");
                }
                installation_queue_.pop_front();
                begin_install_fill(page);
                continue;
            }

            const auto victim_page = *installing->second.victim_page;
            if (installing_.contains(victim_page) ||
                evicting_.contains(victim_page)) {
                note_capacity_stall(installing->second);
                return;
            }
            auto victim = residents_.find(victim_page);
            if (victim == residents_.end() || victim->second.slot != slot) {
                throw std::runtime_error(
                    "behavioral planned eviction lost its victim slot");
            }
            if (victim->second.busy || !victim->second.waiters.empty()) {
                note_capacity_stall(installing->second);
                return;
            }
            const bool dirty = victim->second.dirty;
            if (dirty != installing->second.victim_dirty) {
                throw std::runtime_error(
                    "behavioral physical dirty state diverged from policy");
            }
            const auto owner = slot_owners_.find(slot);
            if (owner == slot_owners_.end() ||
                owner->second != victim_page) {
                throw std::runtime_error(
                    "behavioral slot ownership lost its eviction victim");
            }
            installation_queue_.pop_front();
            if (installing->second.capacity_stalled_ns) {
                result_.placement.capacity_stall_work_ns +=
                    now_ns_ - *installing->second.capacity_stalled_ns;
            }
            slot_owners_.erase(owner);
            residents_.erase(victim);
            if (!dirty) {
                begin_install_fill(page);
                resume_after_transition(victim_page);
                continue;
            }

            const auto [_, inserted] =
                evicting_.emplace(victim_page, Evicting{.slot = slot});
            if (!inserted) {
                throw std::runtime_error(
                    "behavioral dirty victim was already evicting");
            }
            begin_dirty_writeback(victim_page, slot, false,
                                  [this, page] { begin_install_fill(page); });
        }
    }

    void begin_install_fill(std::uint64_t page) {
        auto found = installing_.find(page);
        if (found == installing_.end()) {
            throw std::runtime_error(
                "behavioral install fill lacks a physical slot");
        }
        if (!found->second.needs_backing_fill) {
            finish_install(page);
            return;
        }
        const auto slot = found->second.slot;
        auto completion = hbf_->issue(physical_request(
            "behavioral/page" + std::to_string(page) + "/fill-hbf-read",
            Tier::HBF, Op::Read, now_ns_,
            checked_mul(page, page_size_, "behavioral fill HBF address"),
            page_size_, AddressSpace::Logical, config_.trace,
            HeatmapTrafficSource::PrefetchFill));
        const auto finish_ns = completion.finish_ns;
        ++result_.composition.hbf_background_accesses;
        ++result_.placement.backing_fill_pages;
        result_.placement.backing_fill_bytes =
            checked_add(result_.placement.backing_fill_bytes, page_size_,
                        "behavioral backing fill bytes");
        add_background_completion(result_.composition, std::move(completion));
        schedule(finish_ns, EventPriority::Completion,
                 [this, page, slot] { issue_fill_d2d(page, slot); });
    }

    void issue_fill_d2d(std::uint64_t page, std::uint64_t slot) {
        const auto stack = hbf_->stack_for_logical_page(page);
        auto completion = links_.at(stack).issue(
            "behavioral/page" + std::to_string(page) + "/fill-d2d-read",
            Op::Read, now_ns_, page_size_, config_.trace);
        const auto finish_ns = completion.finish_ns;
        add_background_completion(result_.composition, std::move(completion));
        schedule(finish_ns, EventPriority::Completion,
                 [this, page, slot] { issue_hbm_install(page, slot); });
    }

    void issue_hbm_install(std::uint64_t page, std::uint64_t slot) {
        auto completion = hbm_->issue(physical_request(
            "behavioral/page" + std::to_string(page) + "/hbm-install",
            Tier::HBM, Op::Write, now_ns_, slot_address(slot), page_size_,
            AddressSpace::Physical, config_.trace,
            HeatmapTrafficSource::StreamingInstall));
        const auto finish_ns = completion.finish_ns;
        ++result_.composition.hbm_background_accesses;
        ++result_.placement.hbm_install_pages;
        result_.placement.hbm_install_bytes =
            checked_add(result_.placement.hbm_install_bytes, page_size_,
                        "behavioral HBM install bytes");
        add_background_completion(result_.composition, std::move(completion));
        schedule(finish_ns, EventPriority::Completion,
                 [this, page] { finish_install(page); });
    }

    void finish_install(std::uint64_t page) {
        auto found = installing_.find(page);
        if (found == installing_.end()) {
            throw std::runtime_error(
                "behavioral install completion lost page state");
        }
        auto install = std::move(found->second);
        installing_.erase(found);
        const auto [resident, inserted] =
            residents_.emplace(page, Resident{
                                         .slot = install.slot,
                                         .dirty = false,
                                         .busy = false,
                                     });
        if (!inserted) {
            throw std::runtime_error(
                "behavioral install replaced a resident page");
        }
        const bool slot_inserted =
            slot_owners_.emplace(install.slot, page).second;
        if (!slot_inserted) {
            throw std::runtime_error(
                "behavioral install reused an occupied physical slot");
        }
        const auto& request = requests_.at(install.demand->request_index);
        if (request.op == Op::Write) {
            issue_resident(install.demand);
        }
        request_admission_dispatch();
        if (request.op == Op::Read) {
            complete_transaction(install.demand);
        }
        pump_installations();
    }

    void begin_dirty_writeback(std::uint64_t page, std::uint64_t slot,
                               bool drain, std::function<void()> continuation) {
        auto completion = hbm_->issue(physical_request(
            "behavioral/page" + std::to_string(page) +
                (drain ? "/drain-hbm-read" : "/evict-hbm-read"),
            Tier::HBM, Op::Read, now_ns_, slot_address(slot), page_size_,
            AddressSpace::Physical, config_.trace,
            HeatmapTrafficSource::Destage));
        const auto finish_ns = completion.finish_ns;
        ++result_.composition.hbm_background_accesses;
        add_background_completion(result_.composition, std::move(completion));
        schedule(finish_ns, EventPriority::Completion,
                 [this, page, drain,
                  continuation = std::move(continuation)]() mutable {
                     issue_writeback_d2d(page, drain, std::move(continuation));
                 });
    }

    void issue_writeback_d2d(std::uint64_t page, bool drain,
                             std::function<void()> continuation) {
        const auto stack = hbf_->stack_for_logical_page(page);
        auto completion = links_.at(stack).issue(
            "behavioral/page" + std::to_string(page) +
                (drain ? "/drain-d2d-write" : "/evict-d2d-write"),
            Op::Write, now_ns_, page_size_, config_.trace);
        const auto finish_ns = completion.finish_ns;
        add_background_completion(result_.composition, std::move(completion));
        schedule(finish_ns, EventPriority::Completion,
                 [this, page, drain,
                  continuation = std::move(continuation)]() mutable {
                     issue_hbf_writeback(page, drain, std::move(continuation));
                 });
    }

    void issue_hbf_writeback(std::uint64_t page, bool drain,
                             std::function<void()> continuation) {
        auto completion = hbf_->issue(physical_request(
            "behavioral/page" + std::to_string(page) +
                (drain ? "/drain-hbf-write" : "/evict-hbf-write"),
            Tier::HBF, Op::Write, now_ns_,
            checked_mul(page, page_size_, "behavioral HBF writeback address"),
            page_size_, AddressSpace::Logical, config_.trace,
            HeatmapTrafficSource::Destage));
        const auto finish_ns = completion.finish_ns;
        ++result_.composition.hbf_background_accesses;
        ++result_.placement.dirty_writeback_pages;
        result_.placement.dirty_writeback_bytes =
            checked_add(result_.placement.dirty_writeback_bytes, page_size_,
                        "behavioral dirty writeback bytes");
        if (drain) {
            ++result_.placement.drain_writeback_pages;
            result_.placement.drain_writeback_bytes =
                checked_add(result_.placement.drain_writeback_bytes, page_size_,
                            "behavioral drain writeback bytes");
        }
        add_background_completion(result_.composition, std::move(completion));
        schedule(finish_ns, EventPriority::Completion,
                 [this, page, drain,
                  continuation = std::move(continuation)]() mutable {
                     if (drain) {
                         auto resident = residents_.find(page);
                         if (resident == residents_.end() ||
                             !resident->second.dirty) {
                             throw std::runtime_error(
                                 "behavioral drain lost dirty resident");
                         }
                         resident->second.dirty = false;
                         if (drain_writebacks_remaining_ == 0) {
                             throw std::runtime_error(
                                 "behavioral drain writeback underflow");
                         }
                         --drain_writebacks_remaining_;
                     } else {
                         if (evicting_.erase(page) != 1) {
                             throw std::runtime_error(
                                 "behavioral eviction completion lost victim");
                         }
                     }
                     continuation();
                     if (drain && drain_writebacks_remaining_ == 0) {
                         issue_final_hbf_drain();
                     }
                     if (!drain) {
                         resume_after_transition(page);
                     }
                 });
    }

    void retire_planned_observation(std::uint64_t page,
                                    std::uint64_t observation) {
        auto planned = planned_page_observations_.find(page);
        if (planned == planned_page_observations_.end() ||
            planned->second.empty() ||
            planned->second.front() != observation) {
            throw std::runtime_error(
                "behavioral completion violated planned page-decision order");
        }
        planned->second.pop_front();
        if (planned->second.empty()) {
            planned_page_observations_.erase(planned);
        }
    }

    void complete_transaction(const std::shared_ptr<Transaction>& transaction) {
        if (transaction->complete || !std::isfinite(transaction->admitted_ns) ||
            inflight_transactions_ == 0) {
            throw std::runtime_error(
                "behavioral transaction completion accounting failed");
        }
        transaction->complete = true;
        retire_planned_observation(
            transaction->page, transaction->decision_observation);
        if (transaction->planned_victim) {
            retire_planned_observation(
                *transaction->planned_victim,
                transaction->decision_observation);
        }
        wake_transition_waiters(transaction->page);
        if (transaction->planned_victim) {
            wake_transition_waiters(*transaction->planned_victim);
        }
        --inflight_transactions_;
        auto& parent = parents_.at(transaction->request_index);
        if (parent.remaining == 0) {
            throw std::runtime_error(
                "behavioral parent transaction count underflow");
        }
        --parent.remaining;
        parent.finish_ns = std::max(parent.finish_ns, now_ns_);
        if (parent.remaining == 0) {
            const auto& request = requests_.at(transaction->request_index);
            if (!std::isfinite(parent.first_admitted_ns) ||
                !std::isfinite(parent.source_arrival_ns) ||
                !std::isfinite(parent.offered_arrival_ns) ||
                parent.finish_ns < parent.first_admitted_ns ||
                parent.first_admitted_ns < parent.offered_arrival_ns ||
                parent.offered_arrival_ns < parent.source_arrival_ns) {
                throw std::runtime_error(
                    "behavioral parent latency frontiers are invalid");
            }
            result_.composition.service_latencies_ns.at(
                transaction->request_index) =
                parent.finish_ns - parent.first_admitted_ns;
            result_.composition.offered_latencies_ns.at(
                transaction->request_index) =
                parent.finish_ns - parent.offered_arrival_ns;
            result_.composition.source_latencies_ns.at(
                transaction->request_index) =
                parent.finish_ns - parent.source_arrival_ns;
            ++result_.composition.ops;
            request.op == Op::Read ? ++result_.composition.reads
                                   : ++result_.composition.writes;
            result_.composition.logical_bytes =
                checked_add(result_.composition.logical_bytes, request.bytes,
                            "behavioral logical bytes");
            result_.composition.user_finish_ns =
                std::max(result_.composition.user_finish_ns, parent.finish_ns);
            result_.composition.finish_ns =
                std::max(result_.composition.finish_ns, parent.finish_ns);
            ++completed_parents_;
            if (!phase_groups_.empty()) {
                const auto& group = phase_groups_.at(active_phase_group_);
                if (transaction->request_index < group.begin ||
                    transaction->request_index >= group.end ||
                    active_phase_parents_remaining_ == 0) {
                    throw std::runtime_error(
                        "behavioral parent completed outside its active phase");
                }
                --active_phase_parents_remaining_;
                if (active_phase_parents_remaining_ == 0 &&
                    active_phase_group_ + 1 < phase_groups_.size()) {
                    ++result_.composition.phase_barriers;
                    schedule_phase_group(active_phase_group_ + 1, now_ns_);
                }
            }
        }
        request_admission_dispatch();
        if (completed_parents_ == requests_.size() && !drain_started_) {
            begin_drain();
        }
    }

    void begin_drain() {
        if (!admission_ready_parents_.empty() ||
            pending_admission_transactions_ != 0 || !installing_.empty() ||
            !installation_queue_.empty() || !evicting_.empty() ||
            !planned_page_observations_.empty() ||
            !transition_waiters_.empty() ||
            inflight_transactions_ != 0) {
            throw std::runtime_error(
                "behavioral user completion left controller work in flight");
        }
        if (residents_.size() != policy_residents_.size() ||
            residents_.size() != slot_owners_.size() ||
            resident_lru_.size() != policy_residents_.size()) {
            throw std::runtime_error(
                "behavioral physical/policy residency size diverged at drain");
        }
        for (const auto& [slot, page] : slot_owners_) {
            const auto resident = residents_.find(page);
            if (resident == residents_.end() ||
                resident->second.slot != slot) {
                throw std::runtime_error(
                    "behavioral physical slot ownership diverged at drain");
            }
        }
        for (const auto& [page, policy] : policy_residents_) {
            const auto physical = residents_.find(page);
            if (physical == residents_.end() ||
                physical->second.slot != policy.slot ||
                physical->second.dirty != policy.dirty) {
                throw std::runtime_error(
                    "behavioral physical/policy residency diverged at drain");
            }
        }
        drain_started_ = true;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> dirty;
        for (const auto& [page, resident] : residents_) {
            if (resident.busy || !resident.waiters.empty()) {
                throw std::runtime_error(
                    "behavioral drain found busy resident page");
            }
            if (resident.dirty) {
                dirty.emplace_back(page, resident.slot);
            }
        }
        std::sort(dirty.begin(), dirty.end());
        drain_writebacks_remaining_ = dirty.size();
        if (dirty.empty()) {
            issue_final_hbf_drain();
            return;
        }
        for (const auto& [page, slot] : dirty) {
            begin_dirty_writeback(page, slot, true, [] {});
        }
    }

    void issue_final_hbf_drain() {
        if (drain_done_) {
            throw std::runtime_error(
                "behavioral HBF drain was issued more than once");
        }
        auto completion =
            hbf_->drain_pending("behavioral/HBF/drain", now_ns_, config_.trace);
        const auto finish_ns = completion.finish_ns;
        ++result_.composition.hbf_background_accesses;
        add_background_completion(result_.composition, std::move(completion));
        schedule(finish_ns, EventPriority::Completion,
                 [this] { drain_done_ = true; });
    }

    void finalize() {
        if (!drain_started_ || !drain_done_ ||
            completed_parents_ != requests_.size() ||
            inflight_transactions_ != 0) {
            throw std::runtime_error(
                "behavioral-tiering run did not reach quiescence: "
                "drain_started=" +
                std::to_string(drain_started_) +
                " drain_done=" + std::to_string(drain_done_) +
                " completed=" + std::to_string(completed_parents_) + "/" +
                std::to_string(requests_.size()) +
                " inflight=" + std::to_string(inflight_transactions_) +
                " admission_parents=" +
                std::to_string(admission_ready_parents_.size()) +
                " admission_transactions=" +
                std::to_string(pending_admission_transactions_) +
                " slots=" + std::to_string(slot_owners_.size()) +
                " installing=" + std::to_string(installing_.size()) +
                " install_queue=" + std::to_string(installation_queue_.size()) +
                " evicting=" + std::to_string(evicting_.size()) +
                " planned_page_queues=" +
                std::to_string(planned_page_observations_.size()) +
                " transition_wait_pages=" +
                std::to_string(transition_waiters_.size()));
        }
        if (result_.composition.ops != requests_.size()) {
            throw std::runtime_error(
                "behavioral-tiering lost parent operations");
        }
        for (const auto latency : result_.composition.service_latencies_ns) {
            if (!std::isfinite(latency) || latency < 0.0) {
                throw std::runtime_error(
                    "behavioral-tiering retained invalid latency");
            }
        }
        result_.placement.final_resident_pages = residents_.size();
        result_.placement.final_dirty_pages =
            std::count_if(residents_.begin(), residents_.end(),
                          [](const auto& entry) { return entry.second.dirty; });
        if (result_.placement.final_dirty_pages != 0) {
            throw std::runtime_error(
                "behavioral-tiering drain left dirty HBM pages");
        }
        result_.composition.hbm_stats = hbm_->stats();
        result_.composition.hbf_stats = hbf_->stats();
        result_.composition.hbf_wear_snapshot = hbf_->wear_snapshot_json();
        result_.composition.base_die_link_stats = aggregate_link_stats(links_);
        if (address_heatmap_) {
            result_.composition.address_heatmap = address_heatmap_->snapshot();
        }
    }
};

} // namespace

const char* to_string(BehavioralAdmissionPolicy policy) {
    switch (policy) {
    case BehavioralAdmissionPolicy::AlwaysAdmit:
        return "always-admit";
    case BehavioralAdmissionPolicy::ReuseFiltered:
        return "reuse-filtered";
    }
    throw std::runtime_error("unknown behavioral admission policy");
}

const char* to_string(BehavioralDecisionAction action) {
    switch (action) {
    case BehavioralDecisionAction::HbmHit:
        return "hbm-hit";
    case BehavioralDecisionAction::HbfBypass:
        return "hbf-bypass";
    case BehavioralDecisionAction::Promote:
        return "promote";
    }
    throw std::runtime_error("unknown behavioral decision action");
}

BehavioralTieringRunResult
run_behavioral_tiering_policy(const BehavioralTieringConfig& config,
                                   const std::vector<MemoryRequest>& requests) {
    return RunEngine(config, requests).run();
}

} // namespace hbfsim::policy
