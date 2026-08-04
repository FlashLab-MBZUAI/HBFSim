#include "physical/hybrid/layer_streaming_composition.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace hbfsim::physical::hybrid {
namespace {

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs + rhs;
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs * rhs;
}

std::uint64_t ceil_div(
    std::uint64_t numerator,
    std::uint64_t denominator,
    const char* name) {
    if (denominator == 0) {
        throw std::runtime_error(std::string(name) + " has zero denominator");
    }
    return numerator / denominator +
        static_cast<std::uint64_t>(numerator % denominator != 0);
}

bool is_hbm_only(SemanticKind kind) {
    return kind == SemanticKind::Scratch || kind == SemanticKind::Metadata;
}

struct PageSemantics {
    bool hbm_only = false;
    bool model_weight = false;
    bool shared_context = false;
    bool generated_context = false;
    bool unknown = false;
    bool written = false;
    bool first_touch_shared_context = false;
};

std::uint64_t address_footprint_bytes(
    const std::vector<MemoryRequest>& requests) {
    std::uint64_t footprint = 0;
    for (const auto& request : requests) {
        footprint = std::max(
            footprint,
            checked_add(request.addr, request.bytes, "request footprint"));
    }
    return footprint;
}

PhysicalRequest physical_request(
    std::string id,
    Tier tier,
    Op op,
    double arrival_ns,
    std::uint64_t addr,
    std::uint64_t bytes,
    AddressSpace address_space,
    TraceConfig trace,
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

} // namespace

LayerStreamingComposition::LayerStreamingComposition(LayerStreamingConfig config)
    : config_(std::move(config)),
      hbm_(config_.hbm) {
    if (config_.backing == BackingTier::Hbf) {
        hbf_.emplace(config_.hbf);
    } else {
        external_.emplace(config_.external_backing);
    }
    const auto page_size = backing_page_size();
    if (page_size == 0 || config_.layer_buffer_bytes < page_size ||
        config_.layer_buffer_bytes % page_size != 0) {
        throw std::runtime_error(
            "layer-buffer-bytes must contain an integral positive number of "
            "backing pages");
    }
    if (config_.hbm.capacity_bytes % page_size != 0) {
        throw std::runtime_error(
            "HBM capacity must be aligned to the backing page size");
    }
    if (config_.residency_contract) {
        const auto& contract = *config_.residency_contract;
        const std::array<std::pair<std::uint64_t, const char*>, 10>
            positive_fields{{
                {contract.page_size_bytes, "page size bytes"},
                {contract.unique_resident_footprint_bytes,
                 "unique resident footprint"},
                {contract.immutable_weight_bytes,
                 "immutable weight bytes"},
                {contract.immutable_weight_pages,
                 "immutable weight pages"},
                {contract.runtime_overhead_bytes, "runtime overhead bytes"},
                {contract.block_table_bytes, "block table bytes"},
                {contract.active_buffer_bytes_per_slot,
                 "active buffer bytes per slot"},
                {contract.kv_block_stride_bytes, "KV block stride bytes"},
                {contract.logical_kv_blocks, "logical KV blocks"},
                {contract.hot_kv_blocks, "hot KV blocks"},
            }};
        for (const auto& [value, name] : positive_fields) {
            if (value == 0) {
                throw std::runtime_error(
                    std::string("explicit residency contract ") + name +
                    " must be positive");
            }
        }
        if (contract.page_size_bytes != page_size) {
            throw std::runtime_error(
                "explicit residency contract page size differs from the "
                "selected backing tier");
        }
        if (contract.kv_region_begin % page_size != 0) {
            throw std::runtime_error(
                "explicit residency contract KV arena must begin on a page");
        }
        if (contract.hot_kv_blocks > contract.logical_kv_blocks) {
            throw std::runtime_error(
                "explicit residency contract hot KV exceeds logical KV");
        }
        if (contract.static_weight_resident_pages >
            contract.immutable_weight_pages) {
            throw std::runtime_error(
                "explicit residency contract static weights exceed the "
                "immutable weight population");
        }
        if (contract.kv_region_begin / page_size !=
            contract.immutable_weight_pages) {
            throw std::runtime_error(
                "explicit residency contract immutable weights must occupy "
                "the complete page prefix before the KV arena");
        }
        const auto logical_kv_bytes = checked_mul(
            contract.logical_kv_blocks,
            contract.kv_block_stride_bytes,
            "explicit logical KV bytes");
        const auto hot_kv_bytes = checked_mul(
            contract.hot_kv_blocks,
            contract.kv_block_stride_bytes,
            "explicit hot KV bytes");
        if (contract.hot_kv_blocks < contract.logical_kv_blocks &&
            hot_kv_bytes % page_size != 0) {
            throw std::runtime_error(
                "explicit residency contract hot/cold KV boundary splits a "
                "physical page");
        }
        const auto exact_population = checked_add(
            contract.immutable_weight_bytes,
            checked_add(
                contract.runtime_overhead_bytes,
                checked_add(
                    contract.block_table_bytes,
                    logical_kv_bytes,
                    "explicit block table and logical KV bytes"),
                "explicit overhead, block table, and logical KV bytes"),
            "explicit unique resident population");
        if (exact_population !=
            contract.unique_resident_footprint_bytes) {
            throw std::runtime_error(
                "explicit residency contract unique footprint is not "
                "byte-conservative");
        }
        if (contract.immutable_weight_pages <
            ceil_div(
                contract.immutable_weight_bytes,
                page_size,
                "immutable weight page lower bound")) {
            throw std::runtime_error(
                "explicit residency contract immutable weight page allocation "
                "is smaller than its logical bytes");
        }
        if (contract.active_buffer_bytes_per_slot >
            config_.layer_buffer_bytes) {
            throw std::runtime_error(
                "explicit residency contract active buffer exceeds "
                "layer-buffer-bytes");
        }
        const auto metadata_pages = checked_add(
            ceil_div(
                contract.runtime_overhead_bytes,
                page_size,
                "runtime overhead pages"),
            ceil_div(
                contract.block_table_bytes,
                page_size,
                "block table pages"),
            "explicit metadata pages");
        const auto hot_kv_pages = ceil_div(
            hot_kv_bytes, page_size, "explicit hot KV pages");
        const auto active_buffer_pages = ceil_div(
            contract.active_buffer_bytes_per_slot,
            page_size,
            "explicit active buffer pages");
        const auto runtime_hbm_pages = checked_add(
            metadata_pages,
            checked_add(
                checked_add(
                    contract.static_weight_resident_pages,
                    hot_kv_pages,
                    "explicit static weights and hot KV"),
                checked_mul(
                    active_buffer_pages,
                    2,
                    "two explicit active buffers"),
                "explicit resident data and active buffers"),
            "explicit runtime HBM pages");
        if (runtime_hbm_pages > config_.hbm.capacity_bytes / page_size) {
            throw std::runtime_error(
                "explicit residency contract page allocation exceeds HBM");
        }
    }
    if (config_.backing == BackingTier::Hbf) {
        base_die_links_.reserve(config_.hbf.stacks);
        for (std::uint32_t stack = 0; stack < config_.hbf.stacks; ++stack) {
            base_die_links_.emplace_back(
                config_.base_die_link,
                "layer_stream/base_die_link/stack" + std::to_string(stack));
        }
    }
}

std::uint64_t LayerStreamingComposition::backing_page_size() const {
    return config_.backing == BackingTier::Hbf ?
        config_.hbf.page_size_bytes :
        config_.external_backing.page_size_bytes;
}

hbf::HbfDevice& LayerStreamingComposition::hbf_device() {
    if (!hbf_) {
        throw std::runtime_error("layer streamer has no HBF backing device");
    }
    return *hbf_;
}

const hbf::HbfDevice& LayerStreamingComposition::hbf_device() const {
    if (!hbf_) {
        throw std::runtime_error("layer streamer has no HBF backing device");
    }
    return *hbf_;
}

external::ExternalBackingDevice&
LayerStreamingComposition::external_device() {
    if (!external_) {
        throw std::runtime_error(
            "layer streamer has no external-backing backing device");
    }
    return *external_;
}

BaseDieLink& LayerStreamingComposition::base_die_link_for_stack(std::size_t stack) {
    return base_die_links_.at(stack);
}

BaseDieLinkStats LayerStreamingComposition::aggregate_base_die_link_stats() const {
    BaseDieLinkStats total;
    total.links = base_die_links_.size();
    for (const auto& link : base_die_links_) {
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
        total.first_arrival_ns = std::min(total.first_arrival_ns, stats.first_arrival_ns);
        total.finish_ns = std::max(total.finish_ns, stats.finish_ns);
    }
    return total;
}

void LayerStreamingComposition::absorb_completion(
    LayerStreamingRunResult& result,
    PhysicalCompletion completion) {
    result.finish_ns = std::max(result.finish_ns, completion.finish_ns);
    if (completion.finish_ns < completion.arrival_ns) {
        result.warnings.push_back(completion.id + " finished before arrival");
    }
    if (completion.note.find("unmapped") != std::string::npos) {
        result.warnings.push_back(completion.id + " produced " + completion.note);
    }
    if (config_.trace.mode != TraceMode::Off) {
        result.completions.push_back(std::move(completion));
    }
}

void LayerStreamingComposition::absorb_link_completion(
    LayerStreamingRunResult& result,
    PhysicalCompletion completion) {
    result.finish_ns = std::max(result.finish_ns, completion.finish_ns);
    if (completion.finish_ns < completion.arrival_ns) {
        result.warnings.push_back(completion.id + " finished before arrival");
    }
    if (config_.trace.mode != TraceMode::Off) {
        result.completions.push_back(std::move(completion));
    }
}

class LayerStreamingComposition::RunEngine {
public:
    RunEngine(
        LayerStreamingComposition& composition,
        const std::vector<MemoryRequest>& requests,
        std::uint64_t footprint_bytes)
        : sim_(composition),
          requests_(requests) {
        result_.service_latencies_ns.assign(
            requests.size(), std::numeric_limits<double>::quiet_NaN());
        result_.offered_latencies_ns.assign(
            requests.size(), std::numeric_limits<double>::quiet_NaN());
        result_.source_latencies_ns.assign(
            requests.size(), std::numeric_limits<double>::quiet_NaN());
        result_.streaming_stats.address_footprint_bytes = footprint_bytes;
        result_.streaming_stats.hbm_capacity_bytes =
            sim_.config_.hbm.capacity_bytes;
        result_.streaming_stats.backing = sim_.config_.backing;
        result_.streaming_stats.backing_request_credit_limit =
            sim_.config_.max_outstanding_requests;
        build_layers();
    }

    [[nodiscard]] bool compact_resident_mapping() const {
        return result_.streaming_stats.compact_resident_mapping;
    }

    [[nodiscard]] std::uint64_t resident_physical_bytes() const {
        return result_.streaming_stats.resident_physical_bytes;
    }

    [[nodiscard]] std::uint64_t effective_layer_buffer_bytes() const {
        return result_.streaming_stats.effective_layer_buffer_bytes;
    }

    LayerStreamingRunResult run() {
        prepare_initial_image();
        if (!requests_.empty()) {
            schedule_admission_wake(requests_.front().arrival_ns);
        }
        while (!events_.empty()) {
            auto event = events_.top();
            events_.pop();
            now_ns_ = event.time_ns;
            event.action();
        }
        if (completed_user_ops_ != requests_.size()) {
            throw std::runtime_error(
                "layer streamer quiesced before every user request completed");
        }
        if (!requests_.empty() &&
            sim_.config_.backing == BackingTier::Hbf) {
            auto drain = sim_.hbf_device().drain_pending(
                "layer-stream/drain", now_ns_, sim_.config_.trace);
            // A drain is a causal barrier, not another HBF request. Its media
            // work remains visible in HbfStats and its diagnostic completion,
            // but it must not inflate the composition access census.
            sim_.absorb_completion(result_, std::move(drain));
        }
        finalize_stats();
        result_.hbm_stats = sim_.hbm_.stats();
        if (sim_.config_.backing == BackingTier::Hbf) {
            result_.hbf_stats = sim_.hbf_device().stats();
        } else {
            result_.external_backing_stats = sim_.external_device().stats();
        }
        result_.base_die_link_stats = sim_.aggregate_base_die_link_stats();
        result_.finish_ns = std::max({
            result_.finish_ns,
            result_.hbm_stats.finish_ns,
            result_.hbf_stats.finish_ns,
            result_.external_backing_stats.finish_ns,
            result_.base_die_link_stats.finish_ns,
            result_.user_finish_ns,
        });
        if (sim_.address_heatmap_) {
            result_.address_heatmap = sim_.address_heatmap_->snapshot();
        }
        return std::move(result_);
    }

private:
    struct Event {
        double time_ns = 0.0;
        std::uint64_t sequence = 0;
        std::function<void()> action;
    };

    struct EventLater {
        bool operator()(const Event& lhs, const Event& rhs) const {
            if (lhs.time_ns != rhs.time_ns) {
                return lhs.time_ns > rhs.time_ns;
            }
            return lhs.sequence > rhs.sequence;
        }
    };

    struct PagePlan {
        std::uint64_t source_page = 0;
        std::uint64_t slot = 0;
        bool needs_prefetch = true;
        // Immutable weights use the pre-resolved static extent. Mutable
        // context and writeable unknown data use the logical FTL from their
        // initial version onward.
        bool logical_backing = false;
        std::optional<std::size_t> prior_writer_layer;
    };

    struct PageAccessPlan {
        bool written = false;
        bool needs_prefetch = true;
    };

    struct LayerState {
        std::uint64_t external_id = 0;
        bool explicit_id = false;
        std::vector<std::size_t> requests;
        std::vector<std::uint64_t> data_pages;
        std::vector<PagePlan> pages;
        std::unordered_map<std::uint64_t, std::size_t> page_index;
        std::set<std::uint64_t> dirty_pages;
        std::size_t admitted_requests = 0;
        std::size_t completed_requests = 0;
        std::size_t remaining_installs = 0;
        std::size_t remaining_writebacks = 0;
        bool prefetch_started = false;
        bool data_ready = false;
        bool execution_started = false;
        bool foreground_finished = false;
        bool compute_finished = false;
        bool execution_finished = false;
        double compute_ns = 0.0;
        std::optional<double> declared_compute_ns;
        double first_admit_ns = std::numeric_limits<double>::infinity();
        double prefetch_start_ns = 0.0;
        double data_ready_ns = 0.0;
        double execution_start_ns = 0.0;
        double execution_finish_ns = 0.0;
    };

    struct WaitingPrefetch {
        std::size_t layer = 0;
        std::size_t page_index = 0;
    };

    struct WaitingBufferInstall {
        std::size_t layer = 0;
        std::size_t page_index = 0;
        double ready_ns = 0.0;
    };

    struct PendingUserTransaction {
        std::size_t request_index = 0;
        std::uint64_t page = 0;
        std::uint64_t offset = 0;
        std::uint64_t physical_addr = 0;
        std::uint64_t bytes = 0;
        bool resident = false;
    };

    enum class BackingTransactionKind {
        Read,
        Write,
    };

    struct PendingBackingTransaction {
        BackingTransactionKind kind = BackingTransactionKind::Read;
        std::size_t layer = 0;
        // Read transactions identify the layer page-plan index. Write
        // transactions identify the logical source page.
        std::uint64_t page = 0;
        double ready_ns = 0.0;
    };

    using WriterKey = std::pair<std::size_t, std::uint64_t>;

    LayerStreamingComposition& sim_;
    const std::vector<MemoryRequest>& requests_;
    LayerStreamingRunResult result_;
    std::vector<LayerState> layers_;
    std::vector<std::size_t> request_layer_;
    // Time at which the phase scheduler accepted the parent request. The
    // first physical page transaction may be admitted later after layer,
    // dependency, and transaction-window gates.
    std::vector<double> front_end_ready_ns_;
    std::vector<double> admitted_ns_;
    std::array<double, 2> buffer_reusable_ns_{0.0, 0.0};
    std::array<bool, 2> buffer_writeback_pending_{false, false};
    std::array<std::vector<WaitingBufferInstall>, 2> buffer_install_waiters_;
    std::unordered_map<std::uint64_t, std::uint64_t> resident_page_slots_;
    // The explicit placement compiler always chooses one contiguous logical
    // weight-page prefix. Keep that range implicit so a 70B placement does
    // not allocate tens of millions of hash-table entries merely to encode
    // slot = base + page.
    std::uint64_t static_weight_resident_pages_ = 0;
    std::uint64_t static_weight_resident_slot_base_ = 0;
    std::uint64_t effective_layer_buffer_pages_ = 0;
    std::map<WriterKey, std::vector<WaitingPrefetch>> backing_waiters_;
    std::map<WriterKey, std::vector<WriterKey>> writeback_waiters_;
    std::set<WriterKey> completed_writebacks_;
    std::vector<std::vector<std::size_t>> request_dependents_;
    std::vector<std::size_t> request_dependencies_remaining_;
    std::vector<double> request_dependency_ready_ns_;
    std::vector<bool> request_scheduled_;
    std::vector<std::size_t> request_transactions_remaining_;
    std::deque<PendingUserTransaction> pending_user_transactions_;
    std::size_t inflight_user_transactions_ = 0;
    std::deque<PendingBackingTransaction> pending_backing_transactions_;
    std::size_t inflight_backing_transactions_ = 0;
    std::priority_queue<Event, std::vector<Event>, EventLater> events_;
    std::uint64_t next_event_sequence_ = 0;
    double now_ns_ = 0.0;
    std::size_t next_request_to_admit_ = 0;
    std::size_t outstanding_parent_requests_ = 0;
    std::size_t completed_user_ops_ = 0;
    bool admission_wake_scheduled_ = false;
    std::optional<std::uint64_t> active_phase_;
    double active_phase_ready_ns_ = 0.0;
    // Reduce long streams in extended precision for the same reason as the
    // direct composition: this counter is later compared with independently
    // reduced per-request latency samples.
    long double front_end_admission_wait_work_ns_ = 0.0L;
    long double backing_admission_wait_work_ns_ = 0.0L;

    void schedule(double time_ns, std::function<void()> action) {
        if (!std::isfinite(time_ns) || time_ns < now_ns_) {
            throw std::runtime_error(
                "layer-stream event violated nondecreasing causal time");
        }
        events_.push(Event{
            .time_ns = time_ns,
            .sequence = next_event_sequence_++,
            .action = std::move(action),
        });
    }

    void build_layers() {
        request_layer_.resize(requests_.size());
        front_end_ready_ns_.assign(
            requests_.size(), std::numeric_limits<double>::quiet_NaN());
        admitted_ns_.assign(
            requests_.size(), std::numeric_limits<double>::quiet_NaN());
        request_dependents_.resize(requests_.size());
        request_dependencies_remaining_.assign(requests_.size(), 0);
        request_dependency_ready_ns_.assign(requests_.size(), 0.0);
        request_scheduled_.assign(requests_.size(), false);
        request_transactions_remaining_.assign(requests_.size(), 0);
        const bool any_explicit = std::any_of(
            requests_.begin(), requests_.end(),
            [](const MemoryRequest& request) { return request.layer.has_value(); });
        if (any_explicit && std::any_of(
                requests_.begin(), requests_.end(),
                [](const MemoryRequest& request) { return !request.layer.has_value(); })) {
            throw std::runtime_error(
                "layer-streaming traces must specify layer= on every request or none");
        }

        std::optional<std::uint64_t> previous_id;
        for (std::size_t i = 0; i < requests_.size(); ++i) {
            const auto external_id = requests_[i].layer.value_or(0);
            if (previous_id && external_id < *previous_id) {
                throw std::runtime_error(
                    "layer ids must be nondecreasing in trace order");
            }
            if (!previous_id || external_id != *previous_id) {
                layers_.push_back(LayerState{
                    .external_id = external_id,
                    .explicit_id = any_explicit,
                });
                previous_id = external_id;
            }
            request_layer_[i] = layers_.size() - 1;
            auto& layer = layers_.back();
            layer.requests.push_back(i);
            if (requests_[i].compute_ns) {
                if (layer.declared_compute_ns &&
                    *layer.declared_compute_ns != *requests_[i].compute_ns) {
                    throw std::runtime_error(
                        "layer " + std::to_string(layer.external_id) +
                        " has conflicting compute_ns declarations");
                }
                layer.declared_compute_ns = requests_[i].compute_ns;
                layer.compute_ns = *requests_[i].compute_ns;
            }
        }

        const auto page_size = sim_.backing_page_size();
        const auto hbm_capacity_pages =
            sim_.config_.hbm.capacity_bytes / page_size;
        const auto buffer_limit_pages =
            sim_.config_.layer_buffer_bytes / page_size;
        std::map<std::uint64_t, PageSemantics> page_semantics;
        std::vector<std::uint64_t> hot_kv_candidates;
        std::set<std::uint64_t> hbm_only_pages;
        std::set<std::uint64_t> all_data_pages;
        std::set<std::uint64_t> model_weight_pages;
        std::set<std::uint64_t> kv_pages;
        std::set<std::uint64_t> unknown_pages;
        std::vector<std::set<std::uint64_t>> raw_data_pages_by_layer(
            layers_.size());
        std::uint64_t raw_max_layer_data_pages = 0;
        std::unordered_set<std::uint64_t> hot_kv_resident_pages;
        if (sim_.config_.residency_contract) {
            const auto& contract = *sim_.config_.residency_contract;
            const auto logical_kv_bytes = checked_mul(
                contract.logical_kv_blocks,
                contract.kv_block_stride_bytes,
                "contract logical KV bytes");
            const auto kv_region_end = checked_add(
                contract.kv_region_begin,
                logical_kv_bytes,
                "contract KV region end");
            const auto kv_region_end_page = ceil_div(
                kv_region_end,
                page_size,
                "contract KV region end page");
            const auto hot_kv_end = checked_add(
                contract.kv_region_begin,
                checked_mul(
                    contract.hot_kv_blocks,
                    contract.kv_block_stride_bytes,
                    "contract hot KV bytes"),
                "contract hot KV end");
            for (const auto& request : requests_) {
                if (request.kind == SemanticKind::Unknown) {
                    throw std::runtime_error(
                        "explicit residency contract rejects unknown data pages");
                }
                result_.streaming_stats.semantic_inputs_consumed = true;
                const auto last_byte = checked_add(
                    request.addr,
                    request.bytes - 1,
                    "explicit-residency request end");
                const auto first_page = request.addr / page_size;
                const auto last_page = last_byte / page_size;
                switch (request.kind) {
                case SemanticKind::ModelWeights:
                    if (request.op == Op::Write) {
                        throw std::runtime_error(
                            "model_weights must be read-only under hybrid residency");
                    }
                    if (last_byte >= contract.kv_region_begin) {
                        throw std::runtime_error(
                            "model-weight trace page lies outside the explicit "
                            "immutable-weight prefix");
                    }
                    break;
                case SemanticKind::SharedContext:
                case SemanticKind::GeneratedContext:
                    if (request.addr < contract.kv_region_begin ||
                        last_byte >= kv_region_end) {
                        throw std::runtime_error(
                            "KV trace page lies outside the explicit residency "
                            "contract arena");
                    }
                    for (auto page = first_page;; ++page) {
                        if (checked_mul(
                                page,
                                page_size,
                                "explicit KV page base") < hot_kv_end) {
                            hot_kv_resident_pages.insert(page);
                        }
                        if (page == last_page) {
                            break;
                        }
                    }
                    break;
                case SemanticKind::Scratch:
                case SemanticKind::Metadata:
                    if (first_page < kv_region_end_page) {
                        throw std::runtime_error(
                            "HBM-only trace page aliases the explicit immutable "
                            "weight or KV population");
                    }
                    for (auto page = first_page;; ++page) {
                        hbm_only_pages.insert(page);
                        if (page == last_page) {
                            break;
                        }
                    }
                    break;
                case SemanticKind::Unknown:
                    break;
                }
            }
        } else {
            for (const auto& request : requests_) {
                if (request.kind != SemanticKind::Unknown) {
                    result_.streaming_stats.semantic_inputs_consumed = true;
                }
                if (request.kind == SemanticKind::ModelWeights &&
                    request.op == Op::Write) {
                    throw std::runtime_error(
                        "model_weights must be read-only under hybrid residency");
                }
                const auto last_byte = checked_add(
                    request.addr,
                    request.bytes - 1,
                    "hybrid-residency request end");
                const auto first_page = request.addr / page_size;
                const auto last_page = last_byte / page_size;
                for (auto page = first_page;; ++page) {
                    auto [entry, inserted] =
                        page_semantics.try_emplace(page);
                    auto& semantics = entry->second;
                    if (inserted) {
                        semantics.first_touch_shared_context =
                            request.kind == SemanticKind::SharedContext;
                        if (semantics.first_touch_shared_context) {
                            hot_kv_candidates.push_back(page);
                        }
                    }
                    switch (request.kind) {
                    case SemanticKind::Unknown:
                        semantics.unknown = true;
                        break;
                    case SemanticKind::ModelWeights:
                        semantics.model_weight = true;
                        break;
                    case SemanticKind::SharedContext:
                        semantics.shared_context = true;
                        break;
                    case SemanticKind::GeneratedContext:
                        semantics.generated_context = true;
                        break;
                    case SemanticKind::Scratch:
                    case SemanticKind::Metadata:
                        semantics.hbm_only = true;
                        break;
                    }
                    semantics.written =
                        semantics.written || request.op == Op::Write;
                    if (page == last_page) {
                        break;
                    }
                }
            }

            for (const auto& [page, semantics] : page_semantics) {
                const bool context =
                    semantics.shared_context || semantics.generated_context;
                const bool tiered =
                    semantics.model_weight || context || semantics.unknown;
                if (semantics.hbm_only && tiered) {
                    throw std::runtime_error(
                        "logical page " + std::to_string(page) +
                        " has conflicting HBM-only and tiered-data semantics");
                }
                if (semantics.model_weight && context) {
                    throw std::runtime_error(
                        "logical page " + std::to_string(page) +
                        " has conflicting model-weight and KV semantics");
                }
                if (semantics.model_weight && semantics.written) {
                    throw std::runtime_error(
                        "logical page " + std::to_string(page) +
                        " aliases a write with read-only model weights");
                }
                if (semantics.hbm_only) {
                    hbm_only_pages.insert(page);
                } else if (semantics.model_weight) {
                    model_weight_pages.insert(page);
                    all_data_pages.insert(page);
                } else if (context) {
                    kv_pages.insert(page);
                    all_data_pages.insert(page);
                } else {
                    unknown_pages.insert(page);
                    all_data_pages.insert(page);
                }
            }

            for (std::size_t layer_index = 0;
                 layer_index < layers_.size(); ++layer_index) {
                auto& raw_pages = raw_data_pages_by_layer[layer_index];
                for (const auto request_index :
                     layers_[layer_index].requests) {
                    const auto& request = requests_[request_index];
                    if (is_hbm_only(request.kind)) {
                        continue;
                    }
                    const auto last_byte = checked_add(
                        request.addr,
                        request.bytes - 1,
                        "unfiltered layer request end");
                    const auto first_page = request.addr / page_size;
                    const auto last_page = last_byte / page_size;
                    for (auto page = first_page;; ++page) {
                        raw_pages.insert(page);
                        if (page == last_page) {
                            break;
                        }
                    }
                }
                raw_max_layer_data_pages = std::max<std::uint64_t>(
                    raw_max_layer_data_pages, raw_pages.size());
            }
        }

        auto& stats = result_.streaming_stats;
        if (sim_.config_.residency_contract) {
            const auto& contract = *sim_.config_.residency_contract;
            stats.explicit_residency_contract = true;
            const auto logical_kv_bytes = checked_mul(
                contract.logical_kv_blocks,
                contract.kv_block_stride_bytes,
                "contract logical KV bytes");
            const auto hot_kv_bytes = checked_mul(
                contract.hot_kv_blocks,
                contract.kv_block_stride_bytes,
                "contract hot KV bytes");
            const auto logical_kv_pages = ceil_div(
                logical_kv_bytes,
                page_size,
                "contract logical KV pages");
            const auto hot_kv_pages = ceil_div(
                hot_kv_bytes,
                page_size,
                "contract hot KV pages");
            const auto cold_kv_pages =
                logical_kv_pages - hot_kv_pages;
            const auto kv_region_begin_page =
                contract.kv_region_begin / page_size;

            const auto metadata_pages = checked_add(
                ceil_div(
                    contract.runtime_overhead_bytes,
                    page_size,
                    "contract runtime overhead pages"),
                ceil_div(
                    contract.block_table_bytes,
                    page_size,
                    "contract block table pages"),
                "contract HBM-only pages");
            if (hbm_only_pages.size() > metadata_pages) {
                throw std::runtime_error(
                    "trace-visible HBM-only pages exceed the explicit "
                    "metadata reservation");
            }
            effective_layer_buffer_pages_ = ceil_div(
                contract.active_buffer_bytes_per_slot,
                page_size,
                "contract active buffer pages");

            std::uint64_t next_metadata_slot = 0;
            for (const auto page : hbm_only_pages) {
                resident_page_slots_.emplace(
                    page, next_metadata_slot++);
            }
            static_weight_resident_pages_ =
                contract.static_weight_resident_pages;
            static_weight_resident_slot_base_ = metadata_pages;
            for (const auto page : hot_kv_resident_pages) {
                resident_page_slots_.emplace(
                    page,
                    checked_add(
                        checked_add(
                            metadata_pages,
                            contract.static_weight_resident_pages,
                            "contract resident data prefix"),
                        page - kv_region_begin_page,
                        "contract hot KV resident slot"));
            }

            stats.hbm_only_resident_pages = metadata_pages;
            stats.hot_kv_candidate_pages = logical_kv_pages;
            stats.hot_kv_resident_pages = hot_kv_pages;
            stats.data_pages = checked_add(
                contract.immutable_weight_pages,
                logical_kv_pages,
                "contract data pages");
            stats.model_weight_resident_pages =
                contract.static_weight_resident_pages;
            stats.model_weight_backing_pages =
                contract.immutable_weight_pages -
                contract.static_weight_resident_pages;
            stats.cold_kv_backing_pages = cold_kv_pages;
            stats.unknown_backing_pages = 0;
            stats.capacity_pressure_basis_bytes =
                contract.unique_resident_footprint_bytes;
            stats.immutable_weight_logical_bytes =
                contract.immutable_weight_bytes;
            stats.runtime_overhead_logical_bytes =
                contract.runtime_overhead_bytes;
            stats.block_table_logical_bytes =
                contract.block_table_bytes;
            stats.active_buffer_logical_bytes_per_slot =
                contract.active_buffer_bytes_per_slot;
            stats.residency_page_size_bytes =
                contract.page_size_bytes;
            stats.kv_block_stride_bytes =
                contract.kv_block_stride_bytes;
            stats.logical_kv_blocks = contract.logical_kv_blocks;
            stats.hot_kv_blocks = contract.hot_kv_blocks;
            stats.cold_kv_blocks =
                contract.logical_kv_blocks - contract.hot_kv_blocks;
        } else {
            if (hbm_only_pages.size() > hbm_capacity_pages) {
                throw std::runtime_error(
                    "HBM-only objects exceed configured HBM capacity");
            }
            const auto unfiltered_runtime_hbm_pages = checked_add(
                hbm_only_pages.size(),
                checked_mul(
                    raw_max_layer_data_pages,
                    2,
                    "two unfiltered active-layer buffers"),
                "unfiltered hybrid-residency HBM page demand");
            if (unfiltered_runtime_hbm_pages > hbm_capacity_pages) {
                throw std::runtime_error(
                    "fixed HBM residency plus two unfiltered active-layer "
                    "buffers exceed configured HBM capacity");
            }

            // Generic microbenchmarks have no object map. Their deterministic
            // fallback selects only a prefix of first-touch shared-context
            // pages and never consults future reuse frequency.
            std::size_t selected_hot_kv_pages = std::min<std::size_t>(
                hot_kv_candidates.size(),
                static_cast<std::size_t>(
                    hbm_capacity_pages - unfiltered_runtime_hbm_pages));
            while (true) {
                hot_kv_resident_pages.clear();
                hot_kv_resident_pages.reserve(selected_hot_kv_pages);
                for (std::size_t i = 0; i < selected_hot_kv_pages; ++i) {
                    hot_kv_resident_pages.insert(hot_kv_candidates[i]);
                }

                effective_layer_buffer_pages_ = 0;
                for (const auto& raw_pages : raw_data_pages_by_layer) {
                    std::uint64_t backing_pages = 0;
                    for (const auto page : raw_pages) {
                        if (!hot_kv_resident_pages.contains(page)) {
                            ++backing_pages;
                        }
                    }
                    effective_layer_buffer_pages_ = std::max(
                        effective_layer_buffer_pages_, backing_pages);
                }

                const auto fixed_and_buffers = checked_add(
                    hbm_only_pages.size(),
                    checked_mul(
                        effective_layer_buffer_pages_,
                        2,
                        "two effective hybrid-residency buffers"),
                    "fixed and buffered hybrid-residency pages");
                if (fixed_and_buffers > hbm_capacity_pages) {
                    throw std::runtime_error(
                        "hybrid-residency buffer selection exceeds HBM "
                        "capacity");
                }
                const auto next_selected = std::min<std::uint64_t>(
                    hot_kv_candidates.size(),
                    hbm_capacity_pages - fixed_and_buffers);
                if (next_selected == selected_hot_kv_pages) {
                    break;
                }
                if (next_selected < selected_hot_kv_pages) {
                    throw std::runtime_error(
                        "hybrid-residency hot-KV selection was not monotonic");
                }
                selected_hot_kv_pages =
                    static_cast<std::size_t>(next_selected);
            }

            std::uint64_t resident_slot = 0;
            for (const auto page : hbm_only_pages) {
                resident_page_slots_.emplace(page, resident_slot++);
            }
            for (std::size_t i = 0; i < selected_hot_kv_pages; ++i) {
                resident_page_slots_.emplace(
                    hot_kv_candidates[i], resident_slot++);
            }

            stats.hbm_only_resident_pages = hbm_only_pages.size();
            stats.hot_kv_candidate_pages = hot_kv_candidates.size();
            stats.hot_kv_resident_pages = selected_hot_kv_pages;
            stats.data_pages = all_data_pages.size();
            stats.model_weight_backing_pages = model_weight_pages.size();
            stats.cold_kv_backing_pages =
                kv_pages.size() - selected_hot_kv_pages;
            stats.unknown_backing_pages = unknown_pages.size();
        }

        if (effective_layer_buffer_pages_ > buffer_limit_pages) {
            throw std::runtime_error(
                "a layer's backing data exceeds the layer-buffer-bytes limit");
        }

        stats.backing_unique_pages = checked_add(
            stats.model_weight_backing_pages,
            checked_add(
                stats.cold_kv_backing_pages,
                stats.unknown_backing_pages,
                "cold KV and unknown backing pages"),
            "unique backing pages");
        stats.unique_resident_footprint_pages = checked_add(
            stats.hbm_only_resident_pages,
            stats.data_pages,
            "unique hybrid-residency footprint pages");
        stats.resident_physical_pages = checked_add(
            stats.hbm_only_resident_pages,
            checked_add(
                stats.model_weight_resident_pages,
                stats.hot_kv_resident_pages,
                "resident model weights and hot KV"),
            "resident HBM pages");
        stats.effective_layer_buffer_pages =
            effective_layer_buffer_pages_;

        const auto pages_to_bytes = [page_size](
            std::uint64_t pages,
            const char* name) {
            return checked_mul(pages, page_size, name);
        };
        stats.hbm_only_resident_bytes = pages_to_bytes(
            stats.hbm_only_resident_pages, "HBM-only resident bytes");
        stats.hot_kv_candidate_bytes = pages_to_bytes(
            stats.hot_kv_candidate_pages, "hot-KV candidate bytes");
        stats.hot_kv_resident_bytes = pages_to_bytes(
            stats.hot_kv_resident_pages, "hot-KV resident bytes");
        stats.data_bytes = pages_to_bytes(
            stats.data_pages, "hybrid-residency data bytes");
        stats.model_weight_resident_bytes = pages_to_bytes(
            stats.model_weight_resident_pages,
            "model-weight resident bytes");
        stats.model_weight_backing_bytes = pages_to_bytes(
            stats.model_weight_backing_pages, "model-weight backing bytes");
        stats.cold_kv_backing_bytes = pages_to_bytes(
            stats.cold_kv_backing_pages, "cold-KV backing bytes");
        stats.unknown_backing_bytes = pages_to_bytes(
            stats.unknown_backing_pages, "unknown backing bytes");
        stats.backing_unique_bytes = pages_to_bytes(
            stats.backing_unique_pages, "unique backing bytes");
        if (sim_.config_.backing == BackingTier::External &&
            stats.backing_unique_bytes >
                sim_.config_.external_backing.capacity_bytes) {
            throw std::runtime_error(
                "complete immutable-weight and cold-KV backing population "
                "exceeds external backing capacity");
        }
        stats.unique_resident_footprint_bytes = pages_to_bytes(
            stats.unique_resident_footprint_pages,
            "unique resident footprint bytes");
        if (!stats.explicit_residency_contract) {
            stats.capacity_pressure_basis_bytes =
                stats.unique_resident_footprint_bytes;
        }
        if (stats.capacity_pressure_basis_bytes >
            stats.unique_resident_footprint_bytes) {
            throw std::runtime_error(
                "byte-exact footprint exceeds its physical page allocation");
        }
        stats.footprint_page_rounding_bytes =
            stats.unique_resident_footprint_bytes -
            stats.capacity_pressure_basis_bytes;
        stats.resident_physical_bytes = pages_to_bytes(
            stats.resident_physical_pages, "resident physical bytes");
        stats.effective_layer_buffer_bytes = pages_to_bytes(
            stats.effective_layer_buffer_pages,
            "effective layer-buffer bytes");
        stats.hbm_capacity_pressure =
            static_cast<double>(
                static_cast<long double>(
                    stats.capacity_pressure_basis_bytes) /
                static_cast<long double>(stats.hbm_capacity_bytes));

        const auto runtime_hbm_pages = checked_add(
            stats.resident_physical_pages,
            checked_mul(
                stats.effective_layer_buffer_pages,
                2,
                "two runtime hybrid-residency buffers"),
            "runtime hybrid-residency HBM pages");
        if (runtime_hbm_pages > hbm_capacity_pages) {
            throw std::runtime_error(
                "hybrid-residency runtime allocation exceeds HBM capacity");
        }
        stats.unused_hbm_pages = hbm_capacity_pages - runtime_hbm_pages;
        stats.unused_hbm_bytes = pages_to_bytes(
            stats.unused_hbm_pages, "unused HBM bytes");
        stats.compact_resident_mapping =
            stats.resident_physical_pages != 0;
        if (stats.data_pages != checked_add(
                checked_add(
                    stats.model_weight_resident_pages,
                    stats.hot_kv_resident_pages,
                    "resident data pages"),
                stats.backing_unique_pages,
                "resident and backing data pages")) {
            throw std::runtime_error(
                "hybrid-residency page partition is inconsistent");
        }

        const auto effective_streaming_bytes = checked_mul(
            stats.effective_layer_buffer_bytes,
            2,
            "two effective layer buffers");
        sim_.hbm_streaming_base_addr_ =
            sim_.config_.hbm.capacity_bytes - effective_streaming_bytes;
        if (stats.resident_physical_bytes >
            sim_.hbm_streaming_base_addr_) {
            throw std::runtime_error(
                "resident HBM allocation overlaps effective layer buffers");
        }

        // Accumulate first-touch and write properties only for pages that
        // actually use an active-layer buffer.
        std::vector<std::unordered_map<std::uint64_t, PageAccessPlan>>
            page_access_plans_by_layer(layers_.size());
        for (std::size_t layer_index = 0; layer_index < layers_.size();
             ++layer_index) {
            auto& layer = layers_[layer_index];
            std::set<std::uint64_t> layer_data_pages;
            auto& page_access_plans =
                page_access_plans_by_layer[layer_index];
            for (const auto request_index : layer.requests) {
                const auto& request = requests_[request_index];
                const auto last_byte = checked_add(
                    request.addr,
                    request.bytes - 1,
                    "layer residency request end");
                const auto request_end = checked_add(
                    request.addr, request.bytes, "layer request end");
                const auto first_page = request.addr / page_size;
                const auto last_page = last_byte / page_size;
                for (auto page = first_page;; ++page) {
                    if (!resident_slot_for(page)) {
                        layer_data_pages.insert(page);
                        auto [access, first_touch] =
                            page_access_plans.try_emplace(page);
                        if (first_touch) {
                            const auto page_base =
                                checked_mul(page, page_size, "page base");
                            const auto page_end = checked_add(
                                page_base, page_size, "page end");
                            const auto begin =
                                std::max(page_base, request.addr);
                            const auto end =
                                std::min(page_end, request_end);
                            access->second.needs_prefetch =
                                request.op == Op::Read ||
                                begin != page_base || end != page_end;
                        }
                        if (request.op == Op::Write) {
                            access->second.written = true;
                        }
                    }
                    if (page == last_page) {
                        break;
                    }
                }
            }
            layer.data_pages.assign(
                layer_data_pages.begin(), layer_data_pages.end());
            if (page_access_plans.size() != layer.data_pages.size()) {
                throw std::runtime_error(
                    "layer page-access plan population is inconsistent");
            }
        }

        std::unordered_map<std::uint64_t, std::size_t> last_writer_layer;
        for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
            auto& layer = layers_[layer_index];
            std::uint64_t slot = 0;
            for (const auto page : layer.data_pages) {
                if (slot >= effective_layer_buffer_pages_) {
                    throw std::runtime_error(
                        "layer " + std::to_string(layer.external_id) +
                        " exceeds its dynamically sized data buffer");
                }
                const auto access =
                    page_access_plans_by_layer[layer_index].find(page);
                if (access ==
                    page_access_plans_by_layer[layer_index].end()) {
                    throw std::runtime_error(
                        "layer page is missing its access plan");
                }
                PagePlan plan{
                    .source_page = page,
                    .slot = slot++,
                    .needs_prefetch = access->second.needs_prefetch,
                    .logical_backing = [&] {
                        if (sim_.config_.residency_contract) {
                            return page >=
                                sim_.config_.residency_contract->
                                    kv_region_begin / page_size;
                        }
                        const auto semantics = page_semantics.find(page);
                        if (semantics == page_semantics.end()) {
                            throw std::runtime_error(
                                "layer page lacks semantic classification");
                        }
                        return semantics->second.shared_context ||
                            semantics->second.generated_context ||
                            (semantics->second.unknown &&
                             semantics->second.written);
                    }(),
                };
                if (const auto found = last_writer_layer.find(page);
                    found != last_writer_layer.end()) {
                    plan.prior_writer_layer = found->second;
                }
                layer.page_index.emplace(page, layer.pages.size());
                layer.pages.push_back(plan);
                if (access->second.written) {
                    last_writer_layer[page] = layer_index;
                }
            }
            page_access_plans_by_layer[layer_index].clear();
            page_access_plans_by_layer[layer_index].rehash(0);
            if (layer.pages.size() > effective_layer_buffer_pages_) {
                throw std::runtime_error(
                    "layer data exceeds the effective layer buffer");
            }
            struct PageDependencyState {
                std::optional<std::size_t> last_writer;
                std::vector<std::size_t> readers;
            };
            std::unordered_map<std::uint64_t, PageDependencyState>
                dependencies_by_page;
            for (const auto request_index : layer.requests) {
                const auto& request = requests_[request_index];
                const auto last_byte = checked_add(
                    request.addr, request.bytes - 1, "dependency request end");
                const auto first_page = request.addr / page_size;
                const auto last_page = last_byte / page_size;
                std::set<std::size_t> dependencies;
                for (auto page = first_page;; ++page) {
                    auto& state = dependencies_by_page[page];
                    if (state.last_writer) {
                        dependencies.insert(*state.last_writer);
                    }
                    if (request.op == Op::Write) {
                        dependencies.insert(
                            state.readers.begin(), state.readers.end());
                        state.readers.clear();
                        state.last_writer = request_index;
                    } else {
                        state.readers.push_back(request_index);
                    }
                    if (page == last_page) {
                        break;
                    }
                }
                request_dependencies_remaining_[request_index] = dependencies.size();
                for (const auto dependency : dependencies) {
                    request_dependents_[dependency].push_back(request_index);
                }
            }
            result_.streaming_stats.max_layer_data_pages =
                std::max<std::uint64_t>(
                    result_.streaming_stats.max_layer_data_pages,
                    layer.pages.size());
            if (layer.declared_compute_ns) {
                ++result_.streaming_stats.explicit_compute_layers;
                result_.streaming_stats.compute_work_ns += layer.compute_ns;
                if (!std::isfinite(result_.streaming_stats.compute_work_ns)) {
                    throw std::runtime_error(
                        "layer compute work exceeds finite time");
                }
            }
        }
        result_.streaming_stats.layers = layers_.size();
        result_.streaming_stats.max_layer_data_bytes = checked_mul(
            result_.streaming_stats.max_layer_data_pages,
            page_size,
            "maximum layer data bytes");
        if ((!sim_.config_.residency_contract &&
             result_.streaming_stats.max_layer_data_pages !=
                 effective_layer_buffer_pages_) ||
            (sim_.config_.residency_contract &&
             result_.streaming_stats.max_layer_data_pages >
                 effective_layer_buffer_pages_)) {
            throw std::runtime_error(
                "effective layer buffer does not contain maximum layer data");
        }
        if (any_explicit) {
            result_.streaming_stats.explicit_layer_requests = requests_.size();
        }
    }

    [[nodiscard]] std::uint64_t mapping_pages_for_logical_range(
        std::uint64_t first_lpn,
        std::uint64_t page_count) const {
        if (page_count == 0) {
            return 0;
        }
        const auto& config = sim_.hbf_device().config();
        const auto group_width = checked_mul(
            config.stacks,
            config.mapping_entries_per_page,
            "HBF mapping group width");
        const auto range_end = checked_add(
            first_lpn,
            page_count,
            "HBF logical backing range end");
        const auto first_group = first_lpn / group_width;
        const auto last_group = (range_end - 1) / group_width;
        const auto distinct_vpns = [&config](
            std::uint64_t segment_pages) {
            return std::min<std::uint64_t>(
                config.stacks, segment_pages);
        };
        if (first_group == last_group) {
            return distinct_vpns(page_count);
        }
        const auto first_group_end = checked_mul(
            first_group + 1,
            group_width,
            "HBF first mapping group end");
        const auto first_pages = first_group_end - first_lpn;
        const auto last_group_begin = checked_mul(
            last_group,
            group_width,
            "HBF last mapping group begin");
        const auto last_pages = range_end - last_group_begin;
        const auto middle_groups = last_group - first_group - 1;
        return checked_add(
            distinct_vpns(first_pages),
            checked_add(
                checked_mul(
                    middle_groups,
                    config.stacks,
                    "HBF middle mapping pages"),
                distinct_vpns(last_pages),
                "HBF middle and last mapping pages"),
            "HBF logical backing mapping pages");
    }

    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t>
    reserve_complete_hbf_backing_population() {
        if (!sim_.config_.residency_contract) {
            return {0, 0};
        }
        const auto& contract = *sim_.config_.residency_contract;
        auto& device = sim_.hbf_device();
        const auto& config = device.config();
        const auto total_pages = hbf_page_capacity(device);
        const auto planes = checked_mul(
            checked_mul(
                config.stacks,
                config.channels_per_stack,
                "HBF plane count"),
            checked_mul(
                config.dies_per_channel,
                config.planes_per_die,
                "HBF planes per channel"),
            "HBF plane count");
        const auto pages_per_global_block = checked_mul(
            planes,
            config.pages_per_block,
            "HBF pages per global block coordinate");
        const auto weight_begin_page =
            contract.static_weight_resident_pages;
        const auto weight_end_page = contract.immutable_weight_pages;
        const auto first_complete_block_coordinate = ceil_div(
            weight_begin_page,
            pages_per_global_block,
            "first complete immutable-weight block coordinate");
        const auto last_complete_block_coordinate =
            weight_end_page / pages_per_global_block;
        if (first_complete_block_coordinate >
                std::numeric_limits<std::uint32_t>::max() ||
            last_complete_block_coordinate >
            std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "immutable weight block range exceeds uint32 geometry");
        }

        std::set<std::size_t> edge_blocks;
        const auto first_complete_page = checked_mul(
            first_complete_block_coordinate,
            pages_per_global_block,
            "first complete immutable-weight page");
        for (auto source_page = weight_begin_page;
             source_page < std::min(weight_end_page, first_complete_page);
             ++source_page) {
            const auto ppn =
                map_static_hbf_page_addr(device, source_page) /
                config.page_size_bytes;
            edge_blocks.insert(static_cast<std::size_t>(
                ppn / config.pages_per_block));
        }
        if (first_complete_block_coordinate <
            last_complete_block_coordinate) {
            device.reserve_static_physical_block_extent(
                static_cast<std::uint32_t>(
                    first_complete_block_coordinate),
                static_cast<std::uint32_t>(
                    last_complete_block_coordinate -
                    first_complete_block_coordinate));
        }
        const auto last_complete_page = checked_mul(
            last_complete_block_coordinate,
            pages_per_global_block,
            "last complete immutable-weight page");
        const auto tail_begin_page = std::max(
            std::min(weight_end_page, first_complete_page),
            last_complete_page);
        for (auto source_page = tail_begin_page;
             source_page < weight_end_page;
             ++source_page) {
            const auto ppn =
                map_static_hbf_page_addr(device, source_page) /
                config.page_size_bytes;
            edge_blocks.insert(static_cast<std::size_t>(
                ppn / config.pages_per_block));
        }
        device.reserve_static_physical_block_indices(
            std::vector<std::size_t>(
                edge_blocks.begin(), edge_blocks.end()));

        const auto logical_kv_bytes = checked_mul(
            contract.logical_kv_blocks,
            contract.kv_block_stride_bytes,
            "complete logical KV backing bytes");
        const auto hot_kv_bytes = checked_mul(
            contract.hot_kv_blocks,
            contract.kv_block_stride_bytes,
            "complete hot KV bytes");
        const auto logical_kv_pages = ceil_div(
            logical_kv_bytes,
            config.page_size_bytes,
            "complete logical KV pages");
        const auto hot_kv_pages = ceil_div(
            hot_kv_bytes,
            config.page_size_bytes,
            "complete hot KV pages");
        const auto cold_kv_pages =
            logical_kv_pages - hot_kv_pages;
        const auto cold_kv_first_lpn = checked_add(
            contract.kv_region_begin / config.page_size_bytes,
            hot_kv_pages,
            "cold KV first LPN");
        const auto mapping_pages = mapping_pages_for_logical_range(
            cold_kv_first_lpn,
            cold_kv_pages);
        const auto gc_reserved_pages = checked_mul(
            checked_mul(
                planes,
                config.gc_reserved_free_blocks_per_plane,
                "HBF GC-reserved blocks"),
            config.pages_per_block,
            "HBF GC-reserved pages");
        const auto required_free_pages = checked_add(
            cold_kv_pages,
            checked_add(
                mapping_pages,
                gc_reserved_pages,
                "HBF mapping and GC-reserved pages"),
            "HBF cold backing physical pages");
        const auto free_pages = device.stats().free_pages;
        if (required_free_pages > free_pages ||
            contract.immutable_weight_pages > total_pages) {
            throw std::runtime_error(
                "complete immutable-weight and cold-KV backing population "
                "exceeds physical HBF capacity");
        }
        return {cold_kv_first_lpn, cold_kv_pages};
    }

    void prepare_initial_image() {
        if (sim_.config_.backing != BackingTier::Hbf) {
            return;
        }
        const auto [cold_kv_first_lpn, cold_kv_pages] =
            reserve_complete_hbf_backing_population();
        std::set<std::uint64_t> initial_ppns;
        std::set<std::uint64_t> initial_lpns;
        for (const auto& layer : layers_) {
            for (const auto& page : layer.pages) {
                if (page.needs_prefetch && !page.prior_writer_layer) {
                    if (page.logical_backing) {
                        initial_lpns.insert(page.source_page);
                    } else {
                        initial_ppns.insert(
                            map_static_hbf_page_addr(
                                sim_.hbf_device(), page.source_page) /
                            sim_.backing_page_size());
                    }
                }
            }
        }
        sim_.hbf_device().reserve_static_physical_pages(
            std::vector<std::uint64_t>(initial_ppns.begin(), initial_ppns.end()));
        if (sim_.config_.residency_contract) {
            const auto cold_kv_end_lpn = checked_add(
                cold_kv_first_lpn,
                cold_kv_pages,
                "complete cold KV LPN range end");
            if (std::any_of(
                    initial_lpns.begin(),
                    initial_lpns.end(),
                    [cold_kv_first_lpn, cold_kv_end_lpn](
                        std::uint64_t lpn) {
                        return lpn < cold_kv_first_lpn ||
                            lpn >= cold_kv_end_lpn;
                    })) {
                throw std::runtime_error(
                    "trace-visible logical backing page lies outside the "
                    "complete cold-KV initial population");
            }
            sim_.hbf_device().prepopulate_mutable_logical_page_range(
                cold_kv_first_lpn,
                cold_kv_pages);
        } else {
            sim_.hbf_device().prepopulate_logical_pages(
                std::vector<std::uint64_t>(
                    initial_lpns.begin(), initial_lpns.end()));
        }
    }

    void schedule_admission_wake(double time_ns) {
        if (admission_wake_scheduled_) {
            return;
        }
        admission_wake_scheduled_ = true;
        schedule(time_ns, [this] {
            admission_wake_scheduled_ = false;
            try_admit(now_ns_);
        });
    }

    void try_admit(double now_ns) {
        while (next_request_to_admit_ < requests_.size()) {
            const auto request_index = next_request_to_admit_;
            const auto& request = requests_[request_index];
            if (request.phase) {
                if (!active_phase_) {
                    active_phase_ = request.phase;
                } else if (*request.phase != *active_phase_) {
                    // Trace order is phase-monotonic. Reaching this boundary
                    // means every request in the old phase has been admitted;
                    // wait for their completion frontier before offering the
                    // first request in the next phase.
                    if (outstanding_parent_requests_ != 0) {
                        return;
                    }
                    active_phase_ = request.phase;
                    active_phase_ready_ns_ = now_ns;
                    ++result_.phase_barriers;
                }
            }
            const auto offered = std::max(
                request.arrival_ns,
                request.phase ? active_phase_ready_ns_ : 0.0);
            if (offered > now_ns) {
                schedule_admission_wake(offered);
                return;
            }
            ++next_request_to_admit_;
            ++outstanding_parent_requests_;
            result_.first_offered_arrival_ns = std::min(
                result_.first_offered_arrival_ns, offered);
            result_.last_offered_arrival_ns = std::max(
                result_.last_offered_arrival_ns, offered);
            const auto phase_wait = offered - request.arrival_ns;
            if (phase_wait > 0.0) {
                ++result_.phase_dependency_waited_ops;
                result_.phase_dependency_wait_work_ns += phase_wait;
                if (!std::isfinite(result_.phase_dependency_wait_work_ns)) {
                    throw std::runtime_error(
                        "layer-stream phase-dependency work exceeds finite time");
                }
                result_.phase_dependency_max_wait_ns = std::max(
                    result_.phase_dependency_max_wait_ns, phase_wait);
            }
            admit_request(request_index, now_ns, offered);
        }
    }

    void admit_request(
        std::size_t request_index,
        double admitted_ns,
        double offered_ns) {
        if (admitted_ns != offered_ns) {
            throw std::runtime_error(
                "layer-stream parent scheduler delayed an offered request");
        }
        front_end_ready_ns_[request_index] = admitted_ns;
        auto& layer = layers_[request_layer_[request_index]];
        const bool execution_was_started = layer.execution_started;
        ++layer.admitted_requests;
        layer.first_admit_ns = std::min(layer.first_admit_ns, admitted_ns);
        if (request_layer_[request_index] == 0 && !layer.prefetch_started) {
            start_prefetch(0, admitted_ns);
        }
        if (execution_was_started) {
            schedule_user_request(request_index, admitted_ns);
        } else {
            maybe_start_layer(request_layer_[request_index], admitted_ns);
        }
    }

    void start_prefetch(std::size_t layer_index, double trigger_ns) {
        if (layer_index >= layers_.size()) {
            return;
        }
        auto& layer = layers_[layer_index];
        if (layer.prefetch_started) {
            return;
        }
        layer.prefetch_started = true;
        layer.prefetch_start_ns = trigger_ns;
        for (std::size_t page_index = 0; page_index < layer.pages.size(); ++page_index) {
            const auto& page = layer.pages[page_index];
            if (!page.needs_prefetch) {
                continue;
            }
            ++layer.remaining_installs;
            if (page.prior_writer_layer &&
                !completed_writebacks_.contains(
                    WriterKey{*page.prior_writer_layer, page.source_page})) {
                backing_waiters_[WriterKey{
                    *page.prior_writer_layer, page.source_page}].push_back(
                        WaitingPrefetch{layer_index, page_index});
            } else {
                schedule_backing_read(layer_index, page_index, trigger_ns);
            }
        }
        if (layer.remaining_installs == 0) {
            layer.data_ready = true;
            layer.data_ready_ns = trigger_ns;
            maybe_start_layer(layer_index, trigger_ns);
        }
    }

    void schedule_backing_read(
        std::size_t layer_index,
        std::size_t page_index,
        double arrival_ns) {
        schedule(arrival_ns, [this, layer_index, page_index] {
            ++result_.streaming_stats.streamed_pages;
            result_.streaming_stats.streamed_bytes = checked_add(
                result_.streaming_stats.streamed_bytes,
                sim_.backing_page_size(),
                "streamed bytes");
            pending_backing_transactions_.push_back(PendingBackingTransaction{
                .kind = BackingTransactionKind::Read,
                .layer = layer_index,
                .page = page_index,
                .ready_ns = now_ns_,
            });
            pump_backing_transactions();
        });
    }

    [[nodiscard]] bool backing_credit_available() const {
        const auto limit = sim_.config_.max_outstanding_requests;
        return limit == 0 || inflight_backing_transactions_ < limit;
    }

    void record_backing_admission(double ready_ns) {
        const auto wait_ns = now_ns_ - ready_ns;
        if (!std::isfinite(wait_ns) || wait_ns < 0.0) {
            throw std::runtime_error(
                "layer-stream backing admission wait is invalid");
        }
        if (wait_ns == 0.0) {
            return;
        }
        ++result_.streaming_stats.backing_admission_waited_requests;
        backing_admission_wait_work_ns_ +=
            static_cast<long double>(wait_ns);
        if (!std::isfinite(backing_admission_wait_work_ns_) ||
            backing_admission_wait_work_ns_ >
                static_cast<long double>(
                    std::numeric_limits<double>::max())) {
            throw std::runtime_error(
                "layer-stream backing admission work exceeds finite time");
        }
        result_.streaming_stats.backing_admission_max_wait_ns = std::max(
            result_.streaming_stats.backing_admission_max_wait_ns,
            wait_ns);
    }

    void complete_backing_transaction() {
        if (inflight_backing_transactions_ == 0) {
            throw std::runtime_error(
                "layer-stream backing credit accounting underflow");
        }
        --inflight_backing_transactions_;
        pump_backing_transactions();
    }

    void issue_backing_read(
        std::size_t layer_index,
        std::size_t page_index) {
        auto& layer = layers_[layer_index];
        const auto& page = layer.pages[page_index];
        const auto page_size = sim_.backing_page_size();
        if (sim_.config_.backing == BackingTier::Hbf) {
            // Immutable weights use a pre-resolved striped physical extent.
            // KV and other mutable data use the logical FTL from their
            // initial version onward; a prior write therefore never falls
            // back to stale static media.
            const bool logical_backing =
                page.logical_backing ||
                page.prior_writer_layer.has_value();
            const auto source_addr = logical_backing ?
                checked_mul(
                    page.source_page,
                    page_size,
                    "HBF layer source address") :
                map_static_hbf_page_addr(
                    sim_.hbf_device(), page.source_page);
            auto completion = sim_.hbf_device().issue(physical_request(
                "layer" + std::to_string(layer.external_id) + "/page" +
                    std::to_string(page.source_page) + "/hbf-read",
                Tier::HBF,
                Op::Read,
                now_ns_,
                source_addr,
                page_size,
                logical_backing ?
                    AddressSpace::Logical : AddressSpace::Physical,
                sim_.config_.trace,
                HeatmapTrafficSource::PrefetchFill));
            const auto finish = completion.finish_ns;
            ++result_.hbf_background_accesses;
            sim_.absorb_completion(result_, std::move(completion));
            auto& backing_read_bytes = logical_backing ?
                result_.hbf_logical_read_bytes :
                result_.hbf_static_read_bytes;
            backing_read_bytes = checked_add(
                backing_read_bytes,
                page_size,
                logical_backing ?
                    "HBF logical read bytes" : "HBF static read bytes");
            schedule(finish, [this, layer_index, page_index] {
                issue_d2d_read(layer_index, page_index);
                complete_backing_transaction();
            });
            return;
        }

        auto completion = sim_.external_device().issue(physical_request(
            "layer" + std::to_string(layer.external_id) + "/page" +
                std::to_string(page.source_page) + "/external-read",
            Tier::External,
            Op::Read,
            now_ns_,
            checked_mul(
                page.source_page,
                page_size,
                "external layer source address"),
            page_size,
            AddressSpace::Logical,
            sim_.config_.trace,
            HeatmapTrafficSource::PrefetchFill));
        const auto finish = completion.finish_ns;
        ++result_.external_background_accesses;
        result_.external_backing_read_bytes = checked_add(
            result_.external_backing_read_bytes,
            page_size,
            "external backing read bytes");
        sim_.absorb_completion(result_, std::move(completion));
        schedule(finish, [this, layer_index, page_index] {
            queue_hbm_install(layer_index, page_index, now_ns_);
            complete_backing_transaction();
        });
    }

    void pump_backing_transactions() {
        while (!pending_backing_transactions_.empty() &&
               backing_credit_available()) {
            const auto transaction = pending_backing_transactions_.front();
            pending_backing_transactions_.pop_front();
            record_backing_admission(transaction.ready_ns);
            ++inflight_backing_transactions_;
            result_.streaming_stats.backing_max_inflight_requests = std::max(
                result_.streaming_stats.backing_max_inflight_requests,
                static_cast<std::uint64_t>(
                    inflight_backing_transactions_));
            if (transaction.kind == BackingTransactionKind::Read) {
                issue_backing_read(
                    transaction.layer,
                    static_cast<std::size_t>(transaction.page));
            } else {
                issue_admitted_backing_write(
                    transaction.layer, transaction.page);
            }
        }
    }

    void issue_d2d_read(std::size_t layer_index, std::size_t page_index) {
        const auto& layer = layers_[layer_index];
        const auto& page = layer.pages[page_index];
        const auto page_size = sim_.backing_page_size();
        const auto stack = page.prior_writer_layer ?
            sim_.hbf_device().stack_for_logical_page(page.source_page) :
            static_cast<std::size_t>(sim_.hbf_device().decode(
                map_static_hbf_page_addr(
                    sim_.hbf_device(), page.source_page)).stack);
        auto completion = sim_.base_die_link_for_stack(stack).issue(
            "layer" + std::to_string(layer.external_id) + "/page" +
                std::to_string(page.source_page) + "/d2d-read",
            Op::Read,
            now_ns_,
            page_size,
            sim_.config_.trace);
        const auto finish = completion.finish_ns;
        sim_.absorb_link_completion(result_, std::move(completion));
        queue_hbm_install(layer_index, page_index, finish);
    }

    void queue_hbm_install(
        std::size_t layer_index,
        std::size_t page_index,
        double backing_finish_ns) {
        const auto parity = layer_index % 2;
        if (buffer_writeback_pending_[parity]) {
            buffer_install_waiters_[parity].push_back(WaitingBufferInstall{
                .layer = layer_index,
                .page_index = page_index,
                .ready_ns = backing_finish_ns,
            });
            return;
        }
        const auto install_ns = std::max(
            backing_finish_ns, buffer_reusable_ns_[parity]);
        result_.streaming_stats.buffer_reuse_wait_work_ns +=
            install_ns - backing_finish_ns;
        schedule(install_ns, [this, layer_index, page_index] {
            issue_hbm_install(layer_index, page_index);
        });
    }

    std::uint64_t slot_addr(std::size_t layer_index, std::uint64_t slot) const {
        const auto parity = layer_index % 2;
        return checked_add(
            checked_add(
                sim_.hbm_streaming_base_addr_,
                checked_mul(
                    parity,
                    result_.streaming_stats.effective_layer_buffer_bytes,
                    "layer buffer parity offset"),
                "layer buffer base"),
            checked_mul(
                slot,
                sim_.backing_page_size(),
                "layer buffer slot offset"),
            "layer buffer slot address");
    }

    void issue_hbm_install(std::size_t layer_index, std::size_t page_index) {
        auto& layer = layers_[layer_index];
        const auto& page = layer.pages[page_index];
        const auto page_size = sim_.backing_page_size();
        auto completion = sim_.hbm_.issue(physical_request(
            "layer" + std::to_string(layer.external_id) + "/page" +
                std::to_string(page.source_page) + "/hbm-install",
            Tier::HBM,
            Op::Write,
            now_ns_,
            slot_addr(layer_index, page.slot),
            page_size,
            AddressSpace::Physical,
            sim_.config_.trace,
            HeatmapTrafficSource::StreamingInstall));
        const auto finish = completion.finish_ns;
        ++result_.hbm_background_accesses;
        sim_.absorb_completion(result_, std::move(completion));
        result_.hbm_streaming_write_bytes = checked_add(
            result_.hbm_streaming_write_bytes,
            page_size,
            "HBM streaming write bytes");
        schedule(finish, [this, layer_index] {
            auto& completed_layer = layers_[layer_index];
            if (completed_layer.remaining_installs == 0) {
                throw std::runtime_error("layer install accounting underflow");
            }
            --completed_layer.remaining_installs;
            if (completed_layer.remaining_installs == 0) {
                completed_layer.data_ready = true;
                completed_layer.data_ready_ns = now_ns_;
                maybe_start_layer(layer_index, now_ns_);
            }
        });
    }

    void maybe_start_layer(std::size_t layer_index, double now_ns) {
        if (layer_index >= layers_.size()) {
            return;
        }
        auto& layer = layers_[layer_index];
        if (layer.execution_started || !layer.prefetch_started ||
            !layer.data_ready || layer.admitted_requests == 0) {
            return;
        }
        // A first-touch full-page overwrite has no install event to carry the
        // parity fence. Gate execution itself so every use of the HBM region,
        // not only prefetched pages, waits for the older writeback.
        if (buffer_writeback_pending_[layer_index % 2]) {
            return;
        }
        if (layer_index != 0 && !layers_[layer_index - 1].execution_finished) {
            return;
        }
        layer.execution_started = true;
        layer.execution_start_ns = std::max({
            now_ns, layer.data_ready_ns, layer.first_admit_ns,
        });
        layer.compute_finished = layer.compute_ns == 0.0;
        if (!layer.compute_finished) {
            const auto compute_finish =
                layer.execution_start_ns + layer.compute_ns;
            if (!std::isfinite(compute_finish)) {
                throw std::runtime_error(
                    "layer compute finish exceeds finite time");
            }
            schedule(compute_finish, [this, layer_index] {
                auto& completed_layer = layers_[layer_index];
                completed_layer.compute_finished = true;
                completed_layer.execution_finish_ns = std::max(
                    completed_layer.execution_finish_ns,
                    now_ns_);
                maybe_finish_layer(layer_index);
            });
        }
        for (const auto request_index : layer.requests) {
            if (std::isfinite(front_end_ready_ns_[request_index])) {
                schedule_user_request(request_index, layer.execution_start_ns);
            }
        }
        start_prefetch(layer_index + 1, layer.execution_start_ns);
    }

    void schedule_user_request(std::size_t request_index, double earliest_ns) {
        request_dependency_ready_ns_[request_index] = std::max(
            request_dependency_ready_ns_[request_index], earliest_ns);
        if (request_dependencies_remaining_[request_index] != 0 ||
            request_scheduled_[request_index]) {
            return;
        }
        request_scheduled_[request_index] = true;
        const auto issue_ns = std::max(
            request_dependency_ready_ns_[request_index],
            front_end_ready_ns_[request_index]);
        const auto wait_ns =
            issue_ns - front_end_ready_ns_[request_index];
        if (wait_ns > 0.0) {
            ++result_.streaming_stats.user_waited_ops;
            result_.streaming_stats.user_wait_work_ns += wait_ns;
            result_.streaming_stats.user_max_wait_ns = std::max(
                result_.streaming_stats.user_max_wait_ns, wait_ns);
        }
        schedule(issue_ns, [this, request_index] {
            issue_user_request(request_index);
        });
    }

    [[nodiscard]] std::optional<std::uint64_t> resident_slot_for(
        std::uint64_t logical_page) const {
        if (logical_page < static_weight_resident_pages_) {
            return checked_add(
                static_weight_resident_slot_base_,
                logical_page,
                "contract static weight resident slot");
        }
        const auto found = resident_page_slots_.find(logical_page);
        if (found == resident_page_slots_.end()) {
            return std::nullopt;
        }
        return found->second;
    }

    std::uint64_t resident_addr(
        std::uint64_t logical_page,
        std::uint64_t offset) const {
        const auto slot = resident_slot_for(logical_page);
        if (!slot) {
            throw std::runtime_error(
                "HBM-resident request lacks a compact physical slot");
        }
        return checked_add(
            checked_mul(
                *slot,
                sim_.backing_page_size(),
                "HBM-only resident slot address"),
            offset,
            "HBM-only resident byte address");
    }

    void issue_user_request(std::size_t request_index) {
        const auto& request = requests_[request_index];
        const auto layer_index = request_layer_[request_index];
        auto& layer = layers_[layer_index];
        if (request_transactions_remaining_[request_index] != 0) {
            throw std::runtime_error(
                "layer-stream user request was issued more than once");
        }
        const auto page_size = sim_.backing_page_size();
        auto remaining = request.bytes;
        auto source_addr = request.addr;
        std::size_t transaction_count = 0;
        while (remaining != 0) {
            const auto page = source_addr / page_size;
            const auto offset = source_addr % page_size;
            const auto bytes = std::min(remaining, page_size - offset);
            bool resident = true;
            std::uint64_t physical_addr = 0;
            const auto resident_slot = resident_slot_for(page);
            if (resident_slot) {
                physical_addr = resident_addr(page, offset);
            } else {
                resident = false;
                const auto found = layer.page_index.find(page);
                if (found == layer.page_index.end()) {
                    throw std::runtime_error(
                        "foreground page is neither HBM-only nor present in "
                        "the layer's backed buffer");
                }
                const auto& page_plan = layer.pages[found->second];
                physical_addr = checked_add(
                    slot_addr(layer_index, page_plan.slot),
                    offset,
                    "foreground layer slot address");
                if (request.op == Op::Write) {
                    layer.dirty_pages.insert(page);
                }
            }
            pending_user_transactions_.push_back(PendingUserTransaction{
                .request_index = request_index,
                .page = page,
                .offset = offset,
                .physical_addr = physical_addr,
                .bytes = bytes,
                .resident = resident,
            });
            if (resident) {
                ++result_.streaming_stats.foreground_resident_page_accesses;
            } else {
                ++result_.streaming_stats.foreground_buffer_page_accesses;
            }
            ++transaction_count;
            remaining -= bytes;
            if (remaining != 0) {
                source_addr = checked_add(
                    source_addr,
                    bytes,
                    "next tiered foreground page transaction");
            }
        }
        if (transaction_count == 0) {
            throw std::runtime_error(
                "layer-stream user request produced no page transaction");
        }
        request_transactions_remaining_[request_index] = transaction_count;
        result_.hbm_foreground_bytes = checked_add(
            result_.hbm_foreground_bytes,
            request.bytes,
            "HBM foreground bytes");
        pump_user_transactions();
    }

    void pump_user_transactions() {
        const auto limit = sim_.config_.max_outstanding_requests;
        while (!pending_user_transactions_.empty() &&
               (limit == 0 || inflight_user_transactions_ < limit)) {
            auto transaction = pending_user_transactions_.front();
            pending_user_transactions_.pop_front();
            const auto& request = requests_[transaction.request_index];
            if (!std::isfinite(admitted_ns_[transaction.request_index])) {
                const auto ready_ns =
                    front_end_ready_ns_[transaction.request_index];
                if (!std::isfinite(ready_ns) || now_ns_ < ready_ns) {
                    throw std::runtime_error(
                        "invalid first transaction admission time");
                }
                admitted_ns_[transaction.request_index] = now_ns_;
                const auto admission_wait = now_ns_ - ready_ns;
                if (admission_wait > 0.0) {
                    ++result_.front_end_admission_waited_ops;
                    front_end_admission_wait_work_ns_ +=
                        static_cast<long double>(admission_wait);
                    if (!std::isfinite(
                            front_end_admission_wait_work_ns_) ||
                        front_end_admission_wait_work_ns_ >
                            static_cast<long double>(
                                std::numeric_limits<double>::max())) {
                        throw std::runtime_error(
                            "layer-stream front-end wait exceeds finite time");
                    }
                    result_.front_end_admission_max_wait_ns = std::max(
                        result_.front_end_admission_max_wait_ns,
                        admission_wait);
                }
            }
            auto completion = sim_.hbm_.issue(physical_request(
                request.id +
                    (transaction.resident ?
                        "/resident-page" : "/buffer-page") +
                    std::to_string(transaction.page) + "/off" +
                    std::to_string(transaction.offset),
                Tier::HBM,
                request.op,
                now_ns_,
                transaction.physical_addr,
                transaction.bytes,
                AddressSpace::Physical,
                sim_.config_.trace,
                HeatmapTrafficSource::Workload));
            const auto finish_ns = completion.finish_ns;
            ++result_.hbm_user_accesses;
            ++inflight_user_transactions_;
            sim_.absorb_completion(result_, std::move(completion));
            schedule(finish_ns, [this, request_index =
                    transaction.request_index] {
                complete_user_transaction(request_index);
            });
        }
    }

    void complete_user_transaction(std::size_t request_index) {
        if (inflight_user_transactions_ == 0 ||
            request_transactions_remaining_[request_index] == 0) {
            throw std::runtime_error(
                "layer-stream transaction-window accounting underflow");
        }
        --inflight_user_transactions_;
        --request_transactions_remaining_[request_index];
        if (request_transactions_remaining_[request_index] == 0) {
            complete_user_request(request_index);
        }
        pump_user_transactions();
    }

    void complete_user_request(std::size_t request_index) {
        const auto& request = requests_[request_index];
        const auto service_latency =
            now_ns_ - admitted_ns_[request_index];
        const auto offered_latency =
            now_ns_ - front_end_ready_ns_[request_index];
        const auto source_latency =
            now_ns_ - request.arrival_ns;
        if (!std::isfinite(service_latency) || service_latency < 0.0 ||
            !std::isfinite(offered_latency) ||
            offered_latency < service_latency ||
            !std::isfinite(source_latency) ||
            source_latency < offered_latency) {
            throw std::runtime_error(
                "invalid layer-stream user latency frontiers");
        }
        result_.service_latencies_ns[request.index] = service_latency;
        result_.offered_latencies_ns[request.index] = offered_latency;
        result_.source_latencies_ns[request.index] = source_latency;
        ++result_.ops;
        request.op == Op::Read ? ++result_.reads : ++result_.writes;
        result_.logical_bytes = checked_add(
            result_.logical_bytes, request.bytes, "logical user bytes");
        result_.user_finish_ns = std::max(result_.user_finish_ns, now_ns_);
        result_.finish_ns = std::max(result_.finish_ns, now_ns_);
        ++completed_user_ops_;
        for (const auto dependent : request_dependents_[request_index]) {
            auto& remaining = request_dependencies_remaining_[dependent];
            if (remaining == 0) {
                throw std::runtime_error(
                    "same-page request dependency accounting underflow");
            }
            --remaining;
            request_dependency_ready_ns_[dependent] = std::max(
                request_dependency_ready_ns_[dependent], now_ns_);
            const auto dependent_layer = request_layer_[dependent];
            if (remaining == 0 &&
                std::isfinite(front_end_ready_ns_[dependent]) &&
                layers_[dependent_layer].execution_started) {
                schedule_user_request(
                    dependent,
                    std::max(
                        layers_[dependent_layer].execution_start_ns,
                        request_dependency_ready_ns_[dependent]));
            }
        }
        if (outstanding_parent_requests_ == 0) {
            throw std::runtime_error("front-end outstanding accounting underflow");
        }
        --outstanding_parent_requests_;
        try_admit(now_ns_);

        const auto layer_index = request_layer_[request_index];
        auto& layer = layers_[layer_index];
        ++layer.completed_requests;
        layer.execution_finish_ns = std::max(layer.execution_finish_ns, now_ns_);
        if (layer.completed_requests == layer.requests.size()) {
            layer.foreground_finished = true;
            maybe_finish_layer(layer_index);
        }
    }

    void maybe_finish_layer(std::size_t layer_index) {
        auto& layer = layers_[layer_index];
        if (layer.execution_finished || !layer.foreground_finished ||
            !layer.compute_finished) {
            return;
        }
        finish_layer(layer_index);
    }

    void finish_layer(std::size_t layer_index) {
        auto& layer = layers_[layer_index];
        layer.execution_finished = true;
        layer.execution_finish_ns = std::max(
            layer.execution_finish_ns, layer.execution_start_ns);
        result_.user_finish_ns = std::max(
            result_.user_finish_ns, layer.execution_finish_ns);
        result_.finish_ns = std::max(
            result_.finish_ns, layer.execution_finish_ns);
        const auto parity = layer_index % 2;
        if (layer.dirty_pages.empty()) {
            buffer_reusable_ns_[parity] = layer.execution_finish_ns;
        } else {
            if (buffer_writeback_pending_[parity]) {
                throw std::runtime_error(
                    "layer parity buffer began a second writeback before reuse");
            }
            buffer_writeback_pending_[parity] = true;
            layer.remaining_writebacks = layer.dirty_pages.size();
            begin_writeback(layer_index);
        }
        maybe_start_layer(layer_index + 1, now_ns_);
    }

    void begin_writeback(std::size_t layer_index) {
        auto& layer = layers_[layer_index];
        const auto page_size = sim_.backing_page_size();
        for (const auto page : layer.dirty_pages) {
            const auto found = layer.page_index.find(page);
            if (found == layer.page_index.end()) {
                throw std::runtime_error("dirty layer page lacks a buffer slot");
            }
            const auto& page_plan = layer.pages[found->second];
            auto completion = sim_.hbm_.issue(physical_request(
                "layer" + std::to_string(layer.external_id) + "/page" +
                    std::to_string(page) + "/writeback-hbm-read",
                Tier::HBM,
                Op::Read,
                now_ns_,
                slot_addr(layer_index, page_plan.slot),
                page_size,
                AddressSpace::Physical,
                sim_.config_.trace,
                HeatmapTrafficSource::Destage));
            const auto finish = completion.finish_ns;
            ++result_.hbm_background_accesses;
            sim_.absorb_completion(result_, std::move(completion));
            schedule(finish, [this, layer_index, page] {
                if (sim_.config_.backing == BackingTier::Hbf) {
                    issue_d2d_write(layer_index, page);
                } else {
                    issue_backing_write(layer_index, page);
                }
            });
        }
    }

    void issue_d2d_write(std::size_t layer_index, std::uint64_t page) {
        const auto& layer = layers_[layer_index];
        const auto page_size = sim_.backing_page_size();
        const auto stack = sim_.hbf_device().stack_for_logical_page(page);
        auto completion = sim_.base_die_link_for_stack(stack).issue(
            "layer" + std::to_string(layer.external_id) + "/page" +
                std::to_string(page) + "/d2d-write",
            Op::Write,
            now_ns_,
            page_size,
            sim_.config_.trace);
        const auto finish = completion.finish_ns;
        sim_.absorb_link_completion(result_, std::move(completion));
        schedule(finish, [this, layer_index, page] {
            issue_backing_write(layer_index, page);
        });
    }

    void issue_backing_write(std::size_t layer_index, std::uint64_t page) {
        const auto& layer = layers_[layer_index];
        const auto page_plan = layer.page_index.find(page);
        if (page_plan == layer.page_index.end()) {
            throw std::runtime_error("writeback page lacks a layer-buffer slot");
        }
        const auto& plan = layer.pages[page_plan->second];
        if (plan.prior_writer_layer) {
            const WriterKey prior{*plan.prior_writer_layer, page};
            if (!completed_writebacks_.contains(prior)) {
                writeback_waiters_[prior].push_back(WriterKey{layer_index, page});
                return;
            }
        }
        pending_backing_transactions_.push_back(PendingBackingTransaction{
            .kind = BackingTransactionKind::Write,
            .layer = layer_index,
            .page = page,
            .ready_ns = now_ns_,
        });
        pump_backing_transactions();
    }

    void issue_admitted_backing_write(
        std::size_t layer_index,
        std::uint64_t page) {
        const auto& layer = layers_[layer_index];
        const auto page_size = sim_.backing_page_size();
        const auto backing_addr = checked_mul(
            page, page_size, "backing write address");
        double finish = 0.0;
        if (sim_.config_.backing == BackingTier::Hbf) {
            auto completion = sim_.hbf_device().issue(physical_request(
                "layer" + std::to_string(layer.external_id) + "/page" +
                    std::to_string(page) + "/hbf-writeback",
                Tier::HBF,
                Op::Write,
                now_ns_,
                backing_addr,
                page_size,
                AddressSpace::Logical,
                sim_.config_.trace,
                HeatmapTrafficSource::Destage));
            finish = completion.finish_ns;
            ++result_.hbf_background_accesses;
            ++result_.background_hbf_writes;
            result_.hbf_backing_write_bytes = checked_add(
                result_.hbf_backing_write_bytes,
                page_size,
                "HBF backing write bytes");
            sim_.absorb_completion(result_, std::move(completion));
        } else {
            auto completion = sim_.external_device().issue(physical_request(
                "layer" + std::to_string(layer.external_id) + "/page" +
                    std::to_string(page) + "/external-writeback",
                Tier::External,
                Op::Write,
                now_ns_,
                backing_addr,
                page_size,
                AddressSpace::Logical,
                sim_.config_.trace,
                HeatmapTrafficSource::Destage));
            finish = completion.finish_ns;
            ++result_.external_background_accesses;
            result_.external_backing_write_bytes = checked_add(
                result_.external_backing_write_bytes,
                page_size,
                "external backing write bytes");
            sim_.absorb_completion(result_, std::move(completion));
        }
        ++result_.streaming_stats.dirty_pages_written_back;
        result_.streaming_stats.writeback_bytes = checked_add(
            result_.streaming_stats.writeback_bytes,
            page_size,
            "layer writeback bytes");
        schedule(finish, [this, layer_index, page] {
            const WriterKey key{layer_index, page};
            completed_writebacks_.insert(key);
            const auto found = backing_waiters_.find(key);
            if (found != backing_waiters_.end()) {
                const auto waiters = std::move(found->second);
                backing_waiters_.erase(found);
                for (const auto& waiter : waiters) {
                    schedule_backing_read(
                        waiter.layer, waiter.page_index, now_ns_);
                }
            }
            const auto write_found = writeback_waiters_.find(key);
            if (write_found != writeback_waiters_.end()) {
                const auto write_waiters = std::move(write_found->second);
                writeback_waiters_.erase(write_found);
                for (const auto& [writer_layer, writer_page] : write_waiters) {
                    schedule(now_ns_, [this, writer_layer, writer_page] {
                        issue_backing_write(writer_layer, writer_page);
                    });
                }
            }
            complete_buffer_writeback(layer_index);
            complete_backing_transaction();
        });
    }

    void complete_buffer_writeback(std::size_t layer_index) {
        auto& layer = layers_[layer_index];
        if (layer.remaining_writebacks == 0) {
            throw std::runtime_error("layer writeback accounting underflow");
        }
        --layer.remaining_writebacks;
        if (layer.remaining_writebacks != 0) {
            return;
        }

        const auto parity = layer_index % 2;
        if (!buffer_writeback_pending_[parity]) {
            throw std::runtime_error("layer parity writeback fence was not active");
        }
        buffer_writeback_pending_[parity] = false;
        buffer_reusable_ns_[parity] = now_ns_;
        auto waiters = std::move(buffer_install_waiters_[parity]);
        buffer_install_waiters_[parity].clear();
        for (const auto& waiter : waiters) {
            // The old writeback fence and the new backing read are independent.
            // Whichever finishes last determines when the HBM slot is writable.
            const auto install_ns = std::max(now_ns_, waiter.ready_ns);
            result_.streaming_stats.buffer_reuse_wait_work_ns +=
                install_ns - waiter.ready_ns;
            schedule(install_ns, [this, waiter] {
                issue_hbm_install(waiter.layer, waiter.page_index);
            });
        }
        // A no-prefetch layer can be waiting on the parity fence without an
        // install waiter to wake it.
        maybe_start_layer(layer_index + 2, now_ns_);
    }

    void finalize_stats() {
        if (!pending_user_transactions_.empty() ||
            inflight_user_transactions_ != 0 ||
            !pending_backing_transactions_.empty() ||
            inflight_backing_transactions_ != 0 ||
            std::any_of(
                request_transactions_remaining_.begin(),
                request_transactions_remaining_.end(),
                [](std::size_t remaining) { return remaining != 0; })) {
            throw std::runtime_error(
                "layer-stream transaction window did not quiesce");
        }
        auto& streaming = result_.streaming_stats;
        streaming.backing_admission_wait_work_ns =
            static_cast<double>(backing_admission_wait_work_ns_);
        const auto backing_transactions = checked_add(
            streaming.streamed_pages,
            streaming.dirty_pages_written_back,
            "layer backing transaction count");
        if ((streaming.backing_admission_waited_requests == 0) !=
                (streaming.backing_admission_wait_work_ns == 0.0) ||
            (streaming.backing_admission_waited_requests == 0) !=
                (streaming.backing_admission_max_wait_ns == 0.0) ||
            streaming.backing_admission_waited_requests >
                backing_transactions ||
            streaming.backing_admission_max_wait_ns >
                streaming.backing_admission_wait_work_ns ||
            streaming.backing_max_inflight_requests >
                backing_transactions ||
            (streaming.backing_request_credit_limit != 0 &&
             streaming.backing_max_inflight_requests >
                streaming.backing_request_credit_limit) ||
            (streaming.backing_request_credit_limit == 0 &&
             streaming.backing_admission_waited_requests != 0)) {
            throw std::runtime_error(
                "layer-stream backing admission work did not conserve");
        }
        result_.front_end_admission_wait_work_ns =
            static_cast<double>(front_end_admission_wait_work_ns_);
        if ((result_.front_end_admission_waited_ops == 0) !=
                (result_.front_end_admission_wait_work_ns == 0.0) ||
            (result_.front_end_admission_waited_ops == 0) !=
                (result_.front_end_admission_max_wait_ns == 0.0) ||
            result_.front_end_admission_waited_ops > result_.ops ||
            result_.front_end_admission_max_wait_ns >
                result_.front_end_admission_wait_work_ns) {
            throw std::runtime_error(
                "layer-stream front-end admission work did not conserve");
        }
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            const auto& layer = layers_[i];
            if (!layer.prefetch_started || !layer.data_ready) {
                continue;
            }
            const auto prefetch_ns = std::max(
                0.0, layer.data_ready_ns - layer.prefetch_start_ns);
            double hidden_ns = 0.0;
            if (i != 0 && layers_[i - 1].execution_started) {
                const auto& previous = layers_[i - 1];
                hidden_ns = std::max(
                    0.0,
                    std::min(layer.data_ready_ns, previous.execution_finish_ns) -
                        std::max(layer.prefetch_start_ns, previous.execution_start_ns));
            }
            hidden_ns = std::min(hidden_ns, prefetch_ns);
            result_.streaming_stats.hidden_prefetch_ns += hidden_ns;
            result_.streaming_stats.exposed_prefetch_ns += prefetch_ns - hidden_ns;
        }
    }
};

LayerStreamingRunResult LayerStreamingComposition::run(
    const std::vector<MemoryRequest>& requests) {
    if (has_run_) {
        throw std::runtime_error(
            "LayerStreamingComposition is single-use; construct a fresh simulation");
    }
    validate_memory_requests(requests, "layer-streaming-composition");
    const auto footprint_bytes = address_footprint_bytes(requests);
    RunEngine engine(*this, requests, footprint_bytes);
    const auto effective_buffer_bytes =
        engine.effective_layer_buffer_bytes();
    if (config_.address_heatmap_bins != 0) {
        auto heatmap_config = make_composition_address_heatmap_config(
            config_.hbm,
            config_.hbf,
            requests,
            config_.address_heatmap_bins,
            config_.backing == BackingTier::External ?
                config_.external_backing.capacity_bytes : 0);
        auto& hbm_regions = heatmap_config.domains[
            static_cast<std::size_t>(AddressDomain::HbmPhysical)].regions;
        if (engine.resident_physical_bytes() != 0) {
            hbm_regions.push_back(AddressRegion{
                .name = engine.compact_resident_mapping() ?
                    "hybrid_resident" : "foreground_hbm",
                .kind = AddressRegionKind::Workload,
                .begin = 0,
                .end = engine.resident_physical_bytes(),
            });
        }
        if (effective_buffer_bytes != 0) {
            hbm_regions.push_back(AddressRegion{
                .name = "layer_buffer_0",
                .kind = AddressRegionKind::LayerBuffer,
                .begin = hbm_streaming_base_addr_,
                .end = hbm_streaming_base_addr_ + effective_buffer_bytes,
            });
            hbm_regions.push_back(AddressRegion{
                .name = "layer_buffer_1",
                .kind = AddressRegionKind::LayerBuffer,
                .begin = hbm_streaming_base_addr_ + effective_buffer_bytes,
                .end = config_.hbm.capacity_bytes,
            });
        }
        address_heatmap_.emplace(std::move(heatmap_config));
        record_workload_address_traffic(*address_heatmap_, requests);
        hbm_.attach_address_heatmap(*address_heatmap_);
        if (config_.backing == BackingTier::Hbf) {
            hbf_device().attach_address_heatmap(*address_heatmap_);
        } else {
            external_device().attach_address_heatmap(*address_heatmap_);
        }
    }
    has_run_ = true;
    return engine.run();
}

} // namespace hbfsim::physical::hybrid
