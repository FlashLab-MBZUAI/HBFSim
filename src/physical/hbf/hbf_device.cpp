#include "physical/hbf/hbf_device.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

namespace hbfsim::physical::hbf {
namespace {

void require_positive_count(std::uint64_t value, const char* name) {
    if (value == 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

void require_positive_timing(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string(name) + " must be positive and finite");
    }
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs * rhs;
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs, const char* name) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::runtime_error(std::string(name) + " overflows uint64_t");
    }
    return lhs + rhs;
}

std::uint64_t checked_ceil_div(
    std::uint64_t numerator,
    std::uint64_t denominator,
    const char* name) {
    if (denominator == 0) {
        throw std::runtime_error(std::string(name) + " has zero denominator");
    }
    return numerator == 0 ? 0 : 1 + (numerator - 1) / denominator;
}

// Stable controller-side address scrambler. AI tensor/KV layouts commonly
// advance by power-of-two page strides, so each stack-local mapping group
// rotates the global page lanes before ownership is encoded in the VPN.
// SplitMix64 is only a deterministic bit mixer; it models no randomness and
// has no mutable state.
std::uint64_t placement_mix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

using TemporalTouch = std::pair<double, std::uint64_t>;

std::optional<TemporalTouch> latest_touch_through(
    const std::set<TemporalTouch>& touches,
    double at_ns) {
    const auto after = touches.upper_bound(TemporalTouch{
        at_ns, std::numeric_limits<std::uint64_t>::max()});
    if (after == touches.begin()) {
        return std::nullopt;
    }
    return *std::prev(after);
}

void prune_touch_history(std::set<TemporalTouch>& touches, double through_ns) {
    const auto latest = latest_touch_through(touches, through_ns);
    if (!latest) {
        return;
    }
    touches.erase(touches.begin(), touches.find(*latest));
}

bool has_more_than_combined_headroom(
    std::uint64_t available,
    std::uint64_t required,
    std::uint64_t watermark) {
    // `available > required + watermark` without ever forming the potentially
    // overflowing sum. Keeping this predicate centralized makes both the
    // early return and the GC loop use exactly the same boundary semantics.
    return available > required && available - required > watermark;
}

std::string logic_entity(std::uint32_t stack) {
    return "stack" + std::to_string(stack) + "/logic";
}

std::string channel_entity(const HbfAddress& addr) {
    return "stack" + std::to_string(addr.stack) + "/ch" + std::to_string(addr.channel);
}

std::string die_entity(const HbfAddress& addr) {
    return channel_entity(addr) + "/die" + std::to_string(addr.die);
}

std::string plane_entity(const HbfAddress& addr) {
    return die_entity(addr) + "/plane" + std::to_string(addr.plane);
}

std::string subarray_entity(const HbfAddress& addr, std::size_t subarray) {
    return plane_entity(addr) + "/subarray" + std::to_string(subarray);
}

std::string media_lane_entity(const HbfAddress& addr, std::size_t lane) {
    return plane_entity(addr) + "/lane" + std::to_string(lane);
}

std::string page_buffer_bank_entity(const HbfAddress& addr, std::size_t bank) {
    return plane_entity(addr) + "/page_buffer_bank" + std::to_string(bank);
}

std::uint64_t metadata_lpn(std::uint64_t mapping_vpn) {
    return (std::uint64_t{1} << 63) | mapping_vpn;
}

bool is_metadata_lpn(std::uint64_t lpn) {
    return (lpn & (std::uint64_t{1} << 63)) != 0;
}

std::uint64_t metadata_vpn(std::uint64_t lpn) {
    return lpn & ~(std::uint64_t{1} << 63);
}

void trace_wait(
    std::vector<TraceSpan>* spans,
    const std::string& entity,
    double from_ns,
    double to_ns,
    const std::string& name = "scheduler_queue") {
    add_trace_span(spans, name, "queue", entity, from_ns, to_ns);
}

std::uint64_t range_overlap_bytes(
    std::uint64_t lhs_begin,
    std::uint64_t lhs_end,
    std::uint64_t rhs_begin,
    std::uint64_t rhs_end) {
    const auto begin = std::max(lhs_begin, rhs_begin);
    const auto end = std::min(lhs_end, rhs_end);
    return end > begin ? end - begin : 0;
}

std::uint64_t insert_merged_range(
    std::vector<DirtyRange>& ranges,
    DirtyRange incoming) {
    if (incoming.end <= incoming.begin) {
        return 0;
    }

    // Existing ranges are a sorted, disjoint last-writer map. Preserve the
    // source of bytes outside the incoming write, replace provenance only in
    // the overwritten interval, and coalesce adjacent ranges only when their
    // sources agree. This keeps dirty coverage and source attribution in one
    // canonical representation.
    std::uint64_t overlap = 0;
    std::vector<DirtyRange> merged;
    merged.reserve(ranges.size() + 2);
    const auto append = [&merged](DirtyRange range) {
        if (range.end <= range.begin) {
            return;
        }
        if (!merged.empty()) {
            auto& previous = merged.back();
            if (range.begin < previous.end) {
                throw std::runtime_error(
                    "HBF write-buffer provenance ranges overlap");
            }
            if (range.begin == previous.end &&
                range.heatmap_source == previous.heatmap_source) {
                previous.end = range.end;
                return;
            }
        }
        merged.push_back(range);
    };
    bool inserted = false;
    for (const auto& current : ranges) {
        if (current.end <= incoming.begin) {
            append(current);
            continue;
        }
        if (current.begin >= incoming.end) {
            if (!inserted) {
                append(incoming);
                inserted = true;
            }
            append(current);
            continue;
        }

        overlap += range_overlap_bytes(
            current.begin,
            current.end,
            incoming.begin,
            incoming.end);
        if (current.begin < incoming.begin) {
            append(DirtyRange{
                .begin = current.begin,
                .end = incoming.begin,
                .heatmap_source = current.heatmap_source,
            });
        }
        if (!inserted) {
            append(incoming);
            inserted = true;
        }
        if (current.end > incoming.end) {
            append(DirtyRange{
                .begin = incoming.end,
                .end = current.end,
                .heatmap_source = current.heatmap_source,
            });
        }
    }
    if (!inserted) {
        append(incoming);
    }
    ranges = std::move(merged);
    return overlap;
}

HeatmapTrafficSource dominant_dirty_source(
    const std::vector<DirtyRange>& ranges) {
    std::array<std::uint64_t, kHeatmapTrafficSourceCount> bytes_by_source{};
    for (const auto& range : ranges) {
        const auto source = static_cast<std::size_t>(range.heatmap_source);
        if (source >= bytes_by_source.size() || range.end <= range.begin) {
            throw std::runtime_error(
                "HBF write-buffer contains invalid source provenance");
        }
        const auto bytes = range.end - range.begin;
        if (bytes > std::numeric_limits<std::uint64_t>::max() -
                bytes_by_source[source]) {
            throw std::overflow_error(
                "HBF write-buffer source-byte accounting overflows uint64_t");
        }
        bytes_by_source[source] += bytes;
    }

    std::size_t dominant = bytes_by_source.size();
    std::uint64_t dominant_bytes = 0;
    for (std::size_t source = 0; source < bytes_by_source.size(); ++source) {
        // Iterating in enum order gives equal-byte ties a stable result.
        if (bytes_by_source[source] > dominant_bytes) {
            dominant = source;
            dominant_bytes = bytes_by_source[source];
        }
    }
    if (dominant == bytes_by_source.size()) {
        throw std::runtime_error(
            "HBF cannot flush a write-buffer entry without dirty provenance");
    }
    return static_cast<HeatmapTrafficSource>(dominant);
}

} // namespace

ResidentMappingCapacity derive_resident_mapping_capacity(
    const HbfConfig& config) {
    auto pages_per_stack = checked_mul(
        config.channels_per_stack,
        config.dies_per_channel,
        "HBF pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.planes_per_die,
        "HBF pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.blocks_per_plane,
        "HBF pages per stack");
    pages_per_stack = checked_mul(
        pages_per_stack,
        config.pages_per_block,
        "HBF pages per stack");
    const auto mapping_pages_per_stack = checked_ceil_div(
        pages_per_stack,
        config.mapping_entries_per_page,
        "HBF resident mapping pages per stack");
    const auto bytes_per_stack = checked_mul(
        mapping_pages_per_stack,
        config.page_size_bytes,
        "HBF resident mapping bytes per stack");
    return ResidentMappingCapacity{
        .pages_per_stack = mapping_pages_per_stack,
        .bytes_per_stack = bytes_per_stack,
        .total_bytes = checked_mul(
            bytes_per_stack,
            config.stacks,
            "HBF resident mapping table bytes"),
    };
}

std::string HbfAddress::path() const {
    std::ostringstream out;
    out << "stack" << stack << "/ch" << channel << "/die" << die
        << "/plane" << plane << "/block" << block << "/page" << page
        << "/off" << offset;
    return out.str();
}

std::optional<double> HbfStats::waf() const {
    if (logical_write_bytes == 0) {
        return std::nullopt;
    }
    return static_cast<double>(physical_write_bytes) /
        static_cast<double>(logical_write_bytes);
}

double HbfStats::active_span_ns() const {
    return std::isfinite(first_arrival_ns) && finish_ns > first_arrival_ns ?
        finish_ns - first_arrival_ns : 0.0;
}

double HbfStats::media_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || planes == 0 ? 0.0 :
        media_busy_ns / (span * static_cast<double>(planes));
}

double HbfStats::io_utilization() const {
    // The model has independent command and data HBIO ports per stack.
    const auto span = active_span_ns();
    return span <= 0.0 || stacks == 0 ? 0.0 :
        (hb_io_command_busy_ns + hb_io_data_busy_ns) /
            (span * 2.0 * static_cast<double>(stacks));
}

double HbfStats::media_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : media_busy_ns / span;
}

double HbfStats::read_lane_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : read_lane_busy_ns / span;
}

double HbfStats::subarray_read_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : subarray_read_busy_ns / span;
}

double HbfStats::page_buffer_bank_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : page_buffer_bank_busy_ns / span;
}

double HbfStats::channel_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 :
        (channel_command_busy_ns + channel_data_busy_ns) / span;
}

double HbfStats::hbio_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 :
        (hb_io_command_busy_ns + hb_io_data_busy_ns) / span;
}

double HbfStats::hbio_command_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || stacks == 0 ? 0.0 :
        hb_io_command_busy_ns / (span * static_cast<double>(stacks));
}

double HbfStats::hbio_data_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || stacks == 0 ? 0.0 :
        hb_io_data_busy_ns / (span * static_cast<double>(stacks));
}

double HbfStats::sequencer_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : sequencer_busy_ns / span;
}

double HbfStats::ecc_issue_parallelism() const {
    const auto span = active_span_ns();
    return span <= 0.0 ? 0.0 : ecc_issue_busy_ns / span;
}

double HbfStats::ecc_issue_utilization() const {
    const auto span = active_span_ns();
    return span <= 0.0 || dies == 0 ? 0.0 :
        ecc_issue_busy_ns / (span * static_cast<double>(dies));
}

std::uint64_t HbfDevice::ResourceTimeline::next_priority() {
    // Deterministic xorshift64: priorities depend only on this calendar's
    // insertion sequence, so runs remain reproducible while the treap stays
    // balanced in expectation even for monotonically increasing gap starts.
    priority_state_ ^= priority_state_ << 13;
    priority_state_ ^= priority_state_ >> 7;
    priority_state_ ^= priority_state_ << 17;
    return priority_state_;
}

void HbfDevice::ResourceTimeline::refresh(GapNode& node) {
    node.max_duration_ns = node.gap.end_ns - node.gap.begin_ns;
    if (node.left) {
        node.max_duration_ns = std::max(
            node.max_duration_ns, node.left->max_duration_ns);
    }
    if (node.right) {
        node.max_duration_ns = std::max(
            node.max_duration_ns, node.right->max_duration_ns);
    }
}

void HbfDevice::ResourceTimeline::split(
    std::unique_ptr<GapNode> root,
    double key,
    std::unique_ptr<GapNode>& lower,
    std::unique_ptr<GapNode>& upper) {
    if (!root) {
        lower.reset();
        upper.reset();
        return;
    }
    if (root->gap.begin_ns < key) {
        auto right = std::move(root->right);
        split(std::move(right), key, root->right, upper);
        refresh(*root);
        lower = std::move(root);
        return;
    }
    auto left = std::move(root->left);
    split(std::move(left), key, lower, root->left);
    refresh(*root);
    upper = std::move(root);
}

void HbfDevice::ResourceTimeline::insert_node(
    std::unique_ptr<GapNode>& root,
    std::unique_ptr<GapNode> node) {
    if (!root) {
        root = std::move(node);
        return;
    }
    if (node->gap.begin_ns == root->gap.begin_ns) {
        throw std::runtime_error("HBF resource calendar has duplicate gap starts");
    }
    if (node->priority > root->priority) {
        split(std::move(root), node->gap.begin_ns, node->left, node->right);
        refresh(*node);
        root = std::move(node);
        return;
    }
    if (node->gap.begin_ns < root->gap.begin_ns) {
        insert_node(root->left, std::move(node));
    } else {
        insert_node(root->right, std::move(node));
    }
    refresh(*root);
}

std::unique_ptr<HbfDevice::ResourceTimeline::GapNode>
HbfDevice::ResourceTimeline::merge(
    std::unique_ptr<GapNode> lower,
    std::unique_ptr<GapNode> upper) {
    if (!lower) {
        return upper;
    }
    if (!upper) {
        return lower;
    }
    if (lower->priority > upper->priority) {
        lower->right = merge(std::move(lower->right), std::move(upper));
        refresh(*lower);
        return lower;
    }
    upper->left = merge(std::move(lower), std::move(upper->left));
    refresh(*upper);
    return upper;
}

void HbfDevice::ResourceTimeline::erase_node(
    std::unique_ptr<GapNode>& root,
    double key) {
    if (!root) {
        throw std::runtime_error("HBF resource calendar lost a selected gap");
    }
    if (key == root->gap.begin_ns) {
        root = merge(std::move(root->left), std::move(root->right));
        return;
    }
    if (key < root->gap.begin_ns) {
        erase_node(root->left, key);
    } else {
        erase_node(root->right, key);
    }
    refresh(*root);
}

const HbfDevice::ResourceTimeline::GapNode*
HbfDevice::ResourceTimeline::predecessor(const GapNode* root, double key) {
    const GapNode* result = nullptr;
    while (root != nullptr) {
        if (root->gap.begin_ns <= key) {
            result = root;
            root = root->right.get();
        } else {
            root = root->left.get();
        }
    }
    return result;
}

const HbfDevice::ResourceTimeline::GapNode*
HbfDevice::ResourceTimeline::successor(const GapNode* root, double key) {
    const GapNode* result = nullptr;
    while (root != nullptr) {
        if (root->gap.begin_ns >= key) {
            result = root;
            root = root->left.get();
        } else {
            root = root->right.get();
        }
    }
    return result;
}

const HbfDevice::ResourceTimeline::GapNode*
HbfDevice::ResourceTimeline::first_fitting_from(
    const GapNode* root,
    double minimum_begin_ns,
    double duration_ns) {
    if (root == nullptr || root->max_duration_ns < duration_ns) {
        return nullptr;
    }
    if (root->gap.begin_ns < minimum_begin_ns) {
        return first_fitting_from(root->right.get(), minimum_begin_ns, duration_ns);
    }
    if (const auto* in_left = first_fitting_from(
            root->left.get(), minimum_begin_ns, duration_ns)) {
        return in_left;
    }
    if (root->gap.begin_ns + duration_ns <= root->gap.end_ns) {
        return root;
    }
    return first_fitting_from(root->right.get(), minimum_begin_ns, duration_ns);
}

std::optional<HbfDevice::ResourceTimeline::Gap>
HbfDevice::ResourceTimeline::first_fitting_gap(
    double earliest_ns,
    double duration_ns) const {
    if (!std::isfinite(earliest_ns) || !std::isfinite(duration_ns) ||
        earliest_ns < pruned_through_ns_ || duration_ns <= 0.0) {
        throw std::runtime_error("HBF resource calendar received an invalid reservation");
    }
    if (const auto* containing = predecessor(gap_root_.get(), earliest_ns);
        containing != nullptr &&
        earliest_ns + duration_ns <= containing->gap.end_ns) {
        return containing->gap;
    }
    if (const auto* later = first_fitting_from(
            gap_root_.get(), earliest_ns, duration_ns)) {
        return later->gap;
    }
    return std::nullopt;
}

bool HbfDevice::ResourceTimeline::can_reserve_exact(
    double begin_ns,
    double duration_ns) const {
    if (!std::isfinite(begin_ns) || !std::isfinite(duration_ns) ||
        begin_ns < pruned_through_ns_ || duration_ns <= 0.0 ||
        !std::isfinite(begin_ns + duration_ns)) {
        throw std::runtime_error(
            "HBF resource calendar received an invalid exact-reservation query");
    }
    if (begin_ns >= ready_ns) {
        return true;
    }
    const auto* containing = predecessor(gap_root_.get(), begin_ns);
    return containing != nullptr &&
        begin_ns >= containing->gap.begin_ns &&
        begin_ns + duration_ns <= containing->gap.end_ns;
}

double HbfDevice::ResourceTimeline::preview_start(
    double earliest_ns,
    double duration_ns) const {
    if (!std::isfinite(earliest_ns) || !std::isfinite(duration_ns) ||
        earliest_ns < pruned_through_ns_ || duration_ns <= 0.0) {
        throw std::runtime_error(
            "HBF resource calendar received an invalid reservation preview");
    }
    if (const auto gap = first_fitting_gap(earliest_ns, duration_ns)) {
        return std::max(gap->begin_ns, earliest_ns);
    }
    const double start = std::max(earliest_ns, ready_ns);
    if (!std::isfinite(start) || !std::isfinite(start + duration_ns)) {
        throw std::runtime_error("HBF resource calendar preview time overflowed");
    }
    return start;
}

void HbfDevice::ResourceTimeline::insert_gap(double begin_ns, double end_ns) {
    if (!std::isfinite(begin_ns) || !std::isfinite(end_ns) ||
        begin_ns < pruned_through_ns_ || end_ns <= begin_ns) {
        throw std::runtime_error("HBF resource calendar received an invalid idle gap");
    }
    if (const auto* previous = predecessor(gap_root_.get(), begin_ns);
        previous != nullptr && previous->gap.end_ns > begin_ns) {
        throw std::runtime_error("HBF resource calendar inserted overlapping idle gaps");
    }
    if (const auto* next = successor(gap_root_.get(), begin_ns);
        next != nullptr && next->gap.begin_ns < end_ns) {
        throw std::runtime_error("HBF resource calendar inserted overlapping idle gaps");
    }
    auto node = std::make_unique<GapNode>();
    node->gap = Gap{.begin_ns = begin_ns, .end_ns = end_ns};
    node->priority = next_priority();
    node->max_duration_ns = end_ns - begin_ns;
    insert_node(gap_root_, std::move(node));
}

void HbfDevice::ResourceTimeline::consume_gap(
    const Gap& gap,
    double begin_ns,
    double end_ns) {
    if (begin_ns < gap.begin_ns || end_ns > gap.end_ns || end_ns <= begin_ns) {
        throw std::runtime_error("HBF resource calendar consumed outside an idle gap");
    }
    erase_node(gap_root_, gap.begin_ns);
    if (gap.begin_ns < begin_ns) {
        insert_gap(gap.begin_ns, begin_ns);
    }
    if (end_ns < gap.end_ns) {
        insert_gap(end_ns, gap.end_ns);
    }
}

void HbfDevice::ResourceTimeline::prune_before(double causal_watermark_ns) {
    if (!std::isfinite(causal_watermark_ns) || causal_watermark_ns < 0.0) {
        throw std::runtime_error("HBF resource calendar received an invalid causal watermark");
    }
    if (causal_watermark_ns <= pruned_through_ns_) {
        return;
    }

    std::unique_ptr<GapNode> expired;
    std::unique_ptr<GapNode> future;
    split(std::move(gap_root_), causal_watermark_ns, expired, future);

    // All intervals in `expired` begin before the watermark. Because gaps
    // are disjoint and ordered, only its rightmost interval can cross the
    // watermark; preserve that usable suffix exactly and discard the rest.
    std::optional<double> crossing_end_ns;
    const GapNode* rightmost = expired.get();
    while (rightmost != nullptr && rightmost->right) {
        rightmost = rightmost->right.get();
    }
    if (rightmost != nullptr && rightmost->gap.end_ns > causal_watermark_ns) {
        crossing_end_ns = rightmost->gap.end_ns;
    }

    gap_root_ = std::move(future);
    pruned_through_ns_ = causal_watermark_ns;
    // A frontier behind the causal watermark represents expired idle time,
    // not future capacity. Advancing it is schedule-equivalent for all legal
    // future reservations and prevents reintroducing an expired prefix.
    ready_ns = std::max(ready_ns, causal_watermark_ns);
    if (crossing_end_ns) {
        insert_gap(causal_watermark_ns, *crossing_end_ns);
    }
}

double HbfStats::plane_media_skew() const {
    return avg_active_plane_media_busy_ns <= 0.0 ? 0.0 :
        max_plane_media_busy_ns / avg_active_plane_media_busy_ns;
}

double HbfStats::plane_op_skew() const {
    return avg_active_plane_ops <= 0.0 ? 0.0 :
        static_cast<double>(max_plane_ops) / avg_active_plane_ops;
}

double HbfStats::media_lane_skew() const {
    return avg_active_media_lane_busy_ns <= 0.0 ? 0.0 :
        max_media_lane_busy_ns / avg_active_media_lane_busy_ns;
}

double HbfStats::media_lane_read_skew() const {
    return avg_active_media_lane_reads <= 0.0 ? 0.0 :
        static_cast<double>(max_media_lane_reads) / avg_active_media_lane_reads;
}

double HbfStats::subarray_busy_skew() const {
    return avg_active_subarray_busy_ns <= 0.0 ? 0.0 :
        max_subarray_busy_ns / avg_active_subarray_busy_ns;
}

double HbfStats::subarray_read_skew() const {
    return avg_active_subarray_reads <= 0.0 ? 0.0 :
        static_cast<double>(max_subarray_reads) / avg_active_subarray_reads;
}

double HbfStats::page_buffer_bank_skew() const {
    return avg_active_page_buffer_bank_busy_ns <= 0.0 ? 0.0 :
        max_page_buffer_bank_busy_ns / avg_active_page_buffer_bank_busy_ns;
}

double HbfStats::page_buffer_bank_read_skew() const {
    return avg_active_page_buffer_bank_reads <= 0.0 ? 0.0 :
        static_cast<double>(max_page_buffer_bank_reads) / avg_active_page_buffer_bank_reads;
}

double HbfStats::channel_busy_skew() const {
    return avg_active_channel_busy_ns <= 0.0 ? 0.0 :
        max_channel_busy_ns / avg_active_channel_busy_ns;
}

double HbfStats::die_transaction_skew() const {
    return avg_active_die_transactions <= 0.0 ? 0.0 :
        static_cast<double>(max_die_transactions) / avg_active_die_transactions;
}

HbfDevice::HbfDevice(HbfConfig config, AddressHeatmap* address_heatmap)
    : config_(config), address_heatmap_(address_heatmap) {
    require_positive_count(config_.stacks, "HBF stacks");
    require_positive_count(config_.channels_per_stack, "HBF channels_per_stack");
    require_positive_count(config_.dies_per_channel, "HBF dies_per_channel");
    require_positive_count(config_.planes_per_die, "HBF planes_per_die");
    require_positive_count(config_.blocks_per_plane, "HBF blocks_per_plane");
    require_positive_count(config_.pages_per_block, "HBF pages_per_block");
    require_positive_count(config_.page_size_bytes, "HBF page_size_bytes");
    require_positive_count(config_.media_lanes_per_plane, "HBF media_lanes_per_plane");
    require_positive_count(config_.page_buffer_banks_per_plane, "HBF page_buffer_banks_per_plane");
    subarrays_per_plane_ =
        config_.subarrays_per_plane == 0 ?
        config_.media_lanes_per_plane :
        config_.subarrays_per_plane;
    require_positive_count(subarrays_per_plane_, "HBF subarrays_per_plane");
    if (config_.pages_per_block > 1024) {
        // BlockState's valid-page bitmap holds 16 x 64 bits.
        throw std::runtime_error("HBF pages_per_block must be <= 1024");
    }
    if (config_.page_size_bytes > 4096) {
        throw std::runtime_error("HBF v0 is SLC-only with page_size_bytes <= 4096");
    }
    require_positive_timing(config_.t_read_page_ns, "HBF t_read_page_ns");
    require_positive_timing(config_.t_program_page_ns, "HBF t_program_page_ns");
    require_positive_timing(config_.t_erase_block_ns, "HBF t_erase_block_ns");
    require_positive_timing(config_.t_program_verify_ns, "HBF t_program_verify_ns");
    require_positive_timing(
        config_.ecc_decode_latency_ns, "HBF ecc_decode_latency_ns");
    require_positive_timing(
        config_.ecc_encode_latency_ns, "HBF ecc_encode_latency_ns");
    require_positive_timing(
        config_.ecc_decode_raw_bandwidth_GBps_per_die,
        "HBF ecc_decode_raw_bandwidth_GBps_per_die");
    require_positive_timing(
        config_.ecc_encode_raw_bandwidth_GBps_per_die,
        "HBF ecc_encode_raw_bandwidth_GBps_per_die");
    require_positive_timing(config_.channel_bandwidth_GBps, "HBF channel_bandwidth_GBps");
    require_positive_timing(config_.hb_io_bandwidth_GBps, "HBF hb_io_bandwidth_GBps");
    require_positive_timing(config_.tsv_bandwidth_GBps, "HBF tsv_bandwidth_GBps");
    require_positive_timing(config_.media_lane_bandwidth_GBps, "HBF media_lane_bandwidth_GBps");
    require_positive_timing(config_.logic_sram_bandwidth_GBps, "HBF logic_sram_bandwidth_GBps");
    require_positive_timing(config_.page_buffer_bandwidth_GBps, "HBF page_buffer_bandwidth_GBps");
    require_positive_timing(config_.logic_scheduler_issue_ns, "HBF logic_scheduler_issue_ns");
    require_positive_timing(config_.address_generation_ns, "HBF address_generation_ns");
    require_positive_timing(config_.ctrl_dram_latency_ns, "HBF ctrl_dram_latency_ns");
    require_positive_timing(config_.ctrl_dram_issue_ns, "HBF ctrl_dram_issue_ns");
    require_positive_timing(config_.mapping_update_ns, "HBF mapping_update_ns");
    require_positive_timing(config_.free_page_allocation_ns, "HBF free_page_allocation_ns");
    require_positive_timing(config_.flash_tsu_issue_ns, "HBF flash_tsu_issue_ns");
    require_positive_count(config_.command_address_bytes, "HBF command_address_bytes");
    require_positive_count(config_.mapping_entries_per_page, "HBF mapping_entries_per_page");
    require_positive_count(
        config_.page_read_queue_depth_per_stack,
        "HBF page_read_queue_depth_per_stack");
    if (config_.page_read_queue_depth_per_stack >
        std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "HBF page_read_queue_depth_per_stack exceeds size_t range");
    }
    if (config_.oob_bytes_per_page >= config_.page_size_bytes) {
        throw std::runtime_error("HBF oob_bytes_per_page must be smaller than page_size_bytes");
    }
    const auto codeword_bytes = config_.page_size_bytes + config_.oob_bytes_per_page;
    const double decode_initiation_ns = transfer_time_ns(
        codeword_bytes, config_.ecc_decode_raw_bandwidth_GBps_per_die);
    const double encode_initiation_ns = transfer_time_ns(
        codeword_bytes, config_.ecc_encode_raw_bandwidth_GBps_per_die);
    if (config_.ecc_decode_latency_ns < decode_initiation_ns) {
        throw std::runtime_error(
            "HBF ecc_decode_latency_ns must cover the full codeword initiation interval");
    }
    if (config_.ecc_encode_latency_ns < encode_initiation_ns) {
        throw std::runtime_error(
            "HBF ecc_encode_latency_ns must cover the full codeword initiation interval");
    }
    if (config_.write_coalescing_enabled) {
        require_positive_count(config_.write_buffer_pages, "HBF write_buffer_pages");
    }
    if (config_.gc_wear_leveling_weight < 0.0 || !std::isfinite(config_.gc_wear_leveling_weight)) {
        throw std::runtime_error("HBF gc_wear_leveling_weight must be non-negative and finite");
    }

    const auto total_channels = checked_mul(config_.stacks, config_.channels_per_stack,
        "HBF total_channels");
    const auto total_dies = checked_mul(total_channels, config_.dies_per_channel,
        "HBF total_dies");
    const auto total_planes = checked_mul(total_dies, config_.planes_per_die,
        "HBF total_planes");
    const auto total_blocks = checked_mul(total_planes, config_.blocks_per_plane,
        "HBF total_blocks");
    total_pages_ = checked_mul(total_blocks, config_.pages_per_block, "HBF total_pages");
    (void)checked_mul(total_pages_, config_.page_size_bytes, "HBF capacity bytes");
    if (config_.gc_reserved_free_blocks_per_plane >= config_.blocks_per_plane) {
        throw std::runtime_error(
            "HBF gc_reserved_free_blocks_per_plane must be smaller than blocks_per_plane");
    }
    const auto planes_per_stack = total_planes / config_.stacks;
    const auto usable_blocks_per_plane =
        config_.blocks_per_plane - config_.gc_reserved_free_blocks_per_plane;
    const auto usable_blocks_per_stack = checked_mul(
        planes_per_stack, usable_blocks_per_plane, "HBF usable blocks per stack");
    const auto gc_watermark_capacity = checked_mul(
        usable_blocks_per_stack, config_.pages_per_block, "HBF usable pages per stack");
    if (config_.gc_low_watermark_pages > gc_watermark_capacity) {
        throw std::runtime_error(
            "HBF gc_low_watermark_pages exceeds usable per-stack page capacity");
    }
    if (config_.gc_hard_watermark_pages > gc_watermark_capacity) {
        throw std::runtime_error(
            "HBF gc_hard_watermark_pages exceeds usable per-stack page capacity");
    }
    free_pages_ = total_pages_;
    free_pages_per_stack_.assign(config_.stacks, total_pages_ / config_.stacks);
    next_data_allocation_plane_per_stack_.assign(config_.stacks, 0);
    next_mapping_allocation_plane_per_stack_.assign(config_.stacks, 0);
    next_gc_allocation_plane_per_stack_.assign(config_.stacks, 0);
    causal_state_ready_by_stack_.assign(config_.stacks, 0.0);
    state_observation_by_stack_.assign(config_.stacks, 0.0);
    gc_active_by_stack_.assign(config_.stacks, false);
    materialized_ready_by_block_.assign(
        static_cast<std::size_t>(total_blocks), 0.0);
    write_buffer_lru_by_stack_.resize(config_.stacks);
    write_buffer_by_stack_.resize(config_.stacks);
    write_buffer_slot_release_by_stack_.resize(config_.stacks);
    page_read_credit_release_by_stack_.resize(config_.stacks);

    // The complete stack-local L2P table is resident from time zero. Its
    // representation has the same entry density as a persistent checkpoint
    // page, but no checkpoint read is part of the runtime lookup path.
    const auto resident_mapping =
        derive_resident_mapping_capacity(config_);
    resident_mapping_pages_per_stack_ =
        resident_mapping.pages_per_stack;
    resident_mapping_bytes_per_stack_ =
        resident_mapping.bytes_per_stack;
    const auto required_ctrl_dram_bytes =
        resident_mapping.total_bytes;
    if (config_.ctrl_dram_bytes != 0 &&
        config_.ctrl_dram_bytes / config_.stacks <
            resident_mapping_bytes_per_stack_) {
        throw std::runtime_error(
            "HBF ctrl_dram_bytes cannot hold the complete resident L2P table "
            "in every stack partition");
    }
    if (config_.ctrl_dram_bytes == 0) {
        config_.ctrl_dram_bytes = required_ctrl_dram_bytes;
    }
    stats_.resident_mapping_table_bytes = required_ctrl_dram_bytes;
    stats_.resident_mapping_table_bytes_per_stack =
        resident_mapping_bytes_per_stack_;
    stats_.resident_mapping_pages_per_stack =
        resident_mapping_pages_per_stack_;
    stats_.mapping_dram_resources = config_.stacks;

    channels_.resize(static_cast<std::size_t>(total_channels));
    dies_.resize(static_cast<std::size_t>(total_dies));
    planes_.resize(static_cast<std::size_t>(total_planes));
    logic_dies_.resize(config_.stacks);
    blocks_.resize(static_cast<std::size_t>(total_blocks));
    for (auto& plane : planes_) {
        plane.subarrays.resize(subarrays_per_plane_);
        plane.available_sense_rounds_by_subarray.resize(subarrays_per_plane_);
        plane.media_lanes.resize(config_.media_lanes_per_plane);
        plane.page_buffer_banks.resize(config_.page_buffer_banks_per_plane);
    }
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        auto& block = blocks_[block_index];
        block.free_pages = config_.pages_per_block;
        block.next_page = 0;
        planes_.at(block_plane_index(block_index)).free_blocks.push_back(block_index);
    }
    stats_.free_pages = free_pages_;
    refresh_parallel_stats();
}

HbfAddress HbfDevice::decode(std::uint64_t addr) const {
    if (addr / config_.page_size_bytes >= total_pages_) {
        throw std::runtime_error("HBF physical byte address is out of range");
    }
    HbfAddress decoded;
    decoded.offset = addr % config_.page_size_bytes;
    std::uint64_t unit = addr / config_.page_size_bytes;
    decoded.page = static_cast<std::uint32_t>(unit % config_.pages_per_block);
    unit /= config_.pages_per_block;
    decoded.block = static_cast<std::uint32_t>(unit % config_.blocks_per_plane);
    unit /= config_.blocks_per_plane;
    decoded.plane = static_cast<std::uint32_t>(unit % config_.planes_per_die);
    unit /= config_.planes_per_die;
    decoded.die = static_cast<std::uint32_t>(unit % config_.dies_per_channel);
    unit /= config_.dies_per_channel;
    decoded.channel = static_cast<std::uint32_t>(unit % config_.channels_per_stack);
    unit /= config_.channels_per_stack;
    decoded.stack = static_cast<std::uint32_t>(unit % config_.stacks);
    return decoded;
}

std::uint64_t HbfDevice::encode(const HbfAddress& addr) const {
    return encode_ppn(addr) * config_.page_size_bytes + addr.offset;
}

std::uint64_t HbfDevice::schedule_commit(
    std::size_t stack,
    double at_ns,
    std::function<void()> action) {
    if (!std::isfinite(at_ns) || at_ns < 0.0) {
        throw std::runtime_error("HBF state commit time must be finite and non-negative");
    }
    if (stack >= config_.stacks) {
        throw std::runtime_error("HBF state commit stack is out of range");
    }
    const auto sequence = next_commit_sequence_++;
    pending_commits_.emplace(
        std::make_pair(at_ns, sequence),
        PendingCommit{.stack = stack, .action = std::move(action)});
    return sequence;
}

void HbfDevice::apply_selected_commits_through(
    const std::unordered_set<std::uint64_t>& sequences,
    double at_ns) {
    if (sequences.empty()) {
        return;
    }
    for (auto commit = pending_commits_.begin();
         commit != pending_commits_.end() && commit->first.first <= at_ns;) {
        if (!sequences.contains(commit->first.second)) {
            ++commit;
            continue;
        }
        auto action = std::move(commit->second.action);
        commit = pending_commits_.erase(commit);
        action();
    }
}

void HbfDevice::apply_commits_through(double at_ns) {
    while (!pending_commits_.empty() && pending_commits_.begin()->first.first <= at_ns) {
        auto action = std::move(pending_commits_.begin()->second.action);
        pending_commits_.erase(pending_commits_.begin());
        action();
    }
}

void HbfDevice::apply_stack_commits_through(std::size_t stack, double at_ns) {
    for (;;) {
        const auto limit = pending_commits_.upper_bound(
            std::make_pair(at_ns, std::numeric_limits<std::uint64_t>::max()));
        const auto found = std::find_if(
            pending_commits_.begin(),
            limit,
            [stack](const auto& item) { return item.second.stack == stack; });
        if (found == limit) {
            return;
        }
        auto action = std::move(found->second.action);
        pending_commits_.erase(found);
        action();
    }
}

void HbfDevice::apply_all_commits() {
    while (!pending_commits_.empty()) {
        const double at_ns = pending_commits_.begin()->first.first;
        apply_commits_through(at_ns);
    }
}

bool HbfDevice::advance_to_next_commit(double& at_ns, std::size_t stack) {
    const auto found = std::find_if(
        pending_commits_.begin(),
        pending_commits_.end(),
        [stack](const auto& item) { return item.second.stack == stack; });
    if (found == pending_commits_.end()) {
        return false;
    }
    at_ns = std::max(at_ns, found->first.first);
    apply_stack_commits_through(stack, at_ns);
    causal_state_ready_by_stack_.at(stack) = std::max(
        causal_state_ready_by_stack_.at(stack), at_ns);
    return true;
}

std::optional<std::uint64_t> HbfDevice::visible_lpn_at(
    std::uint64_t lpn,
    double at_ns) const {
    std::optional<std::uint64_t> visible;
    if (const auto base = lpn_to_ppn_.find(lpn); base != lpn_to_ppn_.end()) {
        const auto page = programmed_pages_.find(base->second);
        const auto block_index = static_cast<std::size_t>(
            base->second / config_.pages_per_block);
        const auto& block = blocks_.at(block_index);
        if (page != programmed_pages_.end() &&
            page->second.status == PageStatus::Valid &&
            page->second.owner == PageOwner::Logical &&
            page->second.block_epoch == block.epoch &&
            !block.erase_pending) {
            visible = base->second;
        }
    }
    if (!visible) {
        visible = compact_lpn_ppn(lpn);
    }
    const auto pending = pending_lpn_updates_.find(lpn);
    if (pending == pending_lpn_updates_.end()) {
        return visible;
    }
    const PendingMappingUpdate* newest = nullptr;
    for (const auto& update : pending->second) {
        const auto block_index = static_cast<std::size_t>(
            update.new_ppn / config_.pages_per_block);
        if (blocks_.at(block_index).epoch == update.block_epoch &&
            !blocks_.at(block_index).erase_pending &&
            update.commit_ns <= at_ns &&
            (newest == nullptr || std::tie(update.commit_ns, update.sequence) >
                std::tie(newest->commit_ns, newest->sequence))) {
            newest = &update;
        }
    }
    return newest == nullptr ? visible : std::optional<std::uint64_t>{newest->new_ppn};
}

void HbfDevice::wait_for_prior_lpn_commit(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    double dependency_ready_ns = 0.0;
    if (const auto observed = materialized_ready_by_lpn_.find(lpn);
        observed != materialized_ready_by_lpn_.end()) {
        dependency_ready_ns = observed->second;
    }
    std::unordered_set<std::uint64_t> commit_sequences;
    const auto pending = pending_lpn_updates_.find(lpn);
    if (pending != pending_lpn_updates_.end()) {
        for (const auto& update : pending->second) {
            // Per-LPN ordering survives destruction of the target: a later
            // read waits for the prior write callback and its target erase.
            dependency_ready_ns = std::max({
                dependency_ready_ns,
                update.commit_ns,
                update.destructive_ready_ns,
            });
            commit_sequences.insert(update.program_commit_sequence);
            commit_sequences.insert(update.sequence);
            const auto block_index = static_cast<std::size_t>(
                update.new_ppn / config_.pages_per_block);
            if (const auto erase = pending_physical_erases_.find(block_index);
                erase != pending_physical_erases_.end()) {
                commit_sequences.insert(erase->second.commit_sequence);
            }
        }
    }
    const double required_ns = std::max(at_ns, dependency_ready_ns);
    if (required_ns > at_ns) {
        trace_wait(
            spans,
            logic_entity(static_cast<std::uint32_t>(stack_for_lpn(lpn))),
            at_ns,
            required_ns,
            "wait_prior_lpn_commit");
        breakdown.scheduler_queue_wait_ns += required_ns - at_ns;
        at_ns = required_ns;
    }
    const auto stack = stack_for_lpn(lpn);
    apply_selected_commits_through(commit_sequences, at_ns);
    if (dependency_ready_ns != 0.0) {
        materialized_ready_by_lpn_[lpn] = std::max(
            materialized_ready_by_lpn_[lpn], dependency_ready_ns);
    }
    state_observation_by_stack_.at(stack) = std::max(
        state_observation_by_stack_.at(stack), at_ns);
}

void HbfDevice::wait_for_pending_vpn_erase(
    std::uint64_t mapping_vpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    double dependency_ready_ns = 0.0;
    if (const auto observed = materialized_ready_by_vpn_.find(mapping_vpn);
        observed != materialized_ready_by_vpn_.end()) {
        dependency_ready_ns = observed->second;
    }
    std::unordered_set<std::uint64_t> commit_sequences;
    bool has_destructive_dependency = false;
    const auto pending = pending_vpn_updates_.find(mapping_vpn);
    if (pending != pending_vpn_updates_.end()) {
        for (const auto& update : pending->second) {
            has_destructive_dependency |= update.destructive_ready_ns != 0.0;
            dependency_ready_ns = std::max(
                dependency_ready_ns, update.destructive_ready_ns);
            commit_sequences.insert(update.program_commit_sequence);
            commit_sequences.insert(update.sequence);
            const auto block_index = static_cast<std::size_t>(
                update.new_ppn / config_.pages_per_block);
            if (const auto erase = pending_physical_erases_.find(block_index);
                erase != pending_physical_erases_.end()) {
                commit_sequences.insert(erase->second.commit_sequence);
            }
        }
    }
    if (!has_destructive_dependency && dependency_ready_ns == 0.0) {
        return;
    }
    const double required_ns = std::max(at_ns, dependency_ready_ns);
    const auto stack = stack_for_vpn(mapping_vpn);
    if (required_ns > at_ns) {
        trace_wait(
            spans,
            logic_entity(static_cast<std::uint32_t>(stack)),
            at_ns,
            required_ns,
            "wait_pending_mapping_block_erase");
        breakdown.scheduler_queue_wait_ns += required_ns - at_ns;
        at_ns = required_ns;
    }
    apply_selected_commits_through(commit_sequences, at_ns);
    materialized_ready_by_vpn_[mapping_vpn] = std::max(
        materialized_ready_by_vpn_[mapping_vpn], dependency_ready_ns);
    state_observation_by_stack_.at(stack) = std::max(
        state_observation_by_stack_.at(stack), at_ns);
}

void HbfDevice::wait_for_pending_block_erase(
    std::uint64_t ppn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    double dependency_ready_ns = materialized_ready_by_block_.at(block_index);
    std::unordered_set<std::uint64_t> commit_sequences;
    const auto pending = pending_physical_erases_.find(block_index);
    if (pending != pending_physical_erases_.end()) {
        dependency_ready_ns = std::max(
            dependency_ready_ns, pending->second.finish_ns);
        commit_sequences.insert(pending->second.commit_sequence);
    }
    if (dependency_ready_ns == 0.0) {
        return;
    }
    const auto stack = stack_of_block(block_index);
    if (dependency_ready_ns > at_ns) {
        trace_wait(
            spans,
            logic_entity(static_cast<std::uint32_t>(stack)),
            at_ns,
            dependency_ready_ns,
            "wait_target_block_erase");
        breakdown.scheduler_queue_wait_ns += dependency_ready_ns - at_ns;
        at_ns = dependency_ready_ns;
    }
    apply_selected_commits_through(commit_sequences, at_ns);
    materialized_ready_by_block_.at(block_index) = std::max(
        materialized_ready_by_block_.at(block_index), dependency_ready_ns);
    state_observation_by_stack_.at(stack) = std::max(
        state_observation_by_stack_.at(stack), at_ns);
}

void HbfDevice::tag_pending_mapping_updates_for_erase(
    std::size_t block_index,
    std::uint64_t retired_block_epoch,
    double ready_ns) {
    const auto tag = [this, block_index, retired_block_epoch, ready_ns](auto& table) {
        for (auto& [_, updates] : table) {
            for (auto& update : updates) {
                if (update.new_ppn / config_.pages_per_block == block_index &&
                    update.block_epoch == retired_block_epoch) {
                    update.destructive_ready_ns = std::max(
                        update.destructive_ready_ns, ready_ns);
                }
            }
        }
    };
    tag(pending_lpn_updates_);
    tag(pending_vpn_updates_);
}

void HbfDevice::retire_mapping_update_tombstones(
    std::size_t block_index,
    double ready_ns) {
    const auto retire = [this, block_index, ready_ns](
                            auto& table, auto& materialized_ready) {
        for (auto entry = table.begin(); entry != table.end();) {
            const bool retires_entry = std::any_of(
                entry->second.begin(),
                entry->second.end(),
                [this, block_index, ready_ns](const PendingMappingUpdate& update) {
                    return update.new_ppn / config_.pages_per_block == block_index &&
                        update.destructive_ready_ns != 0.0 &&
                        update.destructive_ready_ns <= ready_ns &&
                        update.commit_ns <= ready_ns;
                });
            std::erase_if(entry->second, [this, block_index, ready_ns](
                              const PendingMappingUpdate& update) {
                return update.new_ppn / config_.pages_per_block == block_index &&
                    update.destructive_ready_ns != 0.0 &&
                    update.destructive_ready_ns <= ready_ns &&
                    update.commit_ns <= ready_ns;
            });
            if (retires_entry) {
                materialized_ready[entry->first] = std::max(
                    materialized_ready[entry->first], ready_ns);
            }
            if (entry->second.empty()) {
                entry = table.erase(entry);
            } else {
                ++entry;
            }
        }
    };
    retire(pending_lpn_updates_, materialized_ready_by_lpn_);
    retire(pending_vpn_updates_, materialized_ready_by_vpn_);
}

std::optional<std::uint64_t> HbfDevice::visible_mapping_vpn_at(
    std::uint64_t mapping_vpn,
    double at_ns) const {
    std::optional<std::uint64_t> visible;
    if (const auto base = mapping_vpn_to_ppn_.find(mapping_vpn);
        base != mapping_vpn_to_ppn_.end()) {
        const auto page = programmed_pages_.find(base->second);
        const auto block_index = static_cast<std::size_t>(
            base->second / config_.pages_per_block);
        const auto& block = blocks_.at(block_index);
        if (page != programmed_pages_.end() &&
            page->second.status == PageStatus::Valid &&
            page->second.owner == PageOwner::Mapping &&
            page->second.block_epoch == block.epoch &&
            !block.erase_pending) {
            visible = base->second;
        }
    }
    if (!visible) {
        visible = compact_mapping_ppn(mapping_vpn);
    }
    const auto pending = pending_vpn_updates_.find(mapping_vpn);
    if (pending == pending_vpn_updates_.end()) {
        return visible;
    }
    const PendingMappingUpdate* newest = nullptr;
    for (const auto& update : pending->second) {
        const auto block_index = static_cast<std::size_t>(
            update.new_ppn / config_.pages_per_block);
        if (blocks_.at(block_index).epoch == update.block_epoch &&
            !blocks_.at(block_index).erase_pending &&
            update.commit_ns <= at_ns &&
            (newest == nullptr || std::tie(update.commit_ns, update.sequence) >
                std::tie(newest->commit_ns, newest->sequence))) {
            newest = &update;
        }
    }
    return newest == nullptr ? visible : std::optional<std::uint64_t>{newest->new_ppn};
}

void HbfDevice::schedule_lpn_mapping_commit(
    std::uint64_t lpn,
    std::uint64_t new_ppn,
    double commit_ns,
    std::uint64_t program_commit_sequence) {
    const auto sequence = next_commit_sequence_++;
    const auto block_index = static_cast<std::size_t>(
        new_ppn / config_.pages_per_block);
    const auto block_epoch = blocks_.at(block_index).epoch;
    blocks_.at(block_index).pending_mapping_publications++;
    pending_lpn_updates_[lpn].push_back(PendingMappingUpdate{
        .sequence = sequence,
        .program_commit_sequence = program_commit_sequence,
        .commit_ns = commit_ns,
        .new_ppn = new_ppn,
        .block_epoch = block_epoch,
    });
    pending_commits_.emplace(
        std::make_pair(commit_ns, sequence),
        PendingCommit{
        .stack = stack_for_lpn(lpn),
        .action = [this, lpn, new_ppn, block_index, block_epoch, sequence]() {
            const auto page = programmed_pages_.find(new_ppn);
            const bool target_survived =
                blocks_.at(block_index).epoch == block_epoch &&
                page != programmed_pages_.end() &&
                page->second.block_epoch == block_epoch &&
                page->second.status == PageStatus::Valid &&
                page->second.owner == PageOwner::Logical;
            if (target_survived) {
                const auto current = lpn_to_ppn_.find(lpn);
                if (current != lpn_to_ppn_.end() && current->second != new_ppn) {
                    invalidate_ppn(current->second);
                } else if (current == lpn_to_ppn_.end() &&
                           compact_lpn_ppn(lpn)) {
                    retire_compact_page(lpn, PageOwner::Logical);
                }
                lpn_to_ppn_[lpn] = new_ppn;
            }
            if (blocks_.at(block_index).epoch == block_epoch) {
                auto& pending_publications =
                    blocks_.at(block_index).pending_mapping_publications;
                if (pending_publications == 0) {
                    throw std::runtime_error(
                        "HBF LPN publication lost its block ownership pin");
                }
                pending_publications--;
            }
            auto pending = pending_lpn_updates_.find(lpn);
            if (pending != pending_lpn_updates_.end()) {
                std::erase_if(
                    pending->second,
                    [sequence](const PendingMappingUpdate& update) {
                        return update.sequence == sequence &&
                            update.destructive_ready_ns <= update.commit_ns;
                    });
                if (pending->second.empty()) {
                    pending_lpn_updates_.erase(pending);
                }
            }
        }});
}

void HbfDevice::schedule_vpn_mapping_commit(
    std::uint64_t mapping_vpn,
    std::uint64_t new_ppn,
    double commit_ns,
    std::uint64_t program_commit_sequence) {
    const auto sequence = next_commit_sequence_++;
    const auto block_index = static_cast<std::size_t>(
        new_ppn / config_.pages_per_block);
    const auto block_epoch = blocks_.at(block_index).epoch;
    blocks_.at(block_index).pending_mapping_publications++;
    pending_vpn_updates_[mapping_vpn].push_back(PendingMappingUpdate{
        .sequence = sequence,
        .program_commit_sequence = program_commit_sequence,
        .commit_ns = commit_ns,
        .new_ppn = new_ppn,
        .block_epoch = block_epoch,
    });
    pending_commits_.emplace(
        std::make_pair(commit_ns, sequence),
        PendingCommit{
        .stack = stack_for_vpn(mapping_vpn),
        .action = [this, mapping_vpn, new_ppn, block_index, block_epoch, sequence]() {
            const auto page = programmed_pages_.find(new_ppn);
            const bool target_survived =
                blocks_.at(block_index).epoch == block_epoch &&
                page != programmed_pages_.end() &&
                page->second.block_epoch == block_epoch &&
                page->second.status == PageStatus::Valid &&
                page->second.owner == PageOwner::Mapping;
            if (target_survived) {
                const auto current = mapping_vpn_to_ppn_.find(mapping_vpn);
                if (current != mapping_vpn_to_ppn_.end() && current->second != new_ppn) {
                    invalidate_ppn(current->second);
                } else if (
                    current == mapping_vpn_to_ppn_.end() &&
                    compact_mapping_ppn(mapping_vpn)) {
                    retire_compact_page(
                        metadata_lpn(mapping_vpn),
                        PageOwner::Mapping);
                }
                mapping_vpn_to_ppn_[mapping_vpn] = new_ppn;
            }
            if (blocks_.at(block_index).epoch == block_epoch) {
                auto& pending_publications =
                    blocks_.at(block_index).pending_mapping_publications;
                if (pending_publications == 0) {
                    throw std::runtime_error(
                        "HBF VPN publication lost its block ownership pin");
                }
                pending_publications--;
            }
            auto pending = pending_vpn_updates_.find(mapping_vpn);
            if (pending != pending_vpn_updates_.end()) {
                std::erase_if(
                    pending->second,
                    [sequence](const PendingMappingUpdate& update) {
                        return update.sequence == sequence &&
                            update.destructive_ready_ns <= update.commit_ns;
                    });
                if (pending->second.empty()) {
                    pending_vpn_updates_.erase(pending);
                }
            }
        }});
}

std::uint64_t HbfDevice::schedule_media_program_commit(
    std::uint64_t ppn,
    std::uint64_t lpn,
    PageOwner owner,
    double commit_ns) {
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    const auto block_epoch = blocks_.at(block_index).epoch;
    const auto page = programmed_pages_.find(ppn);
    if (page == programmed_pages_.end() ||
        page->second.status != PageStatus::Erased ||
        page->second.block_epoch != block_epoch) {
        throw std::runtime_error(
            "HBF media program commit was scheduled without an erased reservation");
    }
    return schedule_commit(
        stack_of_block(block_index),
        commit_ns,
        [this, ppn, lpn, owner, block_index, block_epoch]() {
            // An erase with a newer epoch is authoritative. A stale media
            // completion becomes a no-op instead of recreating page state.
            if (blocks_.at(block_index).epoch != block_epoch) {
                return;
            }
            const auto page = programmed_pages_.find(ppn);
            if (page == programmed_pages_.end() ||
                page->second.block_epoch != block_epoch) {
                return;
            }
            mark_programmed(ppn, lpn, owner);
        });
}

void HbfDevice::schedule_physical_program_commit(
    std::uint64_t ppn,
    double commit_ns) {
    const auto page = programmed_pages_.find(ppn);
    if (page == programmed_pages_.end() || page->second.status != PageStatus::Erased) {
        throw std::runtime_error(
            "HBF physical program commit was scheduled without an erased reservation");
    }
    const auto [pending, inserted] = pending_physical_programs_.emplace(
        ppn,
        PendingPhysicalProgram{.commit_ns = commit_ns});
    if (!inserted) {
        throw std::runtime_error("HBF physical page already has an in-flight program");
    }
    try {
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.pages_per_block);
        const auto block_epoch = blocks_.at(block_index).epoch;
        pending->second.commit_sequence = schedule_commit(
            stack_of_block(block_index), commit_ns,
            [this, ppn, block_index, block_epoch]() {
            const auto pending = pending_physical_programs_.find(ppn);
            if (pending == pending_physical_programs_.end()) {
                throw std::runtime_error(
                    "HBF physical program lost its in-flight reservation");
            }
            const auto page_at_commit = programmed_pages_.find(ppn);
            if (blocks_.at(block_index).epoch != block_epoch) {
                // A later physical erase already retired this incarnation;
                // the bytes were programmed and then destroyed, so only the
                // stale state publication is suppressed.
                pending_physical_programs_.erase(pending);
                return;
            }
            if (page_at_commit == programmed_pages_.end() ||
                page_at_commit->second.block_epoch != block_epoch ||
                page_at_commit->second.status != PageStatus::Erased) {
                throw std::runtime_error(
                    "HBF physical program target stopped being erased before commit");
            }
            mark_programmed(ppn, ppn, PageOwner::RawPhysical);
            materialized_ready_by_ppn_[ppn] = std::max(
                materialized_ready_by_ppn_[ppn], pending->second.commit_ns);
            pending_physical_programs_.erase(pending);
        });
    } catch (...) {
        pending_physical_programs_.erase(ppn);
        throw;
    }
}

void HbfDevice::reserve_physical_program_range(
    std::uint64_t first_ppn,
    std::uint64_t pages) {
    std::unordered_map<std::size_t, std::uint32_t> expected_next;
    std::unordered_map<std::size_t, std::uint32_t> remaining_free;

    // First pass is read-only: reject the whole multi-page request before any
    // block/page allocation state changes.
    for (std::uint64_t i = 0; i < pages; ++i) {
        const auto ppn = first_ppn + i;
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.pages_per_block);
        const auto page_index = static_cast<std::uint32_t>(
            ppn % config_.pages_per_block);
        const auto& block = blocks_.at(block_index);
        if (block.role != BlockRole::Free && block.role != BlockRole::RawPhysical) {
            throw std::runtime_error(
                block.role == BlockRole::StaticReadOnly ?
                "HBF physical write targets static read-only data" :
                "HBF physical write targets a controller-owned block");
        }
        auto [next_it, next_inserted] = expected_next.emplace(
            block_index, block.next_page);
        auto [free_it, free_inserted] = remaining_free.emplace(
            block_index, block.free_pages);
        (void)next_inserted;
        (void)free_inserted;
        if (page_index != next_it->second) {
            throw std::runtime_error(
                "HBF physical program violates sequential page order");
        }
        if (free_it->second == 0) {
            throw std::runtime_error("HBF physical program selected a full block");
        }
        if (programmed_pages_.contains(ppn) ||
            pending_physical_programs_.contains(ppn)) {
            throw std::runtime_error(
                "HBF physical write targets an allocated or in-flight page");
        }
        next_it->second++;
        free_it->second--;
    }

    for (const auto& entry : expected_next) {
        const auto block_index = entry.first;
        const auto& block = blocks_.at(block_index);
        if (block.role != BlockRole::Free) {
            continue;
        }
        const auto& free_blocks = planes_.at(block_plane_index(block_index)).free_blocks;
        if (std::find(free_blocks.begin(), free_blocks.end(), block_index) ==
            free_blocks.end()) {
            throw std::runtime_error(
                "HBF physical program found a free block missing from its plane pool");
        }
    }

    // Second pass reserves capacity exactly as the FTL allocator does. Page
    // validity remains Erased until each media-completion commit executes.
    for (std::uint64_t i = 0; i < pages; ++i) {
        const auto ppn = first_ppn + i;
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.pages_per_block);
        auto& block = blocks_.at(block_index);
        if (block.role == BlockRole::Free) {
            auto& free_blocks = planes_.at(block_plane_index(block_index)).free_blocks;
            const auto found = std::find(free_blocks.begin(), free_blocks.end(), block_index);
            free_blocks.erase(found);
            block.role = BlockRole::RawPhysical;
        }
        block.next_page++;
        block.free_pages--;
        block.pending_program_pages++;
        free_pages_--;
        free_pages_per_stack_.at(stack_of_block(block_index))--;
        const auto [page, inserted] = programmed_pages_.emplace(ppn, PageState{});
        if (!inserted) {
            throw std::runtime_error(
                "HBF physical program reservation unexpectedly reused page state");
        }
        page->second.status = PageStatus::Erased;
        page->second.owner = PageOwner::RawPhysical;
        page->second.lpn = 0;
        page->second.block_epoch = block.epoch;
    }
}

void HbfDevice::prepopulate_logical_pages(const std::vector<std::uint64_t>& lpns) {
    if (compact_logical_image_) {
        throw std::runtime_error(
            "HBF cannot add materialized mappings after a compact logical "
            "image");
    }
    std::unordered_set<std::uint64_t> seen;
    for (const auto lpn : lpns) {
        if (!seen.insert(lpn).second || lpn_to_ppn_.find(lpn) != lpn_to_ppn_.end()) {
            continue;
        }
        if (free_pages_ == 0) {
            throw std::runtime_error("HBF cannot prepopulate logical pages: physical capacity exhausted");
        }
        double at_ns = 0.0;
        Breakdown ignored;
        const auto ppn = allocate_free_page(
            at_ns,
            ignored,
            nullptr,
            BlockRole::Data,
            stack_for_lpn(lpn));
        lpn_to_ppn_[lpn] = ppn;
        mark_programmed(ppn, lpn);
        stats_.initial_logical_data_pages = checked_add(
            stats_.initial_logical_data_pages,
            1,
            "HBF materialized initial logical data pages");
        const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
        if (mapping_vpn_to_ppn_.find(mapping_vpn) == mapping_vpn_to_ppn_.end()) {
            const auto mapping_ppn = allocate_free_page(
                at_ns,
                ignored,
                nullptr,
                BlockRole::Mapping,
                stack_for_vpn(mapping_vpn),
                mapping_plane_for_vpn(mapping_vpn));
            mapping_vpn_to_ppn_[mapping_vpn] = mapping_ppn;
            mark_programmed(
                mapping_ppn,
                metadata_lpn(mapping_vpn),
                PageOwner::Mapping);
            stats_.initial_mapping_pages = checked_add(
                stats_.initial_mapping_pages,
                1,
                "HBF materialized initial mapping pages");
        }
    }
    stats_.free_pages = free_pages_;
    refresh_parallel_stats();
}

void HbfDevice::prepopulate_read_only_logical_page_range(
    std::uint64_t first_lpn,
    std::uint64_t page_count) {
    prepopulate_compact_logical_page_range(
        first_lpn,
        page_count,
        false);
}

void HbfDevice::prepopulate_mutable_logical_page_range(
    std::uint64_t first_lpn,
    std::uint64_t page_count) {
    prepopulate_compact_logical_page_range(
        first_lpn,
        page_count,
        true);
}

void HbfDevice::prepopulate_compact_logical_page_range(
    std::uint64_t first_lpn,
    std::uint64_t page_count,
    bool mutable_image) {
    if (page_count == 0) {
        return;
    }
    const auto last_lpn = checked_add(
        first_lpn,
        page_count - 1,
        "HBF prepopulation LPN range");
    if (is_metadata_lpn(last_lpn)) {
        throw std::runtime_error(
            "HBF compact logical image enters the metadata LPN namespace");
    }
    if (compact_logical_image_ || !lpn_to_ppn_.empty() ||
        !mapping_vpn_to_ppn_.empty()) {
        throw std::runtime_error(
            "HBF compact logical image requires an empty FTL");
    }
    if (!mutable_image &&
        (!programmed_pages_.empty() || free_pages_ != total_pages_)) {
        throw std::runtime_error(
            "HBF compact read-only image must be installed into a fresh "
            "device");
    }
    if (mutable_image) {
        const bool static_only = std::all_of(
            programmed_pages_.begin(),
            programmed_pages_.end(),
            [](const auto& entry) {
                return entry.second.status == PageStatus::StaticReadOnly &&
                    entry.second.owner == PageOwner::StaticReadOnly;
            });
        const bool no_mutable_blocks = std::none_of(
            blocks_.begin(),
            blocks_.end(),
            [](const BlockState& block) {
                return block.role == BlockRole::Data ||
                    block.role == BlockRole::Mapping ||
                    block.role == BlockRole::GC ||
                    block.role == BlockRole::RawPhysical ||
                    block.pending_program_pages != 0 ||
                    block.pending_mapping_publications != 0 ||
                    block.erase_pending;
            });
        if (!static_only || !no_mutable_blocks ||
            !pending_commits_.empty() ||
            !pending_lpn_updates_.empty() ||
            !pending_vpn_updates_.empty() ||
            !dirty_mapping_vpns_.empty()) {
            throw std::runtime_error(
                "HBF compact mutable image must be installed after static "
                "fencing and before mutable FTL work");
        }
    }
    const auto entries_per_mapping_page =
        static_cast<std::uint64_t>(config_.mapping_entries_per_page);
    const auto pps = static_cast<std::uint64_t>(planes_per_stack());
    if (pps == 0) {
        throw std::runtime_error(
            "HBF compact logical image requires at least one plane per stack");
    }
    const auto cursor_was_advanced = [](const auto& cursors) {
        return std::any_of(
            cursors.begin(),
            cursors.end(),
            [](std::size_t cursor) { return cursor != 0; });
    };
    if (cursor_was_advanced(next_data_allocation_plane_per_stack_) ||
        cursor_was_advanced(next_mapping_allocation_plane_per_stack_) ||
        cursor_was_advanced(next_gc_allocation_plane_per_stack_)) {
        throw std::runtime_error(
            "HBF compact logical image requires fresh allocation cursors");
    }

    CompactLogicalImage image;
    image.mutable_image = mutable_image;
    image.first_lpn = first_lpn;
    image.page_count = page_count;
    const auto stacks = static_cast<std::uint64_t>(config_.stacks);
    const auto first_group =
        (first_lpn / stacks) / entries_per_mapping_page;
    const auto last_group =
        (last_lpn / stacks) / entries_per_mapping_page;
    image.first_vpn = checked_mul(
        first_group,
        stacks,
        "HBF compact first mapping VPN");
    const auto group_count = checked_add(
        last_group - first_group,
        1,
        "HBF compact mapping group count");
    image.vpn_slot_count = checked_mul(
        group_count,
        stacks,
        "HBF compact VPN slot count");
    if (image.vpn_slot_count > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error(
            "HBF compact logical image VPN directory exceeds size_t range");
    }
    image.data_blocks_by_plane.resize(planes_.size());
    image.mapping_ppns.resize(
        static_cast<std::size_t>(image.vpn_slot_count));
    image.vpn_ranges.resize(
        static_cast<std::size_t>(image.vpn_slot_count));
    image.vpn_offsets_by_stack.resize(config_.stacks);
    std::vector<std::uint64_t> stack_page_counts(config_.stacks, 0);

    for (std::uint64_t vpn_offset = 0;
         vpn_offset < image.vpn_slot_count;
         ++vpn_offset) {
        const auto mapping_vpn = checked_add(
            image.first_vpn, vpn_offset, "HBF compact mapping VPN");
        const auto group = mapping_vpn / stacks;
        const auto stack = stack_for_vpn(mapping_vpn);
        const auto rotation = placement_mix64(group) % stacks;
        const auto lane = static_cast<std::uint64_t>(stack) >= rotation ?
            static_cast<std::uint64_t>(stack) - rotation :
            stacks - (rotation - static_cast<std::uint64_t>(stack));
        const auto group_first_stripe = checked_mul(
            group,
            entries_per_mapping_page,
            "HBF compact mapping-group first stripe");
        const auto group_last_stripe = checked_add(
            group_first_stripe,
            entries_per_mapping_page - 1,
            "HBF compact mapping-group last stripe");
        const auto first_candidate_stripe = [&] {
            if (first_lpn <= lane) {
                return std::uint64_t{0};
            }
            const auto delta = first_lpn - lane;
            return delta / stacks + (delta % stacks == 0 ? 0 : 1);
        }();
        if (last_lpn < lane) {
            continue;
        }
        const auto last_candidate_stripe = (last_lpn - lane) / stacks;
        const auto range_first_stripe = std::max(
            group_first_stripe,
            first_candidate_stripe);
        const auto range_last_stripe = std::min(
            group_last_stripe,
            last_candidate_stripe);
        if (range_first_stripe > range_last_stripe) {
            continue;
        }
        const auto first_entry =
            range_first_stripe - group_first_stripe;
        const auto range_pages = checked_add(
            range_last_stripe - range_first_stripe,
            1,
            "HBF compact mapping-page range");
        if (range_pages > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "HBF compact mapping-page range exceeds supported page count");
        }
        const auto stack_plane_base = stack * planes_per_stack();
        const auto first_data_cursor =
            static_cast<std::uint64_t>(next_data_allocation_plane_per_stack_.at(stack));
        if (first_data_cursor != stack_page_counts.at(stack) % pps) {
            throw std::runtime_error(
                "HBF compact data-allocation cursor lost round-robin order");
        }
        image.vpn_ranges.at(static_cast<std::size_t>(vpn_offset)) =
            CompactLogicalImage::VpnRange{
            .first_entry = first_entry,
            .page_count = range_pages,
            .stack_page_offset = stack_page_counts.at(stack),
        };
        image.vpn_offsets_by_stack.at(stack).push_back(vpn_offset);

        // In the ordinary prepopulation loop the first data page is reserved
        // before the mapping page for this VPN. Preserve that ordering because
        // Data and Mapping may draw their first blocks from the same plane's
        // free-block deque.
        (void)allocate_compact_pages_on_plane(
            stack_plane_base + static_cast<std::size_t>(first_data_cursor),
            BlockRole::Data,
            1,
            &image.data_blocks_by_plane.at(
                stack_plane_base + static_cast<std::size_t>(first_data_cursor)),
            &image.live_data_pages_by_block);
        image.mapping_ppns.at(static_cast<std::size_t>(vpn_offset)) =
            allocate_compact_pages_on_plane(
            mapping_plane_for_vpn(mapping_vpn),
            BlockRole::Mapping,
            1,
            nullptr,
            &image.live_mapping_pages_by_block);
        image.mapping_page_count = checked_add(
            image.mapping_page_count,
            1,
            "HBF compact mapping-page count");

        for (std::size_t local_plane = 0;
             local_plane < planes_per_stack();
             ++local_plane) {
            const auto plane_distance =
                (local_plane + planes_per_stack() -
                    static_cast<std::size_t>(first_data_cursor)) %
                planes_per_stack();
            std::uint64_t plane_pages = range_pages > plane_distance ?
                1 + (range_pages - 1 - plane_distance) / pps : 0;
            if (local_plane == first_data_cursor) {
                --plane_pages;
            }
            if (plane_pages == 0) {
                continue;
            }
            const auto plane = stack_plane_base + local_plane;
            (void)allocate_compact_pages_on_plane(
                plane,
                BlockRole::Data,
                static_cast<std::uint32_t>(plane_pages),
                &image.data_blocks_by_plane.at(plane),
                &image.live_data_pages_by_block);
        }
        stack_page_counts.at(stack) = checked_add(
            stack_page_counts.at(stack),
            range_pages,
            "HBF compact per-stack data pages");
        next_data_allocation_plane_per_stack_.at(stack) =
            static_cast<std::size_t>(stack_page_counts.at(stack) % pps);
    }
    for (std::size_t plane = 0;
         plane < image.data_blocks_by_plane.size();
         ++plane) {
        const auto& assigned = image.data_blocks_by_plane.at(plane);
        for (std::size_t ordinal = 0; ordinal < assigned.size(); ++ordinal) {
            const bool inserted = image.data_block_locations.emplace(
                assigned[ordinal],
                CompactLogicalImage::DataBlockLocation{
                    .plane = plane,
                    .block_ordinal = ordinal,
                }).second;
            if (!inserted) {
                throw std::runtime_error(
                    "HBF compact data block appears in multiple directories");
            }
        }
    }
    for (std::size_t offset = 0;
         offset < image.mapping_ppns.size();
         ++offset) {
        if (!image.mapping_ppns[offset]) {
            continue;
        }
        const auto mapping_vpn = checked_add(
            image.first_vpn,
            offset,
            "HBF compact mapping inverse VPN");
        const bool inserted = image.mapping_vpn_by_ppn.emplace(
            *image.mapping_ppns[offset],
            mapping_vpn).second;
        if (!inserted) {
            throw std::runtime_error(
                "HBF compact mapping PPN is not unique");
        }
    }
    stats_.initial_logical_data_pages = checked_add(
        stats_.initial_logical_data_pages,
        image.page_count,
        "HBF initial logical data pages");
    stats_.initial_mapping_pages = checked_add(
        stats_.initial_mapping_pages,
        image.mapping_page_count,
        "HBF initial mapping pages");
    stats_.compact_initial_logical_data_pages = checked_add(
        stats_.compact_initial_logical_data_pages,
        image.page_count,
        "HBF compact initial logical data pages");
    stats_.compact_initial_mapping_pages = checked_add(
        stats_.compact_initial_mapping_pages,
        image.mapping_page_count,
        "HBF compact initial mapping pages");
    compact_logical_image_ = std::move(image);
    stats_.free_pages = free_pages_;
    refresh_parallel_stats();
}

void HbfDevice::reserve_static_physical_blocks(
    const std::vector<std::size_t>& block_indices) {
    if (compact_logical_image_) {
        throw std::runtime_error(
            "HBF cannot reserve static pages after a compact logical image");
    }
    std::unordered_set<std::size_t> touched_blocks(
        block_indices.begin(),
        block_indices.end());
    std::vector<std::uint8_t> reserve_mask(blocks_.size(), 0);
    std::vector<std::size_t> reserve_count_by_plane(planes_.size(), 0);
    std::size_t blocks_to_reserve = 0;
    for (const auto block_index : touched_blocks) {
        if (block_index >= blocks_.size()) {
            throw std::runtime_error(
                "HBF static-data block index is out of range");
        }
    }

    // Validate the entire request before changing any pool, role, bitmap, or
    // capacity counter. A mixed legal/illegal list must fail closed rather
    // than leave whichever unordered-set element happened to be visited first
    // as a partially reserved static block.
    for (const auto block_index : touched_blocks) {
        const auto& block = blocks_.at(block_index);
        if (block.role == BlockRole::StaticReadOnly) {
            continue;
        }
        if (block.role != BlockRole::Free || block.free_pages != config_.pages_per_block ||
            block.valid_pages != 0 || block.invalid_pages != 0 || block.next_page != 0 ||
            block.pending_program_pages != 0 ||
            block.pending_mapping_publications != 0) {
            throw std::runtime_error(
                "HBF static-data reservation intersects live FTL state at block " +
                std::to_string(block_index));
        }
        reserve_mask.at(block_index) = 1;
        ++blocks_to_reserve;
        ++reserve_count_by_plane.at(block_plane_index(block_index));
    }
    // Dense extents can cover hundreds of thousands of blocks. Validate pool
    // membership in one linear scan instead of performing one O(blocks-per-
    // plane) find for every reserved block.
    std::size_t pool_matches = 0;
    for (const auto& plane : planes_) {
        for (const auto block_index : plane.free_blocks) {
            pool_matches += reserve_mask.at(block_index) != 0 ? 1 : 0;
        }
    }
    if (pool_matches != blocks_to_reserve) {
        throw std::runtime_error(
            "HBF free block is missing from its plane pool");
    }

    // Reserve complete NAND blocks. Page-granular fencing would let the FTL
    // program around immutable wordlines and later collide with them, which is
    // not a valid sequential-program model.
    for (std::size_t plane_index = 0;
         plane_index < planes_.size();
         ++plane_index) {
        if (reserve_count_by_plane.at(plane_index) == 0) {
            continue;
        }
        auto& free_blocks = planes_.at(plane_index).free_blocks;
        const auto first_removed = std::remove_if(
            free_blocks.begin(),
            free_blocks.end(),
            [&](std::size_t block_index) {
                return reserve_mask.at(block_index) != 0;
            });
        const auto removed = static_cast<std::size_t>(
            std::distance(first_removed, free_blocks.end()));
        if (removed != reserve_count_by_plane.at(plane_index)) {
            throw std::runtime_error(
                "HBF static-data free-pool removal count mismatch");
        }
        free_blocks.erase(first_removed, free_blocks.end());
    }
    for (const auto block_index : touched_blocks) {
        auto& block = blocks_.at(block_index);
        if (block.role == BlockRole::StaticReadOnly) {
            continue;
        }
        block.role = BlockRole::StaticReadOnly;
        block.free_pages = 0;
        block.next_page = config_.pages_per_block;
        free_pages_ -= config_.pages_per_block;
        free_pages_per_stack_.at(stack_of_block(block_index)) -= config_.pages_per_block;
        stats_.static_reserved_pages += config_.pages_per_block;
    }
    stats_.free_pages = free_pages_;
}

void HbfDevice::reserve_static_physical_pages(
    const std::vector<std::uint64_t>& ppns) {
    std::unordered_set<std::uint64_t> unique_pages;
    std::vector<std::size_t> touched_blocks;
    touched_blocks.reserve(ppns.size());
    for (const auto ppn : ppns) {
        if (ppn >= total_pages_) {
            throw std::runtime_error("HBF static-data PPN is out of range");
        }
        if (unique_pages.insert(ppn).second) {
            touched_blocks.push_back(
                static_cast<std::size_t>(ppn / config_.pages_per_block));
        }
    }
    // Page conflicts must be checked before the block helper mutates any
    // ownership state.
    for (const auto ppn : unique_pages) {
        const auto page = programmed_pages_.find(ppn);
        if (page != programmed_pages_.end() &&
            page->second.status != PageStatus::Erased &&
            page->second.status != PageStatus::StaticReadOnly) {
            throw std::runtime_error(
                "HBF static-data reservation intersects a programmed page");
        }
    }
    reserve_static_physical_blocks(touched_blocks);
    for (const auto ppn : unique_pages) {
        auto& page = programmed_pages_[ppn];
        if (page.status == PageStatus::Erased) {
            page.status = PageStatus::StaticReadOnly;
            page.owner = PageOwner::StaticReadOnly;
            page.lpn = ppn;
            auto& block = blocks_.at(static_cast<std::size_t>(ppn / config_.pages_per_block));
            page.block_epoch = block.epoch;
            block.valid_pages++;
            block.set_valid(static_cast<std::uint32_t>(ppn % config_.pages_per_block));
        }
    }
    stats_.free_pages = free_pages_;
}

void HbfDevice::reserve_static_physical_block_indices(
    const std::vector<std::size_t>& block_indices) {
    reserve_static_physical_blocks(block_indices);
}

void HbfDevice::reserve_static_physical_block_extent(
    std::uint32_t first_block,
    std::uint32_t block_count) {
    const auto end_block = static_cast<std::uint64_t>(first_block) +
        static_cast<std::uint64_t>(block_count);
    if (end_block > config_.blocks_per_plane) {
        throw std::runtime_error(
            "HBF static-data block extent exceeds blocks-per-plane geometry");
    }
    if (block_count == 0) {
        return;
    }
    std::vector<std::size_t> block_indices;
    if (planes_.size() >
        std::numeric_limits<std::size_t>::max() / block_count) {
        throw std::runtime_error(
            "HBF static-data block extent size exceeds size_t range");
    }
    block_indices.reserve(
        planes_.size() * static_cast<std::size_t>(block_count));
    for (std::size_t plane = 0; plane < planes_.size(); ++plane) {
        const auto plane_base =
            plane * static_cast<std::size_t>(config_.blocks_per_plane);
        for (std::uint64_t block = first_block; block < end_block; ++block) {
            block_indices.push_back(
                plane_base + static_cast<std::size_t>(block));
        }
    }
    reserve_static_physical_blocks(block_indices);
}

PhysicalCompletion HbfDevice::drain_pending(
    std::string id,
    double arrival_ns,
    TraceConfig trace) {
    PhysicalCompletion out;
    out.id = std::move(id);
    out.tier = Tier::HBF;
    out.op = Op::Write;
    out.arrival_ns = arrival_ns;
    out.start_ns = arrival_ns;
    out.logical_bytes = 0;
    out.resource_path = "logic/write_buffer+mapping_table";
    auto* trace_spans = trace_spans_enabled(trace) ? &out.spans : nullptr;
    const auto physical_bytes_before = stats_.physical_read_bytes + stats_.physical_write_bytes;

    if (!std::isfinite(arrival_ns) || arrival_ns < 0.0) {
        throw std::runtime_error("HBF drain arrival must be finite and non-negative");
    }
    if (last_issue_arrival_ns_ && arrival_ns < *last_issue_arrival_ns_) {
        throw std::runtime_error("HBF drain cannot precede the last issue/drain arrival");
    }
    // drain_pending is a top-level causal barrier. Recording its arrival also
    // prevents a later issue from travelling behind calendar history that the
    // drain is now allowed to reclaim.
    last_issue_arrival_ns_ = arrival_ns;
    reservation_causal_watermark_ns_ = arrival_ns;
    double causal_ready_ns = 0.0;
    for (const auto ready_ns : causal_state_ready_by_stack_) {
        causal_ready_ns = std::max(causal_ready_ns, ready_ns);
    }
    double at_ns = std::max({arrival_ns, background_finish_ns_, causal_ready_ns});
    apply_commits_through(at_ns);
    std::fill(
        state_observation_by_stack_.begin(),
        state_observation_by_stack_.end(),
        at_ns);
    flush_all_write_buffer_entries(at_ns, out.breakdown, trace_spans);
    flush_all_dirty_mapping_pages(at_ns, out.breakdown, trace_spans);
    at_ns = std::max(at_ns, background_finish_ns_);
    if (!pending_commits_.empty()) {
        at_ns = std::max(at_ns, pending_commits_.rbegin()->first.first);
    }
    apply_commits_through(at_ns);
    out.finish_ns = at_ns;
    for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
        causal_state_ready_by_stack_[stack] = std::max(
            causal_state_ready_by_stack_[stack], out.finish_ns);
        state_observation_by_stack_[stack] = std::max(
            state_observation_by_stack_[stack], out.finish_ns);
    }
    out.note = out.finish_ns > out.arrival_ns ? "drained-pending-hbf-state" : "no-pending-hbf-state";
    out.physical_bytes = stats_.physical_read_bytes + stats_.physical_write_bytes -
        physical_bytes_before;

    stats_.free_pages = free_pages_;
    stats_.stage_work += out.breakdown;
    stats_.finish_ns = std::max({stats_.finish_ns, out.finish_ns, background_finish_ns_});
    return out;
}

HbfAuditSnapshot HbfDevice::audit_snapshot() const {
    HbfAuditSnapshot snapshot;
    snapshot.free_pages = free_pages_;
    snapshot.free_pages_per_stack = free_pages_per_stack_;
    snapshot.data_allocation_cursors.assign(
        next_data_allocation_plane_per_stack_.begin(),
        next_data_allocation_plane_per_stack_.end());
    snapshot.mapping_allocation_cursors.assign(
        next_mapping_allocation_plane_per_stack_.begin(),
        next_mapping_allocation_plane_per_stack_.end());
    snapshot.gc_allocation_cursors.assign(
        next_gc_allocation_plane_per_stack_.begin(),
        next_gc_allocation_plane_per_stack_.end());

    snapshot.logical_mappings.reserve(lpn_to_ppn_.size());
    for (const auto& [lpn, ppn] : lpn_to_ppn_) {
        snapshot.logical_mappings.push_back({.key = lpn, .ppn = ppn});
    }
    std::sort(
        snapshot.logical_mappings.begin(),
        snapshot.logical_mappings.end(),
        [](const HbfAuditMapping& lhs, const HbfAuditMapping& rhs) {
            return std::tie(lhs.key, lhs.ppn) < std::tie(rhs.key, rhs.ppn);
        });

    snapshot.mapping_pages.reserve(mapping_vpn_to_ppn_.size());
    for (const auto& [vpn, ppn] : mapping_vpn_to_ppn_) {
        snapshot.mapping_pages.push_back({.key = vpn, .ppn = ppn});
    }
    std::sort(
        snapshot.mapping_pages.begin(),
        snapshot.mapping_pages.end(),
        [](const HbfAuditMapping& lhs, const HbfAuditMapping& rhs) {
            return std::tie(lhs.key, lhs.ppn) < std::tie(rhs.key, rhs.ppn);
        });

    const auto page_status_name = [](PageStatus status) -> std::string {
        switch (status) {
        case PageStatus::Erased:
            return "erased";
        case PageStatus::StaticReadOnly:
            return "static_read_only";
        case PageStatus::Valid:
            return "valid";
        case PageStatus::Invalid:
            return "invalid";
        }
        throw std::runtime_error("unknown HBF page status");
    };
    const auto page_owner_name = [](PageOwner owner) -> std::string {
        switch (owner) {
        case PageOwner::Unassigned:
            return "unassigned";
        case PageOwner::Logical:
            return "logical";
        case PageOwner::Mapping:
            return "mapping";
        case PageOwner::RawPhysical:
            return "raw_physical";
        case PageOwner::StaticReadOnly:
            return "static_read_only";
        }
        throw std::runtime_error("unknown HBF page owner");
    };
    snapshot.materialized_pages.reserve(programmed_pages_.size());
    for (const auto& [ppn, page] : programmed_pages_) {
        snapshot.materialized_pages.push_back({
            .ppn = ppn,
            .status = page_status_name(page.status),
            .owner = page_owner_name(page.owner),
            .logical_key = page.lpn,
            .block_epoch = page.block_epoch,
        });
    }
    std::sort(
        snapshot.materialized_pages.begin(),
        snapshot.materialized_pages.end(),
        [](const HbfAuditPage& lhs, const HbfAuditPage& rhs) {
            return lhs.ppn < rhs.ppn;
        });

    const auto block_role_name = [](BlockRole role) -> std::string {
        switch (role) {
        case BlockRole::Free:
            return "free";
        case BlockRole::StaticReadOnly:
            return "static_read_only";
        case BlockRole::RawPhysical:
            return "raw_physical";
        case BlockRole::Data:
            return "data";
        case BlockRole::Mapping:
            return "mapping";
        case BlockRole::GC:
            return "gc";
        }
        throw std::runtime_error("unknown HBF block role");
    };
    snapshot.blocks.reserve(blocks_.size());
    for (std::size_t index = 0; index < blocks_.size(); ++index) {
        const auto& block = blocks_[index];
        snapshot.blocks.push_back({
            .block = index,
            .role = block_role_name(block.role),
            .valid_pages = block.valid_pages,
            .invalid_pages = block.invalid_pages,
            .free_pages = block.free_pages,
            .next_page = block.next_page,
            .erase_count = block.erase_count,
            .pending_program_pages = block.pending_program_pages,
            .pending_mapping_publications =
                block.pending_mapping_publications,
            .epoch = block.epoch,
            .erase_pending = block.erase_pending,
        });
    }

    snapshot.dirty_mapping_vpns.assign(
        dirty_mapping_vpns_.begin(), dirty_mapping_vpns_.end());
    std::sort(
        snapshot.dirty_mapping_vpns.begin(),
        snapshot.dirty_mapping_vpns.end());
    for (const auto& [_, events] : pending_dirty_mapping_events_) {
        snapshot.pending_dirty_mapping_events += events.size();
    }
    for (const auto& [_, updates] : pending_lpn_updates_) {
        snapshot.pending_lpn_updates += updates.size();
    }
    for (const auto& [_, updates] : pending_vpn_updates_) {
        snapshot.pending_vpn_updates += updates.size();
    }
    snapshot.pending_commits = pending_commits_.size();
    for (const auto& buffer : write_buffer_by_stack_) {
        snapshot.write_buffer_entries += buffer.size();
    }
    for (const auto& [_, generations] : inflight_buffered_writes_) {
        snapshot.inflight_buffered_generations += generations.size();
    }
    snapshot.pending_physical_programs = pending_physical_programs_.size();
    snapshot.pending_physical_erases = pending_physical_erases_.size();
    return snapshot;
}

PhysicalCompletion HbfDevice::issue(const PhysicalRequest& request) {
    if (request.tier != Tier::HBF) {
        throw std::runtime_error("HbfDevice received non-HBF request");
    }
    // Validate provenance before any request state changes. A malformed enum
    // must not be retained in a deferred buffer and fail only at a later
    // eviction or drain.
    (void)resolve_heatmap_source(
        TransactionSource::User, request.heatmap_source);
    if (request.bytes == 0 && request.op != Op::Erase) {
        throw std::runtime_error("HBF request bytes must be positive");
    }
    if (request.op != Op::Erase &&
        request.bytes - 1 > std::numeric_limits<std::uint64_t>::max() - request.addr) {
        throw std::runtime_error("HBF request address range overflows uint64_t");
    }
    if (request.op == Op::Refresh) {
        throw std::runtime_error("HBF v0 does not model background refresh");
    }
    if (!std::isfinite(request.arrival_ns) || request.arrival_ns < 0.0) {
        throw std::runtime_error("HBF request arrival must be finite and non-negative");
    }
    const bool physical_request = request.address_space == AddressSpace::Physical ||
        request.op == Op::Erase;
    if (compact_logical_image_ &&
        !compact_logical_image_->mutable_image &&
        (request.op != Op::Read || physical_request)) {
        throw std::runtime_error(
            "HBF compact read-only initial image accepts logical reads only");
    }
    if (!physical_request) {
        const auto last_addr = request.addr + (request.bytes - 1);
        const auto last_lpn = last_addr / config_.page_size_bytes;
        if (is_metadata_lpn(last_lpn)) {
            throw std::runtime_error(
                "HBF logical request exceeds the user-LPN namespace reserved below bit 63");
        }
    } else {
        const auto capacity_bytes = checked_mul(total_pages_, config_.page_size_bytes,
            "HBF physical capacity bytes");
        if (request.addr >= capacity_bytes ||
            (request.op != Op::Erase && request.bytes > capacity_bytes - request.addr)) {
            throw std::runtime_error("HBF physical request range is out of capacity");
        }
    }
    if (last_issue_arrival_ns_ && request.arrival_ns < *last_issue_arrival_ns_) {
        throw std::runtime_error(
            "HBF requests and drains must be issued in nondecreasing arrival order; "
            "enqueue/sort the workload before simulation so state cannot travel backward "
            "in time");
    }
    // Once a request reaches state-dependent validation, its arrival becomes
    // a causal barrier even if the target is rejected. This prevents a caught
    // future-time error from being followed by an earlier request after prior
    // commits have already been materialized.
    last_issue_arrival_ns_ = request.arrival_ns;
    reservation_causal_watermark_ns_ = request.arrival_ns;
    const auto request_stack = physical_request ?
        decode(request.addr).stack :
        static_cast<std::uint32_t>(
            stack_for_lpn(request.addr / config_.page_size_bytes));
    std::vector<bool> touched_stacks(config_.stacks, false);
    touched_stacks.at(request_stack) = true;
    if (request.op != Op::Erase && physical_request) {
        const auto last_stack = decode(request.addr + request.bytes - 1).stack;
        for (std::size_t stack = request_stack; stack <= last_stack; ++stack) {
            touched_stacks.at(stack) = true;
        }
    } else if (!physical_request) {
        const auto first_lpn = request.addr / config_.page_size_bytes;
        const auto last_lpn =
            (request.addr + request.bytes - 1) / config_.page_size_bytes;
        for (auto lpn = first_lpn;; ++lpn) {
            touched_stacks.at(stack_for_lpn(lpn)) = true;
            if (lpn == last_lpn) {
                break;
            }
        }
    }
    std::vector<double> state_ready_by_stack(
        config_.stacks, request.arrival_ns);
    std::vector<std::unordered_set<std::uint64_t>> selected_commits_by_stack(
        config_.stacks);
    if (physical_request) {
        const auto first_block = block_index(decode(request.addr));
        const auto last_block = request.op == Op::Erase ? first_block :
            block_index(decode(request.addr + request.bytes - 1));
        for (auto block = first_block; block <= last_block; ++block) {
            const auto stack = stack_of_block(block);
            state_ready_by_stack[stack] = std::max(
                state_ready_by_stack[stack],
                materialized_ready_by_block_.at(block));
            const auto pending = pending_physical_erases_.find(block);
            if (pending != pending_physical_erases_.end()) {
                state_ready_by_stack[stack] = std::max(
                    state_ready_by_stack[stack], pending->second.finish_ns);
                selected_commits_by_stack[stack].insert(
                    pending->second.commit_sequence);
            }
        }
        if (request.op != Op::Erase) {
            const auto first_ppn = encode_ppn(decode(request.addr));
            const auto pages = page_count_for(decode(request.addr), request.bytes);
            for (std::uint64_t i = 0; i < pages; ++i) {
                const auto ppn = first_ppn + i;
                const auto stack = stack_of_block(static_cast<std::size_t>(
                    ppn / config_.pages_per_block));
                if (const auto observed = materialized_ready_by_ppn_.find(ppn);
                    observed != materialized_ready_by_ppn_.end()) {
                    state_ready_by_stack[stack] = std::max(
                        state_ready_by_stack[stack], observed->second);
                }
                if (const auto program = pending_physical_programs_.find(ppn);
                    program != pending_physical_programs_.end()) {
                    state_ready_by_stack[stack] = std::max(
                        state_ready_by_stack[stack], program->second.commit_ns);
                    selected_commits_by_stack[stack].insert(
                        program->second.commit_sequence);
                }
            }
        }
    } else {
        const auto first_lpn = request.addr / config_.page_size_bytes;
        const auto last_lpn =
            (request.addr + request.bytes - 1) / config_.page_size_bytes;
        for (auto lpn = first_lpn; lpn <= last_lpn; ++lpn) {
            const auto stack = stack_for_lpn(lpn);
            if (const auto observed = materialized_ready_by_lpn_.find(lpn);
                observed != materialized_ready_by_lpn_.end()) {
                state_ready_by_stack[stack] = std::max(
                    state_ready_by_stack[stack], observed->second);
            }
            const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
            if (const auto observed = materialized_ready_by_vpn_.find(mapping_vpn);
                observed != materialized_ready_by_vpn_.end()) {
                state_ready_by_stack[stack] = std::max(
                    state_ready_by_stack[stack], observed->second);
            }
        }
    }
    // Materialize every causally prior state transition before validating a
    // physical target. Otherwise a completed-but-not-yet-applied program
    // could make the same PPN appear erased and permit illegal reprogramming.
    // Globally materialize only events that have truly completed by host
    // arrival (or a whole-stack causal GC/drain barrier). Future block/LPN/VPN
    // dependencies apply only their selected callback chain; applying the
    // whole stack would make independent planes depend on API call order.
    apply_commits_through(request.arrival_ns);
    for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
        if (!touched_stacks[stack]) {
            continue;
        }
        const double causal_ready_ns = causal_state_ready_by_stack_.at(stack);
        if (causal_ready_ns > request.arrival_ns) {
            apply_stack_commits_through(stack, causal_ready_ns);
        }
        const double ready_ns = std::max(
            state_ready_by_stack[stack], causal_ready_ns);
        state_ready_by_stack[stack] = ready_ns;
        apply_selected_commits_through(
            selected_commits_by_stack[stack], ready_ns);
        state_observation_by_stack_.at(stack) = ready_ns;
    }
    const double ingress_state_ready_ns = state_ready_by_stack.at(request_stack);
    if (physical_request && request.op == Op::Write) {
        const auto first = encode_ppn(decode(request.addr));
        const auto pages = page_count_for(decode(request.addr), request.bytes);
        reserve_physical_program_range(first, pages);
    } else if (physical_request && request.op == Op::Erase) {
        const auto block = block_index(decode(request.addr));
        if (pending_physical_erases_.contains(block) ||
            blocks_.at(block).erase_pending) {
            throw std::runtime_error("HBF block already has an in-flight physical erase");
        }
        if (blocks_.at(block).role == BlockRole::StaticReadOnly) {
            throw std::runtime_error("HBF erase targets a static read-only block");
        }
    }
    stats_.first_arrival_ns = std::min(stats_.first_arrival_ns, request.arrival_ns);

    PhysicalCompletion out;
    out.id = request.id;
    out.tier = Tier::HBF;
    out.op = request.op;
    out.arrival_ns = request.arrival_ns;
    out.logical_bytes = request.bytes;
    auto* trace_spans = trace_spans_enabled(request.trace) ? &out.spans : nullptr;
    const auto physical_bytes_before = stats_.physical_read_bytes + stats_.physical_write_bytes;
    if (ingress_state_ready_ns > request.arrival_ns) {
        out.breakdown.scheduler_queue_wait_ns +=
            ingress_state_ready_ns - request.arrival_ns;
        trace_wait(
            trace_spans,
            logic_entity(static_cast<std::uint32_t>(request_stack)),
            request.arrival_ns,
            ingress_state_ready_ns,
            "wait_stack_state_dependency");
    }

    const auto ingress_stack = request_stack;
    auto& ingress_logic = logic_dies_.at(ingress_stack);
    const double request_command_done = schedule_external_request_command(
        ingress_stack,
        ingress_state_ready_ns,
        out.breakdown,
        trace_spans,
        request.id);
    const auto ingress_slot = reserve(
        request_command_done,
        config_.logic_scheduler_issue_ns,
        ingress_logic.ingress);
    const double logic_start = ingress_slot.start_ns;
    out.breakdown.ingress_queue_wait_ns = ingress_slot.wait_ns;
    add_trace_span(
        trace_spans,
        "logic_die_queue",
        "queue",
        logic_entity(ingress_stack),
        request_command_done,
        logic_start);
    double issued_ns = ingress_slot.finish_ns;
    out.breakdown.command_ns += config_.logic_scheduler_issue_ns;
    add_trace_span(
        trace_spans,
        "logic_scheduler_issue",
        "logic",
        logic_entity(ingress_stack),
        logic_start,
        issued_ns);
    out.start_ns = logic_start;
    const auto wait_for_page_stack = [&](std::size_t stack, double earliest_ns) {
        const double ready_ns = std::max(
            earliest_ns, state_ready_by_stack.at(stack));
        if (ready_ns > earliest_ns) {
            out.breakdown.scheduler_queue_wait_ns += ready_ns - earliest_ns;
            trace_wait(
                trace_spans,
                logic_entity(static_cast<std::uint32_t>(stack)),
                earliest_ns,
                ready_ns,
                "wait_page_stack_state_dependency");
        }
        return ready_ns;
    };

    if (request.op == Op::Read) {
        const auto first_lpn = physical_request ? 0 : request.addr / config_.page_size_bytes;
        const auto logical = physical_request ? decode(request.addr) : logical_page_address(request.addr);
        const auto pages = physical_request ? page_count_for(decode(request.addr), request.bytes) :
            page_count_for(logical, request.bytes);

        double finish_ns = issued_ns;
        std::uint64_t media_pages = 0;
        std::uint64_t read_buffer_pages = 0;
        std::uint64_t write_buffer_pages = 0;
        std::uint64_t erased_pages = 0;
        std::string first_path;
        std::uint64_t remaining_bytes = request.bytes;
        if (pages > 1) {
            const double split_ready_ns = issued_ns;
            const auto split = reserve(
                split_ready_ns,
                config_.address_generation_ns,
                ingress_logic.ingress);
            out.breakdown.ingress_queue_wait_ns += split.wait_ns;
            trace_wait(
                trace_spans,
                logic_entity(ingress_stack),
                split_ready_ns,
                split.start_ns,
                "wait_read_split_ingress");
            add_trace_span(
                trace_spans,
                "read_split",
                "logic",
                logic_entity(ingress_stack),
                split.start_ns,
                split.finish_ns,
                true,
                std::to_string(pages) + " page transactions");
            out.breakdown.address_mapping_ns += config_.address_generation_ns;
            // Address generation is real ingress work. Downstream page
            // transactions cannot observe the split before its actual
            // reservation (including any queueing) has completed.
            issued_ns = split.finish_ns;
            stats_.read_splits++;
            stats_.read_split_pages += pages;
        }
        for (std::uint64_t i = 0; i < pages; ++i) {
            double page_ready_ns = issued_ns;
            const auto range_begin = i == 0 ? logical.offset : 0;
            const auto range_bytes = std::min(
                remaining_bytes, config_.page_size_bytes - range_begin);
            const auto range_end = range_begin + range_bytes;
            remaining_bytes -= range_bytes;
            const auto lpn = first_lpn + i;
            const auto page_stack = physical_request ?
                stack_of_block(static_cast<std::size_t>(
                    (encode_ppn(decode(request.addr)) + i) /
                    config_.pages_per_block)) :
                stack_for_lpn(lpn);
            page_ready_ns = wait_for_page_stack(page_stack, page_ready_ns);
            page_ready_ns = admit_foreground_page_read(
                page_stack,
                page_ready_ns,
                out.breakdown,
                trace_spans);
            const auto complete_page = [&](double page_finish_ns) {
                complete_foreground_page_read(page_stack, page_finish_ns);
                finish_ns = std::max(finish_ns, page_finish_ns);
            };
            std::optional<std::uint64_t> ppn;
            if (physical_request) {
                ppn = encode_ppn(decode(request.addr)) + i;
            } else {
                auto& page_write_buffer = write_buffer(stack_for_lpn(lpn));
                const auto buffered = page_write_buffer.find(lpn);
                const std::vector<DirtyRange>* buffered_ranges = nullptr;
                double buffered_ready_ns = page_ready_ns;
                if (buffered != page_write_buffer.end()) {
                    buffered_ranges = &buffered->second.ranges;
                    buffered_ready_ns = buffered->second.ready_ns;
                } else if (const auto inflight = inflight_buffered_writes_.find(lpn);
                           inflight != inflight_buffered_writes_.end()) {
                    const InflightBufferedWrite* newest = nullptr;
                    for (const auto& generation : inflight->second) {
                        const auto target_block = static_cast<std::size_t>(
                            generation.target_ppn / config_.pages_per_block);
                        const auto& block = blocks_.at(target_block);
                        if (block.epoch == generation.target_block_epoch &&
                            !block.erase_pending &&
                            page_ready_ns < generation.commit_ns &&
                            (newest == nullptr || generation.generation > newest->generation)) {
                            newest = &generation;
                        }
                    }
                    if (newest != nullptr) {
                        buffered_ranges = &newest->ranges;
                        buffered_ready_ns = newest->ready_ns;
                    }
                }
                std::uint64_t buffered_overlap_bytes = 0;
                if (buffered_ranges != nullptr) {
                    for (const auto& range : *buffered_ranges) {
                        const auto lo = std::max(range.begin, range_begin);
                        const auto hi = std::min(range.end, range_end);
                        if (hi > lo) {
                            buffered_overlap_bytes += hi - lo;
                        }
                    }
                    // A dirty range elsewhere in the page is irrelevant to
                    // this sub-page read: it must not create a false WB hit,
                    // wait, overlay, or report classification.
                    if (buffered_overlap_bytes == 0) {
                        buffered_ranges = nullptr;
                    } else if (buffered_ready_ns > page_ready_ns) {
                        trace_wait(
                            trace_spans,
                            logic_entity(static_cast<std::uint32_t>(stack_for_lpn(lpn))),
                            page_ready_ns,
                            buffered_ready_ns,
                            "wait_prior_write_buffer_stage");
                        out.breakdown.scheduler_queue_wait_ns +=
                            buffered_ready_ns - page_ready_ns;
                        page_ready_ns = buffered_ready_ns;
                    }
                }
                if (buffered_ranges != nullptr &&
                    dirty_ranges_cover(*buffered_ranges, range_begin, range_end)) {
                    complete_page(serve_read_from_write_buffer(
                        lpn,
                        range_begin,
                        range_end,
                        *buffered_ranges,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans));
                    if (first_path.empty()) {
                        first_path = "write_buffer/lpn" + std::to_string(lpn);
                    }
                    write_buffer_pages++;
                    continue;
                }
                ppn = lookup_lpn(
                    lpn,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans);
                if (buffered_ranges != nullptr && ppn) {
                    if (first_path.empty()) {
                        first_path = decode_ppn(*ppn).path();
                    }
                    if (config_.read_buffer_pages != 0 &&
                        read_buffer_contains(*ppn, page_ready_ns)) {
                        page_ready_ns = serve_read_from_read_buffer(
                            *ppn,
                            range_bytes,
                            page_ready_ns,
                            out.breakdown,
                            trace_spans,
                            false);
                        read_buffer_pages++;
                    } else {
                        stats_.read_buffer_misses++;
                        page_ready_ns = schedule_read_page(
                            *ppn,
                            page_ready_ns,
                            out.breakdown,
                            trace_spans,
                            TransactionSource::User,
                            request.heatmap_source,
                            ReadPayloadRoute::Internal,
                            0);
                        read_buffer_insert(*ppn, page_ready_ns);
                        media_pages++;
                    }
                    page_ready_ns = schedule_sram_transfer(
                        stack_for_lpn(lpn),
                        buffered_overlap_bytes,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans,
                        "write_buffer_partial_overlay",
                        "dirty bytes over decoded backing page");
                    complete_page(schedule_external_read_egress(
                        stack_for_lpn(lpn),
                        range_bytes,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans,
                        "write_buffer_partial_hbio_out",
                        "backing page + dirty overlay"));
                    stats_.write_buffer_read_hits++;
                    stats_.write_buffer_read_bytes += buffered_overlap_bytes;
                    write_buffer_pages++;
                    continue;
                } else if (buffered_ranges != nullptr) {
                    complete_page(serve_read_from_write_buffer(
                        lpn,
                        range_begin,
                        range_end,
                        *buffered_ranges,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans));
                    if (first_path.empty()) {
                        first_path = "write_buffer+erased/lpn" +
                            std::to_string(lpn);
                    }
                    write_buffer_pages++;
                    continue;
                }
            }
            if (!ppn) {
                // NAND's erased value is the logical image of an unmapped
                // page. Return the exact requested decoded bytes through SRAM
                // and HBIO without inventing a flash/ECC operation.
                const auto stack = stack_for_lpn(lpn);
                page_ready_ns = schedule_sram_transfer(
                    stack,
                    range_bytes,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    "unmapped_read_erased_fill",
                    "decoded erased-value payload");
                complete_page(schedule_external_read_egress(
                    stack,
                    range_bytes,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    "unmapped_read_hbio_out",
                    "decoded erased-value payload"));
                if (first_path.empty()) {
                    first_path = "erased/lpn" + std::to_string(lpn);
                }
                erased_pages++;
                continue;
            }
            if (first_path.empty()) {
                first_path = decode_ppn(*ppn).path();
            }
            if (config_.read_buffer_pages != 0 && read_buffer_contains(*ppn, page_ready_ns)) {
                complete_page(serve_read_from_read_buffer(
                    *ppn,
                    range_bytes,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    true));
                read_buffer_pages++;
                continue;
            }
            stats_.read_buffer_misses++;
            double decoded_ready_ns = 0.0;
            const double page_finish = schedule_read_page(
                *ppn,
                page_ready_ns,
                out.breakdown,
                trace_spans,
                TransactionSource::User,
                request.heatmap_source,
                ReadPayloadRoute::External,
                range_bytes,
                &decoded_ready_ns);
            complete_page(page_finish);
            // The decoded page becomes cacheable when SRAM fill completes,
            // independently of how long its user's HBIO egress waits.
            read_buffer_insert(*ppn, decoded_ready_ns);
            media_pages++;
        }
        out.finish_ns = finish_ns;
        if (write_buffer_pages > 0 && (media_pages > 0 || read_buffer_pages > 0)) {
            out.note = "mapped-read-with-write-buffer-overlay";
        } else if (write_buffer_pages > 0) {
            out.note = "write-buffer-read";
        } else if (erased_pages > 0 && media_pages == 0 && read_buffer_pages == 0) {
            out.note = pages == 1 ? "unmapped-erased-read" : "unmapped-erased-multi-page-read";
        } else if (erased_pages > 0) {
            out.note = "mixed-mapped-and-erased-read";
        } else if (read_buffer_pages > 0 && media_pages == 0) {
            out.note = pages == 1 ? "read-buffer-hit" : "read-buffer-multi-page-hit";
        } else {
            out.note = media_pages == 1 ? "mapped-page-read" : "mapped-multi-page-read";
        }
        if (physical_request) {
            out.resource_path = decode(request.addr).path();
        } else if (!first_path.empty()) {
            out.resource_path = "lpn" + std::to_string(first_lpn) + "->" + first_path;
        } else {
            out.resource_path = "lpn" + std::to_string(first_lpn) + "->unmapped";
        }

        stats_.read_requests++;
        if (!physical_request) {
            stats_.logical_read_bytes += request.bytes;
        }
        stats_.physical_read_bytes += media_pages * config_.page_size_bytes;
        stats_.page_reads += media_pages;
    } else if (request.op == Op::Write) {
        const auto first_lpn = physical_request ? 0 : request.addr / config_.page_size_bytes;
        const auto logical = physical_request ? decode(request.addr) : logical_page_address(request.addr);
        const auto pages = physical_request ? page_count_for(decode(request.addr), request.bytes) :
            page_count_for(logical, request.bytes);

        double finish_ns = issued_ns;
        std::uint64_t first_ppn = 0;
        std::string first_write_path;
        std::uint64_t remaining_bytes = request.bytes;
        for (std::uint64_t i = 0; i < pages; ++i) {
            double page_ready_ns = issued_ns;
            const auto lpn = first_lpn + i;
            const auto range_begin = i == 0 ? logical.offset : 0;
            const auto range_bytes = std::min(
                remaining_bytes, config_.page_size_bytes - range_begin);
            const auto range_end = range_begin + range_bytes;
            remaining_bytes -= range_bytes;
            const auto target_stack = physical_request ?
                stack_of_block(static_cast<std::size_t>(
                    (encode_ppn(decode(request.addr)) + i) /
                    config_.pages_per_block)) :
                stack_for_lpn(lpn);
            page_ready_ns = wait_for_page_stack(target_stack, page_ready_ns);
            const auto old_ppn = physical_request ?
                std::optional<std::uint64_t>{} :
                lookup_lpn(
                    lpn,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans);
            const bool full_page_overwrite =
                range_begin == 0 && range_end == config_.page_size_bytes;
            // Physical writes name their target page directly; GC pressure
            // belongs to that page's stack, not to stack_for_lpn of a raw
            // physical address reinterpreted as an LPN.
            if (!physical_request && config_.write_coalescing_enabled) {
                double page_done = stage_write_buffer_range(
                    lpn,
                    range_begin,
                    range_end,
                    request.heatmap_source,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans);
                if (config_.write_buffer_completion_requires_flush) {
                    flush_write_buffer_entry(lpn, page_done, out.breakdown, trace_spans);
                } else {
                    // Slot admission happens inside stage_write_buffer_range,
                    // before the new range becomes visible. Threshold flushes
                    // run on the device timeline in the background.
                    if (config_.write_buffer_flush_threshold_pages != 0) {
                        double background_ns = page_done;
                        const auto wb_stack = stack_for_lpn(lpn);
                        auto& die_lru = write_buffer_lru(wb_stack);
                        while (write_buffer(wb_stack).size() >=
                               config_.write_buffer_flush_threshold_pages) {
                            const auto victim_lpn = die_lru.back();
                            flush_write_buffer_entry(
                                victim_lpn,
                                background_ns,
                                out.breakdown,
                                trace_spans);
                        }
                        background_finish_ns_ = std::max(background_finish_ns_, background_ns);
                    }
                }
                if (first_write_path.empty()) {
                    first_write_path = "lpn" + std::to_string(first_lpn) + "->write_buffer";
                }
                finish_ns = std::max(finish_ns, page_done);
                continue;
            }
            if (!physical_request && old_ppn) {
                if (!full_page_overwrite) {
                    add_trace_span(
                        trace_spans,
                        "partial_page_merge",
                        "translation",
                        logic_entity(ingress_stack),
                        page_ready_ns,
                        page_ready_ns + config_.mapping_update_ns,
                        true,
                        "lpn" + std::to_string(lpn));
                    out.breakdown.translation_ns += config_.mapping_update_ns;
                    page_ready_ns += config_.mapping_update_ns;
                    page_ready_ns = schedule_read_page(
                        *old_ppn,
                        page_ready_ns,
                        out.breakdown,
                        trace_spans,
                        TransactionSource::User,
                        request.heatmap_source,
                        ReadPayloadRoute::Internal,
                        0);
                    stats_.physical_read_bytes += config_.page_size_bytes;
                    stats_.page_reads++;
                }
            } else if (!full_page_overwrite) {
                // A new partial page has no old media image. NAND's erased
                // value supplies the untouched bytes explicitly; model the
                // logic-die SRAM fill instead of inventing a hidden read.
                page_ready_ns = schedule_sram_transfer(
                    target_stack,
                    config_.page_size_bytes - range_bytes,
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    "partial_page_erased_fill",
                    "erased-value fill before user overlay");
            }
            page_ready_ns = schedule_external_write_ingress(
                target_stack,
                range_bytes,
                page_ready_ns,
                out.breakdown,
                trace_spans,
                "request " + request.id + " page " + std::to_string(i),
                "user/write_ingress_sram");
            if (!physical_request) {
                maybe_run_gc(
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    1,
                    target_stack,
                    BlockRole::Data);
            }

            const auto new_ppn = physical_request ?
                encode_ppn(decode(request.addr)) + i :
                allocate_free_page(
                    page_ready_ns,
                    out.breakdown,
                    trace_spans,
                    BlockRole::Data,
                    stack_for_lpn(lpn));
            if (i == 0) {
                first_ppn = new_ppn;
            }
            const double program_done = schedule_program_page(
                new_ppn,
                page_ready_ns,
                out.breakdown,
                trace_spans,
                TransactionSource::User,
                request.heatmap_source);
            double mapping_ready = program_done;
            double page_done = program_done;
            if (!physical_request) {
                const auto program_commit_sequence = schedule_media_program_commit(
                    new_ppn,
                    lpn,
                    PageOwner::Logical,
                    program_done);
                access_resident_mapping(
                    lpn,
                    TransactionSource::User,
                    MappingAccessKind::Update,
                    mapping_ready,
                    out.breakdown,
                    trace_spans);
                page_done = mapping_ready;
                schedule_lpn_mapping_commit(
                    lpn, new_ppn, page_done, program_commit_sequence);
                mark_mapping_page_dirty(
                    mapping_vpn_for_lpn(lpn), page_done);
                // The resident entry is authoritative immediately; its dirty
                // checkpoint page is persisted by the end-of-run drain.
            } else {
                schedule_physical_program_commit(new_ppn, page_done);
            }
            finish_ns = std::max(finish_ns, page_done);
        }
        out.finish_ns = finish_ns;
        out.resource_path = physical_request ? decode(request.addr).path() :
            (!first_write_path.empty() ? first_write_path :
                ("lpn" + std::to_string(first_lpn) + "->" + decode_ppn(first_ppn).path()));
        if (!physical_request && config_.write_coalescing_enabled) {
            out.note = pages == 1 ? "write-buffer-stage" : "multi-page-write-buffer-stage";
        } else {
            out.note = pages == 1 ? "page-program-map-update" : "multi-page-program-map-update";
        }

        stats_.program_requests++;
        if (!physical_request) {
            stats_.logical_write_bytes += request.bytes;
        }
        if (physical_request || !config_.write_coalescing_enabled) {
            const auto payload_bytes = pages * config_.page_size_bytes;
            stats_.physical_write_bytes += payload_bytes;
            stats_.data_program_payload_bytes += payload_bytes;
            stats_.data_programs += pages;
            stats_.page_programs += pages;
        }
    } else if (request.op == Op::Erase) {
        const auto addr = decode(request.addr);
        const auto block = block_index(addr);
        out.finish_ns = schedule_erase_block(
            block,
            issued_ns,
            out.breakdown,
            trace_spans,
            TransactionSource::User,
            request.heatmap_source);
        out.resource_path = addr.path();
        out.note = "physical-block-erase";
        out.physical_bytes = 0;
        const auto [pending_erase, inserted] = pending_physical_erases_.emplace(
            block,
            PendingPhysicalErase{.finish_ns = out.finish_ns});
        if (!inserted) {
            throw std::runtime_error("HBF block already has an in-flight physical erase");
        }
        auto& block_state = blocks_.at(block);
        if (block_state.epoch == std::numeric_limits<std::uint64_t>::max()) {
            pending_physical_erases_.erase(block);
            throw std::runtime_error("HBF block epoch overflow");
        }
        // A raw erase may legally target an already-erased block. Such a
        // block is still present in the plane's free pool, so retire it from
        // allocation before publishing erase_pending. Otherwise a same-time
        // logical write can claim the block while its erase completion is
        // already scheduled; that completion then destroys the new page.
        auto& erased_plane = planes_.at(block_plane_index(block));
        if (block_state.role == BlockRole::Free) {
            const auto free_block = std::find(
                erased_plane.free_blocks.begin(),
                erased_plane.free_blocks.end(),
                block);
            if (free_block == erased_plane.free_blocks.end()) {
                pending_physical_erases_.erase(block);
                throw std::runtime_error(
                    "HBF free block targeted by erase is missing from its plane pool");
            }
            erased_plane.free_blocks.erase(free_block);
        }
        const auto retired_block_epoch = block_state.epoch;
        tag_pending_mapping_updates_for_erase(
            block, retired_block_epoch, out.finish_ns);
        // Capacity enforcement no longer scans every cache line looking for
        // stale epochs.  Retire this block's decoded lines at the same point
        // that the block becomes erase-pending; this is the visibility point
        // used by the former scan as well.
        read_buffer_purge_block(block);
        block_state.epoch++;
        block_state.erase_pending = true;
        const auto erase_epoch = block_state.epoch;
        if (erased_plane.active_data_block == block) {
            erased_plane.active_data_block = std::nullopt;
        }
        if (erased_plane.active_mapping_block == block) {
            erased_plane.active_mapping_block = std::nullopt;
        }
        if (erased_plane.active_gc_block == block) {
            erased_plane.active_gc_block = std::nullopt;
        }

        pending_erase->second.commit_sequence = schedule_commit(
            addr.stack, out.finish_ns, [this, block, erase_epoch]() {
            const auto pending_erase = pending_physical_erases_.find(block);
            if (pending_erase == pending_physical_erases_.end()) {
                throw std::runtime_error("HBF physical erase lost its in-flight reservation");
            }
            if (blocks_.at(block).epoch != erase_epoch ||
                !blocks_.at(block).erase_pending) {
                throw std::runtime_error(
                    "HBF physical erase lost its block-epoch ownership");
            }
            const auto block_begin = static_cast<std::uint64_t>(
                block) * config_.pages_per_block;
            for (std::uint32_t page = 0; page < config_.pages_per_block; ++page) {
                const auto ppn = block_begin + page;
                const auto it = programmed_pages_.find(ppn);
                if (it == programmed_pages_.end()) {
                    continue;
                }
                if (it->second.status == PageStatus::Valid) {
                    if (it->second.owner == PageOwner::Mapping) {
                        const auto mapping_vpn = metadata_vpn(it->second.lpn);
                        materialized_ready_by_vpn_[mapping_vpn] = std::max(
                            materialized_ready_by_vpn_[mapping_vpn],
                            pending_erase->second.finish_ns);
                        const auto found =
                            mapping_vpn_to_ppn_.find(mapping_vpn);
                        if (found != mapping_vpn_to_ppn_.end() && found->second == ppn) {
                            mapping_vpn_to_ppn_.erase(found);
                        }
                    } else if (it->second.owner == PageOwner::Logical) {
                        materialized_ready_by_lpn_[it->second.lpn] = std::max(
                            materialized_ready_by_lpn_[it->second.lpn],
                            pending_erase->second.finish_ns);
                        const auto found = lpn_to_ppn_.find(it->second.lpn);
                        if (found != lpn_to_ppn_.end() && found->second == ppn) {
                            lpn_to_ppn_.erase(found);
                        }
                    }
                }
            }
            reset_erased_block(block);
            retire_mapping_update_tombstones(
                block, pending_erase->second.finish_ns);
            materialized_ready_by_block_.at(block) = std::max(
                materialized_ready_by_block_.at(block),
                pending_erase->second.finish_ns);
            pending_physical_erases_.erase(pending_erase);
        });

        stats_.erase_requests++;
        stats_.block_erases++;
    }

    if (address_heatmap_ != nullptr && !physical_request &&
        (request.op == Op::Read || request.op == Op::Write)) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfLogical,
            .direction = request.op == Op::Read ?
                TrafficDirection::Read : TrafficDirection::Write,
            .source = request.heatmap_source,
            .address = request.addr,
            .bytes = request.bytes,
        });
    }
    stats_.free_pages = free_pages_;
    stats_.stage_work += out.breakdown;
    stats_.finish_ns = std::max({stats_.finish_ns, out.finish_ns, background_finish_ns_});
    if (request.op == Op::Read || request.op == Op::Write) {
        out.physical_bytes = stats_.physical_read_bytes + stats_.physical_write_bytes -
            physical_bytes_before;
    }
    return out;
}

std::size_t HbfDevice::stack_index(const HbfAddress& addr) const {
    return addr.stack;
}

std::size_t HbfDevice::channel_index(const HbfAddress& addr) const {
    return static_cast<std::size_t>(addr.stack) * config_.channels_per_stack + addr.channel;
}

std::size_t HbfDevice::die_index(const HbfAddress& addr) const {
    std::size_t index = addr.stack;
    index = index * config_.channels_per_stack + addr.channel;
    index = index * config_.dies_per_channel + addr.die;
    return index;
}

std::list<std::uint64_t>& HbfDevice::write_buffer_lru(std::size_t stack) {
    return write_buffer_lru_by_stack_.at(stack);
}

std::unordered_map<std::uint64_t, HbfDevice::WriteBufferEntry>& HbfDevice::write_buffer(
    std::size_t stack) {
    return write_buffer_by_stack_.at(stack);
}

HbfDevice::SenseRoundCandidate HbfDevice::preview_sense_round(
    PlaneState& plane,
    std::size_t subarray_index,
    double ready_ns) {
    const double t_read = config_.t_read_page_ns;
    // issue() accepts only nondecreasing arrivals, and every transaction it
    // creates is ready no earlier than its parent arrival. Therefore a round
    // before the current top-level arrival is causally unreachable by this or
    // any future issue. Reclaim only behind that proven watermark; using the
    // current command's ready time would be unsafe because a later-called
    // sibling can legitimately have an earlier ready time.
    const double causal_watermark_ns = last_issue_arrival_ns_.value_or(0.0);
    if (causal_watermark_ns > plane.sense_rounds_pruned_through_ns) {
        for (auto& available : plane.available_sense_rounds_by_subarray) {
            available.erase(available.begin(), available.lower_bound(causal_watermark_ns));
        }
        plane.sense_rounds_pruned_through_ns = causal_watermark_ns;
    }

    // Availability is indexed independently for each subarray because a
    // round has exactly one slot for each subarray. This is equivalent to a
    // used-subarray mask, but has no fixed history/width limit and provides
    // logarithmic earliest-round lookup.
    auto& subarray = plane.subarrays.at(subarray_index);
    subarray.timeline.prune_before(reservation_causal_watermark_ns_);
    plane.sense_round_calendar.prune_before(reservation_causal_watermark_ns_);
    auto& available = plane.available_sense_rounds_by_subarray.at(subarray_index);
    std::optional<double> join_start;
    for (auto round = available.lower_bound(ready_ns); round != available.end();) {
        const double start_ns = *round;
        if (subarray.timeline.can_reserve_exact(start_ns, t_read)) {
            join_start = start_ns;
            break;
        }
        // A full-plane barrier or an earlier same-subarray read consumed this
        // advertised slot. It can never become available again.
        round = available.erase(round);
    }

    double candidate = ready_ns;
    double new_round_start = 0.0;
    for (;;) {
        const double round_start = plane.sense_round_calendar.preview_start(
            candidate, t_read);
        const double subarray_start = subarray.timeline.preview_start(
            candidate, t_read);
        const double start = std::max(round_start, subarray_start);
        if (plane.sense_round_calendar.can_reserve_exact(start, t_read) &&
            subarray.timeline.can_reserve_exact(start, t_read)) {
            new_round_start = start;
            break;
        }
        if (start <= candidate) {
            throw std::runtime_error(
                "HBF batch-round preview failed to make forward progress");
        }
        candidate = start;
    }

    if (join_start && *join_start <= new_round_start) {
        return SenseRoundCandidate{
            .start_ns = *join_start,
            .joins_existing_round = true,
        };
    }
    return SenseRoundCandidate{
        .start_ns = new_round_start,
        .joins_existing_round = false,
    };
}

HbfDevice::ScheduledTransfer HbfDevice::commit_sense_round(
    PlaneState& plane,
    std::size_t subarray_index,
    double ready_ns,
    const SenseRoundCandidate& candidate) {
    const double t_read = config_.t_read_page_ns;
    auto& subarray = plane.subarrays.at(subarray_index);
    if (candidate.joins_existing_round) {
        auto& available = plane.available_sense_rounds_by_subarray.at(
            subarray_index);
        const auto found = available.find(candidate.start_ns);
        if (found == available.end()) {
            throw std::runtime_error("HBF batch round lost its subarray slot");
        }
        available.erase(found);
    } else {
        const auto round = reserve(
            candidate.start_ns, t_read, plane.sense_round_calendar);
        if (round.start_ns != candidate.start_ns) {
            throw std::runtime_error("HBF new batch round moved after preview");
        }
        for (std::size_t index = 0;
             index < plane.available_sense_rounds_by_subarray.size();
             ++index) {
            if (index != subarray_index) {
                plane.available_sense_rounds_by_subarray[index].insert(
                    candidate.start_ns);
            }
        }
    }
    const auto scheduled = reserve(
        candidate.start_ns, t_read, subarray.timeline);
    if (scheduled.start_ns != candidate.start_ns) {
        throw std::runtime_error("HBF subarray sense moved after preview");
    }
    return ScheduledTransfer{
        .start_ns = scheduled.start_ns,
        .finish_ns = scheduled.finish_ns,
        .wait_ns = std::max(0.0, scheduled.start_ns - ready_ns),
    };
}

double HbfDevice::preview_full_plane_window(
    PlaneState& plane,
    double earliest_ns,
    double duration_ns) {
    plane.sense_round_calendar.prune_before(reservation_causal_watermark_ns_);
    for (auto& subarray : plane.subarrays) {
        subarray.timeline.prune_before(reservation_causal_watermark_ns_);
    }
    for (auto& lane : plane.media_lanes) {
        lane.timeline.prune_before(reservation_causal_watermark_ns_);
    }
    for (auto& bank : plane.page_buffer_banks) {
        bank.timeline.prune_before(reservation_causal_watermark_ns_);
    }

    double candidate = earliest_ns;
    for (;;) {
        double next = plane.sense_round_calendar.preview_start(
            candidate, duration_ns);
        for (const auto& subarray : plane.subarrays) {
            next = std::max(
                next, subarray.timeline.preview_start(candidate, duration_ns));
        }
        for (const auto& lane : plane.media_lanes) {
            next = std::max(
                next, lane.timeline.preview_start(candidate, duration_ns));
        }
        for (const auto& bank : plane.page_buffer_banks) {
            next = std::max(
                next, bank.timeline.preview_start(candidate, duration_ns));
        }

        bool exact = plane.sense_round_calendar.can_reserve_exact(
            next, duration_ns);
        for (const auto& subarray : plane.subarrays) {
            exact = exact && subarray.timeline.can_reserve_exact(next, duration_ns);
        }
        for (const auto& lane : plane.media_lanes) {
            exact = exact && lane.timeline.can_reserve_exact(next, duration_ns);
        }
        for (const auto& bank : plane.page_buffer_banks) {
            exact = exact && bank.timeline.can_reserve_exact(next, duration_ns);
        }
        if (exact) {
            return next;
        }
        if (next <= candidate) {
            throw std::runtime_error(
                "HBF full-plane preview failed to make forward progress");
        }
        candidate = next;
    }
}

void HbfDevice::record_full_plane_window(
    PlaneState& plane,
    double begin_ns,
    double end_ns) {
    const auto position = std::lower_bound(
        plane.full_plane_windows.begin(),
        plane.full_plane_windows.end(),
        begin_ns,
        [](const PlaneState::BusyWindow& window, double begin) {
            return window.begin_ns < begin;
        });
    if ((position != plane.full_plane_windows.begin() &&
         std::prev(position)->end_ns > begin_ns) ||
        (position != plane.full_plane_windows.end() &&
         position->begin_ns < end_ns)) {
        throw std::runtime_error("HBF full-plane windows overlap");
    }
    plane.full_plane_windows.insert(position, PlaneState::BusyWindow{
        .begin_ns = begin_ns,
        .end_ns = end_ns,
    });
}

void HbfDevice::record_plane_media_busy(
    PlaneState& plane,
    double begin_ns,
    double end_ns) {
    if (end_ns <= begin_ns) {
        return;
    }
    auto& windows = plane.media_busy_windows;
    auto first = std::lower_bound(
        windows.begin(),
        windows.end(),
        begin_ns,
        [](const PlaneState::BusyWindow& window, double begin) {
            return window.end_ns < begin;
        });
    double merged_begin = begin_ns;
    double merged_end = end_ns;
    double replaced_ns = 0.0;
    auto last = first;
    while (last != windows.end() && last->begin_ns <= merged_end) {
        merged_begin = std::min(merged_begin, last->begin_ns);
        merged_end = std::max(merged_end, last->end_ns);
        replaced_ns += last->end_ns - last->begin_ns;
        ++last;
    }
    first = windows.erase(first, last);
    windows.insert(first, PlaneState::BusyWindow{
        .begin_ns = merged_begin,
        .end_ns = merged_end,
    });
    plane.media_busy_ns += merged_end - merged_begin - replaced_ns;
}

bool HbfDevice::read_buffer_contains(std::uint64_t ppn, double at_ns) {
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    const auto stack = stack_of_block(block_index);
    auto& logic_die = logic_dies_.at(stack);
    const double capacity_observation_ns =
        std::min(at_ns, state_observation_by_stack_.at(stack));
    enforce_read_buffer_capacity(logic_die, capacity_observation_ns);
    const auto found = logic_die.read_buffer.find(ppn);
    if (found == logic_die.read_buffer.end() ||
        found->second.block_epoch != blocks_.at(block_index).epoch ||
        blocks_.at(block_index).erase_pending) {
        if (found != logic_die.read_buffer.end()) {
            erase_read_buffer_entry(logic_die, ppn);
        }
        return false;
    }
    // The former capacity pass compacted every entry's history.  Compacting
    // the queried entry here retains the same temporal boundary while
    // avoiding O(cache_size) representation work on every lookup.
    prune_touch_history(found->second.touches, capacity_observation_ns);
    const double observed_ns = state_observation_by_stack_.at(stack);
    if (at_ns > observed_ns) {
        // A request may attach to a fill that is still in flight at its
        // causal admission time. It may not retroactively claim a line whose
        // fill completed earlier while other fills could have evicted it.
        if (found->second.ready_ns > observed_ns &&
            at_ns > found->second.ready_ns) {
            return false;
        }
        if (found->second.ready_ns <= observed_ns) {
            const bool intervening_fill = std::any_of(
                logic_die.read_buffer.begin(),
                logic_die.read_buffer.end(),
                [ppn, observed_ns, at_ns](const auto& entry) {
                    return entry.first != ppn &&
                        entry.second.ready_ns > observed_ns &&
                        entry.second.ready_ns <= at_ns;
                });
            if (intervening_fill) {
                return false;
            }
        }
    }
    return true;
}

void HbfDevice::rebalance_read_buffer_ready_index(LogicDieState& logic_die) {
    const auto total = logic_die.read_buffer_ready_prefix.size() +
        logic_die.read_buffer_ready_suffix.size();
    const auto capacity = config_.read_buffer_pages;
    const auto prefix_limit =
        capacity >= std::numeric_limits<std::size_t>::max() ?
        std::numeric_limits<std::size_t>::max() :
        static_cast<std::size_t>(capacity) + 1;
    const auto target = std::min(total, prefix_limit);

    while (logic_die.read_buffer_ready_prefix.size() > target) {
        const auto last = std::prev(logic_die.read_buffer_ready_prefix.end());
        logic_die.read_buffer_ready_suffix.insert(*last);
        logic_die.read_buffer_ready_prefix.erase(last);
    }
    while (logic_die.read_buffer_ready_prefix.size() < target &&
           !logic_die.read_buffer_ready_suffix.empty()) {
        const auto first = logic_die.read_buffer_ready_suffix.begin();
        logic_die.read_buffer_ready_prefix.insert(*first);
        logic_die.read_buffer_ready_suffix.erase(first);
    }
    while (!logic_die.read_buffer_ready_prefix.empty() &&
           !logic_die.read_buffer_ready_suffix.empty()) {
        const auto prefix_last =
            std::prev(logic_die.read_buffer_ready_prefix.end());
        const auto suffix_first = logic_die.read_buffer_ready_suffix.begin();
        if (*prefix_last <= *suffix_first) {
            break;
        }
        const auto low = *suffix_first;
        const auto high = *prefix_last;
        logic_die.read_buffer_ready_prefix.erase(prefix_last);
        logic_die.read_buffer_ready_suffix.erase(suffix_first);
        logic_die.read_buffer_ready_prefix.insert(low);
        logic_die.read_buffer_ready_suffix.insert(high);
    }
}

void HbfDevice::index_read_buffer_entry(
    LogicDieState& logic_die,
    std::uint64_t ppn) {
    const auto entry = logic_die.read_buffer.find(ppn);
    if (entry == logic_die.read_buffer.end() || entry->second.touches.empty()) {
        throw std::runtime_error(
            "HBF cannot index a missing or untouched read-buffer entry");
    }
    const auto ready_key = LogicDieState::ReadBufferReadyKey{
        entry->second.ready_ns, ppn};
    if (!logic_die.read_buffer_ready_prefix.insert(ready_key).second) {
        throw std::runtime_error("HBF duplicate read-buffer ready index entry");
    }
    const auto lru_key = LogicDieState::ReadBufferLruKey{
        *entry->second.touches.rbegin(), ppn};
    if (!logic_die.read_buffer_lru.insert(lru_key).second) {
        logic_die.read_buffer_ready_prefix.erase(ready_key);
        throw std::runtime_error("HBF duplicate read-buffer LRU index entry");
    }
    rebalance_read_buffer_ready_index(logic_die);
}

void HbfDevice::erase_read_buffer_entry(
    LogicDieState& logic_die,
    std::uint64_t ppn) {
    const auto entry = logic_die.read_buffer.find(ppn);
    if (entry == logic_die.read_buffer.end()) {
        return;
    }
    const auto ready_key = LogicDieState::ReadBufferReadyKey{
        entry->second.ready_ns, ppn};
    if (logic_die.read_buffer_ready_prefix.erase(ready_key) == 0 &&
        logic_die.read_buffer_ready_suffix.erase(ready_key) == 0) {
        throw std::runtime_error("HBF read-buffer ready index lost an entry");
    }
    const auto lru_key = LogicDieState::ReadBufferLruKey{
        *entry->second.touches.rbegin(), ppn};
    if (logic_die.read_buffer_lru.erase(lru_key) != 1) {
        throw std::runtime_error("HBF read-buffer LRU index lost an entry");
    }
    logic_die.read_buffer.erase(entry);
    rebalance_read_buffer_ready_index(logic_die);
}

void HbfDevice::touch_read_buffer_entry(
    LogicDieState& logic_die,
    std::uint64_t ppn,
    double at_ns) {
    const auto entry = logic_die.read_buffer.find(ppn);
    if (entry == logic_die.read_buffer.end() || entry->second.touches.empty()) {
        throw std::runtime_error("HBF cannot touch a missing read-buffer entry");
    }
    const auto old_key = LogicDieState::ReadBufferLruKey{
        *entry->second.touches.rbegin(), ppn};
    if (logic_die.read_buffer_lru.erase(old_key) != 1) {
        throw std::runtime_error("HBF read-buffer touch lost its LRU index");
    }
    entry->second.touches.emplace(at_ns, next_cache_touch_sequence_++);
    const auto new_key = LogicDieState::ReadBufferLruKey{
        *entry->second.touches.rbegin(), ppn};
    if (!logic_die.read_buffer_lru.insert(new_key).second) {
        throw std::runtime_error("HBF read-buffer touch duplicated its LRU index");
    }
}

bool HbfDevice::read_buffer_capacity_exceeded_at(
    const LogicDieState& logic_die,
    double at_ns) const {
    if (config_.read_buffer_pages >=
        std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    const auto needed = static_cast<std::size_t>(config_.read_buffer_pages) + 1;
    return logic_die.read_buffer_ready_prefix.size() == needed &&
        std::prev(logic_die.read_buffer_ready_prefix.end())->first <= at_ns;
}

void HbfDevice::enforce_read_buffer_capacity(
    LogicDieState& logic_die,
    double at_ns) {
    while (read_buffer_capacity_exceeded_at(logic_die, at_ns)) {
        if (logic_die.read_buffer_lru.empty() ||
            logic_die.read_buffer_lru.begin()->first.first > at_ns) {
            // Every excess line has an already-scheduled future consumer.
            // Keep those lines pinned until that consumer occurs; evicting
            // now would invalidate a completion already returned to its
            // parent request.
            break;
        }
        erase_read_buffer_entry(
            logic_die, logic_die.read_buffer_lru.begin()->second);
    }
}

void HbfDevice::read_buffer_insert(std::uint64_t ppn, double ready_ns) {
    if (config_.read_buffer_pages == 0) {
        return;
    }
    auto& logic_die = logic_dies_.at(stack_of_block(
        static_cast<std::size_t>(ppn / config_.pages_per_block)));
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    const auto& block = blocks_.at(block_index);
    auto found = logic_die.read_buffer.find(ppn);
    if (found != logic_die.read_buffer.end() &&
        (found->second.block_epoch != block.epoch || block.erase_pending)) {
        erase_read_buffer_entry(logic_die, ppn);
        found = logic_die.read_buffer.end();
    }
    if (block.erase_pending) {
        return;
    }
    if (found != logic_die.read_buffer.end()) {
        const auto old_ready_key = LogicDieState::ReadBufferReadyKey{
            found->second.ready_ns, ppn};
        if (logic_die.read_buffer_ready_prefix.erase(old_ready_key) == 0 &&
            logic_die.read_buffer_ready_suffix.erase(old_ready_key) == 0) {
            throw std::runtime_error(
                "HBF read-buffer fill lost its ready index entry");
        }
        found->second.ready_ns = std::min(found->second.ready_ns, ready_ns);
        logic_die.read_buffer_ready_prefix.emplace(
            found->second.ready_ns, ppn);
        rebalance_read_buffer_ready_index(logic_die);
        touch_read_buffer_entry(logic_die, ppn, ready_ns);
        return;
    }
    LogicDieState::ReadBufferEntry entry{
        .ready_ns = ready_ns,
        .block_epoch = block.epoch,
    };
    entry.touches.emplace(ready_ns, next_cache_touch_sequence_++);
    logic_die.read_buffer.emplace(ppn, std::move(entry));
    index_read_buffer_entry(logic_die, ppn);
    // A future fill is an in-flight merge target, not a resident cache line.
    // Evicting at its future ready time here mutates present state in host
    // call order (A/B/A at the same arrival loses A before either fill). The
    // capacity check is applied only when a lookup/access actually reaches a
    // time at which those fills are ready.
}

void HbfDevice::read_buffer_purge_page(std::uint64_t ppn) {
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    auto& logic_die = logic_dies_.at(stack_of_block(block_index));
    erase_read_buffer_entry(logic_die, ppn);
}

void HbfDevice::read_buffer_purge_block(std::size_t block_index) {
    // PPNs are reused after erase+program; stale data must not be served.
    auto& logic_die = logic_dies_.at(stack_of_block(block_index));
    const auto begin = static_cast<std::uint64_t>(block_index) * config_.pages_per_block;
    const auto end = begin + config_.pages_per_block;
    for (auto ppn = begin; ppn < end; ++ppn) {
        const auto found = logic_die.read_buffer.find(ppn);
        if (found != logic_die.read_buffer.end()) {
            erase_read_buffer_entry(logic_die, ppn);
        }
    }
}

double HbfDevice::serve_read_from_read_buffer(
    std::uint64_t ppn,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    bool external_egress) {
    const auto stack = static_cast<std::uint32_t>(stack_of_block(
        static_cast<std::size_t>(ppn / config_.pages_per_block)));
    auto& logic_die = logic_dies_.at(stack);
    const auto buffered = logic_die.read_buffer.find(ppn);
    if (buffered == logic_die.read_buffer.end()) {
        throw std::runtime_error("HBF attempted to serve a missing read-buffer entry");
    }
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    if (buffered->second.block_epoch != blocks_.at(block_index).epoch ||
        blocks_.at(block_index).erase_pending) {
        throw std::runtime_error(
            "HBF attempted to serve a stale read-buffer epoch");
    }
    earliest_ns = std::max(earliest_ns, buffered->second.ready_ns);
    if (bytes == 0 || bytes > config_.page_size_bytes) {
        throw std::runtime_error("HBF read-buffer transfer size is invalid");
    }
    const double sram_ns = transfer_time_ns(bytes, config_.logic_sram_bandwidth_GBps);
    auto sram = reserve(earliest_ns, sram_ns, logic_die.sram);
    touch_read_buffer_entry(logic_die, ppn, sram.start_ns);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    trace_wait(spans, logic_entity(stack), earliest_ns, sram.start_ns, "wait_sram");
    add_trace_span(
        spans,
        "read_buffer_hit",
        "sram",
        logic_entity(stack),
        sram.start_ns,
        sram.finish_ns,
        true,
        "ppn" + std::to_string(ppn));
    stats_.read_buffer_hits++;
    stats_.read_buffer_read_bytes += bytes;
    if (!external_egress) {
        return sram.finish_ns;
    }
    // The hit leaves over the external interface as requested decoded bytes;
    // OOB never appears beyond ECC/SRAM.
    return schedule_external_read_egress(
        stack,
        bytes,
        sram.finish_ns,
        breakdown,
        spans,
        "read_buffer_hit_hbio_out",
        "read-buffer hit");
}

std::size_t HbfDevice::stack_for_lpn(std::uint64_t lpn) const {
    // Page-granular striping keeps a sequential window active on every HBF
    // stack. mapping_vpn_for_lpn() still groups each stack's local sequence
    // into its own mapping pages, so metadata never needs a cross-stack
    // lookup. This deliberately separates data striping from metadata
    // ownership; binding 512 consecutive global pages to one mapping page
    // serialized an entire W512 stream onto one stack.
    return stack_for_vpn(mapping_vpn_for_lpn(lpn));
}

std::size_t HbfDevice::stack_for_vpn(std::uint64_t mapping_vpn) const {
    return static_cast<std::size_t>(mapping_vpn % config_.stacks);
}

std::size_t HbfDevice::planes_per_stack() const {
    return planes_.size() / config_.stacks;
}

std::size_t HbfDevice::stack_of_plane(std::size_t plane) const {
    return plane / planes_per_stack();
}

std::size_t HbfDevice::stack_of_block(std::size_t block_index) const {
    return stack_of_plane(block_plane_index(block_index));
}

std::size_t HbfDevice::mapping_plane_for_vpn(std::uint64_t mapping_vpn) const {
    // A VPN is encoded as group * stacks + owner_stack. Divide the owner out
    // before selecting a local plane so every stack uses all of its planes.
    return stack_for_vpn(mapping_vpn) * planes_per_stack() +
        static_cast<std::size_t>(
            (mapping_vpn / config_.stacks) % planes_per_stack());
}

std::size_t HbfDevice::plane_index(const HbfAddress& addr) const {
    std::size_t index = die_index(addr);
    index = index * config_.planes_per_die + addr.plane;
    return index;
}

std::size_t HbfDevice::block_index(const HbfAddress& addr) const {
    std::size_t index = plane_index(addr);
    index = index * config_.blocks_per_plane + addr.block;
    return index;
}

std::size_t HbfDevice::block_plane_index(std::size_t block_index) const {
    return block_index / config_.blocks_per_plane;
}

std::size_t HbfDevice::subarray_index(const HbfAddress& addr) const {
    const auto page_in_plane =
        static_cast<std::uint64_t>(addr.block) * config_.pages_per_block + addr.page;
    return static_cast<std::size_t>(page_in_plane % subarrays_per_plane_);
}

std::size_t HbfDevice::media_lane_index(const HbfAddress& addr) const {
    const auto page_in_plane =
        static_cast<std::uint64_t>(addr.block) * config_.pages_per_block + addr.page;
    return static_cast<std::size_t>(page_in_plane % config_.media_lanes_per_plane);
}

std::size_t HbfDevice::page_buffer_bank_index(const HbfAddress& addr) const {
    const auto page_in_plane =
        static_cast<std::uint64_t>(addr.block) * config_.pages_per_block + addr.page;
    return static_cast<std::size_t>(page_in_plane % config_.page_buffer_banks_per_plane);
}

std::uint64_t HbfDevice::page_count_for(const HbfAddress& addr, std::uint64_t bytes) const {
    if (bytes == 0) {
        return 0;
    }
    const auto first_page_bytes = config_.page_size_bytes - addr.offset;
    if (bytes <= first_page_bytes) {
        return 1;
    }
    return 1 + div_ceil(bytes - first_page_bytes, config_.page_size_bytes);
}

HbfAddress HbfDevice::decode_ppn(std::uint64_t ppn) const {
    if (ppn >= total_pages_) {
        throw std::runtime_error("HBF PPN is out of range");
    }
    return decode(ppn * config_.page_size_bytes);
}

std::uint64_t HbfDevice::encode_ppn(const HbfAddress& addr) const {
    if (addr.stack >= config_.stacks || addr.channel >= config_.channels_per_stack ||
        addr.die >= config_.dies_per_channel || addr.plane >= config_.planes_per_die ||
        addr.block >= config_.blocks_per_plane || addr.page >= config_.pages_per_block ||
        addr.offset >= config_.page_size_bytes) {
        throw std::runtime_error("HBF address field out of range");
    }
    std::uint64_t unit = addr.stack;
    unit = unit * config_.channels_per_stack + addr.channel;
    unit = unit * config_.dies_per_channel + addr.die;
    unit = unit * config_.planes_per_die + addr.plane;
    unit = unit * config_.blocks_per_plane + addr.block;
    unit = unit * config_.pages_per_block + addr.page;
    return unit;
}

HbfAddress HbfDevice::logical_page_address(std::uint64_t logical_byte_addr) const {
    HbfAddress addr;
    addr.offset = logical_byte_addr % config_.page_size_bytes;
    return addr;
}

std::uint64_t HbfDevice::mapping_vpn_for_lpn(std::uint64_t lpn) const {
    // One global stripe contains exactly one page per stack. Rotate the lane
    // assignment once per mapping group to prevent a power-of-two logical
    // stride from pinning a tensor/KV phase to one stack, then encode the
    // owner directly in the low VPN digit:
    //
    //   vpn = local_mapping_group * stacks + owner_stack
    //
    // Each VPN therefore owns mapping_entries_per_page consecutive entries
    // from one stack's local page sequence while adjacent global LPNs remain
    // page-striped across the full fabric.
    const auto stacks = static_cast<std::uint64_t>(config_.stacks);
    const auto stripe = lpn / stacks;
    const auto lane = lpn % stacks;
    const auto group = stripe / config_.mapping_entries_per_page;
    const auto rotation = placement_mix64(group) % stacks;
    const auto stack = lane >= stacks - rotation ?
        lane - (stacks - rotation) :
        lane + rotation;
    return checked_add(
        checked_mul(group, stacks, "HBF mapping VPN group"),
        stack,
        "HBF mapping VPN");
}

std::optional<std::uint64_t> HbfDevice::compact_lpn_ppn(
    std::uint64_t lpn) const {
    if (!compact_logical_image_ ||
        lpn < compact_logical_image_->first_lpn ||
        lpn - compact_logical_image_->first_lpn >=
            compact_logical_image_->page_count) {
        return std::nullopt;
    }
    const auto& image = *compact_logical_image_;
    if (image.retired_lpns.contains(lpn)) {
        return std::nullopt;
    }
    const auto mapping_vpn = mapping_vpn_for_lpn(lpn);
    const auto stack = stack_for_vpn(mapping_vpn);
    if (mapping_vpn < image.first_vpn ||
        mapping_vpn - image.first_vpn >= image.vpn_slot_count) {
        throw std::runtime_error(
            "HBF compact LPN maps outside its VPN directory");
    }
    const auto vpn_offset = mapping_vpn - image.first_vpn;
    const auto& range = image.vpn_ranges.at(
        static_cast<std::size_t>(vpn_offset));
    const auto local_entry =
        (lpn / config_.stacks) % config_.mapping_entries_per_page;
    if (local_entry < range.first_entry ||
        local_entry - range.first_entry >= range.page_count) {
        throw std::runtime_error(
            "HBF compact LPN is outside its mapping-page range");
    }
    const auto data_index = checked_add(
        range.stack_page_offset,
        local_entry - range.first_entry,
        "HBF compact per-stack data index");
    const auto pps = static_cast<std::uint64_t>(planes_per_stack());
    const auto plane = stack * planes_per_stack() +
        static_cast<std::size_t>(data_index % pps);
    const auto page_ordinal = data_index / pps;
    const auto block_ordinal = page_ordinal / config_.pages_per_block;
    const auto page = static_cast<std::uint32_t>(
        page_ordinal % config_.pages_per_block);
    const auto& assigned_blocks = image.data_blocks_by_plane.at(plane);
    if (block_ordinal >= assigned_blocks.size()) {
        throw std::runtime_error(
            "HBF compact LPN maps beyond its data-block directory");
    }
    const auto block_index = assigned_blocks[block_ordinal];
    const auto& block = blocks_.at(static_cast<std::size_t>(block_index));
    if (block.role != BlockRole::Data || block.erase_pending ||
        !block.is_valid(page)) {
        throw std::runtime_error(
            "HBF compact LPN maps to a non-live data page");
    }
    return block_index * config_.pages_per_block + page;
}

std::optional<std::uint64_t> HbfDevice::compact_mapping_ppn(
    std::uint64_t mapping_vpn) const {
    if (!compact_logical_image_ ||
        mapping_vpn < compact_logical_image_->first_vpn ||
        mapping_vpn - compact_logical_image_->first_vpn >=
            compact_logical_image_->vpn_slot_count) {
        return std::nullopt;
    }
    if (compact_logical_image_->retired_mapping_vpns.contains(
            mapping_vpn)) {
        return std::nullopt;
    }
    const auto offset = mapping_vpn - compact_logical_image_->first_vpn;
    const auto compact_ppn = compact_logical_image_->mapping_ppns.at(
        static_cast<std::size_t>(offset));
    if (!compact_ppn) {
        return std::nullopt;
    }
    const auto ppn = *compact_ppn;
    const auto block_index = static_cast<std::size_t>(
        ppn / config_.pages_per_block);
    const auto page = static_cast<std::uint32_t>(
        ppn % config_.pages_per_block);
    const auto& block = blocks_.at(block_index);
    if (block.role != BlockRole::Mapping || block.erase_pending ||
        !block.is_valid(page)) {
        throw std::runtime_error(
            "HBF compact VPN maps to a non-live mapping page");
    }
    return ppn;
}

std::optional<std::uint64_t> HbfDevice::compact_lpn_for_ppn(
    std::uint64_t ppn) const {
    if (!compact_logical_image_ || ppn >= total_pages_) {
        return std::nullopt;
    }
    const auto& image = *compact_logical_image_;
    const auto block_index = ppn / config_.pages_per_block;
    const auto location = image.data_block_locations.find(block_index);
    if (location == image.data_block_locations.end()) {
        return std::nullopt;
    }
    const auto pps = static_cast<std::uint64_t>(planes_per_stack());
    const auto plane = location->second.plane;
    const auto stack = stack_of_plane(plane);
    const auto local_plane =
        static_cast<std::uint64_t>(plane - stack * planes_per_stack());
    const auto page_ordinal = checked_add(
        checked_mul(
            location->second.block_ordinal,
            config_.pages_per_block,
            "HBF compact inverse block ordinal"),
        ppn % config_.pages_per_block,
        "HBF compact inverse page ordinal");
    const auto data_index = checked_add(
        checked_mul(
            page_ordinal,
            pps,
            "HBF compact inverse striped data index"),
        local_plane,
        "HBF compact inverse data index");
    const auto& offsets = image.vpn_offsets_by_stack.at(stack);
    std::size_t lower = 0;
    std::size_t upper = offsets.size();
    while (lower < upper) {
        const auto middle = lower + (upper - lower) / 2;
        const auto& range = image.vpn_ranges.at(
            static_cast<std::size_t>(offsets[middle]));
        if (range.stack_page_offset <= data_index) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    if (lower == 0) {
        return std::nullopt;
    }
    const auto vpn_offset = offsets[lower - 1];
    const auto& range = image.vpn_ranges.at(
        static_cast<std::size_t>(vpn_offset));
    if (data_index - range.stack_page_offset >= range.page_count) {
        return std::nullopt;
    }
    const auto mapping_vpn = checked_add(
        image.first_vpn,
        vpn_offset,
        "HBF compact inverse mapping VPN");
    const auto stacks = static_cast<std::uint64_t>(config_.stacks);
    const auto group = mapping_vpn / stacks;
    const auto rotation = placement_mix64(group) % stacks;
    const auto lane = static_cast<std::uint64_t>(stack) >= rotation ?
        static_cast<std::uint64_t>(stack) - rotation :
        stacks - (rotation - static_cast<std::uint64_t>(stack));
    const auto local_entry = checked_add(
        range.first_entry,
        data_index - range.stack_page_offset,
        "HBF compact inverse local entry");
    const auto stripe = checked_add(
        checked_mul(
            group,
            config_.mapping_entries_per_page,
            "HBF compact inverse mapping stripe"),
        local_entry,
        "HBF compact inverse stripe");
    const auto lpn = checked_add(
        checked_mul(stripe, stacks, "HBF compact inverse LPN stripe"),
        lane,
        "HBF compact inverse LPN");
    if (lpn < image.first_lpn ||
        lpn - image.first_lpn >= image.page_count ||
        image.retired_lpns.contains(lpn)) {
        return std::nullopt;
    }
    const auto forward = compact_lpn_ppn(lpn);
    if (!forward || *forward != ppn) {
        throw std::runtime_error(
            "HBF compact forward/inverse data mapping diverged");
    }
    return lpn;
}

std::optional<HbfDevice::CompactPageIdentity>
HbfDevice::compact_page_identity(std::uint64_t ppn) const {
    if (!compact_logical_image_) {
        return std::nullopt;
    }
    const auto& image = *compact_logical_image_;
    if (const auto mapping = image.mapping_vpn_by_ppn.find(ppn);
        mapping != image.mapping_vpn_by_ppn.end() &&
        !image.retired_mapping_vpns.contains(mapping->second)) {
        const auto forward = compact_mapping_ppn(mapping->second);
        if (!forward || *forward != ppn) {
            throw std::runtime_error(
                "HBF compact forward/inverse mapping-page directory "
                "diverged");
        }
        return CompactPageIdentity{
            .logical_key = metadata_lpn(mapping->second),
            .owner = PageOwner::Mapping,
        };
    }
    if (const auto lpn = compact_lpn_for_ppn(ppn)) {
        return CompactPageIdentity{
            .logical_key = *lpn,
            .owner = PageOwner::Logical,
        };
    }
    return std::nullopt;
}

void HbfDevice::retire_compact_page(
    std::uint64_t logical_key,
    PageOwner owner) {
    if (!compact_logical_image_ ||
        !compact_logical_image_->mutable_image) {
        throw std::runtime_error(
            "HBF cannot retire a non-mutable compact logical page");
    }
    auto& image = *compact_logical_image_;
    std::optional<std::uint64_t> ppn;
    std::unordered_map<std::uint64_t, std::uint32_t>* live_pages = nullptr;
    if (owner == PageOwner::Logical) {
        ppn = compact_lpn_ppn(logical_key);
        if (!ppn || !image.retired_lpns.insert(logical_key).second) {
            throw std::runtime_error(
                "HBF compact logical page was already retired");
        }
        live_pages = &image.live_data_pages_by_block;
    } else if (owner == PageOwner::Mapping &&
               is_metadata_lpn(logical_key)) {
        const auto mapping_vpn = metadata_vpn(logical_key);
        ppn = compact_mapping_ppn(mapping_vpn);
        if (!ppn ||
            !image.retired_mapping_vpns.insert(mapping_vpn).second) {
            throw std::runtime_error(
                "HBF compact mapping page was already retired");
        }
        live_pages = &image.live_mapping_pages_by_block;
    } else {
        throw std::runtime_error(
            "HBF compact retirement requires logical or mapping ownership");
    }

    const auto block_index = *ppn / config_.pages_per_block;
    auto live = live_pages->find(block_index);
    if (live == live_pages->end() || live->second == 0) {
        throw std::runtime_error(
            "HBF compact retirement lost its live block count");
    }
    if (--live->second == 0) {
        live_pages->erase(live);
    }
    const auto block_epoch = blocks_.at(
        static_cast<std::size_t>(block_index)).epoch;
    const bool inserted = programmed_pages_.emplace(
        *ppn,
        PageState{
            .status = PageStatus::Valid,
            .owner = owner,
            .lpn = logical_key,
            .block_epoch = block_epoch,
        }).second;
    if (!inserted) {
        throw std::runtime_error(
            "HBF compact retirement collided with materialized page state");
    }
    invalidate_ppn(*ppn);
}

std::uint64_t HbfDevice::logical_mapping_entry_count() const {
    std::uint64_t entries = lpn_to_ppn_.size();
    if (compact_logical_image_) {
        const auto compact_live =
            compact_logical_image_->page_count -
            compact_logical_image_->retired_lpns.size();
        entries = checked_add(
            entries, compact_live, "HBF compact logical mapping entries");
    }
    for (const auto& [lpn, updates] : pending_lpn_updates_) {
        const bool has_live_update = std::any_of(
            updates.begin(), updates.end(), [this](const PendingMappingUpdate& update) {
                const auto block_index = static_cast<std::size_t>(
                    update.new_ppn / config_.pages_per_block);
                return blocks_.at(block_index).epoch == update.block_epoch &&
                    !blocks_.at(block_index).erase_pending;
            });
        if (has_live_update &&
            !lpn_to_ppn_.contains(lpn) &&
            !compact_lpn_ppn(lpn)) {
            ++entries;
        }
    }
    return entries;
}

std::size_t HbfDevice::source_index(TransactionSource source) const {
    switch (source) {
    case TransactionSource::User:
        return 0;
    case TransactionSource::Mapping:
        return 1;
    case TransactionSource::GC:
        return 2;
    case TransactionSource::Prepopulate:
        return 3;
    }
    throw std::runtime_error("unknown HBF transaction source");
}

std::string HbfDevice::source_name(TransactionSource source) const {
    switch (source) {
    case TransactionSource::User:
        return "user";
    case TransactionSource::Mapping:
        return "mapping";
    case TransactionSource::GC:
        return "gc";
    case TransactionSource::Prepopulate:
        return "prepopulate";
    }
    throw std::runtime_error("unknown HBF transaction source");
}

HeatmapTrafficSource HbfDevice::resolve_heatmap_source(
    TransactionSource source,
    HeatmapTrafficSource attribution) const {
    const auto attribution_index = static_cast<std::size_t>(attribution);
    if (attribution_index >= kHeatmapTrafficSourceCount) {
        throw std::runtime_error("invalid HBF heatmap traffic source");
    }

    HeatmapTrafficSource expected = attribution;
    switch (source) {
    case TransactionSource::User:
        return attribution;
    case TransactionSource::Mapping:
        expected = HeatmapTrafficSource::Mapping;
        break;
    case TransactionSource::GC:
        expected = HeatmapTrafficSource::GarbageCollection;
        break;
    case TransactionSource::Prepopulate:
        expected = HeatmapTrafficSource::Prepopulate;
        break;
    default:
        throw std::runtime_error("unknown HBF transaction source");
    }
    if (attribution != expected) {
        throw std::runtime_error(
            "HBF transaction source and heatmap attribution disagree");
    }
    return attribution;
}

std::string HbfDevice::kind_name(TransactionKind kind) const {
    switch (kind) {
    case TransactionKind::Read:
        return "read";
    case TransactionKind::Program:
        return "program";
    case TransactionKind::Erase:
        return "erase";
    }
    throw std::runtime_error("unknown HBF transaction kind");
}

std::string HbfDevice::role_name(BlockRole role) const {
    switch (role) {
    case BlockRole::Free:
        return "free";
    case BlockRole::StaticReadOnly:
        return "static-read-only";
    case BlockRole::RawPhysical:
        return "raw-physical";
    case BlockRole::Data:
        return "data";
    case BlockRole::Mapping:
        return "mapping";
    case BlockRole::GC:
        return "gc";
    }
    throw std::runtime_error("unknown HBF block role");
}

void HbfDevice::flush_all_dirty_mapping_pages(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    std::size_t wave = 0;
    while (!dirty_mapping_vpns_.empty()) {
        std::vector<std::uint64_t> dirty{
            dirty_mapping_vpns_.begin(), dirty_mapping_vpns_.end()};
        std::sort(dirty.begin(), dirty.end());
        // Fan out each wave like the write-buffer drain: media reservations
        // provide serialization. A mapping-page allocation may itself invoke
        // GC, whose data relocations dirty new mapping generations. Repeat
        // until those generations are also durable in this same drain call.
        const double drain_start_ns = at_ns;
        double drain_finish_ns = at_ns;
        for (const auto mapping_vpn : dirty) {
            const auto events = pending_dirty_mapping_events_.find(mapping_vpn);
            if (events == pending_dirty_mapping_events_.end() ||
                events->second.empty()) {
                dirty_mapping_vpns_.erase(mapping_vpn);
                continue;
            }
            double entry_ns = std::max(
                drain_start_ns, events->second.rbegin()->first);
            flush_dirty_mapping_page(
                mapping_vpn, entry_ns, breakdown, spans);
            drain_finish_ns = std::max(drain_finish_ns, entry_ns);
        }
        at_ns = drain_finish_ns;
        if (++wave > blocks_.size() + 1) {
            throw std::runtime_error(
                "HBF mapping drain failed to converge after GC-generated updates");
        }
    }
}

void HbfDevice::mark_mapping_page_dirty(
    std::uint64_t mapping_vpn,
    double at_ns) {
    const auto sequence = next_cache_touch_sequence_++;
    pending_dirty_mapping_events_[mapping_vpn].emplace(at_ns, sequence);
    dirty_mapping_vpns_.insert(mapping_vpn);
}

void HbfDevice::access_resident_mapping(
    std::uint64_t lpn,
    TransactionSource source,
    MappingAccessKind kind,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    if (source != TransactionSource::User &&
        source != TransactionSource::GC) {
        throw std::runtime_error(
            "HBF resident mapping access source must be foreground user or GC");
    }
    const auto stack = static_cast<std::uint32_t>(
        stack_for_lpn(lpn));
    const auto access_name =
        kind == MappingAccessKind::Lookup ?
        "resident_mapping_lookup" :
        "resident_mapping_update_access";

    breakdown.address_mapping_ns += config_.address_generation_ns;
    add_trace_span(
        spans,
        kind == MappingAccessKind::Lookup ?
            "mapping_address_generation" :
            "mapping_update_address_generation",
        "translation",
        logic_entity(stack),
        at_ns,
        at_ns + config_.address_generation_ns,
        true,
        "lpn" + std::to_string(lpn));
    at_ns += config_.address_generation_ns;

    auto& mapping_dram = logic_dies_.at(stack).mapping_dram_issue;
    const auto issue = reserve(
        at_ns,
        config_.ctrl_dram_issue_ns,
        mapping_dram);
    if (issue.wait_ns > 0.0) {
        stats_.mapping_dram_wait_ops++;
        stats_.mapping_dram_wait_ns += issue.wait_ns;
        stats_.mapping_dram_wait_max_ns = std::max(
            stats_.mapping_dram_wait_max_ns,
            issue.wait_ns);
        breakdown.scheduler_queue_wait_ns += issue.wait_ns;
        trace_wait(
            spans,
            logic_entity(stack),
            at_ns,
            issue.start_ns,
            "wait_resident_mapping_dram");
    }
    const double response_done = std::max(
        issue.finish_ns,
        issue.start_ns + config_.ctrl_dram_latency_ns);
    breakdown.mapping_dram_ns += config_.ctrl_dram_latency_ns;
    add_trace_span(
        spans,
        access_name,
        "metadata",
        logic_entity(stack),
        issue.start_ns,
        response_done,
        true,
        "lpn" + std::to_string(lpn));
    at_ns = response_done;

    if (kind == MappingAccessKind::Lookup) {
        stats_.mapping_lookup_ops++;
        if (source == TransactionSource::User) {
            stats_.mapping_user_lookup_ops++;
        } else {
            stats_.mapping_gc_lookup_ops++;
        }
    } else {
        stats_.mapping_update_ops++;
        if (source == TransactionSource::User) {
            stats_.mapping_user_update_ops++;
        } else {
            stats_.mapping_gc_update_ops++;
        }
        const double update_done = at_ns + config_.mapping_update_ns;
        breakdown.translation_ns += config_.mapping_update_ns;
        add_trace_span(
            spans,
            source == TransactionSource::GC ?
                "gc_resident_mapping_update" :
                "resident_mapping_update",
            "translation",
            "logic/mapping_table",
            at_ns,
            update_done,
            true,
            "lpn" + std::to_string(lpn));
        at_ns = update_done;
    }
}

void HbfDevice::flush_dirty_mapping_page(
    std::uint64_t mapping_vpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    auto events = pending_dirty_mapping_events_.find(mapping_vpn);
    if (events == pending_dirty_mapping_events_.end()) {
        dirty_mapping_vpns_.erase(mapping_vpn);
        return;
    }
    const auto snapshot = latest_touch_through(events->second, at_ns);
    if (!snapshot) {
        return;
    }
    const auto clear_snapshot = [this, mapping_vpn, snapshot]() {
        auto pending = pending_dirty_mapping_events_.find(mapping_vpn);
        if (pending == pending_dirty_mapping_events_.end()) {
            return;
        }
        pending->second.erase(
            pending->second.begin(), pending->second.upper_bound(*snapshot));
        if (pending->second.empty()) {
            pending_dirty_mapping_events_.erase(pending);
            dirty_mapping_vpns_.erase(mapping_vpn);
        }
    };
    const auto preferred_plane = mapping_plane_for_vpn(mapping_vpn);
    maybe_run_gc(
        at_ns,
        breakdown,
        spans,
        1,
        stack_for_vpn(mapping_vpn),
        BlockRole::Mapping,
        preferred_plane);
    const auto new_ppn = allocate_free_page(
        at_ns,
        breakdown,
        spans,
        BlockRole::Mapping,
        stack_for_vpn(mapping_vpn),
        preferred_plane);
    const double program_done = schedule_program_page(
        new_ppn,
        at_ns,
        breakdown,
        spans,
        TransactionSource::Mapping,
        HeatmapTrafficSource::Mapping);

    const auto program_commit_sequence = schedule_media_program_commit(
        new_ppn,
        metadata_lpn(mapping_vpn),
        PageOwner::Mapping,
        program_done);
    schedule_vpn_mapping_commit(
        mapping_vpn, new_ppn, program_done, program_commit_sequence);
    at_ns = program_done;
    clear_snapshot();
    stats_.physical_write_bytes += config_.page_size_bytes;
    stats_.mapping_program_payload_bytes += config_.page_size_bytes;
    stats_.page_programs++;
    stats_.mapping_page_programs++;
}

std::optional<std::uint64_t> HbfDevice::lookup_lpn(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // HBF exposes ordered memory semantics per LPN: a later-arriving access
    // waits for every earlier write to commit. This is distinct from the
    // forbidden case of calling issue() with a decreasing arrival timestamp.
    if (const auto mapping = lpn_to_ppn_.find(lpn);
        mapping != lpn_to_ppn_.end()) {
        wait_for_pending_block_erase(
            mapping->second, at_ns, breakdown, spans);
    } else if (const auto compact = compact_lpn_ppn(lpn)) {
        wait_for_pending_block_erase(
            *compact, at_ns, breakdown, spans);
    }
    wait_for_prior_lpn_commit(lpn, at_ns, breakdown, spans);
    access_resident_mapping(
        lpn,
        TransactionSource::User,
        MappingAccessKind::Lookup,
        at_ns,
        breakdown,
        spans);
    return visible_lpn_at(lpn, at_ns);
}

bool HbfDevice::write_buffer_covers(
    const WriteBufferEntry& entry,
    std::uint64_t begin,
    std::uint64_t end) const {
    return dirty_ranges_cover(entry.ranges, begin, end);
}

bool HbfDevice::dirty_ranges_cover(
    const std::vector<DirtyRange>& ranges,
    std::uint64_t begin,
    std::uint64_t end) const {
    std::uint64_t cursor = begin;
    for (const auto& range : ranges) {
        if (range.end <= cursor) {
            continue;
        }
        if (range.begin > cursor) {
            return false;
        }
        cursor = std::max(cursor, range.end);
        if (cursor >= end) {
            return true;
        }
    }
    return cursor >= end;
}

bool HbfDevice::write_buffer_full_page(const WriteBufferEntry& entry) const {
    return write_buffer_covers(entry, 0, config_.page_size_bytes);
}

std::uint64_t HbfDevice::write_buffer_covered_bytes(const WriteBufferEntry& entry) const {
    std::uint64_t bytes = 0;
    for (const auto& range : entry.ranges) {
        bytes += range.end - range.begin;
    }
    return bytes;
}

double HbfDevice::serve_read_from_write_buffer(
    std::uint64_t lpn,
    std::uint64_t begin,
    std::uint64_t end,
    const std::vector<DirtyRange>& ranges,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto stack = static_cast<std::uint32_t>(stack_for_lpn(lpn));
    auto& logic_die = logic_dies_.at(stack);
    const auto bytes = end - begin;
    std::uint64_t buffered_bytes = 0;
    for (const auto& range : ranges) {
        const auto lo = std::max(range.begin, begin);
        const auto hi = std::min(range.end, end);
        if (hi > lo) {
            buffered_bytes += hi - lo;
        }
    }
    const auto erased_bytes = bytes - buffered_bytes;
    const double sram_ns = transfer_time_ns(bytes, config_.logic_sram_bandwidth_GBps);
    auto sram = reserve(earliest_ns, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    trace_wait(spans, logic_entity(stack), earliest_ns, sram.start_ns, "wait_sram");
    add_trace_span(
        spans,
        erased_bytes == 0 ? "write_buffer_read_hit" : "write_buffer_read_assemble",
        "sram",
        logic_entity(stack),
        sram.start_ns,
        sram.finish_ns,
        true,
        "lpn" + std::to_string(lpn) + " buffered=" +
            std::to_string(buffered_bytes) + "B erased=" +
            std::to_string(erased_bytes) + "B");
    // Buffered data still crosses the stack's external interface (already
    // decoded SRAM content: data bytes only, no ECC pass).
    stats_.write_buffer_read_hits++;
    // Count only bytes the buffer actually holds: a read of a partially
    // buffered, never-mapped page is served here with less than full coverage.
    stats_.write_buffer_read_bytes += buffered_bytes;
    return schedule_external_read_egress(
        stack,
        bytes,
        sram.finish_ns,
        breakdown,
        spans,
        "write_buffer_read_hit_hbio_out",
        erased_bytes == 0 ? "buffered payload" : "buffered + erased-fill payload");
}

double HbfDevice::admit_foreground_page_read(
    std::size_t stack,
    double offered_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    auto& releases = page_read_credit_release_by_stack_.at(stack);
    const auto depth =
        static_cast<std::size_t>(config_.page_read_queue_depth_per_stack);
    double admitted_ns = offered_ns;
    if (releases.size() >= depth) {
        const auto earliest = releases.begin();
        admitted_ns = std::max(admitted_ns, *earliest);
        releases.erase(earliest);
    }
    const double wait_ns = admitted_ns - offered_ns;
    stats_.page_read_admission_events = checked_add(
        stats_.page_read_admission_events,
        1,
        "HBF page-read admission event count");
    if (wait_ns > 0.0) {
        stats_.page_read_admission_waited_pages = checked_add(
            stats_.page_read_admission_waited_pages,
            1,
            "HBF page-read admission waited-page count");
        const double updated_wait =
            stats_.page_read_admission_wait_ns + wait_ns;
        if (!std::isfinite(updated_wait)) {
            throw std::runtime_error(
                "HBF page-read admission wait work is not finite");
        }
        stats_.page_read_admission_wait_ns = updated_wait;
        stats_.page_read_admission_max_wait_ns = std::max(
            stats_.page_read_admission_max_wait_ns,
            wait_ns);
        breakdown.scheduler_queue_wait_ns += wait_ns;
        trace_wait(
            spans,
            logic_entity(static_cast<std::uint32_t>(stack)),
            offered_ns,
            admitted_ns,
            "wait_page_read_credit");
    }
    return admitted_ns;
}

void HbfDevice::complete_foreground_page_read(
    std::size_t stack,
    double finish_ns) {
    if (!std::isfinite(finish_ns) || finish_ns < 0.0) {
        throw std::runtime_error(
            "HBF page-read completion frontier is invalid");
    }
    auto& releases = page_read_credit_release_by_stack_.at(stack);
    releases.insert(finish_ns);
    if (releases.size() >
        static_cast<std::size_t>(config_.page_read_queue_depth_per_stack)) {
        throw std::runtime_error(
            "HBF page-read admission exceeded configured per-stack depth");
    }
}

double HbfDevice::schedule_external_write_ingress(
    std::size_t stack,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& detail,
    const std::string& sram_span_name) {
    auto& logic_die = logic_dies_.at(stack);
    const double payload_ns = transfer_time_ns(bytes, config_.hb_io_bandwidth_GBps);

    // The top-level request command has already crossed the command port.
    // Only the exact payload fragment enters here; the later flash-program
    // command is generated by the logic die and starts at TSV.
    auto payload = reserve(earliest_ns, payload_ns, logic_die.hb_io_data);
    logic_die.hb_io_data_busy_ns += payload_ns;
    breakdown.scheduler_queue_wait_ns += payload.wait_ns;
    breakdown.hb_io_transfer_ns += payload_ns;
    trace_wait(
        spans,
        logic_entity(static_cast<std::uint32_t>(stack)),
        earliest_ns,
        payload.start_ns,
        "wait_hbio_data");
    add_trace_span(
        spans,
        "user/data_in_hbio",
        "hbio",
        logic_entity(static_cast<std::uint32_t>(stack)),
        payload.start_ns,
        payload.finish_ns,
        true,
        std::to_string(bytes) + "B payload " + detail);

    const double ingress_ready_ns = payload.finish_ns;
    const double sram_ns = transfer_time_ns(bytes, config_.logic_sram_bandwidth_GBps);
    auto sram = reserve(ingress_ready_ns, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    trace_wait(
        spans,
        logic_entity(static_cast<std::uint32_t>(stack)),
        ingress_ready_ns,
        sram.start_ns,
        "wait_sram");
    add_trace_span(
        spans,
        sram_span_name,
        "sram",
        logic_entity(static_cast<std::uint32_t>(stack)),
        sram.start_ns,
        sram.finish_ns,
        true,
        std::to_string(bytes) + "B payload " + detail);
    return sram.finish_ns;
}

double HbfDevice::schedule_external_request_command(
    std::size_t stack,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& detail) {
    auto& logic_die = logic_dies_.at(stack);
    const double command_ns = transfer_time_ns(
        config_.command_address_bytes, config_.hb_io_bandwidth_GBps);
    auto command = reserve(earliest_ns, command_ns, logic_die.hb_io_command);
    logic_die.hb_io_command_busy_ns += command_ns;
    breakdown.scheduler_queue_wait_ns += command.wait_ns;
    breakdown.hb_io_transfer_ns += command_ns;
    trace_wait(
        spans,
        logic_entity(static_cast<std::uint32_t>(stack)),
        earliest_ns,
        command.start_ns,
        "wait_hbio_request");
    add_trace_span(
        spans,
        "user/request_hbio",
        "hbio",
        logic_entity(static_cast<std::uint32_t>(stack)),
        command.start_ns,
        command.finish_ns,
        true,
        std::to_string(config_.command_address_bytes) + "B request " + detail);
    return command.finish_ns;
}

double HbfDevice::schedule_sram_transfer(
    std::size_t stack,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& name,
    const std::string& detail) {
    if (bytes == 0) {
        return earliest_ns;
    }
    auto& logic_die = logic_dies_.at(stack);
    const double duration_ns = transfer_time_ns(
        bytes, config_.logic_sram_bandwidth_GBps);
    auto transfer = reserve(earliest_ns, duration_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += transfer.wait_ns;
    breakdown.sram_staging_ns += duration_ns;
    trace_wait(
        spans,
        logic_entity(static_cast<std::uint32_t>(stack)),
        earliest_ns,
        transfer.start_ns,
        "wait_sram");
    add_trace_span(
        spans,
        name,
        "sram",
        logic_entity(static_cast<std::uint32_t>(stack)),
        transfer.start_ns,
        transfer.finish_ns,
        true,
        std::to_string(bytes) + "B " + detail);
    return transfer.finish_ns;
}

double HbfDevice::schedule_external_read_egress(
    std::size_t stack,
    std::uint64_t bytes,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    const std::string& name,
    const std::string& detail) {
    if (bytes == 0 || bytes > config_.page_size_bytes) {
        throw std::runtime_error(
            "HBF external read egress must contain 1..page_size payload bytes");
    }
    auto& logic_die = logic_dies_.at(stack);
    const double duration_ns = transfer_time_ns(
        bytes, config_.hb_io_bandwidth_GBps);
    auto transfer = reserve(earliest_ns, duration_ns, logic_die.hb_io_data);
    logic_die.hb_io_data_busy_ns += duration_ns;
    breakdown.scheduler_queue_wait_ns += transfer.wait_ns;
    breakdown.hb_io_transfer_ns += duration_ns;
    trace_wait(
        spans,
        logic_entity(static_cast<std::uint32_t>(stack)),
        earliest_ns,
        transfer.start_ns,
        "wait_hbio_data");
    add_trace_span(
        spans,
        name,
        "hbio",
        logic_entity(static_cast<std::uint32_t>(stack)),
        transfer.start_ns,
        transfer.finish_ns,
        true,
        std::to_string(bytes) + "B payload " + detail);
    return transfer.finish_ns;
}

double HbfDevice::stage_write_buffer_range(
    std::uint64_t lpn,
    std::uint64_t begin,
    std::uint64_t end,
    HeatmapTrafficSource heatmap_source,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto stack = static_cast<std::uint32_t>(stack_for_lpn(lpn));
    const auto bytes = end - begin;
    auto& die_write_buffer = write_buffer(stack);
    auto& die_lru = write_buffer_lru(stack);
    auto found = die_write_buffer.find(lpn);
    double admit_ns = earliest_ns;
    if (found == die_write_buffer.end()) {
        // A new entry needs a free SRAM slot. Occupancy counts both live
        // entries and flushed pages whose programs are still in flight; at
        // capacity the admit waits for the oldest program to complete (the
        // real backpressure of a sustained overload). An overwrite of a live
        // entry consumes no new slot and never waits here.
        auto& slot_releases = write_buffer_slot_release_by_stack_.at(stack);
        const auto release_completed_slots = [&slot_releases](double through_ns) {
            slot_releases.erase(slot_releases.begin(), slot_releases.upper_bound(through_ns));
        };
        const auto slot_capacity_reached = [&]() {
            const auto live = static_cast<std::uint64_t>(die_write_buffer.size());
            const auto inflight = static_cast<std::uint64_t>(slot_releases.size());
            if (live > config_.write_buffer_pages ||
                inflight > config_.write_buffer_pages - live) {
                throw std::runtime_error("HBF write-buffer slot accounting exceeded capacity");
            }
            return live + inflight == config_.write_buffer_pages;
        };

        release_completed_slots(admit_ns);
        while (slot_capacity_reached()) {
            if (!slot_releases.empty()) {
                admit_ns = std::max(admit_ns, *slot_releases.begin());
                release_completed_slots(admit_ns);
                continue;
            }
            if (die_lru.empty()) {
                throw std::runtime_error(
                    "HBF write-buffer capacity is full but has no live or in-flight owner");
            }

            // Every occupied slot is still live. Evict the LRU entry first;
            // flush_write_buffer_entry removes it from the live buffer and
            // records the media-completion release. Only after that release
            // is reached may the new LPN be installed below.
            const auto victim_lpn = die_lru.back();
            double eviction_done_ns = admit_ns;
            flush_write_buffer_entry(
                victim_lpn, eviction_done_ns, breakdown, spans);
            admit_ns = std::max(admit_ns, eviction_done_ns);
            release_completed_slots(admit_ns);
        }
        if (admit_ns > earliest_ns) {
            stats_.write_buffer_slot_wait_ops++;
            stats_.write_buffer_slot_wait_ns += admit_ns - earliest_ns;
            breakdown.scheduler_queue_wait_ns += admit_ns - earliest_ns;
            trace_wait(spans, logic_entity(stack), earliest_ns, admit_ns, "wait_write_buffer_slot");
        }
        if (slot_capacity_reached()) {
            throw std::runtime_error("HBF write-buffer admission did not release a slot");
        }
        die_lru.push_front(lpn);
        found = die_write_buffer.emplace(lpn, WriteBufferEntry{
            .lpn = lpn,
            .ranges = {},
            .iterator = die_lru.begin(),
            .ready_ns = 0.0,
        }).first;
        stats_.write_buffer_misses++;
    } else {
        die_lru.splice(die_lru.begin(), die_lru, found->second.iterator);
        found->second.iterator = die_lru.begin();
        stats_.write_buffer_hits++;
    }

    const auto overlapped = insert_merged_range(
        found->second.ranges,
        DirtyRange{
            .begin = begin,
            .end = end,
            .heatmap_source = heatmap_source,
        });
    stats_.write_buffer_merged_bytes += overlapped;

    // This is the one external payload crossing for a coalesced write. Later
    // destage reads the page from SRAM and must not charge a second full-page
    // HBIO transfer.
    const double stage_done_ns = schedule_external_write_ingress(
        stack,
        bytes,
        admit_ns,
        breakdown,
        spans,
        "buffered lpn" + std::to_string(lpn) + " dirty=" +
            std::to_string(write_buffer_covered_bytes(found->second)) + "/" +
            std::to_string(config_.page_size_bytes) + "B",
        "write_buffer_stage");
    found->second.ready_ns = std::max(found->second.ready_ns, stage_done_ns);
    return stage_done_ns;
}

void HbfDevice::flush_write_buffer_entry(
    std::uint64_t lpn,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto stack = static_cast<std::uint32_t>(stack_for_lpn(lpn));
    auto& die_write_buffer = write_buffer(stack);
    auto found = die_write_buffer.find(lpn);
    if (found == die_write_buffer.end()) {
        return;
    }
    const auto entry = found->second;
    // NAND programs are atomic at page granularity, while buffered writes may
    // combine byte ranges from several request classes. Preserve byte-level
    // last-writer provenance in the entry, then conservatively label the one
    // physical page operation with the source owning the most dirty bytes.
    // Equal-byte ties resolve by HeatmapTrafficSource enum order. This keeps
    // physical byte/access conservation exact without relabeling a deferred
    // flush from whichever request (or drain) happens to trigger it.
    const auto entry_heatmap_source = dominant_dirty_source(entry.ranges);
    at_ns = std::max(at_ns, entry.ready_ns);
    const double flush_ready_ns = at_ns;
    auto& logic_die = logic_dies_.at(stack);
    const auto flush_sram = reserve(
        flush_ready_ns,
        config_.mapping_update_ns,
        logic_die.sram);
    breakdown.scheduler_queue_wait_ns += flush_sram.wait_ns;
    trace_wait(
        spans,
        logic_entity(stack),
        flush_ready_ns,
        flush_sram.start_ns,
        "wait_write_buffer_flush_sram");
    add_trace_span(
        spans,
        "write_buffer_flush",
        "sram",
        logic_entity(stack),
        flush_sram.start_ns,
        flush_sram.finish_ns,
        true,
        "lpn" + std::to_string(lpn));
    breakdown.sram_staging_ns += config_.mapping_update_ns;
    at_ns = flush_sram.finish_ns;

    // Resolve the mapping at flush time: GC may have relocated the page since
    // the write was staged. State-only read; translation time was charged at stage.
    wait_for_prior_lpn_commit(lpn, at_ns, breakdown, spans);
    const auto current_ppn = visible_lpn_at(lpn, at_ns);

    if (current_ppn && !write_buffer_full_page(entry)) {
        add_trace_span(
            spans,
            "partial_page_merge_on_flush",
            "translation",
            logic_entity(stack),
            at_ns,
            at_ns + config_.mapping_update_ns,
            true,
            "lpn" + std::to_string(lpn));
        breakdown.translation_ns += config_.mapping_update_ns;
        at_ns += config_.mapping_update_ns;
        at_ns = schedule_read_page(
            *current_ppn,
            at_ns,
            breakdown,
            spans,
            TransactionSource::User,
            entry_heatmap_source,
            ReadPayloadRoute::Internal,
            0);
        stats_.physical_read_bytes += config_.page_size_bytes;
        stats_.page_reads++;
        at_ns = schedule_sram_transfer(
            stack,
            write_buffer_covered_bytes(entry),
            at_ns,
            breakdown,
            spans,
            "partial_page_buffer_overlay",
            "dirty bytes over old media image");
    } else if (!current_ppn && !write_buffer_full_page(entry)) {
        // Explicit erased-value initialization supplies bytes that have never
        // existed in the logical image. This is a modeled NAND behavior, not
        // an implicit zero-filled backing page or an uncharged media read.
        at_ns = schedule_sram_transfer(
            stack,
            config_.page_size_bytes - write_buffer_covered_bytes(entry),
            at_ns,
            breakdown,
            spans,
            "partial_page_erased_fill",
            "erased-value fill for unmapped buffered page");
    }

    maybe_run_gc(
        at_ns,
        breakdown,
        spans,
        1,
        stack_for_lpn(lpn),
        BlockRole::Data);
    const auto new_ppn = allocate_free_page(
        at_ns,
        breakdown,
        spans,
        BlockRole::Data,
        stack_for_lpn(lpn));

    const double program_done = schedule_program_page(
        new_ppn,
        at_ns,
        breakdown,
        spans,
        TransactionSource::User,
        entry_heatmap_source);
    const auto program_commit_sequence = schedule_media_program_commit(
        new_ppn,
        lpn,
        PageOwner::Logical,
        program_done);
    double mapping_ready = program_done;
    access_resident_mapping(
        lpn,
        TransactionSource::User,
        MappingAccessKind::Update,
        mapping_ready,
        breakdown,
        spans);
    const double page_done = mapping_ready;
    schedule_lpn_mapping_commit(
        lpn, new_ppn, page_done, program_commit_sequence);
    mark_mapping_page_dirty(mapping_vpn_for_lpn(lpn), page_done);
    at_ns = page_done;
    // The dirty resident entry persists through its checkpoint at drain.

    const auto generation = next_buffer_generation_++;
    inflight_buffered_writes_[lpn].push_back(InflightBufferedWrite{
        .generation = generation,
        .ranges = entry.ranges,
        .ready_ns = entry.ready_ns,
        .commit_ns = page_done,
        .target_ppn = new_ppn,
        .target_block_epoch = blocks_.at(
            static_cast<std::size_t>(new_ppn / config_.pages_per_block)).epoch,
    });
    (void)schedule_commit(stack, page_done, [this, lpn, generation]() {
        const auto found_inflight = inflight_buffered_writes_.find(lpn);
        if (found_inflight == inflight_buffered_writes_.end()) {
            return;
        }
        std::erase_if(
            found_inflight->second,
            [generation](const InflightBufferedWrite& write) {
                return write.generation == generation;
            });
        if (found_inflight->second.empty()) {
            inflight_buffered_writes_.erase(found_inflight);
        }
    });

    write_buffer_lru(stack).erase(entry.iterator);
    die_write_buffer.erase(found);
    // The generation remains readable from SRAM until the logical mapping
    // commits. Releasing its physical slot at program_done allowed a new LPN
    // to reuse the same one-page buffer while the old generation was still
    // reported as an SRAM hit. Keep ownership and readability on one deadline.
    write_buffer_slot_release_by_stack_.at(stack).insert(page_done);
    stats_.physical_write_bytes += config_.page_size_bytes;
    stats_.data_program_payload_bytes += config_.page_size_bytes;
    stats_.data_programs++;
    stats_.page_programs++;
    stats_.write_buffer_flushes++;
}

void HbfDevice::flush_all_write_buffer_entries(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    // Drain fans out: every entry's flush is issued from the drain start and
    // the media resource reservations do the real serialization; a single
    // chained cursor would serialize the issue times themselves.
    const double drain_start_ns = at_ns;
    double drain_finish_ns = at_ns;
    for (std::size_t stack = 0; stack < config_.stacks; ++stack) {
        auto& die_lru = write_buffer_lru(stack);
        while (!die_lru.empty()) {
            const auto victim_lpn = die_lru.back();
            double entry_ns = drain_start_ns;
            flush_write_buffer_entry(victim_lpn, entry_ns, breakdown, spans);
            drain_finish_ns = std::max(drain_finish_ns, entry_ns);
        }
    }
    at_ns = drain_finish_ns;
}

std::uint64_t HbfDevice::allocate_free_page(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    BlockRole role,
    std::size_t stack,
    std::optional<std::size_t> preferred_plane) {
    if (free_pages_per_stack_.at(stack) == 0) {
        throw std::runtime_error(
            "HBF stack " + std::to_string(stack) +
            " free-page pool exhausted before GC could free space (shared-nothing "
            "FTL: stacks cannot borrow pages from each other)");
    }
    breakdown.address_mapping_ns += config_.free_page_allocation_ns;
    add_trace_span(
        spans,
        "free_page_alloc",
        "translation",
        "logic/free_page_allocator",
        at_ns,
        at_ns + config_.free_page_allocation_ns,
        true,
        role_name(role));
    at_ns += config_.free_page_allocation_ns;

    const auto try_plane = [&](std::size_t plane_candidate) -> std::optional<std::uint64_t> {
        auto& plane = planes_.at(plane_candidate);
        auto& active = active_block_ref(plane, role);
        if (!active || blocks_.at(*active).free_pages == 0) {
            active = std::nullopt;
            if (plane.free_blocks.empty()) {
                return std::nullopt;
            }
            if (role != BlockRole::GC &&
                plane.free_blocks.size() <= config_.gc_reserved_free_blocks_per_plane) {
                return std::nullopt;
            }
            active = allocate_block_from_plane(plane_candidate, role);
        }
        const auto ppn = allocate_page_from_block(*active);
        if (blocks_.at(*active).free_pages == 0) {
            active = std::nullopt;
        }
        return ppn;
    };

    const auto pps = planes_per_stack();
    const auto stack_base = stack * pps;
    if (preferred_plane &&
        (*preferred_plane < stack_base || *preferred_plane >= stack_base + pps)) {
        throw std::runtime_error(
            "HBF preferred allocation plane belongs to another stack");
    }
    if (preferred_plane) {
        if (auto ppn = try_plane(*preferred_plane)) {
            return *ppn;
        }
    }
    const auto first = choose_allocation_plane(stack, role);
    for (std::size_t offset = 0; offset < pps; ++offset) {
        const auto plane_candidate = stack_base + (first + offset) % pps;
        if (auto ppn = try_plane(plane_candidate)) {
            return *ppn;
        }
    }

    throw std::runtime_error(
        "HBF cannot allocate a page for role " + role_name(role) +
        ": free pages exist only in reserved free blocks or role-active blocks, "
        "and GC found no eligible victim (gc-reserved-free-blocks-per-plane may "
        "be infeasible for this geometry and workload)");
}

std::size_t HbfDevice::choose_allocation_plane(
    std::size_t stack,
    BlockRole role) {
    const auto pps = planes_per_stack();
    if (pps == 0) {
        throw std::runtime_error("HBF has no planes");
    }
    std::vector<std::size_t>* cursors = nullptr;
    switch (role) {
    case BlockRole::Data:
        cursors = &next_data_allocation_plane_per_stack_;
        break;
    case BlockRole::Mapping:
        cursors = &next_mapping_allocation_plane_per_stack_;
        break;
    case BlockRole::GC:
        cursors = &next_gc_allocation_plane_per_stack_;
        break;
    case BlockRole::Free:
    case BlockRole::StaticReadOnly:
    case BlockRole::RawPhysical:
        throw std::runtime_error(
            "HBF allocation cursor requires a Data, Mapping, or GC role");
    }
    auto& cursor = cursors->at(stack);
    const auto selected = cursor % pps;
    cursor = (cursor + 1) % pps;
    return selected;
}

std::optional<std::size_t> HbfDevice::preview_allocation_plane(
    std::size_t stack,
    BlockRole role,
    std::optional<std::size_t> preferred_plane) const {
    const auto pps = planes_per_stack();
    if (pps == 0) {
        throw std::runtime_error("HBF has no planes");
    }
    const auto stack_base = stack * pps;
    if (preferred_plane &&
        (*preferred_plane < stack_base || *preferred_plane >= stack_base + pps)) {
        throw std::runtime_error(
            "HBF preferred allocation plane belongs to another stack");
    }

    const auto can_allocate = [&](std::size_t plane_index) {
        const auto& plane = planes_.at(plane_index);
        const std::optional<std::size_t>* active = nullptr;
        switch (role) {
        case BlockRole::Data:
            active = &plane.active_data_block;
            break;
        case BlockRole::Mapping:
            active = &plane.active_mapping_block;
            break;
        case BlockRole::GC:
            active = &plane.active_gc_block;
            break;
        case BlockRole::Free:
        case BlockRole::StaticReadOnly:
        case BlockRole::RawPhysical:
            throw std::runtime_error(
                "HBF allocation preview requires a Data, Mapping, or GC role");
        }
        if (*active && blocks_.at(**active).free_pages != 0) {
            return true;
        }
        if (plane.free_blocks.empty()) {
            return false;
        }
        return role == BlockRole::GC ||
            plane.free_blocks.size() > config_.gc_reserved_free_blocks_per_plane;
    };

    if (preferred_plane && can_allocate(*preferred_plane)) {
        return preferred_plane;
    }

    const std::vector<std::size_t>* cursors = nullptr;
    switch (role) {
    case BlockRole::Data:
        cursors = &next_data_allocation_plane_per_stack_;
        break;
    case BlockRole::Mapping:
        cursors = &next_mapping_allocation_plane_per_stack_;
        break;
    case BlockRole::GC:
        cursors = &next_gc_allocation_plane_per_stack_;
        break;
    case BlockRole::Free:
    case BlockRole::StaticReadOnly:
    case BlockRole::RawPhysical:
        throw std::runtime_error(
            "HBF allocation preview requires a Data, Mapping, or GC role");
    }
    const auto first = cursors->at(stack) % pps;
    for (std::size_t offset = 0; offset < pps; ++offset) {
        const auto plane_index = stack_base + (first + offset) % pps;
        if (can_allocate(plane_index)) {
            return plane_index;
        }
    }
    return std::nullopt;
}

std::size_t HbfDevice::allocate_block_from_plane(std::size_t plane_index, BlockRole role) {
    auto& plane = planes_.at(plane_index);
    if (plane.free_blocks.empty()) {
        throw std::runtime_error("HBF plane free-block pool exhausted");
    }
    const auto block_index = plane.free_blocks.front();
    plane.free_blocks.pop_front();
    auto& block = blocks_.at(block_index);
    if (block.role != BlockRole::Free || block.free_pages != config_.pages_per_block ||
        block.valid_pages != 0 || block.invalid_pages != 0 || block.next_page != 0 ||
        block.pending_program_pages != 0 || block.erase_pending ||
        block.pending_mapping_publications != 0) {
        throw std::runtime_error("HBF free-block pool contains a non-erased block");
    }
    block.role = role;
    return block_index;
}

std::optional<std::size_t>& HbfDevice::active_block_ref(PlaneState& plane, BlockRole role) {
    switch (role) {
    case BlockRole::Data:
        return plane.active_data_block;
    case BlockRole::Mapping:
        return plane.active_mapping_block;
    case BlockRole::GC:
        return plane.active_gc_block;
    case BlockRole::RawPhysical:
        throw std::runtime_error("HBF FTL cannot allocate a page from a raw-physical block");
    case BlockRole::StaticReadOnly:
        throw std::runtime_error("HBF cannot allocate a page from a static-data block");
    case BlockRole::Free:
        throw std::runtime_error("HBF cannot allocate a page from a free-role block");
    }
    throw std::runtime_error("unknown HBF block role");
}

std::uint64_t HbfDevice::allocate_page_from_block(std::size_t block_index) {
    auto& block = blocks_.at(block_index);
    if (block.role == BlockRole::Free || block.free_pages == 0 ||
        block.next_page >= config_.pages_per_block) {
        throw std::runtime_error("HBF allocator selected a block with no programmable pages");
    }
    const auto page = block.next_page++;
    block.free_pages--;
    block.pending_program_pages++;
    free_pages_--;
    free_pages_per_stack_.at(stack_of_block(block_index))--;
    const auto ppn = static_cast<std::uint64_t>(block_index) * config_.pages_per_block + page;
    const auto inserted = programmed_pages_.emplace(ppn, PageState{});
    if (!inserted.second && inserted.first->second.status != PageStatus::Erased) {
        throw std::runtime_error("HBF allocator selected a programmed physical page");
    }
    inserted.first->second.status = PageStatus::Erased;
    inserted.first->second.owner = PageOwner::Unassigned;
    inserted.first->second.lpn = 0;
    inserted.first->second.block_epoch = block.epoch;
    return ppn;
}

std::uint64_t HbfDevice::allocate_compact_pages_on_plane(
    std::size_t plane_index,
    BlockRole role,
    std::uint32_t page_count,
    std::vector<std::uint64_t>* assigned_blocks,
    std::unordered_map<std::uint64_t, std::uint32_t>*
        live_pages_by_block) {
    if (page_count == 0 ||
        (role != BlockRole::Data && role != BlockRole::Mapping)) {
        throw std::runtime_error(
            "HBF compact allocation requires positive Data/Mapping pages");
    }
    auto& plane = planes_.at(plane_index);
    std::optional<std::uint64_t> first_ppn;
    auto remaining = page_count;
    while (remaining != 0) {
        auto& active = active_block_ref(plane, role);
        if (!active || blocks_.at(*active).free_pages == 0) {
            active.reset();
            if (plane.free_blocks.empty() ||
                plane.free_blocks.size() <=
                    config_.gc_reserved_free_blocks_per_plane) {
                throw std::runtime_error(
                    "HBF compact image exhausted an allocatable plane block");
            }
            active = allocate_block_from_plane(plane_index, role);
            if (assigned_blocks != nullptr) {
                assigned_blocks->push_back(*active);
            }
        }
        auto& block = blocks_.at(*active);
        if (assigned_blocks != nullptr &&
            (assigned_blocks->empty() || assigned_blocks->back() != *active)) {
            throw std::runtime_error(
                "HBF compact data-block directory lost allocator order");
        }
        const auto take = std::min<std::uint32_t>(remaining, block.free_pages);
        const auto first_page = block.next_page;
        const auto ppn =
            static_cast<std::uint64_t>(*active) * config_.pages_per_block +
            first_page;
        if (!first_ppn) {
            first_ppn = ppn;
        }
        block.set_valid_range(first_page, take);
        block.next_page += take;
        block.free_pages -= take;
        block.valid_pages += take;
        if (live_pages_by_block != nullptr) {
            auto& live = (*live_pages_by_block)[*active];
            if (take > std::numeric_limits<std::uint32_t>::max() - live) {
                throw std::runtime_error(
                    "HBF compact per-block live-page count overflowed");
            }
            live += take;
        }
        free_pages_ -= take;
        free_pages_per_stack_.at(stack_of_block(*active)) -= take;
        remaining -= take;
        if (block.free_pages == 0) {
            active.reset();
        }
    }
    return *first_ppn;
}

std::uint64_t HbfDevice::gc_relocation_capacity(std::size_t stack) const {
    return gc_headroom(stack, BlockRole::Data).relocation_pages;
}

HbfDevice::GcHeadroom HbfDevice::gc_headroom(
    std::size_t stack,
    BlockRole allocation_role) const {
    if (allocation_role != BlockRole::Data &&
        allocation_role != BlockRole::Mapping) {
        throw std::runtime_error(
            "HBF GC headroom requires a Data or Mapping allocation role");
    }

    GcHeadroom headroom;
    const auto pps = planes_per_stack();
    for (std::size_t plane_index = stack * pps;
         plane_index < (stack + 1) * pps;
         ++plane_index) {
        const auto& plane = planes_.at(plane_index);
        const auto validate_active = [&](const std::optional<std::size_t>& active,
                                         BlockRole expected_role,
                                         const char* name)
            -> std::uint64_t {
            if (!active) {
                return 0;
            }
            const auto& block = blocks_.at(*active);
            if (block_plane_index(*active) != plane_index ||
                block.role != expected_role || block.erase_pending ||
                block.free_pages == 0) {
                throw std::runtime_error(std::string("HBF invalid active ") + name +
                    " block while computing GC headroom");
            }
            return block.free_pages;
        };

        if ((plane.active_data_block && plane.active_mapping_block &&
                *plane.active_data_block == *plane.active_mapping_block) ||
            (plane.active_data_block && plane.active_gc_block &&
                *plane.active_data_block == *plane.active_gc_block) ||
            (plane.active_mapping_block && plane.active_gc_block &&
                *plane.active_mapping_block == *plane.active_gc_block)) {
            throw std::runtime_error(
                "HBF active Data/Mapping/GC block roles alias each other");
        }

        const auto active_data_pages = validate_active(
            plane.active_data_block, BlockRole::Data, "Data");
        const auto active_mapping_pages = validate_active(
            plane.active_mapping_block, BlockRole::Mapping, "Mapping");
        const auto active_gc_pages = validate_active(
            plane.active_gc_block, BlockRole::GC, "GC");
        const auto active_role_pages = allocation_role == BlockRole::Data ?
            active_data_pages : active_mapping_pages;

        headroom.foreground_pages = checked_add(
            headroom.foreground_pages,
            active_role_pages,
            "HBF foreground active-block headroom");
        headroom.relocation_pages = checked_add(
            headroom.relocation_pages,
            active_gc_pages,
            "HBF active-GC relocation headroom");

        const auto free_blocks =
            static_cast<std::uint64_t>(plane.free_blocks.size());
        const auto free_block_pages = checked_mul(
            free_blocks,
            config_.pages_per_block,
            "HBF whole-free-block GC headroom");
        headroom.relocation_pages = checked_add(
            headroom.relocation_pages,
            free_block_pages,
            "HBF total GC relocation headroom");

        const auto foreground_blocks =
            free_blocks > config_.gc_reserved_free_blocks_per_plane ?
            free_blocks - config_.gc_reserved_free_blocks_per_plane : 0;
        headroom.foreground_pages = checked_add(
            headroom.foreground_pages,
            checked_mul(
                foreground_blocks,
                config_.pages_per_block,
                "HBF foreground whole-free-block headroom"),
            "HBF total foreground GC headroom");
        headroom.reserve_target_pages = checked_add(
            headroom.reserve_target_pages,
            checked_mul(
                config_.gc_reserved_free_blocks_per_plane,
                config_.pages_per_block,
                "HBF GC reserve target per plane"),
            "HBF total GC reserve target");
    }
    return headroom;
}

std::optional<std::size_t> HbfDevice::choose_gc_victim(std::size_t stack) const {
    std::optional<std::size_t> best;
    double best_score = -std::numeric_limits<double>::infinity();
    std::uint32_t best_valid = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t best_erase_count = std::numeric_limits<std::uint32_t>::max();
    const auto relocation_capacity = gc_relocation_capacity(stack);
    const auto block_begin = stack * planes_per_stack() * config_.blocks_per_plane;
    const auto block_end = (stack + 1) * planes_per_stack() * config_.blocks_per_plane;
    for (std::size_t i = block_begin; i < block_end; ++i) {
        const auto& block = blocks_[i];
        if (block.role == BlockRole::Free || block.role == BlockRole::StaticReadOnly ||
            block.role == BlockRole::RawPhysical || block.erase_pending ||
            block.pending_program_pages != 0 ||
            block.pending_mapping_publications != 0 ||
            block.free_pages != 0 || block.invalid_pages == 0) {
            continue;
        }
        const auto plane = block_plane_index(i);
        const auto& plane_state = planes_.at(plane);
        if (plane_state.active_data_block == i || plane_state.active_mapping_block == i ||
            plane_state.active_gc_block == i) {
            continue;
        }
        if (block.valid_pages > relocation_capacity) {
            continue;
        }
        const auto used_pages = block.valid_pages + block.invalid_pages;
        const double invalid_ratio = used_pages == 0 ? 0.0 :
            static_cast<double>(block.invalid_pages) / static_cast<double>(used_pages);
        const double relocation_cost = static_cast<double>(block.valid_pages);
        const double wear_penalty =
            static_cast<double>(block.erase_count) * config_.gc_wear_leveling_weight;
        const double score = invalid_ratio * 1000.0 +
            static_cast<double>(block.invalid_pages) * 10.0 -
            relocation_cost - wear_penalty;
        if (score > best_score ||
            (score == best_score && block.valid_pages < best_valid) ||
            (score == best_score && block.valid_pages == best_valid &&
                block.erase_count < best_erase_count)) {
            best = i;
            best_score = score;
            best_valid = block.valid_pages;
            best_erase_count = block.erase_count;
        }
    }
    return best;
}

void HbfDevice::maybe_run_gc(
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    std::uint64_t required_pages,
    std::size_t stack,
    BlockRole allocation_role,
    std::optional<std::size_t> preferred_plane) {
    if (!config_.auto_gc_enabled) {
        return;
    }
    if (required_pages == 0) {
        return;
    }
    if (required_pages != 1) {
        throw std::runtime_error(
            "HBF GC allocation preview currently requires one page");
    }
    const auto watermark = config_.gc_low_watermark_pages == 0 ?
        config_.pages_per_block : config_.gc_low_watermark_pages;
    const auto hard_watermark = config_.gc_hard_watermark_pages == 0 ?
        required_pages : config_.gc_hard_watermark_pages;

    // GC and foreground allocators have different reachable capacity. A GC
    // relocation may consume reserved free blocks and append to active GC
    // blocks; Data/Mapping may append only to their own active blocks or open
    // whole blocks beyond the per-plane reserve. The soft threshold therefore
    // protects relocation capacity after accounting for the whole block(s)
    // this foreground allocation would have to open, not merely its page count.
    const auto pressure = [this, stack, allocation_role, preferred_plane,
                              required_pages, watermark, hard_watermark]() {
        const auto headroom = gc_headroom(stack, allocation_role);
        const auto target_plane = preview_allocation_plane(
            stack, allocation_role, preferred_plane);
        bool opens_block = true;
        if (target_plane) {
            const auto& plane = planes_.at(*target_plane);
            const auto& active = allocation_role == BlockRole::Data ?
                plane.active_data_block : plane.active_mapping_block;
            opens_block = !active || blocks_.at(*active).free_pages == 0;
        }
        const auto free_pool_debit = opens_block ?
            static_cast<std::uint64_t>(config_.pages_per_block) : 0;
        const auto soft_floor = checked_add(
            headroom.reserve_target_pages,
            watermark,
            "HBF GC soft watermark plus block reserve");
        const bool soft_ok = has_more_than_combined_headroom(
            headroom.relocation_pages, free_pool_debit, soft_floor);
        const bool hard_ok = has_more_than_combined_headroom(
            headroom.foreground_pages, required_pages, hard_watermark);
        return std::tuple{headroom, soft_ok, hard_ok};
    };

    auto [initial_headroom, soft_ok, hard_ok] = pressure();
    (void)initial_headroom;
    if (soft_ok && hard_ok) {
        return;
    }
    if (gc_active_by_stack_.at(stack)) {
        // The active relocation owns a capacity preflight. Dirty mapping
        // writeback encountered inside it is deferred; nested victim
        // selection would invalidate that preflight and can choose the same
        // not-yet-erased block twice.
        return;
    }
    gc_active_by_stack_[stack] = true;
    struct GcActiveGuard {
        std::vector<bool>& active;
        std::size_t stack;
        ~GcActiveGuard() { active[stack] = false; }
    } active_guard{gc_active_by_stack_, stack};
    bool hard_block_recorded = false;
    const auto record_hard_block = [&](double begin_ns,
                                       double end_ns,
                                       const std::string& detail) {
        if (!hard_block_recorded) {
            stats_.gc_user_blocked_runs++;
            hard_block_recorded = true;
        }
        add_trace_span(
            spans,
            "gc/foreground_block",
            "maintenance",
            "logic/maintenance",
            begin_ns,
            end_ns,
            true,
            detail);
    };

    std::uint64_t progress_guard = 0;
    for (;;) {
        auto [headroom, current_soft_ok, current_hard_ok] = pressure();
        if (current_soft_ok && current_hard_ok) {
            return;
        }
        const auto victim = choose_gc_victim(stack);
        if (!victim) {
            // Preventive soft GC must not pull this operation to a future
            // commit merely to manufacture an invalid page. Only hard
            // foreground pressure may wait for already-issued state.
            if (!current_hard_ok) {
                const double wait_begin_ns = at_ns;
                if (advance_to_next_commit(at_ns, stack)) {
                    if (at_ns > wait_begin_ns) {
                        breakdown.scheduler_queue_wait_ns += at_ns - wait_begin_ns;
                        record_hard_block(
                            wait_begin_ns,
                            at_ns,
                            "wait for reclaimable committed state");
                    }
                    continue;
                }
            }
            return;
        }
        const auto invalid_pages = blocks_.at(*victim).invalid_pages;
        const auto relocation_before = headroom.relocation_pages;
        const double relocation_begin_ns = at_ns;
        relocate_and_erase_block(*victim, at_ns, breakdown, spans);
        if (!current_hard_ok) {
            record_hard_block(
                relocation_begin_ns,
                at_ns,
                "foreground hard-watermark GC");
        }
        const auto relocation_after =
            gc_headroom(stack, allocation_role).relocation_pages;
        const auto expected_after = checked_add(
            relocation_before,
            invalid_pages,
            "HBF GC relocation progress");
        // The victim itself contributes exactly invalid_pages. A selected
        // erase dependency may materialize while relocation walks metadata
        // and contribute additional pages, so require at least that delta.
        if (invalid_pages == 0 || relocation_after < expected_after) {
            throw std::runtime_error(
                "HBF GC victim did not increase relocation headroom by at least its invalid pages");
        }
        causal_state_ready_by_stack_.at(stack) = std::max(
            causal_state_ready_by_stack_.at(stack), at_ns);
        progress_guard = checked_add(
            progress_guard, invalid_pages, "HBF GC progress guard");
        if (progress_guard > total_pages_) {
            throw std::runtime_error(
                "HBF GC exceeded finite relocation-headroom progress");
        }
    }
}

void HbfDevice::relocate_and_erase_block(
    std::size_t victim_block,
    double& at_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans) {
    const auto& victim_before_relocation = blocks_.at(victim_block);
    if (victim_before_relocation.pending_program_pages != 0 ||
        victim_before_relocation.pending_mapping_publications != 0) {
        throw std::runtime_error(
            "HBF GC victim retained in-flight page ownership");
    }
    const auto victim_stack = stack_of_block(victim_block);
    if (victim_before_relocation.valid_pages >
        gc_relocation_capacity(victim_stack)) {
        throw std::runtime_error(
            "HBF GC lacks relocation capacity before scheduling victim reads");
    }
    const auto victim_valid_pages =
        static_cast<std::uint64_t>(victim_before_relocation.valid_pages);
    const auto victim_invalid_pages =
        static_cast<std::uint64_t>(victim_before_relocation.invalid_pages);
    if (victim_before_relocation.free_pages != 0 ||
        checked_add(
            victim_valid_pages,
            victim_invalid_pages,
            "HBF GC victim page accounting") != config_.pages_per_block) {
        throw std::runtime_error(
            "HBF GC victim is not a fully programmed valid/invalid block");
    }
    breakdown.maintenance_ns += 50.0;
    const double victim_select_done = at_ns + 50.0;
    add_trace_span(
        spans,
        "gc_victim_select",
        "maintenance",
        "logic/maintenance",
        at_ns,
        victim_select_done,
        true,
        "block" + std::to_string(victim_block));
    at_ns = victim_select_done;
    const auto block_begin = static_cast<std::uint64_t>(victim_block) * config_.pages_per_block;
    // Collect live pages from the block's own valid bitmap (ascending page
    // order), never by scanning the global page table.
    struct LiveVictimPage {
        std::uint64_t ppn = 0;
        std::uint64_t logical_key = 0;
        PageOwner owner = PageOwner::Unassigned;
        bool compact = false;
    };
    std::vector<LiveVictimPage> valid_pages;
    const auto& victim = blocks_.at(victim_block);
    for (std::uint32_t page = 0; page < config_.pages_per_block; ++page) {
        if (!victim.is_valid(page)) {
            continue;
        }
        const auto ppn = block_begin + page;
        if (const auto materialized = programmed_pages_.find(ppn);
            materialized != programmed_pages_.end()) {
            if (materialized->second.status != PageStatus::Valid ||
                (materialized->second.owner != PageOwner::Logical &&
                 materialized->second.owner != PageOwner::Mapping)) {
                throw std::runtime_error(
                    "HBF GC found invalid materialized victim ownership");
            }
            valid_pages.push_back(LiveVictimPage{
                .ppn = ppn,
                .logical_key = materialized->second.lpn,
                .owner = materialized->second.owner,
                .compact = false,
            });
        } else if (const auto compact = compact_page_identity(ppn)) {
            valid_pages.push_back(LiveVictimPage{
                .ppn = ppn,
                .logical_key = compact->logical_key,
                .owner = compact->owner,
                .compact = true,
            });
        } else {
            throw std::runtime_error(
                "HBF GC valid bitmap page has no logical owner");
        }
    }

    for (const auto& page : valid_pages) {
        const auto old_ppn = page.ppn;
        const auto lpn = page.logical_key;
        const double read_done = schedule_read_page(
            old_ppn,
            at_ns,
            breakdown,
            spans,
            TransactionSource::GC,
            HeatmapTrafficSource::GarbageCollection,
            ReadPayloadRoute::Internal,
            0);
        double alloc_ready = read_done;
        const auto new_ppn = allocate_free_page(
            alloc_ready,
            breakdown,
            spans,
            BlockRole::GC,
            victim_stack,
            block_plane_index(victim_block));
        const double program_done = schedule_program_page(
            new_ppn,
            alloc_ready,
            breakdown,
            spans,
            TransactionSource::GC,
            HeatmapTrafficSource::GarbageCollection);
        const bool metadata_page = page.owner == PageOwner::Mapping;
        if (metadata_page != is_metadata_lpn(lpn)) {
            throw std::runtime_error(
                "HBF GC logical key disagrees with page ownership");
        }
        const auto detail = metadata_page ?
            ("vpn" + std::to_string(metadata_vpn(lpn))) :
            ("lpn" + std::to_string(lpn));
        double mapping_ready = program_done;
        if (!metadata_page) {
            // Relocating a data page updates the same resident L2P entry as a
            // user write. Its checkpoint page becomes dirty, but no mapping
            // media read or cache admission lies on this critical path.
            state_observation_by_stack_.at(victim_stack) = std::max(
                state_observation_by_stack_.at(victim_stack), mapping_ready);
            access_resident_mapping(
                lpn,
                TransactionSource::GC,
                MappingAccessKind::Update,
                mapping_ready,
                breakdown,
                spans);
            state_observation_by_stack_.at(victim_stack) = std::max(
                state_observation_by_stack_.at(victim_stack), mapping_ready);
        } else {
            const double mapping_done =
                mapping_ready + config_.mapping_update_ns;
            breakdown.translation_ns += config_.mapping_update_ns;
            add_trace_span(
                spans,
                "gc_mapping_checkpoint_relocate",
                "translation",
                "logic/mapping_table",
                mapping_ready,
                mapping_done,
                true,
                detail);
            mapping_ready = mapping_done;
        }
        const double mapping_done = mapping_ready;
        at_ns = std::max(at_ns, mapping_done);
        if (page.compact) {
            retire_compact_page(lpn, page.owner);
        } else {
            invalidate_ppn(old_ppn);
        }
        if (metadata_page) {
            mapping_vpn_to_ppn_[metadata_vpn(lpn)] = new_ppn;
        } else {
            lpn_to_ppn_[lpn] = new_ppn;
            mark_mapping_page_dirty(
                mapping_vpn_for_lpn(lpn), mapping_done);
        }
        mark_programmed(
            new_ppn,
            lpn,
            metadata_page ? PageOwner::Mapping : PageOwner::Logical);
        stats_.physical_read_bytes += config_.page_size_bytes;
        stats_.physical_write_bytes += config_.page_size_bytes;
        stats_.gc_relocation_payload_bytes += config_.page_size_bytes;
        stats_.page_reads++;
        stats_.page_programs++;
        stats_.gc_relocations++;
        if (metadata_page) {
            stats_.gc_mapping_relocations++;
        } else {
            stats_.gc_data_relocations++;
        }
    }

    at_ns = schedule_erase_block(
        victim_block,
        at_ns,
        breakdown,
        spans,
        TransactionSource::GC,
        HeatmapTrafficSource::GarbageCollection);
    reset_erased_block(victim_block);
    stats_.block_erases++;
    stats_.gc_runs++;
    stats_.gc_reclaimed_invalid_pages = checked_add(
        stats_.gc_reclaimed_invalid_pages,
        victim_invalid_pages,
        "HBF GC reclaimed invalid-page accounting");
}

void HbfDevice::invalidate_ppn(std::uint64_t ppn) {
    auto found = programmed_pages_.find(ppn);
    if (found == programmed_pages_.end() || found->second.status != PageStatus::Valid) {
        return;
    }
    found->second.status = PageStatus::Invalid;
    auto& block = blocks_.at(ppn / config_.pages_per_block);
    if (block.valid_pages > 0) {
        block.valid_pages--;
    }
    block.clear_valid(static_cast<std::uint32_t>(ppn % config_.pages_per_block));
    block.invalid_pages++;
    stats_.invalidations++;
}

void HbfDevice::mark_programmed(
    std::uint64_t ppn,
    std::uint64_t lpn,
    PageOwner owner) {
    if (ppn >= total_pages_) {
        throw std::runtime_error("HBF program target PPN is out of range");
    }
    const auto block_index = static_cast<std::size_t>(ppn / config_.pages_per_block);
    const auto page_index = static_cast<std::uint32_t>(ppn % config_.pages_per_block);
    auto& block = blocks_.at(block_index);
    auto page_state = programmed_pages_.find(ppn);
    if (page_state == programmed_pages_.end()) {
        throw std::runtime_error(
            "HBF program completion has no allocator/raw reservation");
    }
    if (page_state->second.block_epoch != block.epoch || block.erase_pending) {
        throw std::runtime_error("HBF program completion targets a retired block epoch");
    }
    const bool role_matches =
        (owner == PageOwner::Logical &&
            (block.role == BlockRole::Data || block.role == BlockRole::GC)) ||
        (owner == PageOwner::Mapping &&
            (block.role == BlockRole::Mapping || block.role == BlockRole::GC)) ||
        (owner == PageOwner::RawPhysical && block.role == BlockRole::RawPhysical);
    if (!role_matches) {
        throw std::runtime_error("HBF program owner does not match block role");
    }
    if (page_state->second.status != PageStatus::Erased) {
        throw std::runtime_error("HBF attempted to program a non-erased physical page");
    }
    if (block.pending_program_pages == 0) {
        throw std::runtime_error(
            "HBF program completion lost its block ownership pin");
    }
    // The block epoch changes on erase, but not when an erased page becomes
    // programmed. Retire any decoded erased-value cache line exactly when the
    // media program completes so a later physical read cannot observe the old
    // contents under the still-current block epoch.
    read_buffer_purge_page(ppn);
    page_state->second.status = PageStatus::Valid;
    page_state->second.owner = owner;
    page_state->second.lpn = lpn;
    block.pending_program_pages--;
    block.valid_pages++;
    block.set_valid(page_index);
}

void HbfDevice::reset_erased_block(std::size_t block_index) {
    const auto& retiring_block = blocks_.at(block_index);
    if (!retiring_block.erase_pending &&
        (retiring_block.pending_program_pages != 0 ||
         retiring_block.pending_mapping_publications != 0)) {
        throw std::runtime_error(
            "HBF GC attempted to erase a block with in-flight ownership");
    }
    read_buffer_purge_block(block_index);
    const auto block_begin = static_cast<std::uint64_t>(block_index) * config_.pages_per_block;
    for (std::uint32_t page = 0; page < config_.pages_per_block; ++page) {
        programmed_pages_.erase(block_begin + page);
    }

    auto& block = blocks_.at(block_index);
    const bool was_free = block.role == BlockRole::Free;
    const bool was_pending_free_erase = was_free && block.erase_pending;
    const auto next_epoch = block.erase_pending ? block.epoch : block.epoch + 1;
    const auto newly_free_pages = config_.pages_per_block - block.free_pages;
    free_pages_ += newly_free_pages;
    free_pages_per_stack_.at(stack_of_block(block_index)) += newly_free_pages;
    const auto erase_count = block.erase_count + 1;
    block = BlockState{};
    block.role = BlockRole::Free;
    block.free_pages = config_.pages_per_block;
    block.erase_count = erase_count;
    block.epoch = next_epoch;

    auto& plane = planes_.at(block_plane_index(block_index));
    if (plane.active_data_block == block_index) {
        plane.active_data_block = std::nullopt;
    }
    if (plane.active_mapping_block == block_index) {
        plane.active_mapping_block = std::nullopt;
    }
    if (plane.active_gc_block == block_index) {
        plane.active_gc_block = std::nullopt;
    }
    if (!was_free || was_pending_free_erase) {
        plane.free_blocks.push_back(block_index);
    }
}

HbfDevice::ScheduledTransfer HbfDevice::reserve(
    double earliest_ns,
    double duration_ns,
    ResourceTimeline& timeline) {
    // Validate the complete reservation before pruning mutates the calendar.
    // Legal internal work is never ready before its top-level issue/drain
    // arrival; violating that invariant must fail without consuming history.
    if (!std::isfinite(reservation_causal_watermark_ns_) ||
        reservation_causal_watermark_ns_ < 0.0 ||
        !std::isfinite(earliest_ns) ||
        earliest_ns < reservation_causal_watermark_ns_ ||
        !std::isfinite(duration_ns) || duration_ns <= 0.0) {
        throw std::runtime_error("HBF resource calendar received an invalid reservation");
    }
    const double updated_reserved_work_ns =
        timeline.reserved_work_ns + duration_ns;
    if (!std::isfinite(updated_reserved_work_ns)) {
        throw std::runtime_error(
            "HBF resource reservation work exceeds finite double range");
    }
    timeline.prune_before(reservation_causal_watermark_ns_);
    // Use the earliest exact idle interval that fits. No still-usable gap is
    // evicted: losing one changes the simulated schedule based on call order
    // and creates phantom queueing even when the physical resource is idle.
    if (const auto gap = timeline.first_fitting_gap(earliest_ns, duration_ns)) {
        const double start = std::max(gap->begin_ns, earliest_ns);
        const double finish = start + duration_ns;
        timeline.consume_gap(*gap, start, finish);
        timeline.reserved_work_ns = updated_reserved_work_ns;
        return ScheduledTransfer{
            .start_ns = start,
            .finish_ns = finish,
            .wait_ns = std::max(0.0, start - earliest_ns),
        };
    }
    const double start = std::max(earliest_ns, timeline.ready_ns);
    const double finish = start + duration_ns;
    if (!std::isfinite(start) || !std::isfinite(finish)) {
        throw std::runtime_error("HBF resource calendar reservation time overflowed");
    }
    if (start > timeline.ready_ns) {
        timeline.insert_gap(timeline.ready_ns, start);
    }
    timeline.ready_ns = finish;
    timeline.reserved_work_ns = updated_reserved_work_ns;
    return ScheduledTransfer{
        .start_ns = start,
        .finish_ns = finish,
        .wait_ns = std::max(0.0, start - earliest_ns),
    };
}

HbfDevice::ScheduledTransfer HbfDevice::schedule_flash_transaction(
    const HbfAddress& addr,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    TransactionKind kind) {
    auto& die = dies_.at(die_index(addr));
    const auto source_label = source_name(source);
    const auto kind_label = kind_name(kind);
    const auto source_slot = source_index(source);
    stats_.flash_scheduler_enqueues++;

    auto slot = reserve(
        earliest_ns, config_.flash_tsu_issue_ns, die.source_queues.at(source_slot));
    breakdown.scheduler_queue_wait_ns += slot.wait_ns;
    trace_wait(
        spans,
        die_entity(addr),
        earliest_ns,
        slot.start_ns,
        "wait_flash_scheduler_" + source_label);
    const double source_ready = slot.start_ns;

    auto issue = reserve(source_ready, config_.flash_tsu_issue_ns, die.sequencer);
    breakdown.scheduler_queue_wait_ns += issue.wait_ns;
    breakdown.command_ns += config_.flash_tsu_issue_ns;
    trace_wait(spans, die_entity(addr), source_ready, issue.start_ns, "wait_flash_sequencer");
    add_trace_span(
        spans,
        source_label + "/flash_scheduler_issue_" + kind_label,
        "sequencer",
        die_entity(addr),
        issue.start_ns,
        issue.finish_ns,
        true,
        source_label + " " + kind_label);

    die.sequencer_busy_ns += config_.flash_tsu_issue_ns;
    die.transaction_count++;
    stats_.flash_scheduler_issues++;
    return issue;
}

double HbfDevice::schedule_command_path(
    const HbfAddress& addr,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source) {
    auto& logic_die = logic_dies_.at(stack_index(addr));
    auto& channel = channels_.at(channel_index(addr));
    const auto source_label = source_name(source);
    const double tsv_ns = transfer_time_ns(config_.command_address_bytes, config_.tsv_bandwidth_GBps);
    const double channel_ns = transfer_time_ns(config_.command_address_bytes, config_.channel_bandwidth_GBps);

    breakdown.tsv_transfer_ns += tsv_ns;
    auto tsv = reserve(earliest_ns, tsv_ns, logic_die.tsv);
    breakdown.scheduler_queue_wait_ns += tsv.wait_ns;
    trace_wait(
        spans,
        "stack" + std::to_string(addr.stack) + "/tsv",
        earliest_ns,
        tsv.start_ns,
        "wait_tsv_cmd");
    add_trace_span(
        spans,
        source_label + "/cmd_addr_tsv",
        "tsv",
        "stack" + std::to_string(addr.stack) + "/tsv",
        tsv.start_ns,
        tsv.finish_ns,
        true,
        std::to_string(config_.command_address_bytes) + "B " + source_label);

    auto command = reserve(tsv.finish_ns, channel_ns, channel.command);
    breakdown.scheduler_queue_wait_ns += command.wait_ns;
    breakdown.channel_transfer_ns += channel_ns;
    channel.command_busy_ns += channel_ns;
    channel.command_count++;
    trace_wait(spans, channel_entity(addr), tsv.finish_ns, command.start_ns, "wait_channel_cmd");
    add_trace_span(
        spans,
        source_label + "/cmd_addr_channel",
        "flash_channel",
        channel_entity(addr),
        command.start_ns,
        command.finish_ns,
        true,
        std::to_string(config_.command_address_bytes) + "B " + source_label);
    return command.finish_ns;
}

double HbfDevice::schedule_ecc(
    const HbfAddress& addr,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    bool decode) {
    auto& die = dies_.at(die_index(addr));
    const auto source_label = source_name(source);
    const auto operation = decode ? std::string{"decode"} : std::string{"encode"};
    const auto issue_entity = die_entity(addr) + "/ecc_issue_port";
    const auto pipeline_entity =
        die_entity(addr) + "/ecc_" + operation + "_pipeline";
    const double latency_ns = decode ?
        config_.ecc_decode_latency_ns : config_.ecc_encode_latency_ns;
    const double raw_bandwidth_GBps = decode ?
        config_.ecc_decode_raw_bandwidth_GBps_per_die :
        config_.ecc_encode_raw_bandwidth_GBps_per_die;
    const double initiation_ns = transfer_time_ns(page_wire_bytes(), raw_bandwidth_GBps);

    // A codeword consumes only an initiation slot. Its response latency may
    // overlap later codewords in the same per-die pipeline. Decode and encode
    // conservatively share this issue port, so mixed traffic still contends.
    const auto issue = reserve(earliest_ns, initiation_ns, die.ecc_issue);
    const double finish_ns = issue.start_ns + latency_ns;
    if (!std::isfinite(finish_ns)) {
        throw std::runtime_error("HBF ECC completion time overflowed");
    }

    breakdown.ecc_queue_wait_ns += issue.wait_ns;
    breakdown.ecc_latency_ns += latency_ns;
    stats_.ecc_issue_busy_ns += initiation_ns;
    stats_.ecc_codeword_bytes = checked_add(
        stats_.ecc_codeword_bytes, page_wire_bytes(), "HBF ECC codeword bytes");
    die.ecc_issue_busy_ns += initiation_ns;
    die.ecc_inflight_intervals.push_back(DieState::EccInflightInterval{
        .start_ns = issue.start_ns,
        .finish_ns = finish_ns,
    });
    if (decode) {
        stats_.ecc_decode_ops++;
        stats_.ecc_decode_queue_wait_ns += issue.wait_ns;
        stats_.ecc_decode_latency_work_ns += latency_ns;
        stats_.ecc_decode_issue_busy_ns += initiation_ns;
        stats_.ecc_decode_codeword_bytes = checked_add(
            stats_.ecc_decode_codeword_bytes,
            page_wire_bytes(),
            "HBF ECC decode codeword bytes");
        die.ecc_decode_ops++;
    } else {
        stats_.ecc_encode_ops++;
        stats_.ecc_encode_queue_wait_ns += issue.wait_ns;
        stats_.ecc_encode_latency_work_ns += latency_ns;
        stats_.ecc_encode_issue_busy_ns += initiation_ns;
        stats_.ecc_encode_codeword_bytes = checked_add(
            stats_.ecc_encode_codeword_bytes,
            page_wire_bytes(),
            "HBF ECC encode codeword bytes");
        die.ecc_encode_ops++;
    }

    trace_wait(
        spans,
        issue_entity,
        earliest_ns,
        issue.start_ns,
        "wait_ecc_" + operation + "_issue");
    const auto detail = std::to_string(page_wire_bytes()) +
        "B raw, II=" + fixed(initiation_ns, 6) + "ns";
    add_trace_span(
        spans,
        source_label + "/ecc_" + operation + "_issue",
        "ecc_issue",
        issue_entity,
        issue.start_ns,
        issue.finish_ns,
        false,
        detail);
    add_trace_span(
        spans,
        source_label + "/ecc_" + operation + "_latency",
        "ecc_latency",
        pipeline_entity,
        issue.start_ns,
        finish_ns,
        true,
        detail);
    return finish_ns;
}

double HbfDevice::schedule_read_page(
    std::uint64_t ppn,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    HeatmapTrafficSource heatmap_attribution,
    ReadPayloadRoute route,
    std::uint64_t external_payload_bytes,
    double* decoded_ready_ns) {
    const auto addr = decode_ppn(ppn);
    auto& logic_die = logic_dies_.at(stack_index(addr));
    auto& channel = channels_.at(channel_index(addr));
    auto& plane = planes_.at(plane_index(addr));
    const auto source_label = source_name(source);
    const auto subarray_index_value = subarray_index(addr);
    const auto lane_index = media_lane_index(addr);
    const auto buffer_bank_index = page_buffer_bank_index(addr);
    auto& subarray = plane.subarrays.at(subarray_index_value);
    auto& lane = plane.media_lanes.at(lane_index);
    auto& buffer_bank = plane.page_buffer_banks.at(buffer_bank_index);
    const double command_done = schedule_command_path(
        addr, earliest_ns, breakdown, spans, source);
    auto tsu = schedule_flash_transaction(
        addr,
        command_done,
        breakdown,
        spans,
        source,
        TransactionKind::Read);

    const double lane_ns = transfer_time_ns(
        page_wire_bytes(), config_.media_lane_bandwidth_GBps);
    const double page_buffer_ns = transfer_time_ns(
        page_wire_bytes(), config_.page_buffer_bandwidth_GBps);
    ScheduledTransfer sense;
    ScheduledTransfer lane_transfer;
    ScheduledTransfer page_buffer;
    double local_ready_ns = tsu.finish_ns;
    for (;;) {
        subarray.timeline.prune_before(reservation_causal_watermark_ns_);
        lane.timeline.prune_before(reservation_causal_watermark_ns_);
        buffer_bank.timeline.prune_before(reservation_causal_watermark_ns_);

        SenseRoundCandidate batch_candidate;
        double sense_start_ns = 0.0;
        if (config_.batch_activation) {
            batch_candidate = preview_sense_round(
                plane, subarray_index_value, local_ready_ns);
            sense_start_ns = batch_candidate.start_ns;
        } else {
            sense_start_ns = subarray.timeline.preview_start(
                local_ready_ns, config_.t_read_page_ns);
        }
        const double sense_finish_ns = sense_start_ns + config_.t_read_page_ns;
        const double lane_start_ns = lane.timeline.preview_start(
            sense_finish_ns, lane_ns);
        const double lane_finish_ns = lane_start_ns + lane_ns;
        const double buffer_start_ns = buffer_bank.timeline.preview_start(
            lane_finish_ns, page_buffer_ns);
        const double buffer_finish_ns = buffer_start_ns + page_buffer_ns;

        const auto conflict = std::lower_bound(
            plane.full_plane_windows.begin(),
            plane.full_plane_windows.end(),
            sense_start_ns,
            [](const PlaneState::BusyWindow& window, double start) {
                return window.end_ns <= start;
            });
        if (conflict != plane.full_plane_windows.end() &&
            conflict->begin_ns < buffer_finish_ns) {
            local_ready_ns = std::max(local_ready_ns, conflict->end_ns);
            continue;
        }

        if (config_.batch_activation) {
            sense = commit_sense_round(
                plane,
                subarray_index_value,
                local_ready_ns,
                batch_candidate);
        } else {
            sense = reserve(
                sense_start_ns, config_.t_read_page_ns, subarray.timeline);
            if (sense.start_ns != sense_start_ns) {
                throw std::runtime_error(
                    "HBF independent sense moved after path preview");
            }
            sense.wait_ns = std::max(0.0, sense.start_ns - tsu.finish_ns);
        }
        lane_transfer = reserve(lane_start_ns, lane_ns, lane.timeline);
        page_buffer = reserve(
            buffer_start_ns, page_buffer_ns, buffer_bank.timeline);
        if (lane_transfer.start_ns != lane_start_ns ||
            page_buffer.start_ns != buffer_start_ns) {
            throw std::runtime_error(
                "HBF read local data path moved after atomic preview");
        }
        // Exact commits use the already-previewed start as their reservation
        // key, so reserve() correctly reports zero wait. Preserve the actual
        // pipeline queueing relative to the producer instead: otherwise the
        // trace shows lane/page-buffer stalls that disappear from Breakdown.
        lane_transfer.wait_ns = std::max(
            0.0, lane_transfer.start_ns - sense.finish_ns);
        page_buffer.wait_ns = std::max(
            0.0, page_buffer.start_ns - lane_transfer.finish_ns);
        break;
    }
    const double media_start = sense.start_ns;
    breakdown.scheduler_queue_wait_ns += std::max(0.0, media_start - tsu.finish_ns);
    trace_wait(spans, subarray_entity(addr, subarray_index_value), tsu.finish_ns, media_start, "wait_subarray");
    const double sense_done = media_start + config_.t_read_page_ns;
    breakdown.array_read_ns += config_.t_read_page_ns;
    add_trace_span(
        spans,
        source_label + "/array_read",
        "flash_array",
        subarray_entity(addr, subarray_index_value),
        media_start,
        sense_done,
        true,
        "plane" + std::to_string(addr.plane) +
            " subarray" + std::to_string(subarray_index_value) +
            " lane" + std::to_string(lane_index));
    subarray.busy_ns += config_.t_read_page_ns;
    subarray.read_count++;

    breakdown.scheduler_queue_wait_ns += lane_transfer.wait_ns;
    breakdown.media_lane_transfer_ns += lane_ns;
    trace_wait(
        spans,
        media_lane_entity(addr, lane_index),
        sense_done,
        lane_transfer.start_ns,
        "wait_media_lane");
    add_trace_span(
        spans,
        source_label + "/array_to_page_buffer",
        "media_lane",
        media_lane_entity(addr, lane_index),
        lane_transfer.start_ns,
        lane_transfer.finish_ns,
        true,
        std::to_string(page_wire_bytes()) + "B raw from subarray" +
            std::to_string(subarray_index_value));
    lane.busy_ns += lane_ns;
    lane.read_count++;
    record_plane_media_busy(plane, media_start, sense_done);
    plane.read_count++;

    breakdown.scheduler_queue_wait_ns += page_buffer.wait_ns;
    breakdown.page_buffer_ns += page_buffer_ns;
    buffer_bank.busy_ns += page_buffer_ns;
    buffer_bank.read_count++;
    trace_wait(
        spans,
        page_buffer_bank_entity(addr, buffer_bank_index),
        lane_transfer.finish_ns,
        page_buffer.start_ns,
        "wait_page_buffer_bank");
    add_trace_span(
        spans,
        source_label + "/page_buffer_out",
        "page_buffer",
        page_buffer_bank_entity(addr, buffer_bank_index),
        page_buffer.start_ns,
        page_buffer.finish_ns,
        true,
        std::to_string(page_wire_bytes()) + "B raw bank" +
            std::to_string(buffer_bank_index));
    auto& block = blocks_.at(static_cast<std::size_t>(
        ppn / config_.pages_per_block));
    block.issued_media_ready_ns = std::max(
        block.issued_media_ready_ns, page_buffer.finish_ns);

    const double channel_ns = transfer_time_ns(page_wire_bytes(), config_.channel_bandwidth_GBps);
    auto channel_transfer = reserve(page_buffer.finish_ns, channel_ns, channel.data);
    breakdown.scheduler_queue_wait_ns += channel_transfer.wait_ns;
    breakdown.channel_transfer_ns += channel_ns;
    channel.data_busy_ns += channel_ns;
    channel.data_count++;
    trace_wait(spans, channel_entity(addr), page_buffer.finish_ns, channel_transfer.start_ns, "wait_channel_data");
    add_trace_span(
        spans,
        source_label + "/data_out_channel",
        "flash_channel",
        channel_entity(addr),
        channel_transfer.start_ns,
        channel_transfer.finish_ns,
        true,
        std::to_string(page_wire_bytes()) + "B raw");

    const double tsv_ns = transfer_time_ns(page_wire_bytes(), config_.tsv_bandwidth_GBps);
    breakdown.tsv_transfer_ns += tsv_ns;
    auto tsv = reserve(channel_transfer.finish_ns, tsv_ns, logic_die.tsv);
    breakdown.scheduler_queue_wait_ns += tsv.wait_ns;
    trace_wait(
        spans,
        "stack" + std::to_string(addr.stack) + "/tsv",
        channel_transfer.finish_ns,
        tsv.start_ns,
        "wait_tsv_data");
    add_trace_span(
        spans,
        source_label + "/data_out_tsv",
        "tsv",
        "stack" + std::to_string(addr.stack) + "/tsv",
        tsv.start_ns,
        tsv.finish_ns,
        true,
        std::to_string(page_wire_bytes()) + "B raw");

    // Raw codeword bytes cross the flash channel and TSV. Decode happens on
    // the logic die before decoded payload enters SRAM and leaves over the
    // external HBIO; parity/OOB bytes never consume external payload BW.
    const double ecc_done = schedule_ecc(
        addr, tsv.finish_ns, breakdown, spans, source, true);
    const double sram_ns = transfer_time_ns(
        config_.page_size_bytes, config_.logic_sram_bandwidth_GBps);
    auto sram = reserve(ecc_done, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    trace_wait(spans, logic_entity(addr.stack), ecc_done, sram.start_ns, "wait_sram");
    add_trace_span(
        spans,
        source_label + "/sram_stage_read",
        "sram",
        logic_entity(addr.stack),
        sram.start_ns,
        sram.finish_ns,
        true,
        std::to_string(config_.page_size_bytes) + "B payload");
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfPhysical,
            .direction = TrafficDirection::Read,
            .source = resolve_heatmap_source(source, heatmap_attribution),
            .address = checked_mul(
                ppn,
                config_.page_size_bytes,
                "HBF heatmap page-read address"),
            .bytes = config_.page_size_bytes,
        });
    }
    if (decoded_ready_ns != nullptr) {
        *decoded_ready_ns = sram.finish_ns;
    }
    if (route == ReadPayloadRoute::Internal) {
        if (external_payload_bytes != 0) {
            throw std::runtime_error(
                "HBF internal read cannot request external payload bytes");
        }
        return sram.finish_ns;
    }
    return schedule_external_read_egress(
        addr.stack,
        external_payload_bytes,
        sram.finish_ns,
        breakdown,
        spans,
        source_label + "/data_out_hbio",
        "decoded page read");
}

double HbfDevice::schedule_program_page(
    std::uint64_t ppn,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    HeatmapTrafficSource heatmap_attribution) {
    const auto addr = decode_ppn(ppn);
    auto& logic_die = logic_dies_.at(stack_index(addr));
    auto& channel = channels_.at(channel_index(addr));
    auto& plane = planes_.at(plane_index(addr));
    const auto source_label = source_name(source);
    const double command_done = schedule_command_path(
        addr, earliest_ns, breakdown, spans, source);

    // The payload is already resident in logic-die SRAM: user writes crossed
    // HBIO at request ingress, while mapping/GC data was produced internally.
    // Programming reads one full decoded page into ECC, then sends the raw
    // page+OOB codeword over TSV and the flash channel.
    const double sram_ns = transfer_time_ns(
        config_.page_size_bytes, config_.logic_sram_bandwidth_GBps);
    auto sram = reserve(command_done, sram_ns, logic_die.sram);
    breakdown.scheduler_queue_wait_ns += sram.wait_ns;
    breakdown.sram_staging_ns += sram_ns;
    trace_wait(
        spans, logic_entity(addr.stack), command_done, sram.start_ns,
        "wait_sram");
    add_trace_span(
        spans,
        source_label + "/sram_stage_write",
        "sram",
        logic_entity(addr.stack),
        sram.start_ns,
        sram.finish_ns,
        true,
        std::to_string(config_.page_size_bytes) + "B payload");

    const double ecc_done = schedule_ecc(
        addr, sram.finish_ns, breakdown, spans, source, false);

    const double tsv_ns = transfer_time_ns(page_wire_bytes(), config_.tsv_bandwidth_GBps);
    breakdown.tsv_transfer_ns += tsv_ns;
    auto tsv = reserve(ecc_done, tsv_ns, logic_die.tsv);
    breakdown.scheduler_queue_wait_ns += tsv.wait_ns;
    trace_wait(
        spans,
        "stack" + std::to_string(addr.stack) + "/tsv",
        ecc_done,
        tsv.start_ns,
        "wait_tsv_data");
    add_trace_span(
        spans,
        source_label + "/data_in_tsv",
        "tsv",
        "stack" + std::to_string(addr.stack) + "/tsv",
        tsv.start_ns,
        tsv.finish_ns,
        true,
        std::to_string(page_wire_bytes()) + "B raw");

    const double channel_ns = transfer_time_ns(page_wire_bytes(), config_.channel_bandwidth_GBps);
    auto channel_transfer = reserve(tsv.finish_ns, channel_ns, channel.data);
    breakdown.scheduler_queue_wait_ns += channel_transfer.wait_ns;
    breakdown.channel_transfer_ns += channel_ns;
    channel.data_busy_ns += channel_ns;
    channel.data_count++;
    trace_wait(spans, channel_entity(addr), tsv.finish_ns, channel_transfer.start_ns, "wait_channel_data");
    add_trace_span(
        spans,
        source_label + "/data_in_channel",
        "flash_channel",
        channel_entity(addr),
        channel_transfer.start_ns,
        channel_transfer.finish_ns,
        true,
        std::to_string(page_wire_bytes()) + "B raw");

    auto tsu = schedule_flash_transaction(
        addr,
        channel_transfer.finish_ns,
        breakdown,
        spans,
        source,
        TransactionKind::Program);

    const double barrier_ns =
        config_.t_program_page_ns + config_.t_program_verify_ns;
    const double media_start = preview_full_plane_window(
        plane, tsu.finish_ns, barrier_ns);
    breakdown.scheduler_queue_wait_ns += std::max(0.0, media_start - tsu.finish_ns);
    trace_wait(spans, plane_entity(addr), tsu.finish_ns, media_start, "wait_plane_array");
    const double program_done = media_start + config_.t_program_page_ns;
    const double verify_done = program_done + config_.t_program_verify_ns;
    breakdown.array_program_ns += config_.t_program_page_ns;
    breakdown.program_verify_ns += config_.t_program_verify_ns;
    add_trace_span(spans, source_label + "/array_program", "flash_array", plane_entity(addr), media_start, program_done);
    add_trace_span(spans, source_label + "/program_verify", "flash_array", plane_entity(addr), program_done, verify_done);
    const auto round_barrier = reserve(
        media_start, barrier_ns, plane.sense_round_calendar);
    if (round_barrier.start_ns != media_start) {
        throw std::runtime_error(
            "HBF program barrier diverged from batch-round calendar");
    }
    for (auto& subarray : plane.subarrays) {
        const auto barrier = reserve(media_start, barrier_ns, subarray.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF program barrier diverged across subarray calendars");
        }
    }
    // Program is a non-preemptible full-plane operation. The previous
    // suspend path could move this frontier after returning a completion,
    // violating causality, so it has been removed until a preemptible event
    // primitive with revisable completion dependencies exists.
    for (auto& lane : plane.media_lanes) {
        const auto barrier = reserve(media_start, barrier_ns, lane.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF program barrier diverged across media-lane calendars");
        }
    }
    for (auto& bank : plane.page_buffer_banks) {
        const auto barrier = reserve(media_start, barrier_ns, bank.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF program barrier diverged across page-buffer calendars");
        }
    }
    record_full_plane_window(plane, media_start, verify_done);
    record_plane_media_busy(plane, media_start, verify_done);
    plane.program_count++;
    auto& block = blocks_.at(static_cast<std::size_t>(
        ppn / config_.pages_per_block));
    block.issued_media_ready_ns = std::max(
        block.issued_media_ready_ns, verify_done);
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfPhysical,
            .direction = TrafficDirection::Write,
            .source = resolve_heatmap_source(source, heatmap_attribution),
            .address = checked_mul(
                ppn,
                config_.page_size_bytes,
                "HBF heatmap page-program address"),
            .bytes = config_.page_size_bytes,
        });
    }
    return verify_done;
}

double HbfDevice::schedule_erase_block(
    std::size_t block,
    double earliest_ns,
    Breakdown& breakdown,
    std::vector<TraceSpan>* spans,
    TransactionSource source,
    HeatmapTrafficSource heatmap_attribution) {
    const auto ppn = static_cast<std::uint64_t>(block) * config_.pages_per_block;
    const auto addr = decode_ppn(ppn);
    auto& plane = planes_.at(plane_index(addr));
    const auto source_label = source_name(source);

    if (blocks_.at(block).issued_media_ready_ns > earliest_ns) {
        trace_wait(
            spans,
            addr.path(),
            earliest_ns,
            blocks_.at(block).issued_media_ready_ns,
            "wait_target_block_access");
        breakdown.scheduler_queue_wait_ns +=
            blocks_.at(block).issued_media_ready_ns - earliest_ns;
        earliest_ns = blocks_.at(block).issued_media_ready_ns;
    }

    const double command_done = schedule_command_path(
        addr, earliest_ns, breakdown, spans, source);
    auto tsu = schedule_flash_transaction(
        addr,
        command_done,
        breakdown,
        spans,
        source,
        TransactionKind::Erase);

    const double barrier_ns = config_.t_erase_block_ns;
    const double media_start = preview_full_plane_window(
        plane, tsu.finish_ns, barrier_ns);
    breakdown.scheduler_queue_wait_ns += std::max(0.0, media_start - tsu.finish_ns);
    trace_wait(spans, plane_entity(addr), tsu.finish_ns, media_start, "wait_plane_array");
    const double erase_done = media_start + config_.t_erase_block_ns;
    breakdown.array_erase_ns += config_.t_erase_block_ns;
    add_trace_span(
        spans,
        source_label + "/block_erase",
        "flash_array",
        plane_entity(addr),
        media_start,
        erase_done,
        true,
        "block" + std::to_string(addr.block));
    const auto round_barrier = reserve(
        media_start, barrier_ns, plane.sense_round_calendar);
    if (round_barrier.start_ns != media_start) {
        throw std::runtime_error(
            "HBF erase barrier diverged from batch-round calendar");
    }
    for (auto& subarray : plane.subarrays) {
        const auto barrier = reserve(media_start, barrier_ns, subarray.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF erase barrier diverged across subarray calendars");
        }
    }
    for (auto& lane : plane.media_lanes) {
        const auto barrier = reserve(media_start, barrier_ns, lane.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF erase barrier diverged across media-lane calendars");
        }
    }
    for (auto& bank : plane.page_buffer_banks) {
        const auto barrier = reserve(media_start, barrier_ns, bank.timeline);
        if (barrier.start_ns != media_start) {
            throw std::runtime_error(
                "HBF erase barrier diverged across page-buffer calendars");
        }
    }
    record_full_plane_window(plane, media_start, erase_done);
    record_plane_media_busy(plane, media_start, erase_done);
    plane.erase_count++;
    if (address_heatmap_ != nullptr) {
        address_heatmap_->record(AddressTrafficRecord{
            .domain = AddressDomain::HbfPhysical,
            .direction = TrafficDirection::Erase,
            .source = resolve_heatmap_source(source, heatmap_attribution),
            .address = checked_mul(
                ppn,
                config_.page_size_bytes,
                "HBF heatmap block-erase address"),
            .bytes = checked_mul(
                config_.pages_per_block,
                config_.page_size_bytes,
                "HBF heatmap block-erase bytes"),
        });
    }
    return erase_done;
}

void HbfDevice::refresh_accounting_stats() const {
    std::uint64_t free_pages = 0;
    std::uint64_t valid_pages = 0;
    std::uint64_t invalid_pages = 0;
    std::uint64_t pending_program_pages = 0;
    std::uint64_t pending_mapping_publications = 0;
    std::uint64_t static_unmaterialized_pages = 0;
    std::vector<std::uint64_t> free_pages_per_stack(config_.stacks, 0);
    std::vector<std::uint8_t> free_pool_membership(blocks_.size(), 0);

    double mapping_dram_issue_busy_ns = 0.0;
    for (const auto& logic_die : logic_dies_) {
        mapping_dram_issue_busy_ns +=
            logic_die.mapping_dram_issue.reserved_work_ns;
    }
    stats_.mapping_dram_issue_busy_ns = mapping_dram_issue_busy_ns;
    const auto resident_mapping_bytes = checked_mul(
        resident_mapping_bytes_per_stack_,
        config_.stacks,
        "HBF resident mapping accounting");
    if (stats_.resident_mapping_table_bytes != resident_mapping_bytes ||
        stats_.resident_mapping_table_bytes_per_stack !=
            resident_mapping_bytes_per_stack_ ||
        stats_.resident_mapping_pages_per_stack !=
            resident_mapping_pages_per_stack_ ||
        stats_.mapping_dram_resources != config_.stacks ||
        config_.ctrl_dram_bytes / config_.stacks <
            resident_mapping_bytes_per_stack_) {
        throw std::runtime_error(
            "HBF resident mapping capacity accounting diverged");
    }
    if (stats_.mapping_lookup_ops != checked_add(
            stats_.mapping_user_lookup_ops,
            stats_.mapping_gc_lookup_ops,
            "HBF resident mapping lookup-source accounting") ||
        stats_.mapping_update_ops != checked_add(
            stats_.mapping_user_update_ops,
            stats_.mapping_gc_update_ops,
            "HBF resident mapping update-source accounting")) {
        throw std::runtime_error(
            "HBF resident mapping user/GC source accounting diverged");
    }
    const auto valid_mapping_wait = [](std::uint64_t operations,
                                       double total_ns,
                                       double max_ns) {
        if (!std::isfinite(total_ns) || !std::isfinite(max_ns) ||
            total_ns < 0.0 || max_ns < 0.0 || max_ns > total_ns) {
            return false;
        }
        return operations == 0 ?
            total_ns == 0.0 && max_ns == 0.0 :
            total_ns > 0.0 && max_ns > 0.0;
    };
    const double expected_mapping_issue_busy_ns =
        static_cast<double>(checked_add(
            stats_.mapping_lookup_ops,
            stats_.mapping_update_ops,
            "HBF resident mapping access accounting")) *
        config_.ctrl_dram_issue_ns;
    const double mapping_busy_tolerance = std::max(
        1e-9,
        std::abs(expected_mapping_issue_busy_ns) * 1e-12);
    if (!valid_mapping_wait(
            stats_.mapping_dram_wait_ops,
            stats_.mapping_dram_wait_ns,
            stats_.mapping_dram_wait_max_ns) ||
        !std::isfinite(stats_.mapping_dram_issue_busy_ns) ||
        std::abs(
            stats_.mapping_dram_issue_busy_ns -
            expected_mapping_issue_busy_ns) > mapping_busy_tolerance) {
        throw std::runtime_error(
            "HBF resident mapping DRAM telemetry diverged");
    }

    for (std::size_t plane_index = 0; plane_index < planes_.size(); ++plane_index) {
        const auto& plane = planes_.at(plane_index);
        for (const auto block_index : plane.free_blocks) {
            if (block_index >= blocks_.size() ||
                block_plane_index(block_index) != plane_index) {
                throw std::runtime_error(
                    "HBF free-block pool contains an out-of-plane block");
            }
            if (free_pool_membership.at(block_index) != 0) {
                throw std::runtime_error(
                    "HBF free-block pool contains a duplicate block");
            }
            const auto& block = blocks_.at(block_index);
            if (block.role != BlockRole::Free || block.erase_pending) {
                throw std::runtime_error(
                    "HBF free-block pool contains an owned or pending-erase block");
            }
            free_pool_membership[block_index] = 1;
        }
    }

    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        const auto& block = blocks_.at(block_index);
        const auto block_valid = static_cast<std::uint64_t>(block.valid_pages);
        const auto block_invalid = static_cast<std::uint64_t>(block.invalid_pages);
        const auto block_pending =
            static_cast<std::uint64_t>(block.pending_program_pages);
        const auto block_free = static_cast<std::uint64_t>(block.free_pages);
        if (block_valid > config_.pages_per_block ||
            block_invalid > config_.pages_per_block ||
            block_pending > config_.pages_per_block ||
            block_free > config_.pages_per_block) {
            throw std::runtime_error(
                "HBF block page-state counter exceeds block geometry");
        }

        std::uint64_t bitmap_valid = 0;
        if (block.valid_bitmap) {
            for (const auto word : *block.valid_bitmap) {
                bitmap_valid = checked_add(
                    bitmap_valid,
                    static_cast<std::uint64_t>(std::popcount(word)),
                    "HBF valid-page bitmap accounting");
            }
        }
        if (bitmap_valid != block_valid) {
            throw std::runtime_error(
                "HBF valid-page bitmap diverged from the block counter");
        }

        const bool allocatable_free =
            block.role == BlockRole::Free && !block.erase_pending;
        if (static_cast<bool>(free_pool_membership.at(block_index)) !=
            allocatable_free) {
            throw std::runtime_error(
                "HBF block role/pending-erase state diverged from its free pool");
        }

        if (block.role == BlockRole::StaticReadOnly) {
            if (block_invalid != 0 || block_pending != 0 || block_free != 0 ||
                block.next_page != config_.pages_per_block || block.erase_pending) {
                throw std::runtime_error(
                    "HBF static block has mutable or free page state");
            }
            static_unmaterialized_pages = checked_add(
                static_unmaterialized_pages,
                config_.pages_per_block - block_valid,
                "HBF static unmaterialized-page accounting");
        } else {
            const auto used = checked_add(
                checked_add(
                    block_valid,
                    block_invalid,
                    "HBF block valid/invalid accounting"),
                block_pending,
                "HBF block programmed/pending accounting");
            if (checked_add(
                    used, block_free, "HBF block total page accounting") !=
                    config_.pages_per_block ||
                block.next_page != used) {
                throw std::runtime_error(
                    "HBF block valid/invalid/pending/free pages do not conserve capacity");
            }
        }

        free_pages = checked_add(
            free_pages, block_free, "HBF global free-page accounting");
        valid_pages = checked_add(
            valid_pages, block_valid, "HBF global valid-page accounting");
        invalid_pages = checked_add(
            invalid_pages, block_invalid, "HBF global invalid-page accounting");
        pending_program_pages = checked_add(
            pending_program_pages,
            block_pending,
            "HBF global pending-program accounting");
        pending_mapping_publications = checked_add(
            pending_mapping_publications,
            block.pending_mapping_publications,
            "HBF global pending-mapping-publication accounting");
        auto& stack_free = free_pages_per_stack.at(stack_of_block(block_index));
        stack_free = checked_add(
            stack_free, block_free, "HBF per-stack free-page accounting");
    }

    std::uint64_t page_map_valid = 0;
    std::uint64_t page_map_invalid = 0;
    std::uint64_t page_map_pending = 0;
    for (const auto& [ppn, page] : programmed_pages_) {
        if (ppn >= total_pages_) {
            throw std::runtime_error("HBF page-state table contains an out-of-range PPN");
        }
        const auto block_index = static_cast<std::size_t>(
            ppn / config_.pages_per_block);
        const auto page_index = static_cast<std::uint32_t>(
            ppn % config_.pages_per_block);
        const auto& block = blocks_.at(block_index);
        switch (page.status) {
        case PageStatus::Erased:
            page_map_pending++;
            break;
        case PageStatus::StaticReadOnly:
        case PageStatus::Valid:
            if (!block.is_valid(page_index)) {
                throw std::runtime_error(
                    "HBF valid page-state entry is absent from its block bitmap");
            }
            page_map_valid++;
            break;
        case PageStatus::Invalid:
            if (block.is_valid(page_index)) {
                throw std::runtime_error(
                    "HBF invalid page-state entry remains in its block bitmap");
            }
            page_map_invalid++;
            break;
        }
    }
    std::uint64_t compact_valid_pages = 0;
    if (compact_logical_image_) {
        const auto& image = *compact_logical_image_;
        if (image.data_blocks_by_plane.size() != planes_.size() ||
            image.mapping_ppns.size() != image.vpn_slot_count ||
            image.vpn_ranges.size() != image.vpn_slot_count ||
            image.vpn_offsets_by_stack.size() != config_.stacks) {
            throw std::runtime_error(
                "HBF compact image directory dimensions are inconsistent");
        }
        std::vector<std::uint8_t> compact_data_block_seen(blocks_.size(), 0);
        std::uint64_t compact_live_data_pages = 0;
        for (std::size_t plane = 0;
             plane < image.data_blocks_by_plane.size();
             ++plane) {
            const auto& assigned = image.data_blocks_by_plane[plane];
            for (std::size_t ordinal = 0;
                 ordinal < assigned.size();
                 ++ordinal) {
                const auto block_number = assigned[ordinal];
                if (block_number >= blocks_.size() ||
                    block_plane_index(static_cast<std::size_t>(block_number)) != plane ||
                    compact_data_block_seen.at(
                        static_cast<std::size_t>(block_number)) != 0) {
                    throw std::runtime_error(
                        "HBF compact data-block directory is invalid or duplicated");
                }
                compact_data_block_seen[static_cast<std::size_t>(block_number)] = 1;
                const auto location =
                    image.data_block_locations.find(block_number);
                if (location == image.data_block_locations.end() ||
                    location->second.plane != plane ||
                    location->second.block_ordinal != ordinal) {
                    throw std::runtime_error(
                        "HBF compact data-block reverse index diverged");
                }
                const auto& block = blocks_.at(
                    static_cast<std::size_t>(block_number));
                const auto live = image.live_data_pages_by_block.find(
                    block_number);
                const auto live_pages =
                    live == image.live_data_pages_by_block.end() ?
                    0 : static_cast<std::uint64_t>(live->second);
                if (live_pages > block.valid_pages ||
                    (live_pages != 0 &&
                     (block.role != BlockRole::Data ||
                      block.erase_pending))) {
                    throw std::runtime_error(
                        "HBF compact data-block live count references "
                        "non-live physical state");
                }
                compact_live_data_pages = checked_add(
                    compact_live_data_pages,
                    live_pages,
                    "HBF compact live data-page accounting");
            }
        }
        for (const auto& [block_number, live_pages] :
             image.live_data_pages_by_block) {
            if (live_pages == 0 ||
                block_number >= compact_data_block_seen.size() ||
                compact_data_block_seen[
                    static_cast<std::size_t>(block_number)] == 0) {
                throw std::runtime_error(
                    "HBF compact live data-block index is not canonical");
            }
        }
        if (image.retired_lpns.size() > image.page_count ||
            compact_live_data_pages !=
                image.page_count - image.retired_lpns.size()) {
            throw std::runtime_error(
                "HBF compact live/retired data pages do not conserve the "
                "logical image");
        }
        std::unordered_set<std::uint64_t> compact_mapping_seen;
        compact_mapping_seen.reserve(
            static_cast<std::size_t>(image.mapping_page_count));
        std::uint64_t compact_range_pages = 0;
        std::uint64_t compact_live_mapping_pages = 0;
        for (std::size_t index = 0;
             index < image.mapping_ppns.size();
             ++index) {
            const auto& range = image.vpn_ranges[index];
            const auto& compact_ppn = image.mapping_ppns[index];
            if ((range.page_count == 0) != !compact_ppn) {
                throw std::runtime_error(
                    "HBF compact mapping slot activity is inconsistent");
            }
            if (!compact_ppn) {
                continue;
            }
            compact_range_pages = checked_add(
                compact_range_pages,
                range.page_count,
                "HBF compact mapping-range page accounting");
            const auto ppn = *compact_ppn;
            if (ppn >= total_pages_) {
                throw std::runtime_error(
                    "HBF compact mapping-page directory is out of range");
            }
            const auto block_index = static_cast<std::size_t>(
                ppn / config_.pages_per_block);
            const auto page_index = static_cast<std::uint32_t>(
                ppn % config_.pages_per_block);
            const auto mapping_vpn = checked_add(
                image.first_vpn,
                index,
                "HBF compact accounting mapping VPN");
            const auto inverse = image.mapping_vpn_by_ppn.find(ppn);
            if (!compact_mapping_seen.insert(ppn).second ||
                inverse == image.mapping_vpn_by_ppn.end() ||
                inverse->second != mapping_vpn) {
                throw std::runtime_error(
                    "HBF compact mapping-page reverse index diverged");
            }
            if (!image.retired_mapping_vpns.contains(mapping_vpn)) {
                const auto& block = blocks_.at(block_index);
                if (block.role != BlockRole::Mapping ||
                    block.erase_pending ||
                    !block.is_valid(page_index)) {
                    throw std::runtime_error(
                        "HBF compact mapping-page directory references a "
                        "non-live page");
                }
                ++compact_live_mapping_pages;
            }
        }
        if (compact_range_pages != image.page_count ||
            compact_mapping_seen.size() != image.mapping_page_count ||
            image.mapping_vpn_by_ppn.size() != image.mapping_page_count ||
            image.retired_mapping_vpns.size() >
                image.mapping_page_count ||
            compact_live_mapping_pages !=
                image.mapping_page_count -
                    image.retired_mapping_vpns.size()) {
            throw std::runtime_error(
                "HBF compact mapping slots do not conserve the logical image");
        }
        std::uint64_t indexed_live_mapping_pages = 0;
        for (const auto& [block_number, live_pages] :
             image.live_mapping_pages_by_block) {
            if (live_pages == 0 || block_number >= blocks_.size()) {
                throw std::runtime_error(
                    "HBF compact live mapping-block index is invalid");
            }
            indexed_live_mapping_pages = checked_add(
                indexed_live_mapping_pages,
                live_pages,
                "HBF compact indexed live mapping pages");
        }
        if (indexed_live_mapping_pages != compact_live_mapping_pages) {
            throw std::runtime_error(
                "HBF compact mapping live-page block index diverged");
        }
        compact_valid_pages = checked_add(
            compact_live_data_pages,
            compact_live_mapping_pages,
            "HBF compact valid-page accounting");
        stats_.compact_live_logical_data_pages =
            compact_live_data_pages;
        stats_.compact_live_mapping_pages =
            compact_live_mapping_pages;
        stats_.compact_retired_logical_data_pages =
            image.retired_lpns.size();
        stats_.compact_retired_mapping_pages =
            image.retired_mapping_vpns.size();
    } else {
        stats_.compact_live_logical_data_pages = 0;
        stats_.compact_live_mapping_pages = 0;
        stats_.compact_retired_logical_data_pages = 0;
        stats_.compact_retired_mapping_pages = 0;
    }
    if (stats_.compact_initial_logical_data_pages >
            stats_.initial_logical_data_pages ||
        stats_.compact_initial_mapping_pages >
            stats_.initial_mapping_pages ||
        checked_add(
            stats_.compact_live_logical_data_pages,
            stats_.compact_retired_logical_data_pages,
            "HBF compact initial data-page lifecycle") !=
            stats_.compact_initial_logical_data_pages ||
        checked_add(
            stats_.compact_live_mapping_pages,
            stats_.compact_retired_mapping_pages,
            "HBF compact initial mapping-page lifecycle") !=
            stats_.compact_initial_mapping_pages) {
        throw std::runtime_error(
            "HBF compact initial-image lifecycle accounting diverged");
    }
    if (checked_add(
            page_map_valid,
            compact_valid_pages,
            "HBF materialized/compact valid-page accounting") != valid_pages ||
        page_map_invalid != invalid_pages ||
        page_map_pending != pending_program_pages) {
        throw std::runtime_error(
            "HBF page-state table diverged from block page counters");
    }
    const auto validate_mapping_target = [this](
                                             std::uint64_t ppn,
                                             std::uint64_t expected_lpn,
                                             PageOwner expected_owner,
                                             const char* name) {
        const auto page = programmed_pages_.find(ppn);
        if (ppn >= total_pages_ || page == programmed_pages_.end() ||
            page->second.status != PageStatus::Valid ||
            page->second.owner != expected_owner ||
            page->second.lpn != expected_lpn) {
            throw std::runtime_error(
                std::string("HBF ") + name +
                " points to a non-live or wrongly owned physical page");
        }
    };
    for (const auto& [lpn, ppn] : lpn_to_ppn_) {
        validate_mapping_target(ppn, lpn, PageOwner::Logical, "L2P mapping");
    }
    for (const auto& [mapping_vpn, ppn] : mapping_vpn_to_ppn_) {
        validate_mapping_target(
            ppn,
            metadata_lpn(mapping_vpn),
            PageOwner::Mapping,
            "mapping-page directory");
    }
    // Between media-program and mapping-publication callbacks, both old and
    // new versions may legitimately be valid. Once all program/publication
    // pins are gone, every live logical or mapping page must be reachable in
    // the corresponding directory; this catches orphaned GC relocations.
    if (pending_program_pages == 0 && pending_mapping_publications == 0) {
        for (const auto& [ppn, page] : programmed_pages_) {
            if (page.status != PageStatus::Valid) {
                continue;
            }
            if (page.owner == PageOwner::Logical) {
                const auto mapping = lpn_to_ppn_.find(page.lpn);
                if (mapping == lpn_to_ppn_.end() || mapping->second != ppn) {
                    throw std::runtime_error(
                        "HBF quiescent logical page is unreachable from L2P");
                }
            } else if (page.owner == PageOwner::Mapping) {
                const auto mapping_vpn = metadata_vpn(page.lpn);
                const auto mapping = mapping_vpn_to_ppn_.find(mapping_vpn);
                if (mapping == mapping_vpn_to_ppn_.end() || mapping->second != ppn) {
                    throw std::runtime_error(
                        "HBF quiescent mapping page is unreachable from its directory");
                }
            } else if (page.owner != PageOwner::RawPhysical) {
                throw std::runtime_error(
                    "HBF valid page has an invalid owner at quiescence");
            }
        }
    }
    if (free_pages != free_pages_ || free_pages_per_stack != free_pages_per_stack_) {
        throw std::runtime_error(
            "HBF global/per-stack free-page counters do not match block state");
    }
    const auto accounted_pages = checked_add(
        checked_add(
            checked_add(
                free_pages, valid_pages, "HBF free/valid capacity accounting"),
            invalid_pages,
            "HBF free/valid/invalid capacity accounting"),
        checked_add(
            pending_program_pages,
            static_unmaterialized_pages,
            "HBF pending/static capacity accounting"),
        "HBF total physical-page accounting");
    if (accounted_pages != total_pages_) {
        throw std::runtime_error(
            "HBF physical page states do not conserve total capacity");
    }

    const auto classified_programs = checked_add(
        checked_add(
            stats_.data_programs,
            stats_.mapping_page_programs,
            "HBF data/mapping program accounting"),
        stats_.gc_relocations,
        "HBF classified program accounting");
    const auto classified_payload_bytes = checked_add(
        checked_add(
            stats_.data_program_payload_bytes,
            stats_.mapping_program_payload_bytes,
            "HBF data/mapping payload-byte accounting"),
        stats_.gc_relocation_payload_bytes,
        "HBF classified payload-byte accounting");
    if (stats_.page_programs != classified_programs ||
        stats_.data_program_payload_bytes != checked_mul(
            stats_.data_programs,
            config_.page_size_bytes,
            "HBF data-program payload-byte accounting") ||
        stats_.mapping_program_payload_bytes != checked_mul(
            stats_.mapping_page_programs,
            config_.page_size_bytes,
            "HBF mapping-program payload-byte accounting") ||
        stats_.gc_relocation_payload_bytes != checked_mul(
            stats_.gc_relocations,
            config_.page_size_bytes,
            "HBF GC-relocation payload-byte accounting") ||
        stats_.physical_write_bytes != classified_payload_bytes ||
        stats_.physical_write_bytes != checked_mul(
            stats_.page_programs,
            config_.page_size_bytes,
            "HBF physical-write byte accounting") ||
        stats_.physical_read_bytes != checked_mul(
            stats_.page_reads,
            config_.page_size_bytes,
            "HBF physical-read byte accounting")) {
        throw std::runtime_error(
            "HBF program/read byte accounting identities diverged");
    }
    if (stats_.gc_relocations != checked_add(
            stats_.gc_data_relocations,
            stats_.gc_mapping_relocations,
            "HBF GC relocation owner accounting")) {
        throw std::runtime_error(
            "HBF GC data/mapping relocation counts do not conserve relocations");
    }
    const auto gc_victim_pages = checked_mul(
        stats_.gc_runs,
        config_.pages_per_block,
        "HBF GC victim-page accounting");
    if (checked_add(
            stats_.gc_relocations,
            stats_.gc_reclaimed_invalid_pages,
            "HBF GC relocated/reclaimed page accounting") != gc_victim_pages ||
        stats_.block_erases != checked_add(
            stats_.erase_requests,
            stats_.gc_runs,
            "HBF user/GC erase accounting") ||
        stats_.invalidations < stats_.gc_relocations) {
        throw std::runtime_error(
            "HBF GC victim, erase, or invalidation accounting diverged");
    }
    const auto flash_transactions = checked_add(
        checked_add(
            stats_.page_reads,
            stats_.page_programs,
            "HBF flash read/program accounting"),
        stats_.block_erases,
        "HBF flash transaction accounting");
    if (stats_.flash_scheduler_enqueues != flash_transactions ||
        stats_.flash_scheduler_issues != flash_transactions) {
        throw std::runtime_error(
            "HBF flash scheduler counts do not conserve media transactions");
    }

    stats_.total_pages = total_pages_;
    stats_.free_pages = free_pages;
    stats_.valid_pages = valid_pages;
    stats_.invalid_pages = invalid_pages;
    stats_.pending_program_pages = pending_program_pages;
    stats_.pending_mapping_publications = pending_mapping_publications;
    stats_.static_unmaterialized_pages = static_unmaterialized_pages;
    stats_.accounting_verified = true;
}

void HbfDevice::refresh_parallel_stats() const {
    refresh_accounting_stats();
    stats_.mapping_entries = logical_mapping_entry_count();
    // The same physical work is accumulated in request order for the public
    // total and in resource order for directional/per-die totals. Their
    // round-off bound grows with the number of additions; a fixed 32-epsilon
    // tolerance falsely rejected long, otherwise exact replays.
    const auto accumulation_terms = std::max<std::uint64_t>(
        32,
        checked_add(
            checked_add(
                checked_add(
                    stats_.read_requests,
                    stats_.program_requests,
                    "HBF conservation request terms"),
                stats_.erase_requests,
                "HBF conservation request/erase terms"),
            checked_add(
                checked_add(
                    stats_.page_reads,
                    stats_.page_programs,
                    "HBF conservation media terms"),
                checked_add(
                    stats_.read_buffer_hits,
                    stats_.write_buffer_read_hits,
                    "HBF conservation buffer terms"),
                "HBF conservation media/buffer terms"),
            "HBF conservation total terms"));
    const auto work_conserved = [accumulation_terms](
                                    double total, double decode, double encode) {
        const double expected = decode + encode;
        const double scale = std::max({1.0, std::abs(total), std::abs(expected)});
        return std::abs(total - expected) <=
            8.0 * static_cast<double>(accumulation_terms) *
                std::numeric_limits<double>::epsilon() * scale;
    };
    if (stats_.ecc_decode_ops != stats_.page_reads) {
        throw std::runtime_error(
            "HBF ECC decode operation count diverged from physical page reads");
    }
    if (stats_.ecc_encode_ops != stats_.page_programs) {
        throw std::runtime_error(
            "HBF ECC encode operation count diverged from physical page programs");
    }
    const auto expected_decode_bytes = checked_mul(
        stats_.ecc_decode_ops, page_wire_bytes(), "HBF ECC decode conservation");
    const auto expected_encode_bytes = checked_mul(
        stats_.ecc_encode_ops, page_wire_bytes(), "HBF ECC encode conservation");
    if (stats_.ecc_decode_codeword_bytes != expected_decode_bytes ||
        stats_.ecc_encode_codeword_bytes != expected_encode_bytes ||
        stats_.ecc_codeword_bytes != checked_add(
            expected_decode_bytes, expected_encode_bytes,
            "HBF ECC total conservation")) {
        throw std::runtime_error(
            "HBF ECC codeword-byte accounting did not conserve page operations");
    }
    if (!work_conserved(
            stats_.stage_work.ecc_queue_wait_ns,
            stats_.ecc_decode_queue_wait_ns,
            stats_.ecc_encode_queue_wait_ns) ||
        !work_conserved(
            stats_.stage_work.ecc_latency_ns,
            stats_.ecc_decode_latency_work_ns,
            stats_.ecc_encode_latency_work_ns) ||
        !work_conserved(
            stats_.ecc_issue_busy_ns,
            stats_.ecc_decode_issue_busy_ns,
            stats_.ecc_encode_issue_busy_ns)) {
        throw std::runtime_error(
            "HBF ECC directional work accounting did not conserve totals");
    }
    stats_.stacks = config_.stacks;
    stats_.channels = channels_.size();
    stats_.dies = dies_.size();
    stats_.planes = planes_.size();
    stats_.media_lanes = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        config_.media_lanes_per_plane,
        "HBF stats media_lanes");
    stats_.subarrays = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        subarrays_per_plane_,
        "HBF stats subarrays");
    stats_.page_buffer_banks = checked_mul(
        static_cast<std::uint64_t>(planes_.size()),
        config_.page_buffer_banks_per_plane,
        "HBF stats page_buffer_banks");
    stats_.active_channels = 0;
    stats_.active_dies = 0;
    stats_.active_planes = 0;
    stats_.active_media_lanes = 0;
    stats_.active_subarrays = 0;
    stats_.active_page_buffer_banks = 0;
    stats_.max_plane_ops = 0;
    stats_.max_media_lane_reads = 0;
    stats_.max_subarray_reads = 0;
    stats_.max_page_buffer_bank_reads = 0;
    stats_.max_die_transactions = 0;
    stats_.max_plane_media_busy_ns = 0.0;
    stats_.avg_active_plane_media_busy_ns = 0.0;
    stats_.avg_active_plane_ops = 0.0;
    stats_.read_lane_busy_ns = 0.0;
    stats_.subarray_read_busy_ns = 0.0;
    stats_.page_buffer_bank_busy_ns = 0.0;
    stats_.max_media_lane_busy_ns = 0.0;
    stats_.avg_active_media_lane_busy_ns = 0.0;
    stats_.avg_active_media_lane_reads = 0.0;
    stats_.max_subarray_busy_ns = 0.0;
    stats_.avg_active_subarray_busy_ns = 0.0;
    stats_.avg_active_subarray_reads = 0.0;
    stats_.max_page_buffer_bank_busy_ns = 0.0;
    stats_.avg_active_page_buffer_bank_busy_ns = 0.0;
    stats_.avg_active_page_buffer_bank_reads = 0.0;
    stats_.max_channel_busy_ns = 0.0;
    stats_.avg_active_channel_busy_ns = 0.0;
    stats_.avg_active_die_transactions = 0.0;
    stats_.sequencer_busy_ns = 0.0;
    stats_.hb_io_command_busy_ns = 0.0;
    stats_.hb_io_data_busy_ns = 0.0;
    stats_.logic_ingress_busy_ns = 0.0;
    stats_.tsv_busy_ns = 0.0;
    stats_.sram_busy_ns = 0.0;
    stats_.flash_source_queue_busy_ns = 0.0;
    stats_.channel_command_busy_ns = 0.0;
    stats_.channel_data_busy_ns = 0.0;
    stats_.logic_ingress_resources = logic_dies_.size();
    stats_.tsv_resources = logic_dies_.size();
    stats_.sram_resources = logic_dies_.size();
    stats_.flash_source_queue_resources = checked_mul(
        static_cast<std::uint64_t>(dies_.size()),
        static_cast<std::uint64_t>(std::tuple_size_v<
            decltype(DieState::source_queues)>),
        "HBF stats flash source queue resources");
    stats_.channel_command_resources = channels_.size();
    stats_.channel_data_resources = channels_.size();
    stats_.active_ecc_dies = 0;
    stats_.max_ecc_inflight_per_die = 0;
    stats_.max_ecc_issue_busy_ns = 0.0;
    stats_.avg_active_ecc_issue_busy_ns = 0.0;

    double active_plane_busy_ns = 0.0;
    std::uint64_t active_plane_ops = 0;
    for (const auto& plane : planes_) {
        const auto ops = plane.read_count + plane.program_count + plane.erase_count;
        if (ops == 0 && plane.media_busy_ns <= 0.0) {
            continue;
        }
        stats_.active_planes++;
        active_plane_busy_ns += plane.media_busy_ns;
        active_plane_ops += ops;
        stats_.max_plane_media_busy_ns =
            std::max(stats_.max_plane_media_busy_ns, plane.media_busy_ns);
        stats_.max_plane_ops = std::max(stats_.max_plane_ops, ops);

        for (const auto& lane : plane.media_lanes) {
            if (lane.read_count == 0 && lane.busy_ns <= 0.0) {
                continue;
            }
            stats_.active_media_lanes++;
            stats_.read_lane_busy_ns += lane.busy_ns;
            stats_.max_media_lane_busy_ns =
                std::max(stats_.max_media_lane_busy_ns, lane.busy_ns);
            stats_.max_media_lane_reads =
                std::max(stats_.max_media_lane_reads, lane.read_count);
        }

        for (const auto& subarray : plane.subarrays) {
            if (subarray.read_count == 0 && subarray.busy_ns <= 0.0) {
                continue;
            }
            stats_.active_subarrays++;
            stats_.subarray_read_busy_ns += subarray.busy_ns;
            stats_.max_subarray_busy_ns =
                std::max(stats_.max_subarray_busy_ns, subarray.busy_ns);
            stats_.max_subarray_reads =
                std::max(stats_.max_subarray_reads, subarray.read_count);
        }

        for (const auto& bank : plane.page_buffer_banks) {
            if (bank.read_count == 0 && bank.busy_ns <= 0.0) {
                continue;
            }
            stats_.active_page_buffer_banks++;
            stats_.page_buffer_bank_busy_ns += bank.busy_ns;
            stats_.max_page_buffer_bank_busy_ns =
                std::max(stats_.max_page_buffer_bank_busy_ns, bank.busy_ns);
            stats_.max_page_buffer_bank_reads =
                std::max(stats_.max_page_buffer_bank_reads, bank.read_count);
        }
    }
    stats_.media_busy_ns = active_plane_busy_ns;
    if (stats_.active_planes != 0) {
        stats_.avg_active_plane_media_busy_ns =
            active_plane_busy_ns / static_cast<double>(stats_.active_planes);
        stats_.avg_active_plane_ops =
            static_cast<double>(active_plane_ops) / static_cast<double>(stats_.active_planes);
    }
    if (stats_.active_media_lanes != 0) {
        stats_.avg_active_media_lane_busy_ns =
            stats_.read_lane_busy_ns / static_cast<double>(stats_.active_media_lanes);
        std::uint64_t active_lane_reads = 0;
        for (const auto& plane : planes_) {
            for (const auto& lane : plane.media_lanes) {
                if (lane.read_count != 0 || lane.busy_ns > 0.0) {
                    active_lane_reads += lane.read_count;
                }
            }
        }
        stats_.avg_active_media_lane_reads =
            static_cast<double>(active_lane_reads) /
            static_cast<double>(stats_.active_media_lanes);
    }
    if (stats_.active_subarrays != 0) {
        stats_.avg_active_subarray_busy_ns =
            stats_.subarray_read_busy_ns / static_cast<double>(stats_.active_subarrays);
        std::uint64_t active_subarray_reads = 0;
        for (const auto& plane : planes_) {
            for (const auto& subarray : plane.subarrays) {
                if (subarray.read_count != 0 || subarray.busy_ns > 0.0) {
                    active_subarray_reads += subarray.read_count;
                }
            }
        }
        stats_.avg_active_subarray_reads =
            static_cast<double>(active_subarray_reads) /
            static_cast<double>(stats_.active_subarrays);
    }
    if (stats_.active_page_buffer_banks != 0) {
        double active_bank_busy_ns = 0.0;
        std::uint64_t active_bank_reads = 0;
        for (const auto& plane : planes_) {
            for (const auto& bank : plane.page_buffer_banks) {
                if (bank.read_count != 0 || bank.busy_ns > 0.0) {
                    active_bank_busy_ns += bank.busy_ns;
                    active_bank_reads += bank.read_count;
                }
            }
        }
        stats_.avg_active_page_buffer_bank_busy_ns =
            active_bank_busy_ns / static_cast<double>(stats_.active_page_buffer_banks);
        stats_.avg_active_page_buffer_bank_reads =
            static_cast<double>(active_bank_reads) /
            static_cast<double>(stats_.active_page_buffer_banks);
    }

    double active_channel_busy_ns = 0.0;
    for (const auto& channel : channels_) {
        stats_.channel_command_busy_ns += channel.command_busy_ns;
        stats_.channel_data_busy_ns += channel.data_busy_ns;
        const double busy_ns = channel.command_busy_ns + channel.data_busy_ns;
        if (channel.command_count == 0 && channel.data_count == 0 && busy_ns <= 0.0) {
            continue;
        }
        stats_.active_channels++;
        active_channel_busy_ns += busy_ns;
        stats_.max_channel_busy_ns = std::max(stats_.max_channel_busy_ns, busy_ns);
    }
    if (stats_.active_channels != 0) {
        stats_.avg_active_channel_busy_ns =
            active_channel_busy_ns / static_cast<double>(stats_.active_channels);
    }
    if (!work_conserved(
            stats_.stage_work.channel_transfer_ns,
            stats_.channel_command_busy_ns,
            stats_.channel_data_busy_ns)) {
        throw std::runtime_error(
            "HBF channel command/data resource work did not conserve stage work");
    }

    for (const auto& logic_die : logic_dies_) {
        stats_.logic_ingress_busy_ns += logic_die.ingress.reserved_work_ns;
        stats_.tsv_busy_ns += logic_die.tsv.reserved_work_ns;
        stats_.sram_busy_ns += logic_die.sram.reserved_work_ns;
        stats_.hb_io_command_busy_ns += logic_die.hb_io_command_busy_ns;
        stats_.hb_io_data_busy_ns += logic_die.hb_io_data_busy_ns;
    }
    if (!work_conserved(
            stats_.stage_work.hb_io_transfer_ns,
            stats_.hb_io_command_busy_ns,
            stats_.hb_io_data_busy_ns)) {
        throw std::runtime_error(
            "HBF HBIO command/data work accounting did not conserve totals");
    }
    if (!work_conserved(
            stats_.stage_work.tsv_transfer_ns,
            stats_.tsv_busy_ns,
            0.0)) {
        throw std::runtime_error(
            "HBF TSV resource work did not conserve stage work");
    }
    if (!work_conserved(
            stats_.stage_work.sram_staging_ns,
            stats_.sram_busy_ns,
            0.0)) {
        throw std::runtime_error(
            "HBF SRAM resource work did not conserve stage work");
    }

    std::uint64_t active_die_transactions = 0;
    double active_ecc_issue_busy_ns = 0.0;
    std::uint64_t die_ecc_decode_ops = 0;
    std::uint64_t die_ecc_encode_ops = 0;
    double die_ecc_issue_busy_ns = 0.0;
    for (const auto& die : dies_) {
        stats_.sequencer_busy_ns += die.sequencer_busy_ns;
        for (const auto& queue : die.source_queues) {
            stats_.flash_source_queue_busy_ns += queue.reserved_work_ns;
        }
        die_ecc_decode_ops = checked_add(
            die_ecc_decode_ops, die.ecc_decode_ops, "HBF die ECC decode ops");
        die_ecc_encode_ops = checked_add(
            die_ecc_encode_ops, die.ecc_encode_ops, "HBF die ECC encode ops");
        die_ecc_issue_busy_ns += die.ecc_issue_busy_ns;
        if (die.ecc_decode_ops != 0 || die.ecc_encode_ops != 0 ||
            die.ecc_issue_busy_ns > 0.0) {
            stats_.active_ecc_dies++;
            active_ecc_issue_busy_ns += die.ecc_issue_busy_ns;
            stats_.max_ecc_issue_busy_ns =
                std::max(stats_.max_ecc_issue_busy_ns, die.ecc_issue_busy_ns);

            // Exact reservation calendars may backfill issue slots, so call
            // order is not time order. Sweep recorded latency intervals to
            // compute concurrency without assuming monotonic scheduling.
            std::vector<std::pair<double, int>> events;
            events.reserve(die.ecc_inflight_intervals.size() * 2);
            for (const auto& interval : die.ecc_inflight_intervals) {
                events.emplace_back(interval.start_ns, 1);
                events.emplace_back(interval.finish_ns, -1);
            }
            std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                // A completion at t leaves before a new codeword enters at t.
                return lhs.second < rhs.second;
            });
            std::int64_t inflight = 0;
            std::uint64_t max_inflight = 0;
            for (const auto& [_, delta] : events) {
                inflight += delta;
                if (inflight < 0) {
                    throw std::runtime_error(
                        "HBF ECC in-flight accounting became negative");
                }
                max_inflight = std::max(
                    max_inflight, static_cast<std::uint64_t>(inflight));
            }
            if (inflight != 0) {
                throw std::runtime_error(
                    "HBF ECC in-flight accounting did not conserve operations");
            }
            stats_.max_ecc_inflight_per_die =
                std::max(stats_.max_ecc_inflight_per_die, max_inflight);
        }
        if (die.transaction_count == 0 && die.sequencer_busy_ns <= 0.0) {
            continue;
        }
        stats_.active_dies++;
        active_die_transactions += die.transaction_count;
        stats_.max_die_transactions =
            std::max(stats_.max_die_transactions, die.transaction_count);
    }
    if (stats_.active_dies != 0) {
        stats_.avg_active_die_transactions =
            static_cast<double>(active_die_transactions) / static_cast<double>(stats_.active_dies);
    }
    if (stats_.active_ecc_dies != 0) {
        stats_.avg_active_ecc_issue_busy_ns =
            active_ecc_issue_busy_ns / static_cast<double>(stats_.active_ecc_dies);
    }
    if (die_ecc_decode_ops != stats_.ecc_decode_ops ||
        die_ecc_encode_ops != stats_.ecc_encode_ops ||
        !work_conserved(stats_.ecc_issue_busy_ns, die_ecc_issue_busy_ns, 0.0)) {
        throw std::runtime_error(
            "HBF ECC per-die accounting did not conserve aggregate work");
    }
    if (stats_.read_splits > stats_.read_requests) {
        throw std::runtime_error(
            "HBF read-split count exceeds read-request count");
    }
    const auto expected_page_admissions = checked_add(
        stats_.read_split_pages,
        stats_.read_requests - stats_.read_splits,
        "HBF expected page-read admissions");
    if (stats_.page_read_admission_events != expected_page_admissions ||
        stats_.page_read_admission_waited_pages >
            stats_.page_read_admission_events ||
        (stats_.page_read_admission_waited_pages == 0) !=
            (stats_.page_read_admission_wait_ns == 0.0) ||
        (stats_.page_read_admission_waited_pages == 0) !=
            (stats_.page_read_admission_max_wait_ns == 0.0) ||
        stats_.page_read_admission_max_wait_ns >
            stats_.page_read_admission_wait_ns) {
        throw std::runtime_error(
            "HBF page-read admission accounting did not conserve requests");
    }
}

} // namespace hbfsim::physical::hbf
