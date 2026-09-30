#pragma once
#include "physical/physical_types.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace hbfsim::physical {
inline double causal_finish(double start, double duration) {
    if (!std::isfinite(start) || !std::isfinite(duration) || start < 0 || duration < 0)
        throw std::runtime_error("invalid resource reservation");
    const double sum = start + duration;
    const double end = duration == 0 || sum > start ? sum : std::nextafter(start, std::numeric_limits<double>::infinity());
    if (!std::isfinite(end)) throw std::overflow_error("resource time overflow");
    return end;
}


// Exact first-fit resource calendar, in caller-selected time units. Idle
// intervals start in a small inline array, promoting to sorted 64-entry leaves
// and groups with duration summaries when contention requires more history.
// Queries and mutations reuse leaf positions; pruning drops whole leaves.
// A gap fits iff causal_finish(start, duration) <= end. Summary pruning allows
// floating-point rounding slack, so results do not depend on tree shape at
// fractional-nanosecond boundaries. No times are quantized or coarsened.
class ResourceTimeline {
public:
    struct Gap {
        double begin_ns = 0.0;
        double end_ns = 0.0;
        bool operator==(const Gap&) const = default;
    };
    struct Reservation {
        double start_ns = 0.0;
        double finish_ns = 0.0;
        bool operator==(const Reservation&) const = default;
    };

    double ready_ns = 0.0;
    // Sum of all non-overlapping reservations on this one resource.
    double reserved_work_ns = 0.0;

    [[nodiscard]] std::optional<Gap> first_fitting_gap(
        double earliest_ns,
        double duration_ns) const;
    [[nodiscard]] bool can_reserve_exact(
        double begin_ns,
        double duration_ns) const;
    [[nodiscard]] double preview_start(
        double earliest_ns,
        double duration_ns) const;
    void consume_gap(const Gap& gap, double begin_ns, double end_ns);
    void insert_gap(double begin_ns, double end_ns);
    // Records [ready_ns, start_ns) as idle, without searching earlier intervals.
    void insert_frontier_gap(double start_ns);
    void prune_before(double causal_watermark_ns);
    [[nodiscard]] std::vector<Gap> gaps_after(double floor_ns) const;
    // Returns the exact endpoints used to consume capacity. A gap fill can
    // finish before ready_ns; callers must not use the frontier as its finish.
    [[nodiscard]] Reservation reserve(double earliest_ns, double duration_ns);
    // Two fixed-stage streams may share one calendar (for example TSV
    // commands and data). Only a proven first-fit lower bound is reused;
    // arrivals, durations, endpoints and retained idle gaps stay exact.
    [[nodiscard]] Reservation reserve_ordered(
        double earliest_ns, double duration_ns, unsigned stream = 0);
    [[nodiscard]] std::uint64_t ordered_reservation_reuses() const {
        return ordered_reservation_reuses_;
    }
    [[nodiscard]] std::uint64_t ordered_position_reuses() const {
        return ordered_position_reuses_;
    }
    [[nodiscard]] std::size_t gap_count() const;

private:
    // Most bank/lane calendars contain only a few live idle intervals during
    // a parallel read group. Keep them inline; shared resources with long
    // histories promote to the same exact packed index below.
    static constexpr std::uint32_t kInlineGaps = 16;
    Gap inline_gaps_[kInlineGaps];
    std::uint32_t inline_count_ = 0;
    bool inline_mode_ = true;
    void promote_inline();
    void demote_if_small();
    [[nodiscard]] std::optional<std::uint32_t> inline_fit(double earliest, double duration) const;
    void consume_inline(std::uint32_t slot, const Gap& gap, double begin, double end);
    static constexpr std::uint32_t kLeafGaps = 64;
    static constexpr std::uint32_t kGroupLeaves = 64;
    struct Leaf {
        std::uint32_t count = 0;
        // Number of gaps attaining this leaf's maximum duration.
        std::uint32_t max_duration_count = 0;
        Gap gaps[kLeafGaps];
    };
    // Leaves in key order with one summary row per leaf.
    struct Group {
        std::uint32_t count = 0;
        // Number of leaf rows attaining the group's maximum duration.
        std::uint32_t max_duration_count = 0;
        std::uint32_t leaves[kGroupLeaves];
        double first_keys[kGroupLeaves];
        double max_durations[kGroupLeaves];
        double max_ends[kGroupLeaves];
    };
    // (index into order_, leaf slot in that group, gap slot in that leaf);
    // group == order_.size() is the end position.
    struct Position {
        std::size_t group = 0;
        std::uint32_t leaf = 0;
        std::uint32_t slot = 0;
    };
    struct Found {
        Gap gap;
        Position at;
    };
    struct GapQuery {
        bool valid = false;
        double earliest_ns = 0.0;
        double duration_ns = 0.0;
        double frontier_ns = 0.0;
        // Mutations invalidate the query, so the position stays usable for
        // preview -> reserve/consume and for copies of the whole calendar.
        std::optional<Found> result;
    };
    struct OrderedRun {
        bool valid = false;
        double earliest_ns = 0.0;
        double finish_ns = 0.0;
        double duration_ns = 0.0;
        // Interleaved streams keep independent navigation locality. This is
        // only a candidate position; current key bounds validate every use.
        Position navigation;
    };

    std::vector<Leaf> leaves_;
    std::vector<std::uint32_t> free_leaves_;
    std::vector<Group> groups_;
    std::vector<std::uint32_t> free_groups_;
    // Live groups are order_[head_..); the summaries are parallel to order_.
    std::vector<std::uint32_t> order_;
    std::vector<double> first_keys_;
    std::vector<double> max_durations_;
    std::vector<double> max_ends_;
    std::size_t head_ = 0;
    double pruned_through_ns_ = 0.0;
    mutable GapQuery last_gap_query_;
    // A navigation hint, not a cached answer. Current key bounds are checked
    // on every use, including after splits, pruning, recycling and copies.
    mutable Position key_leaf_hint_;
    mutable OrderedRun ordered_runs_[2];
    // ready_ns is public. A direct assignment must not inherit proofs from
    // the previous frontier, including through an intervening public API.
    mutable double observed_frontier_ns_ = 0.0;
    std::uint64_t ordered_reservation_reuses_ = 0;
    std::uint64_t ordered_position_reuses_ = 0;

    void invalidate_ordered_runs() const;
    void observe_frontier() const;
    [[nodiscard]] bool ordered_gaps_within_frontier() const;
    [[nodiscard]] std::optional<Found> ordered_position_gap(
        Position hint, double earliest_ns, double duration_ns) const;
    template <bool Ordered>
    [[nodiscard]] Reservation reserve_impl(
        double earliest_ns, double duration_ns, OrderedRun* ordered);

    [[nodiscard]] bool at_end(Position p) const { return p.group == order_.size(); }
    [[nodiscard]] Group& group_at(std::size_t position) { return groups_[order_[position]]; }
    [[nodiscard]] const Group& group_at(std::size_t position) const {
        return groups_[order_[position]];
    }
    [[nodiscard]] Leaf& leaf_at(Position p) { return leaves_[group_at(p.group).leaves[p.leaf]]; }
    [[nodiscard]] const Leaf& leaf_at(Position p) const {
        return leaves_[group_at(p.group).leaves[p.leaf]];
    }
    [[nodiscard]] const Gap& gap_at(Position p) const { return leaf_at(p).gaps[p.slot]; }
    [[nodiscard]] Position lower_bound(double key) const;  // first begin >= key
    [[nodiscard]] Position upper_bound(double key) const;  // first begin > key
    [[nodiscard]] Position key_leaf(double key) const;  // key >= first live begin
    [[nodiscard]] Position first_after_floor(double floor_ns) const;  // first end > floor
    [[nodiscard]] bool previous(Position& p) const;
    [[nodiscard]] Position next(Position p) const;
    [[nodiscard]] static bool cannot_hold(double max_duration_ns, double max_end_ns, double duration_ns);
    [[nodiscard]] std::uint32_t allocate_leaf();
    [[nodiscard]] std::uint32_t allocate_group();
    void refresh_leaf_row(Group& group, std::uint32_t leaf);
    void refresh_group_summary(std::size_t position);
    void note_leaf_max_changed(Position p, double old_max);
    void note_inserted(Position p, const Gap& gap);
    void note_replaced(Position p, const Gap& old, const Gap& gap);
    void note_erased(Position p, const Gap& old);
    void split_leaf(Position& p);
    void split_group(Position& p);
    void remove_leaf(Position p);
    void insert_at(Position p, const Gap& gap);
    void erase_at(Position p);
    void replace_at(Position p, const Gap& gap);
    void drop_before(Position p);
    void compact();
    void reclaim_sparse_arena();
    [[nodiscard]] std::optional<Found> search(double earliest_ns, double duration_ns) const;
    [[nodiscard]] std::optional<Found> lookup_gap(double earliest_ns, double duration_ns) const;
    void consume_at(Position held, const Gap& gap, double begin_ns, double end_ns);
};

}
