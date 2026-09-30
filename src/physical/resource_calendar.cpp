#include "physical/resource_calendar.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace hbfsim::physical {
namespace {
double duration_of(const ResourceTimeline::Gap& gap) {
    return gap.end_ns - gap.begin_ns;
}
bool reservation_outside(const ResourceTimeline::Gap& gap, double begin_ns, double end_ns) {
    return begin_ns < gap.begin_ns || end_ns > gap.end_ns || end_ns <= begin_ns;
}
template <class T>
void shift_up(T* values, std::uint32_t at, std::uint32_t count) {
    std::move_backward(values + at, values + count, values + count + 1);
}
template <class T>
void shift_down(T* values, std::uint32_t at, std::uint32_t count, std::uint32_t by) {
    std::move(values + at, values + count, values + at - by);
}
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

void ResourceTimeline::promote_inline() {
    for (std::uint32_t i = 0; i < inline_count_; ++i)
        insert_at(Position{order_.size(), 0, 0}, inline_gaps_[i]);
    inline_count_ = 0;
    inline_mode_ = false;
    last_gap_query_.valid = false;
}

void ResourceTimeline::demote_if_small() {
    if (head_ == order_.size()) {
        inline_mode_ = true;
        inline_count_ = 0;
    } else if (order_.size() - head_ == 1 && group_at(head_).count == 1) {
        const auto& leaf = leaf_at(Position{head_, 0, 0});
        if (leaf.count <= kInlineGaps / 2) {
            inline_count_ = leaf.count;
            std::copy(leaf.gaps, leaf.gaps + leaf.count, inline_gaps_);
            drop_before(Position{order_.size(), 0, 0});
            inline_mode_ = true;
        }
    }
}

std::optional<std::uint32_t> ResourceTimeline::inline_fit(double earliest, double duration) const {
    if (earliest >= ready_ns) return std::nullopt;
    for (std::uint32_t i = 0; i < inline_count_; ++i) {
        const auto& gap = inline_gaps_[i];
        if (gap.end_ns > earliest &&
            causal_finish(std::max(earliest, gap.begin_ns), duration) <= gap.end_ns) return i;
    }
    return std::nullopt;
}

void ResourceTimeline::consume_inline(std::uint32_t at, const Gap& gap, double begin, double end) {
    const bool head = gap.begin_ns < begin, tail = end < gap.end_ns;
    if (head && tail) {
        if (inline_count_ == kInlineGaps) {
            promote_inline();
            consume_gap(gap, begin, end);
            return;
        }
        shift_up(inline_gaps_, at + 1, inline_count_);
        inline_gaps_[at] = {gap.begin_ns, begin};
        inline_gaps_[at + 1] = {end, gap.end_ns};
        ++inline_count_;
    } else if (head) inline_gaps_[at].end_ns = begin;
    else if (tail) inline_gaps_[at].begin_ns = end;
    else {
        shift_down(inline_gaps_, at + 1, inline_count_, 1);
        --inline_count_;
    }
}

ResourceTimeline::Position ResourceTimeline::key_leaf(double key) const {
    auto g = key_leaf_hint_.group;
    if (g < head_ || g >= order_.size() || first_keys_[g] > key ||
        (g + 1 < order_.size() && first_keys_[g + 1] <= key)) {
        const auto first = first_keys_.begin() + static_cast<std::ptrdiff_t>(head_);
        g = head_ + static_cast<std::size_t>(
            std::upper_bound(first, first_keys_.end(), key) - first) - 1;
    }
    const Group& group = group_at(g);
    auto l = key_leaf_hint_.leaf;
    if (l >= group.count || group.first_keys[l] > key ||
        (l + 1 < group.count && group.first_keys[l + 1] <= key)) {
        l = static_cast<std::uint32_t>(std::upper_bound(
            group.first_keys, group.first_keys + group.count, key) - group.first_keys) - 1;
    }
    key_leaf_hint_ = Position{g, l, 0};
    return key_leaf_hint_;
}

ResourceTimeline::Position ResourceTimeline::upper_bound(double key) const {
    if (head_ != order_.size()) {
        const auto g = order_.size() - 1;
        const auto l = group_at(g).count - 1;
        const auto& leaf = leaf_at(Position{g, l, 0});
        if (key >= leaf.gaps[leaf.count - 1].begin_ns)
            return Position{order_.size(), 0, 0};
    }
    if (head_ == order_.size() || key < first_keys_[head_]) {
        return Position{head_, 0, 0};
    }
    const auto position = key_leaf(key);
    const auto g = position.group;
    const auto l = position.leaf;
    const Group& group = group_at(g);
    const Leaf& leaf = leaves_[group.leaves[l]];
    const Gap* const begin = leaf.gaps;
    const Gap* const end = leaf.gaps + leaf.count;
    const Gap* slot = std::upper_bound(
        begin, end, key, [](double value, const Gap& gap) { return value < gap.begin_ns; });
    if (slot != end) {
        return Position{g, l, static_cast<std::uint32_t>(slot - begin)};
    }
    if (l + 1 < group.count) {
        return Position{g, l + 1, 0};
    }
    return Position{g + 1, 0, 0};
}

ResourceTimeline::Position ResourceTimeline::lower_bound(double key) const {
    if (head_ != order_.size()) {
        const auto g = order_.size() - 1;
        const auto l = group_at(g).count - 1;
        const auto& leaf = leaf_at(Position{g, l, 0});
        const auto tail = leaf.gaps[leaf.count - 1].begin_ns;
        if (key > tail) return Position{order_.size(), 0, 0};
        if (key == tail) return Position{g, l, leaf.count - 1};
    }
    if (head_ == order_.size() || key < first_keys_[head_]) {
        return Position{head_, 0, 0};
    }
    const auto position = key_leaf(key);
    const auto g = position.group;
    const auto l = position.leaf;
    const Group& group = group_at(g);
    const Leaf& leaf = leaves_[group.leaves[l]];
    const Gap* const begin = leaf.gaps;
    const Gap* const end = leaf.gaps + leaf.count;
    const Gap* slot = std::lower_bound(
        begin, end, key, [](const Gap& gap, double value) { return gap.begin_ns < value; });
    if (slot != end) {
        return Position{g, l, static_cast<std::uint32_t>(slot - begin)};
    }
    if (l + 1 < group.count) {
        return Position{g, l + 1, 0};
    }
    return Position{g + 1, 0, 0};
}

ResourceTimeline::Position ResourceTimeline::first_after_floor(double floor_ns) const {
    // Gaps are disjoint and ordered, so the only gap that can end after the
    // floor while beginning at or before it is the last one beginning there.
    const Position after = upper_bound(floor_ns);
    Position last = after;
    if (previous(last) && gap_at(last).end_ns > floor_ns) {
        return last;
    }
    return after;
}

bool ResourceTimeline::previous(Position& p) const {
    if (p.slot > 0) {
        --p.slot;
        return true;
    }
    if (p.leaf > 0) {
        --p.leaf;
        p.slot = leaf_at(p).count - 1;
        return true;
    }
    if (p.group > head_) {
        --p.group;
        p.leaf = group_at(p.group).count - 1;
        p.slot = leaf_at(p).count - 1;
        return true;
    }
    return false;
}

ResourceTimeline::Position ResourceTimeline::next(Position p) const {
    const Group& group = group_at(p.group);
    if (p.slot + 1 < leaves_[group.leaves[p.leaf]].count) {
        return Position{p.group, p.leaf, p.slot + 1};
    }
    if (p.leaf + 1 < group.count) {
        return Position{p.group, p.leaf + 1, 0};
    }
    return Position{p.group + 1, 0, 0};
}

// ---------------------------------------------------------------------------
// Storage and summaries
// ---------------------------------------------------------------------------

std::uint32_t ResourceTimeline::allocate_leaf() {
    if (free_leaves_.empty()) {
        leaves_.emplace_back();
        return static_cast<std::uint32_t>(leaves_.size() - 1);
    }
    const auto index = free_leaves_.back();
    free_leaves_.pop_back();
    leaves_[index].count = 0;
    return index;
}

std::uint32_t ResourceTimeline::allocate_group() {
    if (free_groups_.empty()) {
        groups_.emplace_back();
        return static_cast<std::uint32_t>(groups_.size() - 1);
    }
    const auto index = free_groups_.back();
    free_groups_.pop_back();
    groups_[index].count = 0;
    return index;
}

bool ResourceTimeline::cannot_hold(double max_duration_ns, double max_end_ns, double duration_ns) {
    // A gap can satisfy causal_finish(begin, duration) <= end while end - begin
    // falls short of the duration by less than one rounding step of end. Two
    // ulps of the largest end in the range cover that and the subtraction's
    // own rounding. The inexpensive epsilon bound is at least two ulps,
    // so no admissible gap is skipped. For integer cycles
    // the slack is below one and never changes the outcome.
    const double slack = 4.0 * std::numeric_limits<double>::epsilon() * max_end_ns +
        2.0 * std::numeric_limits<double>::denorm_min();
    return max_duration_ns + slack < duration_ns;
}

void ResourceTimeline::refresh_leaf_row(Group& group, std::uint32_t slot) {
    Leaf& leaf = leaves_[group.leaves[slot]];
    double longest = 0.0;
    leaf.max_duration_count = 0;
    for (std::uint32_t i = 0; i < leaf.count; ++i) {
        const double duration = duration_of(leaf.gaps[i]);
        if (duration > longest) {
            longest = duration;
            leaf.max_duration_count = 1;
        } else if (duration == longest) {
            ++leaf.max_duration_count;
        }
    }
    group.first_keys[slot] = leaf.gaps[0].begin_ns;
    group.max_durations[slot] = longest;
    group.max_ends[slot] = leaf.gaps[leaf.count - 1].end_ns;
}

void ResourceTimeline::refresh_group_summary(std::size_t position) {
    Group& group = group_at(position);
    double longest = 0.0;
    group.max_duration_count = 0;
    for (std::uint32_t l = 0; l < group.count; ++l) {
        const double duration = group.max_durations[l];
        if (duration > longest) {
            longest = duration;
            group.max_duration_count = 1;
        } else if (duration == longest) {
            ++group.max_duration_count;
        }
    }
    first_keys_[position] = group.first_keys[0];
    max_durations_[position] = longest;
    max_ends_[position] = group.max_ends[group.count - 1];
}

void ResourceTimeline::note_leaf_max_changed(Position p, double old_max) {
    Group& group = group_at(p.group);
    const double new_max = group.max_durations[p.leaf];
    if (old_max == new_max) return;
    double& maximum = max_durations_[p.group];
    if (new_max > maximum) {
        maximum = new_max;
        group.max_duration_count = 1;
    } else if (old_max == maximum) {
        // Only losing the final winning row requires a new maximum search.
        if (--group.max_duration_count == 0) refresh_group_summary(p.group);
    } else if (new_max == maximum) {
        ++group.max_duration_count;
    }
}

void ResourceTimeline::note_inserted(Position p, const Gap& gap) {
    Group& group = group_at(p.group);
    Leaf& leaf = leaves_[group.leaves[p.leaf]];
    const double duration = duration_of(gap);
    const double old_max = group.max_durations[p.leaf];
    if (duration > old_max) {
        group.max_durations[p.leaf] = duration;
        leaf.max_duration_count = 1;
    } else if (duration == old_max) {
        ++leaf.max_duration_count;
    }
    group.first_keys[p.leaf] = leaf.gaps[0].begin_ns;
    group.max_ends[p.leaf] = leaf.gaps[leaf.count - 1].end_ns;
    first_keys_[p.group] = group.first_keys[0];
    max_ends_[p.group] = group.max_ends[group.count - 1];
    note_leaf_max_changed(p, old_max);
}

void ResourceTimeline::note_replaced(Position p, const Gap& old, const Gap& gap) {
    Group& group = group_at(p.group);
    Leaf& leaf = leaves_[group.leaves[p.leaf]];
    if (p.slot == 0) {
        group.first_keys[p.leaf] = gap.begin_ns;
        if (p.leaf == 0) {
            first_keys_[p.group] = gap.begin_ns;
        }
    }
    group.max_ends[p.leaf] = leaf.gaps[leaf.count - 1].end_ns;
    max_ends_[p.group] = group.max_ends[group.count - 1];
    const double old_leaf_max = group.max_durations[p.leaf];
    const double old_duration = duration_of(old), duration = duration_of(gap);
    if (duration > old_leaf_max) {
        group.max_durations[p.leaf] = duration;
        leaf.max_duration_count = 1;
    } else if (old_duration == old_leaf_max && duration < old_leaf_max) {
        if (--leaf.max_duration_count == 0) refresh_leaf_row(group, p.leaf);
    } else if (duration == old_leaf_max && old_duration < old_leaf_max) {
        ++leaf.max_duration_count;
    }
    note_leaf_max_changed(p, old_leaf_max);
}

void ResourceTimeline::note_erased(Position p, const Gap& old) {
    Group& group = group_at(p.group);
    Leaf& leaf = leaves_[group.leaves[p.leaf]];
    if (p.slot == 0) {
        group.first_keys[p.leaf] = leaf.gaps[0].begin_ns;
        if (p.leaf == 0) {
            first_keys_[p.group] = leaf.gaps[0].begin_ns;
        }
    }
    group.max_ends[p.leaf] = leaf.gaps[leaf.count - 1].end_ns;
    max_ends_[p.group] = group.max_ends[group.count - 1];
    const double old_leaf_max = group.max_durations[p.leaf];
    if (duration_of(old) != old_leaf_max) {
        return;
    }
    if (--leaf.max_duration_count != 0) return;
    refresh_leaf_row(group, p.leaf);
    note_leaf_max_changed(p, old_leaf_max);
}

void ResourceTimeline::compact() {
    if (head_ == 0) {
        return;
    }
    const auto dead = static_cast<std::ptrdiff_t>(head_);
    order_.erase(order_.begin(), order_.begin() + dead);
    first_keys_.erase(first_keys_.begin(), first_keys_.begin() + dead);
    max_durations_.erase(max_durations_.begin(), max_durations_.begin() + dead);
    max_ends_.erase(max_ends_.begin(), max_ends_.begin() + dead);
    head_ = 0;
}

void ResourceTimeline::split_group(Position& p) {
    constexpr std::uint32_t half = kGroupLeaves / 2;
    const auto index = allocate_group();
    Group& lower = group_at(p.group);
    Group& upper = groups_[index];
    std::copy(lower.leaves + half, lower.leaves + kGroupLeaves, upper.leaves);
    std::copy(lower.first_keys + half, lower.first_keys + kGroupLeaves, upper.first_keys);
    std::copy(lower.max_durations + half, lower.max_durations + kGroupLeaves, upper.max_durations);
    std::copy(lower.max_ends + half, lower.max_ends + kGroupLeaves, upper.max_ends);
    lower.count = half;
    upper.count = half;
    const auto at = static_cast<std::ptrdiff_t>(p.group) + 1;
    order_.insert(order_.begin() + at, index);
    first_keys_.insert(first_keys_.begin() + at, 0.0);
    max_durations_.insert(max_durations_.begin() + at, 0.0);
    max_ends_.insert(max_ends_.begin() + at, 0.0);
    refresh_group_summary(p.group);
    refresh_group_summary(p.group + 1);
    if (p.leaf >= half) {
        p.group += 1;
        p.leaf -= half;
    }
}

void ResourceTimeline::split_leaf(Position& p) {
    if (group_at(p.group).count == kGroupLeaves) {
        split_group(p);
    }
    constexpr std::uint32_t half = kLeafGaps / 2;
    const auto index = allocate_leaf();
    Group& group = group_at(p.group);
    Leaf& lower = leaves_[group.leaves[p.leaf]];
    Leaf& upper = leaves_[index];
    std::copy(lower.gaps + half, lower.gaps + kLeafGaps, upper.gaps);
    lower.count = half;
    upper.count = half;
    const std::uint32_t at = p.leaf + 1;
    shift_up(group.leaves, at, group.count);
    shift_up(group.first_keys, at, group.count);
    shift_up(group.max_durations, at, group.count);
    shift_up(group.max_ends, at, group.count);
    group.leaves[at] = index;
    ++group.count;
    // The maximum is unchanged, but splitting a winning row can add a tie.
    refresh_leaf_row(group, p.leaf);
    refresh_leaf_row(group, at);
    if (group.max_durations[p.leaf] == max_durations_[p.group] &&
        group.max_durations[at] == max_durations_[p.group])
        ++group.max_duration_count;
    if (p.slot >= half) {
        p.leaf = at;
        p.slot -= half;
    }
}

void ResourceTimeline::insert_at(Position p, const Gap& gap) {
    if (head_ == order_.size()) {
        compact();
        const auto group_index = allocate_group();
        const auto leaf_index = allocate_leaf();
        Leaf& leaf = leaves_[leaf_index];
        leaf.gaps[0] = gap;
        leaf.count = 1;
        Group& group = groups_[group_index];
        group.leaves[0] = leaf_index;
        group.count = 1;
        group.max_duration_count = 1;
        refresh_leaf_row(group, 0);
        order_.push_back(group_index);
        first_keys_.push_back(gap.begin_ns);
        max_durations_.push_back(duration_of(gap));
        max_ends_.push_back(gap.end_ns);
        return;
    }
    if (at_end(p)) {
        p.group = order_.size() - 1;
        p.leaf = group_at(p.group).count - 1;
        p.slot = leaf_at(p).count;
    }
    if (leaf_at(p).count == kLeafGaps) {
        split_leaf(p);
    }
    Leaf& leaf = leaf_at(p);
    shift_up(leaf.gaps, p.slot, leaf.count);
    leaf.gaps[p.slot] = gap;
    ++leaf.count;
    note_inserted(p, gap);
}

void ResourceTimeline::remove_leaf(Position p) {
    Group& group = group_at(p.group);
    const double removed_max = group.max_durations[p.leaf];
    free_leaves_.push_back(group.leaves[p.leaf]);
    if (group.count == 1) {
        free_groups_.push_back(order_[p.group]);
        const auto at = static_cast<std::ptrdiff_t>(p.group);
        order_.erase(order_.begin() + at);
        first_keys_.erase(first_keys_.begin() + at);
        max_durations_.erase(max_durations_.begin() + at);
        max_ends_.erase(max_ends_.begin() + at);
        if (head_ == order_.size()) {
            compact();
        }
        return;
    }
    shift_down(group.leaves, p.leaf + 1, group.count, 1);
    shift_down(group.first_keys, p.leaf + 1, group.count, 1);
    shift_down(group.max_durations, p.leaf + 1, group.count, 1);
    shift_down(group.max_ends, p.leaf + 1, group.count, 1);
    --group.count;
    first_keys_[p.group] = group.first_keys[0];
    max_ends_[p.group] = group.max_ends[group.count - 1];
    if (removed_max == max_durations_[p.group] && --group.max_duration_count == 0)
        refresh_group_summary(p.group);
}

void ResourceTimeline::erase_at(Position p) {
    Leaf& leaf = leaf_at(p);
    const Gap old = leaf.gaps[p.slot];
    shift_down(leaf.gaps, p.slot + 1, leaf.count, 1);
    --leaf.count;
    if (leaf.count == 0) {
        remove_leaf(p);
        return;
    }
    note_erased(p, old);
}

void ResourceTimeline::replace_at(Position p, const Gap& gap) {
    Leaf& leaf = leaf_at(p);
    const Gap old = leaf.gaps[p.slot];
    leaf.gaps[p.slot] = gap;
    note_replaced(p, old, gap);
}

void ResourceTimeline::drop_before(Position p) {
    for (std::size_t position = head_; position < p.group; ++position) {
        const Group& group = group_at(position);
        for (std::uint32_t l = 0; l < group.count; ++l) {
            free_leaves_.push_back(group.leaves[l]);
        }
        free_groups_.push_back(order_[position]);
    }
    head_ = p.group;
    if (at_end(p)) {
        compact();
        return;
    }
    Group& group = group_at(p.group);
    if (p.leaf > 0) {
        for (std::uint32_t l = 0; l < p.leaf; ++l) {
            free_leaves_.push_back(group.leaves[l]);
        }
        shift_down(group.leaves, p.leaf, group.count, p.leaf);
        shift_down(group.first_keys, p.leaf, group.count, p.leaf);
        shift_down(group.max_durations, p.leaf, group.count, p.leaf);
        shift_down(group.max_ends, p.leaf, group.count, p.leaf);
        group.count -= p.leaf;
    }
    if (p.slot > 0) {
        Leaf& leaf = leaves_[group.leaves[0]];
        shift_down(leaf.gaps, p.slot, leaf.count, p.slot);
        leaf.count -= p.slot;
        refresh_leaf_row(group, 0);
    }
    refresh_group_summary(p.group);
    if (head_ > order_.size() / 2) {
        compact();
    }
}

std::size_t ResourceTimeline::gap_count() const {
    if (inline_mode_) return inline_count_;
    std::size_t count = 0;
    for (std::size_t position = head_; position < order_.size(); ++position) {
        const Group& group = group_at(position);
        for (std::uint32_t l = 0; l < group.count; ++l) {
            count += leaves_[group.leaves[l]].count;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

void ResourceTimeline::invalidate_ordered_runs() const {
    for (auto& run : ordered_runs_) run.valid = false;
}

void ResourceTimeline::observe_frontier() const {
    if (ready_ns != observed_frontier_ns_) {
        invalidate_ordered_runs();
        observed_frontier_ns_ = ready_ns;
    }
}

bool ResourceTimeline::ordered_gaps_within_frontier() const {
    if (inline_mode_) {
        if (!inline_count_) return true;
        const auto& last = inline_gaps_[inline_count_ - 1];
        return std::max(last.begin_ns, last.end_ns) <= ready_ns;
    }
    if (head_ == order_.size()) return true;
    const auto& group = group_at(order_.size() - 1);
    const auto& leaf = leaves_[group.leaves[group.count - 1]];
    const auto& last = leaf.gaps[leaf.count - 1];
    // Include begin as well: public consume_gap historically accepts wider
    // caller bounds, which can leave a malformed inline tail before promotion.
    return std::max(last.begin_ns, last.end_ns) <= ready_ns;
}

std::optional<ResourceTimeline::Found> ResourceTimeline::ordered_position_gap(
    Position hint, double earliest_ns, double duration_ns) const {
    const auto valid = [&](Position p) {
        if (p.group < head_ || p.group >= order_.size()) return false;
        const auto& group = group_at(p.group);
        return p.leaf < group.count && p.slot < leaves_[group.leaves[p.leaf]].count;
    };
    if (!valid(hint)) return std::nullopt;
    const auto fit = [&](Position p) -> std::optional<Found> {
        const auto& gap = gap_at(p);
        if (gap.begin_ns <= earliest_ns) {
            if (gap.end_ns <= earliest_ns) return std::nullopt;
        } else {
            // Merely fitting is insufficient: this must be the first gap
            // after earliest, with no still-usable predecessor. previous()
            // crosses leaf/group boundaries, including a candidate at slot 0.
            auto preceding = p;
            if (previous(preceding) && gap_at(preceding).end_ns > earliest_ns)
                return std::nullopt;
        }
        const auto start = std::max(earliest_ns, gap.begin_ns);
        if (causal_finish(start, duration_ns) <= gap.end_ns) return Found{gap, p};
        return std::nullopt;
    };
    if (const auto found = fit(hint)) return found;
    const auto following = next(hint);
    if (valid(following)) {
        if (const auto found = fit(following)) return found;
    }
    auto preceding = hint;
    if (previous(preceding)) return fit(preceding);
    return std::nullopt;
}

std::optional<ResourceTimeline::Found> ResourceTimeline::search(double earliest_ns, double duration_ns) const {
    const Position after = upper_bound(earliest_ns);
    Position containing = after;
    if (previous(containing)) {
        const Gap& gap = gap_at(containing);
        if (causal_finish(earliest_ns, duration_ns) <= gap.end_ns) {
            return Found{gap, containing};
        }
    }
    if (at_end(after)) {
        return std::nullopt;
    }
    // A gap beginning exactly at earliest_ns was just rejected by the same
    // test, so the scan starts at the first gap beginning after it. Leaves
    // and groups whose longest gap cannot hold the request are skipped.
    const auto scan_leaf = [&](std::size_t g, std::uint32_t l, std::uint32_t from)
        -> std::optional<Found> {
        const Leaf& leaf = leaves_[group_at(g).leaves[l]];
        for (std::uint32_t slot = from; slot < leaf.count; ++slot) {
            const Gap& gap = leaf.gaps[slot];
            if (causal_finish(gap.begin_ns, duration_ns) <= gap.end_ns) {
                return Found{gap, Position{g, l, slot}};
            }
        }
        return std::nullopt;
    };
    {
        const Group& group = group_at(after.group);
        if (!cannot_hold(group.max_durations[after.leaf], group.max_ends[after.leaf], duration_ns)) {
            if (const auto found = scan_leaf(after.group, after.leaf, after.slot)) {
                return found;
            }
        }
        for (std::uint32_t l = after.leaf + 1; l < group.count; ++l) {
            if (cannot_hold(group.max_durations[l], group.max_ends[l], duration_ns)) {
                continue;
            }
            if (const auto found = scan_leaf(after.group, l, 0)) {
                return found;
            }
        }
    }
    for (std::size_t g = after.group + 1; g < order_.size(); ++g) {
        if (cannot_hold(max_durations_[g], max_ends_[g], duration_ns)) {
            continue;
        }
        const Group& group = group_at(g);
        for (std::uint32_t l = 0; l < group.count; ++l) {
            if (cannot_hold(group.max_durations[l], group.max_ends[l], duration_ns)) {
                continue;
            }
            if (const auto found = scan_leaf(g, l, 0)) {
                return found;
            }
        }
    }
    return std::nullopt;
}

std::optional<ResourceTimeline::Found> ResourceTimeline::lookup_gap(
    double earliest_ns,
    double duration_ns) const {
    if (earliest_ns >= ready_ns) {
        return std::nullopt;
    }
    if (last_gap_query_.valid && last_gap_query_.duration_ns == duration_ns &&
        last_gap_query_.frontier_ns == ready_ns) {
        if (last_gap_query_.result) {
            const auto& found = *last_gap_query_.result;
            // No earlier gap fit the previous lower bound. Moving that bound
            // towards or within this gap cannot reveal an earlier answer.
            // Bounds inside the gap also work when queries arrive out of order.
            if (earliest_ns >= std::min(last_gap_query_.earliest_ns, found.gap.begin_ns) &&
                causal_finish(std::max(earliest_ns, found.gap.begin_ns), duration_ns) <=
                    found.gap.end_ns) {
                return found;
            }
        } else if (earliest_ns >= last_gap_query_.earliest_ns) {
            return std::nullopt;
        }
    }
    auto found = search(earliest_ns, duration_ns);
    last_gap_query_ = {true, earliest_ns, duration_ns, ready_ns, found};
    return found;
}

std::optional<ResourceTimeline::Gap> ResourceTimeline::first_fitting_gap(
    double earliest_ns,
    double duration_ns) const {
    // These checks stay in Release builds: a reservation placed behind the
    // pruned watermark is a causality violation that must fail closed
    // rather than silently corrupt the schedule.
    if (!std::isfinite(earliest_ns) || !std::isfinite(duration_ns) ||
        earliest_ns < pruned_through_ns_ || duration_ns <= 0.0) {
        throw std::runtime_error(
            "physical resource calendar received an invalid reservation");
    }
    if (inline_mode_) {
        const auto found = inline_fit(earliest_ns, duration_ns);
        return found ? std::optional<Gap>{inline_gaps_[*found]} : std::nullopt;
    }
    const auto found = lookup_gap(earliest_ns, duration_ns);
    return found ? std::optional<Gap>{found->gap} : std::nullopt;
}

bool ResourceTimeline::can_reserve_exact(double begin_ns, double duration_ns) const {
    if (!std::isfinite(begin_ns) || !std::isfinite(duration_ns) ||
        begin_ns < pruned_through_ns_ || duration_ns <= 0.0 ||
        !std::isfinite(causal_finish(begin_ns, duration_ns))) {
        throw std::runtime_error(
            "physical resource calendar received an invalid exact-reservation query");
    }
    if (begin_ns >= ready_ns) {
        return true;
    }
    if (inline_mode_) {
        const auto found = inline_fit(begin_ns, duration_ns);
        return found && inline_gaps_[*found].begin_ns <= begin_ns;
    }
    Position containing = upper_bound(begin_ns);
    if (!previous(containing)) {
        return false;
    }
    const Gap& gap = gap_at(containing);
    return begin_ns >= gap.begin_ns && causal_finish(begin_ns, duration_ns) <= gap.end_ns;
}

double ResourceTimeline::preview_start(double earliest_ns, double duration_ns) const {
    if (!std::isfinite(earliest_ns) || !std::isfinite(duration_ns) ||
        earliest_ns < pruned_through_ns_ || duration_ns <= 0.0) {
        throw std::runtime_error(
            "physical resource calendar received an invalid reservation preview");
    }
    if (inline_mode_) {
        if (const auto found = inline_fit(earliest_ns, duration_ns))
            return std::max(earliest_ns, inline_gaps_[*found].begin_ns);
    } else if (const auto found = lookup_gap(earliest_ns, duration_ns)) {
        return std::max(found->gap.begin_ns, earliest_ns);
    }
    const double start = std::max(earliest_ns, ready_ns);
    if (!std::isfinite(start) ||
        !std::isfinite(causal_finish(start, duration_ns))) {
        throw std::runtime_error("physical resource calendar preview time overflowed");
    }
    return start;
}

std::vector<ResourceTimeline::Gap> ResourceTimeline::gaps_after(double floor_ns) const {
    std::vector<Gap> result;
    if (inline_mode_) {
        for (std::uint32_t i = 0; i < inline_count_; ++i)
            if (inline_gaps_[i].end_ns > floor_ns)
                result.push_back({std::max(floor_ns, inline_gaps_[i].begin_ns), inline_gaps_[i].end_ns});
        return result;
    }
    for (Position p = first_after_floor(floor_ns); !at_end(p); p = next(p)) {
        const Gap& gap = gap_at(p);
        result.push_back({std::max(floor_ns, gap.begin_ns), gap.end_ns});
    }
    return result;
}

void ResourceTimeline::insert_gap(double begin_ns, double end_ns) {
    observe_frontier();
    invalidate_ordered_runs();
    last_gap_query_.valid = false;
    if (!std::isfinite(begin_ns) || !std::isfinite(end_ns) ||
        begin_ns < pruned_through_ns_ || end_ns <= begin_ns) {
        throw std::runtime_error("physical resource calendar received an invalid idle gap");
    }
    if (inline_mode_) {
        const auto at = static_cast<std::uint32_t>(std::lower_bound(
            inline_gaps_, inline_gaps_ + inline_count_, begin_ns,
            [](const Gap& gap, double begin) { return gap.begin_ns < begin; }) - inline_gaps_);
        if ((at && inline_gaps_[at - 1].end_ns > begin_ns) ||
            (at < inline_count_ && inline_gaps_[at].begin_ns < end_ns))
            throw std::runtime_error("physical resource calendar inserted overlapping idle gaps");
        if (inline_count_ < kInlineGaps) {
            shift_up(inline_gaps_, at, inline_count_);
            inline_gaps_[at] = {begin_ns, end_ns};
            ++inline_count_;
            return;
        }
        promote_inline();
    }
    const Position after = lower_bound(begin_ns);
    Position before = after;
    if (previous(before) && gap_at(before).end_ns > begin_ns) {
        throw std::runtime_error("physical resource calendar inserted overlapping idle gaps");
    }
    if (!at_end(after) && gap_at(after).begin_ns < end_ns) {
        throw std::runtime_error("physical resource calendar inserted overlapping idle gaps");
    }
    insert_at(after, Gap{begin_ns, end_ns});
}

void ResourceTimeline::insert_frontier_gap(double start_ns) {
    observe_frontier();
    if (inline_mode_) {
        if (!std::isfinite(ready_ns) || !std::isfinite(start_ns) ||
            ready_ns < pruned_through_ns_ || start_ns <= ready_ns)
            throw std::runtime_error("physical resource calendar received an invalid idle gap");
        if (inline_count_ && inline_gaps_[inline_count_ - 1].end_ns > ready_ns) {
            insert_gap(ready_ns, start_ns);
            return;
        }
        if (inline_count_ < kInlineGaps) {
            inline_gaps_[inline_count_++] = {ready_ns, start_ns};
            return;
        }
        promote_inline();
    }
    if (head_ != order_.size() && max_ends_.back() > ready_ns) {
        // Direct insert_gap callers may hold gaps beyond the frontier.
        insert_gap(ready_ns, start_ns);
        return;
    }
    last_gap_query_.valid = false;
    if (!std::isfinite(ready_ns) || !std::isfinite(start_ns) ||
        ready_ns < pruned_through_ns_ || start_ns <= ready_ns) {
        throw std::runtime_error("physical resource calendar received an invalid idle gap");
    }
    // All retained gaps end by ready_ns, so this is an append with no overlap
    // search. Keep the same leaf insertion and split order as insert_gap.
    insert_at(Position{order_.size(), 0, 0}, Gap{ready_ns, start_ns});
}

void ResourceTimeline::consume_gap(const Gap& gap, double begin_ns, double end_ns) {
    observe_frontier();
    // Caller-supplied bounds may extend the held gap. Internal reservations
    // consume their actual Found gap and do not take this public path.
    invalidate_ordered_runs();
    if (inline_mode_) {
        if (reservation_outside(gap, begin_ns, end_ns))
            throw std::runtime_error("physical resource calendar consumed outside an idle gap");
        std::uint32_t at = 0;
        while (at < inline_count_ && inline_gaps_[at].begin_ns != gap.begin_ns) ++at;
        if (at == inline_count_)
            throw std::runtime_error("physical resource calendar erased an idle gap it does not hold");
        if (at + 1 < inline_count_ && inline_gaps_[at + 1].begin_ns < gap.end_ns)
            throw std::runtime_error("physical resource calendar inserted overlapping idle gaps");
        consume_inline(at, gap, begin_ns, end_ns);
        return;
    }
    std::optional<Position> cached;
    if (last_gap_query_.valid && last_gap_query_.result &&
        last_gap_query_.result->gap.begin_ns == gap.begin_ns) {
        cached = last_gap_query_.result->at;
    }
    last_gap_query_.valid = false;
    if (reservation_outside(gap, begin_ns, end_ns)) {
        throw std::runtime_error(
            "physical resource calendar consumed outside an idle gap (gap=[" +
            std::to_string(gap.begin_ns) + "," +
            std::to_string(gap.end_ns) + "), request=[" +
            std::to_string(begin_ns) + "," +
            std::to_string(end_ns) + "))");
    }
    const Position held = cached ? *cached : lower_bound(gap.begin_ns);
    if (at_end(held) || gap_at(held).begin_ns != gap.begin_ns) {
        throw std::runtime_error(
            "physical resource calendar erased an idle gap it does not hold");
    }
    consume_at(held, gap, begin_ns, end_ns);
}

void ResourceTimeline::consume_at(Position held, const Gap& gap, double begin_ns, double end_ns) {
    const bool keep_head = gap.begin_ns < begin_ns;
    const bool keep_tail = end_ns < gap.end_ns;
    // The remainders are sub-intervals of the caller's gap. They can only
    // collide with the following gap when the caller's bounds exceed the
    // held gap; the treap reported that as an overlapping insertion.
    if (keep_head || keep_tail) {
        const Position following = next(held);
        if (!at_end(following) &&
            gap_at(following).begin_ns < (keep_tail ? gap.end_ns : begin_ns)) {
            throw std::runtime_error("physical resource calendar inserted overlapping idle gaps");
        }
    }
    if (keep_head && keep_tail) {
        replace_at(held, Gap{gap.begin_ns, begin_ns});
        insert_at(next(held), Gap{end_ns, gap.end_ns});
    } else if (keep_head) {
        replace_at(held, Gap{gap.begin_ns, begin_ns});
    } else if (keep_tail) {
        replace_at(held, Gap{end_ns, gap.end_ns});
    } else {
        erase_at(held);
    }
}

void ResourceTimeline::reclaim_sparse_arena() {
    const auto live = leaves_.size() - free_leaves_.size();
    if (leaves_.capacity() < 64 || live > leaves_.capacity() / 4) return;
    std::vector<Leaf> leaves;
    std::vector<Group> groups;
    const auto count = order_.size() - head_;
    if (live) leaves.reserve(std::max<std::size_t>(8, live * 2));
    groups.reserve(count);
    for (auto i = head_; i < order_.size(); ++i) {
        auto group = group_at(i);
        for (std::uint32_t j = 0; j < group.count; ++j) {
            const auto old = group.leaves[j];
            group.leaves[j] = static_cast<std::uint32_t>(leaves.size());
            leaves.push_back(leaves_[old]);
        }
        order_[i] = static_cast<std::uint32_t>(groups.size());
        groups.push_back(group);
    }
    leaves_.swap(leaves);
    groups_.swap(groups);
    std::vector<std::uint32_t>().swap(free_leaves_);
    std::vector<std::uint32_t>().swap(free_groups_);
    // Discard dead summary rows and their old capacity as well.
    const auto trim = [this](auto& rows) {
        using Rows = std::decay_t<decltype(rows)>;
        Rows(rows.begin() + head_, rows.end()).swap(rows);
    };
    trim(order_);
    trim(first_keys_);
    trim(max_durations_);
    trim(max_ends_);
    head_ = 0;
}

void ResourceTimeline::prune_before(double causal_watermark_ns) {
    observe_frontier();
    if (!std::isfinite(causal_watermark_ns) || causal_watermark_ns < 0.0) {
        throw std::runtime_error("physical resource calendar received an invalid causal watermark");
    }
    if (causal_watermark_ns <= pruned_through_ns_) {
        return;
    }
    invalidate_ordered_runs();
    last_gap_query_.valid = false;
    pruned_through_ns_ = causal_watermark_ns;
    ready_ns = std::max(ready_ns, causal_watermark_ns);
    observed_frontier_ns_ = ready_ns;
    if (inline_mode_) {
        std::uint32_t expired = 0;
        while (expired < inline_count_ && inline_gaps_[expired].end_ns <= causal_watermark_ns) ++expired;
        if (expired) {
            std::move(inline_gaps_ + expired, inline_gaps_ + inline_count_, inline_gaps_);
            inline_count_ -= expired;
        }
        if (inline_count_) inline_gaps_[0].begin_ns = std::max(inline_gaps_[0].begin_ns, causal_watermark_ns);
        return;
    }
    if (head_ == order_.size() || causal_watermark_ns <= first_keys_[head_]) {
        // No interval crosses or precedes the watermark. Avoid the lower
        // bound and unchanged group-summary refresh in drop_before.
        reclaim_sparse_arena();
        demote_if_small();
        return;
    }
    if (causal_watermark_ns >= max_ends_.back() && leaves_.capacity() >= 64) {
        // The sparse-arena policy would release this entire arena anyway.
        // Skip visiting every leaf just to fill free lists that will be freed.
        leaves_.clear();
        free_leaves_.clear();
        groups_.clear();
        free_groups_.clear();
        order_.clear();
        first_keys_.clear();
        max_durations_.clear();
        max_ends_.clear();
        head_ = 0;
        reclaim_sparse_arena();
        demote_if_small();
        return;
    }
    // Gaps beginning before the watermark expire. Only the last of them can
    // cross the watermark; its usable suffix is kept exactly.
    const Position future = lower_bound(causal_watermark_ns);
    Position last_expired = future;
    std::optional<double> crossing_end_ns;
    if (previous(last_expired) && gap_at(last_expired).end_ns > causal_watermark_ns) {
        crossing_end_ns = gap_at(last_expired).end_ns;
    }
    drop_before(crossing_end_ns ? last_expired : future);
    if (crossing_end_ns) {
        replace_at(Position{head_, 0, 0}, Gap{causal_watermark_ns, *crossing_end_ns});
    }
    reclaim_sparse_arena();
    demote_if_small();
}

ResourceTimeline::Reservation ResourceTimeline::reserve(double earliest_ns, double duration_ns) {
    return reserve_impl<false>(earliest_ns, duration_ns, nullptr);
}

ResourceTimeline::Reservation ResourceTimeline::reserve_ordered(
    double earliest_ns, double duration_ns, unsigned stream) {
    if (stream >= 2) throw std::out_of_range("physical resource ordered stream is out of range");
    return reserve_impl<true>(earliest_ns, duration_ns, &ordered_runs_[stream]);
}

template <bool Ordered>
ResourceTimeline::Reservation ResourceTimeline::reserve_impl(
    double earliest_ns, double duration_ns, OrderedRun* ordered) {
    observe_frontier();
    if (!std::isfinite(earliest_ns) || !std::isfinite(duration_ns) ||
        earliest_ns < pruned_through_ns_ || duration_ns <= 0.0) {
        throw std::runtime_error(
            "physical resource calendar received an invalid reservation preview");
    }
    if (!std::isfinite(reserved_work_ns + duration_ns))
        throw std::runtime_error("physical resource reservation work overflowed");
    // A same-duration first-fit reservation [S,F) proves that no still-idle
    // capacity can serve that stream before F at any later offered time.
    // Other reservations only consume capacity; appended idle gaps begin at
    // the old frontier, at or beyond F. Public growth/frontier edits invalidate
    // the proof. Future gaps break that frontier invariant and never qualify.
    bool can_record = false;
    double search_ns = earliest_ns;
    if constexpr (Ordered) {
        key_leaf_hint_ = ordered->navigation;
        can_record = ordered_gaps_within_frontier();
        if (can_record && ordered->valid && ordered->duration_ns == duration_ns &&
            earliest_ns >= ordered->earliest_ns && ready_ns >= ordered->finish_ns) {
            search_ns = std::max(earliest_ns, ordered->finish_ns);
        }
    }
    const auto complete = [&](double start, double finish) {
        observed_frontier_ns_ = ready_ns;
        if constexpr (Ordered) {
            if (search_ns > earliest_ns) ++ordered_reservation_reuses_;
            *ordered = {can_record, earliest_ns, finish, duration_ns, key_leaf_hint_};
        }
        return Reservation{start, finish};
    };
    if (inline_mode_) {
        const auto found = inline_fit(search_ns, duration_ns);
        const auto start = found ? std::max(search_ns, inline_gaps_[*found].begin_ns) :
            std::max(earliest_ns, ready_ns);
        const auto finish = causal_finish(start, duration_ns);
        if (found) {
            const auto gap = inline_gaps_[*found];
            consume_inline(*found, gap, start, finish);
        } else {
            if (start > ready_ns) insert_frontier_gap(start);
            ready_ns = finish;
        }
        reserved_work_ns += duration_ns;
        return complete(start, finish);
    }
    // Same lookup as first_fitting_gap, keeping the located position so the
    // consumption below does not search for the gap a second time.
    std::optional<Found> found;
    if constexpr (Ordered) {
        // Future gaps or a lookup at/after the frontier must keep the original
        // lookup early-exit semantics. A valid candidate proves first-fit from
        // current gap bounds, independently of the stream's time certificate.
        if (can_record && search_ns < ready_ns) {
            found = ordered_position_gap(ordered->navigation, search_ns, duration_ns);
            if (found) ++ordered_position_reuses_;
        }
    }
    if (!found) found = lookup_gap(search_ns, duration_ns);
    double start = 0.0;
    if (found) {
        start = std::max(found->gap.begin_ns, search_ns);
    } else {
        start = std::max(earliest_ns, ready_ns);
        if (!std::isfinite(start)) {
            throw std::runtime_error("physical resource calendar preview time overflowed");
        }
    }
    const double finish = causal_finish(start, duration_ns);
    if (found) {
        last_gap_query_.valid = false;
        consume_at(found->at, found->gap, start, finish);
        if constexpr (Ordered) key_leaf_hint_ = found->at;
    } else {
        if (start > ready_ns) {
            insert_frontier_gap(start);
        }
        ready_ns = finish;
    }
    reserved_work_ns += duration_ns;
    return complete(start, finish);
}

}
