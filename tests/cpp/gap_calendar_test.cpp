// Independent interval oracle for fractional HBF and integer HBM calendars.
#include "physical/resource_calendar.hpp"
#include <iostream>
#include <random>
#include <stdexcept>
using namespace hbfsim::physical;
namespace {
void check(bool b, const char* why) { if (!b) throw std::runtime_error(why); }
struct Oracle {
    double ready=0, work=0;
    std::vector<ResourceTimeline::Gap> gaps;
    double preview(double at, double d) const {
        for (auto g:gaps) {
            double begin=std::max(g.begin_ns,at);
            if (causal_finish(begin,d)<=g.end_ns) return begin;
        }
        return std::max(at,ready);
    }
    ResourceTimeline::Reservation reserve(double at,double d) {
        double begin=preview(at,d),end=causal_finish(begin,d);
        for (auto i=gaps.begin();i!=gaps.end();++i) {
            if (i->begin_ns<=begin && end<=i->end_ns) {
                auto g=*i; i=gaps.erase(i);
                if(end<g.end_ns) i=gaps.insert(i,{end,g.end_ns});
                if(g.begin_ns<begin) gaps.insert(i,{g.begin_ns,begin});
                work+=d; return {begin,end};
            }
        }
        if(ready<begin) gaps.push_back({ready,begin});
        ready=end;work+=d;return {begin,end};
    }
    void prune(double floor) {
        auto i=gaps.begin();while(i!=gaps.end() && i->end_ns<=floor) ++i;
        gaps.erase(gaps.begin(),i);
        if(!gaps.empty()) gaps.front().begin_ns=std::max(floor,gaps.front().begin_ns);
        ready=std::max(ready,floor);
    }
};
void replay(double scale,double now) {
    ResourceTimeline c;Oracle o;std::mt19937_64 random(921);
    c.prune_before(now);o.prune(now);
    for(unsigned step=0;step<30000;++step) {
        double at=now+double(random()%2000000)*scale,d=double(1+random()%100)*scale;
        check(c.preview_start(at,d)==o.preview(at,d),"preview differs from first-fit oracle");
        check(c.reserve(at,d)==o.reserve(at,d),"reservation differs from oracle");
        if(step%257==0) {
            now+=double(random()%10000)*scale;c.prune_before(now);o.prune(now);
            check(c.ready_ns==o.ready && c.reserved_work_ns==o.work,"accounting differs");
            check(c.gaps_after(now)==o.gaps,"pruning changed live capacity");
            auto copy=c;check(copy.gaps_after(now)==c.gaps_after(now),"copy differs");
        }
    }
    c.prune_before(c.ready_ns+1);check(c.gap_count()==0,"expired gaps remain");
    bool rejected=false;try{(void)c.reserve(0,1);}catch(const std::runtime_error&){rejected=true;}
    check(rejected,"reservation behind frontier accepted");
}
void rounding_boundary() {
    ResourceTimeline c;
    double begin=1e8+0.1,d=0.1,end=causal_finish(begin,d);
    c.ready_ns=end+10;c.insert_gap(begin,end);
    check(c.reserve(begin-1,d)==ResourceTimeline::Reservation{begin,end},
        "floating summary discarded a fitting gap or changed its finish");
    check(c.gap_count()==0,"consumed gap remains");
}
void interleaved_service_streams() {
    ResourceTimeline c; Oracle o;
    // Long data transfers and short commands share a resource; command
    // arrivals lag behind its booked data frontier for many read rounds.
    for (unsigned i = 0; i < 6000; ++i) {
        const double command = (i / 256) * 4000.0 + (i % 256) / 13.0;
        for (const auto& [at, duration] : {std::pair{command, 64.0 / 96},
                 std::pair{command + 4000, 4224.0 / 96}}) {
            check(c.preview_start(at, duration) == o.preview(at, duration), "stream preview differs");
            check(c.reserve(at, duration) == o.reserve(at, duration), "stream reservation differs");
        }
        if (i % 257 == 256) {
            // An out-of-order GC/metadata arrival must search before a cursor.
            check(c.reserve(0, 64.0 / 96) == o.reserve(0, 64.0 / 96), "backfill skipped capacity");
        }
    }
    check(c.gaps_after(0) == o.gaps, "streams lost idle intervals");
    ResourceTimeline inserted;
    inserted.ready_ns = 100;
    (void)inserted.preview_start(0, 1);
    inserted.insert_gap(10, 20);
    check(inserted.reserve(0, 1) == ResourceTimeline::Reservation{10, 11},
        "new capacity did not invalidate stream bound");
}
void inline_promotion_and_pruning() {
    ResourceTimeline c; Oracle o;
    for (unsigned round = 0; round < 40; ++round) {
        const double base = round * 10000.0;
        for (unsigned i = 0; i < 80; ++i)
            check(c.reserve(base + 10 * i + 3, 0.25) == o.reserve(base + 10 * i + 3, 0.25),
                "inline promotion changes reservations");
        const double floor = base + 750;
        c.prune_before(floor); o.prune(floor);
        check(c.gaps_after(floor) == o.gaps, "inline demotion changes gaps");
        auto copied = c;
        for (unsigned i = 0; i < 50; ++i) {
            const double at = floor + (i % 5) * 0.1;
            const auto expected = o.reserve(at, 0.3);
            check(c.reserve(at, 0.3) == expected && copied.reserve(at, 0.3) == expected,
                "split/promotion after copy changes reservations");
        }
        c.prune_before(base + 9000); o.prune(base + 9000);
        check(c.gaps_after(base + 9000) == o.gaps, "full pruning changes inline state");
    }
}

void tied_maxima_split_prune_copy() {
    ResourceTimeline c; Oracle o;
    // More than 64 leaves forces both leaf and group splits. Every leaf has
    // several equal winners, and many leaves share the group maximum.
    constexpr unsigned count = 6000;
    for (unsigned i = 0; i < count; ++i) {
        const double begin = i * 32.0 + 4;
        const double duration = i % 4 < 2 ? 16 : 8;
        c.insert_gap(begin, begin + duration);
        o.gaps.push_back({begin, begin + duration});
    }
    c.ready_ns = o.ready = count * 32.0 + 32;
    // Alternate distant key ranges and exact/adjacent floating boundaries
    // before the mutations below. Navigation locality must not skip gaps.
    for (const unsigned i : {4096U, 63U, 2048U, 0U, count - 1, 64U, 4095U, 127U, 2047U}) {
        const double begin = i * 32.0 + 4;
        for (const double at : {std::nextafter(begin, 0.0), begin,
                 std::nextafter(begin, std::numeric_limits<double>::infinity())}) {
            check(c.preview_start(at, 0.25) == o.preview(at, 0.25),
                "out-of-order boundary preview differs");
        }
    }
    auto copied = c;
    const auto reserve = [&](double at, double duration) {
        const double expected_start = o.preview(at, duration);
        check(c.preview_start(at, duration) == expected_start &&
            copied.preview_start(at, duration) == expected_start, "tied maximum preview differs");
        const auto expected = o.reserve(at, duration);
        check(expected.start_ns == expected_start && c.reserve(at, duration) == expected &&
            copied.reserve(at, duration) == expected, "tied maximum reservation differs");
    };
    // Delete one winner, then split its tied neighbour into smaller gaps.
    // Ties survive some updates and disappear entirely from other leaves.
    for (unsigned i = 0; i < count / 2; i += 4) {
        reserve(i * 32.0 + 4, 16);
        reserve((i + 1) * 32.0 + 8, 4);
    }
    check(c.gaps_after(0) == o.gaps && copied.gaps_after(0) == o.gaps,
        "shortening and deleting tied maxima changed intervals");
    // Consume every remaining 16-unit winner, including the unique last
    // winner in each leaf/group; only shorter gaps remain after this pass.
    while (o.preview(0, 16) < o.ready) reserve(0, 16);
    check(c.preview_start(0, 16) == o.ready && copied.preview_start(0, 16) == o.ready,
        "last maximum removal leaves a fitting interval");
    check(c.gaps_after(0) == o.gaps && copied.gaps_after(0) == o.gaps,
        "last maximum removal changes smaller intervals");

    // Cross a surviving eight-unit winner as whole preceding groups expire.
    const double floor = count * 24.0 + 70;
    c.prune_before(floor); copied.prune_before(floor); o.prune(floor);
    check(c.gaps_after(floor) == o.gaps && copied.gaps_after(floor) == o.gaps,
        "pruning tied maxima changes intervals");
    copied = c;
    // Reuse pruned storage and split rows again, with a new larger maximum.
    for (unsigned i = 0; i < 2300; ++i) {
        const double at = count * 32.0 + 64 + i * 32.0;
        reserve(at, 4);
    }
    // Erase all remaining intervals to cover empty leaf/group removal,
    // arena recycling, and copies whose summaries evolved independently.
    while (!o.gaps.empty()) {
        const auto gap = o.gaps.front();
        reserve(gap.begin_ns, gap.end_ns - gap.begin_ns);
    }
    check(c.gap_count() == 0 && copied.gap_count() == 0,
        "erasing tied maxima leaves stale intervals");
    check(c.ready_ns == o.ready && copied.ready_ns == o.ready &&
        c.reserved_work_ns == o.work && copied.reserved_work_ns == o.work,
        "tied maximum accounting differs");
}

void reservation_endpoints_and_failed_state() {
    // Exercise the same endpoint contract in inline and promoted storage.
    for (const unsigned gaps : {1U, 24U}) {
        ResourceTimeline c; Oracle o;
        c.ready_ns = o.ready = 1000;
        for (unsigned i = 0; i < gaps; ++i) {
            const double begin = 4 * i + 10;
            c.insert_gap(begin, begin + 1);
            o.gaps.push_back({begin, begin + 1});
        }
        const auto actual = c.reserve(0, 0.25);
        check(actual == o.reserve(0, 0.25) && actual.start_ns == 10 &&
            actual.finish_ns == 10.25 && actual.finish_ns < c.ready_ns,
            "gap fill returned the booked frontier instead of its finish");
        check(c.ready_ns == o.ready && c.reserved_work_ns == o.work &&
            c.gaps_after(0) == o.gaps, "endpoint return changed gap capacity");

        // A positive duration smaller than one ULP still advances time.
        const double large = 0x1p54, tiny = 0.25;
        check(large + tiny == large, "sub-ULP fixture rounded forward");
        const auto next = c.reserve(large, tiny);
        check(next == o.reserve(large, tiny) && next.start_ns == large &&
            next.finish_ns == std::nextafter(large, std::numeric_limits<double>::infinity()),
            "reservation result lost causal sub-ULP advancement");
        check(c.ready_ns == o.ready && c.reserved_work_ns == o.work &&
            c.gaps_after(0) == o.gaps, "sub-ULP reservation changed capacity");
    }

    const auto expect_unchanged = [](ResourceTimeline& c, double at, double duration) {
        const auto ready = c.ready_ns, work = c.reserved_work_ns;
        const auto intervals = c.gaps_after(0);
        const auto count = c.gap_count();
        bool rejected = false;
        try { (void)c.reserve(at, duration); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "overflowing reservation was accepted");
        check(c.ready_ns == ready && c.reserved_work_ns == work &&
            c.gap_count() == count && c.gaps_after(0) == intervals,
            "overflow changed resource capacity or accounting");
    };
    for (const unsigned gaps : {1U, 24U}) {
        ResourceTimeline c;
        for (unsigned i = 0; i < gaps; ++i)
            c.insert_gap(4 * i + 10, 4 * i + 11);
        const double maximum = std::numeric_limits<double>::max();
        c.ready_ns = maximum / 2;
        expect_unchanged(c, c.ready_ns, maximum); // Addition overflows.
        c.ready_ns = maximum;
        expect_unchanged(c, c.ready_ns, 1); // nextafter would become infinity.
        c.ready_ns = 1000;
        c.reserved_work_ns = maximum;
        expect_unchanged(c, 1000, maximum); // Work accounting overflows first.
    }
}

void ordered_streams_match_scalar_and_oracle() {
    ResourceTimeline ordered, scalar; Oracle oracle;
    // Start with packed gap storage, then interleave fixed-stage streams with
    // scalar metadata/GC reservations, duration changes and earlier arrivals.
    for (unsigned i = 0; i < 96; ++i) {
        const auto expected = oracle.reserve(i * 32.0 + 8, 2);
        check(ordered.reserve(i * 32.0 + 8, 2) == expected &&
            scalar.reserve(i * 32.0 + 8, 2) == expected, "ordered fixture differs");
    }
    std::mt19937_64 random(20260922);
    double offered[2] = {0, 0};
    double floor = 0;
    for (unsigned i = 0; i < 6000; ++i) {
        const unsigned stream = random() % 2;
        offered[stream] = std::max(floor, offered[stream]) + double(random() % 7) / 13;
        const double at = i % 191 == 0 ? floor : offered[stream];
        const double duration = (stream == 0 ? 64.0 : 4224.0) / 96 +
            (i % 137 == 0 ? 0.125 : 0.0);
        const auto expected = oracle.reserve(at, duration);
        check(scalar.reserve(at, duration) == expected &&
            ordered.reserve_ordered(at, duration, stream) == expected,
            "ordered reservation differs from scalar/oracle");
        if (i % 113 == 0) {
            const double backfill = floor + double(random() % 64) / 13;
            const auto normal = oracle.reserve(backfill, 0.375);
            check(ordered.reserve(backfill, 0.375) == normal &&
                scalar.reserve(backfill, 0.375) == normal,
                "intervening scalar reservation changed ordered capacity");
        }
        if (i % 997 == 996) {
            floor += 23;
            ordered.prune_before(floor); scalar.prune_before(floor); oracle.prune(floor);
            auto copy = ordered;
            auto scalar_copy = scalar;
            for (unsigned j = 0; j < 8; ++j) {
                const double copy_at = std::max(floor, offered[j % 2]);
                check(copy.reserve_ordered(copy_at, 0.5, j % 2) ==
                    scalar_copy.reserve(copy_at, 0.5), "copied ordered streams differ");
            }
            check(copy.gaps_after(floor) == scalar_copy.gaps_after(floor),
                "copied ordered stream loses gaps");
        }
        if (i % 64 == 0) {
            check(ordered.gaps_after(floor) == oracle.gaps &&
                scalar.gaps_after(floor) == oracle.gaps,
                "ordered streams lost live idle intervals");
            check(ordered.ready_ns == oracle.ready && scalar.ready_ns == oracle.ready &&
                ordered.reserved_work_ns == oracle.work && scalar.reserved_work_ns == oracle.work,
                "ordered streams changed floating accounting");
        }
    }
    check(ordered.ordered_reservation_reuses() > 100, "ordered path never advanced search bounds");
    check(ordered.gaps_after(floor) == scalar.gaps_after(floor) &&
        ordered.ready_ns == scalar.ready_ns && ordered.reserved_work_ns == scalar.reserved_work_ns,
        "ordered final state differs from scalar");
}

void ordered_capacity_and_frontier_changes() {
    for (const unsigned count : {1U, 24U}) {
        ResourceTimeline ordered, scalar;
        ordered.ready_ns = scalar.ready_ns = 1000;
        for (unsigned i = 0; i < count; ++i) {
            ordered.insert_gap(4 * i + 10, 4 * i + 11);
            scalar.insert_gap(4 * i + 10, 4 * i + 11);
        }
        check(ordered.reserve_ordered(0, 5) == scalar.reserve(0, 5), "capacity setup differs");
        // Public consume_gap can expand a caller-specified tail beyond the
        // actual held interval. It must invalidate, rather than assume shrink.
        const double last = 4 * (count - 1) + 10;
        ordered.consume_gap({last, 200}, last + 0.25, last + 0.5);
        scalar.consume_gap({last, 200}, last + 0.25, last + 0.5);
        check(ordered.reserve_ordered(0, 5) == scalar.reserve(0, 5),
            "public expanded consume skipped new earlier capacity");
        check(ordered.gaps_after(0) == scalar.gaps_after(0), "expanded consume changed gaps");
    }
    {
        ResourceTimeline ordered, scalar;
        ordered.ready_ns = scalar.ready_ns = 100;
        check(ordered.reserve_ordered(0, 1) == scalar.reserve(0, 1), "frontier setup differs");
        ordered.ready_ns = scalar.ready_ns = 10;
        ordered.insert_frontier_gap(20); scalar.insert_frontier_gap(20);
        // Merely testing the restored current frontier would miss this edit.
        ordered.ready_ns = scalar.ready_ns = 101;
        check(ordered.reserve_ordered(0, 1) == scalar.reserve(0, 1),
            "temporary public frontier change retained an unsafe proof");
        ordered.insert_gap(1, 2); scalar.insert_gap(1, 2);
        check(ordered.reserve_ordered(0, 1) == scalar.reserve(0, 1),
            "public insertion did not invalidate ordered lower bound");
    }
    {
        ResourceTimeline ordered, scalar;
        ordered.ready_ns = scalar.ready_ns = 100;
        ordered.insert_gap(10, 60); scalar.insert_gap(10, 60);
        check(ordered.reserve_ordered(0, 20) == scalar.reserve(0, 20), "prune setup differs");
        auto copy = ordered;
        auto scalar_copy = scalar;
        check(copy.reserve_ordered(0, 20) == scalar_copy.reserve(0, 20) &&
            copy.ordered_reservation_reuses() == 1, "copy did not retain its own ordered run");
        ordered.prune_before(15); scalar.prune_before(15);
        const auto work = ordered.reserved_work_ns;
        bool rejected = false;
        try { (void)ordered.reserve_ordered(0, 20); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected && ordered.reserved_work_ns == work,
            "ordered clamp hid invalid original arrival or changed work");
        check(ordered.reserve_ordered(15, 20) == scalar.reserve(15, 20),
            "pruned ordered run changed valid reservation");
    }
}

void ordered_future_gaps_and_rounding() {
    for (const unsigned count : {1U, 24U}) {
        ResourceTimeline ordered, scalar;
        ordered.ready_ns = scalar.ready_ns = 10;
        for (unsigned i = 0; i < count; ++i) {
            ordered.insert_gap(100 + 4 * i, 101 + 4 * i);
            scalar.insert_gap(100 + 4 * i, 101 + 4 * i);
        }
        for (unsigned i = 0; i < 3; ++i)
            check(ordered.reserve_ordered(10, 0.25) == scalar.reserve(10, 0.25),
                "ordered future-gap handling differs from scalar");
        check(ordered.ordered_reservation_reuses() == 0,
            "ordered stream certified gaps beyond the public frontier");
    }
    {
        ResourceTimeline ordered, scalar;
        ordered.ready_ns = scalar.ready_ns = 1;
        ordered.insert_gap(2, 1e20); scalar.insert_gap(2, 1e20);
        // The first tail advances ready beyond a preexisting future gap.
        // Eligibility must be checked before reservation, not only afterward.
        for (unsigned i = 0; i < 2; ++i)
            check(ordered.reserve_ordered(1, 1e20) == scalar.reserve(1, 1e20),
                "ordered path certified a future gap after moving its frontier");
    }
    {
        ResourceTimeline ordered, scalar;
        const double large = 0x1p54;
        for (unsigned i = 0; i < 8; ++i)
            check(ordered.reserve_ordered(large, 0.25) == scalar.reserve(large, 0.25),
                "ordered recurrence changed sub-ULP causal advancement");
        check(ordered.ready_ns == scalar.ready_ns &&
            ordered.reserved_work_ns == scalar.reserved_work_ns,
            "ordered rounding changed frontier or work");
        const auto maximum = std::numeric_limits<double>::max();
        ordered.ready_ns = scalar.ready_ns = maximum;
        const auto work = ordered.reserved_work_ns;
        bool rejected = false;
        try { (void)ordered.reserve_ordered(maximum, 0.25); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected && ordered.ready_ns == maximum && ordered.reserved_work_ns == work,
            "ordered overflow changed resource state");
    }
}

void ordered_slot_hints_survive_shrink_and_split() {
    ResourceTimeline ordered, scalar; Oracle oracle;
    for (unsigned i = 0; i < 128; ++i) {
        const double begin = i * 256.0 + 16;
        ordered.insert_gap(begin, begin + 112);
        scalar.insert_gap(begin, begin + 112);
        oracle.gaps.push_back({begin, begin + 112});
    }
    ordered.ready_ns = scalar.ready_ns = oracle.ready = 128 * 256.0 + 32;
    for (unsigned step = 0; step < 80; ++step) {
        if (step == 24) {
            // Insert before and around both repeatedly consumed gaps. These
            // additions shift slots and split packed leaves while stream
            // hints still name their previous numeric positions.
            for (unsigned i = 60; i < 116; ++i) {
                const double begin = i * 256.0 + 144;
                ordered.insert_gap(begin, begin + 32);
                scalar.insert_gap(begin, begin + 32);
                const auto at = std::lower_bound(oracle.gaps.begin(), oracle.gaps.end(), begin,
                    [](const auto& gap, double value) { return gap.begin_ns < value; });
                oracle.gaps.insert(at, {begin, begin + 32});
            }
        }
        for (unsigned stream = 0; stream < 2; ++stream) {
            const double begin = (stream ? 110 : 70) * 256.0 + 32;
            const double at = begin - (step % 11 == 10 ? 8 : 0);
            const double duration = stream ? 0.5 : 0.25;
            // A distant query overwrites the shared navigation hint between
            // the two ordered streams; each stream must revalidate its own.
            const double other = (stream ? 4 : 124) * 256.0 + 16;
            check(ordered.preview_start(other, 0.125) == oracle.preview(other, 0.125),
                "interleaved distant slot query differs");
            const auto expected = oracle.reserve(at, duration);
            check(ordered.reserve_ordered(at, duration, stream) == expected &&
                scalar.reserve(at, duration) == expected, "shrinking slot reservation differs");
            check(ordered.gaps_after(0) == oracle.gaps && scalar.gaps_after(0) == oracle.gaps &&
                ordered.ready_ns == oracle.ready && ordered.reserved_work_ns == oracle.work,
                "slot hint changed idle capacity or accounting");
        }
        if (step == 48) {
            auto copy = ordered;
            auto expected = oracle;
            copy.prune_before(4 * 256.0 + 20);
            expected.prune(4 * 256.0 + 20);
            const double at = 70 * 256.0 + 32;
            check(copy.reserve_ordered(at, 0.25) == expected.reserve(at, 0.25) &&
                copy.gaps_after(0) == expected.gaps && copy.ready_ns == expected.ready &&
                copy.reserved_work_ns == expected.work,
                "copied slot hint survived prune with a wrong result");
        }
    }
    check(ordered.ordered_position_reuses() > 10,
        "rolling reservation positions never bypassed the calendar lookup");
    // A remembered later gap must not win over newly available earlier
    // capacity, even though it can fit the request and its index is valid.
    ordered.insert_gap(1, 5); scalar.insert_gap(1, 5);
    oracle.gaps.insert(oracle.gaps.begin(), {1, 5});
    const auto early = oracle.reserve(0, 0.25);
    check(early.start_ns == 1 && ordered.reserve_ordered(0, 0.25) == early &&
        scalar.reserve(0, 0.25) == early && ordered.gaps_after(0) == oracle.gaps,
        "stale rolling position skipped an earlier fitting interval");
}
}
int main() {
    try { replay(1,0);replay(0.1,0);replay(1.0/3,1e8);rounding_boundary();interleaved_service_streams();inline_promotion_and_pruning();tied_maxima_split_prune_copy();reservation_endpoints_and_failed_state();ordered_streams_match_scalar_and_oracle();ordered_capacity_and_frontier_changes();ordered_future_gaps_and_rounding();ordered_slot_hints_survive_shrink_and_split(); }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    std::cout<<"Packed calendar matches independent first-fit oracle\n";
}
