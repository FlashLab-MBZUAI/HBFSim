#include <catch2/catch_test_macros.hpp>

#include <deque>
#include <memory>

#include "src/controller/scheduler.hh"

using namespace obelisk;

static std::unique_ptr<Packet> make_pkt(PacketType type, uint64_t page) {
    auto p = std::make_unique<Packet>();
    p->id = page;
    p->type = type;
    p->phys.page_id = page;
    return p;
}

TEST_CASE("FCFS: picks head of read queue with read priority", "[scheduler]") {
    auto sched = make_scheduler("fcfs");
    std::deque<std::unique_ptr<Packet>> rq;
    std::deque<std::unique_ptr<Packet>> wq;
    rq.push_back(make_pkt(PacketType::READ, 1));
    rq.push_back(make_pkt(PacketType::READ, 2));
    wq.push_back(make_pkt(PacketType::WRITE, 100));

    auto pkt = sched->select_next({rq, wq, 0, true, nullptr});
    REQUIRE(pkt);
    REQUIRE(pkt->phys.page_id == 1);
}

TEST_CASE("FCFS: falls back to write queue when reads empty", "[scheduler]") {
    auto sched = make_scheduler("fcfs");
    std::deque<std::unique_ptr<Packet>> rq;
    std::deque<std::unique_ptr<Packet>> wq;
    wq.push_back(make_pkt(PacketType::WRITE, 100));

    auto pkt = sched->select_next({rq, wq, 0, true, nullptr});
    REQUIRE(pkt);
    REQUIRE(pkt->phys.page_id == 100);
}

TEST_CASE("FRFCFS: prefers page hit", "[scheduler]") {
    auto sched = make_scheduler("frfcfs");
    std::deque<std::unique_ptr<Packet>> rq;
    std::deque<std::unique_ptr<Packet>> wq;

    // First access sets last_page_id_ to 5.
    rq.push_back(make_pkt(PacketType::READ, 5));
    auto first = sched->select_next({rq, wq, 0, true, nullptr});
    REQUIRE(first->phys.page_id == 5);

    // Now two pending: one at page 9, one at page 5 (hit).
    rq.push_back(make_pkt(PacketType::READ, 9));
    rq.push_back(make_pkt(PacketType::READ, 5));
    auto next = sched->select_next({rq, wq, 1, true, nullptr});
    REQUIRE(next);
    REQUIRE(next->phys.page_id == 5);
}

TEST_CASE("HBFAware: prefers READ_KV over plain READ", "[scheduler]") {
    auto sched = make_scheduler("hbf_aware");
    std::deque<std::unique_ptr<Packet>> rq;
    std::deque<std::unique_ptr<Packet>> wq;

    rq.push_back(make_pkt(PacketType::READ, 1));
    rq.push_back(make_pkt(PacketType::READ_KV, 2));
    rq.push_back(make_pkt(PacketType::READ, 3));

    auto pkt = sched->select_next({rq, wq, 0, true, nullptr});
    REQUIRE(pkt);
    REQUIRE(pkt->type == PacketType::READ_KV);
    REQUIRE(pkt->phys.page_id == 2);
}

TEST_CASE("make_scheduler: unknown type throws", "[scheduler]") {
    REQUIRE_THROWS(make_scheduler("nonsense"));
}
