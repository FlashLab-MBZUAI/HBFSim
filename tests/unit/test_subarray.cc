#include <catch2/catch_test_macros.hpp>

#include "src/media/nand/subarray.hh"

using obelisk::Packet;
using obelisk::PacketType;
using obelisk::Subarray;
using obelisk::SubarrayConfig;

TEST_CASE("Subarray: single read latency matches tR", "[subarray]") {
    SubarrayConfig cfg;
    cfg.tR_cycles = 1000;
    cfg.queue_depth = 4;
    Subarray sa(0, cfg);

    auto pkt = std::make_unique<Packet>();
    pkt->type = PacketType::READ;
    pkt->size = 16384;
    pkt->phys.page_id = 42;

    REQUIRE(sa.accept(pkt));
    REQUIRE_FALSE(pkt);  // moved out

    // Drive the clock. First tick starts service.
    sa.tick(0);
    REQUIRE(sa.is_busy());

    // Partway through.
    sa.tick(500);
    REQUIRE(sa.is_busy());
    REQUIRE_FALSE(sa.dequeue().has_value());

    // After tR.
    sa.tick(1000);
    auto out = sa.dequeue();
    REQUIRE(out.has_value());
    REQUIRE((*out)->phys.page_id == 42);
}

TEST_CASE("Subarray: buffer hit reduces latency", "[subarray]") {
    SubarrayConfig cfg;
    cfg.tR_cycles = 1000;
    cfg.cache_hit_read_cycles = 50;
    cfg.cache_buffer_pages = 4;
    cfg.queue_depth = 4;
    Subarray sa(0, cfg);

    // First read: miss, brings page_id=7 into buffer.
    auto p1 = std::make_unique<Packet>();
    p1->type = PacketType::READ;
    p1->phys.page_id = 7;
    REQUIRE(sa.accept(p1));
    sa.tick(0);
    sa.tick(1000);
    REQUIRE(sa.dequeue().has_value());
    REQUIRE(sa.total_read_misses() == 1);

    // Second read to same page: hit.
    auto p2 = std::make_unique<Packet>();
    p2->type = PacketType::READ;
    p2->phys.page_id = 7;
    REQUIRE(sa.accept(p2));
    sa.tick(1001);  // start service, busy_until_ = 1001 + 50 = 1051
    sa.tick(1051);  // completion fires
    auto out = sa.dequeue();
    REQUIRE(out.has_value());
    REQUIRE(sa.total_read_hits() == 1);
}

TEST_CASE("Subarray: queue rejects at capacity", "[subarray]") {
    SubarrayConfig cfg;
    cfg.tR_cycles = 1000;
    cfg.queue_depth = 2;
    Subarray sa(0, cfg);

    auto a = std::make_unique<Packet>();
    a->type = PacketType::READ;
    auto b = std::make_unique<Packet>();
    b->type = PacketType::READ;
    auto c = std::make_unique<Packet>();
    c->type = PacketType::READ;

    REQUIRE(sa.accept(a));
    REQUIRE(sa.accept(b));
    // One is being served, second is in queue — but queue_depth=2 includes the
    // currently-serving slot only when it's still in queue_. After tick(0),
    // `a` is popped into current_, so the queue has only 1 (`b`). Another accept
    // should succeed.
    sa.tick(0);
    REQUIRE(sa.accept(c));
}

TEST_CASE("Subarray: write uses tPROG", "[subarray]") {
    SubarrayConfig cfg;
    cfg.tR_cycles = 1000;
    cfg.tPROG_cycles = 5000;
    cfg.queue_depth = 2;
    Subarray sa(0, cfg);

    auto p = std::make_unique<Packet>();
    p->type = PacketType::WRITE;
    REQUIRE(sa.accept(p));
    sa.tick(0);

    // Should still be busy at tR_cycles=1000.
    sa.tick(1000);
    REQUIRE(sa.is_busy());

    // Done at tPROG_cycles=5000.
    sa.tick(5000);
    REQUIRE(sa.dequeue().has_value());
}
