#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <fstream>
#include <string>

#include "src/frontend/trace_reader.hh"

using namespace obelisk;

static std::string write_tmp_trace(const std::string& content) {
    std::string path = std::tmpnam(nullptr);
    path += ".trace";
    std::ofstream out(path);
    out << content;
    return path;
}

TEST_CASE("TraceReader: parses basic records", "[trace_reader]") {
    auto path = write_tmp_trace(
        "0 R 0x1800000000 16384 0 seq_read\n"
        "-1 W 0x0000000040 64 3 kv_write\n"
        "100 RKV 0x1800004000 16384\n");
    TraceReader r(path);

    auto a = r.next();
    REQUIRE(a.has_value());
    REQUIRE(a->timestamp == 0);
    REQUIRE(a->type == PacketType::READ);
    REQUIRE(a->addr == 0x1800000000ULL);
    REQUIRE(a->size == 16384);
    REQUIRE(a->op_name == "seq_read");

    auto b = r.next();
    REQUIRE(b.has_value());
    REQUIRE(b->timestamp == -1);
    REQUIRE(b->type == PacketType::WRITE);
    REQUIRE(b->layer_id == 3);

    auto c = r.next();
    REQUIRE(c.has_value());
    REQUIRE(c->type == PacketType::READ_KV);
    REQUIRE(c->op_name.empty());

    REQUIRE_FALSE(r.next().has_value());

    std::remove(path.c_str());
}

TEST_CASE("TraceReader: skips blank and comment lines", "[trace_reader]") {
    auto path = write_tmp_trace(
        "# comment\n"
        "\n"
        "0 R 0x100 64\n");
    TraceReader r(path);
    auto a = r.next();
    REQUIRE(a.has_value());
    REQUIRE(a->addr == 0x100);
    REQUIRE_FALSE(r.next().has_value());
    std::remove(path.c_str());
}
