#include "../../verification/probes/hbf_with_hbm.hpp"
#include "host/hbf_controller.hpp"
#include "host/hbf_persistent_image.hpp"
#include "host/hbf_zone_endurance.hpp"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace hbfsim::physical;
using namespace hbfsim::host;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F function) {
    bool rejected = false;
    try { function(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "unsafe zone operation was accepted");
}

HbfConfig geometry() {
    HbfConfig c;
    c.device.stacks = 2; c.device.channels_per_stack = 2; c.device.dies_per_channel = 1;
    c.device.planes_per_die = 1; c.device.blocks_per_plane = 12; c.device.pages_per_block = 4;
    c.device.page_size_bytes = 4096; c.host.zone_size_blocks = 4;
    return c;
}
void exercise_base_die_and_initial_image() {
    auto cfg = geometry();
    cfg.host.mapping_mode = MappingMode::RawPhysical;
    hbf::HbfDevice base(cfg.device);
    rejects([&] { base.execute_zone_remap({.first=0,.second=1}); });
    base.configure_host_zones(cfg.host.zone_size_blocks);
    rejects([&] { base.execute_zone_remap({.opcode=0x7,.first=0,.second=1}); });
    rejects([&] { base.execute_zone_remap({.remap_type=1,.first=0,.second=1}); });
    base.seed_block(0, 0, 50000);
    base.seed_block(4, 0, 1000);
    base.execute_zone_remap({.first=0,.second=1});
    require(base.zone_pec_sum(0)==1000 && base.zone_pec_sum(1)==50000,
        "PEC query did not follow the base-die mapping");
    require(base.block_pec(0)==50000 && base.block_pec(4)==1000,
        "device remap moved physical wear");
    rejects([&] { base.execute_zone_remap({.first=0,.second=3}); });

    hbfsim::verification::HbfWithHbm d(cfg);
    d.prepopulate_channel_zones(0,0,0,2);
    rejects([&] { d.prepopulate_channel_zones(0,0,1,1); });
    auto image=d.persistent_image();
    require(image.state.materialized_pages.empty(), "zone image expanded into per-page records");
    require(d.stats().page_programs==0 && d.stats().block_erases==0 && d.stats().accounting_verified,
        "initial zone data incurred warmup traffic or broke the capacity census");
    const auto path=std::filesystem::temp_directory_path()/"hbfsim-compact-zone-test.image";
    std::filesystem::remove(path);
    write_persistent_image_file(path,image);
    hbfsim::verification::HbfWithHbm restored(cfg);
    restored.restore_persistent_image(read_persistent_image_file(path));
    std::filesystem::remove(path);
    double at=restored.issue_channel_local(PhysicalRequest{.id="initial-read",.tier=Tier::HBF,
        .op=Op::Read,.arrival_ns=0,.addr=0,.bytes=4096}).finish_ns;
    rejects([&] { (void)restored.reset_zone(0,0,0,at); });
    rejects([&] { (void)restored.remap_zones(0,0,0,1,at); });
    at=restored.invalidate_zone(0,0,0,at).finish_ns;
    const auto erases=restored.block_erase_counts();
    at=restored.reset_zone(0,0,0,at).finish_ns;
    require(restored.stats().host_zone_remaps==0 && restored.block_erase_counts()==erases,
        "host reset implicitly remapped or erased media");
    rejects([&] { (void)restored.issue_channel_local(PhysicalRequest{.id="invalid-read",.tier=Tier::HBF,
        .op=Op::Read,.arrival_ns=at,.addr=0,.bytes=4096}); });
    at=restored.recycle_zone(0,0,0,at).finish_ns;
    at=restored.issue_channel_local(PhysicalRequest{.id="new-data",.tier=Tier::HBF,.op=Op::Write,
        .arrival_ns=at,.addr=0,.bytes=4096}).finish_ns;
    at=restored.drain_pending("complete",at).finish_ns;
    require(restored.stats().block_erases==1 && restored.stats().accounting_verified,
        "rewrite must incur exactly one page-zero erase");
    require(restored.zone_state(0,0,1,at).valid_pages==16,
        "live neighboring zone was used as a wear-leveling destination");
    auto after=restored.persistent_image();
    hbfsim::verification::HbfWithHbm continued(cfg);
    continued.restore_persistent_image(after);
    require(continued.stats().accounting_verified && continued.block_erase_counts()==restored.block_erase_counts(),
        "mixed initial/runtime zone state did not restore");
}
void exercise_host_gc() {
    auto cfg = geometry();
    cfg.device.stacks = cfg.device.channels_per_stack = cfg.host.zone_size_blocks = 1;
    cfg.device.planes_per_die = 2; cfg.device.blocks_per_plane = 16; cfg.device.pages_per_block = 8;
    hbfsim::verification::HbfWithHbm d(cfg);
    d.prepopulate_mutable_logical_page_range(0, 128);
    double at = 0;
    for (unsigned i = 0; i < 1200; ++i)
        at = d.issue(PhysicalRequest{.id="overwrite", .tier=Tier::HBF, .op=Op::Write,
            .arrival_ns=at, .addr=(i*37 % 96)*cfg.device.page_size_bytes,
            .bytes=cfg.device.page_size_bytes}).finish_ns;
    at = d.drain_pending("gc-drain",at).finish_ns;
    const auto& stats = d.stats();
    require(stats.gc_relocations > 0, "host roundtrip test never copied live data");
    require(stats.host_gc_read_bytes == stats.gc_relocations * cfg.device.page_size_bytes &&
        stats.host_gc_read_bytes == stats.host_gc_write_bytes, "host GC byte conservation failed");
    require(stats.host_hbm_read_bytes >= stats.host_gc_write_bytes &&
        stats.host_hbm_write_bytes >= stats.host_gc_read_bytes,
        "GC copies omitted their HBM staging write or source read");
    require(stats.hb_io_data_busy_ns >=
        (stats.host_gc_read_bytes + stats.host_gc_write_bytes) /
            hbf::speed_grade(cfg.device.speed_grade).payload_GBps_per_channel,
        "host GC copies bypassed ordinary HBF transport");
    require(stats.host_gc_control_ns > 0 && stats.accounting_verified,
        "host decision work was uncharged or media accounting diverged");
}
void exercise_endurance_equivalence() {
    for (const bool remap : {false, true}) for (const unsigned zone_blocks : {1u, 4u}) {
        auto cfg=geometry();
        cfg.host.zone_size_blocks=zone_blocks;
        cfg.host.mapping_mode=MappingMode::RawPhysical;
        cfg.host.auto_gc_enabled=false;
        hbfsim::verification::HbfWithHbm timed(cfg);
        HbfZoneEndurance exact(cfg.device,zone_blocks,cfg.host.host_zone_wear_gap);
        const auto zones_per_channel=cfg.device.blocks_per_plane/zone_blocks;
        const auto zone_bytes=std::uint64_t{zone_blocks}*cfg.device.pages_per_block*4096;
        for (unsigned ch=0;ch<4;++ch) {
            timed.prepopulate_channel_zones(ch/2,ch%2,0,2);
            exact.prepopulate(ch*zones_per_channel,2);
        }
        double at=0;
        for (unsigned i=0;i<80;++i) {
            const auto ch=(i*3)%4;
            const auto zone=1u; // zone zero holds live weights throughout.
            at=timed.invalidate_zone(ch/2,ch%2,zone,at).finish_ns;
            at=(remap ? timed.recycle_zone(ch/2,ch%2,zone,at) :
                timed.reset_zone(ch/2,ch%2,zone,at)).finish_ns;
            at=timed.issue_channel_local(PhysicalRequest{.id="zone-rewrite",.tier=Tier::HBF,
                .op=Op::Write,.arrival_ns=at,.addr=(ch*zones_per_channel+zone)*zone_bytes,
                .bytes=zone_bytes}).finish_ns;
            timed.materialize_committed_state_through(at);
            exact.rewrite(ch*zones_per_channel+zone,remap);
            require(exact.wear()==timed.block_erase_counts(), "whole-zone arithmetic changed physical erase distribution");
            require(exact.mapping()==timed.persistent_image().zone_remapping,
                "whole-zone arithmetic chose a different remap destination");
        }
        const auto& s=timed.stats();
        require(s.page_programs==exact.programs && s.block_erases==exact.erases &&
            s.host_zone_remaps==exact.remaps && s.host_zone_resets==exact.resets &&
            s.host_zone_invalidations==exact.invalidations && s.gc_relocations==0 && s.accounting_verified,
            "timed/native and exact zone lifecycle ledgers differ");
    }
}
int main() {
    try {
        exercise_base_die_and_initial_image();
        exercise_endurance_equivalence();
        exercise_host_gc();
        auto cfg = geometry();
        auto invalid = cfg; invalid.host.zone_size_blocks = 5;
        rejects([&] { hbfsim::verification::HbfWithHbm d(invalid); });
        hbfsim::verification::HbfWithHbm d(cfg);
        double at = 0;
        // Figure 46: zone 1 is hot; zone 2 is cold in the same channel.
        for (int cycle = 0; cycle < 3; ++cycle) for (std::uint64_t block = 4; block < 8; ++block)
            at = d.issue(PhysicalRequest{.id="prewear", .tier=Tier::HBF, .op=Op::Erase,
                .address_space=AddressSpace::Physical, .arrival_ns=at,
                .addr=block*cfg.device.pages_per_block*cfg.device.page_size_bytes}).finish_ns;
        const auto zone_bytes = cfg.host.zone_size_blocks * cfg.device.pages_per_block * cfg.device.page_size_bytes;
        const auto write = [&](std::uint64_t address) {
            at = d.issue_channel_local(PhysicalRequest{.id="zone-write", .tier=Tier::HBF,
                .op=Op::Write, .arrival_ns=at, .addr=address, .bytes=cfg.device.page_size_bytes}).finish_ns;
        };
        const auto read = [&](std::uint64_t address, double arrival) {
            return d.issue_channel_local(PhysicalRequest{.id="zone-read", .tier=Tier::HBF,
                .op=Op::Read, .arrival_ns=arrival, .addr=address, .bytes=cfg.device.page_size_bytes});
        };
        write(zone_bytes);
        rejects([&] { (void)d.remap_zones(0,0,1,2,at); });
        const auto r = read(zone_bytes, at);
        rejects([&] { (void)d.invalidate_zone(0,0,1,at); });
        at = r.finish_ns;
        at = d.invalidate_zone(0,0,1,at).finish_ns;
        const auto wear_before = d.block_erase_counts();
        const auto programs_before = d.stats().page_programs;
        at = d.remap_zones(0,0,1,2,at).finish_ns;
        require(wear_before == d.block_erase_counts(), "remap moved physical PEC history");
        require(programs_before == d.stats().page_programs, "remap secretly copied data");
        rejects([&] { (void)read(zone_bytes,at); });
        rejects([&] { (void)read(2*zone_bytes,at); });
        write(zone_bytes);
        const auto result = read(zone_bytes,at); at = result.finish_ns;
        require(result.resource_path.find("physical-zone2") != std::string::npos,
            "rewrite did not use the swapped physical destination");
        // Other channels stay identity, even after a swap in channel 0.
        write(3*zone_bytes);
        require(read(3*zone_bytes,at).resource_path.find("physical-zone3") != std::string::npos,
            "remap escaped its channel");
        at = d.stats().finish_ns;
        at = d.drain_pending("persist",at).finish_ns;
        auto image = d.persistent_image();
        require(image.zone_remapping.at(1)==2 && image.zone_remapping.at(2)==1,
            "checkpoint lost the zone permutation");
        const auto path = std::filesystem::temp_directory_path()/"hbfsim-host-zone-test.image";
        std::filesystem::remove(path);
        write_persistent_image_file(path,image);
        hbfsim::verification::HbfWithHbm restored(cfg);
        restored.restore_persistent_image(read_persistent_image_file(path));
        std::filesystem::remove(path);
        require(restored.block_erase_counts()==d.block_erase_counts(), "restore lost physical wear");
        require(restored.wear_snapshot_json().find("\"workload_erases\":0")!=std::string::npos,
            "restored history was counted as new workload wear");
        rejects([&] { auto bad=image;bad.zone_remapping[1]=3;hbfsim::verification::HbfWithHbm x(cfg);x.restore_persistent_image(bad); });
        rejects([&] { auto bad=image;bad.channels_per_stack=1;hbfsim::verification::HbfWithHbm x(cfg);x.restore_persistent_image(bad); });
        // Automated WL is invalid-zone remapping, with no live relocation.
        at = restored.invalidate_zone(0,0,1,0).finish_ns;
        const auto reset_pec = restored.block_erase_counts();
        at = restored.recycle_zone(0,0,1,at).finish_ns;
        require(restored.block_erase_counts() == reset_pec, "reset invented wear");
        for (unsigned b = 0; b < cfg.host.zone_size_blocks; ++b)
            at = restored.issue_channel_local(PhysicalRequest{.id="wear-zone", .tier=Tier::HBF,
                .op=Op::Write, .arrival_ns=at,
                .addr=zone_bytes + b * cfg.device.pages_per_block * cfg.device.page_size_bytes,
                .bytes=cfg.device.page_size_bytes}).finish_ns;
        at = restored.invalidate_zone(0,0,1,at).finish_ns;
        at = restored.recycle_zone(0,0,1,at).finish_ns;
        require(restored.stats().host_zone_remaps > 0, "reset did not select a colder invalid zone");
        require(restored.stats().gc_relocations==0 && restored.stats().static_wear_leveling_relocations==0,
            "zone reset activated a live-copy policy");
        require(restored.stats().accounting_verified, "zone accounting did not conserve capacity");
        std::cout << "host zone mapping, IO fencing, physical PEC, and restore passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
