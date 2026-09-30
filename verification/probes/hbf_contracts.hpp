#pragma once

// Included by physical.cpp inside its anonymous namespace to share diagnostic
// rendering. These probes exercise the current OCP bank model through public
// interfaces; host GC, mapping and persistence have dedicated C++ regressions.

void hbf_require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("HBF physical contract: " + message);
}
bool hbf_close(double a, double b) {
    return std::abs(a-b) <= 1e-8 * std::max({1.0, std::abs(a), std::abs(b)});
}
HbfConfig hbf_probe_config() {
    HbfConfig c;
    c.device.channels_per_stack = 2;
    c.device.dies_per_channel = 1;
    c.device.planes_per_die = 2;
    c.device.blocks_per_plane = 16;
    c.device.pages_per_block = 8;
    c.device.t_read_page_ns = 200;
    c.device.t_erase_block_ns = 2000;
    c.device.t_program_page_ns = 750;
    c.host.mapping_mode = hbfsim::host::MappingMode::RawPhysical;
    c.host.auto_gc_enabled = false;
    c.host.gc_reserved_free_blocks_per_plane = 0;
    return c;
}
PhysicalRequest hbf_raw(std::string id, Op op, std::uint64_t address,
    std::uint64_t bytes, double arrival = 0) {
    auto q = request(std::move(id), Tier::HBF, op, arrival, address, bytes,
        AddressSpace::Physical);
    q.trace.mode = TraceMode::Full;
    return q;
}
const TraceSpan& hbf_span(const PhysicalCompletion& c, const std::string& name) {
    const auto at = std::find_if(c.spans.begin(), c.spans.end(),
        [&](const auto& span) { return span.name == name; });
    hbf_require(at != c.spans.end(), "missing trace span " + name);
    return *at;
}
void probe_hbf_bank_pipeline() {
    std::cout << "\n== OCP HBF ordered bank pipeline ==\n";
    auto cfg = hbf_probe_config();
    hbfsim::verification::HbfWithHbm d(cfg);
    const auto other = d.encode(HbfAddress{.plane=1});
    d.reserve_static_physical_pages({0, 1, other / 4096});
    const auto a=d.issue(hbf_raw("bank-a",Op::Read,0,4096));
    const auto b=d.issue(hbf_raw("bank-a-next",Op::Read,4096,4096));
    const auto c=d.issue(hbf_raw("bank-b",Op::Read,other,4096));
    // Copies: GCC 13's -Wdangling-reference cannot see that hbf_span returns
    // a reference into the completion, not into its temporary name argument.
    const TraceSpan sa=hbf_span(a,"user/array_read");
    const TraceSpan sb=hbf_span(b,"user/array_read");
    const TraceSpan sc=hbf_span(c,"user/array_read");
    hbf_require(sb.start_ns >= sa.end_ns, "same bank sensing overlapped");
    hbf_require(sc.start_ns < sb.end_ns, "independent banks lost concurrency");
    hbf_require(hbf_close(d.stats().subarray_read_busy_ns, 3*cfg.device.t_read_page_ns),
        "sense work not conserved");
    hbf_require(hbf_close(d.stats().hb_io_data_busy_ns, 3*4096.0/96),
        "external payload work must use per-channel grade bandwidth");
    print_rows({a,b,c});
    std::cout << "  same_bank_serial=yes independent_banks_overlap=yes payload_conserved=yes\n";

    hbfsim::verification::HbfWithHbm traced(cfg), quiet(cfg);
    traced.reserve_static_physical_pages({0}); quiet.reserve_static_physical_pages({0});
    auto q=hbf_raw("trace-invariance",Op::Read,0,4096);
    auto full=traced.issue(q); q.trace.mode=TraceMode::Off;
    auto off=quiet.issue(q);
    hbf_require(full.finish_ns==off.finish_ns &&
        traced.stats().stage_work==quiet.stats().stage_work &&
        traced.stats().hb_io_data_busy_ns==quiet.stats().hb_io_data_busy_ns,
        "diagnostics changed hardware work");
}
void probe_hbf_ecc_pipeline() {
    std::cout << "\n== HBF ECC latency and throughput ==\n";
    auto cfg=hbf_probe_config();
    cfg.device.oob_bytes_per_page=0;
    cfg.device.ecc_decode_raw_bandwidth_GBps_per_die=64;
    cfg.device.ecc_encode_raw_bandwidth_GBps_per_die=64;
    cfg.device.ecc_decode_latency_ns=500;
    cfg.device.ecc_encode_latency_ns=500;
    cfg.device.channel_bandwidth_GBps=1e9;
    cfg.device.tsv_bandwidth_GBps=1e9;
    cfg.device.logic_sram_bandwidth_GBps=1e9;
    hbfsim::verification::HbfWithHbm d(cfg);
    auto other=d.encode(HbfAddress{.plane=1});
    d.reserve_static_physical_pages({0,other/4096});
    const auto a=d.issue(hbf_raw("ecc-a",Op::Read,0,4096));
    const auto b=d.issue(hbf_raw("ecc-b",Op::Read,other,4096));
    const TraceSpan ai=hbf_span(a,"user/ecc_decode_issue");
    const TraceSpan bi=hbf_span(b,"user/ecc_decode_issue");
    const TraceSpan al=hbf_span(a,"user/ecc_decode_latency");
    const TraceSpan bl=hbf_span(b,"user/ecc_decode_latency");
    hbf_require(hbf_close(ai.duration_ns(),64) && hbf_close(bi.start_ns-ai.start_ns,64),
        "same-die ECC initiation interval must be raw bytes/bandwidth");
    hbf_require(hbf_close(al.duration_ns(),500) && bl.start_ns<al.end_ns,
        "ECC response latency must overlap later codewords");
    auto longer=cfg; longer.device.ecc_decode_latency_ns=900;
    hbfsim::verification::HbfWithHbm slow(longer); slow.reserve_static_physical_pages({0});
    const auto latency=slow.issue(hbf_raw("ecc-long-latency",Op::Read,0,4096));
    hbf_require(hbf_close(latency.finish_ns-a.finish_ns,400), "latency-only delta changed issue rate");
    auto parity=cfg; parity.device.oob_bytes_per_page=512;
    hbfsim::verification::HbfWithHbm oob(parity); oob.reserve_static_physical_pages({0});
    const auto p=oob.issue(hbf_raw("ecc-parity",Op::Read,0,4096));
    hbf_require(hbf_close(hbf_span(p,"user/ecc_decode_issue").duration_ns(),72),
        "OOB must occupy raw ECC bandwidth");
    hbf_require(hbf_close(oob.stats().hb_io_data_busy_ns,d.stats().hb_io_data_busy_ns/2),
        "OOB leaked into external decoded payload");
    auto independent=cfg; independent.device.planes_per_die=1;
    hbfsim::verification::HbfWithHbm two(independent);
    const auto channel=two.encode(HbfAddress{.channel=1});
    two.reserve_static_physical_pages({0,channel/4096});
    const auto ca=two.issue(hbf_raw("die-a",Op::Read,0,4096));
    const auto cb=two.issue(hbf_raw("die-b",Op::Read,channel,4096));
    hbf_require(std::abs(hbf_span(cb,"user/ecc_decode_issue").start_ns-
        hbf_span(ca,"user/ecc_decode_issue").start_ns)<64,
        "independent dies shared an ECC issue port");
    print_rows({a,b,latency,p,ca,cb});
    std::cout << "  ecc_raw_ii=yes latency_overlap=yes per_die_ports=yes oob_internal_only=yes\n";
}
void probe_hbf_host_boundary() {
    std::cout << "\n== HBF Host assembly and full-page transport ==\n";
    auto cfg=hbf_probe_config();
    cfg.host.mapping_mode=hbfsim::host::MappingMode::FullResident;
    cfg.host.auto_gc_enabled=true; cfg.host.gc_reserved_free_blocks_per_plane=2;
    cfg.host.write_coalescing_enabled=true; cfg.host.write_buffer_pages=4;
    cfg.host.write_buffer_flush_threshold_pages=4;
    hbfsim::verification::HbfWithHbm d(cfg);
    double now=0;
    for (unsigned i=0;i<8;++i) {
        auto q=request("fragment-"+std::to_string(i),Tier::HBF,Op::Write,now,i*64,64);
        now=d.issue(q).finish_ns;
    }
    const auto drained=d.drain_pending("full-page-destage",now);
    const auto& s=d.stats();
    hbf_require(s.data_programs==1 && s.mapping_page_programs==1,
        "Host fragments did not coalesce into one data page plus checkpoint");
    hbf_require(s.physical_write_bytes==8192 && s.logical_write_bytes==512,
        "Host logical/full-page write accounting");
    hbf_require(hbf_close(s.hb_io_data_busy_ns,8192.0/96),
        "program payload must cross interface once after Host assembly");
    hbf_require(s.block_erases==2 && s.auto_erase_requests==2,
        "each first data/mapping page must autoerase its block");
    hbf_require(s.write_buffer_dram_write_bytes == 512 && s.accounting_verified,
        "Host buffer acceptance or capacity accounting diverged");
    auto read_cfg=cfg; read_cfg.host.write_coalescing_enabled=false;
    hbfsim::verification::HbfWithHbm reader(read_cfg); reader.prepopulate_logical_pages({0});
    const auto short_read=reader.issue(request("read-64B-coverage",Tier::HBF,Op::Read,0,63,2));
    hbf_require(hbf_close(reader.stats().hb_io_data_busy_ns,128.0/96),
        "Host range crossing a 64B boundary must transfer both device units");
    print_rows({drained,short_read});
    std::cout << "  host_buffer_accounted=yes full_page_program=yes rounded_device_read=yes\n";
}
void probe_hbf_program_barriers() {
    std::cout << "\n== HBF program visibility and destructive ordering ==\n";
    auto cfg=hbf_probe_config();
    hbfsim::verification::HbfWithHbm d(cfg);
    const auto first=d.issue(hbf_raw("program-page-zero",Op::Write,0,4096));
    const auto read=d.issue(hbf_raw("read-pending-program",Op::Read,0,64));
    hbf_require(hbf_span(read,"user/array_read").start_ns >= first.finish_ns,
        "raw read became visible before nonposted program completion");
    const auto erase=d.issue(hbf_raw("erase-after-read",Op::Erase,0,0));
    // Once the raw page has left the bank, ECC/egress may overlap an erase;
    // a destructive command must fence NAND capture, not the whole response.
    hbf_require(erase.finish_ns >= hbf_span(read,"user/page_buffer_out").end_ns+
        cfg.device.t_erase_block_ns, "erase did not fence the raw-page capture");
    (void)d.drain_pending("retire-erase",erase.finish_ns);
    bool rejected=false;
    try { (void)d.issue(hbf_raw("erased-read",Op::Read,0,64,erase.finish_ns)); }
    catch (const std::exception&) { rejected=true; }
    hbf_require(rejected,"erased raw page remained readable");
    const auto pec=d.block_erase_counts();
    hbf_require(std::accumulate(pec.begin(),pec.end(),std::uint64_t{0})==2,
        "page-zero and explicit erase wear not conserved");
    print_rows({first,read,erase});
    std::cout << "  nonposted_visibility=yes erase_consumer_fence=yes erased_read_rejected=yes\n";
}
void probe_hbf_calendar() {
    std::cout << "\n== HBF exact channel reservation calendar ==\n";
    auto cfg=hbf_probe_config();
    hbfsim::physical::hbf::HbfDevice media(cfg.device);
    using Direction=hbfsim::physical::hbf::HbfDevice::Direction;
    // 128 future intervals leave real 10ns gaps. Every one must remain
    // reusable; no bounded history truncation or overlap is permitted.
    for (unsigned i=0;i<128;++i)
        (void)media.transfer(0,Direction::HostToDevice,960,10.0+20*i,0);
    for (unsigned i=0;i<128;++i) {
        const auto r=media.transfer(0,Direction::HostToDevice,960,0,0);
        hbf_require(hbf_close(r.start_ns,20.0*i),"lost a reusable calendar interval");
    }
    const auto rx=media.transfer(1,Direction::HostToDevice,960,0,0);
    const auto tx=media.transfer(0,Direction::DeviceToHost,960,0,0);
    hbf_require(rx.start_ns==0 && tx.start_ns==0,"channel or directional isolation lost");
    hbf_require(hbf_close(media.channels()[0].rx_work_ns,2560),"calendar occupied work mismatch");
    std::cout << "  128_gaps_preserved=yes full_duplex=yes channel_isolation=yes\n";
}
void probe_hbf_standard() {
    probe_hbf_bank_pipeline();
    probe_hbf_ecc_pipeline();
    probe_hbf_host_boundary();
    probe_hbf_program_barriers();
    probe_hbf_calendar();
}
