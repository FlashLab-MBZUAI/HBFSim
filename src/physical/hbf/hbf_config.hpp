#pragma once
#include <cstdint>
#include "physical/hbf/hbf_standard.hpp"
#include <string_view>

namespace hbfsim::physical::hbf {

struct HbfDeviceConfig {
    std::uint32_t speed_grade = 2;
    static constexpr auto standard = kStandard;
    std::uint32_t stacks = 1;
    std::uint32_t channels_per_stack = 16;
    std::uint32_t dies_per_channel = 1;
    std::uint32_t planes_per_die = 16;
    std::uint32_t blocks_per_plane = 2048;
    std::uint32_t pages_per_block = 256;
    std::uint64_t page_size_bytes = 4096;
    // Out-of-band (spare) bytes per page: ECC parity, reverse LPN mapping,
    // block status. Shares the page's wordline, so array timings are
    // unchanged. Flash-side raw transfers and ECC move page + OOB; decoded
    // SRAM and the external HBIO move payload only. Logical/physical byte
    // statistics stay data-area based.
    std::uint64_t oob_bytes_per_page = 224;
    std::uint32_t media_lanes_per_plane = 1;
    std::uint32_t page_buffer_banks_per_plane = 2;
    double t_read_page_ns = 4000.0;
    // Total NAND page-program time, including internal program verification.
    double t_program_page_ns = 75000.0;
    double t_erase_block_ns = 2'000'000.0;
    // ECC response latency is independent of pipeline throughput. The raw
    // bandwidth includes data + OOB codeword bytes and sets the initiation
    // interval of one shared decode/encode issue port per flash die.
    // The default 101.25 GB/s carries 96 GB/s of payload plus 224 B
    // OOB per 4 KiB page. These are assumptions, not vendor measurements.
    double ecc_decode_latency_ns = 500.0;
    double ecc_encode_latency_ns = 500.0;
    double ecc_decode_raw_bandwidth_GBps_per_die = 101.25;
    double ecc_encode_raw_bandwidth_GBps_per_die = 101.25;
    // Flash channel/media/page-buffer data bandwidths are raw codeword rates
    // (payload + OOB). The stack TSV uses one shared timeline for internal
    // command bytes and those raw codewords. HBIO is external payload
    // bandwidth and logic SRAM holds decoded payload.
    double channel_bandwidth_GBps = 101.25;
    [[nodiscard]] double hb_io_bandwidth_GBps() const {
        return channels_per_stack * physical::hbf::speed_grade(speed_grade).payload_GBps_per_channel;
    }
    double tsv_bandwidth_GBps = 1644.0;
    double media_lane_bandwidth_GBps = 2048.0;
    double logic_sram_bandwidth_GBps = 2048.0;
    double page_buffer_bandwidth_GBps = 2048.0;
    std::uint64_t command_address_bytes = 64;
    // Per-request dispatch cost of the stack's logic scheduler. HBM-class
    // PHYs issue a command every ~2 ns; at 4 KiB pages a 20 ns dispatch would
    // cap a stack at 200 GB/s and silently dominate the read fabric.
    double logic_scheduler_issue_ns = 2.0;
    double address_generation_ns = 5.0;
    double flash_tsu_issue_ns = 10.0;
    // Finite controller credits for page-granular foreground reads, per HBF
    // stack. A large memory-object request is split into page transactions and
    // every child holds one credit from translation admission through its
    // user-visible completion. This prevents one giant parent request from
    // bypassing the intended outstanding-work bound. The value is an explicit
    // architecture assumption, not a published HBF product parameter.
    std::uint64_t page_read_queue_depth_per_stack = 4096;
    // OCP 0.7.0 section 5.4.1: host-configured accumulation timeout and
    // product-specific outstanding logical-unit limit. These defaults are
    // model assumptions, not mandated OCP timing/capacity values.
    double write_accumulation_timeout_ns = 1'000'000.0;
    std::uint32_t outstanding_write_pages_per_channel = 64;
    // Workload-coupled stack thermal model: one lumped RC node per stack
    // (junction/sensor) plus a firmware pacing governor. Media operations
    // deposit energy; between deposits the node decays toward
    // ambient + static_power * resistance. When the sensor crosses
    // throttle_c the governor paces media admissions so dissipated power
    // stays at the pacing budget until the node decays below release_c.
    // Array timings (tR/tProg/tErase) are never altered: real firmware
    // throttles by delaying command issue, not by reprogramming the array.
    // Disabled by default because enabling it changes completion times.
    bool thermal_enabled = false;
    double thermal_ambient_c = 45.0;
    double thermal_resistance_c_per_w = 2.5;
    double thermal_capacitance_j_per_c = 3.0;
    double thermal_throttle_c = 85.0;
    double thermal_release_c = 80.0;
    double thermal_static_power_w = 2.0;
    // Active energy per payload bit for a media page read/program (array
    // sense/program plus the on-stack data path), and per block erase.
    // These are explicit architecture assumptions, not vendor measurements;
    // the sustainable throttled bandwidth they imply is
    // pacing_power / energy_per_bit.
    double thermal_read_energy_pj_per_bit = 6.0;
    double thermal_program_energy_pj_per_bit = 60.0;
    double thermal_erase_energy_uj_per_block = 150.0;
    // Conducted boundary-temperature rise from package neighbors (the GPU
    // and HBM stacks sharing the interposer and cold plate) at the declared
    // operating load. The node decays toward
    // ambient + neighbor_heat + static_power * resistance: ambient stays
    // the documented coolant-supply reference while this term carries the
    // cross-heating assumption. Zero models a thermally isolated stack.
    double thermal_neighbor_heat_c = 0.0;
    // Media power budget per stack while throttled. Zero derives the
    // ceiling-sustainable budget
    // (throttle_c - ambient_c - neighbor_heat_c) / resistance minus static
    // power.
    double thermal_throttle_power_w = 0.0;
    // Boot/restore temperature state. false boots every stack at the idle
    // steady state, so short windows from cold are unaffected by the
    // model. true boots at the throttle ceiling with the governor already
    // engaged: the measured window is then an excerpt of sustained serving
    // whose thermal steady state pins the stack at its governor ceiling.
    // The start state is an experiment initial-state boundary rather than
    // a device property, so system profiles keep it false and studies opt
    // in per row.
    bool thermal_start_at_ceiling = false;
};

} // namespace hbfsim::physical::hbf
