const TYPE_META = {
  system: { label: "System / front-end", className: "type-system", color: "#dceaf3" },
  hbm: { label: "HBM hardware", className: "type-hbm", color: "#d8e8fb" },
  hbf: { label: "HBF hardware", className: "type-hbf", color: "#d9f0ec" },
  external: { label: "External backing", className: "type-external", color: "#e8e2f0" },
  logic: { label: "Logic / policy", className: "type-logic", color: "#e5e0f5" },
  media: { label: "Flash media", className: "type-media", color: "#eee4d2" },
  flow: { label: "Flow stage", className: "type-flow", color: "#f3e4cf" },
  maintenance: { label: "Maintenance", className: "type-maintenance", color: "#f1d9d7" },
  output: { label: "Output / metric", className: "type-output", color: "#dde6df" },
};

const LEVEL_META = {
  source: {
    code: "L0",
    label: "Workload evidence",
    detail: "External traces, generators, model descriptors, and demand inputs.",
  },
  contract: {
    code: "L1",
    label: "Trace contract",
    detail: "Validated operations and semantic metadata consumed by the simulator.",
  },
  composition: {
    code: "L2",
    label: "Composition model",
    detail: "Reusable tier routing, dependency, admission, and streaming control.",
  },
  device: {
    code: "L3",
    label: "Device / controller",
    detail: "One modeled HBM, HBF, or external-backing device and controller-visible state.",
  },
  resource: {
    code: "L4",
    label: "Timed resource",
    detail: "Queues, links, schedulers, banks, buses, ECC, and other contention points.",
  },
  media: {
    code: "L5",
    label: "Persistent media state",
    detail: "Flash hierarchy, page ownership, program/erase state, GC, and wear.",
  },
  evidence: {
    code: "L6",
    label: "Observable evidence",
    detail: "Schema-v12 results, timing views, heatmaps, traces, and validation guards.",
  },
};

const DEFAULT_LEVEL_BY_TYPE = {
  system: "contract",
  hbm: "device",
  hbf: "device",
  external: "device",
  logic: "resource",
  media: "media",
  flow: "resource",
  maintenance: "media",
  output: "evidence",
};

// Template changes intentionally advance the storage namespace. Otherwise an
// old browser-local snapshot silently hides newly added architecture entities.
const STORAGE_KEY = "hbfsim.architecture.editor.v13";

const templates = {
  system_map: {
    id: "system_map",
    name: "End-to-end simulator architecture",
    description: "Workload evidence becomes one validated trace, one selected reusable composition model, timed devices, and auditable outputs.",
    nodes: [
      n("synthetic", "Controlled Synthetic Workloads", "sequential, random, locality, working-set, and read/write mixes", "system", 40, 40, 270, 94, "source"),
      n("astra", "ASTRA-sim Workload Adapter", "manifest-v3 model/profile identity plus layer/phase/compute metadata", "system", 360, 40, 290, 94, "source"),
      n("qwen", "Qwen-Bailian Demand Trace", "sanitized online request arrivals, lengths, and request identity", "system", 700, 40, 270, 94, "source"),
      n("frontier", "Frontier Memory-object Exporter", "request demand + model descriptor -> weights/KV memory operations", "system", 1020, 40, 300, 94, "source"),
      n("trace", "Canonical Memory Trace", "addr, R/W, bytes, at, kind, phase, layer, compute_ns", "flow", 260, 230, 310, 94, "contract"),
      n("guard", "Fail-closed Input & Provenance", "schema, digest, model profile, address accounting, ordering, and range checks", "logic", 640, 230, 330, 100, "contract"),
      n("request", "MemoryRequest Contract", "byte range + arrival + optional semantic/dependency hints; hints never required for direct routing", "system", 1040, 230, 340, 104, "contract"),
      n("runner", "scenario_compare Experiment Runner", "same trace, resolved configuration, selected composition model, bounded collection", "logic", 200, 430, 330, 98, "composition"),
      n("direct", "Direct Composition Engine", "all-HBM, all-HBF, Flat, and static-direct policies; page-level credits", "logic", 610, 430, 330, 98, "composition"),
      n("streaming", "Layer-streaming Engine", "selectable backing, finite backing credits, independent HBM credits, two dynamic buffers", "logic", 1020, 430, 350, 104, "composition"),
      n("hbm", "HBM Stack Device Model", "derived interface rate, address map, FR-FCFS, commands, refresh, pseudo-channel bus", "hbm", 440, 640, 340, 106, "device"),
      n("hbf", "HBF Stack Device Model", "resident L2P, decoded SRAM, ECC, TSU, flash resources, program/erase, GC", "hbf", 940, 640, 350, 106, "device"),
      n("external", "External-backing Device Model", "CXL memory / SSD profiles; global E2E credits and explicit M2S/controller/media/S2M pipeline", "external", 940, 790, 350, 106, "device"),
      n("summary", "Schema-v12 Summary + CSV", "config, trace digest, throughput/latency, device stats, WAF, warnings", "output", 1500, 300, 320, 98, "evidence"),
      n("timing", "Four-view Time Breakdown", "additive wall clock; parent latency work; overlapping stage work; resource busy", "output", 1500, 440, 340, 104, "evidence"),
      n("heatmap", "Five-domain Address Heatmap", "workload, HBM, HBF logical/physical, and external physical with traffic sources", "output", 1500, 590, 340, 104, "evidence"),
      n("guards", "Sanity / Conservation Guards", "fail on schema drift, impossible timelines, byte mismatch, capacity overflow, or invalid WAF", "output", 1500, 740, 340, 104, "evidence"),
    ],
    edges: [
      e("synthetic", "trace", "synthetic"),
      e("astra", "trace", "ASTRA adapter"),
      e("qwen", "frontier", "demand"),
      e("frontier", "trace", "memory objects"),
      e("trace", "guard", "artifacts"),
      e("guard", "request", "validated"),
      e("request", "runner", "requests"),
      e("runner", "direct", "direct policy"),
      e("runner", "streaming", "layer schedule"),
      e("direct", "hbm", "HBM pages"),
      e("direct", "hbf", "HBF logical/static"),
      e("streaming", "hbm", "foreground/DMA"),
      e("streaming", "hbf", "backing IO"),
      e("streaming", "external", "no-HBF backing IO"),
      e("runner", "summary", "aggregate"),
      e("hbm", "summary", "device counters"),
      e("hbf", "summary", "device counters"),
      e("external", "summary", "device counters"),
      e("summary", "timing", "timing"),
      e("summary", "heatmap", "heatmap"),
      e("summary", "guards", "invariants"),
    ],
  },
  compositions: {
    id: "compositions",
    name: "Reusable composition and access paths",
    description: "This architecture view contains implementation mechanisms only. Runner-owned experiment cases select their parameters outside the diagram.",
    nodes: [
      n("input", "Validated MemoryRequest Stream", "ordered operations plus optional semantic and dependency hints", "system", 30, 430, 300, 94, "contract"),
      n("window", "Page-transaction Admission", "completion-order credit window; shared or read-only tier-specific; 0 is open loop", "logic", 380, 430, 300, 100, "composition"),
      n("direct", "DirectComposition", "all-HBM, all-HBF, address-split FLAT, or static-direct routing policy", "logic", 750, 210, 330, 104, "composition"),
      n("streaming", "LayerStreamingComposition", "layer residency, backing admission, parity-buffer reuse, prefetch dependencies, and dirty writeback", "logic", 750, 730, 340, 104, "composition"),
      n("hbm", "HBM Device(s)", "HBM-routed pages plus streaming foreground and ping-pong buffers", "hbm", 1140, 80, 300, 98, "device"),
      n("hbf_ftl", "Managed HBF Logical Path", "resident per-stack L2P, write coalescing, checkpoint, GC, and drain", "hbf", 1140, 250, 300, 100, "device"),
      n("static_map", "Static HBF Page Permutation", "capacity-checked bijection for immutable reads that bypass resident translation", "logic", 1140, 450, 310, 100, "composition"),
      n("static_fabric", "Raw HBF Physical Read Fabric", "no L2P lookup, mapping checkpoint, program, drain, or FTL-owned block", "hbf", 1510, 450, 310, 100, "device"),
      n("stream_ctrl", "Shared Layer Residency Controller", "unique-page accounting, parity reuse, prefetch, foreground dependencies, dirty writeback", "logic", 1140, 700, 320, 106, "composition"),
      n("buffers", "Two Dynamic HBM Ping-pong Buffers", "each sized to max per-layer footprint; scratch/metadata compactly HBM-resident", "hbm", 1510, 650, 310, 104, "device"),
      n("d2d", "Per-stack Base-die D2D Links", "independent read/write lanes; prefetch HBF->HBM and writeback HBM->HBF", "flow", 1510, 800, 310, 100, "resource"),
      n("backing", "Full HBF Model/KV Backing", "static initial image, then logical full-page reads/writes after dirty generations", "hbf", 1140, 850, 320, 100, "device"),
      n("external", "Full External Model/KV Backing", "CXL memory or SSD; global E2E credits and address-striped media channels", "external", 1140, 1010, 320, 100, "device"),
      n("external_pipeline", "External Request Pipeline", "M2S -> propagation -> controller -> media -> S2M; earliest-gap resources", "flow", 1510, 1010, 310, 100, "resource"),
      n("result", "CompositionRunResult", "user completion, drain tail, latency views, traffic, stats, spans, and heatmap", "output", 1880, 300, 310, 104, "evidence"),
    ],
    edges: [
      e("input", "window", "ordered parent ops"),
      e("window", "direct", "selected direct policy"),
      e("window", "streaming", "selected layer model"),
      e("direct", "hbm", "HBM-routed pages"),
      e("direct", "hbf_ftl", "managed HBF pages"),
      e("direct", "static_map", "eligible static reads"),
      e("static_map", "static_fabric", "physical pages"),
      e("streaming", "stream_ctrl", "layer/phase schedule"),
      e("stream_ctrl", "buffers", "foreground residency"),
      e("backing", "d2d", "prefetch read"),
      e("d2d", "buffers", "buffer install"),
      e("buffers", "d2d", "dirty full page"),
      e("d2d", "backing", "logical writeback"),
      e("external", "external_pipeline", "prefetch read"),
      e("external_pipeline", "buffers", "buffer install"),
      e("buffers", "external_pipeline", "dirty full page"),
      e("external_pipeline", "external", "writeback"),
      e("hbm", "result", "HBM stats"),
      e("hbf_ftl", "result", "managed HBF stats"),
      e("static_fabric", "result", "static read stats"),
      e("stream_ctrl", "result", "layer/wait stats"),
      e("backing", "result", "backing stats"),
      e("external", "result", "backing stats"),
    ],
  },
  full: {
    id: "full",
    name: "Device hierarchy and physical links",
    description: "Implemented HBM/HBF controller, resource, and media hierarchy. External HBIO and internal TSV traffic are deliberately separate.",
    nodes: [
      n("workload", "Workload / Compute Interface", "memory ops, tensor/KV/page requests", "system", 20, 350, 210, 82, "source"),
      n("front", "Memory Front-end", "normalize, dependencies, tier routing, stats", "system", 280, 350, 230, 92, "contract"),
      n("hbm_ctrl", "HBM Stack Model", "pin rate, DQ width, BL, derived tCK/bandwidth", "hbm", 610, 80, 250, 104),
      n("hbm_timing", "HBM Control Path", "scheduler, row policy, refresh, timing, bus", "logic", 900, 80, 220, 82),
      n("hbm_ch", "Channel", "command and access domain", "hbm", 1190, 20, 170, 70, "resource"),
      n("hbm_pch", "Pseudo-channel", "HBM sub-channel parallelism", "hbm", 1420, 20, 180, 70, "resource"),
      n("hbm_bg", "Bank Group", "tCCD_L / tRRD_L scope", "hbm", 1190, 130, 170, 70, "resource"),
      n("hbm_bank", "Bank", "open row, ready time", "hbm", 1420, 130, 170, 70, "resource"),
      n("hbm_row", "Row / Column / Burst", "row buffer and data burst", "hbm", 1650, 80, 210, 70, "resource"),
      n("hbf_stack", "HBF Stack Model", "flash-as-memory stack geometry and timing", "hbf", 610, 330, 240, 84),
      n("logic_die", "HBF Logic Die (per stack)", "queues, decoded SRAM, resident L2P DRAM, per-die ECC", "hbf", 610, 450, 250, 96),
      n("req_sched", "Logic Request Issue", "ingress queue and per-request dispatch", "logic", 900, 300, 250, 82),
      n("mapping", "Resident L2P Translation", "LPN -> PPN in per-stack DRAM; pipelined lookup", "logic", 900, 420, 260, 88),
      n("write_buf", "Write Buffer / Coalescer", "dirty ranges, read hits, flush to page program", "hbf", 900, 545, 260, 92),
      n("map_batch", "Mapping Checkpoint / Drain", "batch dirty mapping VPNs into final-drain programs", "logic", 1190, 545, 250, 86),
      n("maint", "GC / Wear Policy", "free reserve, victim score, relocation, erase", "maintenance", 1190, 660, 250, 88),
      n("tsu", "Flash Transaction Scheduler", "source queues, die sequencer, read/program/erase issue", "logic", 1190, 420, 270, 92),
      n("hbio", "External HBIO / Internal TSV", "HBIO carries user requests/payload; TSV carries flash commands/raw codewords", "hbf", 1490, 420, 270, 96, "resource"),
      n("fch", "Flash Channel", "command/data bus ready times", "hbf", 1810, 420, 190, 74, "resource"),
      n("fdie", "Flash Die", "source queues, shared sequencer, and per-die ECC pipeline", "media", 1810, 530, 210, 86, "resource"),
      n("plane", "Plane", "program serialization plus read-resource collection", "media", 2070, 530, 220, 82, "media"),
      n("subarray", "Independent Subarray", "one array-sense admission per sense round", "media", 2070, 650, 230, 82, "resource"),
      n("media_lane", "Local Media Lane", "raw codeword transfer; independent lanes may overlap", "media", 2360, 650, 230, 86, "resource"),
      n("page_buffer", "Page-buffer Bank", "independent raw-codeword latch/output serialization", "media", 2650, 650, 240, 88, "resource"),
      n("block", "Block", "sequential program, valid/invalid/free pages, erase count", "media", 2070, 780, 240, 82, "media"),
      n("page", "SLC Page <= 4 KiB", "read/program granularity, owner, and logical mapping", "media", 2380, 780, 230, 82, "media"),
      n("breakdown", "Completion / Trace Evidence", "queue, translation, media, IO, ECC, maintenance, resource busy", "output", 2650, 470, 250, 94, "evidence"),
    ],
    edges: [
      e("workload", "front", "requests"),
      e("front", "hbm_ctrl", "HBM tier"),
      e("hbm_ctrl", "hbm_timing", "commands"),
      e("hbm_timing", "hbm_ch", "ready"),
      e("hbm_ch", "hbm_pch", ""),
      e("hbm_pch", "hbm_bg", ""),
      e("hbm_bg", "hbm_bank", ""),
      e("hbm_bank", "hbm_row", ""),
      e("front", "hbf_stack", "external HBIO request/payload"),
      e("hbf_stack", "logic_die", "local base die"),
      e("logic_die", "req_sched", "issue"),
      e("logic_die", "mapping", "translate"),
      e("logic_die", "write_buf", "SRAM stage"),
      e("req_sched", "mapping", "logical requests"),
      e("mapping", "write_buf", "old PPN / dirty checkpoint"),
      e("write_buf", "map_batch", "flush updates"),
      e("map_batch", "tsu", "mapping page program"),
      e("mapping", "maint", "free pressure"),
      e("maint", "tsu", "GC txns"),
      e("req_sched", "tsu", "read/program txns"),
      e("write_buf", "tsu", "data program"),
      e("tsu", "hbio", "internal TSV command/raw data"),
      e("hbio", "fch", "raw page + OOB"),
      e("fch", "fdie", ""),
      e("fdie", "plane", ""),
      e("plane", "subarray", "sense admission"),
      e("subarray", "media_lane", "raw codeword"),
      e("media_lane", "page_buffer", "latch/output"),
      e("plane", "block", ""),
      e("block", "page", ""),
      e("page", "subarray", "mapped access"),
      e("page_buffer", "breakdown", "latency + busy"),
      e("hbm_row", "breakdown", "latency"),
      e("breakdown", "front", "complete"),
    ],
  },
  hbm_flow: {
    id: "hbm_flow",
    name: "HBM request flow",
    description: "Ramulator-style HBM command and timing path.",
    nodes: [
      n("req", "HBM Request", "addr, size, op, stream", "system", 30, 190, 180, 74, "contract"),
      n("map", "Address Mapper", "addr -> ch/pch/bg/bank/row/col", "hbm", 260, 190, 230, 82, "resource"),
      n("queue", "Read/Write Queues", "admission and ordering", "hbm", 540, 190, 200, 74, "resource"),
      n("sched", "Request Scheduler", "FCFS / FR-FCFS / priority", "hbm", 790, 190, 210, 74, "resource"),
      n("row", "Row State Check", "hit, miss, conflict", "hbm", 1050, 190, 200, 74, "resource"),
      n("prep", "PRE / ACT Path", "tRP, tRAS, tRCD, tRC", "hbm", 1300, 90, 210, 80, "resource"),
      n("cas", "RD / WR", "tCL, tCWL, tCCD, burst", "hbm", 1300, 280, 210, 80, "resource"),
      n("bus", "Pseudo-channel Bus", "derived BL duration, turnaround, contention", "hbm", 1560, 190, 220, 80, "resource"),
      n("refresh", "Refresh Interference", "tREFI, tRFC, REFsb/REFab", "hbm", 1560, 80, 220, 80, "resource"),
      n("done", "Completion", "latency and breakdown", "output", 1560, 310, 220, 74, "evidence"),
    ],
    edges: [
      e("req", "map", ""),
      e("map", "queue", ""),
      e("queue", "sched", ""),
      e("sched", "row", ""),
      e("row", "prep", "miss/conflict"),
      e("prep", "cas", "row ready"),
      e("row", "cas", "row hit"),
      e("cas", "bus", ""),
      e("bus", "refresh", "blocked by"),
      e("bus", "done", ""),
      e("refresh", "done", "stall"),
    ],
  },
  hbf_read: {
    id: "hbf_read",
    name: "HBF read flow",
    description: "Source-aware HBF read: the external request command enters on HBIO; flash commands and raw page+OOB bytes use TSV/channel; ECC produces decoded SRAM/HBIO payload.",
    nodes: [
      n("req", "HBF Read Request", "logical/physical address, bytes", "system", 30, 230, 220, 78, "contract"),
      n("ingress", "External Request Command", "command/address enters the stack over HBIO", "hbf", 290, 230, 240, 78, "resource"),
      n("queue", "Logic-die Queue", "ingress wait and scheduler issue", "hbf", 570, 230, 220, 78, "resource"),
      n("trans", "Resident L2P Lookup", "address generation + pipelined per-stack DRAM response", "logic", 830, 230, 280, 88),
      n("buffer_hit", "Decoded SRAM Buffer Hit", "read/write-buffer payload; no media or ECC pass", "hbf", 1160, 80, 270, 82, "resource"),
      n("split", "Page Transactions", "split into independent page reads", "logic", 1160, 230, 240, 78),
      n("pre", "Internal Flash Command", "logic die -> TSV -> flash channel; never external HBIO", "flow", 1440, 95, 270, 82),
      n("tsu", "Flash Scheduler", "source queue + die sequencer issue_read", "logic", 1760, 95, 250, 82),
      n("array", "Subarray Sense", "one admitted read per subarray in each batch sense round", "media", 1760, 230, 240, 82, "resource"),
      n("media_lane", "Local Media-lane Transfer", "independent raw-codeword lane selected within the plane", "media", 2050, 230, 270, 84, "resource"),
      n("page_buffer", "Page-buffer Bank Output", "independent latch/output bank; serialization is separately visible", "media", 2370, 230, 280, 86, "resource"),
      n("post", "Raw Read Return", "page buffer -> channel -> TSV; page + OOB bytes", "flow", 2700, 230, 290, 84, "resource"),
      n("ecc", "Pipelined ECC Decode", "raw codeword; shared per-die issue II + overlapped latency", "logic", 3040, 230, 280, 82, "resource"),
      n("payload", "Decoded Payload Return", "ECC -> SRAM; user read -> HBIO, internal read stops in SRAM", "hbf", 3370, 230, 300, 88, "resource"),
      n("done", "Read Completion", "latency, raw physical bytes, payload bytes, trace spans", "output", 3720, 230, 270, 84, "evidence"),
    ],
    edges: [
      e("req", "ingress", "external command"),
      e("ingress", "queue", ""),
      e("queue", "trans", ""),
      e("trans", "buffer_hit", "dirty range"),
      e("buffer_hit", "payload", "decoded SRAM payload"),
      e("trans", "split", "PPNs / old PPN"),
      e("split", "pre", "internal command"),
      e("pre", "tsu", ""),
      e("tsu", "array", ""),
      e("array", "media_lane", "raw codeword"),
      e("media_lane", "page_buffer", ""),
      e("page_buffer", "post", ""),
      e("post", "ecc", ""),
      e("ecc", "payload", "decoded page"),
      e("payload", "done", "requested payload bytes"),
    ],
  },
  hbf_write: {
    id: "hbf_write",
    name: "HBF write flow",
    description: "Source-aware HBF write: external user command/payload enters over HBIO and lands in SRAM; destage, mapping, and GC programs continue internally through ECC and raw TSV/channel transfers.",
    nodes: [
      n("req", "HBF Write Request", "logical address, bytes, write policy", "system", 30, 240, 210, 78, "contract"),
      n("ingress", "External Request Command", "command/address enters the stack over HBIO", "hbf", 270, 240, 240, 78, "resource"),
      n("queue", "Logic-die Queue", "ingress wait and scheduler issue", "hbf", 550, 240, 220, 78, "resource"),
      n("lookup", "Lookup Old Mapping", "resident per-stack L2P DRAM supplies old PPN", "logic", 810, 240, 250, 84),
      n("payload", "External Payload In", "exact user bytes: HBIO payload -> decoded SRAM", "hbf", 1100, 80, 270, 82, "resource"),
      n("buffer", "Write Buffer Stage", "dirty range merge in base-die SRAM", "hbf", 1410, 80, 260, 84, "resource"),
      n("flush", "Flush / Drain Decision", "capacity, threshold, completion policy, final drain", "logic", 1710, 80, 270, 84),
      n("partial", "Partial-page Merge", "old-page read -> decoded SRAM; no external HBIO", "flow", 1410, 240, 270, 82),
      n("alloc", "Free Page Allocation", "plane round-robin plus GC reserve", "logic", 2020, 80, 250, 78),
      n("gc_check", "Free-space Check", "low/hard watermark and reserve pressure", "maintenance", 2020, 300, 250, 78),
      n("gc", "GC Relocation + Erase", "internal SRAM read/program path; no external HBIO", "maintenance", 2310, 300, 290, 88),
      n("pre", "Internal Program Command", "logic die -> TSV -> flash channel; no external HBIO", "flow", 2310, 80, 280, 84),
      n("encode", "SRAM + ECC Encode", "full decoded page -> pipelined raw codeword", "logic", 2630, 80, 270, 82),
      n("raw", "Raw Program Data-in", "page + OOB: TSV -> channel -> page-buffer latch", "flow", 2940, 80, 290, 84),
      n("tsu", "Flash Scheduler", "user-destage/mapping/GC source queues", "logic", 3270, 80, 270, 84),
      n("prog", "Array Program", "page-buffer codeword -> non-preemptible plane program", "media", 3580, 80, 280, 82, "media"),
      n("verify", "Program Verify", "verify latency and completion point", "logic", 3580, 210, 250, 78),
      n("map", "Mapping Update", "install new PPN, invalidate old PPN", "logic", 3270, 210, 280, 84),
      n("map_batch", "Dirty Mapping Checkpoint", "batched internal mapping-page program at drain", "logic", 2940, 330, 300, 88),
      n("done", "Write Completion / WAF", "physical payload writes / logical HBF writes + trace spans", "output", 3580, 330, 280, 84, "evidence"),
    ],
    edges: [
      e("req", "ingress", "external command"),
      e("ingress", "queue", ""),
      e("queue", "lookup", ""),
      e("lookup", "payload", "user bytes"),
      e("payload", "buffer", "decoded SRAM"),
      e("lookup", "partial", "old PPN"),
      e("buffer", "flush", ""),
      e("flush", "partial", "sub-page"),
      e("partial", "alloc", "merged page"),
      e("flush", "alloc", "full page"),
      e("alloc", "gc_check", "pressure"),
      e("gc_check", "gc", "if low"),
      e("gc", "pre", "internal GC program"),
      e("alloc", "pre", "user destage"),
      e("pre", "encode", "decoded SRAM page"),
      e("encode", "raw", "raw page + OOB"),
      e("raw", "tsu", ""),
      e("tsu", "prog", ""),
      e("prog", "verify", ""),
      e("verify", "map", ""),
      e("map", "map_batch", "dirty checkpoint"),
      e("map_batch", "pre", "internal mapping program"),
      e("map_batch", "done", "deferred"),
      e("map", "done", "logical mapping visible"),
    ],
  },
  observability: {
    id: "observability",
    name: "Observability and reliability contract",
    description: "What every E2E run must expose, how each timing view is interpreted, and which conservation/provenance checks prevent misleading results.",
    nodes: [
      n("request", "MemoryRequest + PhysicalRequest", "parent workload op and device-facing page transactions", "system", 30, 300, 300, 92, "contract"),
      n("completion", "PhysicalCompletion", "arrival/start/finish, logical/physical bytes, Breakdown, and TraceSpan list", "output", 390, 170, 320, 100, "evidence"),
      n("hbm_stats", "HBM Device Statistics", "traffic, commands, queue/row/bus/refresh work, and resource occupancy", "hbm", 390, 20, 320, 94, "evidence"),
      n("hbf_stats", "HBF Device Statistics", "logical/physical traffic, program classes, GC, ECC/link/media work, occupancy", "hbf", 390, 320, 330, 100, "evidence"),
      n("external_backing_stats", "External-backing Statistics", "media/link traffic, queue waits, fixed latency work, busy time, and occupancy", "external", 820, 20, 340, 100, "evidence"),
      n("composition_stats", "Composition Statistics", "tier traffic, backing-credit peaks/waits, dependency waits, streaming residency, user/drain frontiers", "logic", 390, 480, 350, 104, "evidence"),
      n("traffic", "Source-attributed Address Records", "workload, HBM, HBF, external, static, mapping, GC, and buffer activity", "flow", 390, 650, 350, 104, "evidence"),
      n("waf", "Canonical HBF Device WAF", "physical_write_bytes / logical_write_bytes; undefined when denominator is zero", "output", 820, 320, 320, 96, "evidence"),
      n("heatmap_acc", "Bounded AddressHeatmap", "byte-conserving records into fixed bins with canonical domains/sources/regions", "output", 820, 650, 340, 100, "evidence"),
      n("summary", "Schema-v12 Scenario Summary", "single authoritative artifact with resolved config, trace digest, metrics, snapshots, warnings", "output", 1240, 300, 350, 106, "evidence"),
      n("wall", "Additive Wall Clock", "non-overlapping offered span + completion tail + drain tail = makespan", "output", 1680, 20, 330, 98, "evidence"),
      n("latency", "Parent Latency Work", "service, offered, and source latency sums/distributions; operations may overlap", "output", 1680, 160, 330, 98, "evidence"),
      n("stage", "Overlapping Stage Work", "summed device/controller stage durations; not percentages of makespan", "output", 1680, 300, 330, 98, "evidence"),
      n("busy", "Resource Busy / Capacity-time", "busy_ns / (resource_count × active_span_ns), plus effective parallelism", "output", 1680, 440, 340, 100, "evidence"),
      n("domains", "Five Canonical Address Domains", "workload logical · HBM physical · HBF logical/physical · external physical", "output", 1240, 650, 350, 96, "evidence"),
      n("time_html", "Time-breakdown HTML", "cross-scenario bars, component definitions, totals, and resource utilization", "output", 2100, 230, 320, 96, "evidence"),
      n("heatmap_html", "Address-heatmap HTML", "read/write/erase colors, full-capacity view, source and region tables", "output", 1680, 650, 340, 96, "evidence"),
      n("provenance", "Parameter / Artifact Provenance", "evidence grade, resolved config, trace and manifest hashes, model identity", "output", 1240, 820, 350, 96, "evidence"),
      n("sanity", "Fail-closed E2E Guards", "timeline, schema, capacity, traffic conservation, placement, WAF, and replay identity", "output", 2100, 500, 330, 104, "evidence"),
    ],
    edges: [
      e("request", "completion", "device completion"),
      e("request", "traffic", "logical access"),
      e("completion", "summary", "latency + spans"),
      e("hbm_stats", "summary", "HBM counters"),
      e("hbf_stats", "summary", "HBF counters"),
      e("external_backing_stats", "summary", "external counters"),
      e("composition_stats", "summary", "policy counters"),
      e("hbf_stats", "waf", "two byte counters"),
      e("waf", "summary", "one WAF definition"),
      e("traffic", "heatmap_acc", "record"),
      e("heatmap_acc", "domains", "snapshot"),
      e("domains", "summary", "embedded snapshot"),
      e("summary", "wall", ""),
      e("summary", "latency", ""),
      e("summary", "stage", ""),
      e("summary", "busy", ""),
      e("wall", "time_html", ""),
      e("latency", "time_html", ""),
      e("stage", "time_html", ""),
      e("busy", "time_html", ""),
      e("domains", "heatmap_html", ""),
      e("summary", "sanity", "verify"),
      e("provenance", "sanity", "verify"),
    ],
  },
};

const REPO_ROOT = (new URLSearchParams(window.location.search).get("repo") || "")
  .replace(/\/+$/, "");
const EDITOR_PROTOCOL = "vscode";
const DEFAULT_CODE_REF = "planned: no C++ entity yet";

const CODE_REFS = {
  "system_map.synthetic": "tools/generate_synthetic_trace.py:372::generate",
  "system_map.astra": "tools/replay_astra_trace.py:141::load_workload_manifest",
  "system_map.qwen": "tools/prepare_qwen_bailian_workload.py:748::prepare_suite / tools/verify_qwen_bailian_workload.py:252::verify_suite",
  "system_map.frontier": "tools/export_frontier_memory_trace.py:1689::export_memory_trace",
  "system_map.trace": "src/tools/scenario_compare.cpp:638::parse_trace_line",
  "system_map.guard": "tools/replay_astra_trace.py:141::load_workload_manifest / src/physical/hybrid/composition_common.cpp:57::validate_memory_requests",
  "system_map.request": "src/physical/hybrid/composition_common.hpp:45::MemoryRequest",
  "system_map.runner": "src/tools/scenario_compare.cpp:7872::main",
  "system_map.direct": "src/physical/hybrid/direct_composition.hpp:15::DirectPolicy",
  "system_map.streaming": "src/physical/hybrid/layer_streaming_composition.hpp:219::LayerStreamingComposition",
  "system_map.hbm": "src/physical/hbm/hbm_device.hpp:164::HbmDevice",
  "system_map.hbf": "src/physical/hbf/hbf_device.hpp:422::HbfDevice",
  "system_map.external": "src/physical/external/external_backing_device.hpp:132::ExternalBackingDevice",
  "system_map.summary": "src/tools/scenario_compare.cpp:6312::write_summary_json",
  "system_map.timing": "tools/time_breakdown_report.py:1502::build_time_breakdown_report",
  "system_map.heatmap": "tools/plot_address_heatmap.py:1006::render_html",
  "system_map.guards": "tools/check_ec_sanity.py:523::main / tools/check_parameter_provenance.py:192::main",

  "compositions.input": "src/physical/hybrid/composition_common.hpp:45::MemoryRequest",
  "compositions.window": "src/physical/hybrid/composition_common.hpp:98::ClosedLoopWindow",
  "compositions.direct": "src/physical/hybrid/direct_composition.hpp:15::DirectPolicy",
  "compositions.streaming": "src/physical/hybrid/layer_streaming_composition.hpp:219::LayerStreamingComposition",
  "compositions.hbm": "src/physical/hbm/hbm_device.hpp:164::HbmDevice",
  "compositions.hbf_ftl": "src/physical/hbf/hbf_device.cpp:4456::HbfDevice::lookup_lpn / src/physical/hbf/hbf_device.cpp:4293::HbfDevice::access_resident_mapping",
  "compositions.static_map": "src/physical/hybrid/composition_common.cpp:239::map_static_hbf_page_addr",
  "compositions.static_fabric": "src/physical/hbf/hbf_device.cpp:2313::HbfDevice::issue",
  "compositions.stream_ctrl": "src/physical/hybrid/layer_streaming_composition.hpp:219::LayerStreamingComposition",
  "compositions.buffers": "src/physical/hybrid/layer_streaming_composition.hpp:74::LayerStreamingStats",
  "compositions.d2d": "src/physical/hybrid/composition_common.hpp:202::BaseDieLink",
  "compositions.backing": "src/physical/hbf/hbf_device.hpp:422::HbfDevice",
  "compositions.external": "src/physical/external/external_backing_device.hpp:132::ExternalBackingDevice",
  "compositions.external_pipeline": "src/physical/external/external_backing_device.cpp:301::ExternalBackingDevice::schedule / src/physical/external/external_backing_device.cpp:177::ExternalBackingDevice::SerialResourceTimeline::reserve",
  "compositions.result": "src/physical/hybrid/composition_common.hpp:227::CompositionRunResult",

  "full.workload": "src/physical/hybrid/composition_common.hpp:45::MemoryRequest",
  "full.front": "src/physical/physical_types.hpp:81::PhysicalRequest / src/physical/physical_types.hpp:188::PhysicalCompletion",
  "full.hbm_ctrl": "src/physical/hbm/hbm_device.hpp:54::HbmConfig / src/physical/hbm/hbm_device.hpp:164::HbmDevice",
  "full.hbm_timing": "src/physical/hbm/hbm_device.cpp:1359::HbmDevice::pick_next / src/physical/hbm/hbm_device.cpp:1823::HbmDevice::schedule_command",
  "full.hbm_ch": "src/physical/hbm/hbm_device.hpp:125::HbmAddress::channel / src/physical/hbm/hbm_device.cpp:533::HbmDevice::decode",
  "full.hbm_pch": "src/physical/hbm/hbm_device.hpp:227::HbmDevice::PseudoChannelState",
  "full.hbm_bg": "src/physical/hbm/hbm_device.hpp:127::HbmAddress::bank_group",
  "full.hbm_bank": "src/physical/hbm/hbm_device.hpp:200::HbmDevice::BankState",
  "full.hbm_row": "src/physical/hbm/hbm_device.hpp:129::HbmAddress::row / src/physical/hbm/hbm_device.cpp:1592::HbmDevice::service_request",
  "full.hbf_stack": "src/physical/hbf/hbf_device.hpp:29::HbfConfig / src/physical/hbf/hbf_device.hpp:422::HbfDevice",
  "full.logic_die": "src/physical/hbf/hbf_device.hpp:692::HbfDevice::LogicDieState",
  "full.req_sched": "src/physical/hbf/hbf_device.cpp:2313::HbfDevice::issue",
  "full.mapping": "src/physical/hbf/hbf_device.cpp:4456::HbfDevice::lookup_lpn / src/physical/hbf/hbf_device.cpp:4293::HbfDevice::access_resident_mapping",
  "full.write_buf": "src/physical/hbf/hbf_device.cpp:4795::HbfDevice::stage_write_buffer_range / src/physical/hbf/hbf_device.cpp:4901::HbfDevice::flush_write_buffer_entry",
  "full.map_batch": "src/physical/hbf/hbf_device.cpp:4391::HbfDevice::flush_dirty_mapping_page / src/physical/hbf/hbf_device.cpp:4249::HbfDevice::flush_all_dirty_mapping_pages",
  "full.maint": "src/physical/hbf/hbf_device.cpp:5545::HbfDevice::maybe_run_gc / src/physical/hbf/hbf_device.cpp:5497::HbfDevice::choose_gc_victim",
  "full.tsu": "src/physical/hbf/hbf_device.cpp:6050::HbfDevice::schedule_flash_transaction",
  "full.hbio": "src/physical/hbf/hbf_device.cpp:4692::HbfDevice::schedule_external_request_command / src/physical/hbf/hbf_device.cpp:4635::HbfDevice::schedule_external_write_ingress / src/physical/hbf/hbf_device.cpp:4758::HbfDevice::schedule_external_read_egress / src/physical/hbf/hbf_device.cpp:6094::HbfDevice::schedule_command_path",
  "full.fch": "src/physical/hbf/hbf_device.hpp:683::HbfDevice::ChannelState",
  "full.fdie": "src/physical/hbf/hbf_device.hpp:660::HbfDevice::DieState",
  "full.plane": "src/physical/hbf/hbf_device.hpp:610::HbfDevice::PlaneState",
  "full.subarray": "src/physical/hbf/hbf_device.hpp:615::HbfDevice::PlaneState::SubarrayState",
  "full.media_lane": "src/physical/hbf/hbf_device.hpp:621::HbfDevice::PlaneState::MediaLaneState",
  "full.page_buffer": "src/physical/hbf/hbf_device.hpp:627::HbfDevice::PlaneState::PageBufferBankState",
  "full.block": "src/physical/hbf/hbf_device.hpp:742::HbfDevice::BlockState",
  "full.page": "src/physical/hbf/hbf_device.hpp:735::HbfDevice::PageState",
  "full.breakdown": "src/physical/physical_types.hpp:94::Breakdown / src/physical/physical_types.hpp:174::TraceSpan",

  "hbm_flow.req": "src/physical/physical_types.hpp:81::PhysicalRequest",
  "hbm_flow.map": "src/physical/hbm/hbm_device.cpp:533::HbmDevice::decode / src/physical/hbm/hbm_device.cpp:581::HbmDevice::encode",
  "hbm_flow.queue": "src/physical/hbm/hbm_device.hpp:227::HbmDevice::PseudoChannelState",
  "hbm_flow.sched": "src/physical/hbm/hbm_device.cpp:1359::HbmDevice::pick_next",
  "hbm_flow.row": "src/physical/hbm/hbm_device.hpp:200::HbmDevice::BankState",
  "hbm_flow.prep": "src/physical/hbm/hbm_device.cpp:1592::HbmDevice::service_request / src/physical/hbm/hbm_device.cpp:1823::HbmDevice::schedule_command",
  "hbm_flow.cas": "src/physical/hbm/hbm_device.cpp:1592::HbmDevice::service_request / src/physical/hbm/hbm_device.cpp:1823::HbmDevice::schedule_command",
  "hbm_flow.bus": "src/physical/hbm/hbm_device.hpp:231::HbmDevice::PseudoChannelState::bus_ready_ns",
  "hbm_flow.refresh": "src/physical/hbm/hbm_device.cpp:1994::HbmDevice::reserve_refresh_free_window",
  "hbm_flow.done": "src/physical/physical_types.hpp:188::PhysicalCompletion",

  "hbf_read.req": "src/physical/physical_types.hpp:81::PhysicalRequest",
  "hbf_read.ingress": "src/physical/hbf/hbf_device.cpp:4692::HbfDevice::schedule_external_request_command",
  "hbf_read.queue": "src/physical/hbf/hbf_device.cpp:2313::HbfDevice::issue",
  "hbf_read.trans": "src/physical/hbf/hbf_device.cpp:4456::HbfDevice::lookup_lpn / src/physical/hbf/hbf_device.cpp:4293::HbfDevice::access_resident_mapping",
  "hbf_read.buffer_hit": "src/physical/hbf/hbf_device.cpp:3679::HbfDevice::serve_read_from_read_buffer / src/physical/hbf/hbf_device.cpp:4522::HbfDevice::serve_read_from_write_buffer",
  "hbf_read.split": "src/physical/hbf/hbf_device.cpp:2313::HbfDevice::issue",
  "hbf_read.tsu": "src/physical/hbf/hbf_device.cpp:6050::HbfDevice::schedule_flash_transaction",
  "hbf_read.pre": "src/physical/hbf/hbf_device.cpp:6094::HbfDevice::schedule_command_path",
  "hbf_read.array": "src/physical/hbf/hbf_device.cpp:6233::HbfDevice::schedule_read_page",
  "hbf_read.media_lane": "src/physical/hbf/hbf_device.hpp:621::HbfDevice::PlaneState::MediaLaneState",
  "hbf_read.page_buffer": "src/physical/hbf/hbf_device.hpp:627::HbfDevice::PlaneState::PageBufferBankState",
  "hbf_read.post": "src/physical/hbf/hbf_device.cpp:6233::HbfDevice::schedule_read_page",
  "hbf_read.ecc": "src/physical/hbf/hbf_device.cpp:6143::HbfDevice::schedule_ecc",
  "hbf_read.payload": "src/physical/hbf/hbf_device.cpp:4723::HbfDevice::schedule_sram_transfer / src/physical/hbf/hbf_device.cpp:4758::HbfDevice::schedule_external_read_egress",
  "hbf_read.done": "src/physical/physical_types.hpp:188::PhysicalCompletion",

  "hbf_write.req": "src/physical/physical_types.hpp:81::PhysicalRequest",
  "hbf_write.ingress": "src/physical/hbf/hbf_device.cpp:4692::HbfDevice::schedule_external_request_command",
  "hbf_write.queue": "src/physical/hbf/hbf_device.cpp:2313::HbfDevice::issue",
  "hbf_write.lookup": "src/physical/hbf/hbf_device.cpp:4456::HbfDevice::lookup_lpn / src/physical/hbf/hbf_device.cpp:4293::HbfDevice::access_resident_mapping",
  "hbf_write.payload": "src/physical/hbf/hbf_device.cpp:4635::HbfDevice::schedule_external_write_ingress",
  "hbf_write.buffer": "src/physical/hbf/hbf_device.cpp:4795::HbfDevice::stage_write_buffer_range",
  "hbf_write.flush": "src/physical/hbf/hbf_device.cpp:4901::HbfDevice::flush_write_buffer_entry / src/physical/hbf/hbf_device.cpp:2110::HbfDevice::drain_pending",
  "hbf_write.partial": "src/physical/hbf/hbf_device.cpp:4901::HbfDevice::flush_write_buffer_entry / src/physical/hbf/hbf_device.cpp:2313::HbfDevice::issue",
  "hbf_write.alloc": "src/physical/hbf/hbf_device.cpp:5098::HbfDevice::allocate_free_page",
  "hbf_write.gc_check": "src/physical/hbf/hbf_device.cpp:5545::HbfDevice::maybe_run_gc",
  "hbf_write.gc": "src/physical/hbf/hbf_device.cpp:5696::HbfDevice::relocate_and_erase_block",
  "hbf_write.tsu": "src/physical/hbf/hbf_device.cpp:6050::HbfDevice::schedule_flash_transaction",
  "hbf_write.pre": "src/physical/hbf/hbf_device.cpp:6094::HbfDevice::schedule_command_path",
  "hbf_write.encode": "src/physical/hbf/hbf_device.cpp:6143::HbfDevice::schedule_ecc",
  "hbf_write.raw": "src/physical/hbf/hbf_device.cpp:6497::HbfDevice::schedule_program_page",
  "hbf_write.prog": "src/physical/hbf/hbf_device.cpp:6497::HbfDevice::schedule_program_page",
  "hbf_write.verify": "src/physical/hbf/hbf_device.cpp:6497::HbfDevice::schedule_program_page",
  "hbf_write.map": "src/physical/hbf/hbf_device.cpp:4901::HbfDevice::flush_write_buffer_entry / src/physical/hbf/hbf_device.cpp:1288::HbfDevice::schedule_lpn_mapping_commit",
  "hbf_write.map_batch": "src/physical/hbf/hbf_device.cpp:4249::HbfDevice::flush_all_dirty_mapping_pages / src/physical/hbf/hbf_device.cpp:4391::HbfDevice::flush_dirty_mapping_page",
  "hbf_write.done": "src/physical/physical_types.hpp:94::Breakdown / src/physical/hbf/hbf_device.hpp:142::HbfStats",

  "observability.request": "src/physical/hybrid/composition_common.hpp:45::MemoryRequest / src/physical/physical_types.hpp:81::PhysicalRequest",
  "observability.completion": "src/physical/physical_types.hpp:188::PhysicalCompletion / src/physical/physical_types.hpp:94::Breakdown / src/physical/physical_types.hpp:174::TraceSpan",
  "observability.hbm_stats": "src/physical/hbm/hbm_device.hpp:135::HbmStats",
  "observability.hbf_stats": "src/physical/hbf/hbf_device.hpp:142::HbfStats",
  "observability.external_backing_stats": "src/physical/external/external_backing_device.hpp:74::ExternalBackingStats",
  "observability.composition_stats": "src/physical/hybrid/composition_common.hpp:227::CompositionRunResult",
  "observability.traffic": "src/physical/address_heatmap.hpp:144::AddressTrafficRecord",
  "observability.waf": "src/physical/hbf/hbf_device.cpp:312::HbfStats::waf",
  "observability.heatmap_acc": "src/physical/address_heatmap.hpp:194::AddressHeatmap",
  "observability.summary": "src/tools/scenario_compare.cpp:6312::write_summary_json",
  "observability.wall": "tools/time_breakdown_report.py:370::_validate_wall",
  "observability.latency": "tools/time_breakdown_report.py:428::_validate_latency",
  "observability.stage": "tools/time_breakdown_report.py:758::_validate_stage_work",
  "observability.busy": "tools/time_breakdown_report.py:849::_validate_resources",
  "observability.domains": "src/physical/address_heatmap.hpp:19::AddressDomain / src/physical/address_heatmap.hpp:182::AddressHeatmapSnapshot",
  "observability.time_html": "tools/plot_time_breakdown.py:770::render_html",
  "observability.heatmap_html": "tools/plot_address_heatmap.py:1006::render_html",
  "observability.provenance": "tools/check_parameter_provenance.py:70::validate_registry",
  "observability.sanity": "tools/check_ec_sanity.py:291::validate_summary",
};

function n(
  id,
  label,
  subtitle,
  type,
  x,
  y,
  w,
  h,
  level = DEFAULT_LEVEL_BY_TYPE[type] || "resource",
) {
  return { id, label, subtitle, type, level, x, y, w, h, codeRef: "" };
}

function e(source, target, label = "") {
  return { id: `edge_${source}_${target}_${Math.random().toString(36).slice(2, 8)}`, source, target, label };
}

const app = {
  diagrams: enrichDiagrams(clone(templates)),
  currentId: "system_map",
  selected: null,
  mode: "select",
  connectSource: null,
  routeMode: "orthogonal",
  snap: true,
  wheelSensitivity: 0.5,
  panels: {
    left: { w: 280, collapsed: false },
    right: { w: 360, collapsed: true },
  },
  viewport: { x: 40, y: 40, scale: 0.62 },
  drag: null,
  undoStack: [],
  redoStack: [],
};

const GRID = 10;
const SETTINGS_KEY = "hbfsim.architecture.editor.settings.v2";
const SUBTITLE_MIN_SCALE = 0.45;
const PANEL_MIN = 180;
const PANEL_MAX = 560;
const MIN_W = 90;
const MIN_H = 70;

const el = {
  svg: document.getElementById("canvas"),
  bg: document.getElementById("canvasBg"),
  viewport: document.getElementById("viewport"),
  nodesLayer: document.getElementById("nodesLayer"),
  edgesLayer: document.getElementById("edgesLayer"),
  edgeLabelsLayer: document.getElementById("edgeLabelsLayer"),
  diagramList: document.getElementById("diagramList"),
  legend: document.getElementById("legend"),
  levelLegend: document.getElementById("levelLegend"),
  modeHint: document.getElementById("modeHint"),
  statusText: document.getElementById("statusText"),
  positionText: document.getElementById("positionText"),
  zoomLabel: document.getElementById("zoomLabel"),
  selectModeBtn: document.getElementById("selectModeBtn"),
  connectModeBtn: document.getElementById("connectModeBtn"),
  addNodeBtn: document.getElementById("addNodeBtn"),
  deleteBtn: document.getElementById("deleteBtn"),
  zoomInBtn: document.getElementById("zoomInBtn"),
  zoomOutBtn: document.getElementById("zoomOutBtn"),
  fitBtn: document.getElementById("fitBtn"),
  routeBtn: document.getElementById("routeBtn"),
  snapBtn: document.getElementById("snapBtn"),
  undoBtn: document.getElementById("undoBtn"),
  redoBtn: document.getElementById("redoBtn"),
  wheelSens: document.getElementById("wheelSens"),
  wheelSensVal: document.getElementById("wheelSensVal"),
  nodeTooltip: document.getElementById("nodeTooltip"),
  leftPanel: document.getElementById("leftPanel"),
  rightPanel: document.getElementById("rightPanel"),
  resizerLeft: document.getElementById("resizerLeft"),
  resizerRight: document.getElementById("resizerRight"),
  toggleLeft: document.getElementById("toggleLeft"),
  toggleRight: document.getElementById("toggleRight"),
  saveBtn: document.getElementById("saveBtn"),
  exportBtn: document.getElementById("exportBtn"),
  importBtn: document.getElementById("importBtn"),
  importFile: document.getElementById("importFile"),
  resetBtn: document.getElementById("resetBtn"),
  applyJsonBtn: document.getElementById("applyJsonBtn"),
  jsonBox: document.getElementById("jsonBox"),
  emptyInspector: document.getElementById("emptyInspector"),
  nodeInspector: document.getElementById("nodeInspector"),
  edgeInspector: document.getElementById("edgeInspector"),
  nodeLabel: document.getElementById("nodeLabel"),
  nodeSubtitle: document.getElementById("nodeSubtitle"),
  nodeCodeRef: document.getElementById("nodeCodeRef"),
  nodeOpenCodeBtn: document.getElementById("nodeOpenCodeBtn"),
  nodeType: document.getElementById("nodeType"),
  nodeLevel: document.getElementById("nodeLevel"),
  nodeX: document.getElementById("nodeX"),
  nodeY: document.getElementById("nodeY"),
  nodeW: document.getElementById("nodeW"),
  nodeH: document.getElementById("nodeH"),
  edgeLabel: document.getElementById("edgeLabel"),
  edgeSource: document.getElementById("edgeSource"),
  edgeTarget: document.getElementById("edgeTarget"),
  newEdgeSource: document.getElementById("newEdgeSource"),
  newEdgeTarget: document.getElementById("newEdgeTarget"),
  newEdgeLabel: document.getElementById("newEdgeLabel"),
  createEdgeBtn: document.getElementById("createEdgeBtn"),
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function codeRefFor(diagram, node) {
  return node.codeRef || CODE_REFS[`${diagram.id}.${node.id}`] || DEFAULT_CODE_REF;
}

function shouldRefreshCodeRef(diagram, node) {
  const key = `${diagram.id}.${node.id}`;
  const current = String(node.codeRef || "").trim();
  if (!current) return true;
  if (!Object.prototype.hasOwnProperty.call(CODE_REFS, key)) return false;
  if (current === DEFAULT_CODE_REF && CODE_REFS[key] !== DEFAULT_CODE_REF) return true;
  return current.startsWith("src/") && !codeRefIsOpenable(current);
}

function enrichDiagram(diagram) {
  if (!diagram || !Array.isArray(diagram.nodes)) return diagram;
  diagram.nodes.forEach((node) => {
    if (!Object.prototype.hasOwnProperty.call(LEVEL_META, node.level)) {
      node.level = DEFAULT_LEVEL_BY_TYPE[node.type] || "resource";
    }
    if (typeof node.codeRef !== "string" || shouldRefreshCodeRef(diagram, node)) {
      node.codeRef = CODE_REFS[`${diagram.id}.${node.id}`] || DEFAULT_CODE_REF;
    }
  });
  return diagram;
}

function enrichDiagrams(diagrams) {
  Object.values(diagrams || {}).forEach(enrichDiagram);
  return diagrams;
}

function nodeCodeRef(node) {
  return codeRefFor(currentDiagram(), node);
}

function nodeLevelMeta(node) {
  return LEVEL_META[node.level] || LEVEL_META.resource;
}

function compactCodeRef(ref) {
  return ref
    .replace(/^src\/physical\//, "")
    .replace(/^src\//, "")
    .replace("::HbfDevice::", "::")
    .replace("::HbmDevice::", "::");
}

function parseCodeRefTarget(ref) {
  const parts = String(ref || "")
    .split(/\s+\/\s+/)
    .map((part) => part.trim())
    .filter(Boolean);
  const firstConcrete = parts.find((part) => !part.toLowerCase().startsWith("planned:"));
  if (!firstConcrete) return null;

  const symbolStart = firstConcrete.indexOf("::");
  const location = symbolStart >= 0 ? firstConcrete.slice(0, symbolStart) : firstConcrete;
  const symbol = symbolStart >= 0 ? firstConcrete.slice(symbolStart + 2) : "";
  const match = location.match(/^(.+):(\d+)(?::(\d+))?$/);
  if (!match) return null;

  const relativePath = match[1].replace(/^\.\//, "");
  const path = relativePath.startsWith("/")
    ? relativePath
    : (REPO_ROOT ? `${REPO_ROOT}/${relativePath}` : "");
  if (!path) return null;
  const line = Math.max(1, Number(match[2]) || 1);
  const column = Math.max(1, Number(match[3]) || 1);
  return {
    path,
    line,
    column,
    symbol,
    label: `${relativePath}:${line}${symbol ? `::${symbol}` : ""}`,
  };
}

function codeRefIsOpenable(ref) {
  return Boolean(parseCodeRefTarget(ref));
}

function editorHrefForCodeRef(ref) {
  const target = parseCodeRefTarget(ref);
  if (!target) return "";
  return `${EDITOR_PROTOCOL}://file${encodeURI(target.path)}:${target.line}:${target.column}`;
}

function openCodeRef(ref) {
  const target = parseCodeRefTarget(ref);
  if (!target) {
    setStatus(REPO_ROOT
      ? "No concrete code definition for this entity yet"
      : "Add ?repo=/absolute/path/to/HBFSim to enable code links");
    return;
  }
  const anchor = document.createElement("a");
  anchor.href = editorHrefForCodeRef(ref);
  anchor.rel = "noreferrer";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setStatus(`Opening ${target.label}`);
}

function syncCodeOpenButton() {
  const canOpen = codeRefIsOpenable(el.nodeCodeRef.value);
  el.nodeOpenCodeBtn.disabled = !canOpen;
  el.nodeOpenCodeBtn.title = canOpen ? "Open code definition" : "No concrete code definition yet";
}

function currentDiagram() {
  return app.diagrams[app.currentId];
}

function nodeById(id) {
  return currentDiagram().nodes.find((node) => node.id === id);
}

function edgeById(id) {
  return currentDiagram().edges.find((edge) => edge.id === id);
}

function setStatus(message) {
  el.statusText.textContent = message;
}

function setMode(mode) {
  app.mode = mode;
  app.connectSource = null;
  el.selectModeBtn.classList.toggle("active", mode === "select");
  el.connectModeBtn.classList.toggle("active", mode === "connect");
  el.modeHint.textContent =
    mode === "connect"
      ? "Connect mode: click a source node, then click a target node."
      : "Select and drag nodes. Drag blank canvas to pan. Use mouse wheel to zoom.";
  setStatus(mode === "connect" ? "Connect mode active" : "Select mode active");
}

function renderAll() {
  renderDiagramList();
  renderLegend();
  renderCanvas();
  renderInspector();
  syncJsonBox();
  updateZoomLabel();
  updateRouteButton();
  updateHistoryButtons();
}

function renderDiagramList() {
  el.diagramList.replaceChildren();
  Object.values(app.diagrams).forEach((diagram) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `diagram-button${diagram.id === app.currentId ? " active" : ""}`;
    button.dataset.diagramId = diagram.id;
    button.innerHTML = `<strong>${escapeHtml(diagram.name)}</strong><span>${escapeHtml(diagram.description)}</span>`;
    button.addEventListener("click", () => {
      app.currentId = diagram.id;
      app.selected = null;
      app.connectSource = null;
      fitView();
      renderAll();
    });
    el.diagramList.appendChild(button);
  });
}

function renderLegend() {
  el.legend.replaceChildren();
  Object.entries(TYPE_META).forEach(([type, meta]) => {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-swatch" style="background:${meta.color}"></span><span>${escapeHtml(meta.label)}</span>`;
    el.legend.appendChild(item);
  });
  el.levelLegend.replaceChildren();
  Object.entries(LEVEL_META).forEach(([level, meta]) => {
    const item = document.createElement("div");
    item.className = "level-item";
    item.dataset.level = level;
    item.innerHTML = `<span class="level-code">${escapeHtml(meta.code)}</span><span><strong>${escapeHtml(meta.label)}</strong><small>${escapeHtml(meta.detail)}</small></span>`;
    el.levelLegend.appendChild(item);
  });
}

function renderCanvas() {
  el.viewport.setAttribute("transform", `translate(${app.viewport.x} ${app.viewport.y}) scale(${app.viewport.scale})`);
  el.nodesLayer.replaceChildren();
  el.edgesLayer.replaceChildren();
  el.edgeLabelsLayer.replaceChildren();
  const routeContext = buildRouteContext(currentDiagram());

  // Paint order (bottom -> top): edge paths, node boxes, edge labels.
  // Labels sit in their own top layer so node boxes can never obscure them.
  const labelSpecs = [];
  let selectedEdgeRoute = null;
  currentDiagram().edges.forEach((edge) => {
    const source = nodeById(edge.source);
    const target = nodeById(edge.target);
    if (!source || !target) return;
    const route =
      app.routeMode === "orthogonal" ? orthogonalRoute(edge, source, target, routeContext) : curvedRoute(edge, source, target);
    el.edgesLayer.appendChild(edgePathElement(edge, route));
    if (isSelected("edge", edge.id)) selectedEdgeRoute = route;
    if (edge.label) {
      const anchor = routeLabelPoint(route, edge.label, currentDiagram().nodes);
      const off = edge.labelOffset || { dx: 0, dy: 0 };
      const pinned = Boolean(off.dx || off.dy);
      labelSpecs.push({
        edge,
        anchor,
        mid: { x: anchor.x + off.dx, y: anchor.y + off.dy },
        width: labelWidth(edge.label),
        height: 22,
        pinned,
      });
    }
  });
  separateLabels(labelSpecs, labelScale());
  labelSpecs.forEach((spec) => el.edgeLabelsLayer.appendChild(edgeLabelElement(spec)));

  // Endpoint handles for the selected edge, drawn on top so they're grabbable over nodes.
  if (selectedEdgeRoute && app.selected && app.selected.type === "edge") {
    const selectedEdge = edgeById(app.selected.id);
    if (selectedEdge) {
      const pts = selectedEdgeRoute.points;
      el.edgeLabelsLayer.appendChild(endpointHandle(selectedEdge, "source", pts[0]));
      el.edgeLabelsLayer.appendChild(endpointHandle(selectedEdge, "target", pts[pts.length - 1]));
    }
  }

  currentDiagram().nodes.forEach((node) => {
    el.nodesLayer.appendChild(nodeElement(node));
  });
}

function selectEdgeOnMousedown(edge) {
  return (event) => {
    event.stopPropagation();
    select("edge", edge.id);
  };
}

function edgePathElement(edge, route) {
  const group = svgEl("g", {
    class: `edge ${isSelected("edge", edge.id) ? "selected" : ""}`,
    "data-id": edge.id,
  });
  const path = routeToPath(route);
  group.appendChild(svgEl("path", { class: "edge-hit", d: path }));
  group.appendChild(svgEl("path", { class: "edge-path", d: path }));
  group.addEventListener("mousedown", selectEdgeOnMousedown(edge));
  group.addEventListener("dblclick", (event) => {
    event.stopPropagation();
    if (edge.sourceAnchor || edge.targetAnchor) {
      recordHistory();
      delete edge.sourceAnchor;
      delete edge.targetAnchor;
      renderAll();
      setStatus("Edge endpoints reset to auto-routing");
    }
  });
  return group;
}

// A draggable handle at one end of the selected edge. Constant ~7px on screen.
function endpointHandle(edge, end, point) {
  const handle = svgEl("circle", {
    class: `endpoint-handle endpoint-${end}`,
    cx: point.x,
    cy: point.y,
    r: 7 / app.viewport.scale,
  });
  handle.addEventListener("mousedown", (event) => {
    event.stopPropagation();
    select("edge", edge.id);
    app.drag = { kind: "endpoint", edgeId: edge.id, end, preState: historyString(), moved: false };
  });
  return handle;
}

function labelWidth(label) {
  return Math.max(56, label.length * 7 + 22);
}

// Edge tags receive partial counter-scaling below 100%. A strict cap keeps the
// fitted overview proportional to its nodes instead of covering the topology.
function labelScale() {
  const s = app.viewport.scale;
  return s < 1 ? Math.min(1 / s, 1.75) : 1;
}

// Gentle vertical de-collision so adjacent edge tags don't stack on each other.
// The footprint is inflated by the current label scale so chips stay separated
// at the zoom they are actually drawn at, not just at 100%.
function separateLabels(specs, k = 1) {
  // A few relaxation passes so chains of stacked tags fully separate, not just pairs.
  for (let pass = 0; pass < 4; pass += 1) {
    let moved = false;
    for (let i = 0; i < specs.length; i += 1) {
      for (let j = 0; j < i; j += 1) {
        const a = specs[j];
        const b = specs[i];
        const overlapX = ((a.width + b.width) / 2) * k + 6 - Math.abs(a.mid.x - b.mid.x);
        const overlapY = ((a.height + b.height) / 2) * k + 4 - Math.abs(a.mid.y - b.mid.y);
        if (overlapX > 0 && overlapY > 0 && !b.pinned) {
          // Pinned (user-dragged) labels stay put; auto labels flow around them.
          b.mid.y += (b.mid.y >= a.mid.y ? 1 : -1) * overlapY;
          moved = true;
        }
      }
    }
    if (!moved) break;
  }
}

function edgeLabelElement(spec) {
  const { edge, mid, anchor, width, height, pinned } = spec;
  const k = labelScale();
  const group = svgEl("g", {
    class: `edge-label-chip ${isSelected("edge", edge.id) ? "selected" : ""}`,
    "data-id": edge.id,
    transform: `translate(${mid.x} ${mid.y}) scale(${k})`,
  });
  if (pinned && anchor) {
    // Leader back to the edge so a tag dragged away still reads as belonging to it.
    group.appendChild(svgEl("line", {
      class: "edge-label-leader",
      x1: (anchor.x - mid.x) / k,
      y1: (anchor.y - mid.y) / k,
      x2: 0,
      y2: 0,
    }));
  }
  group.appendChild(svgEl("rect", {
    class: "edge-label-bg",
    x: -width / 2,
    y: -height / 2,
    width,
    height,
    rx: 6,
  }));
  const text = svgEl("text", {
    class: "edge-label",
    x: 0,
    y: 0,
    "text-anchor": "middle",
    "dominant-baseline": "central",
  });
  text.textContent = edge.label;
  group.appendChild(text);
  group.addEventListener("mousedown", startLabelDrag(edge));
  group.addEventListener("dblclick", (event) => {
    event.stopPropagation();
    if (edge.labelOffset) {
      recordHistory();
      delete edge.labelOffset;
      renderAll();
      setStatus("Label position reset");
    }
  });
  return group;
}

function startLabelDrag(edge) {
  return (event) => {
    event.stopPropagation();
    select("edge", edge.id);
    const p = screenToWorld(event);
    const off = edge.labelOffset || { dx: 0, dy: 0 };
    app.drag = {
      kind: "label",
      edgeId: edge.id,
      startX: p.x,
      startY: p.y,
      baseDx: off.dx,
      baseDy: off.dy,
      preState: historyString(),
      moved: false,
    };
  };
}

function nodeElement(node) {
  const meta = TYPE_META[node.type] || TYPE_META.system;
  const level = nodeLevelMeta(node);
  const group = svgEl("g", {
    class: `node ${isSelected("node", node.id) ? "selected" : ""}`,
    transform: `translate(${node.x} ${node.y})`,
    "data-id": node.id,
    "data-level": node.level,
    role: "group",
    "aria-label": `${level.code} ${level.label}: ${node.label}`,
  });
  group.appendChild(svgEl("rect", {
    class: `node-rect ${meta.className}`,
    width: node.w,
    height: node.h,
  }));

  const levelText = svgEl("text", { class: "node-level", x: 14, y: 15 });
  levelText.textContent = `${level.code} · ${level.label}`;
  group.appendChild(levelText);

  const title = svgEl("text", { class: "node-title", x: 14, y: 34 });
  title.textContent = clipText(node.label, Math.max(10, Math.floor((node.w - 26) / 7.2)));
  group.appendChild(title);

  // Level of detail: below SUBTITLE_MIN_SCALE the subtitle is unreadable noise,
  // so drop it and let the hover tooltip carry the detail instead.
  if (app.viewport.scale >= SUBTITLE_MIN_SCALE) {
    const fullCodeRef = nodeCodeRef(node);
    const codeRef = compactCodeRef(fullCodeRef);
    const codeY = node.h - 9;
    const subtitleLines = Math.max(0, Math.min(3, Math.floor((codeY - 55) / 15)));
    wrapLines(node.subtitle || "", Math.max(12, Math.floor((node.w - 26) / 7)), subtitleLines).forEach((line, index) => {
      const text = svgEl("text", { class: "node-subtitle", x: 14, y: 54 + index * 15 });
      text.textContent = line;
      group.appendChild(text);
    });
    const openable = codeRefIsOpenable(fullCodeRef);
    const ref = svgEl("text", {
      class: `node-code-ref${openable ? " openable" : ""}`,
      x: 14,
      y: codeY,
      role: openable ? "link" : "text",
      tabindex: openable ? "0" : "-1",
      "aria-label": openable ? `Open ${fullCodeRef}` : fullCodeRef,
    });
    ref.textContent = `code: ${clipText(codeRef, Math.max(12, Math.floor((node.w - 26) / 6.2)))}`;
    if (openable) {
      ref.addEventListener("mousedown", (event) => event.stopPropagation());
      ref.addEventListener("click", (event) => {
        event.stopPropagation();
        openCodeRef(fullCodeRef);
      });
      ref.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        event.stopPropagation();
        openCodeRef(fullCodeRef);
      });
    }
    group.appendChild(ref);
  }

  group.addEventListener("mouseenter", (event) => showNodeTooltip(node, event));
  group.addEventListener("mousemove", positionNodeTooltip);
  group.addEventListener("mouseleave", hideNodeTooltip);

  group.addEventListener("mousedown", (event) => {
    event.stopPropagation();
    hideNodeTooltip();
    if (app.mode === "connect") {
      handleConnectClick(node.id);
      return;
    }
    select("node", node.id);
    const p = screenToWorld(event);
    app.drag = {
      kind: "node",
      id: node.id,
      offsetX: p.x - node.x,
      offsetY: p.y - node.y,
      preState: historyString(),
      moved: false,
    };
  });

  if (isSelected("node", node.id)) appendResizeHandles(group, node);
  return group;
}

// Eight resize handles on the selected node. Sized in inverse of the zoom so they
// stay a constant ~9px on screen and grabbable at any zoom.
function appendResizeHandles(group, node) {
  const s = 9 / app.viewport.scale;
  const handles = [
    { dir: "nw", x: 0, y: 0 },
    { dir: "n", x: node.w / 2, y: 0 },
    { dir: "ne", x: node.w, y: 0 },
    { dir: "e", x: node.w, y: node.h / 2 },
    { dir: "se", x: node.w, y: node.h },
    { dir: "s", x: node.w / 2, y: node.h },
    { dir: "sw", x: 0, y: node.h },
    { dir: "w", x: 0, y: node.h / 2 },
  ];
  handles.forEach(({ dir, x, y }) => {
    const handle = svgEl("rect", {
      class: `node-handle handle-${dir}`,
      x: x - s / 2,
      y: y - s / 2,
      width: s,
      height: s,
      rx: s * 0.25,
    });
    handle.addEventListener("mousedown", (event) => {
      event.stopPropagation();
      const p = screenToWorld(event);
      app.drag = {
        kind: "resize",
        id: node.id,
        dir,
        startX: p.x,
        startY: p.y,
        start: { x: node.x, y: node.y, w: node.w, h: node.h },
        preState: historyString(),
        moved: false,
      };
    });
    group.appendChild(handle);
  });
}

// Compute new box geometry for a resize drag. The edge(s) named by `dir` move;
// the opposite edge(s) stay fixed. Minimums are enforced against the fixed edge.
function resizeNode(dir, start, dx, dy, snap) {
  const snapTo = (v) => (snap ? Math.round(v / GRID) * GRID : v);
  let { x, y, w, h } = start;
  if (dir.includes("e")) w = snapTo(start.w + dx);
  if (dir.includes("s")) h = snapTo(start.h + dy);
  if (dir.includes("w")) {
    x = snapTo(start.x + dx);
    w = start.x + start.w - x;
  }
  if (dir.includes("n")) {
    y = snapTo(start.y + dy);
    h = start.y + start.h - y;
  }
  if (w < MIN_W) {
    if (dir.includes("w")) x = start.x + start.w - MIN_W;
    w = MIN_W;
  }
  if (h < MIN_H) {
    if (dir.includes("n")) y = start.y + start.h - MIN_H;
    h = MIN_H;
  }
  return { x, y, w, h };
}

// Hover readout keeps the complete label, abstraction meaning, and code
// definition accessible even when the fitted view clips in-node text.
function showNodeTooltip(node, event) {
  if (app.drag) return;
  const level = nodeLevelMeta(node);
  el.nodeTooltip.innerHTML = `<span class="tooltip-level">${escapeHtml(level.code)} · ${escapeHtml(level.label)}</span><strong>${escapeHtml(node.label)}</strong>${
    node.subtitle ? `<span>${escapeHtml(node.subtitle)}</span>` : ""
  }<span class="tooltip-code">${escapeHtml(nodeCodeRef(node))}</span>`;
  el.nodeTooltip.classList.remove("hidden");
  positionNodeTooltip(event);
}

function positionNodeTooltip(event) {
  if (el.nodeTooltip.classList.contains("hidden")) return;
  const wrap = el.svg.parentElement.getBoundingClientRect();
  const tw = el.nodeTooltip.offsetWidth;
  const th = el.nodeTooltip.offsetHeight;
  const x = Math.min(event.clientX - wrap.left + 14, wrap.width - tw - 8);
  const y = Math.min(event.clientY - wrap.top + 16, wrap.height - th - 8);
  el.nodeTooltip.style.left = `${Math.max(8, x)}px`;
  el.nodeTooltip.style.top = `${Math.max(8, y)}px`;
}

function hideNodeTooltip() {
  el.nodeTooltip.classList.add("hidden");
}

function select(type, id) {
  app.selected = { type, id };
  let expandedInspector = false;
  if (app.panels.right.collapsed) {
    app.panels.right.collapsed = false;
    applyPanels();
    saveSettings();
    expandedInspector = true;
  }
  if (expandedInspector) fitView();
  else renderCanvas();
  renderInspector();
  setStatus(`${type === "node" ? "Node" : "Edge"} selected`);
}

function isSelected(type, id) {
  return app.selected && app.selected.type === type && app.selected.id === id;
}

function handleConnectClick(nodeId) {
  if (!app.connectSource) {
    app.connectSource = nodeId;
    select("node", nodeId);
    setStatus(`Source selected: ${nodeById(nodeId).label}`);
    return;
  }
  if (app.connectSource === nodeId) {
    setStatus("Pick a different target node");
    return;
  }
  addEdge(app.connectSource, nodeId, "");
  const sourceLabel = nodeById(app.connectSource).label;
  const targetLabel = nodeById(nodeId).label;
  app.connectSource = null;
  setStatus(`Connected ${sourceLabel} -> ${targetLabel}`);
  renderAll();
}

function renderInspector() {
  populateNodeSelects();
  el.emptyInspector.classList.remove("hidden");
  el.nodeInspector.classList.add("hidden");
  el.edgeInspector.classList.add("hidden");

  if (!app.selected) return;

  if (app.selected.type === "node") {
    const node = nodeById(app.selected.id);
    if (!node) return;
    el.emptyInspector.classList.add("hidden");
    el.nodeInspector.classList.remove("hidden");
    el.nodeLabel.value = node.label;
    el.nodeSubtitle.value = node.subtitle || "";
    el.nodeCodeRef.value = nodeCodeRef(node);
    syncCodeOpenButton();
    el.nodeType.value = node.type;
    el.nodeLevel.value = node.level;
    el.nodeX.value = Math.round(node.x);
    el.nodeY.value = Math.round(node.y);
    el.nodeW.value = Math.round(node.w);
    el.nodeH.value = Math.round(node.h);
  }

  if (app.selected.type === "edge") {
    const edge = edgeById(app.selected.id);
    if (!edge) return;
    el.emptyInspector.classList.add("hidden");
    el.edgeInspector.classList.remove("hidden");
    el.edgeLabel.value = edge.label || "";
    el.edgeSource.value = edge.source;
    el.edgeTarget.value = edge.target;
  }
}

function populateNodeSelects() {
  const options = currentDiagram().nodes
    .map((node) => `<option value="${escapeAttr(node.id)}">${escapeHtml(node.label)}</option>`)
    .join("");
  [el.edgeSource, el.edgeTarget, el.newEdgeSource, el.newEdgeTarget].forEach((select) => {
    const previous = select.value;
    select.innerHTML = options;
    if ([...select.options].some((option) => option.value === previous)) {
      select.value = previous;
    }
  });

  const typeOptions = Object.entries(TYPE_META)
    .map(([type, meta]) => `<option value="${escapeAttr(type)}">${escapeHtml(meta.label)}</option>`)
    .join("");
  el.nodeType.innerHTML = typeOptions;
  const levelOptions = Object.entries(LEVEL_META)
    .map(([level, meta]) => `<option value="${escapeAttr(level)}">${escapeHtml(`${meta.code} · ${meta.label}`)}</option>`)
    .join("");
  el.nodeLevel.innerHTML = levelOptions;
}

function updateSelectedNode(patch) {
  if (!app.selected || app.selected.type !== "node") return;
  const node = nodeById(app.selected.id);
  if (!node) return;
  Object.assign(node, patch);
  renderCanvas();
  syncJsonBox();
}

function updateSelectedEdge(patch) {
  if (!app.selected || app.selected.type !== "edge") return;
  const edge = edgeById(app.selected.id);
  if (!edge) return;
  Object.assign(edge, patch);
  renderCanvas();
  syncJsonBox();
}

function addNode() {
  recordHistory();
  const point = screenToWorldCenter();
  const id = uniqueId("node");
  currentDiagram().nodes.push(n(id, "New node", "edit this concept", "logic", point.x - 100, point.y - 40, 210, 86, "resource"));
  select("node", id);
  renderAll();
}

function addEdge(source, target, label) {
  if (!source || !target || source === target) {
    setStatus("Choose two different nodes");
    return;
  }
  recordHistory();
  const edge = { id: uniqueId("edge"), source, target, label: label || "" };
  currentDiagram().edges.push(edge);
  app.selected = { type: "edge", id: edge.id };
}

function deleteSelected() {
  if (!app.selected) return;
  recordHistory();
  const diagram = currentDiagram();
  if (app.selected.type === "node") {
    const id = app.selected.id;
    diagram.nodes = diagram.nodes.filter((node) => node.id !== id);
    diagram.edges = diagram.edges.filter((edge) => edge.source !== id && edge.target !== id);
  } else {
    diagram.edges = diagram.edges.filter((edge) => edge.id !== app.selected.id);
  }
  app.selected = null;
  renderAll();
  setStatus("Deleted");
}

function fitView() {
  const diagram = currentDiagram();
  if (!diagram.nodes.length) return;
  const bounds = diagram.nodes.reduce(
    (acc, node) => ({
      minX: Math.min(acc.minX, node.x),
      minY: Math.min(acc.minY, node.y),
      maxX: Math.max(acc.maxX, node.x + node.w),
      maxY: Math.max(acc.maxY, node.y + node.h),
    }),
    { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity },
  );
  const rect = el.svg.getBoundingClientRect();
  const contentW = Math.max(1, bounds.maxX - bounds.minX);
  const contentH = Math.max(1, bounds.maxY - bounds.minY);
  const scale = clamp(Math.min((rect.width - 120) / contentW, (rect.height - 120) / contentH), 0.22, 1.25);
  app.viewport.scale = scale;
  app.viewport.x = (rect.width - contentW * scale) / 2 - bounds.minX * scale;
  app.viewport.y = (rect.height - contentH * scale) / 2 - bounds.minY * scale;
  renderCanvas();
  updateZoomLabel();
}

// Convert a wheel event into a multiplicative zoom factor. Using the event's
// pixel delta (normalized across deltaMode) makes a trackpad flick and a mouse
// notch behave consistently, and the per-event delta is clamped so one big
// flick can't jump. The sensitivity slider scales the whole thing.
function wheelZoomFactor(event) {
  let delta = event.deltaY;
  if (event.deltaMode === 1) delta *= 16; // lines -> ~px
  else if (event.deltaMode === 2) delta *= 400; // pages -> ~px
  delta = clamp(delta, -240, 240);
  return Math.exp(-delta * 0.0016 * app.wheelSensitivity);
}

function updateWheelSensLabel() {
  el.wheelSens.value = String(app.wheelSensitivity);
  el.wheelSensVal.value = `${app.wheelSensitivity.toFixed(2).replace(/0$/, "")}×`;
}

function saveSettings() {
  try {
    localStorage.setItem(
      SETTINGS_KEY,
      JSON.stringify({ wheelSensitivity: app.wheelSensitivity, snap: app.snap, panels: app.panels }),
    );
  } catch {
    /* ignore */
  }
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);
    if (typeof s.wheelSensitivity === "number") app.wheelSensitivity = clamp(s.wheelSensitivity, 0.1, 1.5);
    if (typeof s.snap === "boolean") app.snap = s.snap;
    ["left", "right"].forEach((side) => {
      const saved = s.panels && s.panels[side];
      if (!saved) return;
      if (typeof saved.w === "number") app.panels[side].w = clamp(saved.w, PANEL_MIN, PANEL_MAX);
      if (typeof saved.collapsed === "boolean") app.panels[side].collapsed = saved.collapsed;
    });
  } catch {
    /* ignore */
  }
}

function zoomAt(delta, screenPoint) {
  const oldScale = app.viewport.scale;
  const newScale = clamp(oldScale * delta, 0.16, 2.5);
  const rect = el.svg.getBoundingClientRect();
  const sx = screenPoint ? screenPoint.x - rect.left : rect.width / 2;
  const sy = screenPoint ? screenPoint.y - rect.top : rect.height / 2;
  const wx = (sx - app.viewport.x) / oldScale;
  const wy = (sy - app.viewport.y) / oldScale;
  app.viewport.scale = newScale;
  app.viewport.x = sx - wx * newScale;
  app.viewport.y = sy - wy * newScale;
  renderCanvas();
  updateZoomLabel();
}

function updateZoomLabel() {
  el.zoomLabel.value = `${Math.round(app.viewport.scale * 100)}%`;
}

function updateRouteButton() {
  el.routeBtn.textContent = app.routeMode === "orthogonal" ? "Orthogonal" : "Curved";
  el.routeBtn.classList.toggle("active", app.routeMode === "orthogonal");
}

function toggleRouteMode() {
  app.routeMode = app.routeMode === "orthogonal" ? "curved" : "orthogonal";
  updateRouteButton();
  renderCanvas();
  setStatus(`Routing: ${app.routeMode}`);
}

function syncJsonBox() {
  el.jsonBox.value = JSON.stringify(currentDiagram(), null, 2);
  persist();
}

function applyJson() {
  try {
    const parsed = JSON.parse(el.jsonBox.value);
    validateDiagram(parsed);
    enrichDiagram(parsed);
    recordHistory();
    app.diagrams[parsed.id || app.currentId] = parsed;
    app.currentId = parsed.id || app.currentId;
    app.selected = null;
    renderAll();
    setStatus("JSON applied");
  } catch (error) {
    setStatus(`JSON error: ${error.message}`);
  }
}

function exportJson() {
  const blob = new Blob([JSON.stringify(currentDiagram(), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${currentDiagram().id}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  setStatus("JSON exported");
}

function importJsonFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const parsed = JSON.parse(String(reader.result));
      validateDiagram(parsed);
      enrichDiagram(parsed);
      recordHistory();
      app.diagrams[parsed.id || uniqueId("diagram")] = parsed;
      app.currentId = parsed.id;
      app.selected = null;
      renderAll();
      setStatus("JSON imported");
    } catch (error) {
      setStatus(`Import error: ${error.message}`);
    }
  };
  reader.readAsText(file);
}

function validateDiagram(diagram) {
  if (!diagram || typeof diagram !== "object") throw new Error("Diagram must be an object");
  if (!Array.isArray(diagram.nodes)) throw new Error("Diagram.nodes must be an array");
  if (!Array.isArray(diagram.edges)) throw new Error("Diagram.edges must be an array");
  if (!diagram.id) diagram.id = uniqueId("diagram");
  if (!diagram.name) diagram.name = diagram.id;
  if (!diagram.description) diagram.description = "Imported diagram";
  diagram.nodes.forEach((node) => {
    if (typeof node.codeRef !== "string") node.codeRef = "";
  });
}

// ---- History (undo / redo) ----
// A snapshot is the full set of diagrams plus the active id. recordHistory() is
// called just before a mutation; node drags stash their pre-state and commit it
// on mouseup only if the node actually moved.
function historyString() {
  return JSON.stringify({ d: app.diagrams, c: app.currentId });
}

function pushUndo(stateStr) {
  if (app.undoStack.at(-1) === stateStr) return;
  app.undoStack.push(stateStr);
  if (app.undoStack.length > 80) app.undoStack.shift();
  app.redoStack.length = 0;
  updateHistoryButtons();
}

function recordHistory() {
  pushUndo(historyString());
}

function restoreState(stateStr) {
  const state = JSON.parse(stateStr);
  app.diagrams = enrichDiagrams(state.d);
  app.currentId = app.diagrams[state.c] ? state.c : Object.keys(app.diagrams)[0];
  app.selected = null;
  app.connectSource = null;
  renderAll();
}

function undo() {
  if (!app.undoStack.length) return;
  app.redoStack.push(historyString());
  restoreState(app.undoStack.pop());
  updateHistoryButtons();
  setStatus("Undo");
}

function redo() {
  if (!app.redoStack.length) return;
  app.undoStack.push(historyString());
  restoreState(app.redoStack.pop());
  updateHistoryButtons();
  setStatus("Redo");
}

function updateHistoryButtons() {
  el.undoBtn.disabled = app.undoStack.length === 0;
  el.redoBtn.disabled = app.redoStack.length === 0;
}

// ---- Persistence (autosave) ----
let saveTimer = null;
let lastSaved = "";

function persist(immediate = false) {
  const write = () => {
    const serialized = JSON.stringify(app.diagrams);
    if (serialized === lastSaved) return;
    localStorage.setItem(STORAGE_KEY, serialized);
    lastSaved = serialized;
    setStatus(`Auto-saved ${new Date().toLocaleTimeString()}`);
  };
  clearTimeout(saveTimer);
  if (immediate) write();
  else saveTimer = setTimeout(write, 600);
}

function saveLocal() {
  persist(true);
  setStatus("Saved to browser local storage");
}

function loadLocal() {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (!saved) return false;
  try {
    const parsed = JSON.parse(saved);
    if (parsed && typeof parsed === "object") {
      app.diagrams = parsed;
      enrichDiagrams(app.diagrams);
      lastSaved = saved;
      if (!app.diagrams[app.currentId]) app.currentId = Object.keys(app.diagrams)[0] || "full";
      return true;
    }
  } catch {
    return false;
  }
  return false;
}

function toggleSnap() {
  app.snap = !app.snap;
  el.snapBtn.classList.toggle("active", app.snap);
  saveSettings();
  setStatus(`Grid snap ${app.snap ? "on" : "off"}`);
}

// ---- Side panels (collapse + resize) ----
function applyPanels() {
  const { left, right } = app.panels;
  document.documentElement.style.setProperty("--left-w", `${left.collapsed ? 0 : left.w}px`);
  document.documentElement.style.setProperty("--right-w", `${right.collapsed ? 0 : right.w}px`);
  el.leftPanel.classList.toggle("collapsed", left.collapsed);
  el.rightPanel.classList.toggle("collapsed", right.collapsed);
  el.resizerLeft.classList.toggle("hidden", left.collapsed);
  el.resizerRight.classList.toggle("hidden", right.collapsed);
  // Chevron points the way the panel will move when clicked.
  el.toggleLeft.textContent = left.collapsed ? "⟩" : "⟨";
  el.toggleRight.textContent = right.collapsed ? "⟨" : "⟩";
}

function togglePanel(side) {
  const panel = app.panels[side];
  panel.collapsed = !panel.collapsed;
  applyPanels();
  fitView();
  saveSettings();
  setStatus(`${side === "left" ? "Left" : "Right"} panel ${panel.collapsed ? "hidden" : "shown"}`);
}

function startPanelResize(side, event) {
  event.preventDefault();
  const panel = app.panels[side];
  if (panel.collapsed) return;
  const startX = event.clientX;
  const startW = panel.w;
  const resizer = side === "left" ? el.resizerLeft : el.resizerRight;
  resizer.classList.add("active");
  document.body.classList.add("resizing");

  const onMove = (e) => {
    const delta = e.clientX - startX;
    panel.w = clamp(side === "left" ? startW + delta : startW - delta, PANEL_MIN, PANEL_MAX);
    applyPanels();
  };
  const onUp = () => {
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    resizer.classList.remove("active");
    document.body.classList.remove("resizing");
    saveSettings();
  };
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
}

function resetTemplate() {
  recordHistory();
  app.diagrams[app.currentId] = enrichDiagram(clone(templates[app.currentId] || templates.full));
  app.selected = null;
  fitView();
  renderAll();
  setStatus("Template restored");
}

function uniqueId(prefix) {
  const existing = new Set([
    ...currentDiagram().nodes.map((node) => node.id),
    ...currentDiagram().edges.map((edge) => edge.id),
  ]);
  let id;
  do {
    id = `${prefix}_${Math.random().toString(36).slice(2, 9)}`;
  } while (existing.has(id));
  return id;
}

function screenToWorld(event) {
  const rect = el.svg.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left - app.viewport.x) / app.viewport.scale,
    y: (event.clientY - rect.top - app.viewport.y) / app.viewport.scale,
  };
}

function screenToWorldCenter() {
  const rect = el.svg.getBoundingClientRect();
  return {
    x: (rect.width / 2 - app.viewport.x) / app.viewport.scale,
    y: (rect.height / 2 - app.viewport.y) / app.viewport.scale,
  };
}

function centerOf(node) {
  return { x: node.x + node.w / 2, y: node.y + node.h / 2 };
}

function buildRouteContext(diagram) {
  const nodeMap = new Map(diagram.nodes.map((node) => [node.id, node]));
  const edgeInfo = new Map();
  const portGroups = new Map();
  const laneGroups = new Map();

  diagram.edges.forEach((edge) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    if (!source || !target) return;
    const [autoSourceSide, autoTargetSide] = routeSides(source, target);
    const sourceSide = edge.sourceAnchor ? edge.sourceAnchor.side : autoSourceSide;
    const targetSide = edge.targetAnchor ? edge.targetAnchor.side : autoTargetSide;
    const info = { sourceSide, targetSide, sourcePort: 0, sourcePortCount: 1, targetPort: 0, targetPortCount: 1, lane: 0 };
    edgeInfo.set(edge.id, info);

    // Manually-anchored ends keep their fixed point, so they don't take part in
    // the automatic port spreading / lane offsetting of the other edges.
    if (!edge.sourceAnchor) {
      const sourceKey = `${edge.source}:${sourceSide}`;
      if (!portGroups.has(sourceKey)) portGroups.set(sourceKey, []);
      portGroups.get(sourceKey).push({ edge, role: "source", other: target });

      const laneKey = `${edge.source}:${sourceSide}`;
      if (!laneGroups.has(laneKey)) laneGroups.set(laneKey, []);
      laneGroups.get(laneKey).push({ edge, other: target, horizontal: sourceSide === "left" || sourceSide === "right" });
    }
    if (!edge.targetAnchor) {
      const targetKey = `${edge.target}:${targetSide}`;
      if (!portGroups.has(targetKey)) portGroups.set(targetKey, []);
      portGroups.get(targetKey).push({ edge, role: "target", other: source });
    }
  });

  portGroups.forEach((items, key) => {
    const side = key.split(":").at(-1);
    items.sort((a, b) => {
      const ac = centerOf(a.other);
      const bc = centerOf(b.other);
      return side === "left" || side === "right" ? ac.y - bc.y : ac.x - bc.x;
    });
    items.forEach((item, index) => {
      const info = edgeInfo.get(item.edge.id);
      if (!info) return;
      if (item.role === "source") {
        info.sourcePort = index;
        info.sourcePortCount = items.length;
      } else {
        info.targetPort = index;
        info.targetPortCount = items.length;
      }
    });
  });

  laneGroups.forEach((items) => {
    items.sort((a, b) => {
      const ac = centerOf(a.other);
      const bc = centerOf(b.other);
      return a.horizontal ? ac.y - bc.y : ac.x - bc.x;
    });
    const count = items.length;
    items.forEach((item, index) => {
      const info = edgeInfo.get(item.edge.id);
      if (!info) return;
      info.lane = index - (count - 1) / 2;
    });
  });

  return { edgeInfo };
}

function edgeAnchor(node, side, index = 0, count = 1) {
  const c = centerOf(node);
  if (side === "left" || side === "right") {
    const min = node.y + Math.min(22, node.h * 0.28);
    const max = node.y + node.h - Math.min(22, node.h * 0.28);
    const y = count <= 1 ? c.y : min + ((max - min) * index) / (count - 1);
    return { x: side === "left" ? node.x : node.x + node.w, y };
  }
  const min = node.x + Math.min(34, node.w * 0.28);
  const max = node.x + node.w - Math.min(34, node.w * 0.28);
  const x = count <= 1 ? c.x : min + ((max - min) * index) / (count - 1);
  return { x, y: side === "top" ? node.y : node.y + node.h };
}

function routeSides(source, target) {
  const s = centerOf(source);
  const t = centerOf(target);
  const dx = t.x - s.x;
  const dy = t.y - s.y;
  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0 ? ["right", "left"] : ["left", "right"];
  }
  return dy >= 0 ? ["bottom", "top"] : ["top", "bottom"];
}

// A manual attach point: `t` is the fraction along the chosen side (0..1).
function anchorPointOn(node, side, t) {
  const f = clamp(t, 0, 1);
  if (side === "left") return { x: node.x, y: node.y + node.h * f };
  if (side === "right") return { x: node.x + node.w, y: node.y + node.h * f };
  if (side === "top") return { x: node.x + node.w * f, y: node.y };
  return { x: node.x + node.w * f, y: node.y + node.h };
}

// Nearest border side + fraction on a node for a world point (used while dragging
// an endpoint). t is kept off the exact corners so the elbow routing stays sane.
function nearestAnchor(node, p) {
  const dLeft = Math.abs(p.x - node.x);
  const dRight = Math.abs(p.x - (node.x + node.w));
  const dTop = Math.abs(p.y - node.y);
  const dBottom = Math.abs(p.y - (node.y + node.h));
  const min = Math.min(dLeft, dRight, dTop, dBottom);
  const tx = clamp((p.x - node.x) / node.w, 0.08, 0.92);
  const ty = clamp((p.y - node.y) / node.h, 0.08, 0.92);
  if (min === dLeft) return { side: "left", t: ty };
  if (min === dRight) return { side: "right", t: ty };
  if (min === dTop) return { side: "top", t: tx };
  return { side: "bottom", t: tx };
}

// Topmost node containing a world point, or null.
function nodeAtPoint(p) {
  const nodes = currentDiagram().nodes;
  for (let i = nodes.length - 1; i >= 0; i -= 1) {
    const node = nodes[i];
    if (p.x >= node.x && p.x <= node.x + node.w && p.y >= node.y && p.y <= node.y + node.h) return node;
  }
  return null;
}

function orthogonalRoute(edge, source, target, routeContext) {
  const [sourceSide, targetSide] = routeSides(source, target);
  const info = routeContext.edgeInfo.get(edge.id) || {
    sourceSide,
    targetSide,
    sourcePort: 0,
    sourcePortCount: 1,
    targetPort: 0,
    targetPortCount: 1,
    lane: 0,
  };
  const s = edge.sourceAnchor
    ? anchorPointOn(source, edge.sourceAnchor.side, edge.sourceAnchor.t)
    : edgeAnchor(source, info.sourceSide, info.sourcePort, info.sourcePortCount);
  const t = edge.targetAnchor
    ? anchorPointOn(target, edge.targetAnchor.side, edge.targetAnchor.t)
    : edgeAnchor(target, info.targetSide, info.targetPort, info.targetPortCount);
  const points = [s];
  const laneOffset = info.lane * 22;

  if (info.sourceSide === "right" || info.sourceSide === "left") {
    if (Math.abs(s.y - t.y) < 14 && Math.abs(s.x - t.x) < 92) {
      return { kind: "orthogonal", points: simplifyPoints([s, { x: t.x, y: s.y }, t]) };
    }
    const gap = info.sourceSide === "right" ? 30 : -30;
    const start = { x: s.x + gap, y: s.y };
    const endGap = info.targetSide === "left" ? -30 : 30;
    const end = { x: t.x + endGap, y: t.y };
    let midX = (start.x + end.x) / 2 + laneOffset;
    const lo = Math.min(start.x, end.x) + 10;
    const hi = Math.max(start.x, end.x) - 10;
    if (hi > lo) midX = clamp(midX, lo, hi);
    points.push(start, { x: midX, y: start.y }, { x: midX, y: end.y }, end);
  } else {
    if (Math.abs(s.x - t.x) < 14 && Math.abs(s.y - t.y) < 92) {
      return { kind: "orthogonal", points: simplifyPoints([s, { x: s.x, y: t.y }, t]) };
    }
    const gap = info.sourceSide === "bottom" ? 30 : -30;
    const start = { x: s.x, y: s.y + gap };
    const endGap = info.targetSide === "top" ? -30 : 30;
    const end = { x: t.x, y: t.y + endGap };
    let midY = (start.y + end.y) / 2 + laneOffset;
    const lo = Math.min(start.y, end.y) + 10;
    const hi = Math.max(start.y, end.y) - 10;
    if (hi > lo) midY = clamp(midY, lo, hi);
    points.push(start, { x: start.x, y: midY }, { x: end.x, y: midY }, end);
  }

  points.push(t);
  return { kind: "orthogonal", points: simplifyPoints(points) };
}

function curvedRoute(edge, source, target) {
  const s = edge.sourceAnchor ? anchorPointOn(source, edge.sourceAnchor.side, edge.sourceAnchor.t) : centerOf(source);
  const t = edge.targetAnchor ? anchorPointOn(target, edge.targetAnchor.side, edge.targetAnchor.t) : centerOf(target);
  const dx = Math.max(90, Math.abs(t.x - s.x) * 0.42);
  return {
    kind: "curved",
    points: [s, { x: s.x + dx, y: s.y }, { x: t.x - dx, y: t.y }, t],
  };
}

function routeToPath(route) {
  if (route.kind === "curved") {
    const [s, c1, c2, t] = route.points;
    return `M ${s.x} ${s.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${t.x} ${t.y}`;
  }
  const [first, ...rest] = route.points;
  return `M ${first.x} ${first.y} ${rest.map((p) => `L ${p.x} ${p.y}`).join(" ")}`;
}

function routeLabelPoint(route, label = "", nodes = []) {
  const w = labelWidth(label);
  const clearOfNodes = (x, y) =>
    !nodes.some((node) => x + w / 2 > node.x && x - w / 2 < node.x + node.w && y + 11 > node.y && y - 11 < node.y + node.h);

  if (route.kind === "curved") {
    const [p0, p1, p2, p3] = route.points;
    return cubicPoint(p0, p1, p2, p3, 0.5);
  }
  const segments = [];
  for (let i = 0; i < route.points.length - 1; i += 1) {
    const a = route.points[i];
    const b = route.points[i + 1];
    const length = Math.hypot(b.x - a.x, b.y - a.y);
    const horizontal = Math.abs(a.y - b.y) < 0.1;
    const vertical = Math.abs(a.x - b.x) < 0.1;
    segments.push({ a, b, length, horizontal, vertical });
  }

  // Build candidate label points (longest horizontal first, then vertical),
  // then pick the first one that does not sit on top of a node box.
  const candidates = [];
  segments
    .filter((segment) => segment.horizontal && segment.length >= w)
    .sort((a, b) => b.length - a.length)
    .forEach((segment) => candidates.push({ x: (segment.a.x + segment.b.x) / 2, y: segment.a.y - 15 }));
  segments
    .filter((segment) => segment.vertical && segment.length >= 28)
    .sort((a, b) => b.length - a.length)
    .forEach((segment) => candidates.push({ x: segment.a.x + 18, y: (segment.a.y + segment.b.y) / 2 }));

  const clear = candidates.find((candidate) => clearOfNodes(candidate.x, candidate.y));
  if (clear) return clear;
  if (candidates.length) return candidates[0];

  const longest = segments.slice().sort((a, b) => b.length - a.length)[0];
  if (longest) {
    return { x: (longest.a.x + longest.b.x) / 2, y: (longest.a.y + longest.b.y) / 2 - 15 };
  }
  return route.points[Math.floor(route.points.length / 2)];
}

function simplifyPoints(points) {
  const deduped = points.filter((point, index) => {
    if (index === 0) return true;
    const prev = points[index - 1];
    return Math.abs(prev.x - point.x) > 0.1 || Math.abs(prev.y - point.y) > 0.1;
  });
  const simplified = [];
  deduped.forEach((point) => {
    simplified.push(point);
    while (simplified.length >= 3) {
      const a = simplified[simplified.length - 3];
      const b = simplified[simplified.length - 2];
      const c = simplified[simplified.length - 1];
      const sameX = Math.abs(a.x - b.x) < 0.1 && Math.abs(b.x - c.x) < 0.1;
      const sameY = Math.abs(a.y - b.y) < 0.1 && Math.abs(b.y - c.y) < 0.1;
      if (!sameX && !sameY) break;
      simplified.splice(simplified.length - 2, 1);
    }
  });
  return simplified;
}

function cubicPoint(p0, p1, p2, p3, t) {
  const u = 1 - t;
  return {
    x: u ** 3 * p0.x + 3 * u ** 2 * t * p1.x + 3 * u * t ** 2 * p2.x + t ** 3 * p3.x,
    y: u ** 3 * p0.y + 3 * u ** 2 * t * p1.y + 3 * u * t ** 2 * p2.y + t ** 3 * p3.y,
  };
}

function wrapLines(text, maxChars, maxLines) {
  if (maxLines <= 0) return [];
  const words = text.split(/\s+/).filter(Boolean);
  const lines = [];
  let line = "";
  words.forEach((word) => {
    const next = line ? `${line} ${word}` : word;
    if (next.length > maxChars && line) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  });
  if (line) lines.push(line);
  if (lines.length > maxLines) {
    const clipped = lines.slice(0, maxLines);
    clipped[maxLines - 1] = `${clipped[maxLines - 1].replace(/\.+$/, "")}...`;
    return clipped;
  }
  return lines;
}

function clipText(text, maxChars) {
  if (text.length <= maxChars) return text;
  return `${text.slice(0, Math.max(1, maxChars - 3))}...`;
}

function svgEl(tag, attrs) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs || {}).forEach(([key, value]) => {
    element.setAttribute(key, String(value));
  });
  return element;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function bindEvents() {
  el.selectModeBtn.addEventListener("click", () => setMode("select"));
  el.connectModeBtn.addEventListener("click", () => setMode("connect"));
  el.addNodeBtn.addEventListener("click", addNode);
  el.deleteBtn.addEventListener("click", deleteSelected);
  el.zoomInBtn.addEventListener("click", () => zoomAt(1.16));
  el.zoomOutBtn.addEventListener("click", () => zoomAt(1 / 1.16));
  el.fitBtn.addEventListener("click", fitView);
  el.routeBtn.addEventListener("click", toggleRouteMode);
  el.snapBtn.addEventListener("click", toggleSnap);
  el.undoBtn.addEventListener("click", undo);
  el.redoBtn.addEventListener("click", redo);
  el.saveBtn.addEventListener("click", saveLocal);
  el.exportBtn.addEventListener("click", exportJson);
  el.importBtn.addEventListener("click", () => el.importFile.click());
  el.importFile.addEventListener("change", () => {
    const file = el.importFile.files && el.importFile.files[0];
    if (file) importJsonFile(file);
    el.importFile.value = "";
  });
  el.applyJsonBtn.addEventListener("click", applyJson);
  el.resetBtn.addEventListener("click", resetTemplate);
  el.createEdgeBtn.addEventListener("click", () => {
    addEdge(el.newEdgeSource.value, el.newEdgeTarget.value, el.newEdgeLabel.value.trim());
    el.newEdgeLabel.value = "";
    renderAll();
    setStatus("Edge created");
  });

  // Snapshot once when the user starts editing a field so a whole edit undoes together.
  el.nodeInspector.addEventListener("focusin", recordHistory);
  el.edgeInspector.addEventListener("focusin", recordHistory);

  el.nodeLabel.addEventListener("input", () => updateSelectedNode({ label: el.nodeLabel.value }));
  el.nodeSubtitle.addEventListener("input", () => updateSelectedNode({ subtitle: el.nodeSubtitle.value }));
  el.nodeCodeRef.addEventListener("input", () => {
    updateSelectedNode({ codeRef: el.nodeCodeRef.value });
    syncCodeOpenButton();
  });
  el.nodeOpenCodeBtn.addEventListener("click", () => openCodeRef(el.nodeCodeRef.value));
  el.nodeType.addEventListener("change", () => updateSelectedNode({ type: el.nodeType.value }));
  el.nodeLevel.addEventListener("change", () => updateSelectedNode({ level: el.nodeLevel.value }));
  el.nodeX.addEventListener("input", () => updateSelectedNode({ x: Number(el.nodeX.value) || 0 }));
  el.nodeY.addEventListener("input", () => updateSelectedNode({ y: Number(el.nodeY.value) || 0 }));
  el.nodeW.addEventListener("input", () => updateSelectedNode({ w: Math.max(90, Number(el.nodeW.value) || 90) }));
  el.nodeH.addEventListener("input", () => updateSelectedNode({ h: Math.max(MIN_H, Number(el.nodeH.value) || MIN_H) }));
  el.edgeLabel.addEventListener("input", () => updateSelectedEdge({ label: el.edgeLabel.value }));
  el.edgeSource.addEventListener("change", () => updateSelectedEdge({ source: el.edgeSource.value }));
  el.edgeTarget.addEventListener("change", () => updateSelectedEdge({ target: el.edgeTarget.value }));

  el.svg.addEventListener("mousedown", (event) => {
    if (event.target !== el.bg && event.target !== el.svg) return;
    hideNodeTooltip();
    app.selected = null;
    renderInspector();
    app.drag = {
      kind: "pan",
      startX: event.clientX,
      startY: event.clientY,
      viewportX: app.viewport.x,
      viewportY: app.viewport.y,
    };
  });

  window.addEventListener("mousemove", (event) => {
    const p = screenToWorld(event);
    el.positionText.textContent = `x ${Math.round(p.x)}, y ${Math.round(p.y)}`;
    if (!app.drag) return;

    if (app.drag.kind === "node") {
      const node = nodeById(app.drag.id);
      if (!node) return;
      let nx = p.x - app.drag.offsetX;
      let ny = p.y - app.drag.offsetY;
      if (app.snap && !event.altKey) {
        nx = Math.round(nx / GRID) * GRID;
        ny = Math.round(ny / GRID) * GRID;
      }
      if (nx !== node.x || ny !== node.y) app.drag.moved = true;
      node.x = nx;
      node.y = ny;
      renderCanvas();
      renderInspector();
      syncJsonBox();
    }

    if (app.drag.kind === "resize") {
      const node = nodeById(app.drag.id);
      if (!node) return;
      const next = resizeNode(
        app.drag.dir,
        app.drag.start,
        p.x - app.drag.startX,
        p.y - app.drag.startY,
        app.snap && !event.altKey,
      );
      if (next.w !== node.w || next.h !== node.h || next.x !== node.x || next.y !== node.y) app.drag.moved = true;
      Object.assign(node, next);
      renderCanvas();
      renderInspector();
      syncJsonBox();
    }

    if (app.drag.kind === "endpoint") {
      const edge = edgeById(app.drag.edgeId);
      if (!edge) return;
      const otherId = app.drag.end === "source" ? edge.target : edge.source;
      // Attach to the node under the cursor; reconnecting to the other endpoint's
      // own node would make a self-loop, so fall back to the current node there.
      let node = nodeAtPoint(p);
      if (!node || node.id === otherId) node = nodeById(app.drag.end === "source" ? edge.source : edge.target);
      if (!node) return;
      const anchor = nearestAnchor(node, p);
      if (app.drag.end === "source") {
        edge.source = node.id;
        edge.sourceAnchor = anchor;
      } else {
        edge.target = node.id;
        edge.targetAnchor = anchor;
      }
      app.drag.moved = true;
      renderCanvas();
      renderInspector();
      syncJsonBox();
    }

    if (app.drag.kind === "label") {
      const edge = edgeById(app.drag.edgeId);
      if (!edge) return;
      const dx = p.x - app.drag.startX;
      const dy = p.y - app.drag.startY;
      if (dx || dy) app.drag.moved = true;
      edge.labelOffset = { dx: app.drag.baseDx + dx, dy: app.drag.baseDy + dy };
      renderCanvas();
      syncJsonBox();
    }

    if (app.drag.kind === "pan") {
      app.viewport.x = app.drag.viewportX + (event.clientX - app.drag.startX);
      app.viewport.y = app.drag.viewportY + (event.clientY - app.drag.startY);
      renderCanvas();
    }
  });

  window.addEventListener("mouseup", () => {
    if (
      app.drag &&
      (app.drag.kind === "node" ||
        app.drag.kind === "label" ||
        app.drag.kind === "resize" ||
        app.drag.kind === "endpoint") &&
      app.drag.moved &&
      app.drag.preState
    ) {
      pushUndo(app.drag.preState);
    }
    app.drag = null;
  });

  el.svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomAt(wheelZoomFactor(event), { x: event.clientX, y: event.clientY });
  }, { passive: false });

  el.wheelSens.addEventListener("input", () => {
    app.wheelSensitivity = Number(el.wheelSens.value) || 0.5;
    updateWheelSensLabel();
    saveSettings();
  });

  el.resizerLeft.addEventListener("mousedown", (event) => startPanelResize("left", event));
  el.resizerRight.addEventListener("mousedown", (event) => startPanelResize("right", event));
  el.toggleLeft.addEventListener("click", () => togglePanel("left"));
  el.toggleRight.addEventListener("click", () => togglePanel("right"));

  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
    const mod = event.metaKey || event.ctrlKey;
    if (mod && event.key.toLowerCase() === "z") {
      event.preventDefault();
      if (event.shiftKey) redo();
      else undo();
      return;
    }
    if (mod && event.key.toLowerCase() === "y") {
      event.preventDefault();
      redo();
      return;
    }
    if (mod) return;
    if (event.key === "Delete" || event.key === "Backspace") deleteSelected();
    if (event.key === "c") setMode("connect");
    if (event.key === "v" || event.key === "Escape") setMode("select");
    if (event.key === "f") fitView();
    if (event.key === "r") toggleRouteMode();
  });

  window.addEventListener("resize", () => {
    renderCanvas();
  });
}

loadSettings();
loadLocal();
bindEvents();
updateWheelSensLabel();
el.snapBtn.classList.toggle("active", app.snap);
applyPanels();
renderAll();
requestAnimationFrame(() => {
  fitView();
  renderAll();
});
