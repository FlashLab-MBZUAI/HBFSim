# Obelisk: HBF 模拟器项目完整技术方案

> 本文档是 Obelisk(High-Bandwidth Flash 模拟器)的完整设计规范。目标读者是 C++ 开发者或 AI 编码助手。所有架构决策、模块接口、数据 schema、实验计划已在此文档中固化,实现阶段应严格遵循本文档,如需偏离需明确标注并说明原因。

---

## 0. 速览

| 项目 | 内容 |
|---|---|
| **项目名** | Obelisk |
| **目标** | 产出 1-2 篇架构顶会论文(ISCA/MICRO/ASPLOS/HPCA),次选 FAST/ATC |
| **Research Question** | 在 LLM inference 场景下,HBM+HBF 混合 3D 堆叠内存的最优数据放置与调度策略是什么 |
| **语言/工具** | C++20, CMake ≥ 3.20, Python 3.10+, TOML 配置 |
| **License** | MIT |
| **参考骨架** | PKU ChaseLab 的 Xerxes(FAST'26) |
| **M1 工期** | 4 周,2 人并行 |
| **完整项目工期** | 12-15 个月,至论文中稿 |
| **代码规模预估** | ~ 4000-6000 行 C++ 自写 + DRAMsim3 submodule |

### 名字由来

**Obelisk**(方尖碑)—— 古埃及建筑,由单一石材向上垂直堆叠而成,自下而上象征从物理基座到顶端的贯穿式结构。这一意象恰好对应 3D 堆叠内存:logic die 作为基座,多层 memory die 垂直堆叠,TSV 如贯穿石柱的中轴。项目名同时寓意一个"独立、垂直、纪念碑式"的开源模拟器,为 3D 异构内存架构的研究留下一座学术地标。

---

## 1. 项目定位与 Research Question

### 1.1 一句话定位

> **Obelisk 是一个面向 3D 堆叠异构内存(HBM + HBF)的周期级模拟器,用于评估 LLM inference 场景下数据放置和 sub-array 调度策略对性能、带宽、能耗的影响。**

### 1.2 主 Research Question(RQ)

**RQ-Main**: 在 LLM inference workload 下,HBM 和 HBF 异构堆叠内存系统中,什么样的数据放置与 sub-array 调度策略能最大化有效带宽利用率,同时满足 inference 的延迟约束?

### 1.3 子问题(Sub-RQs)

- **RQ-1**: HBF 的 sub-array 并行度从 N=32 到 N=128 变化时,logic die 调度器的瓶颈在哪里?
- **RQ-2**: KV cache 与 weight 在 HBM/HBF 之间的放置策略对 TTFT(Time-To-First-Token)和 TPOT(Time-Per-Output-Token)的影响如何?
- **RQ-3**: 当 batch size、context length、模型尺寸变化时,HBM:HBF 的最优容量比例是多少?
- **RQ-4**: 对比"all-HBM"、"all-HBF"、"HBM+HBF hybrid"、"HBM+SSD tiered",各自的 perf/TCO 对比?

### 1.4 成功标准

- **M1(4 周)**: 模拟器能跑通 LLM trace,产出合理的 bandwidth/latency 数据,复现 SanDisk fact sheet 的 1.6 TB/s baseline ±20% 内
- **M2(3 个月)**: 完成 RQ-1 和 RQ-2 的全部实验,产出 arxiv vision paper
- **M3(6 个月)**: 完成 RQ-3 和 RQ-4,投稿 ISCA 或 MICRO
- **M4(12 个月)**: 论文中稿或转投 HPCA/ASPLOS/FAST

---

## 2. 架构原则

### 2.1 精度分界线原则

整个系统分为五层,在 **HBF Memory Controller 入口**设置一条"精度分界线":

```
┌─────────────────────────────────────────────────────────┐
│ Layer 5: Workload Generator (LLM trace / synthetic)     │  粗粒度(参数)
├─────────────────────────────────────────────────────────┤
│ Layer 4: Host / CPU Model  [freq, compute_time_fn]      │  粗粒度(参数)
├─────────────────────────────────────────────────────────┤
│ Layer 3: Host Bus (PCIe/CXL) [bw, latency]              │  粗粒度(参数)
├═════════════════════════════════════════════════════════┤ ← 精度分界线
│ Layer 2: HBF Memory Controller (host-side)              │  精细建模
│   - request queue, scheduler, command generator         │
├─────────────────────────────────────────────────────────┤
│ Layer 1: HBF Logic Die / Base Die                       │  精细建模
│   - channel mux, sub-array dispatcher, TSV model        │
├─────────────────────────────────────────────────────────┤
│ Layer 0: Media (NAND core die stack / HBM via DRAMsim3) │  精细建模
└─────────────────────────────────────────────────────────┘
```

**原则**: 分界线以上全是参数化、无内部状态;分界线以下是完整事件驱动建模。

### 2.2 模块化原则

- **单一职责**: 每个 .hh/.cc 对只负责一个模块
- **接口隔离**: 模块间通过 `Packet` 对象通信,不共享内部状态
- **可替换性**: 每个核心模块必须有 Interface 抽象,至少支持 2 种实现(真实 + mock)
- **参数化一切**: 任何硬编码的数字都必须可以从 TOML 配置覆盖

### 2.3 事件驱动原则

- 中央 `Simulation` 类持有全局 clock(`uint64_t` 周期计数)
- 所有模块派生自 `IModule`,在每个 tick 被调用
- 请求通过 `Packet` 在模块间传递,携带所有 timing 统计字段
- **不使用线程**,单线程事件驱动

---

## 3. 完整目录结构

```
obelisk/
├── CMakeLists.txt
├── LICENSE                      # MIT
├── README.md
├── .clang-format
├── .gitignore
├── .gitmodules                  # DRAMsim3 as submodule
│
├── third_party/
│   └── DRAMsim3/                # submodule: https://github.com/umd-memsys/DRAMsim3
│
├── include/                     # 公共头文件
│   └── obelisk/
│       ├── types.hh             # 全局类型定义
│       ├── packet.hh            # Packet 数据结构
│       ├── imodule.hh           # 模块接口基类
│       └── imedia.hh            # 介质接口基类
│
├── src/
│   ├── base/
│   │   ├── simulation.hh        # 顶层事件驱动引擎
│   │   ├── simulation.cc
│   │   ├── clock.hh             # Clock 类
│   │   ├── stats.hh             # 统计收集器
│   │   └── stats.cc
│   │
│   ├── frontend/
│   │   ├── requester.hh         # 请求注入器
│   │   ├── requester.cc
│   │   ├── trace_reader.hh      # Trace 文件读取
│   │   └── trace_reader.cc
│   │
│   ├── host/
│   │   ├── host_cpu.hh          # 粗粒度 CPU 模型
│   │   ├── host_cpu.cc
│   │   ├── host_bus.hh          # 粗粒度 host bus (PCIe/CXL)
│   │   └── host_bus.cc
│   │
│   ├── controller/
│   │   ├── hbf_controller.hh    # HBF 内存控制器
│   │   ├── hbf_controller.cc
│   │   ├── scheduler.hh         # 请求调度器(可插拔)
│   │   ├── scheduler_fcfs.cc    # FCFS 实现
│   │   ├── scheduler_frfcfs.cc  # FR-FCFS 实现
│   │   └── scheduler_hbf_aware.cc  # HBF-aware 实现(论文贡献)
│   │
│   ├── logic_die/
│   │   ├── logic_die.hh         # HBF base die
│   │   ├── logic_die.cc
│   │   ├── channel_mux.hh       # Channel 多路复用
│   │   ├── subarray_dispatcher.hh  # Sub-array 分发器
│   │   └── tsv_model.hh         # TSV 带宽模型
│   │
│   ├── media/
│   │   ├── nand/
│   │   │   ├── nand_stack.hh    # NAND die stack
│   │   │   ├── nand_stack.cc
│   │   │   ├── nand_die.hh      # 单个 NAND die
│   │   │   ├── subarray.hh      # Sub-array 模型
│   │   │   └── subarray.cc
│   │   └── hbm/
│   │       ├── hbm_interface.hh # DRAMsim3 封装
│   │       └── hbm_interface.cc
│   │
│   ├── memory_system/
│   │   ├── memory_system.hh     # 顶层 orchestrator
│   │   ├── memory_system.cc
│   │   └── address_mapper.hh    # 物理地址 → (channel, die, subarray, page)
│   │
│   └── main.cc                  # 入口
│
├── configs/
│   ├── hbf_gen1_sandisk.toml    # SanDisk Gen1 HBF
│   ├── hbf_gen2.toml            # 推测的 Gen2 参数
│   ├── hbm3e_baseline.toml      # HBM3E baseline
│   ├── hybrid_hbm_hbf.toml      # HBM+HBF 混合
│   ├── gen_config.py            # Python 配置生成器
│   └── sweep_configs.py         # 参数扫描生成器
│
├── traces/
│   ├── samples/                 # 小 trace 用于快速测试
│   │   ├── synthetic_seq.trace
│   │   ├── synthetic_rand.trace
│   │   └── llama7b_decode_small.trace
│   └── gen_llm_trace.py         # LLM trace 生成器
│
├── tests/
│   ├── unit/                    # 每模块的 unit test
│   │   ├── test_subarray.cc
│   │   ├── test_logic_die.cc
│   │   ├── test_scheduler.cc
│   │   └── ...
│   ├── integration/             # 端到端集成测试
│   │   └── test_e2e.cc
│   └── regression/              # 黄金输出回归
│       ├── golden/              # 存储 golden CSV
│       └── run_regression.sh
│
├── output/                      # 模拟输出目录
│   ├── .gitkeep
│   └── plot_results.py          # 结果可视化脚本
│
├── AE-scripts/                  # Artifact Evaluation 脚本
│   ├── run_all.sh
│   ├── fig_main.sh              # Figure 1: 主结果图
│   ├── fig_sensitivity.sh       # Figure 2: 敏感性分析
│   └── ...
│
├── docs/
│   ├── architecture.md          # 架构详细说明
│   ├── hbf_primer.md            # HBF 技术背景
│   ├── config_reference.md      # 配置参数完整参考
│   ├── output_format.md         # 输出格式说明
│   └── development.md           # 开发者指南
│
└── .github/
    └── workflows/
        └── ci.yml               # CMake build + unit tests
```

---

## 4. 核心数据结构

### 4.1 `Packet`(请求包)

**文件**: `include/obelisk/packet.hh`

这是整个系统最核心的数据结构,在模块间传递。每个字段都对应最终 CSV 输出的一列。

```cpp
#pragma once
#include <cstdint>
#include <string>
#include "obelisk/types.hh"

namespace obelisk {

enum class PacketType : uint8_t {
    READ,               // 普通读(weight fetch)
    WRITE,              // 普通写(KV cache store)
    READ_KV,            // KV cache 读(带 locality hint)
    READ_NON_TEMPORAL,  // 非时序读(无 cache)
    PREFETCH,           // 预取
    GC_READ,            // GC 读(M2+)
    GC_WRITE,           // GC 写(M2+)
};

enum class MediaType : uint8_t {
    HBM,
    HBF,
};

// 物理位置(由 AddressMapper 填入)
struct PhysicalLocation {
    uint32_t stack_id       = 0;
    uint32_t channel_id     = 0;
    uint32_t die_id         = 0;
    uint32_t subarray_id    = 0;  // HBF 特有
    uint32_t bank_id        = 0;  // HBM 特有
    uint32_t row            = 0;
    uint32_t column         = 0;
    uint64_t page_id        = 0;
};

// Timing 统计(对应 CSV 每列)
struct TimingStats {
    uint64_t t_send              = 0;  // 请求由 requester 发出
    uint64_t t_arrive            = 0;  // 请求最终完成
    uint64_t host_bus_queuing    = 0;
    uint64_t host_bus_time       = 0;
    uint64_t controller_queuing  = 0;
    uint64_t controller_time     = 0;
    uint64_t logic_die_queuing   = 0;
    uint64_t logic_die_time      = 0;
    uint64_t subarray_queuing    = 0;  // HBF
    uint64_t subarray_time       = 0;  // HBF
    uint64_t bank_queuing        = 0;  // HBM
    uint64_t bank_time           = 0;  // HBM
    uint64_t tsv_time            = 0;

    uint64_t total_time() const { return t_arrive - t_send; }
};

struct Packet {
    // 静态属性(创建时确定)
    uint64_t     id          = 0;           // 全局唯一 ID
    PacketType   type        = PacketType::READ;
    uint64_t     addr        = 0;           // 虚拟/逻辑地址
    uint32_t     size        = 64;          // 字节数
    MediaType    target_media = MediaType::HBF;

    // 动态属性(运行时填入)
    PhysicalLocation phys;
    TimingStats      timing;

    // 元数据(可选,用于调试和分析)
    uint32_t     workload_tag = 0;   // 来自哪个 workload stream
    uint32_t     layer_id     = 0;   // LLM layer(如果适用)
    std::string  op_name;            // "attention_q_weight_fetch" 等

    // 序列化到 CSV 行
    std::string to_csv_row() const;
};

} // namespace obelisk
```

**关键设计**:
- `Packet` 是值类型,通过 `std::unique_ptr<Packet>` 在模块间 move
- `TimingStats` 里每个字段对应 CSV 一列,**这是输出契约的核心**
- `PhysicalLocation` 是由 `AddressMapper` 在进入 controller 前填好的

### 4.2 `IModule`(模块接口)

**文件**: `include/obelisk/imodule.hh`

```cpp
#pragma once
#include <memory>
#include <string>
#include <optional>
#include "obelisk/packet.hh"

namespace obelisk {

class Simulation;  // forward decl

class IModule {
public:
    virtual ~IModule() = default;

    // 模块名(用于日志/stats)
    virtual const std::string& name() const = 0;

    // 每个 tick 被 Simulation 调用一次
    virtual void tick(uint64_t current_cycle) = 0;

    // 接收上游模块的 packet(非阻塞)
    // 返回 true 表示接收成功,false 表示本模块 busy,上游需要重试
    virtual bool accept(std::unique_ptr<Packet> pkt) = 0;

    // 从本模块取出已完成的 packet(如果有)
    virtual std::optional<std::unique_ptr<Packet>> dequeue() { return std::nullopt; }

    // 是否还有 pending work(用于判断仿真结束)
    virtual bool has_pending() const = 0;

    // 注册 Simulation(用于回调)
    virtual void set_simulation(Simulation* sim) { sim_ = sim; }

protected:
    Simulation* sim_ = nullptr;
};

} // namespace obelisk
```

### 4.3 `IMedia`(介质接口)

**文件**: `include/obelisk/imedia.hh`

```cpp
#pragma once
#include "obelisk/imodule.hh"

namespace obelisk {

class IMedia : public IModule {
public:
    // 介质类型
    virtual MediaType media_type() const = 0;

    // 总容量(字节)
    virtual uint64_t capacity_bytes() const = 0;

    // 最小访问粒度(HBM = 32B/64B cache line;HBF = page,典型 4KB/16KB)
    virtual uint32_t min_access_granularity() const = 0;

    // peak bandwidth (GB/s),用于 sanity check
    virtual double peak_bandwidth_gbps() const = 0;
};

} // namespace obelisk
```

---

## 5. 模块详细接口

以下每个模块给出完整的类接口。实现者只需要填 `.cc` 即可。

### 5.1 Requester(请求注入器)

**文件**: `src/frontend/requester.hh`

```cpp
#pragma once
#include "obelisk/imodule.hh"
#include "src/frontend/trace_reader.hh"

namespace obelisk {

struct RequesterConfig {
    std::string trace_file;
    uint32_t    max_in_flight      = 64;   // 最大在途请求数
    uint32_t    issue_rate_per_cyc = 1;    // 每周期最多发出几个请求
    bool        warm_up            = true; // 是否做 warm-up
    uint64_t    warm_up_cycles     = 1000;
};

class Requester : public IModule {
public:
    explicit Requester(const RequesterConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(uint64_t current_cycle) override;
    bool accept(std::unique_ptr<Packet> pkt) override;  // 接收返回的完成包
    bool has_pending() const override;

    // 连接下游
    void connect_downstream(IModule* m) { downstream_ = m; }

    // 统计
    uint64_t total_issued() const { return total_issued_; }
    uint64_t total_completed() const { return total_completed_; }

private:
    std::string name_ = "Requester";
    RequesterConfig cfg_;
    std::unique_ptr<TraceReader> trace_;
    IModule* downstream_ = nullptr;
    uint32_t in_flight_ = 0;
    uint64_t total_issued_ = 0;
    uint64_t total_completed_ = 0;
};

} // namespace obelisk
```

**行为**:
- 每个 tick 从 trace 读下一条请求,生成 `Packet`,push 给 downstream
- 如果 `in_flight_ >= max_in_flight`,stall
- 接收 downstream 回传的完成 packet,记录 timing,写入 CSV

### 5.2 Host Bus

**文件**: `src/host/host_bus.hh`

```cpp
#pragma once
#include "obelisk/imodule.hh"
#include <queue>

namespace obelisk {

struct HostBusConfig {
    double   bandwidth_gbps     = 128.0;  // PCIe 6.0 x16
    uint32_t base_latency_cycles = 50;    // bus 基础延迟
    uint32_t queue_depth        = 32;
    uint32_t cpu_compute_cycles = 100;    // 粗粒度 CPU 处理时间
};

class HostBus : public IModule {
public:
    explicit HostBus(const HostBusConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(uint64_t current_cycle) override;
    bool accept(std::unique_ptr<Packet> pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    void connect_downstream(IModule* m) { downstream_ = m; }
    void connect_upstream_return(IModule* m) { upstream_return_ = m; }

private:
    struct InFlightPacket {
        std::unique_ptr<Packet> pkt;
        uint64_t ready_cycle;
    };

    std::string name_ = "HostBus";
    HostBusConfig cfg_;
    std::queue<InFlightPacket> forward_queue_;   // host → HBF
    std::queue<InFlightPacket> return_queue_;    // HBF → host
    IModule* downstream_ = nullptr;
    IModule* upstream_return_ = nullptr;

    uint64_t compute_latency_cycles(const Packet& pkt) const;
};

} // namespace obelisk
```

**行为**:
- 正向(读/写请求):模拟 CPU 发出请求 → 经过 bus 到达 HBF controller,加 `cpu_compute_cycles + bus_latency + size/bandwidth`
- 反向(响应):模拟 HBF → CPU 的返回,同样加延迟
- 记录 `host_bus_queuing` 和 `host_bus_time` 到 packet timing

### 5.3 HBF Controller

**文件**: `src/controller/hbf_controller.hh`

```cpp
#pragma once
#include "obelisk/imodule.hh"
#include "src/controller/scheduler.hh"

namespace obelisk {

struct HBFControllerConfig {
    uint32_t    read_queue_depth    = 64;
    uint32_t    write_queue_depth   = 32;
    std::string scheduler_type      = "frfcfs";  // fcfs, frfcfs, hbf_aware
    uint32_t    controller_latency_cycles = 10;
    bool        enable_read_priority = true;
    uint32_t    max_outstanding_per_subarray = 4;
};

class HBFController : public IModule {
public:
    explicit HBFController(const HBFControllerConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(uint64_t current_cycle) override;
    bool accept(std::unique_ptr<Packet> pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    void connect_downstream(IModule* logic_die) { logic_die_ = logic_die; }

private:
    std::string name_ = "HBFController";
    HBFControllerConfig cfg_;
    std::unique_ptr<IScheduler> scheduler_;
    std::queue<std::unique_ptr<Packet>> read_queue_;
    std::queue<std::unique_ptr<Packet>> write_queue_;
    std::queue<std::unique_ptr<Packet>> completed_queue_;
    IModule* logic_die_ = nullptr;
};

} // namespace obelisk
```

### 5.4 Scheduler(可插拔调度器)

**文件**: `src/controller/scheduler.hh`

```cpp
#pragma once
#include "obelisk/packet.hh"
#include <memory>
#include <vector>

namespace obelisk {

class IScheduler {
public:
    virtual ~IScheduler() = default;

    // 从 ready queue 中选出下一个要发射的 packet
    virtual std::unique_ptr<Packet> select_next(
        std::vector<std::unique_ptr<Packet>>& candidates,
        uint64_t current_cycle) = 0;

    virtual const std::string& name() const = 0;
};

// 工厂函数
std::unique_ptr<IScheduler> make_scheduler(const std::string& type);

} // namespace obelisk
```

**三种实现**:
- `SchedulerFCFS`: 按到达顺序
- `SchedulerFRFCFS`: First-Ready First-Come-First-Served(row hit 优先)
- `SchedulerHBFAware`: **这是论文贡献**。考虑 sub-array 并行度、read/write 比例、KV cache hint

### 5.5 Logic Die

**文件**: `src/logic_die/logic_die.hh`

```cpp
#pragma once
#include "obelisk/imodule.hh"
#include "src/logic_die/channel_mux.hh"
#include "src/logic_die/subarray_dispatcher.hh"
#include "src/logic_die/tsv_model.hh"

namespace obelisk {

struct LogicDieConfig {
    uint32_t num_channels           = 8;
    uint32_t num_dies_per_stack     = 16;
    uint32_t num_subarrays_per_die  = 32;   // 关键参数!论文要 sweep
    uint32_t tsv_bandwidth_gbps     = 200;
    uint32_t logic_die_freq_mhz     = 1000;
    uint32_t command_translation_cycles = 2;
    uint32_t max_concurrent_subarray_cmds = 64;  // logic die 并发能力
};

class LogicDie : public IModule {
public:
    explicit LogicDie(const LogicDieConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(uint64_t current_cycle) override;
    bool accept(std::unique_ptr<Packet> pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    // 连接到 NAND die stack(每个 die 的入口)
    void connect_nand_die(uint32_t die_id, IModule* die);

private:
    std::string name_ = "LogicDie";
    LogicDieConfig cfg_;
    std::unique_ptr<ChannelMux>   channel_mux_;
    std::unique_ptr<SubarrayDispatcher> dispatcher_;
    std::unique_ptr<TSVModel>     tsv_;
    std::vector<IModule*>         nand_dies_;
    std::queue<std::unique_ptr<Packet>> completed_;
};

} // namespace obelisk
```

### 5.6 Sub-Array(HBF 精髓)

**文件**: `src/media/nand/subarray.hh`

```cpp
#pragma once
#include "obelisk/imodule.hh"

namespace obelisk {

struct SubarrayConfig {
    uint32_t page_size_bytes    = 16384;   // 16 KB
    uint32_t pages_per_block    = 512;
    uint32_t blocks_per_subarray = 1024;
    uint32_t tR_cycles          = 50000;   // NAND read latency ~50us @ 1GHz
    uint32_t tPROG_cycles       = 500000;  // NAND program ~500us
    uint32_t tBERS_cycles       = 3000000; // NAND erase ~3ms
    uint32_t cache_buffer_pages = 4;       // sub-array 内 buffer
};

class Subarray : public IModule {
public:
    Subarray(uint32_t id, const SubarrayConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(uint64_t current_cycle) override;
    bool accept(std::unique_ptr<Packet> pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    uint32_t id() const { return id_; }
    bool is_busy() const { return busy_; }

private:
    std::string name_;
    uint32_t id_;
    SubarrayConfig cfg_;

    bool busy_ = false;
    uint64_t busy_until_ = 0;
    std::unique_ptr<Packet> current_;
    std::queue<std::unique_ptr<Packet>> completed_;

    // Buffer(用于捕捉 page locality)
    std::vector<uint64_t> cached_pages_;
};

} // namespace obelisk
```

**关键行为**:
- 一次只能处理一个请求(busy_until_)
- 根据请求类型应用对应 latency(tR / tPROG / tBERS)
- 检查 page 是否在 buffer(命中则延迟大幅减小)
- **这是 HBF 性能建模的核心**

### 5.7 NAND Stack

**文件**: `src/media/nand/nand_stack.hh`

```cpp
#pragma once
#include "obelisk/imedia.hh"
#include "src/media/nand/nand_die.hh"

namespace obelisk {

struct NANDStackConfig {
    uint32_t num_dies_per_stack     = 16;
    uint32_t num_subarrays_per_die  = 32;
    uint64_t capacity_per_die_bytes = 32ULL * 1024 * 1024 * 1024;  // 32 GB per die → 512 GB stack
    SubarrayConfig subarray_cfg;
};

class NANDStack : public IMedia {
public:
    explicit NANDStack(const NANDStackConfig& cfg);

    const std::string& name() const override { return name_; }
    void tick(uint64_t current_cycle) override;
    bool accept(std::unique_ptr<Packet> pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    MediaType media_type() const override { return MediaType::HBF; }
    uint64_t capacity_bytes() const override;
    uint32_t min_access_granularity() const override;
    double peak_bandwidth_gbps() const override;

    // 给 LogicDie 访问
    IModule* get_die(uint32_t die_id);

private:
    std::string name_ = "NANDStack";
    NANDStackConfig cfg_;
    std::vector<std::unique_ptr<NANDDie>> dies_;
};

} // namespace obelisk
```

### 5.8 HBM Interface(复用 DRAMsim3)

**文件**: `src/media/hbm/hbm_interface.hh`

```cpp
#pragma once
#include "obelisk/imedia.hh"

// Forward decl from DRAMsim3
namespace dramsim3 {
    class MemorySystem;
}

namespace obelisk {

struct HBMConfig {
    std::string dramsim3_config_file;  // DRAMsim3 .ini path
    std::string output_dir;
};

class HBMInterface : public IMedia {
public:
    explicit HBMInterface(const HBMConfig& cfg);
    ~HBMInterface();

    const std::string& name() const override { return name_; }
    void tick(uint64_t current_cycle) override;
    bool accept(std::unique_ptr<Packet> pkt) override;
    std::optional<std::unique_ptr<Packet>> dequeue() override;
    bool has_pending() const override;

    MediaType media_type() const override { return MediaType::HBM; }
    uint64_t capacity_bytes() const override;
    uint32_t min_access_granularity() const override { return 64; }
    double peak_bandwidth_gbps() const override;

private:
    std::string name_ = "HBM";
    HBMConfig cfg_;
    std::unique_ptr<dramsim3::MemorySystem> dramsim_;

    // DRAMsim3 是 callback 风格,需要维护 pending 映射
    struct Pending {
        std::unique_ptr<Packet> pkt;
        uint64_t issue_cycle;
    };
    std::unordered_map<uint64_t, Pending> pending_;  // addr → pkt
    std::queue<std::unique_ptr<Packet>> completed_;

    void on_dram_complete(uint64_t addr, bool is_write);
};

} // namespace obelisk
```

### 5.9 Address Mapper

**文件**: `src/memory_system/address_mapper.hh`

```cpp
#pragma once
#include "obelisk/packet.hh"

namespace obelisk {

enum class AddressMappingPolicy {
    ROW_INTERLEAVE,            // 破坏 locality,最大化并行
    CHANNEL_INTERLEAVE_FIRST,  // 典型平衡
    PAGE_CONTIGUOUS,           // 保留 locality
    CUSTOM,                    // 从 TOML 读映射
};

struct AddressMapperConfig {
    AddressMappingPolicy policy = AddressMappingPolicy::CHANNEL_INTERLEAVE_FIRST;
    uint32_t num_stacks      = 1;
    uint32_t num_channels    = 8;
    uint32_t num_dies        = 16;
    uint32_t num_subarrays   = 32;
    uint32_t page_size_bytes = 16384;
    uint64_t hbm_base_addr   = 0x0000000000000000;
    uint64_t hbm_size_bytes  = 96ULL * 1024 * 1024 * 1024;   // 96 GB HBM
    uint64_t hbf_base_addr   = 0x0000001800000000;
    uint64_t hbf_size_bytes  = 512ULL * 1024 * 1024 * 1024;  // 512 GB HBF
};

class AddressMapper {
public:
    explicit AddressMapper(const AddressMapperConfig& cfg);

    // 填入 packet 的 target_media 和 phys 字段
    void map(Packet& pkt) const;

    MediaType which_media(uint64_t addr) const;

private:
    AddressMapperConfig cfg_;
};

} // namespace obelisk
```

**论文实验要点**: `policy` 是一个关键扫描维度。不同策略下 HBF 带宽能差 10x。

### 5.10 Simulation 引擎

**文件**: `src/base/simulation.hh`

```cpp
#pragma once
#include "obelisk/imodule.hh"
#include "src/base/stats.hh"
#include <vector>
#include <memory>

namespace obelisk {

struct SimulationConfig {
    uint64_t    max_cycles     = 100'000'000;
    uint64_t    warmup_cycles  = 1'000'000;
    uint32_t    freq_mhz       = 1000;  // 用于时间单位换算
    std::string output_csv_path;
    std::string output_stats_json_path;
    uint32_t    log_level      = 1;  // 0=off, 1=info, 2=debug, 3=trace
};

class Simulation {
public:
    explicit Simulation(const SimulationConfig& cfg);

    void register_module(std::shared_ptr<IModule> m);

    // 运行直到 all modules has_pending()==false 或 max_cycles
    void run();

    uint64_t current_cycle() const { return cycle_; }

    Stats& stats() { return stats_; }

    // 输出 per-request CSV
    void log_completed(const Packet& pkt);

private:
    SimulationConfig cfg_;
    uint64_t cycle_ = 0;
    std::vector<std::shared_ptr<IModule>> modules_;
    Stats stats_;
    std::ofstream csv_out_;

    void write_csv_header();
};

} // namespace obelisk
```

---

## 6. TOML 配置 Schema

### 6.1 完整示例: `configs/hybrid_hbm_hbf.toml`

```toml
# Obelisk Configuration File
# Target: HBM3E + HBF Gen1 hybrid configuration
# Source parameters: SanDisk fact sheet + SK hynix IEEE paper + JEDEC HBM3 spec

[simulation]
max_cycles              = 100_000_000
warmup_cycles           = 1_000_000
freq_mhz                = 1000
output_csv_path         = "output/hybrid_hbm_hbf.csv"
output_stats_json_path  = "output/hybrid_hbm_hbf_stats.json"
log_level               = 1

[requester]
trace_file              = "traces/samples/llama7b_decode_small.trace"
max_in_flight           = 128
issue_rate_per_cyc      = 2
warm_up                 = true
warm_up_cycles          = 10_000

[host_bus]
bandwidth_gbps          = 128.0    # PCIe 6.0 x16
base_latency_cycles     = 50
queue_depth             = 64
cpu_compute_cycles      = 100

[address_mapping]
policy                  = "channel_interleave_first"
num_stacks              = 8         # 每侧 8 stacks (典型 GPU 配置)
num_channels            = 8
num_dies_hbm            = 12        # HBM3E 12-Hi
num_dies_hbf            = 16        # HBF 16-Hi
num_subarrays_hbf       = 32
page_size_bytes         = 16384
hbm_base_addr           = "0x0000000000000000"
hbm_size_bytes          = "96 GB"
hbf_base_addr           = "0x0000001800000000"
hbf_size_bytes          = "512 GB"  # SanDisk fact sheet

[hbf_controller]
read_queue_depth              = 128
write_queue_depth             = 64
scheduler_type                = "hbf_aware"
controller_latency_cycles     = 10
enable_read_priority          = true
max_outstanding_per_subarray  = 4

[logic_die]
num_channels                  = 8
num_dies_per_stack            = 16
num_subarrays_per_die         = 32
tsv_bandwidth_gbps            = 200
logic_die_freq_mhz            = 1000
command_translation_cycles    = 2
max_concurrent_subarray_cmds  = 64

[nand_media]
# SanDisk BiCS 3D NAND parameters (conservative estimates)
page_size_bytes           = 16384
pages_per_block           = 512
blocks_per_subarray       = 1024
tR_cycles                 = 50_000      # 50us @ 1GHz = read latency
tPROG_cycles              = 500_000     # 500us = program
tBERS_cycles              = 3_000_000   # 3ms = block erase
cache_buffer_pages        = 4
num_dies_per_stack        = 16
num_subarrays_per_die     = 32
capacity_per_die_bytes    = "32 GB"

[hbm_media]
# HBM3E via DRAMsim3
dramsim3_config_file      = "third_party/DRAMsim3/configs/HBM3_8Gb_x128.ini"
output_dir                = "output/dramsim3/"

[stats]
# 哪些 metrics 要输出到 JSON
report_bandwidth          = true
report_latency_histogram  = true
report_per_subarray_util  = true
report_per_module_time    = true
histogram_bins            = 100
```

### 6.2 Python 配置生成器

**文件**: `configs/gen_config.py`

```python
#!/usr/bin/env python3
"""
Generate Obelisk TOML config programmatically.
Supports parameter sweeps for DSE.

Usage:
  python gen_config.py --preset hbf_gen1 --output configs/my_config.toml
  python gen_config.py --preset hybrid --num_subarrays 64 --output configs/sweep_64.toml
"""

import argparse
import toml
from pathlib import Path

PRESETS = {
    "hbf_gen1_sandisk": {
        "logic_die.num_dies_per_stack": 16,
        "logic_die.num_subarrays_per_die": 32,
        "nand_media.tR_cycles": 50_000,
        # ... 填入 SanDisk fact sheet 参数
    },
    "hbm3e_baseline": {
        "hbm_media.dramsim3_config_file": "third_party/DRAMsim3/configs/HBM3_8Gb_x128.ini",
        # ...
    },
    "hybrid": {
        # 混合模式
    },
}

def build_config(preset_name: str, overrides: dict) -> dict:
    """Build TOML dict from preset + overrides"""
    cfg = load_preset(preset_name)
    apply_overrides(cfg, overrides)
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS.keys(), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_subarrays", type=int, default=None)
    parser.add_argument("--num_dies", type=int, default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--trace", default=None)
    args = parser.parse_args()

    overrides = {}
    if args.num_subarrays:
        overrides["logic_die.num_subarrays_per_die"] = args.num_subarrays
    # ... 其他 overrides

    cfg = build_config(args.preset, overrides)
    with open(args.output, "w") as f:
        toml.dump(cfg, f)
    print(f"Written {args.output}")

if __name__ == "__main__":
    main()
```

---

## 7. 输出 CSV Schema

**文件**: `docs/output_format.md`

### 7.1 Per-Request CSV

文件路径由 `simulation.output_csv_path` 指定。每个完成的请求一行。

**Header**:
```
id,type,target_media,addr,size,layer_id,op_name,send,arrive,
host_bus_queuing,host_bus_time,
controller_queuing,controller_time,
logic_die_queuing,logic_die_time,
subarray_queuing,subarray_time,
bank_queuing,bank_time,
tsv_time,total_time,
stack_id,channel_id,die_id,subarray_id,bank_id,page_id
```

**示例行**:
```
0,READ,HBF,0x1800000040,16384,12,attn_k_weight,100,52340,0,50,10,12,15,80,20,52120,0,0,40,52240,0,3,5,17,0,4587520
```

### 7.2 JSON Stats(聚合)

```json
{
  "simulation": {
    "total_cycles": 87_543_210,
    "total_requests": 1_234_567,
    "completed_requests": 1_234_000,
    "sim_time_seconds": 42.3
  },
  "bandwidth": {
    "hbm_read_gbps": 782.1,
    "hbm_write_gbps": 0.5,
    "hbf_read_gbps": 1423.6,
    "hbf_write_gbps": 12.1,
    "aggregate_gbps": 2218.3
  },
  "latency": {
    "p50_cycles": 150,
    "p95_cycles": 52_000,
    "p99_cycles": 85_000,
    "mean_cycles": 8_234
  },
  "utilization": {
    "hbf_subarray_avg": 0.73,
    "hbf_subarray_p99": 0.98,
    "logic_die_dispatcher": 0.81,
    "tsv_utilization": 0.64
  }
}
```

---

## 8. Trace 格式

### 8.1 输入 Trace 格式(Ramulator 兼容扩展)

纯文本,每行一个请求:
```
<timestamp> <type> <addr> <size> [<layer_id>] [<op_name>]
```

- `timestamp`: 绝对 cycle(或 `-1` 表示 back-to-back)
- `type`: `R` / `W` / `RKV` / `RNT`(non-temporal)
- `addr`: 16 进制,64-bit
- `size`: 字节数
- `layer_id`(可选): LLM layer 索引
- `op_name`(可选): 操作名

**示例**:
```
0 R 0x0000000000 64
0 R 0x0000000040 64
100 W 0x1800001000 4096 3 attn_kv_write
200 R 0x0000100000 16384 3 attn_q_weight
```

### 8.2 LLM Trace 生成器完整 Spec

**文件**: `traces/gen_llm_trace.py`

```python
#!/usr/bin/env python3
"""
Generate LLM inference memory access trace for Obelisk.

Models the memory access pattern of transformer inference:
- Weight fetches (HBM or HBF)
- KV cache reads/writes (HBM typically)
- Activations (fit in cache, ignored)

Reference: Llama-3.1 architecture, SanDisk fact sheet Llama-405B baseline
"""

import argparse
import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    name: str
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    ffn_dim: int
    vocab_size: int
    dtype_bytes: int  # 1 for int8, 2 for fp16

@dataclass
class Scenario:
    phase: str  # "prefill" or "decode"
    batch_size: int
    context_length: int
    num_generate_tokens: int

@dataclass
class MemoryLayout:
    weights_base: int
    kv_cache_base: int
    weights_in_hbf: bool  # True = HBF, False = HBM
    kv_cache_in_hbf: bool

MODELS = {
    "llama-3.1-7b": ModelConfig(
        name="llama-3.1-7b",
        num_layers=32, hidden_dim=4096, num_heads=32,
        num_kv_heads=8, ffn_dim=14336, vocab_size=128000, dtype_bytes=2),
    "llama-3.1-70b": ModelConfig(
        name="llama-3.1-70b",
        num_layers=80, hidden_dim=8192, num_heads=64,
        num_kv_heads=8, ffn_dim=28672, vocab_size=128000, dtype_bytes=2),
    "llama-3.1-405b": ModelConfig(
        name="llama-3.1-405b",
        num_layers=126, hidden_dim=16384, num_heads=128,
        num_kv_heads=16, ffn_dim=53248, vocab_size=128000, dtype_bytes=1),  # int8 weights
}

def attention_weight_size(m: ModelConfig) -> int:
    """QKVO projection weights per layer"""
    head_dim = m.hidden_dim // m.num_heads
    q_size = m.hidden_dim * m.hidden_dim
    k_size = m.hidden_dim * m.num_kv_heads * head_dim
    v_size = m.hidden_dim * m.num_kv_heads * head_dim
    o_size = m.hidden_dim * m.hidden_dim
    return (q_size + k_size + v_size + o_size) * m.dtype_bytes

def ffn_weight_size(m: ModelConfig) -> int:
    """FFN weights per layer (gate + up + down, SwiGLU)"""
    return 3 * m.hidden_dim * m.ffn_dim * m.dtype_bytes

def kv_cache_size_per_token(m: ModelConfig) -> int:
    """Per-token KV cache size across all layers"""
    head_dim = m.hidden_dim // m.num_heads
    return 2 * m.num_layers * m.num_kv_heads * head_dim * 2  # 2 bytes fp16

def emit_read(f, ts, addr, size, layer_id=0, op=""):
    f.write(f"{ts} R 0x{addr:016x} {size} {layer_id} {op}\n")

def emit_write(f, ts, addr, size, layer_id=0, op=""):
    f.write(f"{ts} W 0x{addr:016x} {size} {layer_id} {op}\n")

def gen_decode_trace(m: ModelConfig, s: Scenario, layout: MemoryLayout, out_path: str):
    """
    Generate trace for ONE decode step (generating one new token)
    Memory access pattern:
      For each layer:
        1. Fetch attention weights (Q, K, V, O)
        2. Read KV cache up to current context
        3. Write new KV for this token
        4. Fetch FFN weights
    """
    with open(out_path, "w") as f:
        ts = 0  # use -1 for back-to-back in practice
        token_pos = s.context_length  # current position

        for layer in range(m.num_layers):
            # (1) Attention weight fetch
            attn_size = attention_weight_size(m)
            attn_addr = layout.weights_base + layer * (attn_size + ffn_weight_size(m))
            # Split into page-sized reads (16KB typical HBF page)
            PAGE = 16384
            for off in range(0, attn_size, PAGE):
                emit_read(f, ts, attn_addr + off, min(PAGE, attn_size - off),
                          layer, "attn_weight")

            # (2) KV cache read (all previous tokens)
            kv_per_token = kv_cache_size_per_token(m) // m.num_layers
            kv_read_size = kv_per_token * token_pos
            kv_addr = layout.kv_cache_base + layer * kv_per_token * s.context_length * 2
            for off in range(0, kv_read_size, PAGE):
                emit_read(f, ts, kv_addr + off, min(PAGE, kv_read_size - off),
                          layer, "kv_read")

            # (3) KV cache write (for this new token)
            kv_write_addr = kv_addr + token_pos * kv_per_token
            emit_write(f, ts, kv_write_addr, kv_per_token, layer, "kv_write")

            # (4) FFN weight fetch
            ffn_size = ffn_weight_size(m)
            ffn_addr = attn_addr + attn_size
            for off in range(0, ffn_size, PAGE):
                emit_read(f, ts, ffn_addr + off, min(PAGE, ffn_size - off),
                          layer, "ffn_weight")

            ts += 1  # increment per layer (fine-grained ts later)

    print(f"Written trace to {out_path}")

def gen_prefill_trace(m: ModelConfig, s: Scenario, layout: MemoryLayout, out_path: str):
    """
    Prefill: process entire input context in one forward pass.
    Access pattern is similar to decode but with full batched KV writes.
    """
    # Similar structure but process s.context_length tokens at once
    pass  # TODO: implement

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--phase", choices=["prefill", "decode"], default="decode")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=2048)
    parser.add_argument("--num_tokens", type=int, default=128)
    parser.add_argument("--weights_in_hbf", action="store_true")
    parser.add_argument("--kv_in_hbf", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    m = MODELS[args.model]
    s = Scenario(args.phase, args.batch_size, args.context_length, args.num_tokens)
    layout = MemoryLayout(
        weights_base = 0x1800000000 if args.weights_in_hbf else 0x0000000000,
        kv_cache_base = 0x1C00000000 if args.kv_in_hbf else 0x4000000000,
        weights_in_hbf = args.weights_in_hbf,
        kv_cache_in_hbf = args.kv_in_hbf,
    )

    if args.phase == "decode":
        gen_decode_trace(m, s, layout, args.output)
    else:
        gen_prefill_trace(m, s, layout, args.output)

if __name__ == "__main__":
    main()
```

---

## 9. HBF 参数来源与真实值

### 9.1 SanDisk HBF Gen1 参数(来自公开资料)

| 参数 | 值 | 来源 |
|---|---|---|
| Per-die capacity | 256 Gb (32 GB) | SanDisk fact sheet |
| Dies per stack | 16 | SanDisk fact sheet |
| Total capacity per stack | 512 GB | SanDisk fact sheet |
| Peak read bandwidth per stack | 1.6 TB/s | SanDisk fact sheet |
| Page size | 16 KB (推断) | BiCS NAND typical |
| tR (read latency) | ~50 μs | 3D NAND typical,需校准 |
| Sub-arrays per die | 32 (推断) | "parallel sub-array" 描述 |

### 9.2 HBM3E 参数(来自 JEDEC + vendor 规格)

| 参数 | 值 | 来源 |
|---|---|---|
| Per-stack bandwidth | ~1 TB/s | SK hynix/Micron 规格 |
| Per-stack capacity | 24-36 GB (8-12 Hi) | JEDEC HBM3 |
| Channels per stack | 16 (pseudo-channels) | JEDEC |
| tCL | ~14 ns | JEDEC HBM3 |

### 9.3 参数不确定性标注

所有 `configs/*.toml` 必须有三类注释标记:
- `# CONFIRMED: <source>` — 有权威来源
- `# ESTIMATED: <reasoning>` — 基于合理推断
- `# TODO: needs validation` — 占位,M2 前必须解决

---

## 10. 构建与依赖

### 10.1 CMakeLists.txt 结构

```cmake
cmake_minimum_required(VERSION 3.20)
project(Obelisk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(OBELISK_BUILD_TESTS "Build unit tests" ON)
option(OBELISK_ENABLE_THERMAL "Enable DRAMsim3 thermal module" OFF)

# DRAMsim3 submodule
add_subdirectory(third_party/DRAMsim3)

# Third-party via FetchContent
include(FetchContent)
FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.14.1)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.6.0)
FetchContent_MakeAvailable(tomlplusplus spdlog)
if(OBELISK_BUILD_TESTS)
    FetchContent_MakeAvailable(Catch2)
endif()

# Core library
add_library(obelisk_core
    src/base/simulation.cc
    src/base/stats.cc
    src/frontend/requester.cc
    src/frontend/trace_reader.cc
    src/host/host_cpu.cc
    src/host/host_bus.cc
    src/controller/hbf_controller.cc
    src/controller/scheduler_fcfs.cc
    src/controller/scheduler_frfcfs.cc
    src/controller/scheduler_hbf_aware.cc
    src/logic_die/logic_die.cc
    src/media/nand/nand_stack.cc
    src/media/nand/nand_die.cc
    src/media/nand/subarray.cc
    src/media/hbm/hbm_interface.cc
    src/memory_system/memory_system.cc
)
target_include_directories(obelisk_core PUBLIC include src)
target_link_libraries(obelisk_core PUBLIC tomlplusplus::tomlplusplus spdlog::spdlog dramsim3)

# Main executable
add_executable(obelisk src/main.cc)
target_link_libraries(obelisk PRIVATE obelisk_core)

# Tests
if(OBELISK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

### 10.2 构建命令

```bash
# 初始克隆
git clone --recursive https://github.com/YOUR_ORG/obelisk.git
cd obelisk

# 构建
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j8

# 运行
./obelisk ../configs/hybrid_hbm_hbf.toml

# 测试
ctest --output-on-failure
```

---

## 11. M1 四周详细开发计划

### Week 1: 骨架与核心数据结构

**目标**: 能编译,能跑 synthetic trace,输出空 CSV

**Day 1-2**:
- 搭项目骨架(CMakeLists, .gitignore, .clang-format)
- 引入 DRAMsim3 submodule、tomlplusplus、spdlog、Catch2
- CI 跑通(GitHub Actions)
- 实现 `include/obelisk/types.hh`、`packet.hh`、`imodule.hh`、`imedia.hh`

**Day 3-5**:
- 实现 `Simulation` 事件驱动引擎
- 实现 `Stats` 统计收集器
- 实现 `TraceReader` 和 `Requester`
- 实现 dummy `HostBus`(仅加固定延迟)
- 写通一个 echo pipeline: Requester → HostBus → (dummy media returns immediately)

**验收**: `./obelisk configs/echo.toml` 能输出带正确 id 和 timing 的 CSV

### Week 2: HBF 核心建模

**目标**: 完整的 HBF-only 读路径能跑

**Day 6-8**:
- 实现 `AddressMapper`
- 实现 `HBFController`(仅 FCFS scheduler)
- 实现 `LogicDie`(channel mux + simple dispatcher)

**Day 9-10**:
- 实现 `Subarray`(只读,tR 参数化)
- 实现 `NANDDie`、`NANDStack`
- 实现 `TSVModel`

**验收**:
- `./obelisk configs/hbf_only_read.toml traces/synthetic_seq.trace`
- 输出的聚合 bandwidth 在 SanDisk fact sheet 的 1.6 TB/s ±20% 内
- 每个 packet 的 `subarray_time` 和 `tsv_time` 字段非零且合理

### Week 3: HBM baseline 与混合架构

**目标**: 能对比 HBM vs HBF 的 baseline 实验

**Day 11-13**:
- 实现 `HBMInterface`(封装 DRAMsim3)
- 实现 `MemorySystem` orchestrator,能同时挂 HBM 和 HBF

**Day 14-15**:
- 实现 `SchedulerFRFCFS`
- 实现 LLM trace generator `gen_llm_trace.py`
- 跑 Llama-7B decode trace 的 HBM-only vs HBF-only 对比

**验收**:
- HBM-only 跑出来的 bandwidth 匹配 DRAMsim3 standalone 输出(误差 <5%)
- HBM-only vs HBF-only 的 latency 曲线有明显差异,方向符合预期(HBF 延迟更高但容量更大)

### Week 4: HBF-aware scheduler + 论文图

**目标**: 产出第一张"hero chart"

**Day 16-18**:
- 实现 `SchedulerHBFAware`(论文贡献占位版本,利用 sub-array 并行信息)
- 实现 address mapping policy 切换
- 实现 `plot_results.py`

**Day 19-20**:
- Sweep 实验: num_subarrays × address_policy × scheduler
- 产出 3-5 张草图版本的论文 figure
- 写 arxiv vision paper 草稿(4-6 页)

**验收**:
- 能画出 "Effective bandwidth vs num_subarrays" 曲线,说明 subarray 并行度的 trade-off
- 有初步 insight 可以写成 paper

---

## 12. M2+ 路线图概要

**M2(Week 5-16, ~3 个月)**:
- 补 write 路径、简化 FTL(不做 GC 也不做 wear leveling,只做 page mapping)
- 跑完整的 RQ-1 和 RQ-2 实验矩阵
- 投 arxiv preprint
- 实现 4-5 个 address mapping policy
- 完整的 `SchedulerHBFAware` 算法设计

**M3(Week 17-32, ~4 个月)**:
- 完成 RQ-3(容量比例 sweep)
- 完成 RQ-4(HBF vs SSD tiered 对比)
- 完整实验、ablation study、sensitivity analysis
- 投 ISCA 或 MICRO(通常有 winter/summer deadline)

**M4(Week 33-52)**:
- 根据审稿意见 revise
- 代码开源(论文中稿后)
- 写 tutorial、做 artifact evaluation

---

## 13. 测试策略

### 13.1 单元测试(Week 1 起就要写)

每个核心模块至少 3 个测试:

**`tests/unit/test_subarray.cc`**:
```cpp
#include <catch2/catch_test_macros.hpp>
#include "src/media/nand/subarray.hh"

TEST_CASE("Subarray: single read latency matches tR", "[subarray]") {
    SubarrayConfig cfg{.tR_cycles = 50000};
    Subarray sa(0, cfg);
    auto pkt = std::make_unique<Packet>();
    pkt->type = PacketType::READ;
    // Accept at cycle 0
    REQUIRE(sa.accept(std::move(pkt)));
    // Tick 49999 cycles, should still be busy
    for (uint64_t c = 0; c < 49999; c++) sa.tick(c);
    REQUIRE(sa.is_busy());
    // Tick one more, should complete
    sa.tick(49999);
    auto out = sa.dequeue();
    REQUIRE(out.has_value());
}

TEST_CASE("Subarray: buffer hit reduces latency", "[subarray]") { ... }
TEST_CASE("Subarray: back-to-back requests queue correctly", "[subarray]") { ... }
```

### 13.2 集成测试

**`tests/integration/test_e2e.cc`**: 端到端跑一个小 trace,检查输出 CSV 的 packet 数量、bandwidth 范围、latency 分布。

### 13.3 回归测试

**`tests/regression/golden/`**: 存储每个 config + trace 组合的"黄金输出"。`run_regression.sh` 跑所有组合,diff 输出是否和 golden 一致(允许 timing 字段有小于 1% 的差异)。

**关键**: **M1 结束前建立至少 5 组 golden baseline,防止后续开发退化。**

---

## 14. AE Scripts 模板

为论文 artifact evaluation 准备,**每张 figure 一个脚本**。

**文件**: `AE-scripts/fig_main.sh`

```bash
#!/bin/bash
# Reproduce Figure 1: HBM vs HBF vs Hybrid bandwidth comparison
# Expected runtime: ~30 min

set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

OUTDIR="output/fig_main"
mkdir -p "$OUTDIR"

# Generate traces if not cached
if [ ! -f "traces/llama7b_decode_1k.trace" ]; then
    python3 traces/gen_llm_trace.py \
        --model llama-3.1-7b --phase decode \
        --context_length 1024 --num_tokens 128 \
        --output traces/llama7b_decode_1k.trace
fi

# Run three configurations
for cfg in hbm3e_baseline hbf_gen1_sandisk hybrid_hbm_hbf; do
    echo "Running $cfg..."
    python3 configs/gen_config.py \
        --preset "$cfg" \
        --trace traces/llama7b_decode_1k.trace \
        --output "$OUTDIR/${cfg}.toml"
    ./build/obelisk "$OUTDIR/${cfg}.toml" > "$OUTDIR/${cfg}.log"
done

# Plot
python3 output/plot_results.py \
    --inputs "$OUTDIR/hbm3e_baseline.csv" "$OUTDIR/hbf_gen1_sandisk.csv" "$OUTDIR/hybrid_hbm_hbf.csv" \
    --labels "HBM3E" "HBF Gen1" "Hybrid" \
    --output "$OUTDIR/fig_main.png"

echo "Figure saved to $OUTDIR/fig_main.png"
```

---

## 15. 论文实验计划(倒推模拟器需求)

### 15.1 计划中的论文主图

- **Fig 1**: HBM-only / HBF-only / Hybrid 三种配置在 Llama-7B/70B/405B decode 下的有效带宽 bar chart
- **Fig 2**: 效能对比曲线 - TTFT/TPOT vs Context length(HBM vs Hybrid)
- **Fig 3**: Sub-array 并行度 sweep(N=8,16,32,64,128),有效带宽曲线
- **Fig 4**: Address mapping policy 对比(row_interleave vs channel_first vs page_contig)
- **Fig 5**: Scheduler 对比(FCFS vs FR-FCFS vs HBF-aware)
- **Fig 6**: HBM:HBF 容量比例 sweep,perf/TCO curve
- **Fig 7**: Sensitivity - tR latency ±30% 下 end-to-end 影响
- **Fig 8**: Power / Energy 分解(M3+)

### 15.2 实验矩阵

| 维度 | 取值 |
|---|---|
| Model | Llama-7B, Llama-70B, Llama-405B |
| Phase | decode (primary), prefill |
| Context length | 1K, 4K, 16K, 64K |
| Config | HBM-only, HBF-only, Hybrid-50/50, Hybrid-25/75 |
| Sub-arrays | 8, 16, 32, 64, 128 |
| Scheduler | FCFS, FRFCFS, HBFAware |
| Mapping | row_interleave, channel_first, page_contig |

**全矩阵 = 3×2×4×4×5×3×3 = 4320 runs**。M2 末期用 sweep 脚本并行跑。

---

## 16. 工程实践要求(给 AI 编码助手的附加指令)

### 16.1 代码风格
- 4 空格缩进,`.clang-format` 配置为 Google style (120 列)
- 所有 public API 写 Doxygen 注释
- 头文件用 `#pragma once`
- 不使用 `using namespace` 在头文件中
- 优先 `std::unique_ptr`,避免裸 new

### 16.2 命名
- 类: `PascalCase`
- 函数/变量: `snake_case`
- 成员变量: `snake_case_` 带下划线后缀
- 常量: `kPascalCase`
- 文件名: `snake_case.hh/.cc`

### 16.3 错误处理
- 配置解析错误 → 立即 exit,打印清晰错误
- 运行时异常(bug)→ `assert` + 日志
- 文件 I/O 失败 → 抛 `std::runtime_error`,main() 捕获
- **不吞异常**

### 16.4 日志
- 用 `spdlog`,分级: trace/debug/info/warn/error
- info 级别:只打 per-module 启动和聚合结果
- debug 级别:每个 packet 关键事件
- trace 级别:所有 tick

### 16.5 性能
- 主循环 `Simulation::run()` 必须是 **hot path** → 所有 virtual call 在此路径上要有数据
- Packet 对象在事件驱动中频繁创建,考虑用 `std::pmr::pool_resource` 或对象池(M2 优化)
- 大 trace 文件必须流式读,不能一次全加载

### 16.6 可调试性
- 每个模块的状态都要能 dump(`to_string()` 方法)
- `--log_level trace` 模式下,能输出完整的 packet 流转时间线

---

## 17. 给 AI 编码助手的 prompt 模板

当你让 AI 实现某个模块时,推荐的 prompt 格式:

```
你在实现 Obelisk 项目中的 [模块名]。

项目总体设计文档: [贴 Obelisk_Design_Spec.md 相关章节]

你的任务是实现以下文件:
- src/[path]/[module].hh (接口已在 Section X.X 定义)
- src/[path]/[module].cc (需你实现)

要求:
1. 严格按照 Section X.X 的接口,不改动 public API
2. 遵循 Section 16 的代码风格
3. 实现对应的 unit test: tests/unit/test_[module].cc,至少 3 个测试用例
4. 内部数据结构可自主设计,但要在 .cc 开头用注释说明思路
5. 所有 config 字段必须可从构造函数读取,不硬编码

参考模块实现:
[贴一个已完成模块的 .cc 作为 style reference]

请先输出你对该模块的实现计划(3-5 bullet 点),再输出代码。
```

---

## 18. 决策日志(未来变更记录)

这一节随项目演进更新。M1 期间所有偏离本文档的决策,都记录在此。

| 日期 | 决策 | 原因 | 影响 |
|---|---|---|---|
| Day 0 | 初版发布 | - | - |

---

## 附录 A: 关键参考

1. SanDisk HBF Fact Sheet (2025): 1.6 TB/s, 512GB/stack, 16-Hi BiCS NAND
2. SK hynix HBM+HBF IEEE paper (2025): 2.69× perf/W improvement on Llama inference
3. PKU Xerxes (FAST'26): 参考代码骨架、TOML+Python 配置模式、per-request CSV 设计
4. CMU-SAFARI Ramulator 2.0: C++20 模块化设计参考
5. UMD DRAMsim3: 直接作为 HBM baseline,thermal 模块可选
6. CMU-SAFARI MQSim: NAND 建模思路参考(不直接依赖)
7. JEDEC HBM3 Standard (JESD238A): HBM3 timing 参数
8. Sandisk-SK hynix OCP HBF Standardization (2026-02): 后续标准跟踪

---

## 附录 B: 术语表

- **HBF**: High Bandwidth Flash - SanDisk 提出的 3D 堆叠 NAND 内存架构
- **HBM**: High Bandwidth Memory - 3D 堆叠 DRAM
- **Sub-array**: HBF 中 NAND die 内部的独立可访问子阵列
- **Logic die**: HBF stack 底部的逻辑芯片,负责命令分发
- **TSV**: Through-Silicon Via,3D 堆叠垂直互连
- **CBA**: CMOS directly Bonded to Array (SanDisk 3D NAND 工艺)
- **TTFT**: Time To First Token (LLM inference 指标)
- **TPOT**: Time Per Output Token
- **FTL**: Flash Translation Layer
- **tR / tPROG / tBERS**: NAND read / program / erase 延迟
- **KV Cache**: LLM 推理中存储的 Key-Value 矩阵

---

**文档版本**: 1.0
**最后更新**: 2026-04-20
**维护者**: Obelisk 项目组
