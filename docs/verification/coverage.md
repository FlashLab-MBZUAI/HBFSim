# Verification coverage

> Status: Current
> Last reviewed: 2026-09-29

本文件回答的是“代码是否忠实执行了声明的模型”。它不回答“声明的模型是否
就是未来 HBF 硅片”。前者由仓库内测试覆盖；参数校准和外部硬件验证的现状见
[evidence policy](../reference/evidence-policy.md)。

## 证据类型

| 类型 | 检查内容 | 主要能发现 | 不能证明 |
|---|---|---|---|
| 精确守恒 | 请求、页、字节、映射、搬迁、链路流量逐项相等 | 丢失、重复、错路由、错记账 | 时间模型或参数真实 |
| 手算时序 | 从显式配置推导下界、上界或容差区间 | 隐藏延迟、漏 gate、提前完成 | 配置值来自实测 |
| 锐利 regime | 构造应产生数量级或方向性差异的状态 | 机制未生效或走错分支 | 机制幅度与硬件一致 |
| 理论边界 | 数学界、单调性、容量界、时间平移不变性 | 规模性偏差、指标定义错误 | 完整微架构 |
| 交叉一致 | 两条独立路径或 replay 必须相同 | 组合层、序列化或报告分歧 | 两条路径不会同错 |
| Fail-closed | 非法范围、容量、NaN/Inf、因果倒退必须拒绝 | “看似合理”的无效结果 | 合法输入就一定物理真实 |

这些证据互补，但即使全部通过，也只代表当前实现对当前模型具有较高内部可信度。
现有 Ramulator2/MQSim 对拍只覆盖两个明确的窄 facet；更广的外部可信度仍需要参数
校准和未参与校准的数据集验证。校准 overlay 背后的硬件测量见 [evidence](../../evidence/README.md)。

## 模块覆盖

### HBM

当前唯一模型为 `channel-aggregate-v2`；不再验证或报告 DRAM 行状态和刷新事件。
服务组默认包含 4 个 pseudochannel（`service_group_channels`）；组大小 1 用于同一路径的精度敏感性对照。
canonical ledger 以服务组为分辨率：每个被触及的组记一个 `hbm/group{N}` 事件，其服务时长为组内最忙
lane 的 burst 数（隔离传输保持 lane 带宽，组内不相交并发 lane 串行），`physical_bytes` 为该组全部 lane
的 burst 取整字节。ledger reducer 据此精确复算读写字节、总线 work、完成时间与活动组数；逐 lane 计数
（`active_pseudo_channels`、`max_pseudo_channel_accesses`）不在事件中，reducer 只用组事件给出上下界
（每个活动组含 1..组宽个活动 lane；最忙 lane 介于 ceil(组 burst/组宽) 与组 burst 之间），精确值由独立
oracle 的逐 burst 计数与 production summary 逐字段比对。

| 机制 | 检验 | 入口 |
|---|---|---|
| 频率/位宽/BL 自洽 | 接口推导物理字节与原始带宽，有效带宽决定服务时间 | `physical_probe hbm-interface` |
| 地址与容量 | 非对齐 burst 取整、地址映射可逆、越容量拒绝；独立逐 burst 枚举核对聚合流量 | `physical_probe hbm-boundaries`, `hbm_channel_model` |
| 通道并行 | 固定工作量下多通道加速、通道独立 | `physical_probe hbm-channels` |
| 有限队列与公平性 | 大小请求轮转、容量背压、外部边界限制聚合、单 parent 合并与逐片段推进一致 | `hbm_channel_model`, `simulation_session_contract` |
| 控制器缓冲共享 HBM | 物理容量预留、应用地址隔离、通道争用、未来预约和空隙回填、共同前沿回收、HBM/HBF 流量守恒 | `hbm_controller_buffer`, `hbm_channel_model` |
| 读写转换 | 配置的方向转换开销作用于应用通道服务 | `physical_probe hbm-turnaround` |
| 时钟舍入 | 一 ULP 边界残差、时间平移不变性、长时运行 | `hbm_clock_roundoff` |
| 同步账本 | 独立逐 burst 地址计数、按服务组取最忙 lane 的区间参考、按组的读写方向切换，核对完成时间、物理字节与逐 lane 计数；fuzz 同时变化组宽 1/4 | `hbm.channel-service`, `hbm.burst-boundary`, `hbm.map-exhaustive`, `hbm.dual-pch-overlap`, `hbm.rw-turnaround`, `oracle_property_fuzz` |

### HBF device contract and Host management

The current boundary is OCP v0.7.0. Protocol framing and complete electrical
conformance are outside this transaction simulator; see [model](../reference/model.md).

| Contract | Evidence entry |
|---|---|
| Grades 1/2/3, per-channel command/RX/TX work, geometry limits, aligned reads and sequential writes | `ocp_standard_contract`, `hbf_ocp_v070_grades_effective` |
| Same-Bank sensing serializes; different Banks overlap; trace on/off preserves service | `physical_probe hbf-bank-pipeline` |
| ECC initiation throughput differs from response latency; OOB stays internal | `physical_probe hbf-ecc-pipeline` |
| Two decoded pages per Bank, temporal fills, eviction, invalidation and future-gap scheduling | `hbf_read_buffer_pressure_test`, `hbf_read_buffer_handoff_test` |
| Host subpage assembly sends one complete device page; alignment covers both request ends | `physical_probe hbf-host-boundary` |
| Program/read/erase barriers; first and reused page-zero programs pay actual erase | `physical_probe hbf-program-barriers` |
| Independent interval oracle for logical reads, subpage writes, Host copies, zero-live and live-page reclamation, and hybrid routing | `canonical_oracle_cases` |
| Host mapping ledger: every lookup or update pays `mapping_control_compute_ns` (`mapping_lookup_compute` / `mapping_update_compute`) on the shared host compute pool (one worker per stack; equal starts choose the most recently freed worker, idle since no earlier than the request arrival), then one `ctrl_dram_issue_ns` issue on `host/mapping/partition{stack}` whose entry transfer shares the HBM service-group calendars; an update then publishes for `mapping_update_ns` (`mapping_publish_compute`) on the pool. User and GC updates share that stage: the reducer derives GC updates from relocated data pages, requires one publication per update access, and reconstructs `mapping_compute_work_ns` / `mapping_compute_ops` from the `mapping_compute` events. Case notes state the rule; fuzz also charges zero control work | `canonical_oracle_cases`, `oracle_property_fuzz` |
| Resident/cached Host mapping, page/entry/affine layouts, GC reserve progress and exact metadata capacity | `hbf_mapping_cache_test`, `hbf_mapping_layout_test`, `hbf_gc_cached_mapping_progress` |
| Host GC and static wear leveling use ordinary HBF transfers and finite copy slots | `hbf_gc_reserve_test`, `hbf_static_wear_leveling_test`, `verify_write_amplification` |
| Program classes and WAF conserve bytes; logical reclaim count differs from physical erase count | `verify_write_amplification`, `gc_waf_quick_accounting` |
| Channel-local, block-aligned zones; invalid-only swaps; P/E stays with physical blocks | `hbf_host_zones_test`, `test_hbf_host_zones.py` |
| Canonical v5 persistent Host/media image; fresh-process restore preserves physical wear and future work | `hbf_persistent_image_test`, `simulation_session_checkpoint_contract` |
| Exact ownership and publication accounting for compact mutable populations | `hbf_compact_mutable_test`, `simulation_session_contract` |
| Optional RC thermal sensitivity is an assumption, not the complete standard fault-state machine | `hbf_thermal_test` |

A passing test establishes its stated contract only. The independent Python
Host oracle is bounded to its declared tiny request sequences; arbitrary
concurrent GC policy behavior is covered by native causal/conservation tests,
not claimed as a separately implemented full Host controller.

### 组成层与初态

| 机制 | 检验 | 入口 |
|---|---|---|
| FLAT 路由 | 边界下/上请求计数精确；跨页请求逐 placement 拆分 | `flat-address-routing`, `verify_physical` |
| HBF placement | 连续 data stripe 对每个 stack 恰好一页，mapping page 保持 stack-local；`8×k` 对抗 stride 仍经 group rotation 分散；`direct-read` 静态页复用同一 data placement，并在小几何上全域双射、可逆、越界拒绝 | `composition-static-mapping` |
| 静态块隔离 | 静态页所属 block 不再进入可变 FTL 分配 | `composition-static-mapping` |
| `direct-read` 绕 FTL | eligible read 只走 static physical HBF；logical read/write、mapping、program、D2D 均为零，write 留在 HBM | `composition-kind-blind`, `verify_physical` |
| 初始镜像 | read 观察到未被此前 write 覆盖的字节才推断旧页；full-page write→read 不读取未来 | `composition-initial-image`, `composition-reuse-routing` |
| Lazy 顺序初始镜像 | HBF 区间首尾落在 partial mapping page 时，紧凑表示与 materialized trace 的场景结果逐字段相同；未启用 hbf-streaming 时其 buffer 参数不占 HBM、也不阻塞 direct-only | `lazy_sequential_equivalence_and_option_scoping` |
| 协作写 | 根据显式 watermark、有限 slot 压力或最终 drain 执行 destage；固定页 slot；同页/重叠子页合并；未对齐跨页拆分；destaged bytes 等于 D2D write bytes；按 stack 拆分 | `composition-coop-write`, `composition-write-conservation` |
| 写后读一致性 | parked/foreground-write 版本走 HBM，不回退到 stale raw HBF | `composition-reuse-routing`, `composition-read-after-write` |
| hbf-streaming 初始层 DMA | 完整只读权重 population 的所有 NAND block 在任何 FTL 分配前被物理隔离，实际权重读走预解析、跨 stack 条带化 static extent；mutable cold/overflow KV 从初始版本起始终走 logical FTL，hot KV 常驻 HBM，绝不回退到 stale static 版本 | `summary_hybrid_residency_observability`, `composition-static-mapping` |
| hbf-streaming 读写顺序 | read-before-overwrite 会读取旧页；first-touch full-page overwrite 跳过旧页；同页写回按版本完成 | `summary_hybrid_residency_observability` |
| hbf-streaming 双缓冲守恒 | HBF(static+logical)→D2D→HBM install 与 HBM→D2D→HBF writeback 逐页/逐字节守恒；奇偶 buffer 复用有因果 fence | `summary_hybrid_residency_observability` |
| hbf-streaming 容量与调度模式 | 显式 residency contract 执行时核对 trace bytes/SHA-256 并拒绝 `max-ops`；runtime overhead/block table 与 plan 指定的 hot KV 常驻，未触达对象仍计容量，权重及 cold/overflow KV 由所选 backing tier 后备；byte-exact pressure basis、逐对象页舍入、双 buffer、resident/backing/unused 分区守恒且不超 HBM；未知对象、地址越界或别名、KV 越界、页粒度漂移、partial contract 和 buffer 不足均 fail-closed；phase/layer 独立 | `summary_hybrid_residency_observability`, `external_backing_layer_streaming_contract` |
| 无 HBF 外部后备 | on-package LPDDR、host DRAM、CXL memory 与 NVMe SSD 共用同一 M2S→controller→media→S2M 设备路径；address page、caller range 与 transport segment 分层记账；read/write caller range 分方向 round-robin 选择 queue，range 内 segment 固定 queue、再按地址 stripe 到 per-queue channel，同一 queue/channel timeline 在读写间共享；方向 aggregate bandwidth 按该方向 queues×channels 均分；全局 E2E credit、earliest-gap backfill、returned completion、payload/protocol/wire、local-span utilization 与地址热图守恒；同页粒度下 HBF/external 的 hbf-streaming plan 和 HBM traffic 逐字段恒等 | `external_backing_device_contract`, `external_backing_layer_streaming_contract` |
| External 独立实现 | 八个 canonical case 覆盖 read/write/full-duplex/controller-media queue/global outstanding/cross-channel backfill、host-DRAM 16 KiB range 按 8 KiB transport command 分成 2 段并覆盖 4 个 address page，以及两个同到达 read ranges 分配到两个 queue 且各自 segments 不跨 queue；独立 Python scheduler 与 ledger reducer不调用 production helper，并验证共享 queue/channel 资源不重叠、方向带宽按 queues×channels 均分；fuzz 同时变化 read/write queue 数，32-seed、time-shift/ID-rename metamorphic 和两个人为 production fault 全被门禁识别 | `canonical_oracle_cases`, `oracle_property_fuzz`, `verify_mutations` |
| kind 可选 | 无 semantic kind 的 trace 在每种组合中都有完整定义 | `composition-kind-blind` |
| Behavior-only placement | HBF 权威后备、有限 HBM page tier；首次 bypass/复用提升/真实 LRU/脏页写回；完整 decision ledger 与独立 Python oracle 相同；semantic 擦除、地址双射、时间平移、future suffix、window/device timing 均不改变策略；十类 production mutation 全须被 kill | `behavioral_tiering_policy`, `placement_oracle_differential`, `verify_mutations` |
| Behavior workload 资格 | occupied footprint、address-span inflation、读写数、reuse、精确 LRU stack distance 和 working-set/HBM 关系由 case 预声明并 fail-closed；性能不参与资格判定 | `trace_locality_analyzer`, `workload_quality_contracts` |
| 时间语义 | completion 不早于到达+服务；Direct 同 phase 父请求按页轮转且同页读写不越序；hbf-streaming layer prefetch 服从 admission/层依赖；独立 HBM/HBF 资源不被全局 HOL 阻塞 | `arrival-time-causality`, `summary_hybrid_residency_observability`, `composition-reuse-routing`, `verify_physical` |
| outstanding credit | credit 以 4 KiB 物理页事务而非父 trace record 计数；满窗只等待最早 completion；等价 4 KiB/大请求表达产生相同物理完成跨度 | `closed_loop_window_completion_order`, `composition-reuse-routing` |

### 指标、配置与可复现性

| 合同 | 检验 | 入口 |
|---|---|---|
| 时间原点不变 | 所有 arrival 同移 1 ms，active span、吞吐与利用率/并行度差异不超过 4 ULP | `time-origin-invariance` |
| 非法配置拒绝 | 负 timing、NaN、Inf 均非零退出 | `invalid-config-rejection` |
| 配置 replay | 导出的 resolved config 重放，科学 payload 完全相同 | `ctest` replay tests |
| provenance | schema、git/dirty、build、argv/cwd、trace SHA-256、完整 config 均存在 | `analytical_microbench`, `certificate_contracts` |
| 解析边界 | HBM busy work 由 bytes/bandwidth 精确重算；隔离 HBF read pipeline 的 25 点 media/HBIO 相界由两组长度消去启动项；Direct mixed makespan 等于独立 tier 较慢者且两侧 resource work 不变；删点、改区或串行化均拒绝 | `analytical_microbench`, `certificate_contracts` |
| 证书绑定 | certificate schema v8 同时绑定 `hbfsim`、ledger probe、physical probe、全部 external-evidence tracked input、sanitizer platform/runtime policy，并要求 behavior-only placement differential 与 analytical gate；直接得到的 summary 固定为 `exploratory_unattached`，只有校验 clean commit、输入 digest 与 exact binary 后才附证 | `certificate_contracts` |
| 指标命名 | 用户完成吞吐与含 drain 的 makespan 吞吐分别报告；主延迟为 offer→completion，service/source 分布独立保留；wall-clock span、latency work、stage work、resource busy 分域 | summary schema v18, `summary_time_breakdown_contract` |
| 时间分解合同 | offered + post-offer = user completion；user completion + drain = makespan；JSON/CSV scalar 一致；trace off/full 不改变结果 | `summary_time_breakdown_contract` |
| 时间报告 | 报告渲染器从 summary 生成 long-form CSV、Markdown 与自包含 HTML；可视化将 additive wall、latency work、overlapping stage work 与 resource capacity 分面呈现；父子归属、坏合同写前拒绝及原子产物均有门禁 | `time_breakdown_report_renderer`, `time_breakdown_visualization_renderer`, `component_contracts` |
| 地址热图记账 | 五 domain、固定 source 顺序、读写擦方向、bin byte 精确守恒、跨 bin access fan-out、`UINT64_MAX` 顶地址与 `2^64` exclusive end、固定容量与溢出 fail-closed | `address_heatmap_accounting` |
| 地址热图集成/渲染 | 每个 v18 scenario 均嵌入热图；HBM/HBF/external physical bytes 对齐设备统计；零流量 hottest 为 n/a；多场景须显式选择；C++ JSON 经 Python 严格校验并生成无网络依赖 HTML | `summary_time_breakdown_contract`, `address_heatmap_renderer`, `address_heatmap_cpp_python_interop`, `external_backing_layer_streaming_contract` |
| hbf-streaming 层流式可观测性 | mode、explicit-contract 标志、byte-exact pressure basis、page-allocated footprint/rounding、层数、显式 layer/compute、完整 fixed/hot-KV/backing/unused HBM 分区、stream/writeback 页与字节守恒；backing credit limit、实际峰值并发及 admission wait 可复核；用户访问全走 HBM；exposed/hidden prefetch 与 buffer-reuse wait 可复核；JSON/CSV/console 一致 | `summary_hybrid_residency_observability` |
| HBF 等待可观测性 | ingress、下游 scheduler、ECC issue、write-buffer slot 四类累计 work 在 JSON/CSV/console 一致；底层队列压力由物理 probe 独立触发 | `summary_time_breakdown_contract`, `physical_probe all` |
| ECC 与数据边界 | 原始 page+OOB 的内部工作与外部 payload 分开计数；ECC 启动间隔、响应延迟与 Host 复制分别建模 | `hbf_ecc_pipeline_contract`, `verify_physical` |
| Bank decoded cache | 每 Bank 两页、第三页逐出、跨 Bank 隔离、未来填充不可见、page-zero erase 清除缓存；Host 逻辑失效不会伪造设备缓存清除 | `ocp_standard_test`, `oracle_property_fuzz` |
| Program completion 与回收 | 无跳页写入；每页各自的 program 完成时间约束读取；page-zero 自动擦除；Host zone reset 不重复计擦除 | `ocp_standard_test`, `hbf_host_zones_test` |
| Mapping / GC 发布顺序 | 悬挂 program、映射发布和 Host reclaim 的所有权在 drain 后守恒；每个搬迁的 live data page 经用户 L2P 路径恰好发布一次 GC 映射更新，mapping page 搬迁只做 checkpoint relocate；随机案例对比独立 oracle，压力测试检查回收进展 | `canonical_oracle_cases`, `oracle_property_fuzz`, `hbf_gc_cached_mapping_progress` |

## 质量门

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
cmake --build build --target verify_physical
cmake --build build --target verify_components
cmake --build build --target verify_write_amplification
```

CI 还会在 Debug/Release、Linux/macOS 和 Address/UndefinedBehavior Sanitizer
组合中运行相应门。任何模型改动都应同时更新测试期望、参数证据和本地图；不允许只更新
结果基线来“接受”漂移。
