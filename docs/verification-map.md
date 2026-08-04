# 验证地图：实现正确性、校准与外部验证的边界

本文件回答的是“代码是否忠实执行了声明的模型”。它不回答“声明的模型是否
就是未来 HBF 硅片”。前者由仓库内测试覆盖；参数校准和外部硬件验证的现状见
[`model-validation.md`](model-validation.md)。

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
外部可信度还需要独立模拟器对拍、参数校准和未参与校准的数据集验证。

## 模块覆盖

### 真实 workload 输入与 Frontier memory object

| 合同 | 检验 | 入口 |
|---|---|---|
| Qwen-Bailian 生产 suite | pinned Thinking 源、70B workload identity、steady/burst/long-context-decode-tail 三个 256-request 选择器、会话祖先、到达时间、token 长度与 prefix block hash 确定；独立 verifier 从源 JSONL 逐字节重建三个 CSV 并拒绝 unattached artifact；不伪造 token ID 或地址 | `qwen_bailian_frontier_adapter` |
| Frontier scheduler/KV 守恒 | memory contract v4 的 batch 顺序、token/KV frontier、output boundary、block count、allocator-state digest、residency plan v1 与完整 lookup→admission→touch/allocate→release 状态机逐事件重放；`full_request_kv_reservation_v1` 在每个 batch cursor 独立复算所有活跃请求的最大 KV frontier 并证明不超过逻辑 block pool；物理状态只有一个 lifecycle source，任一 plan 字段或 lifecycle 篡改均 fail-closed | `frontier_integration_bundle`, `frontier_hybrid_residency_contract`, `frontier_replay_accounting_audit` |
| 模型与地址绑定 | 模型 JSON 路径/SHA-256、HF identity、selected precision profile、embedding sharing、dense shape、RMSNorm 与 Frontier config 精确相等；schema-v2 descriptor 独立反算 W8A16/BF16 的 matrix payload、non-matrix payload、per-output-channel scale metadata、attention/FFN/norm/boundary分项和最大 active-weight buffer；trace schema v4 独立复算 unique footprint、双 active buffer、hot/cold KV、backing、unused HBM 与 whole-block slack；有限 KV slot 复用同一地址，相邻 layer 精确相差一个 stride；partial-prefill 不读取 lm_head；任一账本、plan 或 object-map 漂移均 fail-closed | `frontier_hybrid_residency_contract`, `frontier_memory_object_export` |
| HBFSim 结构接入与大 span | 导出 trace 可由 C++ parser/census 消费；读写/phase/layer/byte census 与 manifest 相等，phase 因果与 layer 窗口字段独立；同一 trace 的 all-HBM/HBF/CXL/NVMe 执行器逐项锁定 traffic、credit、HBM geometry、backed placement/HBM traffic/backing bytes，trace 漂移即失败；该门只证明结构集成，不授予性能资格；TiB 重叠 span 用区间并集计数，空 trace、坏 phase 与重复字段均 fail-closed | `frontier_memory_structural_integration`, `trace_span_census_scalability` |
| Frontier 结构接入边界 | 7B descriptor 仅用于 CI smoke；schema-v7 replay auditor 强制绑定经过独立验证的三窗口 suite、源/config/manifest digest、70B W8A16/BF16 账本、完整 prefix block 语义、admission policy 与 hybrid plan；唯一 structural runner 固定三窗口和 96 GiB/pressure=1.0 sensitivity 点，并只在全部通过后生成 `paper_result_eligible=false` 的聚合 receipt；单-trace四类 memory baseline runner 同样只产生 memory-system service evidence，并已从真实 HBF 计数闭环唯一 canonical WAF、零写 `null` 和 physical-write/usable-capacity；不可执行且从未产出证据的 72-cell full-replay runner 已删除。当前执行路径只接受 burst 的五个正交 cell，并由独立 verifier 精确聚合 q=0/q=0.5/q=1 的 60 行分层结果；任一 digest、时间分位行、baseline 指标、WAF、claim 或 census 漂移都 fail-closed，且禁止跨分层平均、full-replay 冒充与 `paper_result_eligible` 提升；dummy timing 明确禁止 TTFT/TPOT/SLO/end-to-end/time-throughput claim | `frontier_70b_structural_suite_contract`, `frontier_70b_two_axis_slice_contract`, `frontier_70b_two_axis_evidence_certificate`, `frontier_memory_structural_integration`, `frontier_canonical_waf_contract`, `experiment_certificate_gates` |
| Frontier 完整 cold-KV 初态 | 显式 residency contract 的全部 cold-KV（包括窗口未触达页）在 time zero 占用真实 Data/Mapping 页和 free-page budget；紧凑 mutable image 的 overwrite、checkpoint、GC、物理写与最终 FTL 状态必须和逐页物化差分一致；初态不计 workload write/WAF | `hbf_compact_mutable_initial_image`, `layer_streaming_stats` |
| Frontier 容量网格 preflight | 从 tracked W8/BF16 descriptor 独立推导权重、最大 active buffer 和 KV page；覆盖 2H6F/4H4F/6H2F 的 96/192/288 GiB 与 0.75/1.0/1.25/1.5/2.0 共 30 点；逐字节校验三个 request CSV 并复算最大单请求 `full_request_kv_reservation_v1`，明确区分 planner-infeasible、request-infeasible、execution-candidate；preflight 不冒充运行结果 | `frontier_70b_capacity_grid_preflight` |
| Frontier 70B WAF/磨损合同 | 完整 256-request burst lifecycle 精确计算三种 HBM/HBF 配比的 cold-KV 逻辑写；full-lifecycle placement fixed point、三组预声明 block residue、one/eight epoch、trace/summary digest、程序页/物理字节/GC/erase 守恒和唯一 canonical WAF 任一漂移均 fail-closed；寿命仅允许未校准 uniform-wear sensitivity | `frontier_70b_hbf_wear_study_contract` |
| ASTRA Llama 3.1 8B | sibling profile 精确反算 8,030,261,248 参数、14.96 GiB BF16 常驻权重、16 GiB batch-1 128K KV 容量和逐 kind 流量；HBFSim 独立复算 profile/权重/KV/初态/地址对象并对篡改 fail-closed | sibling `test_llama31_8b_profile_has_exact_full_model_and_128k_boundary`, `astra_replay_artifact_transaction` |

### HBM

| 机制 | 检验 | 入口 |
|---|---|---|
| 频率/位宽/BL 自洽 | pin rate、DQ width、PC split、BL 唯一推导带宽/字节/tCK/tBL；burst→PC→bank-group 映射在非 2 次幂拓扑及 uint64 边界也可逆；顺序流达到 ≥85% 派生峰值；命令因果对齐 CK；tCCD 精确；非法组合拒绝 | `physical_probe hbm-interface` |
| burst/mapping 边界拆分 | 66 B 非对齐请求覆盖 4 bursts、2 pseudo-channels；父完成聚合；越容量拒绝 | `physical_probe hbm-boundaries` |
| 行状态 | miss 无 PRE、hit 无 ACT、conflict 先 PRE 后 ACT | `hbm-row` |
| 通道并行 | 固定工作量下多通道加速且所有目标 pch 活跃 | `hbm-channels` |
| refresh 副作用 | refresh 增加 stall；长 idle 跨多个窗口仍关行并重新 ACT | `hbm-refresh` |
| 派生 burst 时序 | RD↔WR、WR→PRE gate 使用 width/rate/BL 唯一派生的 data burst | `hbm-turnaround` |
| 激活约束 | 任意 tFAW 窗口最多四次 ACT | `hbm-tfaw` |
| FR-FCFS 与同步等价路径 | 已到达 hit 可越队、QD1 顺序、无未卜先知、年龄上限；diagnostics-free、空队列、单 stripe 请求的同步路径与显式 enqueue/pump 在逐 completion、累计状态和后续 ticket 上 bit-exact；跨 stripe 必须回退，扩大 eligibility 的 production mutation 必须被杀死 | `hbm-frfcfs`, `foundational_mutation_validation` |

### HBF 读写与映射

| 机制 | 检验 | 入口 |
|---|---|---|
| 子页写与 OOB | 小写仍产生完整数据页及映射页物理工作 | `hbf-rounding` |
| 全量 L2P 常驻 | 每次逻辑访问经过 per-stack controller DRAM；物理读只含数据页，不读取 mapping page；容量不足 fail-closed | `hbf-resident-mapping`, `hbf-exact-calendar`, `uc-r1` |
| 映射持久化 | resident L2P 更新立即可见；不同 mapping page 的 dirty 项在 drain 时持久化 | `hbf-mapping-writeback`, `uc-w3` |
| 写缓冲合并 | 同页多写合并、读己之写、最终只编程所需页 | `hbf-coalesce`, `uc-w1`, `uc-w7` |
| slot 生命周期 | slot 与旧 generation 同时保留到 mapping publish，过载产生可计算背压且不出现双 owner | `hbf-write-backpressure`, `hbf-exact-calendar`, `uc-w5` |
| program 排他 | 同一 plane 的 program/verify 不重叠 | `uc-w6` |
| 批激活 | ideal independent ≤ batch；单 subarray 明显更慢；页/字节守恒相同 | `uc-r2` |
| 精确预约日历 | 超过 64 个 idle gaps 与 sense rounds 后仍完整回填；早到 read 可回填 future program 前空档，但完整 sense/lane/page-buffer 路径不得跨越 full-plane barrier；issue/drain 逆序 fail-closed | `hbf-exact-calendar` |
| 擦除 epoch / block 因果 | erase 退休旧 block epoch，迟到 mapping/cache 回调不可复活数据；只等待目标 block，同 stack 独立 plane 可并行 | `hbf-exact-calendar` |
| raw physical 所有权 | `RawPhysical` block 与 FTL data/mapping/GC 分离，跨所有权 program fail-closed | `hbf-exact-calendar` |
| unmapped read | 返回 NAND erased-value 的 exact-byte SRAM/HBIO payload，不产生虚假 media/ECC/mapping | `hbf-ecc-pipeline` |
| 内部资源分解 | plane、subarray、media lane、page-buffer bank、HB IO 均有定向并行测试 | 对应 `hbf-*` probes |
| TSU | read/program/erase 均经 source queue 与 die sequencer | `hbf-tsu` |
| 提交因果 | 未完成的 mapping/buffer/read fill 不得提前命中；同 LPN 写后读等待 | `physical_probe all` 中定向断言 |

程序和擦除当前不可抢占，因此不存在 suspend 快速读用例。重新引入 suspend 前，必须先
支持可修订完成事件，并添加总 plane-time 守恒与已返回 completion 不可失真的测试。

### FTL / GC

| 机制 | 检验 | 入口 |
|---|---|---|
| GC 触发 | 真实低水位触发 relocation 和 erase | `hbf-gc` |
| 顺序失效 | 有充分 aging room 时 GC 发生且搬迁为零 | `uc-g1` |
| 写放大恒等式 | `page_programs = data_programs + mapping_page_programs + gc_relocations`；物理写字节等于页程序数×页大小 | `waf_cases` |
| 运行时页状态审计 | 每个 block 的 valid/invalid/pending/free、valid bitmap、全局与 per-stack free pages、free-pool 唯一所有权、L2P/mapping-directory 可达性以及总容量逐次 fail-closed 守恒 | `physical_probe all`, summary `hbf_accounting_verified` |
| GC victim 守恒 | data+mapping relocation 等于总搬迁；搬迁+回收无效页等于 `gc_runs×pages_per_block`；user+GC erase、flash scheduler、ECC、物理字节跨层精确守恒 | `gc_waf_quick_accounting`, `waf_verifier_mutation_contract`, `uc-g1` |
| pending raw erase 所有权 | 正在 raw erase 的空 free block 必须退出 allocation/headroom pool，commit 后恰好归还一次，不得擦除同到达新分配 | `pending_free_block_erase_ownership_ok` |
| 唯一 WAF 口径 | `waf = physical_write_bytes/logical_write_bytes`；包含数据页取整、mapping checkpoint 与 GC relocation，排除 OOB/ECC/链路字节；零分母为未定义 | `hbf-rounding`, `waf_cases` |
| 组成层写口径 | cooperative host bytes（含覆盖）与 dirty-union destaged bytes 分开；端到端 media/user 不复用 HBF-device 分母 | `composition-coop-write`, `check_ec_sanity.py` |
| 多 regime | 顺序、KV FIFO、子页 RMW 的锐利边界；两档随机搅拌要求非零 relocation、FIFO 保守上界，以及 WAF/平均 victim 有效页比例随占用率单调增加，不从 FIFO 稳态近似臆造 finite-run 下界 | `waf_cases` |
| Frontier 70B KV wear | exact full-lifecycle cold-KV 逻辑写与 capacity-scaled media replay 分离；主 epoch 必须真实执行 GC/erase，最终表只保留 logical bytes、physical payload bytes、一个 WAF 和 physical-write/capacity | `frontier_70b_hbf_wear_study_contract` |
| resident mapping exact oracle | 4096 页 read→write 精确得到 8192 次 lookup、4096 次 update、0 次 mapping read、8 次 drain checkpoint program，并守恒 page/free/ECC/scheduler/WAF 计数 | `gc_waf_quick_accounting` |
| 延迟 invalidation | hard-pressure GC 可等待最早 commit；soft GC 不推进未来状态来制造 victim | `physical_probe all` |
| relocation metadata 持久化 | data-page relocation 更新 resident L2P、标脏对应 VPN，并在 drain 写回；前台不读取 mapping page，checkpoint program 单独计入 | `gc_relocation_mapping_persistence_ok` |
| role-aware headroom | Data/Mapping active block append 不误扣完整 free block，不触发虚假 GC | `gc_role_headroom_no_false_trigger_ok` |
| soft-watermark 收敛 | active-GC 剩余页计入 relocation headroom；受压场景只搬一次目标 live set，不跑满 block guard | `gc_active_relocation_headroom_converges_ok` |
| 多 plane 分配预测 | role 独立 cursor；完整 block debit 记到实际 opening plane/request，不转嫁给下一次 active-block append | `gc_multiplane_opening_debit_ok` |

### 组成层与初态

| 机制 | 检验 | 入口 |
|---|---|---|
| FLAT 路由 | 边界下/上请求计数精确；跨页请求逐 placement 拆分 | `uc-c1`, `physical_probe all` |
| HBF placement | 连续 data stripe 对每个 stack 恰好一页，mapping page 保持 stack-local；`8×k` 对抗 stride 仍经 group rotation 分散；EC5 静态页复用同一 data placement，并在小几何上全域双射、可逆、越界拒绝 | `composition-static-mapping` |
| 静态块隔离 | 静态页所属 block 不再进入可变 FTL 分配 | `composition-static-mapping` |
| EC5 绕 FTL | eligible read 只走 static physical HBF；logical read/write、mapping、program、D2D 均为零，write 留在 HBM | `check_ec_sanity.py`, `composition-kind-blind` |
| 初始镜像 | read 观察到未被此前 write 覆盖的字节才推断旧页；full-page write→read 不读取未来 | `composition-initial-image`, `composition-reuse-routing` |
| Lazy 顺序初始镜像 | HBF 区间首尾落在 partial mapping page 时，紧凑表示与 materialized trace 的场景结果逐字段相同；未启用 EC6 时其 buffer 参数不占 HBM、也不阻塞 direct-only | `lazy_sequential_equivalence_and_option_scoping` |
| 协作写 | 固定页 slot；同页/重叠子页合并；未对齐跨页拆分；destaged bytes 等于 D2D write bytes；按 stack 拆分 | `composition-coop-write`, `uc-c2` |
| 写后读一致性 | parked/foreground-write 版本走 HBM，不回退到 stale raw HBF | `composition-reuse-routing`, `uc-c3` |
| EC6 初始层 DMA | 完整只读权重 population 的所有 NAND block 在任何 FTL 分配前被物理隔离，实际权重读走预解析、跨 stack 条带化 static extent；mutable cold/overflow KV 从初始版本起始终走 logical FTL，hot KV 常驻 HBM，绝不回退到 stale static 版本 | `summary_hybrid_residency_observability`, `composition-static-mapping` |
| EC6 读写顺序 | read-before-overwrite 会读取旧页；first-touch full-page overwrite 跳过旧页；同页写回按版本完成 | `summary_hybrid_residency_observability` |
| EC6 双缓冲守恒 | HBF(static+logical)→D2D→HBM install 与 HBM→D2D→HBF writeback 逐页/逐字节守恒；奇偶 buffer 复用有因果 fence | `summary_hybrid_residency_observability`, `check_ec_sanity.py` |
| EC6 容量与调度模式 | Frontier manifest/object-map/trace 经 SHA-256 绑定的独立生成器转成自包含显式 residency contract，执行时再次核对 trace bytes/SHA-256 并拒绝 `max-ops`；完整 runtime overhead/block table 与 plan 指定的低 ID hot KV 常驻，未触达对象仍计容量，权重及 cold/overflow KV 由所选 backing tier 后备；byte-exact pressure basis、逐对象页舍入、双 buffer、resident/backing/unused 分区守恒且不超 HBM；canonical per-trace runner 原子生成 all-HBM/HBF/CXL/NVMe receipt 并拒绝 trace、placement、scheduler/credit、HBM geometry 或 backing traffic 漂移；未知对象、权重/metadata 地址越界或别名、KV 越界、页粒度漂移、partial contract 和 buffer 不足均 fail-closed；phase/layer 独立 | `frontier_residency_object_binding`, `frontier_memory_structural_integration`, `summary_hybrid_residency_observability`, `external_backing_layer_streaming_contract` |
| 无 HBF 外部后备 | CXL memory/外置 SSD 共用同一 M2S→controller→media→S2M 设备路径；全局 E2E credit、earliest-gap backfill、returned completion、payload/protocol/wire、local-span utilization 与地址热图守恒；同页粒度下 HBF/external 的 EC6 plan 和 HBM traffic 逐字段恒等 | `external_backing_device_contract`, `external_backing_layer_streaming_contract` |
| HBM 容量溢出对照 | 4H4F 的 192 GiB HBM 上精确记账 200 GiB 饱和写入与 8.000977 GiB FIFO offloading/readback；offered/service 分布分开；三种 backing 均扫 1/4/16/64/256-page exact service curve；converged layer sample 推导 1/2/4/8/16/32 次 restore、winner 与 break-even；raw sample/executable digest、direct exploratory 状态和坏 artifact 均 fail-closed | `capacity_overflow_composition_contract`, `capacity_overflow_experiment_end_to_end` |
| External 独立实现 | 六个 canonical case 覆盖 read/write/full-duplex/controller-media queue/global outstanding/cross-channel backfill；独立 Python scheduler 与 ledger reducer 不调用 production helper；32-seed fuzz、time-shift/ID-rename metamorphic 和两个人为 production fault 全被门禁识别 | `foundational_independent_oracles`, `foundational_property_fuzz`, `foundational_mutation_validation` |
| kind 可选 | 无 semantic kind 的 trace 在每种组合中都有完整定义 | `composition-kind-blind` |
| Behavior-only placement | HBF 权威后备、有限 HBM page tier；首次 bypass/复用提升/真实 LRU/脏页写回；完整 decision ledger 与独立 Python oracle 相同；semantic 擦除、地址双射、时间平移、future suffix、window/device timing 均不改变策略；六类 production mutation 全须被 kill | `behavioral_tiering_composition`, `foundational_behavioral_differential`, `behavioral_placement_experiment_contracts`, `foundational_mutation_validation` |
| Behavior workload 资格 | occupied footprint、address-span inflation、读写数、reuse、精确 LRU stack distance 和 working-set/HBM 关系由 case 预声明并 fail-closed；性能不参与资格判定 | `trace_locality_analyzer`, `behavioral_workload_quality_contracts` |
| 时间语义 | completion 不早于到达+服务；Direct 同 phase 父请求按页轮转且同页读写不越序；EC6 layer prefetch 服从 admission/层依赖；独立 HBM/HBF 资源不被全局 HOL 阻塞 | `uc-c4`, `summary_hybrid_residency_observability`, `composition-reuse-routing`, `physical_probe all` |
| outstanding credit | credit 以 4 KiB 物理页事务而非父 trace record 计数；满窗只等待最早 completion；等价 4 KiB/大请求表达产生相同物理完成跨度 | `closed_loop_window_completion_order`, `composition-reuse-routing` |

### 指标、配置与可复现性

| 合同 | 检验 | 入口 |
|---|---|---|
| 时间原点不变 | 所有 arrival 同移 1 ms，active span、吞吐与利用率/并行度差异不超过 4 ULP | `uc-q1` |
| 非法配置拒绝 | 负 timing、NaN、Inf 均非零退出 | `uc-q2` |
| 配置 replay | 导出的 resolved config 重放，科学 payload 完全相同 | `ctest` replay tests |
| provenance | schema、git/dirty、build、argv/cwd、trace SHA-256、完整 config 均存在 | `compare_summary_results.py` |
| 解析边界 | HBM busy work 由 bytes/bandwidth 精确重算；隔离 HBF read pipeline 的 25 点 media/HBIO 相界由两组长度消去启动项；Direct mixed makespan 等于独立 tier 较慢者且两侧 resource work 不变；删点、改区或串行化均拒绝 | `foundational_analytical_microbench`, `foundational_certificate_contracts` |
| overflow 发布绑定 | certificate schema v6 同时绑定 scenario/validation/physical/overflow 四个 executable、全部 external-validation tracked input、sanitizer platform/runtime policy，并要求 behavior-only placement differential 与 analytical gate；direct artifact 固定为 `exploratory_unattached`；paper runner 校验 clean commit、输入 digest 与 exact overflow binary 后才附证 | `foundational_certificate_contracts`, `experiment_certificate_gates`, `capacity_overflow_experiment_end_to_end` |
| 指标命名 | 用户完成吞吐与含 drain 的 makespan 吞吐分别报告；主延迟为 offer→completion，service/source 分布独立保留；wall-clock span、latency work、stage work、resource busy 分域 | summary schema v16, `summary_time_breakdown_contract` |
| 时间分解合同 | offered + post-offer = user completion；user completion + drain = makespan；JSON/CSV scalar 一致；trace off/full 不改变结果 | `summary_time_breakdown_contract` |
| E2E 时间报告 | 四个生产 runner 默认生成 suite-level long-form CSV、Markdown 与自包含 HTML；可视化将 additive wall、latency work、overlapping stage work 与 resource capacity 分面呈现；父子归属、坏合同写前拒绝及原子产物均有门禁 | `time_breakdown_report_renderer`, `time_breakdown_visualization_renderer`, `astra_replay_artifact_transaction`, `synthetic_output_smoke_end_to_end`, `use_cases_quick_contracts` |
| 地址热图记账 | 五 domain、固定 source 顺序、读写擦方向、bin byte 精确守恒、跨 bin access fan-out、`UINT64_MAX` 顶地址与 `2^64` exclusive end、固定容量与溢出 fail-closed | `address_heatmap_accounting` |
| 地址热图集成/渲染 | 每个 v16 scenario 均嵌入热图；HBM/HBF/external physical bytes 对齐设备统计；零流量 hottest 为 n/a；多场景须显式选择；C++ JSON 经 Python 严格校验并生成无网络依赖 HTML | `summary_time_breakdown_contract`, `address_heatmap_renderer`, `address_heatmap_cpp_python_interop`, `external_backing_layer_streaming_contract` |
| EC6 层流式可观测性 | mode、explicit-contract 标志、byte-exact pressure basis、page-allocated footprint/rounding、层数、显式 layer/compute、完整 fixed/hot-KV/backing/unused HBM 分区、stream/writeback 页与字节守恒；backing credit limit、实际峰值并发及 admission wait 可复核；用户访问全走 HBM；exposed/hidden prefetch 与 buffer-reuse wait 可复核；JSON/CSV/console 一致 | `summary_hybrid_residency_observability` |
| HBF 等待可观测性 | ingress、下游 scheduler、ECC issue、write-buffer slot 四类累计 work 在 JSON/CSV/console 一致 | `summary_hbf_stage_work_observability` |
| ECC/数据边界 | 首读 latency、同 die II、跨 die 并行、OOB/raw-payload 字节、饱和吞吐、内部流量绕过 HBIO、partial erased-fill、重复 physical program fail-closed、读写共享 issue port 均有守卫 | `hbf_ecc_pipeline_contract`, `physical_guard` |
| Read-buffer 内容代际 | erase 清空 block cache；program 在不改变 block epoch 时仍精确失效目标 PPN 的 erased-value cache line | `program_invalidates_read_buffer_ok`, `erase_read_buffer_epoch_ok` |
| Pending mapping / erase 因果 | 尚未发布的 LPN/metadata target 被 raw erase retire 后，后续访问仍等待 mapping callback 与目标 block erase，且不会复活旧 epoch | `erase_pending_mapping_dependency_ok`, `erase_pending_vpn_dependency_ok`, `erase_epoch_no_resurrection_ok` |
| destructive ordering / state time | 同 block erase 不越过已发 read/program；raw read 等待同 PPN program；targeted wait 保持 program→mapping 顺序，独立 plane 的结果不随 API 调用排列改变 | `erase_waits_target_read_ok`, `erase_waits_target_program_ok`, `raw_read_waits_target_program_ok`, `targeted_state_no_reverse_time_travel_ok`, `erase_plane_call_order_invariant_ok` |

## 质量门

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
cmake --build build --target physical_guard
cmake --build build --target use_cases
cmake --build build --target waf_cases
```

CI 还会在 Debug/Release、Linux/macOS 和 Address/UndefinedBehavior Sanitizer
组合中运行相应门。任何模型改动都应同时更新测试期望、参数证据和本地图；不允许只更新
结果基线来“接受”漂移。
