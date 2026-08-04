# Behavior-only HBM/HBF placement：从“看起来合理”到可反证

## 1. 为什么不能用吞吐曲线证明 simulator 正确

同一条曲线可能同时由正确机制、错误 workload、错误 placement、错误流量记账，
或两个相互抵消的 bug 产生。因此，`all-HBF < hybrid < all-HBM` 之类的排序只能是
待解释的模型输出，不能成为正确性判据。反过来，hybrid 超过 all-HBM 或 all-HBF
也不自动代表 bug：两个介质和 D2D 链路可能并行，关键在于资源账本是否允许这种
重叠、访存是否真的走了两条用户路径，以及迁移开销是否被完整计入。

本仓采用下面的信任链。前一层失败时，后一层的性能数字没有解释资格。

| 层 | 要回答的问题 | 机械判据 | 失败归因 |
|---|---|---|---|
| L0 基础器件 | 单个 HBM/HBF/link/FTL 原语是否符合规格和手算时序 | foundational certificate、canonical oracle、ledger reducer | 基础设备模型 |
| L1 workload 资格 | trace 是否真的具有它声称的规模、地址占用和局部性 | case-declared quality contract | workload/实验设计 |
| L2 placement 语义 | 给定 address/op 顺序，策略状态转移是否正确 | 独立 Python oracle 逐字段相等 | placement 实现 |
| L3 变形关系 | 无关输入变化是否错误地改变策略 | semantic 擦除、地址双射、时间平移、future suffix、window/timing invariance | 因果或调度耦合 |
| L4 守恒和终态 | 数据移动、容量、脏页和 completion 是否闭合 | action/byte/path/capacity/drain/quiescence identities | 组成层或记账 |
| L5 门禁有效性 | 上述检查是否真的能发现错误 | production-code mutation 必须全部被 kill | 验证盲区 |
| L6 性能探索 | 在已通过 L0–L5 的前提下，模型预测什么 | 同输入、同窗口的多场景 stress run | 可解释的模型结果 |

`sanity=PASS` 只覆盖 L4 中的一部分；它不是整条信任链的缩写。

## 2. 当前 behavior-only policy 的可执行规格

HBF 是完整、权威的 backing store，HBM 是有限物理 page tier。控制器只能读取
请求的地址、读写类型和因果顺序；`kind`、`label`、`phase`、`layer`、
`compute_ns` 均不得影响 placement 决策。这里必须区分“策略输入”和“执行
依赖”：`phase` 的 complete-before-next barrier 仍由 composition front-end
执行并计入 source latency，只是不能被 admission/replacement policy 当成未来
提示。多页 parent 也必须和 direct composition 一样按 parent round-robin 分享
page-transaction window。

当前提供两个明确的策略：

- `always-admit`：每个 HBM miss 都装入 HBM，作为会受 scan pollution 影响的
  demand-fill 对照。
- `reuse-filtered`：首次观察留在 HBF 并进入有界 ghost history；达到复用阈值后
  才提升；HBM 满时按真实 LRU 淘汰。

提升读页必须依次产生 HBF logical read、D2D read、HBM physical install。
整页写提升可跳过旧 backing fill。脏淘汰必须依次产生 HBM read、D2D write、
HBF logical write；结束时所有剩余脏页也必须写回。

策略状态在 trace 到达顺序中规划，物理设备随后并发执行该计划。这一分离是硬
合同：改变 HBM/HBF latency、arrival spacing 或 outstanding window 可以改变
完成时间和利用率，但不得改变 action、slot、victim 或 dirty decision。否则模型
实际上是在用偶然的设备完成顺序充当未声明的 placement 输入。
物理 round-robin 通过 per-page observation queue 和 victim dependency edge
执行这个计划：同页的 hit 不能越过更早 promotion，replacement 也不能越过
victim page 上更早的访问；无依赖的 page 仍可并行。

另一个硬合同是退化等价：当 reuse-filtered 面对纯冷 trace、所有 page 都 bypass
时，它必须与同配置、同 phase、同窗口的 all-HBF direct path 具有相同 front-end
时序和 HBF 账本。这个对照能直接抓出 phase barrier 丢失和大请求独占窗口，而
不依赖“吞吐看起来像不像”。

## 3. 独立 oracle 与完整决策指纹

`validation/behavioral_oracle.py` 是顺序 shadow state machine，不调用 production
C++ helper，也不知道 HBM/HBF 时序。qualification run 使用 `W=1` 和充分大的
interarrival，把生产 event system 化约到同一个数学问题，然后逐字段比较：

- hit、bypass、cold/history hit、promotion；
- clean/dirty eviction 和 ghost eviction；
- fill、install、writeback、drain 的 page/byte 数；
- peak/final resident、dirty 和 history 状态；
- 完整决策指纹。

指纹覆盖 observation、request/fragment、page、op、action、history evidence、
HBM slot、victim page 和 victim dirty bit；它排除 semantic annotation 和物理
时延。这样，“错误 LRU 恰好得到相同 hit 数”也不能通过。

## 4. Workload 资格不是事后描述

每个 case 在执行前声明最小操作数、实际占用页数、读写数、每页触达数、地址
span inflation、是否应有 reuse、working set 与 HBM 容量的关系，以及需要时的
`reuse_within_hbm_ratio` 下界。分析器使用精确 LRU stack distance，而不是地址
跨度或抽样近似。

这能拒绝几类常见伪实验：

- “500 GB”只是最高地址很大，实际只触达少量页；
- trace 太短，warm-up 后没有可测 steady state；
- 声称 hot set fits HBM，但 reuse distance 实际超过 tier capacity；
- 声称 scan 无复用，但生成器重复了页；
- 地址洞导致 nominal footprint 远大于 occupied footprint。

## 5. 当前机制矩阵

`tools/run_behavioral_placement_experiments.py` 对每个 case 先跑 qualification，
再跑共同 stress window。十个 case 不是十次重复，而是十种可判定机制：

| case | 角色 | 必须出现的机制签名 |
|---|---|---|
| sequential/random scan | negative control | 全 bypass、零 promotion/hit/migration |
| hotset fits | positive control | 每页一次 bypass、一次 promotion，随后稳定 hit |
| capacity edge | boundary control | 正好装满且零 replacement |
| cyclic over capacity | negative control | 有 reuse，但 stack distance 不适合 HBM |
| hot+cold pollution | placement opportunity | cold bypass，hot resident，fill 少于 demand-fill |
| phase shift | adaptation control | 不用 future hint 学会第二个 hot set |
| skewed read | distribution sweep | HBM/HBF 两条用户路径都被使用 |
| dirty hotset fits | write control | 无脏淘汰，只在 drain 精确写回 |
| dirty over capacity | write pressure | 脏淘汰及 HBM→D2D→HBF 字节精确闭合 |

stress 表同时给出 all-HBM、all-HBF、case-local mixed Flat、demand-fill 和
reuse-filtered。性能排序不参与 PASS/FAIL。

## 6. 故障注入证明门禁有牙齿

behavioral mutation gate 会逐一改坏生产 C++，重新构建，并要求测试失败：

1. 首次触达就提升；
2. HBM write 后丢弃 dirty state；
3. 偷看 `SemanticKind::Scratch`；
4. 用 MRU 代替 LRU；
5. promotion 时跳过 D2D fill；
6. replacement 越过 victim page 上更早的决策；
7. 忽略显式 phase dependency；
8. 让一个多页 parent 独占 foreground window；
9. planned HBM hit 越过同页更早的 promotion；
10. replacement 不在 victim page 留下 eviction tombstone。

baseline、每个 mutant 和恢复后的 HEAD 都必须经过同一 gate。某个 mutant
survive 代表验证存在已知盲区，不能签发证书，也不能把性能结果升级为论文证据。

## 7. 遇到“奇怪结果”时的固定诊断顺序

1. quality fail：先修 trace；禁止解释吞吐。
2. oracle fail：最小化首个 decision divergence；这是 placement bug。
3. metamorphic fail：检查策略是否偷看 semantic/timing，或请求是否发生越序。
4. conservation/quiescence fail：沿 HBF、D2D、HBM 和 dirty lifecycle 查丢失或
   重复流量。
5. L0–L5 全过但性能意外：再读 user-path census、migration amplification、
   HBM data-bus、HBF channel-data、HBF plane-media 和 D2D utilization，验证
   并行关键路径。
6. 若结论依赖绝对 GB/s：必须再加入真实硬件校准和误差区间；内部自洽不能替代
   外部有效性。

## 8. 可重复入口与证据边界

```bash
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure

python3 -B validation/run_behavioral_differential.py \
  --scenario-compare build/scenario_compare \
  --config configs/scenario_compare/server-4k-hbf4x.cfg \
  --seeds 32 --operations 48 \
  --report build/foundational-behavioral-differential.json

python3 -B tools/run_behavioral_placement_experiments.py \
  --profile core \
  --scenario-compare build/scenario_compare \
  --config configs/scenario_compare/server-4k-hbf4x.cfg \
  --out-dir out/behavioral-placement-core
```

实验目录是事务性产物，包含 trace digest、locality、quality、W=1 qualification、
stress summaries、逐项 checks、CSV、Markdown 和 manifest。未绑定有效
foundation certificate 时，manifest 明确标记为 `exploratory_unattached`。

这套方法能证明“实现忠实于当前可执行规格、因果关系稳定、账本闭合，而且门禁能
抓住所列错误”。它不能单独证明参数等同于某一代真实器件，也不能证明当前简单
reuse filter 是最优 placement。后者正是后续 policy sweep 和真实 workload
探索要回答的问题。
