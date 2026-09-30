# HBF 映射算法实现与运行入口

> Status: Current
> Last reviewed: 2026-09-24

六类候选已落到原生模拟器，块映射进一步分为纯块和块加日志，共七个可运行方案。它们使用相同的 NAND、ECC、HBIO 和共享 HBM 设备模型。算法选择、内存记录尺寸和软件时间是模拟器策略，不是 OCP 指定的 FTL。

## 选择算法

日常使用直接选 `configs/overlays/hbf/mapping/` 下的 overlay（`resident`、`page-cache`、`entry-cache`、`compressed-cache`、`block`、`block-log-striped`、`extent-striped`），例如
`python3 -m hbfsim sweep --overlay cached-l2p-1-over-1000 --vary overlay=mapping/page-cache,mapping/block --scenarios all-hbf`。
下表的 `.cfg` 是 `tests/fixtures/mapping-organizations/` 中的测试夹具：一个 32 MiB 单 plane 的 HBF，先加载 `configs/systems/eight-stack-baseline.cfg`，再加载 `base.cfg` 与对应文件，用于在引擎协议上逐一覆盖七种组织。

| 配置文件 | 主映射 / 驻留组织 | 实际读写和回收路径 |
| --- | --- | --- |
| `resident.cfg` | `page` + `full-resident` | 常驻 L2P；异地更新；页级 GC；按脏映射页 checkpoint |
| `page-cache.cfg` | `page` + `cached` + `page` | 常驻 translation directory；整映射页 LRU；缺页读、脏逐出和 checkpoint |
| `entry-cache.cfg` | `page` + `cached` + `entry` | 有限条目缓存；后备映射页读取；脏条目按后备页合并；GC 与恢复 |
| `block.cfg` | `hbf-mapping-organization=block` | LBN→数据块，块内固定偏移；单页覆盖执行块 COW；完整块请求直接替换；洞显式编程为 padding |
| `block-log.cfg` | `hbf-mapping-organization=block-log` | 每个活跃逻辑块至多一个专属日志块；全局日志数量有界；读先查更新；LRU 逐出日志，执行 switch / partial / full merge |
| `extent.cfg` | `hbf-mapping-organization=extent` | 常驻、有序的权威 extent 索引；更新拆段，相邻连续段合并；追加分配与有效页最少的 segment cleaning |
| `object-segment.cfg` | `hbf-mapping-organization=object-segment` | 对象 ID / 逻辑范围目录与 extent 索引；共享 segment 追加；CREATE / SEAL / DELETE；部分失效后搬移仍存活的数据 |

主实现是 `src/host/hbf_mapping_policy.cpp`；前三种保留 `src/host/hbf_controller.cpp` 中成熟的页 FTL。`MappingCacheLayout::Extent` 仍用于独立的“压缩映射页缓存”研究，它的后备表是 page-L2P；新 `MappingOrganization::Extent` 的权威映射本身就是 extent。两个配置研究不同问题，不能互换。

块日志采用 **BAST 风格的专属日志**，没有声称实现 FAST 的共享日志策略。`hbf-mapping-log-blocks` 是整个设备的活跃日志块上限。完整有序日志用 switch merge，无有效页复制；有序前缀用 partial merge 补齐旧尾部；随机日志用 full merge 重建数据块。纯块映射仅在调用方一次提交完整块时直接替换；多个独立单页请求不会被偷偷合成整块请求。

对象模式允许多个对象共享物理段。SEAL 刷出该对象的数据并禁止继续覆盖，DELETE 丢弃该对象的缓冲和映射，完全失效的块立即释放；与其他对象混合的块保留到 cleaning。对象目录并不自动提供按寿命隔离的物理池。没有注册对象的逻辑范围不能读写。

## 粒度、写回与恢复

- 页、extent 和日志更新的最小定位单位为 4 KiB；纯块数据基址以 NAND block 为单位，块内用偏移定位。Host 接受 64 B 等部分写，必要时读取旧页完成 RMW，最终提交完整 NAND 页。
- `hbf-write-coalescing=true` 开启每栈有界的数据页缓存；`hbf-write-buffer-pages` 配容量，`hbf-write-buffer-flush-threshold-pages` 配逐出阈值。容量压力按该栈 LRU 逐出；显式 checkpoint 刷出剩余页。
- `hbf-write-buffer-completion-requires-flush=false` 可在数据进入 HBM 后返回，阈值写回仍计入设备完成时间；设为 `true` 则等待数据 NAND program。两者都需要 checkpoint 才能导出包含新映射的可恢复镜像。
- 开启 HBF 设备 DRAM（`hbf-device-dram-capacity-denominator`，见 [host-hbf-management.md](host-hbf-management.md#hbf-device-dram)）后，写缓冲不再占 HBM：结构主映射的 HBM 预算只含索引和每栈工作区。缓冲页按 LPN 所在 stack 暂存在该 stack 的设备 DRAM 中，写入在进入设备 DRAM 时确认；刷写时若目标块在同一 stack，则直接从设备 DRAM program，不再经过 HBIO，否则经 HBIO 搬到目标 stack。读取与 Page FTL 共用同一个按物理页索引的设备 DRAM 读缓存，刷写后的页保留为干净缓存行。
- 新主映射使用两个预留的 NAND checkpoint 槽，写 body 后写 commit root，交替重用。按实际固定宽度记录数收取 metadata programs；恢复读取当前槽、核对校验和、布局、所有权和有效页。
- 持久化镜像格式为 **7**。这是显式 quiescent checkpoint / restart；没有未完成 checkpoint 的掉电恢复、逐笔 WAL 或每次写入的持久性保证。旧格式不再接受。
- 新组织会明确拒绝异步 CRASH 注入。数据块在空闲块中选择较低磨损者；两个 metadata 槽固定保留，反复 checkpoint 的集中磨损会反映在 P/E 中，没有隐藏的 metadata wear leveling。
- 释放块只更新 Host 分配状态；下次 page-0 program 触发真实 erase。所有 live-page copies 均实际执行 NAND→HBIO→HBM→HBIO→NAND，没有免费 copyback。

结构主映射的 Host 按到达顺序执行命令：写入侧的查找与索引更新、分配、元数据 HBM 流量和载荷搬运都排在同一条 Host 时间线上（读取侧的查找见下一段）；它发出的 NAND 操作则在各自的 plane 上执行，不必等完成。program 发出后 Host 立即继续，每个 channel 在途的 program 页数受 `hbf-outstanding-write-pages-per-channel` 限制，超过时 Host 等待最早的一页完成。后续命令只在真正依赖这些操作时等待：读某页会等它的 program 完成（native 媒体就绪时间）；复制、合并、RMW 和恢复读需要数据才能继续，因此 Host 等到读完成；program 仍在途时被覆盖或 TRIM 的旧版本，在该 program 完成时才作废；回收一个块要等它收到的所有 program 完成；checkpoint（排空）等待之前发出的全部操作完成后才提交。不经缓冲或要求刷写完成的写入，在数据 program 完成后确认；经缓冲的写入在进入 HBM 缓冲时确认。

只读逻辑范围分成前景页事务，使用与 Page 路径相同的 `hbf-page-read-queue-depth-per-stack` 有限 credits；credit 从查找前保留到该页完整响应结束，按实际物理页所属 stack 计费，HBM 缓冲与未映射页也受队列约束。一次查找解析一个索引条目（逻辑块或 extent run）在该范围内的全部页，与块映射或 extent FTL 的做法一致；同一条目的后续页只在块带有日志时查询日志目录。读取查找与 Page 路径使用同样的 Host 资源：一组共享的映射计算 worker（数量与 Page 路径相同）和每个 stack 一个控制器内存发起槽。一次查找的各级探测是相互依赖的往返，但不同查找可以重叠，因此 Host 时间线只前进到查找发起为止，该条目的页则等到自己的地址翻译完成才发起 NAND 读。这样比较的是索引组织本身，而不是把结构映射放在单线程 Host 上、把 Page 映射放在多 worker 上造成的差别。完成查找的独立页可重叠；同 bank 的 NAND sense、共享 channel/TSV/ECC/SRAM/HBIO 与热功率预算仍由 native 调度器限制。读缓存仍在完整页解码后可用，部分读的 HBIO 只传输覆盖请求的 64 B 单元。

## 超级块条带化

`hbf-mapping-superblock-planes=N` 让结构主映射以超级块为单位：在 N 条相邻分配 lane（先跨 stack，再跨 stack 内的 plane）上各取同一偏移的一个物理块组成超级块，超级块第 o 页位于第 o mod N 条 lane 的第 o / N 页。块映射、日志、extent segment、分配、按最磨损成员的磨损均衡、合并、cleaning、checkpoint 与恢复都以超级块为单位，所以连续页的 program 和读取分布到 N 个 plane 上并行执行。N 必须整除 plane 总数；`hbf-mapping-superblock-planes=all` 在解析系统配置时取整个 HBF 的 plane 总数（`hbf-stacks` × `hbf-channels` × `hbf-dies-per-channel` × `hbf-planes-per-die`），`mapping/extent-striped` 与 `mapping/block-log-striped` 两个 overlay 就这样写；Page FTL 自己按页条带化，要求 N=1。N=1 时每个逻辑块的页都在同一个 plane 上，分配只在逻辑块之间轮换 lane。

条带化以更大的更新与回收单位换取并行度：逻辑块不按块内顺序追加的更新、合并和 cleaning 都按超级块计。连续追加的数据（例如 extent 的追加 segment、按块内顺序写入的块）同时得到并行度和原有的写放大；块内随机更新的块映射则按超级块放大复制。检查点头部记录超级块几何，宽度不同的镜像不能相互恢复。

查找、树遍历、分配、GC 选择和 HBM 流量都有时间账本，但不是校准过的 CPU 指令模拟。Page 映射按页条带化；结构映射在 N=1 时按连续物理块分配，因此单个请求的物理并行度不同：比较索引成本应使用单 bank 控制，或读取完全相同 PPN 的 raw 媒体控制；多 stack 结果需要另外报告布局与命中的 bank，不能把 Page 的条带收益直接归因于查找算法。

## 内存和元数据成本

内存报告是明确的模拟记录格式，不是 C++ `std::map` / `vector` 的进程堆占用：

| 新组织的记录 | Host 计费 |
| --- | ---: |
| 数据块 / 分配状态与物理有效位图 | 32 B / 物理块 + 1 bit / 物理页 |
| 活跃逻辑块表头 | 32 B + 块内有效位图 |
| 日志更新条目 | 16 B / 活跃 offset→PPN |
| extent 树节点 | 48 B / run；包含 24 B 映射记录及树链接 |
| 对象目录与范围索引 | 96 B / 对象 |
| 数据缓存 | 每页 4 KiB + 32 B 状态 |
| 每栈复制 / 编码工作区 | 8 KiB + 每（超级）块页数 × 16 B |

extent 与对象在 checkpoint 上分别编码为 24 B / run、32 B / 对象；内存树链接无需写入媒体。新主映射默认预留最坏情况预算，允许显式 `hbf-ctrl-dram-bytes` 限制预算；碎片增长超出预算时实验报错，不暗中回退到无限页表。

cleaner 不假定 Host 可以免费读取 NAND OOB 中的反向地址。它扫描权威 extent 索引重建 victim 中的逻辑地址，收取 HBM 扫描流量和计算工作，计入 `reverse_scan_records`。模拟器的 generation 标记仅验证复制与恢复没有丢失最新内容，不参与主索引查找。

`mapping_index_bytes` 可直接比较主映射与缓存结构，`block_state_bytes` 单列共同辅助状态；`metadata_bytes` 是二者之和。`controller_reserved_bytes` 是实际配置的 HBM 预留。页 FTL 的预算包含映射、缓存、scratch、复制槽；结构化主映射的预算包含索引、写缓冲（每页 4096 B，与页 FTL 相同）和合并/清理工作区。两者都不把公共块状态计入映射预算，块状态在预算之外单独预留并单独报告，因此同一预算分母下各组织比较的是映射专有状态。结构化索引常驻、没有缺失路径，所以扫描预算只会得到平线；比较时应同时查看预留和实际使用量。

新组织内部使用 raw media scheduler，原有底层 `data_programs` 会包含 Host 发出的复制和 checkpoint。跨算法归因请使用统一 `hbf_mapping_snapshot()`：`data_programs + copy_programs + padding_programs + metadata_programs == page_programs`。保留底层物理总量用于流量与 P/E 核算。

内部 raw program 不属于额外的 Host 请求，也不计入 `raw_physical_program_payload_bytes`；结构映射的 WAF 分母只使用调用方的 `logical_write_bytes`。复制、padding 和 checkpoint 仅进入物理写入分子。2026-09-22 之前的结构映射构建曾重复计入该 raw 分母，通用 WAF/host-write 报告需由原始 logical/physical 计数重算；统一 mapping snapshot 的物理总量与按 logical 写量计算的 WAF 不受影响。

Session 结束回执的 `hbf_mapping_observations.foreground/post_drain` 在原 terminal drain 前后捕获同一种快照。Serving 结果另给出 `foreground`、`terminal_drain`、`total` 三个窗口的写入分解，各自与对应物理总量对账。Page 的 GC 和 static-WL copy 分开报告；Block 使用 `block_replacements`，BlockLog 使用三种 merge，Extent/ObjectSegment 使用 `cleaned_segments`。结构组织的 Page 专用 GC 字段为 `null`；底层通用 `gc_runs=0` 不表示没有结构 cleaning。不能把 foreground 分项配上 post-drain 物理总量。

## 可直接运行

通过参考运行器比较各组织（同一 trace、同一 HBM 映射预算）：

```bash
python3 -m hbfsim sweep --overlay cached-l2p-1-over-1000 \
  --vary overlay=mapping/page-cache,mapping/entry-cache,mapping/compressed-cache,mapping/block,mapping/block-log-striped,mapping/extent-striped \
  --scenarios all-hbf --demo-tokens 16 --demo-layers 8 --jobs 4 \
  --metrics makespan_us,mean_latency_us,p95_latency_us,hbf_waf,hbf_block_erases
```

直接驱动引擎协议时，`tests/python/mapping_organizations.py` 的 `run()` 用同一随机种子和访问序列依次执行完整块写、单页顺序覆盖、随机覆盖、64 B 片段、读取、checkpoint、重启、失效与替换，对七种组织各输出配置、镜像、恢复结果与 wear 报告，每阶段保留写回前和 checkpoint 后的计数；`mapping_observations()` 校验 foreground / terminal-drain 的写入分解。

```python
session.hbf_object_command("CREATE", "create", object_id=7, first_lpn=0, page_count=256)
# 使用普通 HBF_LOGICAL transactions 读写此范围。
session.hbf_object_command("SEAL", "seal", object_id=7)
session.checkpoint_image("save", output / "state.image")
stats = session.hbf_mapping_snapshot("mapping-cost")
session.hbf_object_command("DELETE", "expire", object_id=7)
```

C++ 测试 `hbf_mapping_organization_test` 用独立 generation oracle 验证持续更新、trim、cleaning、三种日志合并、跨栈缓冲、完整块替换和恢复。Python 测试 `test_hbf_mapping_organizations.py` 覆盖七种配置、同轨迹执行、初始镜像、对象生命周期和 checkpoint/restart。已有页缓存、条目缓存与压缩缓存测试继续检查映射缺页、脏逐出及 GC。
