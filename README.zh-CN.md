# HBFSim（中文说明）

> Status: Current
> Last reviewed: 2026-09-29

**面向 AI 的高带宽闪存（HBF）存储系统模拟器与实验平台。**

[English README](README.md) · [快速上手](#快速上手) · [示例](examples/README.md) ·
[上手指南（英文）](docs/getting-started.md) · [文档索引](docs/README.md)

HBF 像 HBM 堆叠 DRAM 一样堆叠 NAND 闪存：它紧挨加速器，带宽接近 HBM，容量是
HBM 的数倍；但它的读取以微秒计、按页写入、按块擦除，并且会磨损。它是否值得用，
取决于数据放在哪里、闪存如何管理、以及工作负载本身。HBFSim 用来回答这些问题。

HBFSim 把 HBM4、遵循 OCP HBF v0.7.0 的 HBF 及主机侧闪存转换层（FTL），
二者之间的链路，以及外部后备存储（主机 DRAM、CXL、LPDDR、NVMe、CXL-SSD）
建模为**同一个因果事件驱动系统**：时序、容量、放置、数据搬移、映射、垃圾回收、
写放大与磨损。每个结果都记录产生它的源码版本、配置、可执行文件与负载。

## 快速上手

需要 CMake 3.20+、C++20 编译器（已在 GCC 13、Clang 18、Apple Clang 15 上测试）和
Python 3.10+，无需任何 Python 第三方包。

```bash
git clone https://github.com/FlashLab-MBZUAI/HBFSim.git
cd HBFSim
python3 -m hbfsim doctor        # 检查工具链
python3 -m hbfsim quickstart    # 首次自动编译（约一分钟），然后运行第一组对比
```

`quickstart` 在 4 个 HBM 栈 + 4 个 HBF 栈的服务器配置上，用一条类 LLM 访存
trace（权重、KV cache、临时数据）比较五种数据放置策略，并逐列解释输出含义。
也可以在 dev container / GitHub Codespaces 中打开仓库，容器创建时会自动编译。

## 可以探索的问题

| 问题 | 从这里开始 |
| --- | --- |
| HBF 离 HBM 有多远？哪种混合放置能缩小差距？ | `python3 -m hbfsim quickstart`、[示例 1](examples/01_first_comparison.py) |
| 读带宽受限于 HBF 接口速度等级还是 NAND 阵列？ | [示例 3](examples/03_read_bandwidth_ceiling.py) |
| 不同 FTL 映射策略在延迟、写放大和磨损上的代价？ | [示例 2](examples/02_ftl_mapping_tradeoffs.py) |
| 磨损均衡能换来多少寿命，代价是什么？ | [设计空间配方](docs/guides/design-space.md#wear-leveling-versus-write-amplification) |
| 作为 HBM 之后的容量层，HBF 与 DRAM、CXL、SSD 相比如何？ | [示例 6](examples/06_capacity_tier_media.py) |
| 我自己的缓存/迁移策略有没有用？ | [示例 5](examples/05_custom_policy.py)：约 60 行 Python 实现一个策略 |
| 在我自己的负载上表现如何？ | [示例 4](examples/04_custom_trace.py)，或通过 [ServeLoop](docs/guides/serveloop.md) 模拟 LLM 推理服务 |

每个参数都是一个配置项，因此探索通常只需一条命令：

```bash
python3 -m hbfsim list systems            # 硬件配置、overlay、场景与指标
python3 -m hbfsim run --system server-hbm128-hbf1024 --overlay ocp-v070-grade1 --set hbf-read-ns=8000
python3 -m hbfsim sweep --vary hbf-read-ns=2000,4000,8000 --vary overlay=ocp-v070-grade1,ocp-v070-grade2
python3 -m hbfsim show out/run-*/         # 并排比较之前的运行结果
```

每次运行都写入独立的 `out/` 子目录，其中 `command.txt` 记录了底层模拟器的完整
命令，因此任何结果都可以脱离 Python 复现。

## 工作方式

- `build/hbfsim`：语义无关的引擎，执行事务 DAG（目标设备、地址、大小、发起时间、
  依赖），返回每个事务的物理完成时间。放置、缓存、迁移、预取策略都在引擎之外，
  新想法不需要修改物理模型。
- `build/hbfsim-reference`：用八种参考放置策略回放地址 trace（`all-hbm`、`all-hbf`、
  `flat`、`direct-read`、`hbf-streaming`、`external-streaming`、`demand-fill`、
  `reuse-filtered`）。
- `python3 -m hbfsim`：面向使用者的入口，负责编译、按名称查找配置、运行、扫描参数、
  汇总结果；`hbfsim.open_session` 用于编写自己的 Python 策略。

概念、术语表和完整流程见 [Concepts](docs/concepts.md)（英文）。

## 可信的结果

- 每个结果都带有来源：源码提交与是否有未提交修改、完整配置、可执行文件摘要、trace 摘要。
- 每个物理默认值都在 [参数来源登记](configs/parameter-provenance.json) 中标注证据等级。
- 非法输入直接拒绝，而不是给出看似合理的数字；需要谨慎解读的结果会给出警告。
- 契约、独立 oracle、性质模糊测试、解析边界和变异测试都以 CTest 门禁的形式运行。

HBFSim 是事务级研究模拟器，不是厂商产品模型，也不能替代硬件校准。详见
[模型参考](docs/reference/model.md) 与 [证据策略](docs/reference/evidence-policy.md)。

## 参与贡献

欢迎各种规模的贡献：新的配置 overlay、示例、负载导入器、策略、文档修正，或分享你
做过的研究。请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 与
[扩展指南](docs/guides/extending.md)，并遵守 [行为准则](CODE_OF_CONDUCT.md)。
提交 issue 时可以选择 “Share a study, result, or question” 模板分享你的发现。

如果 HBFSim 对你的研究有帮助，请按 [CITATION.cff](CITATION.cff) 引用。

## 许可证

HBFSim 以 [MIT License](LICENSE) 发布。
