**简体中文** ｜ [English](README.md)

# 本地大模型部署调优指南

**无 NVLink 多卡场景的配置决策库。**

不是原理科普，是决策依据 —— 每条结论都标注它的来源和适用条件。

---

## 为什么需要这个项目

国内绝大多数本地推理部署跑在**没有 NVLink 的机器**上：RTX 4090 / 5090、RTX PRO 6000、各代 PCIe 版计算卡，通过 PCIe 5.0 x16 或 PCIe Switch 互联。

但主流推理框架的文档、调优建议和默认配置，几乎都建立在"卡间有高速互联"这个假设上。结果就是：

- 同样 4 张卡，社区报告里吞吐能差 2 倍以上（见[实测数据](benchmarks/zh-CN/README.md)），差别全在配置，不在硅片
- 中文资料里"讲原理"的文章很多，但**参数建议互相矛盾，且几乎都不说明测试条件**
- 你无法判断一条建议到底适不适用于你的机器

这个项目要补的是另一半：**把结论和它的适用条件绑在一起。**

## 适合谁

- 你在一台机器上跑 2–8 张卡，走 PCIe，没有 NVLink。
- 你正在选 TP / PP / EP / 副本、量化或 KV 配置，想要一个带依据的起点。
- 你手上有一个跑分数字，想知道它能不能迁移到你的机器。

**不适合**：多机 InfiniBand/NVSwitch 集群、训练、以及“哪个模型更聪明”的排行榜。这里讨论的是推理配置，不是模型能力。

## 从这里开始

1. 跑 `python tools/detect_topology.py`，把 JSON 留下来——后面每个决策都以它为输入。
2. 打开[模型矩阵](MODEL_MATRIX.zh-CN.md)，确认模型能不能放下、需要几张卡。
3. 找最接近的[配方](recipes/zh-CN/)，复制参数，然后用 [bench/run_matrix.py](bench/run_matrix.py) 实测，不要直接相信默认值。

## 快速决策表

先回答最高频的六个问题。

| 你的情况 | 建议 | 依据 |
|---|---|---|
| 模型（含 KV）能塞进单卡 | **多副本（replicas）优先于张量并行** | 实测 + 可推导 |
| 模型必须切，且卡在同一 PCIe Switch 下 | **先从 TP = 该 Switch 下的卡数起步，再实测**。8×72 机器上 TP8 的 c8 聚合吞吐比单节点 TP4 高 12%——但用的是 **2 倍卡数**，**每卡效率 TP4 反而高 1.7 倍** | 实测 |
| 卡跨 PCIe Switch 或跨 CPU Socket | **默认不要让 TP 跨过去**，改用 TP×PP 或副本。同一台机器也测到过例外（TP8 跨 SYS 仍赢 TP4），所以这是默认值，不是铁律，要实测 | 可推导 + 实测 |
| 想要更长上下文 | 先降 KV 精度 / 加卡，DCP 在 PCIe 上收益有限 | 实测 + 可推导 |
| 想用推测解码提速 | **draft 长度不是越大越好**，从 2–3 开始测。公开的 19–22% 差距是 MTP3 与 DFlash2-K7 的对比，两者草稿方法也不同 | 实测（有混杂变量） |
| 量化掉点明显 | **先把激活精度提上去**——W4A4 → W4A16 让 checksum 失败率从 5.9% 降到 0；FP32 router 是免费的二次保险 | 实测 |

## 目录

| 章节 | 内容 | 一句话 |
|---|---|---|
| [01 硬件与拓扑](docs/zh-CN/01-hardware.md) | PCIe 与 NVLink 的真实差距 | 先搞清楚你的卡怎么连的 |
| [02 并行策略](docs/zh-CN/02-parallelism.md) | TP / PP / DP / EP / DCP 怎么选 | 加卡什么时候反而更慢 |
| [03 通信优化](docs/zh-CN/03-communication.md) | all-reduce 模式与 NCCL 调优 | PCIe 上为什么该换 one-shot |
| [04 量化选型](docs/zh-CN/04-quantization.md) | FP8 / MXFP8 / NVFP4 / AWQ 的取舍 | 省显存的代价在哪里 |
| [05 推测解码](docs/zh-CN/05-speculative.md) | MTP / DFlash / EAGLE 系 | 猜得越多为什么反而越慢 |
| [06 KV 缓存](docs/zh-CN/06-kv-cache.md) | 显存预算、前缀缓存、长上下文 | 上下文是怎么吃掉显存的 |
| [07 排错手册](docs/zh-CN/07-troubleshooting.md) | 症状 → 原因 → 处理 | OOM、输出异常、SM120 内核缺口、容器卡住 |
| [08 功耗与成本](docs/zh-CN/08-power-cost.md) | 每瓦与每块钱的 token 数 | 300 W 卡为什么停止扩展 |
| [09 失败的配置](docs/zh-CN/09-failures.md) | 跑不通、或跑通但结论错的配置 | 决策库的反面 |
| [模型矩阵](MODEL_MATRIX.zh-CN.md) | 72 GB 卡上什么能跑 | 权重、KV/token、最少卡数、已验证配置 |
| [配置配方](recipes/zh-CN/) | 按硬件型号的现成配置 | [4×72](recipes/zh-CN/4x-72gb-pcie.md) · [8×72](recipes/zh-CN/8x-72gb-pcie.md) · [4×96](recipes/zh-CN/4x-96gb-pcie.md) · [8×24](recipes/zh-CN/8x-24gb-pcie.md) |
| [实测数据](benchmarks/zh-CN/README.md) | 数据表与测量方法 | 你自己也能填表 |
| [规划工具](tools/) | 输入硬件 → 输出并行策略 | 命令行 + 离线网页 |
| [术语表](docs/zh-CN/GLOSSARY.md) | 术语与缩写 | TP / PP / EP / MLA / NVFP4 / SM120 |
| [文章](articles/) | 长文记录 | 英文摘要 + 中文全文 |

## 数据可信度分级

本项目所有结论都带依据标记：

| 标记 | 含义 |
|---|---|
| **实测** | 由真实硬件上的运行产出，记录硬件、软件版本与方差；新提交还必须记录时钟策略、镜像 digest 与完整的服务端/客户端命令（见 [SCHEMA](benchmarks/zh-CN/SCHEMA.md)）。2026-11 之前提交的条目缺后三项，已在文件中标注——它们确实跑过，但尚不可完全复现 |
| **可推导** | 由硬件规格或框架机制直接推导，不依赖具体某次测量；表里的推导值会注明公式（如 [MODEL_MATRIX](MODEL_MATRIX.zh-CN.md) 的 KV/token 推导表） |
| **待测** | 方向明确但缺公开数据；本项目给出测量方法，等你用自己机器填 |

**我们不做的事：不给没有依据的具体数字。** 空格比编造填得更满有价值。

## 核心结论速览

以下六条是当前最有把握的判断，展开见各章节。

1. **能塞进单卡，就别做张量并行。** 两个独立副本之间零通信，吞吐直接线性叠加；而 TP 每层都要 all-reduce，在 PCIe 上这个代价会吃掉大部分收益。
2. **TP 的边界是 PCIe Switch，不是显卡数量。** 跨 Switch 的通信要走 CPU 内存，延迟骤增。四张卡插在两个 Switch 上时，TP=4 往往不如 TP=2 × PP=2，甚至不如两个 TP=2 副本。
3. **PCIe 拓扑下应优先 one-shot all-reduce。** 帧同步式的 two-shot 在高速互联上更优，但它的优势建立在带宽充裕的前提上——PCIe 恰好相反。
4. **推测解码的收益受带宽约束。** draft 越长，单次验证失败浪费的计算越多，而草稿模型本身还要占用带宽。公开对比（MTP3 vs DFlash2-K7）里短 draft 领先 19–22%，但两者的草稿方法和权重也不同——应当读作“两种方案都要实测”，而不是“7 一定比 3 差”。
5. **先提激活精度，再谈路由层。** MoE 模型里，W4A4 → W4A16 一项就把 greedy-checksum 失败率从 5.9% 降到 0。FP32 router 权重不带来可测量的吞吐代价，且在 pre-QAD checkpoint 上是 19.5% → 4.0% 的差别，但它替代不了 A16 激活。
6. **同一台机器，配置差异可以造成 2 倍以上的吞吐差距。** 这不是玄学，是每一个参数都在和你的实际拓扑对齐。

## 怎么用这个项目

**如果你要部署一台新机器**：从 [01 硬件与拓扑](docs/zh-CN/01-hardware.md) 开始，先跑一次 `nvidia-smi topo -m`，再去 [配置配方](recipes/zh-CN/) 找最接近你硬件的配置。

**如果你已经有机器但吞吐不理想**：跑一下规划工具，对比它给出的参数和你现在的差异。

```bash
python tools/planner.py --gpus A800-80 --count 8 --model qwen2.5-72b --ctx 8192
```

它会输出互联带宽上限、显存预算、以及 TP / PP / 副本数的推荐组合，每一条都标注依据等级。详见 [tools/README.md](tools/README.md)。

**如果你想贡献数据**：见 [CONTRIBUTING.md](CONTRIBUTING.zh-CN.md)。数据表格已经做好，缺的就是真实机器的测量值。

## 相关项目

- [llm-inference-bench](https://github.com/Lowaltitudeflight/llm-inference-bench) —— 配套的压测与评测平台，负责产出本项目需要的数据
- [llm-private-deploy-handbook](https://github.com/Lowaltitudeflight/llm-private-deploy-handbook) —— 私有化交付流程手册

## 许可

[MIT](LICENSE)
