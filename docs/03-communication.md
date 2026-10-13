# 03 通信优化

> TP 的通信是每一层都要发生的。这一章讲怎么让它在 PCIe 上少花时间。

## 3.1 两种 all-reduce 与它们的适用条件

两阶段式（常称 two-shot）把 all-reduce 拆成 reduce-scatter 加 all-gather 两步：

| | 两阶段（reduce-scatter + all-gather） | 单步（one-shot） |
|---|---|---|
| 通信总量 | 与 rank 数近似无关 | 随 rank 数线性上升 |
| 同步次数 | 两次 | 一次 |
| 优势场景 | **大消息**、带宽受限 | **小消息**、延迟受限 |
| PCIe 上 | 消息大时仍然可用 | 解码阶段通常更优 |

这条取舍的核心是**延迟还是带宽在主导**：

- **Prefill 阶段**：一次处理几千个 token，all-reduce 的消息很大 → 带宽主导 → 两阶段仍有价值
- **Decode 阶段**：每步只处理很少的 token，消息很小但同步极频繁 → 延迟主导 → 单步更优

所以正确的问法不是"哪个算法更好"，而是"我这个阶段的消息有多大"。

**依据：可推导**

## 3.2 真实项目的做法

公开的 PCIe 优化实践（Local Inference Lab 的 GLM-5.3-Flash 配置）中有两点值得参考：

1. **默认关闭两阶段 all-reduce** —— 其配置文档明确写着 "Two-shot all-reduce is off"，目标配置走单步路径。
2. **提供了按消息大小切换的阈值** —— 配置项里存在 `VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE` 这类参数，说明这个选择不是全局一刀切，而是随消息规模变化的。

第 2 点比第 1 点更有启发：**结论不是一个算法，而是一个阈值。**

> 该项目的目标是 4 张 PCIe 卡（无 NVLink）运行 NVFP4 量化的 MoE 模型，TP=4。其配置可作为 4 卡 PCIe 场景的参考起点。

**依据：实测**（来源：local-inference-lab/rtx6kpro 公开配置文件）

## 3.3 NCCL 环境变量

以下变量在 PCIe 机器上最常需要调整。**更改前请先建立基线测量**，否则无法判断改动是否有效。

| 变量 | 作用 | 典型用途 |
|---|---|---|
| `NCCL_P2P_LEVEL` | 允许 P2P 传输的最高拓扑层级（`PIX`/`PXB`/`PHB`/`SYS`） | 跨 NUMA 卡对可限制为 `PIX` 避免走慢链路 |
| `NCCL_P2P_DISABLE` | 完全禁用 GPU 间 P2P | 排查 P2P 不稳定时的诊断手段 |
| `NCCL_ALGO` | 选择集合通信算法（`Ring` / `Tree`） | 卡数少时 `Tree` 可降低延迟 |
| `NCCL_PROTO` | 传输协议（`LL` / `LL128` / `Simple`） | 小消息场景 `LL` 常优，大消息 `Simple` |
| `NCCL_MIN_NCHANNELS` | 通道数下限 | 带宽利用率不足时可上调 |
| `NCCL_BUFFSIZE` | 分片缓冲大小 | 与消息规模匹配 |
| `NCCL_ASYNC_ERROR_HANDLING` | 异步错误处理 | 排查超时时临时关闭，**不要用于生产** |

**通用排查顺序**：

```bash
# 1. 先确认拓扑，再动手（见 01 章）
nvidia-smi topo -m

# 2. 用 nccl-tests 建立基线
all_reduce_perf -b 8 -e 128M -f 2 -g 4

# 3. 一次只改一个变量，重新测量
# 4. 记录：改动 + 前后数值 + 硬件与软件版本
```

**依据：可推导**（NCCL 变量语义以其官方文档为准）

## 3.4 与推测解码的相互作用

推测解码会**放大通信压力**：

- 草稿阶段产生的候选 token 需要在卡间同步验证结果
- draft 越长，单次验证的数据量越大
- 草稿模型本身如果也是 TP 切分，它的 all-reduce 会和主模型的通信竞争同一条 PCIe 链路

这就是为什么在 PCIe 机器上，推测解码的收益比在 NVLink 机器上更小，且 draft 长度需要更保守。展开见 [05 推测解码](05-speculative.md)。

**依据：可推导**

## 3.5 测量模板

```markdown
### 配置
- GPU / 数量：
- 拓扑（topo -m 摘要）：
- 并行配置（TP/PP）：
- NCCL 版本：
- 相关环境变量：

### nccl-tests all_reduce_perf 结果
| 消息大小 | 算法/协议 | 耗时 | 带宽 |
|---|---|---|---|

### 推理侧对照
| 配置 | C1 输出 | C8 聚合 | 备注 |
|---|---|---|---|
```

**依据：待测**

---

上一章：[02 并行策略](02-parallelism.md) ｜ 下一章：[04 量化选型](04-quantization.md)
