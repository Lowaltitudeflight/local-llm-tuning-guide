# 01 硬件与拓扑

> 动手调参之前，先搞清楚你的卡是怎么连的。这一步做错，后面所有参数都是错的。

## 1.1 差距有多大

| 互联方式 | 典型双向带宽 | 出现场景 |
|---|---|---|
| PCIe 4.0 x16 | ~64 GB/s | 上代平台、消费级主板 |
| PCIe 5.0 x16 | ~128 GB/s | 当前主流工作站与服务器 |
| NVLink 3 (A100) | ~600 GB/s | A100 SXM |
| NVLink 4 (H100) | ~900 GB/s | H100 SXM |
| NVLink 5 (B200) | ~1800 GB/s | B200 / GB200 |

两者差 **7 到 14 倍**。这是为什么所有"照抄 NVLink 集群配置"的做法在没有 NVLink 的机器上都会失效。

**依据：可推导**（均为厂商公开规格）

## 1.2 关键：PCIe 也不是均质的

同样是四张 PCIe 卡，拓扑可能完全不同。带宽从高到低：

| `nvidia-smi topo -m` 标记 | 含义 | 相对速度 |
|---|---|---|
| `NV#` | NVLink，数字是链路数 | 最快 |
| `PIX` | 经过**单个** PCIe Switch | 快 |
| `PXB` | 经过**多个** PCIe Switch | 中 |
| `PHB` | 经过 PCIe Host Bridge（CPU 引出） | 慢 |
| `SYS` | 经过 CPU 间 SMP 互连（跨 NUMA） | 最慢 |
| `NODE` | 同一 NUMA 节点内 | — |

**对推理的意义**：TP 的每一步 all-reduce 都会走这条路径。如果卡对之间是 `SYS`，那部分通信要先出 CPU、过 QPI/UPI、再进另一颗 CPU——延迟可能是 `PIX` 的若干倍。

**依据：可推导**

## 1.3 先跑这三条命令

```bash
# 1. GPU 之间的连接关系 —— 最重要的一条
nvidia-smi topo -m

# 2. 每张卡挂在哪个 PCIe 插槽、什么代数、什么宽度
nvidia-smi -q | grep -A2 "GPU Link Info"

# 3. NUMA 布局（几颗 CPU、各自管辖哪些卡）
lscpu | grep -i numa
numactl --hardware
```

`nvidia-smi topo -m` 的输出是一张矩阵，交叉点就是两张卡之间的连接类型：

```
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      PIX     PXB     SYS     0-31            0
GPU1    PIX      X      PXB     SYS     0-31            0
GPU2    PXB     PXB      X      SYS     0-31            0
GPU3    SYS     SYS     SYS      X      32-63           1
```

这个例子里：GPU0/1/2 在同一组 PCIe Switch 下（`PIX`/`PXB`），而 GPU3 在另一颗 CPU 上（`SYS`）。**结论直接就有了：TP=4 会跨越最慢的那条链路，应该考虑 TP=3（走 Switch 内）或 2×2 的组合。**

## 1.4 拓扑决定了什么

| 观察到的情况 | 配置含义 |
|---|---|
| 全部 `NV#` | 正常 NVLink 集群，可按常规建议配置 |
| 全部 `PIX` | 四卡同 Switch，TP=4 可行，但需关注 all-reduce 模式 |
| 混有 `PXB` / `PHB` | 谨慎放大 TP，考虑 PP 拆分 |
| 出现 `SYS` | **TP 不要跨越 SYS 边界**，那是性能悬崖 |

## 1.5 测量模板

本项目不给出未经核实的带宽数字。请用下面这份模板测量并提交（见 [CONTRIBUTING.md](../../CONTRIBUTING.md)）：

```markdown
### 硬件
- GPU 型号 / 数量：
- 显卡互联（topo -m 摘要）：
- CPU / NUMA 域：
- 主板 / PCIe Switch：
- PCIe 代数与宽度：

### 测量
| 卡对 | topo 标记 | all-reduce 32MB 耗时 | 实测带宽 |
|---|---|---|---|
| 0-1 |  |  |  |
| 0-3 |  |  |  |

### 工具与版本
- NCCL 版本：
- CUDA 版本：
- 测量命令：
```

推荐测量工具：`nccl-tests`（`all_reduce_perf`）、`nvbandwidth`。

**依据：待测**

---

下一章：[02 并行策略](02-parallelism.md)
