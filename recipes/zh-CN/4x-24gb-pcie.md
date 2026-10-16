# 配方：4 × 24 GB，PCIe 互联

适用硬件：4 张 24 GB 显存的消费级卡（RTX 4090 / 3090 类），无 NVLink，PCIe 4.0 x16 互联。这是消费级无 NVLink 平台最常见的配置。

## 这一类的特殊之处

与 [8x-24gb-pcie](8x-24gb-pcie.md) 相同：消费级 Ada 及之后的显卡 **P2P 访问在驱动层被限制**，NCCL 只能绕道主机内存，TP 在这类硬件上效率极低。**能不做 TP 就不做。**

4 卡相比 8 卡多了一个现实约束：卡少，副本数也少。三个方案的实际取舍：

| 方案 | 前提 | 适用 |
|---|---|---|
| 4 个独立副本 | 模型（+KV）能放进单卡 | 小模型、高并发吞吐场景 |
| 2 个副本 × TP=2 | 模型能放进双卡 | 30B 上下的量化模型，**最常见** |
| TP=4 | 仅当必须四卡才装得下 | 尽量避免，见"避免" |

## 拓扑前提

```bash
nvidia-smi topo -m
```

消费级平台上卡对通常显示 `PHB` 或 `SYS`，P2P 行不可用——这是正常现象，不是故障。若你的平台带 PCIe Switch（`PIX`），TP=2 的收益会比 `PHB` 好一些，但策略不变。

## 启动配置

### 方案 A：模型能放进双卡（推荐，2 副本 × TP=2）

```bash
# 副本 1
CUDA_VISIBLE_DEVICES=0,1 vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --port 8001

# 副本 2
CUDA_VISIBLE_DEVICES=2,3 vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --port 8002
```

前面挂一层负载均衡即可。

| 参数 | 为什么这么设 |
|---|---|
| `--tensor-parallel-size 2` | TP 封顶在 2；通信代价从 3 开始压过收益 |
| `--gpu-memory-utilization 0.90` | 消费卡显存碎片更多，别压太满 |
| `--max-model-len 8192` | 24 GB 卡的 KV 空间有限，从短上下文起步，按 [06 KV 缓存](../../docs/zh-CN/06-kv-cache.md) 的公式上调 |
| `--kv-cache-dtype fp8` | 24 GB 卡上这是代价最低的显存节省手段 |

**依据**：同构硬件上的公开对比数据——4×RTX 3090、AWQ 量化 35B 级 MoE：双实例各 TP=2 达 1437 tok/s，单实例 TP=4 仅 610 tok/s（见 [benchmarks](../../benchmarks/zh-CN/README.md)，`../../benchmarks/data/rtx3090-replicas-vs-tp.json`）。**可迁移的结论**是"副本优于把卡做大并行"；具体数字不可迁移（单次测量、特定模型）。

### 方案 B：模型能放进单卡（4 个独立副本）

```bash
# 副本 1，其余副本改 CUDA_VISIBLE_DEVICES 和 --port
CUDA_VISIBLE_DEVICES=0 vllm serve <your-model> \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --port 8001
```

30B 以下的量化模型在 24 GB 单卡上完全够用。此时四张卡的最佳用法是 4 个零通信副本，吞吐近似线性叠加。

### 避免

| 配置 | 原因 |
|---|---|
| `--tensor-parallel-size 4` | 4 卡全绕道 CPU 内存，公开数据下它只有 TP=2 副本的 4 成吞吐 |
| 高 `--max-num-seqs` + 长上下文组合 | 24 GB 卡容易 OOM；KV 是第一个被压爆的 |
| 期望 4 卡线性加速 | TP 在这类硬件上不成立；线性的是副本 |

## 已知限制与待验证项

| 项 | 状态 |
|---|---|
| P2P 受限导致 TP 低效 | **可推导**（驱动限制 + 公开实测佐证） |
| 双副本 × TP=2 优于 TP=4 | **实测**（4×RTX 3090 公开对比，非同卡型） |
| 你的平台上 TP=2 相对副本的实际收益 | **待测** |
| `--max-num-seqs 64` 在你的模型 + 上下文下是否 OOM | **待测**（从 32 起步往上试） |

## 测完之后

请按 [METHODOLOGY.md](../../benchmarks/zh-CN/METHODOLOGY.md) 记录并提交结果。**尤其是 TP=2 副本 vs TP=4 在 4090（而非 3090）上的直接对比——目前只有 3090 的公开数据，这一格是空的。**
