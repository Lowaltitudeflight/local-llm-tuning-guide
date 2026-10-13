# 配方：8 × 24 GB，PCIe 互联

适用硬件：8 张 24 GB 显存的消费级卡（RTX 4090 / 3090 类），无 NVLink。

## 这一类的特殊之处

消费级 Ada 及之后的显卡，**GPU 间 P2P 访问在驱动层被限制**。后果是：

- NCCL 无法在卡之间直接传输数据，只能**绕道主机内存**（GPU → CPU → GPU）
- 每一次 all-reduce 都要走两趟 PCIe，延迟和带宽双重劣化
- **TP 在这类硬件上效率极低**，加卡带来的收益远小于预期

所以这类机器的配置策略与数据中心卡完全不同：**能不做 TP 就不做。**

## 拓扑前提

```bash
nvidia-smi topo -m
```

在这类硬件上，卡对之间通常显示 `PHB` 或 `SYS`，且 P2P 行会显示为不可用。这是正常现象，不是故障。

## 启动配置

### 首选：多副本（前提是模型能放进单卡或双卡）

```bash
# 每个副本用一个端口、一组卡，各自独立服务
# 副本之间零通信，吞吐近似线性叠加

# 副本 1
CUDA_VISIBLE_DEVICES=0,1 vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8 \
  --port 8001

# 副本 2
CUDA_VISIBLE_DEVICES=2,3 vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8 \
  --port 8002
```

前面挂一层负载均衡即可。**这是这类硬件上性价比最高的方案**——参考 [02 并行策略](../docs/02-parallelism.md) 中的实测对比（4×RTX 3090：双实例 TP=2 达 1437 tok/s，单实例 TP=4 仅 610 tok/s）。

### 次选：必须切分时，TP 尽量小

```bash
vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000
```

| 参数 | 说明 |
|---|---|
| `--tensor-parallel-size 2` | 不超过 2；再大通信代价压过收益 |
| `--pipeline-parallel-size 2` | 用 PP 扩展容量；通信只在段边界 |
| `--gpu-memory-utilization 0.90` | 消费卡的显存碎片更多，别压太满 |
| `--max-model-len 8192` | 24 GB 卡的 KV 空间有限，从短上下文起步 |

### 避免

| 配置 | 原因 |
|---|---|
| `--tensor-parallel-size 8` | 8 卡全 P2P 绕道 CPU，几乎必然是负收益 |
| 高 `--max-num-seqs` + 长上下文 | 24 GB 卡上容易 OOM |
| 期望 8 卡线性加速 | 这类硬件上不成立 |

## 已知限制与待验证项

| 项 | 状态 |
|---|---|
| P2P 受限导致 TP 低效 | **可推导**（驱动限制 + 公开实测佐证） |
| 多副本优于 TP | **实测**（4×RTX 3090 的公开对比数据） |
| 你的具体卡型是否有 P2P | **待测**（跑 `nvidia-smi topo -m` 确认） |
| 8 卡的具体最优副本数 | **待测**（取决于模型大小与显存） |

**一个务实的提醒**：如果模型小到能放进单卡，就不要用多卡。24 GB 卡跑 30B 以下的量化模型完全够用，此时 8 张卡的最佳用法是**8 个独立副本**，而不是一个巨大的并行实例。

## 测完之后

请按 [METHODOLOGY.md](../benchmarks/METHODOLOGY.md) 记录并提交结果。
