# 配方：4 × 96 GB，PCIe 互联

适用硬件：4 张 96 GB 显存卡（RTX PRO 6000 Blackwell 类），无 NVLink，通过 PCIe 5.0 x16 互联。

## 适用边界

| 能用 | 不能用 |
|---|---|
| 总显存 384 GB 的单机部署 | 需要跨节点扩展的场景 |
| 中小规模 MoE（量化后） | 未量化的超大稠密模型 |
| 上下文 32K–128K 的常规服务 | 需要 1M 上下文的超长场景（见 [06 KV 缓存](../docs/06-kv-cache.md)） |

## 拓扑前提

**先跑这一步，再决定参数：**

```bash
nvidia-smi topo -m
```

判断规则（见 [02 并行策略](../docs/02-parallelism.md)）：

| 观察结果 | TP 怎么设 |
|---|---|
| 四张卡两两为 `PIX` / `PXB`，且同属一个 PCIe Switch 组 | TP=4 |
| 四张卡分属两个 Switch | TP=2 × PP=2 |
| 出现 `SYS`（跨 CPU Socket） | 按 Switch 边界切，不要让 TP 跨过去 |

## 启动配置

### 情况 A：四卡同组（TP=4）

```bash
vllm serve <your-model> \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --host 0.0.0.0 --port 8000
```

| 参数 | 为什么这么设 |
|---|---|
| `--tensor-parallel-size 4` | 用满四卡，全在快链路上 |
| `--gpu-memory-utilization 0.90` | 留出工作区；**不要一上手就 0.95**，突发峰值容易失败 |
| `--max-model-len 32768` | 保守起点，先跑通再按 6.1 的公式往上调 |
| `--max-num-seqs 32` | 中等并发起点 |
| `--kv-cache-dtype fp8` | 代价最低的显存节省手段 |
| `--enable-prefix-caching` | 有重复前缀时收益显著 |
| `--enable-chunked-prefill` | 长 prompt 时不阻塞解码 |

### 情况 B：分属两个 Switch（TP=2 × PP=2）

```bash
vllm serve <your-model> \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --host 0.0.0.0 --port 8000
```

不需要额外参数——让 PP 承担跨 Switch 的通信，它只在层边界传激活值，代价远小于 TP 的逐层 all-reduce。

## 已知限制与待验证项

| 项 | 状态 |
|---|---|
| 上述参数是结构合理的起点 | **可推导** |
| 本配方在你的具体硬件上的最优值 | **待测** |
| 是否需要调整 NCCL 环境变量 | **待测**（见 [03 通信优化](../docs/03-communication.md)） |
| 推测解码能否带来收益 | **待测**（从 draft=3 开始，见 [05 推测解码](../docs/05-speculative.md)） |

**特别注意**：如果你的卡是低功耗版本（Max-Q 类，约 300 W），功耗墙可能先于显存成为瓶颈。请在你的测量记录里标明功耗上限与散热条件。

## 测完之后

请按 [METHODOLOGY.md](../benchmarks/METHODOLOGY.md) 记录，并把结果提交到 [benchmarks/](../benchmarks/)。配方库的价值来自真实数据，不是猜测。
