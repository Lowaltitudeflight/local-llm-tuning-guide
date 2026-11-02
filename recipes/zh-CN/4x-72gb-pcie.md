# 配方：4 × 72 GB，PCIe 互联

目标硬件：4 张 72 GB 显存的卡（RTX PRO 5000 Blackwell 级别），无 NVLink，合计 288 GB。在 4+4 的双路机器上，这就是一个 NUMA 节点；TP 不要跨出这个节点。

这份配方有一半是“负面结果”：这一档卡上，**很多看起来“应该能装下”的模型其实装不下**——权重加上运行时开销在分配 KV 之前就已经超过单卡预算。

## 哪些能跑，哪些不能

| 模型 | 权重 | 单卡（TP4） | 能否 | 说明 |
|---|---:|---:|---|---|
| Qwen3-8B | ~16 GB | ~4 GB | ✅ | KV 余量充足 |
| Qwen3.8-27B BF16 | ~52 GB | ~13 GB | ✅ | 长上下文和并发都可行 |
| Qwen3.8-27B NVFP4 | ~21 GB | ~5 GB | ✅ | 这台机器上每瓦吞吐最好 |
| Qwen3-32B | ~62 GB | ~16 GB | ✅ | 放得下，还有 KV 余量 |
| Qwen3-235B-A22B NVFP4 | ~134 GB | ~34 GB | ✅ | MoE；支持时用 TP4 + EP4，实测为准 |
| GLM-5.3-Flash FP8 | ~306 GB | ~77 GB | ❌ | 还没算 KV/激活就超过 72 GB |
| MiniMax-M3.1-Flash NVFP4 | ~252 GB | ~63 GB | ❌ | 只剩 8–12 GB 给 KV + 运行时，必 OOM（见 8×72 配方方案 C） |
| DeepSeek-V4.1-Flash-1017 | ~476 GB | ~119 GB | ❌ | 必须 8 卡 |

> 经验法则：72 GB 卡上，想要长上下文或真实并发，**权重最好 ≤ 50 GB**。剩下的 20 多 GB 会被 KV、激活、CUDA 图、框架 workspace 吃掉，速度比大多数人预期快得多。

## 拓扑前提

```bash
nvidia-smi topo -m
```

- 4 张卡在同一个 NUMA 节点（`NODE` 链路）：TP=4 是最简单的快路径。
- 4 张卡跨两个 NUMA 节点：优先 2× TP=2 副本，或 TP=2 × PP=2；不要让 TP 跨 `SYS`。

## 启动配置

### Qwen3-32B / Qwen3.8-27B，TP4，单 NUMA 节点

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve <model> \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --host 0.0.0.0 --port 8000
```

### Qwen3-235B-A22B NVFP4，TP4 + EP4

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve <model> \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000
```

> vLLM 和 SGLang 的 EP 参数名不同；保持 `--tp 4`，加上对应框架的专家并行开关。TP4 与 2× TP=2 副本谁更快，MoE 上并不显然，要实测。

## 什么时候该用 4 卡而不是 8 卡

- 模型放得下（见上表），另外 4 张卡可以跑第二个副本或另一个模型。
- 更看重单流延迟而不是聚合吞吐：TP 组越小，集合通信越少。
- 功耗受限：4 × 300 W 比 8 × 300 W 更容易供电和散热。

## 欢迎负面结果

如果你在 4×72 GB 上把“不能跑”那一列的模型跑起来了，这就是有价值的修正——带上完整启动命令、显存日志和 benchmark JSON 开 issue。见 [CONTRIBUTING.md](../../CONTRIBUTING.zh-CN.md)。
