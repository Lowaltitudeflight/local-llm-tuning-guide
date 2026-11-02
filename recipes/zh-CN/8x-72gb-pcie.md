# 配方：8 × 72 GB，PCIe 互联，双 NUMA

目标硬件：8 张 72 GB 显存的卡（RTX PRO 5000 Blackwell 级别），无 NVLink，共 576 GB。每颗 CPU 挂 4 张卡，跨路互联为 `SYS`。

本配方起草所依据的环境（第一方快照，2026-10-23）：

| 项目 | 数值 |
|---|---|
| GPU | 8 × NVIDIA RTX PRO 5000 72GB Blackwell |
| 拓扑（`nvidia-smi topo -m`） | GPU0–3 在 NUMA0 内为 `NODE`，GPU4–7 在 NUMA1 内为 `NODE`，跨组为 `SYS` |
| CPU | 2 × Xeon Platinum 8488C（192 线程），GPU0–3 → NUMA0，GPU4–7 → NUMA1 |
| 内存 | 1.5 TB |
| 驱动 / CUDA | 595.71.05 / 13.2 |
| NCCL | 2.26.5（系统自带） |
| 系统 | Ubuntu 22.04.5 LTS |

## 适用边界

| 适用 | 不适用 |
|---|---|
| 单机部署大参数量 MoE 模型（数百 B、量化后） | 未量化的超大规模稠密模型 |
| 576 GB 总显存——大模型 + 长上下文都有余量 | 需要跨节点扩展的场景 |
| NUMA 节点内 TP=4 是快路径 | 天真地 TP=8（每层 all-reduce 都要跨 `SYS`） |

## 拓扑前置检查

**先跑这条命令，再决定参数：**

```bash
nvidia-smi topo -m
```

针对这种拓扑的决策规则（见 [02 并行策略](../../docs/zh-CN/02-parallelism.md)）：

| 观察结果 | 并行方式 |
|---|---|
| 4+4 分属两个 NUMA 节点、跨组 `SYS`（本机如此） | 节点内 TP=4 × 跨节点 PP=2 |
| 8 卡同组（不是本机的情况） | TP=8 |
| 模型装得下 4 × 72 GB | 单 NUMA 节点 TP=4，另一组留空 |

## 启动配置

### 方案 A：TP=4 × PP=2（8 卡全用，跨路走 PP）

```bash
vllm serve <your-model> \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
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
| `--tensor-parallel-size 4` | 让每层的 all-reduce 留在同一个 NUMA 节点内（`NODE` 链路） |
| `--pipeline-parallel-size 2` | 跨路流量只在层边界发生，远比 TP 跨 `SYS` 便宜 |
| `--gpu-memory-utilization 0.90` | 留余量；不要一上来就 0.95 |
| `--kv-cache-dtype fp8` | 这个规模下最省显存的手段 |

SGLang 等价写法（快照机器上就是用 Docker 跑 SGLang）：

```bash
python -m sglang.launch_server \
  --model-path <your-model> \
  --tp 4 --dp 2 \
  --mem-fraction-static 0.90 \
  --context-length 32768 \
  --host 0.0.0.0 --port 30000
```

> 注意：SGLang 的 `--dp 2` 是把模型在两个 NUMA 组之间**复制一份**（数据并行），完全避开跨路 all-reduce，代价是权重显存翻倍。PP=2 和 DP=2 在这种拓扑上谁更强是**待实测问题**——两个都要测。

### 方案 B：模型装得下 4 × 72 GB（TP=4，单 NUMA 节点）

用 `CUDA_VISIBLE_DEVICES=0,1,2,3`（NUMA0）或 `4,5,6,7`（NUMA1）配 TP=4，不开 PP。每 token 路径最简单最快。

### 方案 C：大 NVFP4 MoE（MiniMax-M3.1-Flash，252 GB）——SM120 绕行方案

252 GB 的模型装不进 4 张 72 GB 卡，必须用满 8 张。镜像的 NVFP4 原生路径面向 SM100/SM103，在 SM120（RTX PRO 5000）上需要三个只读补丁加显式参数。2026-11-02 已在本快照机器上验证。

```bash
sglang serve \
  --model-path /models2/MiniMax-M3.1-Flash \
  --trust-remote-code \
  --tp 8 \
  --attention-backend triton \
  --chunked-prefill-size 2048 \
  --mem-fraction-static 0.85 \
  --context-length 1048576 \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode 128 \
  --host 0.0.0.0 --port 8200
```

需要的改动（用只读挂载覆盖镜像内文件，不要改镜像）：

| 文件 | 改动 | 原因 |
|---|---|---|
| `srt/configs/model_config.py` | 在 SM120 上让 `uses_minimax_m3_nvfp4_native_stack` 返回 False | 原生 MSA-v1 decode indexer 要求 SM100/SM103 |
| `srt/arg_groups/model_overrides/minimax_m3.py` | 在 SM120 上走非 Blackwell 分支 | 绕开硬编码到 SM100/103 的 MegaMoE |
| `srt/layers/quantization/minimax_m3_nvfp4.py` | `use_marlin = not is_blackwell or is_sm120` | SM120 上只剩 Marlin 这条 NVFP4 MoE 路径 |

另外：**不要传 `--quantization`**——必须让 SGLang 自动识别成 `minimax_m3_nvfp4`；强制 `mxfp8` 会把打包的 NVFP4 专家按 FP8 加载，报 192 vs 384 的张量尺寸错误。

本机实测（vLLM bench，OpenAI completions，输入 1024 / 输出 256，seed 42，cache-on 3 次均值）：

| 配置 | C1 单流 | C8 聚合 | C32 聚合 | C64 聚合 | ShareGPT C64 |
|---|---:|---:|---:|---:|---:|
| TP8 + Marlin（本方案） | 94.4 tok/s | 489.8 tok/s | 740.4 tok/s | **1060.8 tok/s** | **1093.8 tok/s** |

长上下文与显存：

- `--mem-fraction-static 0.85` 时 KV 池 1,329,280 token；每卡占用 67.7–72.3 GB。
- 1M 上下文下，冷启动 999,988 token 请求 **517.7 秒**完成，无 OOM；1,047,992 token 的边界请求也成功。
- 1M 请求占 KV 池约 79%，同一时间基本只能跑 1 条；1M 配置的 C64 吞吐为 815.7 tok/s，比 128K 低约 23%。
- 完整数据与方法：[`benchmarks/data/minimax-m31-flash-8x72-sm120.json`](../../benchmarks/data/minimax-m31-flash-8x72-sm120.json)。

## 第一方实测数据（2026-10-23，稠密 27B）

在快照机器上用 [sglang 0.5.22](../../benchmarks/data/qwen38-27b-8x-72gb-pcie-parallelism.json) 实测——Qwen3.8-27B BF16，输入 4096 / 输出 512 token，每次冷启动换新容器：

| 配置 | C1 单流输出 | C8 聚合输出 | TTFT（C1） |
|---|---|---|---|
| TP8（朴素，跨 NUMA） | **75.3 tok/s** | **352.4 tok/s** | 592 ms |
| TP8 + DP-attention 8 | 37.7 tok/s | 233.6 tok/s | 2825 ms |
| 仅 NUMA0 的 TP4 | 64.2 tok/s | 314.5 tok/s | 603 ms |

结论（复用前先看 JSON 的 `transferable` / `not_transferable`）：

- **朴素 TP8 赢了。** 在这个模型规模上，算力扩展的收益大于跨路 all-reduce 的代价——"绝不跨 `SYS`"的经验法则对稠密 27B 不成立。
- **DP-attention 把 C1 吞吐砍半。** 并发 1 时大部分 DP rank 闲置，权重却按 8 份切开。这是 SM100 上面向 MoE 的布局；本机 SM120 上的 M3.1 走的是 TP8 + Marlin（方案 C），不是 DP-attention。
- 单 NUMA 节点 TP4 用一半硬件保住了 TP8 约 89% 的 C1 吞吐——另外 4 张卡要挪作他用时这是正确选择。

## 已知局限与待验证问题

| 项目 | 状态 |
|---|---|
| 上面的拓扑、CPU/内存、驱动/CUDA/NCCL 数值 | **已实测**（第一方快照，2026-10-23） |
| TP8 vs 单节点 TP4 vs TP8+DP-attention（稠密 27B，BF16） | **已实测**——朴素 TP8 最快（见上表） |
| TP4 × PP2 vs TP8（需要 >4 卡余量的模型） | **未验证**——本轮未测试 sglang 的 PP |
| 该拓扑上的 MoE 模型（DP-attention + EP，M3.1 布局） | **MiniMax-M3.1-Flash NVFP4 已用 Marlin 绕行方案实测通过（方案 C）**；DP-attention/EP 布局本身在 SM120 上仍未验证 |
| NUMA 绑核（`numactl --cpunodebind`）是否提升吞吐 | **未验证** |
| 这台机器上 NCCL 是否需要调参（如 `NCCL_P2P_DISABLE`） | **未验证**（见 [03 通信](../../docs/zh-CN/03-communication.md)） |
