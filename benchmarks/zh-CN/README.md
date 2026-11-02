# 实测数据

本目录存放可复现的性能数据。**所有条目必须遵循 [METHODOLOGY.md](METHODOLOGY.md) 的测量纪律。**

## 机器可读数据

外部参考数据与社区提交同时以结构化 JSON 存放在 [`data/`](../data/) 中(格式见 [data/SCHEMA.md](SCHEMA.md))。表格给人看,JSON 给工具用:

```bash
python tools/planner.py --gpus RTX4090-24 --count 4 --params 35 --quant awq \
    --benchmarks benchmarks/data
```

planner 会把已有实测数据直接带进规划输出。

## 本项目的实测数据

| 硬件 | 模型 | 配置 | C1 输出 | C8 聚合 | 提交者 | 链接 |
|---|---|---|---|---|---|---|
| 8 × RTX PRO 5000 72GB（PCIe，双 NUMA，无 NVLink） | Qwen3.8-27B BF16 | TP8（朴素） | 75.3 tok/s | 352.4 tok/s | Lowaltitudeflight | [JSON](../data/qwen38-27b-8x-72gb-pcie-parallelism.json) |
| 同上 | Qwen3.8-27B BF16 | TP8 + DP-attention 8 | 37.7 tok/s | 233.6 tok/s | Lowaltitudeflight | 同上 |
| 同上 | Qwen3.8-27B BF16 | TP4（单 NUMA 节点） | 64.2 tok/s | 314.5 tok/s | Lowaltitudeflight | 同上 |

2026-10-23 实测，输入 4096 / 输出 512，sglang 0.5.22。关键结论：这个稠密模型上朴素 TP8 反而最快——复用前先看 JSON 的 `transferable` / `not_transferable`。

### 三台 Blackwell 单卡 BF16 对比（2026-10-26）

Qwen3.8-27B BF16，TP1，vLLM 0.32.0 同 digest，输入 1024 / 输出 256（[JSON](../data/qwen38-27b-bf16-singlecard-three-machines.json)）：

| 机器 | c1 | c8 | c32 | c64 | 32K c1 | ShareGPT c64 |
|---|---:|---:|---:|---:|---:|---:|
| A：RTX 5090 96GB（575 W） | 29.68 | 177.58 | 434.42 | 559.63 | 14.46 | 900.34 |
| B：RTX PRO 6000 Server 96GB（600 W） | 26.10 | 182.08 | 491.85 | 697.91 | 16.67 | 933.12 |
| C：RTX PRO 5000 72GB（300 W） | 21.94 | 137.42 | **300.72** | **300.50** | 11.31 | 483.84 |

关键结论：c1 由显存带宽决定；C 的 300 W 功耗墙让 c32 和 c64 几乎一样（硬天花板）；B 的算力优势随并发放大；ShareGPT 不能从定长结果外推。

### GLM-5.3-Flash FP8，TP8+EP8（两台 96GB PCIe 主机，2026-10-22）

306 GB FP8 MoE，vLLM 0.32.0 TP8+EP8，max-model-len 131072（[JSON](../data/glm53-flash-fp8-8card-tp8ep.json)）：

| 机器 | c1 | c8 | c32 | c64 | ShareGPT c64 |
|---|---:|---:|---:|---:|---:|
| A：8× RTX 5090 96GB | 74.37 | 405.55 | 955.44 | 1226.62 | 1621.41 |
| B：8× RTX PRO 6000 Server 96GB | 63.99 | 378.19 | 865.24 | 1136.27 | 1462.01 |

关键结论：A 领先 7–16%；BIOS 修复后 B 的 1 GiB NCCL 已与 A 持平，差距来自显存带宽和小消息 all-to-all，不是大消息互联。

### MiniMax-M3.1-Flash NVFP4，TP8 + Marlin（8× RTX PRO 5000 72GB，SM120，2026-11-02）

252 GB checkpoint，MXFP8 dense + NVFP4 打包专家，走 SM120 的 Marlin 绕行方案（见 [8×72 配方方案 C](../../recipes/zh-CN/8x-72gb-pcie.md#方案-c大-nvfp4-moeminimax-m31-flash252-gbsm120-绕行方案)）。vLLM bench，OpenAI completions，输入 1024 / 输出 256，seed 42，cache-on 3 次均值（[JSON](../data/minimax-m31-flash-8x72-sm120.json)）：

| 配置 | C1 输出 | C8 聚合 | C32 聚合 | C64 聚合 | ShareGPT C64 |
|---|---:|---:|---:|---:|---:|
| TP8 + Marlin（SM120） | 94.4 | 489.8 | 740.4 | **1060.8** | **1093.8** |
| 关前缀缓存，C64（1 次） | — | 383.3 | 588.3 | **821.0** | 1084.1 |

关键结论：

- 前缀缓存是刚需：关闭后 c8–c64 吞吐下降 **20.5%–22.6%**，TTFT p50 变为 3.7–5.8 倍；`ctx32k_c4` 下降 **41.9%**。
- 1M 上下文可用：冷启动 **999,988 token** 请求 **517.7 秒**完成，无 OOM；1,047,992 token 边界请求也成功。1M 请求占 1,329,280 token KV 池约 79%，基本只能跑 1 条。
- 1M 配置的 C64 吞吐为 815.7 tok/s，比 128K 低约 23%（prefill 图关闭 + chunk 2048）。

## 外部参考数据

以下是**公开可查的第三方测量**，列在这里是为了给参数选择提供锚点。**注意它们是特定条件下的结果，不可直接外推。**

### 1. 推测解码：draft 长度增加反而变慢

| 项 | 条件 |
|---|---|
| 硬件 | 4 × RTX PRO 6000 **Max-Q**（约 300 W 低功耗版），无 NVLink |
| 互联 | PCIe，TP4 / DCP1 |
| 模型 | GLM-5.3-Flash，NVFP4 量化 MoE |
| 服务配置 | 4096 token prefill 预算，32 并发槽位，FP8 KV 缓存，GPU-only 前缀缓存 |
| 采样 | temperature 1 / top-p 0.95 |
| 测量 | 每格 5 个预热后的 30 秒窗口；C8 为聚合值 |

| 模式 | draft token | C1 输出 | C8 聚合 | 32K prefill | 验证器速率 |
|---|---|---|---|---|---|
| MTP3 | 3 | **281.3 tok/s** | **897.6 tok/s** | 13,816 tok/s | 112.14 steps/s |
| DFlash2 K7 | 7 | 229.0 tok/s | 702.6 tok/s | 13,955 tok/s | 87.56 steps/s |

**来源**：Local Inference Lab，`local-inference-lab/rtx6kpro`，GLM-5.3-Flash 模型页。

**该来源自行声明的波动范围**：另做的一组五次 C1 检查测得 MTP3 为 265.7–277.0 tok/s、DFlash2 为 212.9–225.2 tok/s。即约 **±5% 的启动间波动**。

**可迁移的结论**：在同一硬件上，draft 从 3 加到 7 并未带来收益。**不可迁移的部分**：具体数字，以及"3 一定优于 7"这个判断——它依赖该模型、该量化方案和该硬件的组合。

### 2. 无 NVLink 下副本优于张量并行

| 项 | 条件 |
|---|---|
| 硬件 | 4 × RTX 3090，无 NVLink |
| 模型 | AWQ 量化的 Qwen3.6-35B-A3B |
| 框架 | vLLM 0.21.0 |
| 工作负载 | 8192 token 上下文，128 并发 |

| 配置 | 聚合吞吐 |
|---|---|
| 双实例，各 TP=2 | **1437 tok/s** |
| 单实例，TP=4 | 610 tok/s |

**来源**：vllm-topology-bench，经第三方技术资料汇总。

**可迁移的结论**：模型能放进较少卡时，独立副本优于把卡做大并行。**不可迁移的部分**：具体数字（消费级卡、单次测量、特定模型）。

## 提交数据

见 [CONTRIBUTING.md](../../CONTRIBUTING.zh-CN.md)。提交时请：

1. 使用 [METHODOLOGY.md](METHODOLOGY.md) 中的模板
2. 附上完整的服务端与客户端命令
3. **包含波动范围**，不要只给最好的一次
4. 负面结果同样欢迎
