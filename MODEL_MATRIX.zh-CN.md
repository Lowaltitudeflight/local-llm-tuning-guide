# 模型矩阵

这张表回答三个问题：哪些模型真的在本指南覆盖的硬件上跑过、各需要多少显存、证据在哪里。

**怎么读**

- **权重**是磁盘上的 checkpoint 大小，不是单卡占用；除以 TP 只是粗略估算。
- **KV/token** 默认指整个模型每 token 的开销，标注“每 rank”的除外。它取决于层数、KV 头数、head_dim 和 KV 精度，只能当预算输入，不是常数。
- **每个 KV 数字都可以重算**：`2 × 层数 × KV 头数 × head_dim × 每元素字节`（系数 2 代表 K 和 V 各一份）。下面的[推导表](#kvtoken-推导)列出了我们核对过的模型几何。带 **†** 的数字来自部署笔记，没有可重算的公开 config——当预算输入用，别当已验证值。
- **最少 72 GB 卡数**是“权重 + 一小块 KV”的下限。每卡还要再加 10–30 GB 给 CUDA context、激活、CUDA 图、框架 workspace。刚好“装得下”的配置通常服务不好。
- **证据**列链接到配方或跑分 JSON；“推导”表示按模型 config 算出来的，不是在本机实测。

| 模型 | 参数（激活） | 量化 | 权重 | KV/token | 最少 72 GB 卡 | 推荐并行 | 证据 |
|---|---|---:|---:|---:|---:|---|---|
| Qwen3-8B | 8B | BF16 | ~16 GB | ~144 KiB (fp16) | 1 | 单卡/副本 | 推导 |
| Qwen3.8-27B | 27B | BF16 | ~52 GB | ~256 KiB (fp16) | 2 | 单 NUMA 节点内 TP2 | [三机单卡研究](benchmarks/zh-CN/README.md) |
| Qwen3.8-27B | 27B | NVFP4 | ~21 GB | ~128 KiB (fp8 KV) | 1 | 单卡/副本 | 推导 |
| Qwen3-32B | 32B | BF16 | ~62 GB | ~256 KiB (fp16) | 2 | 单 NUMA 节点内 TP2/TP4 | 推导 |
| Qwen3-32B | 32B | NVFP4 | ~20 GB | ~128 KiB (fp8 KV) | 1 | 单卡/副本 | 推导 |
| Qwen3-235B-A22B | 235B (22B) | NVFP4 | ~134 GB | ~94 KiB (fp8 KV) | 3–4 | TP4 + EP4；与 2 副本对比实测 | 推导 |
| GLM-5.3-Flash | ~355B | FP8 | ~306 GB | ~96 KiB (fp8 KV) † | 6–8 | TP8 + EP8 | [GLM TP8+EP8 数据](benchmarks/zh-CN/README.md) |
| GLM-5.3-Flash | ~355B | NVFP4 | ~190 GB | ~96 KiB (fp8 KV) † | 4–6 | TP4/TP8 + EP；实测 | 推导 |
| MiniMax-M3.1-Flash | 428B (23B) | MXFP8 + NVFP4 专家 | ~252 GB | **~7.5 KiB 推导（每 rank，fp8，TP8）** | 8 | SM120：TP8 + Marlin；SM100：TP8 + MegaMoE | [8×72 配方方案 C](recipes/zh-CN/8x-72gb-pcie.md#方案-c大-nvfp4-moeminimax-m31-flash252-gbsm120-绕行方案) · [JSON](benchmarks/data/minimax-m31-flash-8x72-sm120.json) |
| DeepSeek-V4.1-Flash-1017 | 538B 级 | FP8 + FP4 专家 | ~510 GB（476 GiB） | **~785 B 实测** · ~2.6 KiB 报告值 † | 8 | TP8 + EP8；Engram 表名义上放主机（实测暂未释放显存） | [8×72 TP8 矩阵](benchmarks/data/deepseek-v41-flash-8x72-tp8.json) · [部署手册](https://github.com/Lowaltitudeflight/llm-private-deploy-handbook) |

## KV/token 推导

公式：`2 × 层数 × KV 头数 × head_dim × 每元素字节`，系数 2 是 K 和 V，`每元素字节` 在 fp16/bf16 下是 2、fp8 下是 1。

| 模型 | 层数 | KV 头数 | head_dim | fp16 | fp8 |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 36 | 8 | 128 | 144 KiB | 72 KiB |
| Qwen3.8-27B | 64 | 4 | 256 | 256 KiB | 128 KiB |
| Qwen3-32B | 64 | 8 | 128 | 256 KiB | 128 KiB |
| Qwen3-235B-A22B | 94 | 4 | 128 | 188 KiB | 94 KiB |
| MiniMax-M3.1-Flash | 60 | 4 | 128 | 120 KiB（TP8 每 rank 15 KiB） | TP8 每 rank 7.5 KiB |

这张表想抓住两个坑：

- **head_dim 不总是 128。** Qwen3.8-27B 是 4 个 KV 头、head_dim 256，所以它的 KV/token 与 8 头/128 维的模型相同——只看头数减半就去推算另一行是错的。
- **fp8 砍的是每元素字节，不是头数。** 如果某行的 fp16 与 fp8 数字相差 4 倍，其中一个一定错了。

`python tools/check_model_matrix.py` 会重算上表中几何公开的行，对不上就报错；CI 里会跑。

† GLM-5.3-Flash 与 DeepSeek-V4.1-Flash-1017 的 config 我们无法从公开 checkpoint 重算（DeepSeek 是 MLA + Engram，GQA 公式对它根本不适用）。

DeepSeek-V4.1-Flash-1017 的两个数字相差 3.4 倍，**两个都刻意保留**：

- `~2.6 KiB` 来自部署笔记，没有写明分母（每 rank？整个模型？含不含 Engram？）。
- `~785 B` 是在本机实测的：服务启动报告 `max_total_num_tokens=12,443,136`，KV 可用 `9.10 GiB`，即 `9.10 GiB ÷ 1244 万 = 785 B/token`。

实测值才是调度器真正分配的量，在这套构建上做容量规划应当用它；报告值保留是因为我们无法复现它的分母。按 config 几何推算（40 层 × 576 × 1 B）会得到约 23 KiB/token，是实测值的 30 倍——这个模型的压缩注意力与 Engram 布局完全不能用 GQA 公式描述。请用 `--kv-bytes-per-token` 在自己的构建上填数，不要直接信这两个值中的任何一个。

## 这张表想避免什么

1. **“能装进单卡”不等于“单卡能服务”。** 52 GB 的 checkpoint 放在 72 GB 卡上，只剩约 20 GB 给 KV、激活和图；长上下文或并发一样会 OOM。
2. **252 GB 的模型装不进 4 张 72 GB 卡。** 每卡 63 GB 权重，只剩 8–12 GB，连运行时开销都不够。见 [4×72 配方](recipes/zh-CN/4x-72gb-pcie.md)。
3. **MoE 不是“换了个名字的稠密模型”。** 专家并行、router 精度、all-to-all 会同时改变显存布局和最优 TP/EP 组合。

## 怎么加一行

用 [benchmark report 模板](https://github.com/Lowaltitudeflight/local-llm-tuning-guide/issues/new?template=benchmark-report.yml)，或直接提 PR 新增符合 `lltg.benchmark.v1` 的 `benchmarks/data/*.json`。如果只有“装不下”的结果，也算一行——把 OOM 日志放进 [4×72 配方](recipes/zh-CN/4x-72gb-pcie.md) 的负面列。
