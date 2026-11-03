# 09 失败的配置

大多数调优指南只发布"什么能跑"。那是最没用的那一半：配置失败时通常不会报错——服务照样启动、照样回答，只是比应有的速度慢；或者崩在一个指向错误层级的日志上。

本章是决策库的反面：**跑不通的配置**，附现象、真实原因和结论。标 **实测** 的是在 8×72 参考机上真实踩到的；其余可从框架/内核机制推导。

## 9.1 怎么读

| 列 | 含义 |
|---|---|
| **现象** | 你最先看到的东西——报错行或可疑的数字 |
| **真实原因** | 排查后确认的问题所在 |
| **结论** | `never`（没有参数能救）/ `avoid`（能跑但得不偿失）/ `conditional`（满足条件才成立） |

## 9.2 8×72 机器上实测到的

| ID | 配置 | 现象 | 真实原因 | 结论 |
|---|---|---|---|---|
| F-01 | MiniMax-M3.1-Flash NVFP4 跑在 SM120（RTX PRO 5000/6000） | 启动失败；绕过 prefill 后端后，decode indexer 直接拒绝这张卡 | 内核源码硬编码 `arch not in ((10,0),(10,3))`——MSA 稀疏注意力的 decode indexer 只认 SM100/SM103 | 该内核版本 `never`。改用 NVFP4 + Marlin 路径，或等 sm_120 移植 |
| F-02 | Qwen3.5 系列跑在 MiniMax 定制镜像（triton 3.9） | 能启动，随后在线性注意力内核崩溃 | 镜像里的 triton 移除了 `tl.make_block_ptr`，模型内核仍在调用 | 该镜像 `never`。换主线镜像（triton 3.8） |
| F-03 | 容器内跑 `bench_serving` | tokenizer 读不到本地路径；随机数据集生成卡住 | tokenizer 需要挂载模型目录；数据集生成要联网拉配置 | `conditional`——挂载模型目录，并传 `-e HF_ENDPOINT=...` |
| F-04 | **稠密**模型用 TP=8 + `--enable-dp-attention --dp 8` | c1 只有 37.7 tok/s，朴素 TP8 是 75.3 | DP-attention 多一套 all-to-all 调度；稠密模型没有可切分的东西 | 稠密模型 `avoid`。它是 MoE 工具 |
| F-05 | TP4 但不指定卡（8 卡机器上随机选 4 张） | TP4 跨到 SYS 边界，比预期慢 20~30% | 调度器不会自动挑同一 NUMA 下的 4 张 | `avoid`。用 `CUDA_VISIBLE_DEVICES=0,1,2,3` 钉住 |
| F-06 | MoE 用 W4A4 激活 | 输出通顺但不可复现；greedy checksum 失败率 5.9% | 问题在激活精度，不在 router | `avoid`。先升到 A16（0/500 失败）；router 保持 FP32 是二次保险，且依赖 checkpoint |

## 9.3 根本起不来的配置

| ID | 配置 | 报错 / 现象 | 原因 | 结论 |
|---|---|---|---|---|
| F-10 | 模型有 4 个 KV 头却用 TP=3 | 引擎拒绝启动 | TP 必须整除注意力头数**和** KV 头数；4 % 3 ≠ 0 | `never` |
| F-11 | TP 大于注意力头数 | 引擎拒绝启动 | 头数无法分配 | `never` |
| F-12 | A100（SM80）上开 `--kv-cache-dtype fp8` | 启动失败或静默回退 fp16 | FP8 KV 需要 SM≥8.9，A100 没有 FP8 tensor core | A100 上 `never`，保持 fp16 KV |
| F-13 | 非 Blackwell 卡上跑 NVFP4 权重 | 找不到内核 / 不支持该量化 | NVFP4 需要 Blackwell 级 tensor core | `never`。退回 AWQ/GPTQ 或 FP8 |
| F-14 | AWQ 权重配 `--dtype bfloat16` | 启动失败或精度被静默改写 | vLLM 的 AWQ 内核只声明 fp16 激活 | `never`。用 `--dtype float16`（或不写） |
| F-15 | 4 × 72 GB 跑 72B BF16，`--gpu-memory-utilization 0.99` | 启动即 OOM，尽管"权重装得下" | 145 GB 权重 + 每卡 workspace 超过 4 × 72 × 0.99 | 该规模 `never`。上 8 卡或量化 |
| F-16 | `--max-model-len` 超过 KV 池容量 | `max seq len is larger than the maximum number of tokens that can be stored in KV cache` | 声明的上下文剩下的显存服务不了 | 该组合 `never`。降上下文或降 KV 精度 |
| F-17 | DeepSeek-V4.1-Flash-1017 跑 **4 × 72 GB**，任何 `--mem-fraction-static` | 加载权重时 OOM：`create_fp8_moe_weight_` → `Tried to allocate 1.05 GiB. GPU 1 has a total capacity of 71.12 GiB of which 384.00 KiB is free` | 476 GiB 权重 ÷ 4 卡 = **每卡 119 GiB**，而可用只有 71.12 GiB。这是算术问题，不是调参问题 | 4 × 72 GB 上 `never`。需要 8 卡、4 × 141 GB（H200），或量化到约 250 GiB |

## 9.4 能跑但结论是错的

这一类最贵：不崩、数字好看、结论错。

| ID | 配置 | 看起来对的地方 | 实际错在哪 |
|---|---|---|---|
| F-20 | 所有请求用同一个 prompt | TTFT 非常漂亮 | 前缀缓存替你做了 prefill；你测的是缓存不是模型 |
| F-21 | 每档只跑一次 | 吞吐曲线很平滑 | 没有方差、没有误差棒；±5% 的启动间波动就能翻转排名 |
| F-22 | 只报吞吐不报延迟 | "c64 是最佳配置" | c64 可能要先排队 1.6 秒；按 SLO 算 goodput 可能不如 c8 |
| F-23 | 对比不同引擎/量化时改了 `--max-tokens` 或并发档 | "AWQ 快 2 倍" | 不止一个变量变了 |
| F-24 | 压测机与服务同机 | 省事 | 压测进程抢 CPU/PCIe，污染 TTFT 与 P99 |
| F-25 | 标成 `measured` 的 KV/token，其实是 `2 × 层数 × KV 头数 × head_dim × 字节` | 数字很精确 | 那是推导不是实测；标 `derived`，公式要能看见 |
| F-26 | 短请求每档只测一次就画曲线 | 曲线看起来很平滑 | DeepSeek-V4.1-Flash-1017 上同一条命令、开着前缀缓存时，c8 三次得到 **71.4 / 71.8 / 162.6 tok/s**（128% 波动）；关掉 `--disable-radix-cache` 后同一档三次稳定在 **70.6~72.3**——波动来自缓存状态，c8 本身是真实低谷而非噪声。无论哪种情况，每档只跑一次都无法区分「配置差异」和「缓存/调度假象」 |

## 9.5 怎么加一条

提 issue 时附上：
1. 完整命令行（镜像 digest、框架版本）；
2. 第一行报错（不要贴整份日志）；
3. `nvidia-smi topo -m`；
4. 还试过什么、同样失败的。

"装不下"也算一条记录——[4×72 配方](../../recipes/zh-CN/4x-72gb-pcie.md)的负面列就是收这些的。
