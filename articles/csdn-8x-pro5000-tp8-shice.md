# 教科书让你避开 NUMA，实测朴素 TP8 快了 12%：8×RTX PRO 5000 72GB 双路服务器首测

> 上一篇文章的规则是"TP 不要跨过 `SYS` 边界，宁可改 TP=2×PP=2"。这次我把这套规则带到一台更大的机器上验证：8 张 RTX PRO 5000 72GB、双路 Xeon、纯 PCIe 无 NVLink——结果**最朴素的 TP8（每层 all-reduce 都跨 NUMA）反而是最快的配置**。
> 这是我自己机器上的第一方实测（18 次独立冷启动、方差 ±1.5% 以内），不是转载数据。本文讲清为什么会这样、什么情况下经验法则仍然成立、以及可以直接抄的配置。

---

## 一、先看机器和那组"打脸"的数据

测试硬件：

| 特性 | 规格 |
|---|---|
| GPU | **8 × RTX PRO 5000 72GB Blackwell**（共 576 GB 显存） |
| 互联 | PCIe，**没有 NVLink** |
| CPU | 2 × Xeon Platinum 8488C（192 线程） |
| 内存 | 1.5 TB |
| 系统 | Ubuntu 22.04.5，驱动 595.71.05 |

拓扑结构是理解一切的前提，`nvidia-smi topo -m` 一跑：

```
        GPU0   GPU1   GPU2   GPU3   GPU4   GPU5   GPU6   GPU7
GPU0     X     NODE   NODE   NODE   SYS    SYS    SYS    SYS
...
GPU4    SYS    SYS    SYS    SYS     X     NODE   NODE   NODE
```

**4+4 布局**：GPU0–3 挂在 NUMA0 上，GPU4–7 挂在 NUMA1 上；同组卡之间走 `NODE`，跨组要过 CPU 间的 `SYS` 链路。这是双路服务器的典型形态，和上一篇 4 卡单路机器的结构完全不同。

然后是三种并行布局的正面对决。测试条件：**Qwen3.8-27B（BF16 未量化，稠密模型），SGLang 0.5.22，输入 4096 / 输出 512 token，temperature 0**；每个配置独立冷启动 3 次（每次换全新容器），结果取均值：

| 配置 | C1 单流输出 | C8 聚合输出 | C1 首字延迟 |
|---|---|---|---|
| **TP8（朴素，每层 all-reduce 跨 NUMA）** | **75.3 tok/s** | **352.4 tok/s** | **592 ms** |
| TP4（只用 NUMA0 的 4 张卡） | 64.2 tok/s | 314.5 tok/s | 603 ms |
| TP8 + DP-attention 8 | 37.7 tok/s | 233.6 tok/s | 2825 ms |

三个值得盯住的数字：

- **朴素 TP8 比"避 NUMA"的 TP4 快 17%（C1）/ 12%（C8）**——每层都跨 `SYS` 通信的方案赢了
- TP8 + DP-attention 的**单流吞吐直接砍半**（37.7 vs 75.3），首字延迟差 4.8 倍
- 三个配置的启动间方差都在 ±1.5% 以内，这个排序不是噪声

如果你看到这里的第一反应是"不对吧，跨 NUMA 不是最慢的链路吗？"——那这篇文章就是为你写的。

---

## 二、为什么"跨 SYS 的 TP8"能赢：算力和通信的账要一起算

`SYS` 确实是最慢的链路，这一点没被推翻。被推翻的是隐含假设——**"通信慢 → 总吞吐慢"**。这两笔账要分开算：

**通信的代价**：TP 的 all-reduce 发生在每一层。27B 稠密模型、64 层，每次 all-reduce 传输的是整层的激活值，跨 `SYS` 每次多付的延迟是真实的。但注意：这是**每次几毫秒级**的开销，不是数量级的差距——PCIe 5.0 跨 NUMA 依然有几十 GB/s 的有效带宽。

**算力的收益**：TP4 → TP8，矩阵乘的并行度翻倍，每层计算时间近乎减半。27B 模型单层的计算量和通信量是同一量级的，**省下来的计算时间 > 多付的通信时间**。

一句话：**经验法则"别跨 SYS"成立的隐含前提是"通信开销和计算开销同量级或更大"。模型够小、卡够多时，算力翻倍的收益会盖过跨 NUMA 的通信税。**

这也解释了 TP4 为什么只慢 12–17% 而不是慢一半：它省下了全部跨 NUMA 通信，但丢了一半算力，两头相抵后净亏一点点。

---

## 三、DP-attention 为什么在这里输得最难看

TP8 + DP-attention 8 是"教科书式"的避 NUMA 布局：attention 按数据并行切开（每个 rank 独立处理自己的请求，无需跨卡通信），MLP 部分保持 TP8。这台机器上 MiniMax M3.1 的官方启动脚本用的正是这套布局（`--tp 8 --attn-dp-size 8 --ep 8`，注释写明"规避 PCIe all-reduce"）。

但它有个对稠密模型致命的特性：**权重是按 8 份切开的，请求却是按 rank 独立处理的**。

- **并发 1 时**：只有 1 个 rank 在干活，其余 7 个 rank 的算力全部闲置——但权重照样切了 8 份，等效于"用 1/8 的算力跑一个被切碎的模型"。这就是 37.7 tok/s 的来历
- **并发 8 时**：8 个 rank 都有活干，但每个请求的 attention 都只吃一张卡，聚合吞吐仍然只有 233.6，比朴素 TP8 低 34%

**DP-attention 是为 MoE 大模型设计的**：几百 B 参数的 MoE，专家权重占大头，用 EP（专家并行）+ DP attention 换来的通信节省远超闲置代价。MiniMax M3.1（NVFP4 量化 MoE）用它是合理的。**稠密 27B 用它，就是拿大炮的炮架扛步枪。**

---

## 四、实测前的三次翻车（这部分可能比结论更有用）

这台机器上并没有"一次跑通"。以下三个坑，每一个都对应一类会在你自己机器上遇到的场景：

**坑 1：MiniMax M3.1-Flash 在 SM120 上根本起不来。** 我最初想测的就是这台机器的"原配"模型（同事的启动脚本都在）。启动即报错，顺着源码查下去是个死结：

- 模型侧强制要求 `--kv-cache-dtype fp8_e4m3`（NVFP4 量化的硬性要求）
- SGLang 的 SM120 兼容性检查又禁止 trtllm_mha prefill 使用 fp8 KV cache
- 换 prefill 后端绕过第一层后，撞上真正的硬墙：**MSA 稀疏注意力的 decode indexer 内核源码里硬编码 `arch not in ((10,0),(10,3))`——只认 SM100/SM103，SM120（RTX PRO 5000/6000 这类工作站卡）直接被拒**

这属于内核级限制，没有参数可绕。机器属主正在做 sm_120 移植（他们当天就在调 fused_moe 的架构分支）。**给你的启示：新模型 + 新卡组合起不来时，先 grep 报错源码里的架构检查，能省掉几小时的盲试参数。**

**坑 2：Qwen3.5 系列对 triton 版本敏感。** 换 Qwen3.8-27B 后，在 MiniMax 定制镜像里启动即崩：Qwen3.5 的线性注意力内核用了 `tl.make_block_ptr`，而那个镜像的 triton 3.9 把这个 API **移除**了（不再只是弃用警告）。换成官方主线镜像（triton 3.8，只是 warning）一次通过。**定制 dev 镜像跑通用模型，先查 triton/内核版本兼容性。**

**坑 3：官方镜像里也有两处小坑。** bench_serving 的 tokenizer 要挂载模型目录才能读本地路径；随机数据集生成要联网拉配置，容器内需加 `-e HF_ENDPOINT=https://hf-mirror.com`。

---

## 五、能直接抄的配置

**8 卡全用（本轮实测最快，稠密模型首选）：**

```bash
docker run -d --name llm --gpus all --ipc host --shm-size 64g --network host \
  -v /path/to/model:/models/mymodel:ro \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server --model-path /models/mymodel \
    --tp 8 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 32768 \
    --max-running-requests 64 \
    --host 0.0.0.0 --port 30000
```

**只用 4 张卡（另外 4 张要跑别的活时）：**

```bash
docker run -d --name llm --gpus all --ipc host --shm-size 64g --network host \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -v /path/to/model:/models/mymodel:ro \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server --model-path /models/mymodel \
    --tp 4 \
    --mem-fraction-static 0.85 \
    --host 0.0.0.0 --port 30000
```

要点：

- `--tp 8` 不需要任何"避 NUMA"的花哨参数——**对稠密模型，朴素的就是快的**（至少到 27B 这个规模）
- `CUDA_VISIBLE_DEVICES=0,1,2,3` 选的是 NUMA0 那 4 张卡，TP4 天然在快链路内
- 完整配方（含拓扑判断流程、参数逐条解释）在仓库 `recipes/8x-72gb-pcie.md`

---

## 六、这个结论的适用边界（比结论本身更重要）

按本项目的规矩，所有结论标依据：

| 结论 | 依据 | 能不能迁移 |
|---|---|---|
| 稠密 27B：朴素 TP8 > TP4 单节点 > TP8+DP-attention | **实测**（本机，3 次冷启动/配置，±1.5%） | 同类拓扑（双路、纯 PCIe、无 NVLink）+ 同量级模型，大概率成立 |
| TP8 比 TP4 只慢 12–17% 而非 50% | **实测 + 可推导**（计算/通信两笔账） | 模型越大、层数越深，跨 NUMA 通信税越重，TP8 的优势会缩小——**更大模型请重测** |
| DP-attention 适合 MoE、不适合稠密 | **实测（稠密侧）+ 可推导（机制）** | MoE 侧的"适合"是机制推断 + 该机 M3.1 生产配置佐证，**MoE 本机实测还没跑** |
| 具体吞吐数字 | **实测** | **不可迁移**——换了量化、换了框架版本、换了功耗墙，数字全变 |

特别说明两点：

1. **MoE 布局的本机实测是空缺的**——M3.1 等 SM120 移植完成后我会补测"DP-attention + EP 对 MoE 是否真比朴素 TP8 快"，那才是这套布局的正名之战
2. **TP4 × PP2（跨 NUMA 用流水线而非张量并行）这轮没测**，它理论上能同时拿到"不跨 SYS 通信"和"8 卡算力"，是最值得补的实验，配方里已挂为待验证项

---

## 七、仓库与工具

- GitHub：https://github.com/Lowaltitudeflight/local-llm-tuning-guide
- Gitee：https://gitee.com/wang-guangshuooo/local-llm-tuning-guide

**上一篇文章说"实测数据表是空的"，这次它有第一行了**：`benchmarks/README.md` 的 first-party 表格已收录本文全部数据（含 18 份原始运行记录，可逐条复查），机器可读的 JSON 已接入 `tools/planner.py` 的决策链。仓库里还有新配方的中英文版本，以及六章节方法论（拓扑 / 并行 / 通信 / 量化 / 推测解码 / KV 缓存）。

*你的机器也是双路 + 纯 PCIe？欢迎在评论区贴出 `nvidia-smi topo -m` 输出和你测的模型规模，尤其想知道：多大参数量的稠密模型上，跨 NUMA 的通信税会反超算力收益——那个翻转点目前只有理论猜测，没有公开数据。*
