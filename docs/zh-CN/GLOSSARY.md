# 术语表

| 术语 | 在本指南中的含义 |
|---|---|
| **TP** | 张量并行：把每层切到多卡，每层都要 all-reduce。同一 NVLink/PCIe 组内快，跨组代价高。 |
| **PP** | 流水并行：按层切 stage，只在边界通信。 |
| **DP** | 数据并行：独立模型副本，没有逐层通信。 |
| **EP** | 专家并行：把 MoE 专家分散到多卡，需要 all-to-all 分发/合并。 |
| **DCP** | 解码上下文并行：KV 切到多卡；PCIe 上收益有限。 |
| **GQA** | 分组查询注意力：KV 头少于查询头，KV 比 MHA 小。 |
| **MLA** | 多头潜在注意力：压缩 KV 表示，每 token KV 远小于 GQA。 |
| **MSA** | MiniMax 稀疏注意力：块稀疏 + lightning indexer。 |
| **FP8 / MXFP8 / NVFP4** | 8 位浮点、块缩放 8 位、NVIDIA 4 位浮点。精度越低越省显存和带宽，但内核/可移植性限制越多。 |
| **AWQ / GPTQ** | 仅权重的 4 位量化，加载时反量化。 |
| **前缀缓存** | 复用重复 prompt 前缀的 KV；收益取决于流量里前缀重复度。 |
| **TTFT** | 首 token 时间：prefill + 排队。 |
| **TPOT / ITL** | 每输出 token 时间 / token 间延迟：decode 速度。 |
| **Goodput** | 满足延迟/SLO 约束的有效吞吐；只看 tok/s 会误导。 |
| **PCIe Switch / NUMA 组** | 不跨 CPU socket 就能通信的那组卡；实际上的 TP 边界。 |
| **SM100 / SM103 / SM120** | NVIDIA Blackwell 计算能力：数据中心（B200/B300/GB300）与消费/工作站（RTX PRO 5000/6000、RTX 50）。内核支持不同。 |
