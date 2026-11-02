# Glossary

| Term | Meaning in this guide |
|---|---|
| **TP** | Tensor parallelism: split each layer across GPUs; all-reduce every layer. Fast inside one NVLink/PCIe group, expensive across groups. |
| **PP** | Pipeline parallelism: split layers into stages; cross-stage traffic only at boundaries. |
| **DP** | Data parallelism: independent model replicas, no per-layer communication. |
| **EP** | Expert parallelism: split MoE experts across GPUs; requires all-to-all dispatch/combine. |
| **DCP** | Decode context parallelism: split the KV cache across GPUs; limited benefit over PCIe. |
| **GQA** | Grouped-query attention: fewer KV heads than query heads; KV cache is smaller than MHA. |
| **MLA** | Multi-head latent attention: compressed KV representation; KV/token is much smaller than GQA. |
| **MSA** | MiniMax sparse attention: block-sparse attention with a lightning indexer. |
| **FP8 / MXFP8 / NVFP4** | 8-bit float, block-scaled 8-bit, and 4-bit NVIDIA float formats. Lower precision = less memory and bandwidth, more kernel/portability constraints. |
| **AWQ / GPTQ** | Weight-only 4-bit quantization methods; dequantize on load. |
| **Prefix cache** | Reuse KV for a repeated prompt prefix; gains depend on how much traffic shares prefixes. |
| **TTFT** | Time to first token: prefill + queueing. |
| **TPOT / ITL** | Time per output token / inter-token latency: decode speed. |
| **Goodput** | Throughput that meets a latency or SLO constraint; raw tok/s alone can mislead. |
| **PCIe switch / NUMA group** | The set of GPUs that communicate without crossing the CPU socket; the practical TP boundary. |
| **SM100 / SM103 / SM120** | NVIDIA Blackwell compute capabilities: datacenter (B200/B300/GB300), and consumer/workstation (RTX PRO 5000/6000, RTX 50). Kernel support differs. |
