# 07 Troubleshooting

Symptom → likely cause → what to check → what to change. Every row is tagged
with the same evidence grades used elsewhere: **Measured**, **Derivable**, or
**Open**.

## Startup and memory

| Symptom | Likely cause | Check | Fix | Grade |
|---|---|---|---|---|
| `ValueError: Loaded weights leave no GPU memory for the KV cache` | `mem-fraction-static` / `gpu-memory-utilization` is below the weight fraction | Compare weight size with the budget in the log | Raise the fraction; shrink weights (quantize); add GPUs | Measured |
| OOM during CUDA-graph capture, not during weight load | Capture pool + activations exceed the remaining memory | Log shows `Capture target ... begin` followed by OOM | Disable prefill graphs, reduce `cuda-graph-max-bs-decode`, lower chunk size, lower mem fraction | Measured |
| `NCCL error: unhandled cuda error` during capture | Often a hidden allocation failure inside NCCL, not a network error | Look for the OOM line immediately above; rerun with `NCCL_DEBUG=INFO` | Same as above: reduce capture memory, then retest | Measured |
| OOM only on a long prompt, not at startup | KV pool is too small for that request or concurrency | Compute `tokens × KV/token`; compare with `max_total_num_tokens` | Shorter context, lower KV precision, fewer concurrent requests, more GPUs | Derivable |
| Model starts, then dies every ~2 minutes | Restart policy is hiding a startup exception | `docker inspect` RestartCount; read logs from the beginning of a crash cycle | Fix the first exception; do not rely on restarts | Measured |

## Wrong or empty output

| Symptom | Likely cause | Check | Fix | Grade |
|---|---|---|---|---|
| Fluent but wrong / garbled answers | Quantization mismatch, especially MoE router quantized with experts | Compare output with the unquantized model; check the quant config | Keep the router at higher precision; re-quantize experts only | Measured |
| Expert weights fail to load with a 2× size mismatch | Packed FP4/NVFP4 experts loaded as FP8 | Error like `size of tensor a (192) must match ... (384)` | Do not force `--quantization`; let the framework auto-detect the packed format | Measured |
| Empty `content` with a reasoning model | All output tokens were spent on `reasoning_content` | Inspect the raw response | Raise `max_tokens`; enable the framework’s reasoning parser; for chat UIs, render reasoning separately | Measured |
| HTTP 404 for the model name | Client sends a different id than the server exposes | `GET /v1/models` | Set `--served-model-name` or use the exact id | Derivable |
| HTTP 503 right after “startup complete” | Server warmup still running | Poll `/health_generate`; watch logs | Wait for “ready to roll”; increase client timeout | Measured |

## SM120 / Blackwell consumer cards

| Symptom | Likely cause | Check | Fix | Grade |
|---|---|---|---|---|
| `requires SM100 or SM103` | Kernel path is gated to datacenter Blackwell | Log line with the kernel/plan name | Use a supported Marlin/fallback path, or wait for an upstream port | Measured |
| MegaMoE “weights are not ready” | MegaMoE is hard-gated to SM100/103 | `is_mega_moe_experts_ready` returns False | Use Marlin (NVFP4) or another backend; see [8×72 Case C](../recipes/8x-72gb-pcie.md#case-c-large-nvfp4-moe-minimax-m31-flash-252-gb--sm120-workaround) | Measured |
| TRT-LLM FMHAv2 rejects fp8 KV | SM120 path does not support that combination | `TRTLLM FMHAv2 ... does not support fp8_e4m3 KV cache` | Choose a different attention backend or KV dtype | Measured |
| First request stalls for minutes | Triton/JIT cold compile | Log: `Triton kernel ... device-loaded after serving started` | Pre-warm the service; keep a warm container; allow long first-request timeout | Measured |

## Interconnect and container operations

| Symptom | Likely cause | Check | Fix | Grade |
|---|---|---|---|---|
| Multi-GPU scaling far below expectation on PCIe | Cross-socket or cross-switch TP; small-message all-to-all | `nvidia-smi topo -m`; `nccl-tests` before/after | Keep TP inside one group; prefer one-shot all-reduce; consider replicas | Measured |
| NCCL bandwidth much lower than the same model on another host | BIOS/PCIe root-port configuration | Compare `nccl-tests` 1 GiB busbw; check `lspci` redirect flags | Apply the vendor BIOS fix, then re-measure; do not attribute to NCCL before that | Measured |
| `docker stop` hangs with “did not receive an exit event” | GPU container teardown race | `docker ps` shows the container still up | `docker rm -f <name>`, wait a few seconds, then recreate | Measured |
| Prefix cache seems useless | Prompts have no shared prefix | Compare cache-on vs cache-off on the same workload | Expect gains only when prefixes repeat; ShareGPT-style unique prompts will not show them | Measured |

## How to report a new row

Open an issue with the exact command, the first error line and the 20 lines
above it, `nvidia-smi topo -m`, and the framework/image versions. A reproducible
failure is as valuable as a fix.
