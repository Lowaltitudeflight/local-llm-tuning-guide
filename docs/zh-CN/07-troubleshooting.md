# 07 排错手册

症状 → 可能原因 → 先查什么 → 改什么。每条都带证据等级：**实测**、**可推导**、**未验证**。

## 启动与显存

| 症状 | 可能原因 | 先查 | 处理 | 等级 |
|---|---|---|---|---|
| `ValueError: Loaded weights leave no GPU memory for the KV cache` | `mem-fraction-static` / `gpu-memory-utilization` 低于权重占比 | 用日志里的权重大小对比预算 | 提高比例；量化压权重；加卡 | 实测 |
| 不是加载权重时 OOM，而是 CUDA 图捕获时 OOM | 捕获池 + 激活超过剩余显存 | 日志出现 `Capture target ... begin` 后紧跟 OOM | 关 prefill 图、降 `cuda-graph-max-bs-decode`、减小 chunk、降 mem fraction | 实测 |
| 捕获时 `NCCL error: unhandled cuda error` | 多半是 NCCL 内部的隐藏分配失败，不是网络问题 | 往上看是否有一行 OOM；`NCCL_DEBUG=INFO` 复跑 | 同上：先降捕获显存，再复测 | 实测 |
| 只有长 prompt 才 OOM | KV 池装不下该请求或该并发 | 算 `tokens × KV/token`，对比 `max_total_num_tokens` | 缩短上下文、降 KV 精度、减并发、加卡 | 可推导 |
| 服务起来后每 ~2 分钟死一次 | 重启策略掩盖了启动异常 | `docker inspect` 的 RestartCount；从一轮崩溃的开头读日志 | 修第一个异常；不要依赖自动重启 | 实测 |

## 输出错误或为空

| 症状 | 可能原因 | 先查 | 处理 | 等级 |
|---|---|---|---|---|
| 语句通顺但答案错 / 乱码 | MoE 激活量化过激，有时叠加 router 被量化 | 跑 greedy checksum 测试（同 prompt、temperature 0、重复 50+ 次）；看量化 config 的激活精度 | **先把激活提到 A16**（W4A4 → A16 实测失败率 5.9% → 0%）；router 保持 FP32/BF16 作为免费的二次保险 | 实测 |
| 专家权重加载报 2 倍尺寸不匹配 | 打包的 FP4/NVFP4 专家被按 FP8 加载 | 类似 `size of tensor a (192) must match ... (384)` | 不要强制 `--quantization`，让框架自动识别打包格式 | 实测 |
| 推理模型 `content` 为空 | 输出 token 全花在 `reasoning_content` 上 | 看原始响应 | 提高 `max_tokens`；启用 reasoning parser；聊天界面单独渲染思考过程 | 实测 |
| 模型名 404 | 客户端发的 id 和服务端不一致 | `GET /v1/models` | 设 `--served-model-name`，或使用完全一致的 id | 可推导 |
| “startup complete” 后仍 503 | 服务还在 warmup | 轮询 `/health_generate`，看日志 | 等到 “ready to roll”，加长客户端超时 | 实测 |

## SM120 / 消费级 Blackwell

| 症状 | 可能原因 | 先查 | 处理 | 等级 |
|---|---|---|---|---|
| `requires SM100 or SM103` | 内核路径只对数据中心 Blackwell 开放 | 日志里的内核 / plan 名 | 走受支持的 Marlin/回退路径，或等上游移植 | 实测 |
| MegaMoE 报 “weights are not ready” | MegaMoE 硬编码到 SM100/103 | `is_mega_moe_experts_ready` 返回 False | NVFP4 改用 Marlin；见 [8×72 方案 C](../../recipes/zh-CN/8x-72gb-pcie.md#方案-c大-nvfp4-moeminimax-m31-flash252-gbsm120-绕行方案) | 实测 |
| TRT-LLM FMHAv2 拒绝 fp8 KV | SM120 不支持该组合 | `TRTLLM FMHAv2 ... does not support fp8_e4m3 KV cache` | 换 attention backend 或 KV dtype | 实测 |
| 第一条请求卡几分钟 | Triton/JIT 冷编译 | 日志 `Triton kernel ... device-loaded after serving started` | 提前 warmup；保持容器热；放宽首请求超时 | 实测 |

## 互联与容器运维

| 症状 | 可能原因 | 先查 | 处理 | 等级 |
|---|---|---|---|---|
| PCIe 上多卡扩展远低于预期 | TP 跨 socket/switch；小消息 all-to-all | `nvidia-smi topo -m`；改动前后跑 `nccl-tests` | TP 留在同一组内；优先 one-shot all-reduce；考虑副本 | 实测 |
| 同一模型换一台机器 NCCL 带宽差很多 | BIOS / PCIe 根端口配置 | 对比 1 GiB `nccl-tests` busbw；看 `lspci` 的 redirect 标志 | 应用厂商 BIOS 修复后复测；在此之前不要归因于 NCCL | 实测 |
| `docker stop` 卡住并报 “did not receive an exit event” | GPU 容器退出竞态 | `docker ps` 里容器还在 | `docker rm -f <name>`，等几秒再重建 | 实测 |
| 前缀缓存好像没用 | prompt 之间没有共享前缀 | 同一负载对比 cache-on / cache-off | 只有前缀重复时才有收益；ShareGPT 这种唯一 prompt 不会体现 | 实测 |

## 怎么报新的一条

开 issue，带上完整命令、第一行错误及其上方 20 行日志、`nvidia-smi topo -m`、框架与镜像版本。可复现的失败和修复同样有价值。
