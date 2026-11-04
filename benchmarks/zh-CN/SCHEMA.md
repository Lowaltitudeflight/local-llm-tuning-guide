# 机器可读数据格式 / Machine-readable data format

`benchmarks/data/*.json` 存放结构化的测量数据。`benchmarks/README.md` 里的表格由这些文件支撑:**表格给人看,JSON 给工具用**。

`tools/planner.py --benchmarks benchmarks/data` 会读取本目录,把已有实测数据直接带进规划输出——数据不只是文档,还能进决策链。

## 规则

1. **schema 字段必须是 `lltg.benchmark.v1`**,否则 planner 拒绝加载(宁可报错,不猜格式)。
2. **不知道的字段写 `null`,不写猜测值。** 这条纪律与 [METHODOLOGY.md](METHODOLOGY.md) 一致:空白比编造有价值。
3. `kind` 只有两种:
   - `first-party` —— 本项目或贡献者在自己机器上测的(必须满足 METHODOLOGY 的重复次数与波动要求)
   - `external-reference` —— 公开可查的第三方测量(必须写明来源与不可迁移部分)
4. `variance` 必填于 `first-party`(至少 3 次独立启动的范围);外部数据有则填。
5. `transferable` / `not_transferable` 必填:明确哪些结论可迁移、哪些不可。

## 字段说明

| 字段 | 含义 |
|---|---|
| `id` | 文件名同名的稳定标识(kebab-case) |
| `kind` | `first-party` 或 `external-reference` |
| `hardware` | GPU 型号×数量、显存、功耗上限、互联、`nvidia-smi topo -m` 摘要、CPU/NUMA |
| `software` | 驱动 / CUDA / NCCL / 框架及版本 / 镜像 **digest**（不能只写 tag）/ 模型 revision |
| `software.image_digest` | `repo@sha256:...`；可变 tag 无法定位构建 |
| `software.clocks` | 运行时的时钟策略（锁定 / 默认 / 未知）；没有它，"实测"无法复现 |
| `model` | 模型名、量化方式与工具、revision |
| `workload` | 并发档位、输入/输出长度、采样参数、prompt 来源或 seed |
| `commands.server` | 服务端完整启动命令（占位符不算可复现） |
| `commands.client` | 压测端完整命令与版本 |
| `results[]` | 每个配置一行;指标名用 `*_c1_*`(单并发)/ `*_c8_*`(聚合)区分 |
| `variance` | 各指标多次独立启动的范围 |
| `methodology` | 窗口长度、预热、重复次数的一句话概括 |
| `source` | 来源项目与 URL(external-reference 必填) |
| `transferable` / `not_transferable` | 迁移边界 |
| `submitted_by` | GitHub 用户名(first-party 填) |
| `date` | 测量日期(YYYY-MM-DD;未知写 null,不要猜) |
| `reproducibility` | 必填。`{status, commands_recorded, image_pinned, clocks_recorded, variance_runs, missing[], note?}`，见下 |

## 可复现性字段

每个文件都必须有，缺了 `tools/check_repo.py` 会直接失败。

| `status` | 含义 | 要求 |
|---|---|---|
| `full` | 只靠这个文件就能重跑 | 命令、固定镜像 digest、时钟、方差齐全，`missing` 必须为空 |
| `partial` | 在真实硬件上测过，但部分复现要素当时没记录 | `missing` 必须列出 |
| `external-unverified` | 第三方数字，来源已无法定位 | `missing` 必须列出 |

这个字段存在的原因：「测过」和「可复现」不是同一个声明。一份数据可以是真实测量，同时无法重跑——诚实的做法是写清楚缺什么，而不是让读者自己去发现。

## 实测 vs 推导

表里两个长得一样的数字，立场可能不同：

- **实测**：由某次运行产出，且有 `variance`、`commands.*`、`software.clocks`。
- **推导**：由模型 config 或硬件规格算出（KV/token 用 `2 × 层数 × KV 头数 × head_dim × 字节`，权重用 `参数量 × 字节`）。它仍然有用，但必须标 `derived` 并让公式可见——因为它不会和现实矛盾，只会和错误的 config 矛盾。

如果数字是从启动日志反推的，而不是直接观测的，写 `derived-from-log` 并注明是哪一行。本目录里早于 `commands`/`clocks` 字段的文件在这条轴上是不完整的；它们的 `measured` 应读作"在真实硬件上跑过"，而不是"完全可复现"。

## 提交方式

通过 [benchmark report issue 模板](https://github.com/Lowaltitudeflight/local-llm-tuning-guide/issues/new?template=benchmark-report.yml)提交,由维护者整理入库;或直接 PR 按本 schema 增加文件(见 [CONTRIBUTING.md](../../CONTRIBUTING.zh-CN.md))。
