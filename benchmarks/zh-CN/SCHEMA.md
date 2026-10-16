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
| `software` | 驱动 / CUDA / NCCL / 框架及版本 / 镜像 tag / 模型 revision |
| `model` | 模型名、量化方式与工具、revision |
| `workload` | 并发档位、输入/输出长度、采样参数 |
| `results[]` | 每个配置一行;指标名用 `*_c1_*`(单并发)/ `*_c8_*`(聚合)区分 |
| `variance` | 各指标多次独立启动的范围 |
| `methodology` | 窗口长度、预热、重复次数的一句话概括 |
| `source` | 来源项目与 URL(external-reference 必填) |
| `transferable` / `not_transferable` | 迁移边界 |
| `submitted_by` | GitHub 用户名(first-party 填) |
| `date` | 测量日期(YYYY-MM-DD;未知写 null,不要猜) |

## 提交方式

通过 [benchmark report issue 模板](https://github.com/Lowaltitudeflight/local-llm-tuning-guide/issues/new?template=benchmark-report.yml)提交,由维护者整理入库;或直接 PR 按本 schema 增加文件(见 [CONTRIBUTING.md](../../CONTRIBUTING.zh-CN.md))。
