# 08 Power and Cost

Throughput per watt is a first-class metric on multi-GPU workstations. Two
cards with the same silicon can differ by 2× in power limit, and the cheaper
one can win on tokens per dollar while losing on tokens per second.

## The power wall is a throughput ceiling

A 300 W card and a 600 W card can have the same peak FLOPS on paper. Under a
sustained decode load the 300 W card hits its cap and stops scaling: adding
concurrency raises per-watt efficiency but not total throughput.

First-party example (Qwen3.8-27B BF16, TP1, c64, GPU-only power):

| Machine | Power limit | c64 output | tok/s per W (measured draw) | tok/s per W (at the limit) |
|---|---:|---:|---:|---:|
| A — RTX 5090 96GB (modded) | 575 W | 559.6 tok/s | ~1.04 | ~0.97 |
| B — RTX PRO 6000 Server 96GB | 600 W | 697.9 tok/s | **~1.69** | ~1.16 |
| C — RTX PRO 5000 72GB | 300 W | 300.5 tok/s | ~1.02 | ~1.00 |

The two efficiency columns answer different questions, and mixing them is how the same run ends up with two "correct" numbers. The measured column divides by board power logged during the run; the limit column divides by the card's power cap. B never reached its 600 W cap at c64 — at ~1.69 tok/s/W it drew roughly 413 W, so normalising by the limit understates it by about 45%. A and C were at their caps, so the two columns nearly agree. **Use measured draw to compare cards; use the limit to size a PSU.**

The PRO 5000 result is the interesting one: **c32 and c64 were identical**
(300.72 vs 300.50 tok/s) — the card was already at its power ceiling. The
per-watt number stayed respectable, but the absolute throughput could not grow.

> These are GPU-only numbers from `nvidia-smi`. Wall power is higher: add PSU
> efficiency (~90%), CPU, fans, and networking. Use them for relative
> comparisons, not for a power bill.

## How to measure it

Run a steady-state window (30 s is enough), sample every second, and discard
the first few seconds:

```bash
# terminal 1: steady load (any client that keeps the batch busy)
python bench/run_matrix.py --labels c64 --reps 1 --base-url http://127.0.0.1:8000 \
  --model <model> --tokenizer <path> --result-dir /tmp/power

# terminal 2: sample power while it runs
while true; do
  nvidia-smi --query-gpu=index,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits
  sleep 1
done
```

Compute `tokens_per_watt = output_throughput / mean(power.draw)`. Record the
clock policy: locked clocks make runs comparable; automatic clocks drift with
cooling.

## Cost per million output tokens

```text
cost_per_1M = (mean_kW × hours × price_per_kWh) / (tokens / 1,000,000)
```

Example: 8 cards × 300 W = 2.4 kW, 1,060 tok/s aggregate, 1 hour:
`2.4 kWh × price / (1,060 × 3600 / 1e6) = 2.4 × price / 3.816`.
At 1 ¥/kWh that is about **0.63 ¥ per million output tokens** of GPU power —
before utilization, cooling, and idle overhead. A machine that idles at 200 W
spends most of its budget doing nothing; schedule batch work densely.

## Practical rules

1. **Measure at the concurrency you will actually serve.** A power-limited card
   can look fine at c1 and hit a wall at c32.
2. **Do not buy a 600 W card for a workload that is bandwidth-bound at c1.** The
   extra power budget buys compute headroom, not single-stream speed.
3. **Report tok/s/W together with tok/s.** They answer different questions:
   “how fast” vs “how much does fast cost”.
4. **Idle power counts.** A 24/7 service at 200 W idle costs more than the
   difference between two cards under load.

## What is still open

- Wall-power measurements (PSU + host) for the 72 GB and 96 GB configurations.
- Tokens per watt for MoE models at c32/c64, where all-to-all and expert
  routing change the power profile.
- Whether locked clocks change the ranking between 300 W and 600 W cards.

If you measure any of these, submit it with the [benchmark
template](https://github.com/Lowaltitudeflight/local-llm-tuning-guide/issues/new?template=benchmark-report.yml).
