# Articles

Long-form write-ups that accompany the guide. Full text is currently Chinese
(originally published on CSDN); each entry has an English abstract so the
reasoning is usable without reading Chinese.

| Article | English abstract | Full text |
|---|---|---|
| RTX PRO 6000 without NVLink: measured deployment notes | A first-party record of deploying on an RTX PRO 6000 host without NVLink: what the PCIe topology actually looks like, where TP stops paying off, and the counterintuitive speculative-decoding result (3 draft tokens beat 7). | [中文](csdn-rtx-pro-6000-ceshi.md) |
| 8× RTX PRO 5000, TP8 vs TP4: three failures and one counterexample | A first-party log of the 8× RTX PRO 5000 build: three startup failures (SM120 kernel gate, triton API removal, container tokenizer paths) and the measured surprise that naive TP8 beat TP4-on-one-node — with the per-GPU efficiency caveat. | [中文](csdn-8x-pro5000-tp8-shice.md) |
| Multi-GPU tuning without NVLink | A practical walk-through of the decisions in chapters 01–06: find the PCIe/NUMA boundary, keep TP inside it (with the dense-27B counterexample), choose KV precision before adding GPUs, and measure instead of copying defaults. | [中文](csdn-wu-nvlink-duoka-tiaoyou.md) |
| Cross-machine PD at 128K over 91 Mbps gigabit | Three engine traps (vLLM KV connectors all fail on MLA+sparse-retrieval models), a hidden 8x KV-pool over-provisioning, the mrr sweet spot, why dual-instance beats 4+4 PD on 8 cards, and a bandwidth formula that predicted the measured 25.7 s TTFT exactly. | [中文](csdn-kua-ji-pd-fenli-128k.md) |

## Contributing a translation

An English translation of either article is welcome. Keep commands, parameter
names, and log excerpts verbatim; put the translation next to the Chinese file
with an `-en.md` suffix and add it to the table above.
