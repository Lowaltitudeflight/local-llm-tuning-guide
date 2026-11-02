# Articles

Long-form write-ups that accompany the guide. Full text is currently Chinese
(originally published on CSDN); each entry has an English abstract so the
reasoning is usable without reading Chinese.

| Article | English abstract | Full text |
|---|---|---|
| RTX PRO 6000 without NVLink: measured deployment notes | A first-party record of deploying on an RTX PRO 6000 host without NVLink: what the PCIe topology actually looks like, where TP stops paying off, and the counterintuitive speculative-decoding result (3 draft tokens beat 7). | [中文](csdn-rtx-pro-6000-ceshi.md) |
| Multi-GPU tuning without NVLink | A practical walk-through of the decisions in chapters 01–06: find the PCIe/NUMA boundary, keep TP inside it (with the dense-27B counterexample), choose KV precision before adding GPUs, and measure instead of copying defaults. | [中文](csdn-wu-nvlink-duoka-tiaoyou.md) |

## Contributing a translation

An English translation of either article is welcome. Keep commands, parameter
names, and log excerpts verbatim; put the translation next to the Chinese file
with an `-en.md` suffix and add it to the table above.
