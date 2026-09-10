# Runbook — weather ablation

Companion to `rahi_runbook.md`. Run this **in addition to** the jobs already going; nothing here touches them.

## What we're running and why

**Question:** how much weather diversity in *training* buys robustness across *all* weather at test time?

Eight jobs — 2 baselines × 4 training-weather subsets. Every job uses the **same segment split, same index, same `frame_stride`, and the same all-weather held-out segments**. The only thing that varies is which weather variants the model is allowed to train on.

**The weather filter applies to training only.** Validation always keeps all 10 variants, on the same held-out segments, for every job. That is what makes the results comparable — a clear-only model is still scored on fog, snow, dust and everything else. Do not "fix" this.

| task | model | trains on |
|---|---|---|
| 1 | box3d | clear only |
| 2 | box3d | clear + rain, snow, fog, dust |
| 3 | box3d | clear + rain, fog |
| 4 | box3d | clear + 4 random alterations (seeded) |
| 6 | monodetr | clear only |
| 7 | monodetr | clear + rain, snow, fog, dust |
| 8 | monodetr | clear + rain, fog |
| 9 | monodetr | clear + 4 random alterations (seeded) |

Tasks **0 and 5** are the all-weather control arm — they duplicate the `train_box3d.sbatch` / `train_monodetr.sbatch` jobs already running, so they're skipped below.

## Run it

```bash
cd /blue/iruchkin/patel.rahi/visuals-dataset-generation
git checkout main && git pull
export GROUP=iruchkin HPG_USER=$USER
cd visuals-ml/hipergator
sbatch --account=$GROUP --qos=$GROUP --array=1-4,6-9 train_weather_ablation.sbatch
```

`--array=1-4,6-9` skips the two control tasks. All eight run in parallel, one GPU each. box3d tasks ~6–8 h, MonoDETR tasks ~19 h. No index rebuild — they reuse `data/output/det_records.jsonl`.

If the existing control jobs die or you want the controls produced by this identical code path, add them back with `--array=0-9`.

## Check the first 2 minutes of any one task

```bash
tail -f logs/wabl_<jobid>_1.out
```

Four lines must appear:

1. `[info] Train on: clear   (validation always = all 10)`
2. `Frame stride 4: ...`
3. `Group split fingerprint: groups=... val=...`
4. `Weather ablation -> training on ['clear']` then
   `Train weathers ['clear']: kept N / M train samples ... Validation keeps ALL variants.`

**If line 4 is missing, stop** — the ablation didn't apply and the job is just another all-weather run.

## The one invariant that must hold

Every task must produce the **same val fingerprint**. That is the proof they're all scored on the same test set and the numbers can be compared.

```bash
grep -h "Group split fingerprint" logs/wabl_*.out | sed 's/.*val=/val=/' | awk '{print $1}' | sort -u
```

**Exactly one line expected.** More than one means the jobs were held out on different segments — report it, and don't compare the numbers.

Cross-check against the control runs too:

```bash
grep -h "Group split fingerprint" logs/wabl_*.out logs/box3d_*.out logs/monodetr_*.out | sed 's/.*val=/val=/' | awk '{print $1}' | sort -u
```

Ideally still one line — the ablations and the already-running controls share the split.

## Sanity check on train sizes

Train size should scale with the number of weathers kept; val size should be identical everywhere.

```bash
grep -h "kept .* train" logs/wabl_*.out
```

Roughly: clear ≈ 1 unit, clear+rain/fog ≈ 3 units, the two 5-weather sets ≈ 5 units, control ≈ 10 units. If clear-only is not about a tenth of the control's train size, something is wrong.

## What to send back

```
visuals-ml/reports/final_box3d_w-clear.json
visuals-ml/reports/final_box3d_w-clear4.json
visuals-ml/reports/final_box3d_w-clear2.json
visuals-ml/reports/final_box3d_w-rand4.json
visuals-ml/reports/final_monodetr_w-clear.json
visuals-ml/reports/final_monodetr_w-clear4.json
visuals-ml/reports/final_monodetr_w-clear2.json
visuals-ml/reports/final_monodetr_w-rand4.json
visuals-ml/logs/wabl_<jobid>_*.out
```

Plus:

1. The unique val fingerprint(s) — one, or a list if more
2. Which alterations `random4` drew (printed in tasks 4 and 9 as `Weather ablation -> training on [...]`)
3. Anything that died, with the task number and traceback

## Notes

- Each task writes its own checkpoint dir (`baselines/checkpoints/<model>_w-<tag>/`) and report, so nothing overwrites the running jobs.
- `random4` is seeded from the config `seed`, so it's reproducible; the log records the exact draw.
- Don't edit `configs/box3d.yaml` or `configs/monodetr.yaml` — the ablation is passed on the command line by the sbatch, and editing the configs would change the running jobs too.
