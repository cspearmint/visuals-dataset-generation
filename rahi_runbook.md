# Rahi's runbook

Everything to run tonight, in order. Three parts:

| Part | What | Where | Time |
|---|---|---|---|
| **1** | Launch box3d + MonoDETR training | GPU (sbatch) | submit in 2 min, runs 7–20 h |
| **2** | Evaluate the MonoDETR run that already finished | GPU (interactive) | 20–40 min |
| **3** | UQ wrapper shakedown | GPU (interactive) | 1–2 h |

**Do Part 1 first.** Those are long jobs — get them queued, then do Parts 2 and 3 while they run.

## Setup (once)

```bash
cd /blue/iruchkin/patel.rahi/visuals-dataset-generation
git checkout main && git pull
export GROUP=iruchkin HPG_USER=$USER
```

Confirm you have the files this runbook needs:

```bash
ls visuals-ml/configs/box3d.yaml visuals-ml/configs/monodetr.yaml visuals-ml/configs/monodetr_legacy_eval.yaml visuals-ml/hipergator/train_box3d.sbatch visuals-ml/hipergator/train_monodetr.sbatch
```

**If any are missing, stop** — you're not on current main.

---

# Part 1 — launch the two training runs

```bash
cd /blue/iruchkin/patel.rahi/visuals-dataset-generation/visuals-ml/hipergator
sbatch --account=$GROUP --qos=$GROUP train_box3d.sbatch
sbatch --account=$GROUP --qos=$GROUP train_monodetr.sbatch
```

No config editing, no index rebuild. Both reuse `data/output/det_records.jsonl`.

| Job | Per epoch | Estimate | Walltime |
|---|---|---|---|
| box3d | ~233k objects | 6–8 h | 12 h |
| MonoDETR | ~8k batches | ~19 h | 96 h |

Running past Friday is fine — correctness matters more than the deadline here.

## The split design (don't "optimise" this)

Split is by **segment**, with **all 10 weather variants on both sides**. A training segment contributes every one of its renderings; the held-out segments are entirely different drives, scored across every rendering. This mirrors the data the visuals agent will be trained on.

Compute is cut with `frame_stride: 4` — frames are ~10 Hz so neighbours are near-duplicates, and thinning them costs little information. **Weather variants are never dropped**; they are the experiment.

Subsampling runs *before* the split, so dropped in-between frames land in neither train nor val.

## Watch the first 5 minutes of each

```bash
tail -f logs/box3d_<jobid>.out
```

**box3d must show, in order:**

1. `[info] resnet18 IMAGENET1K_V1 served from baked cache`
   → if this errors, **stop**: `train.sif` doesn't have resnet18 baked and needs rebuilding from `train.def`. box3d builds two resnet18 encoders, and compute nodes have no internet.
2. `Frame stride 4: ... -> ... records (... frames x all weathers)`
3. `Dimension anchor (h, w, l) = [...] (encode and decode now share this)`
4. `Group split: <N> groups -> ...` then `Group split fingerprint: groups=... val=...`

**MonoDETR must show:**

1. `[info] MSDeformAttn: compiled CUDA op (fast path)`
   → if it says **pure-PyTorch fallback (SLOW)**, kill the job. Correct but far too slow for this budget.
2. `[info] resnet50 IMAGENET1K_V1 served from baked cache`
3. `Frame stride 4: ...`
4. `Group split fingerprint: ...`

**Red flag for both:** if you see `Filtered to weathers`, something is wrong — `train_weathers` should be unset.

Then watch `dlogvar_min` on MonoDETR's train lines (your divergence canary). If it walks steadily toward −10, flag it.

## The check that matters at the end

```bash
grep -h "Group split fingerprint" logs/box3d_*.out logs/monodetr_*.out | sort -u
```

Ideally **one unique `val=` hash**. Both models train on identical data, so they should hold out identical segments. More than one hash means they aren't comparable — report it, don't report the numbers.

Results land in `reports/final_box3d.json` and `reports/final_monodetr.json`.

---

# Part 2 — evaluate the MonoDETR run that already finished

Inference only. Run from `visuals-ml/`:

```bash
cd /blue/iruchkin/patel.rahi/visuals-dataset-generation/visuals-ml
python -m baselines.eval --config configs/monodetr_legacy_eval.yaml --report reports/final_monodetr_legacy.json
```

**Use `monodetr_legacy_eval.yaml`, not `monodetr.yaml`.** The completed run used the old row-level split. Evaluating it under the current segment split would build a *different* val set and score the model on its own training frames. The legacy config restores the original split via `split_mode: record` — you should see a warning saying exactly that. If you don't, you used the wrong config.

Then:

```bash
python -c "import json; m=json.load(open('reports/final_monodetr_legacy.json'))['metrics']; [print(f'{k:12s}', m.get(k)) for k in ('recall','depth_mae','center_mae','monitor')]"
```

**If `recall` is 0.0**, the model never detected anything above `score_thresh: 0.2` and nothing else in the report means anything. Report it, don't debug it.

⚠️ **These numbers are in-distribution fit, not generalization.** The index has 10 weather renderings of every frame and the row-level split scattered them across train and val, so nearly every val frame had siblings in training. Preliminary only.

---

# Part 3 — UQ wrapper shakedown

**Read first:** `visuals-ml/hipergator/RUN_INTROSPECTION.md` and `visuals-ml/baselines/README.md`.

The UQ wrapper is the Paper 1 introspection baseline (Daftry et al. 2016): a two-stream CNN plus a linear SVM that predicts *when the perception model will fail*, from the input alone.

### Scope — do not skip this

- This chain wraps **PositionNet**, not MonoDETR. The label pre-pass imports PositionNet directly (`introspection_label_prepass.py:44`). Wrapping MonoDETR needs code changes that aren't written yet.
- The PositionNet checkpoint came from a run with a leaky split, and introspection itself still uses the leaky `split_dataset`.
- **So this is a code shakedown and label characterization, not a result.** Nothing here gets reported. The goal is to find what breaks and whether the failure labels carry any signal — before we invest in wiring MonoDETR in.

## Step 1 — preflight

```bash
cd /blue/iruchkin/patel.rahi/visuals-dataset-generation/visuals-ml
ls -la baselines/checkpoints/positionnet/best.pt
python -c "import cv2; print('cv2', cv2.__version__, '| TVL1:', hasattr(cv2,'optflow'))"
```

If `TVL1: False`, the code silently falls back to Farneback — a different, worse flow engine. Worth reporting.

## Step 2 — build the index, and time it

```bash
time python -m baselines.data.build_introspection_index --source-dir /blue/iruchkin/patel.rahi/waymo/output --index-file data/output/introspection_records.jsonl --cameras 1 --flow-stack 5
```

```bash
wc -l data/output/introspection_records.jsonl
```

Metadata only, no images decoded — should be minutes. If it's hours, that's a finding.

## Step 3 — cut it down before spending GPU time

There is no `--max-segments` flag. Truncate the index directly, contiguously so frames stay temporally adjacent (the flow stream needs neighbours):

```bash
head -n 4000 data/output/introspection_records.jsonl > data/output/introspection_small.jsonl
```

## Step 4 — generate failure labels (GPU)

```bash
python -m baselines.data.introspection_label_prepass --index-file data/output/introspection_small.jsonl --checkpoint baselines/checkpoints/positionnet/best.pt --out-file data/output/introspection_labeled_small.jsonl --tau-percentile 50 --fail-thresh 0.5
```

**Most likely step to break.** If it fails loading the checkpoint, the fix is probably `weights_only=False` (torch 2.6 flipped the default — see commit `5e74dc8`). Report the exact traceback.

## Step 5 — characterize the labels (highest value, costs nothing)

**Do this before training anything.** Save as `check_labels.py` and run it:

```python
import json, collections, statistics as st
P = 'data/output/introspection_labeled_small.jsonl'
rs = [json.loads(l) for l in open(P, encoding='utf-8') if l.strip()]
ff = [r['fail_frac'] for r in rs]
print(f"frames: {len(rs)}  tau: {rs[0].get('tau'):.2f} m")
print(f"fail_frac: mean={st.mean(ff):.3f} median={st.median(ff):.3f} min={min(ff)} max={max(ff)}")
print(f"binary fail rate: {sum(r['fail'] for r in rs)/len(rs):.3f}   (0.0 or 1.0 == DEGENERATE)")
print(f"fail_frac exactly 0: {sum(1 for f in ff if f == 0)/len(ff):.1%}")
print(f"fail_frac exactly 1: {sum(1 for f in ff if f == 1)/len(ff):.1%}")
print("\nper weather (fail rate / mean err):")
by = collections.defaultdict(list)
for r in rs:
    by[r.get('weather', '?')].append(r)
for w in sorted(by):
    g = by[w]
    fail = sum(x['fail'] for x in g) / len(g)
    err = st.mean([x['mean_err'] for x in g])
    print(f"  {w:16s} n={len(g):5d}  fail={fail:.3f}  mean_err={err:.2f} m")
```

**Answer these three:**

1. Is the binary fail rate near 0.0 or 1.0? → labels are **degenerate**, the CNN has no negative class. Stop and report.
2. Is `fail_frac` mostly exactly 0 or exactly 1? → same problem in continuous form.
3. **Does fog / snow / dust fail more than clear?** If not, either the perception model isn't weather-sensitive or the labels are wrong. **This is the single most diagnostic number in the whole exercise.**

## Step 6 — the cheap control (before training the CNN)

Can weather alone predict failure? If a 10-way lookup matches the CNN, the wrapper isn't learning from pixels. Save as `check_control.py`:

```python
import json, collections, random
from sklearn.metrics import roc_auc_score
P = 'data/output/introspection_labeled_small.jsonl'
rs = [json.loads(l) for l in open(P, encoding='utf-8') if l.strip()]
random.Random(0).shuffle(rs)
k = int(0.8 * len(rs))
tr, va = rs[:k], rs[k:]
rate = collections.defaultdict(list)
for r in tr:
    rate[r.get('weather', '?')].append(r['fail'])
p = {w: sum(v) / len(v) for w, v in rate.items()}
base = sum(r['fail'] for r in tr) / len(tr)
y = [r['fail'] for r in va]
s = [p.get(r.get('weather', '?'), base) for r in va]
print(f"WEATHER-ONLY AUROC: {roc_auc_score(y, s):.4f}   (0.5 = no signal)")
print("This is the bar the two-stream CNN must beat to be worth anything.")
```

**Report that AUROC.** It sets the bar for everything downstream.

## Step 7 — train the wrapper (only if Steps 5–6 look sane)

Epoch 1 pays the entire optical-flow cost (TV-L1, ~5 fields per sample, cached to disk after). Cap it:

```bash
cp configs/introspection.yaml configs/introspection_small.yaml
```

Then append these four lines to `configs/introspection_small.yaml`:

```yaml
index_file: data/output/introspection_labeled_small.jsonl
max_samples: 2000
epochs: 5
num_workers: 4
```

```bash
time python -m baselines.train --config configs/introspection_small.yaml
```

**Report epoch 1 time vs epoch 2 time.** The gap is the flow cost being paid and cached — we need it to size a full run.

## Step 8 — the SVM stage

```bash
python -m baselines.introspection_svm --config configs/introspection_small.yaml
```

```bash
cat baselines/checkpoints/introspection/introspection_svm_report.json
```

---

# What to send back

**Files:**

```
visuals-ml/reports/final_box3d.json
visuals-ml/reports/final_monodetr.json
visuals-ml/reports/final_monodetr_legacy.json
visuals-ml/logs/box3d_<jobid>.out
visuals-ml/logs/monodetr_<jobid>.out
visuals-ml/baselines/checkpoints/introspection/introspection_svm_report.json
```

**Answers:**

1. The split fingerprints — one unique `val=` hash across box3d and MonoDETR, or not?
2. Label distribution from Part 3 Step 5 — **especially whether bad weather fails more than clear**
3. Weather-only AUROC from Step 6
4. Epoch 1 vs epoch 2 timing from Step 7
5. Anything that broke, with tracebacks

For Part 3, **don't fix anything beyond obvious import and checkpoint-loading errors.** We want the honest list of what's broken.
