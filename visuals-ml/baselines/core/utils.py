"""Shared helpers used across baselines."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml
from torch.utils.data import Subset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_index_suffix(cfg: dict, suffix: str | None) -> dict:
    """Rewrite cfg['index_file'] to carry a suffix before its extension.

    Lets one config serve the smoke / full / full-eval indexes:
        data/output/det_records.jsonl -> data/output/det_records_full.jsonl
    Returns cfg unchanged when suffix is falsy.
    """
    if not suffix or not cfg.get("index_file"):
        return cfg
    p = Path(cfg["index_file"])
    cfg = dict(cfg)
    cfg["index_file"] = str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))
    print(f"index_file -> {cfg['index_file']}")
    return cfg


def split_dataset(dataset, val_fraction: float, seed: int):
    """Deterministic train/val split by shuffling indices with a fixed seed.

    Matches the convention used by the original visuals-ml train.py/eval.py so a
    model re-registered under the harness sees the same split it always did.

    WARNING: this splits on individual records and therefore LEAKS on the visuals
    dataset, where each physical object appears once per weather variant (10x)
    and again in every neighbouring ~10 Hz frame. Use split_by_group() with the
    segment id for any number you intend to report.
    """
    n = len(dataset)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    split = int(n * (1 - val_fraction))
    return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])


def split_by_group(dataset, groups, val_fraction: float, seed: int):
    """Train/val split that keeps every sample of a group on one side.

    `groups[i]` is the group key of sample i (use the segment id). Whole groups
    are assigned to val until the val fraction is reached, so a frame's clear /
    rain / fog renderings — and its temporal neighbours — can never straddle the
    split. Group sizes differ, so the realised fraction is approximate.

    Returns (train_subset, val_subset).
    """
    if len(groups) != len(dataset):
        raise ValueError(
            f"groups has {len(groups)} entries but dataset has {len(dataset)}"
        )

    by_group = {}
    for idx, key in enumerate(groups):
        by_group.setdefault(key, []).append(idx)

    keys = sorted(by_group)
    random.Random(seed).shuffle(keys)

    target_val = len(dataset) * val_fraction
    val_idx, train_idx, n_val = [], [], 0
    for key in keys:
        members = by_group[key]
        if n_val < target_val:
            val_idx.extend(members)
            n_val += len(members)
        else:
            train_idx.extend(members)

    if not train_idx or not val_idx:
        raise ValueError(
            f"Group split degenerate: {len(keys)} group(s) gave "
            f"{len(train_idx)} train / {len(val_idx)} val samples. "
            "Need at least 2 groups (segments) to split on."
        )

    train_idx.sort()
    val_idx.sort()

    # Fingerprint the GROUP SET, not the sample count. The split is derived by
    # shuffling sorted(keys), so if training and evaluation are run over indexes
    # whose segment sets differ by even one segment, the shuffle diverges and
    # the "held-out" set silently fills with segments the model trained on.
    # Measured: dropping a single segment from an 800-segment index moved 82 of
    # 160 val segments into that category. The counts look normal either way --
    # this hash is the only thing that catches it, so grep it in both logs.
    fp = hashlib.sha1("|".join(keys).encode()).hexdigest()[:12]
    val_fp = hashlib.sha1(
        "|".join(sorted({groups[i] for i in val_idx})).encode()
    ).hexdigest()[:12]
    print(f"Group split: {len(keys)} groups -> "
          f"{len(train_idx)} train / {len(val_idx)} val samples "
          f"({len(val_idx) / len(dataset):.1%} val)")
    print(f"Group split fingerprint: groups={fp} val={val_fp} "
          f"({len(set(groups[i] for i in val_idx))} val groups) — "
          "must match between the training run and the final eval")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def subsample_frames(records, stride: int):
    """Keep every Nth FRAME within each (segment, camera), with ALL of its
    weather variants.

    The right knob for cutting compute on this dataset. Frames arrive at ~10 Hz,
    so neighbours are near-duplicates and dropping them costs little. The 10
    weather renderings of a kept frame are all retained: they are the axis the
    experiment is about, and the downstream visuals agent trains on all of them,
    so the baselines must too.

    Decimation is per (segment, camera) and ordered by frame stem, so it is a
    genuine temporal thinning rather than an arbitrary slice of the file order.
    """
    if not stride or stride <= 1:
        return records

    by_group = {}
    for r in records:
        by_group.setdefault((r.get("segment"), r.get("camera")), set()).add(r.get("stem"))

    keep = set()
    for (seg, cam), stems in by_group.items():
        for stem in sorted(stems)[::stride]:
            keep.add((seg, cam, stem))

    out = [r for r in records
           if (r.get("segment"), r.get("camera"), r.get("stem")) in keep]
    n_frames = len(keep)
    print(f"Frame stride {stride}: {len(records)} -> {len(out)} records "
          f"({n_frames} frames x all weathers)")
    return out


WEATHER_VARIANTS = [
    "clear", "rain", "fog", "snow", "frost",
    "sunglare", "brightness", "wildfire_smoke", "dust", "waterdrop",
]
ALTERATIONS = [w for w in WEATHER_VARIANTS if w != "clear"]


def resolve_train_weathers(spec, seed: int):
    """Turn a train-weather spec into an explicit list, or None for 'all'.

    Accepts:
      None / "all"        -> None (every variant; the default run)
      list of names       -> that list, validated
      "clear"             -> ["clear"]
      "clear,rain,fog"    -> those three
      "random<N>"         -> clear + N alterations drawn WITHOUT replacement from
                             the 9 non-clear variants, seeded so the draw is
                             reproducible from (spec, seed) alone.

    The chosen set is returned sorted; callers print it so the run's log records
    exactly which variants trained, including for a random draw.
    """
    if spec is None or spec == "all":
        return None

    if isinstance(spec, str):
        spec = spec.strip()
        if spec.startswith("random"):
            n = int(spec[len("random"):] or 0)
            if not 1 <= n <= len(ALTERATIONS):
                raise ValueError(
                    f"random{n}: N must be 1..{len(ALTERATIONS)} (the alteration pile)"
                )
            drawn = random.Random(seed).sample(ALTERATIONS, n)
            return sorted(["clear"] + drawn)
        spec = [w.strip() for w in spec.split(",") if w.strip()]

    spec = list(spec)
    unknown = set(spec) - set(WEATHER_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown weather variant(s): {sorted(unknown)}")
    return sorted(spec)


def filter_train_subset(dataset, train_subset, weathers_of, keep):
    """Restrict a TRAIN Subset to samples whose weather is in `keep`.

    Applied AFTER the split, on the train side only, so validation keeps all 10
    variants for every ablation. That is what makes the ablations comparable:
    each one trains on a different weather subset but is scored on the same
    all-weather held-out segments. Filtering before the split would give each
    variant its own test set and the numbers could not be compared.

    `weathers_of` maps a dataset index -> weather string.
    """
    from torch.utils.data import Subset

    if keep is None:
        return train_subset
    keep = set(keep)
    idx = [i for i in train_subset.indices if weathers_of(i) in keep]
    if not idx:
        raise ValueError(
            f"No training samples left after restricting to {sorted(keep)}. "
            "Check the weather names against what the index actually contains."
        )
    dropped = len(train_subset.indices) - len(idx)
    print(f"Train weathers {sorted(keep)}: kept {len(idx)} / "
          f"{len(train_subset.indices)} train samples (dropped {dropped}). "
          "Validation keeps ALL variants.")
    return Subset(dataset, idx)
