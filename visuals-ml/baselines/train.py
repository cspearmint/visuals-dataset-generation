"""
Harness training entrypoint. Model-agnostic — picks the baseline from the
config's `model:` key.

Run from the visuals-ml/ directory:
    python -m baselines.train --config configs/positionnet.yaml
"""

import argparse

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import baselines.models  # noqa: F401  (registers all baselines)
from baselines.core.registry import build_model
from baselines.core.runner import train
from baselines.core.utils import apply_index_suffix, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--index-suffix", default=None,
                        help="Insert '_<suffix>' before .jsonl in index_file, so "
                             "smoke/full/full-eval indexes coexist without "
                             "separate configs.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Override the config's checkpoint_dir.")
    parser.add_argument("--train-weathers", default=None,
                        help="Weather ablation. 'all' (default), a comma list "
                             "like 'clear,rain,fog', or 'randomN' for clear plus "
                             "N alterations drawn with the config seed. Applied "
                             "to the TRAIN side only -- validation always keeps "
                             "all 10 variants so ablations stay comparable.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_index_suffix(cfg, args.index_suffix)
    if args.checkpoint_dir:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.train_weathers:
        cfg["train_weathers"] = (
            None if args.train_weathers == "all" else args.train_weathers)

    # cfg['seed'] previously only seeded the train/val split (a local
    # random.Random in core/utils.split_dataset); torch's global RNG was left
    # unseeded, so weight init AND DataLoader shuffle order differed on every
    # run. That makes a divergence impossible to reproduce or bisect -- two
    # runs can't be compared when neither the starting weights nor the batch
    # order match. Seed torch here, before build_model() initialises weights.
    seed = cfg.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Seeded torch RNG with {seed} (weight init + shuffle order now "
              "reproducible across runs)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  model: {cfg['model']}")

    model = build_model(cfg)
    train(model, cfg, device, resume=args.resume)


if __name__ == "__main__":
    main()
