"""
diagnose_type_leak.py

Iterates the dataset directly (NOT through the DataLoader/collate_fn), so
we catch type problems (ndarray vs. tensor/MetaTensor) per-sample, per-key,
BEFORE collation ever gets a chance to crash on a mixed-type batch. This
tells you exactly which key, on which bdmap id pair, comes out as a plain
ndarray -- which transform produced it is then inferable from which key
it is (e.g. "tumor_mask_1" pointing at CropForegroundd/RandCropByLabelClassesd,
"heatmap" pointing at GenerateTumorHeatmapd).

Also fetches ct0_bdmap/ct1_bdmap directly from the sample dict when
available, since MapTransform passthrough should preserve these unless a
transform explicitly deletes them.
"""

import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
from omegaconf import DictConfig, open_dict
import hydra

from dataset.dataloader import get_loader


NUM_SAMPLES = None  # set to an int to limit; None = full dataset
EXPECTED_KEYS = ["tumor_mask_0", "tumor_mask_1", "organ_mask_0", "organ_mask_1", "heatmap"]


def get_ids(sample):
    ct0 = sample.get("ct0_bdmap", "<missing>")
    ct1 = sample.get("ct1_bdmap", "<missing>")
    organ = sample.get("organ", "<missing>")
    delta_t = sample.get("delta_t", "<missing>")
    return ct0, ct1, organ, delta_t


def check_types(sample, idx):
    """Returns a dict {key: (type_name, shape_or_None)} for each expected key,
    and a list of keys that are NOT torch.Tensor/MetaTensor (the leak)."""
    results = {}
    bad_keys = []
    for key in EXPECTED_KEYS:
        if key not in sample:
            results[key] = ("<MISSING KEY>", None)
            continue
        val = sample[key]
        type_name = type(val).__name__
        shape = tuple(val.shape) if hasattr(val, "shape") else None
        results[key] = (type_name, shape)
        if not torch.is_tensor(val):
            bad_keys.append(key)
    return results, bad_keys


@hydra.main(config_path='config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    with open_dict(cfg):
        cfg.dataset.phase = "train"

    # get_loader returns (dataloader, sampler, dataset_len) -- we want the
    # underlying dataset object, not the DataLoader, so we bypass collate.
    dataloader, _, dataset_len = get_loader(cfg.dataset)
    dataset = dataloader.dataset  # the stochastic-wrapped Dataset (outer layer)

    n = dataset_len if NUM_SAMPLES is None else min(NUM_SAMPLES, dataset_len)
    print(f"Iterating {n} samples directly from the dataset (bypassing DataLoader/collate)...\n")

    total = 0
    leak_count = 0
    leak_records = []
    error_count = 0

    for i in range(n):
        try:
            sample = dataset[i]
        except Exception as e:
            error_count += 1
            print(f"[ERROR at index {i}] {type(e).__name__}: {e}")
            continue

        # RandCropByLabelClassesd with num_samples > 1 returns a LIST of
        # dicts (one per crop) instead of a single dict -- handle both.
        samples = sample if isinstance(sample, list) else [sample]

        for sub_idx, s in enumerate(samples):
            total += 1
            ct0, ct1, organ, delta_t = get_ids(s)
            results, bad_keys = check_types(s, i)

            if bad_keys:
                leak_count += 1
                print(f"[TYPE LEAK] idx={i} sub={sub_idx}  ct0={ct0}  ct1={ct1}  "
                      f"organ={organ}  delta_t={delta_t}")
                for key in EXPECTED_KEYS:
                    type_name, shape = results[key]
                    marker = " <-- LEAK" if key in bad_keys else ""
                    print(f"    {key:16s} type={type_name:20s} shape={shape}{marker}")
                leak_records.append({
                    "idx": i, "sub_idx": sub_idx, "ct0_bdmap": ct0, "ct1_bdmap": ct1,
                    "organ": organ, "delta_t": delta_t, "bad_keys": ",".join(bad_keys),
                })

        if i % 25 == 0:
            print(f"--- {i}/{n} checked, {leak_count} leaks so far ---")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples checked: {total}")
    print(f"Load errors:            {error_count}")
    print(f"Type leaks found:       {leak_count} ({100 * leak_count / max(total, 1):.2f}%)")

    if leak_records:
        import pandas as pd
        df = pd.DataFrame(leak_records)
        out_path = "type_leak_log.csv"
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} rows to {out_path}")

        print("\nBreakdown of which key(s) leak most often:")
        all_bad_keys = [k for row in leak_records for k in row["bad_keys"].split(",")]
        from collections import Counter
        for key, count in Counter(all_bad_keys).most_common():
            print(f"  {key}: {count}")

        print("\nBreakdown by organ:")
        print(df["organ"].value_counts())


if __name__ == '__main__':
    run()