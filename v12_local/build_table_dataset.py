"""Table 1 for the manuscript: per-class dataset composition + annotation statistics.

Pure join of material that already exists - no new measurement:
  * image / box counts per split      <- journal_analysis/<dataset>_distribution.csv
  * density, median box area, group   <- results/<dataset>_protocol_analysis.csv
  * baseline per-class mAP@50         <- mean of the three local baseline seed CSVs
                                         (the same 3-seed mean the density law is fitted on)

Also re-prints the density-law correlations so the numbers quoted in the manuscript can be
checked against a single command.

    qpy build_table_dataset.py                 # v13 -> paper/table_dataset.csv
    qpy build_table_dataset.py --dataset v12   # v12 -> paper_v12/table_dataset.csv
"""
import argparse

import pandas as pd
from scipy import stats

import paper_data as P

D = P.dataset(P.add_dataset_arg(argparse.ArgumentParser()).parse_args().dataset)
D.out.mkdir(exist_ok=True)

dist = pd.read_csv(D.dist)
dist["images"] = dist.train_imgs + dist.valid_imgs + dist.test_imgs
dist["boxes"] = dist.train_boxes + dist.valid_boxes + dist.test_boxes

prot = pd.read_csv(D.protocol)

pc = pd.concat([P.perclass(f)[1] for f in D.val_arms["baseline"]])
pc = pc.groupby("class", as_index=False)[["mAP50", "recall"]].mean()

t = (dist[["class", "images", "boxes", "train_imgs", "valid_imgs", "test_imgs"]]
     .merge(prot[["class", "group", "boxes_per_img", "median_area", "mixed_frac"]], on="class")
     .merge(pc, on="class"))
t["crop"] = t["class"].str.split("_").str[0]
t = t.sort_values(["group", "boxes_per_img"], ascending=[True, False])
t = t[["class", "crop", "group", "images", "boxes", "boxes_per_img", "median_area",
       "mixed_frac", "mAP50", "recall"]].round(4)
t.to_csv(D.out / "table_dataset.csv", index=False)
print(t.to_string(index=False))

# Appendix table (Shakib S-5): per-split images and boxes per class, in Table 1's row order.
splits = dist.set_index("class").loc[t["class"], ["train_imgs", "valid_imgs", "test_imgs",
                                                   "train_boxes", "valid_boxes", "test_boxes"]]
splits.insert(0, "group", t.set_index("class")["group"])
splits.to_csv(D.out / "tableA2_splits.csv")
print("\nper-split counts:\n" + splits.to_string())
# NOTE: the per-class `images` column counts images CONTAINING at least one box of that class,
# so summing it over classes double-counts the few multi-class images. The dataset totals are
# the on-disk file counts, not this sum.
n = P.images_on_disk(D)
print(f"\n{D.root.name}: {n['train']:,} / {n['valid']:,} / {n['test']:,} = {sum(n.values()):,} "
      f"images on disk; sum of per-class image counts {t.images.sum()} (multi-class images "
      f"counted once per class); boxes {t.boxes.sum()}; classes {len(t)}")
b = t.set_index("class")["boxes"]
print(f"box-level class imbalance ratio: {b.max() / b.min():.1f}x  "
      f"(max {b.idxmax()} {b.max()}, min {b.idxmin()} {b.min()})")
print("groups: " + ", ".join(f"{g} {k}" for g, k in t.group.value_counts().sort_index().items()))

print("\n--- density law (n = %d classes, baseline 3-seed mean, validation) ---" % len(t))
for name, col in (("boxes per image", "boxes_per_img"),
                  ("median box area", "median_area"),
                  ("within-class protocol mixing", "mixed_frac")):
    rp, pp = stats.pearsonr(t[col], t.mAP50)
    rs, ps = stats.spearmanr(t[col], t.mAP50)
    print(f"{name:30s} Pearson {rp:+.3f} (p={pp:.2e})   Spearman {rs:+.3f} (p={ps:.2e})")

print("\n--- who gains from 1024? Delta(1024 - 640) per class, vs the 3-seed baseline mean ---")
lo = pc.set_index("class")["mAP50"]
for tag, arm in (("resolution only: n @1024", "n @1024"),
                 ("resolution + capacity: s @1024 (3-seed)", "s @1024")):
    files = [f for f in D.val_arms[arm] if f.exists()]
    if not files:
        print(f"{tag:48s} not run yet")
        continue
    hi = pd.concat([P.perclass(f)[1] for f in files]).groupby("class")["mAP50"].mean()
    d = (hi - lo).dropna()
    d = d[d.index.isin(t["class"])]
    j = t.set_index("class").loc[d.index]
    for name, col in (("boxes per image", "boxes_per_img"), ("median box area", "median_area")):
        rp, pp = stats.pearsonr(j[col], d)
        print(f"{tag:48s} vs {name:16s} r {rp:+.3f} (p={pp:.3f})")
    print(f"{'':48s} potato_blackleg {lo['potato_blackleg']:.3f} -> {hi['potato_blackleg']:.3f}, "
          f"n classes moving 0.000: {(d.abs() < 5e-4).sum()}")
