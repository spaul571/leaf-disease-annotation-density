"""Numbers quoted in the v13 manuscript text that no other script writes.

The tables and figures come from build_paper_assets.py, build_table_dataset.py,
build_reviewer_stats.py, analyse_frequency.py and analyse_object_scale.py. This script adds the
smaller facts the prose cites (2026-10-21, manuscript revision to v13):

  - seed-to-seed behaviour per class on validation (classes that never move, the A-B gap bar);
  - per-class training-image and box counts, and the extremes the text names;
  - protocol mixing counted per image, as build_harmonized.py did on v12, with polygons read as
    their bounding boxes (labels.py);
  - where the v13 test images sat in v12, including the v12 validation images on which the final
    configuration was chosen before the rebuild;
  - training time and best epoch of every v13 run;
  - per-class gains at 1024 pixels.

    qpy manuscript_numbers_v13.py      # writes results/v13_manuscript_numbers.json
"""
import csv
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import paper_data as P
from classes import NAMES, group_of
from labels import read_boxes

D = P.dataset("v13")
OUT = P.RES / "v13_manuscript_numbers.json"
WHOLE_LEAF_MIN, LESION_MAX, MIXED_THRESHOLD = 0.50, 0.10, 0.10   # as build_harmonized.py
out = {}

# ---------- seed behaviour on validation ----------
seeds = [P.summarise(f)[1] for f in D.val_arms["baseline"]]
pc = pd.concat(seeds, axis=1)
swing = (pc.max(axis=1) - pc.min(axis=1)).round(4)
out["val_seed_scores_ALL"] = [round(s["ALL"], 4) for s in
                              (P.summarise(f)[0] for f in D.val_arms["baseline"])]
out["classes_identical_across_seeds"] = sorted(swing[swing == 0].index)
out["classes_swing_gt_0.02"] = {c: float(v) for c, v in swing[swing > 0.02].sort_values(
    ascending=False).items()}
S = P.val_per_seed(D)
gap_sd = float(S["A_B_gap"].std(ddof=1))
out["A_B_gap"] = {"mean": round(float(S["A_B_gap"].mean()), 4), "sd": round(gap_sd, 5),
                  "bar": round(1.96 * np.sqrt(2) * gap_sd, 4),
                  "ratio": round(float(S["A_B_gap"].mean()) / (1.96 * np.sqrt(2) * gap_sd), 1)}

# ---------- class composition ----------
dist = pd.read_csv(D.dist).set_index("class")
prot = pd.read_csv(D.protocol).set_index("class")
base = pc.mean(axis=1)
out["train_images"] = {"min": [dist.train_imgs.idxmin(), int(dist.train_imgs.min())],
                       "max": [dist.train_imgs.idxmax(), int(dist.train_imgs.max())],
                       "ratio": round(dist.train_imgs.max() / dist.train_imgs.min(), 1)}
nbox = dist[["train_boxes", "valid_boxes", "test_boxes"]].sum(axis=1)
out["boxes"] = {"min": [nbox.idxmin(), int(nbox.min())], "max": [nbox.idxmax(), int(nbox.max())],
                "ratio": round(nbox.max() / nbox.min(), 1), "total": int(nbox.sum())}
out["val_boxes_lt20"] = sorted(dist.index[dist.valid_boxes < 20])
lesion = [c for c in NAMES if group_of(c) == "B_lesion"]
out["lesion_boxes"] = {"valid": int(dist.loc[lesion, "valid_boxes"].sum()),
                       "test": int(dist.loc[lesion, "test_boxes"].sum()),
                       "test_per_class": {c: int(dist.loc[c, "test_boxes"]) for c in lesion}}
dens = prot["boxes_per_img"]
one = dens[(dens >= 1.0) & (dens <= 1.02)].index
out["classes_at_1.00_1.02"] = {"n": len(one), "map50_min": round(float(base[one].min()), 3),
                               "map50_max": round(float(base[one].max()), 3),
                               "above_0.98": int((base[one] > 0.98).sum()),
                               "lowest": base[one].idxmin()}
out["map50_range"] = {"max": round(float(base.max()), 3), "min": round(float(base.min()), 3),
                      "diff": round(float(base.max() - base.min()), 3)}
for g in ("A_whole_leaf", "B_lesion"):
    m = prot.loc[[c for c in NAMES if group_of(c) == g], "median_area"]
    out[f"median_area_{g}"] = [round(float(m.min()), 3), round(float(m.max()), 3)]
fewer = dist.index[dist.train_imgs < dist.loc["potato_blackleg", "train_imgs"]]
out["fewer_train_images_than_potato_blackleg"] = {
    "n": len(fewer), "blackleg": int(dist.loc["potato_blackleg", "train_imgs"]),
    "range": [int(dist.loc[fewer, "train_imgs"].min()), int(dist.loc[fewer, "train_imgs"].max())],
    "map50": [round(float(base[fewer].min()), 3), round(float(base[fewer].max()), 3)]}

# ---------- protocol mixing per image (train split), as build_harmonized.py ----------
def img_protocol(areas):
    if any(a >= WHOLE_LEAF_MIN for a in areas):
        return "whole"
    if all(a <= LESION_MAX for a in areas):
        return "lesion"
    return "mid"


counts = defaultdict(Counter)
per_img = []
box_counts = defaultdict(Counter)
for lbl in sorted((D.root / "labels" / "train").glob("*.txt")):
    by_cls = defaultdict(list)
    for c, _, _, w, h in read_boxes(lbl):
        by_cls[c].append(w * h)
        a = w * h
        box_counts[c]["whole" if a >= WHOLE_LEAF_MIN else "lesion" if a <= LESION_MAX else "mid"] += 1
    prots = {c: img_protocol(v) for c, v in by_cls.items()}
    per_img.append(prots)
    for c, m in prots.items():
        counts[c][m] += 1
mix = {}
targets = {}
for c, d in counts.items():
    tot = sum(d.values())
    fw, fl = d["whole"] / tot, d["lesion"] / tot
    mix[NAMES[c]] = {"images": tot, "whole": round(fw, 3), "lesion": round(fl, 3),
                     "mid": round(d["mid"] / tot, 3), "mixed": round(min(fw, fl), 3)}
    if min(fw, fl) >= MIXED_THRESHOLD:
        targets[c] = "whole" if fw >= fl else "lesion"
changed = sum(1 for prots in per_img
              if any(c in targets and m in ("whole", "lesion") and m != targets[c]
                     for c, m in prots.items()))
out["mixing_per_image"] = {"classes_ge_0.10": sorted(NAMES[c] for c in targets),
                           "train_images_changed_by_harmonising": changed,
                           "train_images": len(per_img), "per_class": mix}
ca = NAMES.index("cucumber_Antraknosa")
bc = box_counts[ca]
out["cucumber_Antraknosa_by_box"] = {k: round(v / sum(bc.values()), 3) for k, v in bc.items()}

# ---------- v13 test images and their v12 split ----------
with open(P.JA / "v13_manifest.csv", newline="", encoding="utf-8") as f:
    man = list(csv.DictReader(f))
out["v13_split_by_v12_split"] = {s: dict(Counter(r["v12_split"] for r in man if r["split"] == s))
                                 for s in ("train", "valid", "test")}
g = Counter(int(r["twin_group_size"]) for r in man)
out["twin_groups"] = {"kept_images_from_groups_gt1": sum(n for k, n in g.items() if k > 1),
                      "images_removed": sum((k - 1) * n for k, n in g.items())}

# ---------- training time and best epoch ----------
runs = {}
for r in sorted((P.HERE / "runs").glob("v13_*_W2")):
    rc = r / "results.csv"
    if not rc.exists():
        continue
    df = pd.read_csv(rc)
    df.columns = [c.strip() for c in df.columns]
    fit = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    runs[r.name] = {"epochs": int(df["epoch"].max()), "best_epoch": int(df.loc[fit.idxmax(), "epoch"]),
                    "hours": round(float(df["time"].max()) / 3600, 2),
                    "hours_per_100_epochs": round(float(df["time"].max()) / 3600
                                                  * 100 / int(df["epoch"].max()), 2)}
out["runs"] = runs

# ---------- per-class gains at 1024 pixels ----------
for arm in ("n @1024", "s @1024"):
    g = pd.concat([P.summarise(f)[1] for f in D.val_arms[arm]], axis=1).mean(axis=1) - base
    out[f"gain_{arm}"] = {"classes_moving_lt_0.0005": int((g.abs() < 0.0005).sum()),
                          "potato_blackleg": [round(float(base["potato_blackleg"]), 3),
                                              round(float(base["potato_blackleg"] + g["potato_blackleg"]), 3)]}

OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "mixing_per_image"}, indent=1))
print("mixing per image:", {k: v for k, v in out["mixing_per_image"].items() if k != "per_class"})
print(f"wrote {OUT}")
