"""Class frequency and imbalance: where it does and does not matter.

1. Imbalance ratios at box and image level, and per-split instance counts.
2. Which classes each imbalance arm moved (Environment 1, single seed each, against the
   Environment 1 baseline): the classes the resampling floor duplicated vs the rest, the classes
   naive resampling cut vs the rest, and per-class deltas correlated with density and frequency.
3. Frequency as a source of measurement noise: per-class seed spread across the three
   Environment 2 baseline seeds against the number of validation boxes.

Parts 1 and 3 use the chosen dataset (v13 by default). Part 2 is always v12: the Environment 1
arms were trained on v12, so the classes they floored or cut, the densities they are correlated
with and the bars they are judged against all come from v12. Groups come from classes.py.
Since 2026-11-01 (Shakib S-1) the same arms retrained on v13 are analysed too ("v13_arms"),
against the three-seed v13 baseline mean, once their per-class CSVs exist. Every bar is the exact
t-based bar of paper_data.bar (Shakib S-4).

Run: qpy -I analyse_frequency.py [--dataset v12]
Writes: results/v13_frequency_analysis.json or results/v12_frequency_analysis.json
"""
import argparse
import csv
import json
import math

from scipy import stats

import paper_data as P
from classes import group_of

FLOOR = 400  # images per primary class, used by both resampling arms
D = P.dataset(P.add_dataset_arg(argparse.ArgumentParser()).parse_args().dataset)
V12 = P.dataset("v12")


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def corr(x, y):
    r, p = stats.pearsonr(x, y)
    rho, prho = stats.spearmanr(x, y)
    return {"pearson": round(r, 3), "p": round(p, 4), "spearman": round(rho, 3), "p_s": round(prho, 4)}


def mean(xs):
    xs = list(xs)
    return round(sum(xs) / len(xs), 4)


def load(ds):
    """Per-class counts, density and baseline seeds of one dataset."""
    dist = {r["class"]: r for r in read(ds.dist)}
    proto = {r["class"]: r for r in read(ds.protocol)}
    seeds = [{r["class"]: r for r in read(f)} for f in ds.val_arms["baseline"]]
    return dist, proto, seeds


dist, proto, seeds = load(D)
classes = list(dist)
group = {c: group_of(c) for c in classes}
density = {c: float(proto[c]["boxes_per_img"]) for c in classes}
n_boxes = {c: int(proto[c]["n_boxes"]) for c in classes}
train_imgs = {c: int(dist[c]["train_imgs"]) for c in classes}
val_boxes = {c: int(dist[c]["valid_boxes"]) for c in classes}
test_boxes = {c: int(dist[c]["test_boxes"]) for c in classes}

out = {
    "dataset": D.name,
    "box_imbalance": round(max(n_boxes.values()) / min(n_boxes.values()), 1),
    "train_image_imbalance": round(max(train_imgs.values()) / min(train_imgs.values()), 1),
    "lesion_boxes": {s: sum(int(dist[c][f"{s}_boxes"]) for c in classes if group[c] == "B_lesion")
                     for s in ("train", "valid", "test")},
    "lesion_test_boxes_per_class": {c: test_boxes[c] for c in classes if group[c] == "B_lesion"},
}

# Environment 2 three-seed baseline: per-class mean and spread.
base_mean = {c: sum(float(s[c]["mAP50"]) for s in seeds) / len(seeds) for c in classes}
swing = {c: max(float(s[c]["mAP50"]) for s in seeds) - min(float(s[c]["mAP50"]) for s in seeds)
         for c in classes}
rec_swing = {c: max(float(s[c]["recall"]) for s in seeds) - min(float(s[c]["recall"]) for s in seeds)
             for c in classes}

fewer_than_blackleg = [c for c in classes if train_imgs[c] < train_imgs["potato_blackleg"]]
out["classes_with_fewer_train_images_than_potato_blackleg"] = {
    "n": len(fewer_than_blackleg),
    "train_images_range": [min(train_imgs[c] for c in fewer_than_blackleg),
                           max(train_imgs[c] for c in fewer_than_blackleg)],
    "map50_range": [round(min(base_mean[c] for c in fewer_than_blackleg), 3),
                    round(max(base_mean[c] for c in fewer_than_blackleg), 3)],
}
out["frequency_vs_map50"] = corr([math.log(n_boxes[c]) for c in classes], [base_mean[c] for c in classes])

# Noise: classes whose mAP@50 range across seeds exceeds 0.02.
out["unstable_classes"] = {c: {"swing": round(swing[c], 4), "val_boxes": val_boxes[c],
                               "density": density[c]} for c in classes if swing[c] > 0.02}
nonsat = [c for c in classes if base_mean[c] < 0.98]  # saturated classes cannot swing
out["swing_vs_log_val_boxes_nonsaturated"] = corr([math.log(val_boxes[c]) for c in nonsat],
                                                  [swing[c] for c in nonsat])
out["swing_vs_density_nonsaturated"] = corr([density[c] for c in nonsat], [swing[c] for c in nonsat])
out["n_nonsaturated"] = len(nonsat)
out["potato_blackleg_seed_range"] = {"map50": round(swing["potato_blackleg"], 4),
                                     "recall": round(rec_swing["potato_blackleg"], 4)}
out["swing_by_threshold"] = {
    str(t): {"vs_log_val_boxes": corr([math.log(val_boxes[c]) for c in classes if base_mean[c] < t],
                                      [swing[c] for c in classes if base_mean[c] < t]),
             "vs_density": corr([density[c] for c in classes if base_mean[c] < t],
                                [swing[c] for c in classes if base_mean[c] < t]),
             "n": sum(base_mean[c] < t for c in classes)}
    for t in (0.98, 0.99, 1.01)}
pb = [float(s["potato_blackleg"]["mAP50"]) for s in seeds]
out["potato_blackleg_seeds_map50"] = [round(v, 4) for v in pb]
out["potato_blackleg_sd_map50"] = round(stats.tstd(pb), 4)

def imbalance_arms(ds, ref, n_ref, files):
    """Which classes each imbalance arm moved, against a reference of n_ref runs.

    ref: {class: {"mAP50", "recall"}} of the reference (a single run, or the baseline-seed mean).
    Group bars are the exact t-based bars of Table 2 (Shakib S-4): the group's mean mAP@50 in each
    baseline seed, its sd (n-1), then t(0.975, n-1) * sd * sqrt(1 + 1/n_ref).
    """
    dist_, proto_, seeds_ = load(ds)
    cs = list(dist_)
    dens = {c: float(proto_[c]["boxes_per_img"]) for c in cs}
    nb = {c: int(proto_[c]["n_boxes"]) for c in cs}
    ti = {c: int(dist_[c]["train_imgs"]) for c in cs}
    floored = [c for c in cs if ti[c] < FLOOR]   # raised by both resampling arms
    cut = [c for c in cs if ti[c] > FLOOR]        # cut down by naive only

    def bar(group):
        per_seed = [sum(float(s[c]["mAP50"]) for c in group) / len(group) for s in seeds_]
        return round(P.bar(stats.tstd(per_seed), len(seeds_), 1, n_ref), 4)

    res = {"floored_classes": len(floored), "cut_classes": len(cut),
           "group_bars": {"floored": bar(floored), "not_floored": bar([c for c in cs if c not in floored]),
                          "lesion": bar([c for c in cs if group_of(c) == "B_lesion"]),
                          "wholeleaf": bar([c for c in cs if group_of(c) == "A_whole_leaf"]),
                          "cut": bar(cut)}}
    for arm, path in files.items():
        if not path.exists():
            print(f"skip {arm}: {path.name} not written yet")
            continue
        rows = {r["class"]: r for r in read(path)}
        d = {c: float(rows[c]["mAP50"]) - ref[c]["mAP50"] for c in cs}
        res[arm] = {
            "mean_delta_floored": mean(d[c] for c in floored),
            "mean_delta_not_floored": mean(d[c] for c in cs if c not in floored),
            "mean_delta_cut": mean(d[c] for c in cut),
            "mean_delta_lesion": mean(d[c] for c in cs if group_of(c) == "B_lesion"),
            "mean_delta_wholeleaf": mean(d[c] for c in cs if group_of(c) == "A_whole_leaf"),
            "delta_vs_density": corr([dens[c] for c in cs], [d[c] for c in cs]),
            "delta_vs_log_frequency": corr([math.log(nb[c]) for c in cs], [d[c] for c in cs]),
            "potato_blackleg": {"map50": [round(ref["potato_blackleg"]["mAP50"], 4),
                                          round(float(rows["potato_blackleg"]["mAP50"]), 4)],
                                "recall": [round(ref["potato_blackleg"]["recall"], 4),
                                           round(float(rows["potato_blackleg"]["recall"]), 4)]},
            "cut_classes": {c: [ti[c], round(d[c], 4)] for c in cut},
            "worst5": sorted(((c, round(d[c], 4)) for c in cs), key=lambda kv: kv[1])[:5],
            "best3": sorted(((c, round(d[c], 4)) for c in cs), key=lambda kv: -kv[1])[:3],
        }
    return res


# Environment 1 imbalance arms, on v12, against the single-run Environment 1 baseline.
env1 = {r["class"]: {"mAP50": float(r["mAP50"]), "recall": float(r["recall"])}
        for r in read(P.KAG / "out6b" / "perclass_v12_100ep.csv")}
out["env1_v12"] = imbalance_arms(V12, env1, 1, {
    "oversample": P.KAG / "out7/perclass_v12_oversample.csv",
    "naive": P.KAG / "out14/perclass_v12_naive.csv",
    "focal": P.KAG / "out13/perclass_v12_focal.csv"})

# The same arms retrained on v13 (Shakib S-1), against the three-seed v13 baseline mean.
if D.name == "v13":
    base_ref = {c: {"mAP50": base_mean[c],
                    "recall": sum(float(s[c]["recall"]) for s in seeds) / len(seeds)} for c in classes}
    out["v13_arms"] = imbalance_arms(D, base_ref, len(seeds), {
        a: P.RES / f"perclass_v13_{a}_100ep_s0_W2.csv" for a in ("oversample", "naive", "focal")})

path = P.RES / f"{D.tag}frequency_analysis.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
print("wrote", path)
