"""Measure annotation-protocol consistency inside each class, and assign Groups A and B.

The campaign's thesis is that Group A (whole-leaf boxes) is easy and Group B (lesion boxes) is
hard. The actionable question for a data-centric fix is narrower: is the protocol consistent
WITHIN a class? A class that contains both whole-leaf-sized boxes and small lesion boxes is
giving the detector contradictory supervision for the same label, which no architecture can
resolve - and which relabelling CAN.

Note the 1,735 "conflicting label groups" from build_v12.py are already resolved in v12: each
near-duplicate group was collapsed to its most-annotated representative. So they cannot be
re-fixed; what can be fixed is systematic within-class protocol mixing.

Signals per class:
  area_frac   - box area as a fraction of image area (whole-leaf ~ >0.5, lesion ~ <0.1)
  bimodality  - fraction of boxes that fall in the OPPOSITE mode from the class majority
  n_per_img   - boxes per image (whole-leaf ~ 1, lesion ~ many)

Groups (rule chosen by the user on 2026-10-21). A class is whole-leaf (Group A) when its median box
covers at least half the image; otherwise it is lesion (Group B). The split does not depend on the
exact threshold: any value from 0.26 to 0.67 gives the same groups on v12 and on v13. The rule
counts boxes, as the density law does.

An alternative rule compares the share of whole-leaf boxes with the share of lesion boxes. It was
rejected because it would also move both bean classes to Group A. Most of their boxes (70-82%) are
mid-sized, so that rule decides them on 12-18% of their boxes.

Until 2026-10-21 the groups were a hand-written list. The median rule changes only one class:
potato_healthy, which polygon labels read as boxes had made look like a lesion class. The old
list is kept as `group_v12_paper`, so every change of group can be traced.

Potato labels are polygons; labels.py reads them as their bounding box, as Ultralytics does.

    qpy analyse_protocol.py                  # v13, the dataset of record
    qpy analyse_protocol.py --dataset v12
Writes results/v13_protocol_analysis.csv or results/v12_protocol_analysis.csv. The unprefixed
results/protocol_analysis.csv is the record behind the published v12 manuscript (polygon bug and
old groups included) and is never rewritten.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from classes import GROUP_B, GROUP_B_V12_PAPER, NAMES
from labels import parse_line

HERE = Path(__file__).resolve().parent
DATASETS = {   # dataset -> (root, output csv); results/protocol_analysis.csv is the published record
    "v12": (Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v12"),
            HERE / "results" / "v12_protocol_analysis.csv"),
    "v13": (Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v13"),
            HERE / "results" / "v13_protocol_analysis.csv"),
}

WHOLE_LEAF_MIN = 0.50   # area fraction above which a box is effectively "the whole leaf"
LESION_MAX = 0.10       # ...and below which it is a localized lesion
GROUP_MEDIAN_MIN = 0.50 # a class whose median box reaches this is Group A (whole-leaf)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=sorted(DATASETS), default="v13")
ROOT, OUT = DATASETS[ap.parse_args().dataset]

areas = defaultdict(list)
per_img = defaultdict(list)
n_poly = n_lines = n_mixed_files = 0

for split in ("train", "valid", "test"):
    d = ROOT / "labels" / split
    for lbl in d.glob("*.txt"):
        counts = defaultdict(int)
        kinds = set()
        for line in lbl.read_text().splitlines():
            r = parse_line(line)
            if r is None:
                continue
            c, _, _, w, h, is_poly = r
            areas[c].append(w * h)
            counts[c] += 1
            kinds.add(is_poly)
            n_lines += 1
            n_poly += is_poly
        n_mixed_files += len(kinds) == 2
        for c, n in counts.items():
            per_img[c].append(n)

rows = []
for ci, name in enumerate(NAMES):
    a = np.array(areas.get(ci, []))
    if a.size == 0:
        continue
    npi = np.array(per_img.get(ci, []))
    whole = float((a >= WHOLE_LEAF_MIN).mean())
    lesion = float((a <= LESION_MAX).mean())
    majority = "whole_leaf" if whole >= lesion else "lesion"
    minority_frac = min(whole, lesion)          # boxes contradicting the class's own majority
    rows.append({
        "class": name,
        "group": "A_whole_leaf" if np.median(a) >= GROUP_MEDIAN_MIN else "B_lesion",
        "group_v12_paper": "B_lesion" if name in GROUP_B_V12_PAPER else "A_whole_leaf",
        "n_boxes": int(a.size),
        "median_area": round(float(np.median(a)), 4),
        "frac_whole_leaf": round(whole, 3),
        "frac_lesion": round(lesion, 3),
        "majority": majority,
        "mixed_frac": round(minority_frac, 3),   # <- the protocol-inconsistency signal
        "boxes_per_img": round(float(npi.mean()), 2),
    })

# Known mixed files: potato_late_blight train images Potato_Potato_train_0483/0484 (and 0485 in
# v12), 4 box lines each, which Ultralytics reads as polygons. 8 wrong train boxes in v13; accepted.
print(f"{ROOT.name}: {n_lines} label lines, {n_poly} of them polygons; "
      f"{n_mixed_files} files mix boxes and polygons (known: 2 in v13, 3 in v12; see labels.py)\n")
rows.sort(key=lambda r: -r["mixed_frac"])
w = max(len(r["class"]) for r in rows)
print(f"{'class':{w}s} {'group':13s} {'boxes':>7s} {'medA':>6s} {'whole':>6s} {'lesion':>7s} "
      f"{'mixed':>6s} {'b/img':>6s}")
for r in rows:
    print(f"{r['class']:{w}s} {r['group']:13s} {r['n_boxes']:7d} {r['median_area']:6.3f} "
          f"{r['frac_whole_leaf']:6.2f} {r['frac_lesion']:7.2f} {r['mixed_frac']:6.2f} "
          f"{r['boxes_per_img']:6.2f}")

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)

print(f"\nSAVED {OUT}")
for g in ("A_whole_leaf", "B_lesion"):
    sel = [r for r in rows if r["group"] == g]
    print(f"{g:13s} {len(sel):2d} classes | mean mixed_frac "
          f"{np.mean([r['mixed_frac'] for r in sel]):.3f} | "
          f"mean boxes/img {np.mean([r['boxes_per_img'] for r in sel]):.2f} | "
          f"mean median_area {np.mean([r['median_area'] for r in sel]):.3f}")
moved = [r for r in rows if r["group"] != r["group_v12_paper"]]
print(f"\n{len(moved)} classes change group against the v12 paper:")
for r in moved:
    print(f"  {r['class']:34s} {r['group_v12_paper']} -> {r['group']} "
          f"(median box {r['median_area']:.3f})")
top_b = max(r["median_area"] for r in rows if r["group"] == "B_lesion")
low_a = min(r["median_area"] for r in rows if r["group"] == "A_whole_leaf")
print(f"any median threshold in ({top_b:.3f}, {low_a:.3f}] gives these groups")
measured = {r["class"] for r in rows if r["group"] == "B_lesion"}
assert measured == GROUP_B, (f"classes.GROUP_B is out of date: measured B = {sorted(measured)}, "
                             f"classes.py has {sorted(GROUP_B)}")
print("classes.GROUP_B matches the measurement")
bad = [r for r in rows if r["mixed_frac"] >= 0.10]
print(f"\n{len(bad)}/{len(rows)} classes mix protocols in >=10% of boxes:")
for r in bad:
    print(f"  {r['class']:34s} {r['group']:13s} mixed {r['mixed_frac']:.2f} "
          f"(majority {r['majority']})")
