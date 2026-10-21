"""Protocol harmonization of the v12 TRAIN split (the data-centric arm).

Rationale: per-class mAP is an almost deterministic function of annotation density (r=-0.92),
and the classes that mix two box protocols inside one label are the ones that underperform.
Seven model-side interventions failed to move this; the remaining lever is the supervision.

Repair rule - drop the IMAGE, never the box. Deleting a contradicting box would leave a real
object in the image unlabelled, i.e. teach the detector that the object is background. Dropping
the whole image removes the contradictory supervision without manufacturing false negatives.

CRITICAL: TRAIN ONLY. val/test labels are untouched, so every number stays comparable to the
3-seed noise floor. Changing evaluation ground truth would invalidate the entire results table.

No files are copied: ultralytics accepts a .txt list of image paths as `train:`, so the repaired
split is just a file list.

    qpy build_repair.py --dry-run     # report what would be removed
    qpy build_repair.py               # write train_repaired.txt + data_v12_repaired.yaml
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v12")
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "repair"

WHOLE_LEAF_MIN = 0.50
LESION_MAX = 0.10
MIXED_THRESHOLD = 0.10     # classes above this are treated as protocol-inconsistent

NAMES = [
    "bean_angular_leaf_spot", "bean_bean_rust", "bean_healthy",
    "cauliflower_bacterial_spot_rot", "cauliflower_black_rot",
    "cauliflower_downy_mildew", "cauliflower_no_disease",
    "chilli_Curling-Leafs", "chilli_Healthy-Leafs", "chilli_Leaf Spot-Leafs",
    "chilli_Leveillula taurica-Leafs", "chilli_Mozaik Leaf",
    "cucumber_Antraknosa", "cucumber_Bacterial Wilt", "cucumber_Downy Mildew",
    "potato_healthy", "potato_blackleg", "potato_late_blight", "potato_early_blight",
    "tomato_bacterial_spot", "tomato_curl_virus", "tomato_two_spotted_spider_mite",
]


def mode_of(area):
    if area >= WHOLE_LEAF_MIN:
        return "whole"
    if area <= LESION_MAX:
        return "lesion"
    return "mid"


def load_split(split):
    """-> list of (image_path, [(cls, area_mode), ...])"""
    out = []
    for lbl in sorted((ROOT / "labels" / split).glob("*.txt")):
        boxes = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            boxes.append((int(p[0]), mode_of(float(p[3]) * float(p[4]))))
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            cand = ROOT / "images" / split / (lbl.stem + ext)
            if cand.exists():
                img = cand
                break
        if img is not None:
            out.append((img, boxes))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    train = load_split("train")
    print(f"train images: {len(train)}")

    # majority protocol per class, computed on TRAIN only (no peeking at val/test)
    counts = defaultdict(lambda: defaultdict(int))
    for _, boxes in train:
        for c, m in boxes:
            counts[c][m] += 1

    targets = {}
    print(f"\n{'class':34s} {'whole':>7s} {'lesion':>7s} {'mid':>6s} {'mixed':>7s} majority")
    for c in sorted(counts):
        w, l, mid = counts[c]["whole"], counts[c]["lesion"], counts[c]["mid"]
        tot = w + l + mid
        if not tot:
            continue
        fw, fl = w / tot, l / tot
        majority = "whole" if fw >= fl else "lesion"
        mixed = min(fw, fl)
        flag = ""
        if mixed >= MIXED_THRESHOLD:
            targets[c] = majority
            flag = "  <-- REPAIR"
        print(f"{NAMES[c]:34s} {fw:7.2f} {fl:7.2f} {mid/tot:6.2f} {mixed:7.3f} {majority}{flag}")

    # an image is dropped if it carries a TARGET class box in the minority protocol
    keep, drop = [], []
    drop_by_class = defaultdict(int)
    for img, boxes in train:
        bad = False
        for c, m in boxes:
            if c in targets and m in ("whole", "lesion") and m != targets[c]:
                bad = True
                drop_by_class[c] += 1
        (drop if bad else keep).append(img)

    print(f"\nrepair targets: {[NAMES[c] for c in targets]}")
    print(f"images kept   : {len(keep)}  ({100*len(keep)/len(train):.1f}%)")
    print(f"images dropped: {len(drop)}  ({100*len(drop)/len(train):.1f}%)")
    for c, n in sorted(drop_by_class.items(), key=lambda kv: -kv[1]):
        print(f"   {NAMES[c]:34s} contributed {n} offending images")

    if a.dry_run:
        print("\n(dry run - nothing written)")
        return

    OUT_DIR.mkdir(exist_ok=True)
    lst = OUT_DIR / "train_repaired.txt"
    lst.write_text("\n".join(str(p) for p in keep), encoding="utf-8")

    data = {
        "path": str(ROOT),
        "train": str(lst),
        "val": "images/valid",     # UNCHANGED - comparability depends on this
        "test": "images/test",
        "nc": len(NAMES),
        "names": NAMES,
    }
    ymlp = OUT_DIR / "data_v12_repaired.yaml"
    yaml.safe_dump(data, open(ymlp, "w"), sort_keys=False)
    # control arm: same number of images removed at random, to separate "protocol repair"
    # from "less training data"
    rng = np.random.default_rng(0)
    ctrl_keep = list(train)
    idx = rng.choice(len(ctrl_keep), size=len(train) - len(keep), replace=False)
    ctrl = [p for i, (p, _) in enumerate(ctrl_keep) if i not in set(idx.tolist())]
    lstc = OUT_DIR / "train_control.txt"
    lstc.write_text("\n".join(str(p) for p in ctrl), encoding="utf-8")
    datac = dict(data, train=str(lstc))
    yaml.safe_dump(datac, open(OUT_DIR / "data_v12_control.yaml", "w"), sort_keys=False)

    print(f"\nSAVED {ymlp}  ({len(keep)} imgs)")
    print(f"SAVED {OUT_DIR/'data_v12_control.yaml'}  ({len(ctrl)} imgs, random removal control)")


if __name__ == "__main__":
    main()
