"""Protocol-harmonized subset of v12 - the data-centric arm, done so it is actually evaluable.

TWO BUGS IN THE FIRST ATTEMPT, both fixed here:

1. Majority was computed by BOX count. Lesion-style images carry many boxes each, so box-weighting
   inverted the answer: cucumber_Antraknosa is 54% lesion BY BOX but 89% whole-leaf BY IMAGE.
   Dropping the "minority" would have deleted 89% of the class and produced a data-quantity
   artifact, not a protocol result. Majority is now IMAGE-weighted.

2. Harmonizing TRAIN only cannot work, because val is protocol-mixed as well: a model trained on
   one protocol is then scored against images annotated in the other. So the harmonized subset is
   applied to train AND val AND test, and BOTH arms are evaluated on the SAME harmonized val.

Resulting comparison (fair - identical evaluation set for both):
    arm A  existing baseline weights            -> evaluate on harmonized val   (no training)
    arm B  model trained on harmonized train    -> evaluate on harmonized val
If B - A exceeds the noise floor, consistent supervision beats mixed supervision, which is the
thesis stated as a positive result rather than another null.

Note this is NOT comparable to the 0.8990 whole-dataset baseline - different evaluation set.
It is its own self-contained experiment with its own control.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v12")
HERE = Path(__file__).resolve().parent
OUT = HERE / "repair"

WHOLE_LEAF_MIN = 0.50
LESION_MAX = 0.10
MIXED_THRESHOLD = 0.10

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


def img_protocol(boxes):
    """Protocol of ONE image for ONE class: whole-leaf if any box covers >= half the frame."""
    if any(a >= WHOLE_LEAF_MIN for a in boxes):
        return "whole"
    if all(a <= LESION_MAX for a in boxes):
        return "lesion"
    return "mid"


def load(split):
    out = []
    for lbl in sorted((ROOT / "labels" / split).glob("*.txt")):
        per_cls = defaultdict(list)
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                per_cls[int(p[0])].append(float(p[3]) * float(p[4]))
        img = next((ROOT / "images" / split / (lbl.stem + e)
                    for e in (".jpg", ".jpeg", ".png", ".JPG", ".PNG")
                    if (ROOT / "images" / split / (lbl.stem + e)).exists()), None)
        if img:
            out.append((img, {c: img_protocol(v) for c, v in per_cls.items()}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    splits = {s: load(s) for s in ("train", "valid", "test")}

    # IMAGE-weighted majority protocol per class, from TRAIN only
    img_counts = defaultdict(lambda: defaultdict(int))
    for _, prot in splits["train"]:
        for c, m in prot.items():
            img_counts[c][m] += 1

    targets = {}
    print(f"{'class':34s} {'imgs':>6s} {'whole':>7s} {'lesion':>7s} {'mid':>6s} {'mixed':>7s} majority")
    for c in sorted(img_counts):
        d = img_counts[c]
        tot = sum(d.values())
        fw, fl = d["whole"] / tot, d["lesion"] / tot
        majority = "whole" if fw >= fl else "lesion"
        mixed = min(fw, fl)
        flag = ""
        if mixed >= MIXED_THRESHOLD:
            targets[c] = majority
            flag = "  <-- HARMONIZE"
        print(f"{NAMES[c]:34s} {tot:6d} {fw:7.2f} {fl:7.2f} {d['mid']/tot:6.2f} "
              f"{mixed:7.3f} {majority}{flag}")

    print(f"\nharmonize targets: {[NAMES[c] for c in targets]}")
    kept = {}
    for s, items in splits.items():
        k = [img for img, prot in items
             if not any(c in targets and m in ("whole", "lesion") and m != targets[c]
                        for c, m in prot.items())]
        kept[s] = k
        print(f"  {s:6s} {len(k):6d}/{len(items):6d} kept ({100*len(k)/len(items):.1f}%)")

    # per-class retention in train, so a data-quantity artifact cannot hide
    has, keptc = defaultdict(int), defaultdict(int)
    keepset = set(kept["train"])
    for img, prot in splits["train"]:
        for c in prot:
            has[c] += 1
            if img in keepset:
                keptc[c] += 1
    print("\nper-class TRAIN retention (worst first):")
    for c in sorted(has, key=lambda c: keptc[c] / has[c])[:6]:
        print(f"   {NAMES[c]:34s} {keptc[c]:5d}/{has[c]:5d}  {100*keptc[c]/has[c]:5.1f}%")

    if a.dry_run:
        print("\n(dry run - nothing written)")
        return

    OUT.mkdir(exist_ok=True)
    files = {}
    for s in ("train", "valid", "test"):
        f = OUT / f"harmonized_{s}.txt"
        f.write_text("\n".join(str(p) for p in kept[s]), encoding="utf-8")
        files[s] = str(f)
    yaml.safe_dump({"path": str(ROOT), "train": files["train"], "val": files["valid"],
                    "test": files["test"], "nc": len(NAMES), "names": NAMES},
                   open(OUT / "data_v12_harmonized.yaml", "w"), sort_keys=False)
    print(f"\nSAVED {OUT/'data_v12_harmonized.yaml'}")


if __name__ == "__main__":
    main()
