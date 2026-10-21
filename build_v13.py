"""Build merged-dataset-v13: v12 with its rotated/flipped twins collapsed, then re-split.

v12 removed exact and re-compressed copies, but not the rotated and flipped copies that the
Roboflow export augmentation made of the Cucumber, Potato and Tomato (and some Chilli) training
images. v12_local/audit_dihedral_twins.py found that two thirds of the v12 test images have such
a twin in train (2026-10-21).

v13 repeats the v12 recipe with those twins:
  1. Group the v12 images by every candidate twin pair from the audit, with union-find. This is
     the audit's lenient definition: a false twin costs one clean image, a missed twin leaks.
     --strict uses only the pixel-confirmed pairs instead (sensitivity check).
  2. Keep ONE image per group: the one with the most boxes, ties broken by file name.
  3. Re-split the survivors 70/20/10, stratified by primary class, seed 42, as v12 did.
Because each group is reduced to one image before splitting, no twin can span two splits.

Writes 05_Datasets/merged-dataset-v13/ (images, labels, data.yaml) and, next to this script:
  v13_manifest.csv      one row per image: split, v12 split, SHA-256, dHash, primary class,
                        boxes, crop, size of its twin group
  v13_distribution.csv  boxes and images per class per split (same columns as v12's)

    qpy v12_local/audit_dihedral_twins.py      # once, on v12: writes the twin pair list
    qpy build_v13.py --dry-run                 # statistics only, writes nothing
    qpy build_v13.py                           # build
    qpy v12_local/audit_dihedral_twins.py --data f:/0BScDefense/05_Datasets/merged-dataset-v13 --tag v13_
                                               # must report 0 candidate pairs
"""
import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "05_Datasets" / "merged-dataset-v12"
DST = ROOT / "05_Datasets" / "merged-dataset-v13"
PAIRS = HERE / "v12_local" / "results" / "dihedral_twins_pairs.csv"

SPLITS = ["train", "valid", "test"]
RATIOS = (0.70, 0.20, 0.10)
SEED = 42


def dhash(path, size=8):
    """Same 64-bit dHash as build_v12.py, so manifest hashes are comparable across versions."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = 0
    for b in (a[:, 1:] > a[:, :-1]).flatten():
        bits = (bits << 1) | int(b)
    return bits


def read_label(p):
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def primary(it):
    return Counter(it["classes"]).most_common(1)[0][0] if it["classes"] else "_empty"


def main(dry_run, force, strict):
    names = yaml.safe_load((SRC / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    nc = len(names)

    # ---------- 1. inventory of v12 ----------
    items, key = [], {}
    for s in SPLITS:
        for p in sorted((SRC / "images" / s).iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lines = read_label(SRC / "labels" / s / (p.stem + ".txt"))
            key[(s, p.name)] = len(items)
            items.append({"img": p, "split": s, "lines": lines,
                          "classes": [int(l.split()[0]) for l in lines],
                          "crop": p.name.split("_")[0]})
    print(f"v12 inventory: {len(items)} images")

    # ---------- 2. twin groups ----------
    if not PAIRS.exists():
        raise SystemExit(f"missing {PAIRS} - run v12_local/audit_dihedral_twins.py first")
    with open(PAIRS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if strict:
        rows = [r for r in rows if r["confirmed"] == "1"]
    missing = [r for r in rows if (r["split_a"], r["img_a"]) not in key
               or (r["split_b"], r["img_b"]) not in key]
    if missing:
        raise SystemExit(f"{len(missing)} twin pairs name images not in v12 - stale pair list?")
    print(f"twin pairs used: {len(rows)} ({'strict' if strict else 'lenient'} definition)")

    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        parent[find(key[(r["split_a"], r["img_a"])])] = find(key[(r["split_b"], r["img_b"])])
    groups = defaultdict(list)
    for i, it in enumerate(items):
        groups[find(i)].append(it)

    # ---------- 3. one representative per group ----------
    reps, disagree = [], 0
    for g in groups.values():
        if len({len(it["lines"]) for it in g}) > 1:
            disagree += 1
        rep = sorted(g, key=lambda it: (-len(it["lines"]), it["img"].name))[0]
        rep["group_size"] = len(g)
        reps.append(rep)
    multi = sum(1 for g in groups.values() if len(g) > 1)
    print(f"twin groups (size > 1): {multi}")
    print(f"representatives kept  : {len(reps)}   dropped: {len(items) - len(reps)}")
    print(f"twin groups whose members disagree on the number of boxes: {disagree}")

    kept_crop = Counter(it["crop"] for it in reps)
    all_crop = Counter(it["crop"] for it in items)
    print("\nimages per crop, v12 -> v13")
    for c in sorted(all_crop):
        print(f"  {c:12s} {all_crop[c]:6d} -> {kept_crop[c]:6d}  ({kept_crop[c] / all_crop[c]:.0%})")

    # ---------- 4. stratified re-split by primary class (as build_v12.py) ----------
    random.seed(SEED)
    by_primary = defaultdict(list)
    for it in reps:
        by_primary[primary(it)].append(it)
    assign = {}
    for cls in sorted(by_primary, key=str):
        lst = sorted(by_primary[cls], key=lambda it: it["img"].name)
        random.shuffle(lst)
        n = len(lst)
        n_tr = int(round(n * RATIOS[0]))
        n_va = int(round(n * RATIOS[1]))
        if n >= 3:  # every class present in every split when it has >= 3 images
            n_tr = max(1, min(n_tr, n - 2))
            n_va = max(1, min(n_va, n - n_tr - 1))
        for i, it in enumerate(lst):
            assign[id(it)] = "train" if i < n_tr else "valid" if i < n_tr + n_va else "test"

    box = {s: Counter() for s in SPLITS}
    img = {s: Counter() for s in SPLITS}
    for it in reps:
        s = assign[id(it)]
        for c in it["classes"]:
            box[s][c] += 1
        for c in set(it["classes"]):
            img[s][c] += 1
    print("\n--- v13 split sizes ---")
    for s in SPLITS:
        n = sum(1 for it in reps if assign[id(it)] == s)
        print(f"  {s:<6} {n:>6} images  {sum(box[s].values()):>6} boxes")
    counts = [sum(box[s][i] for s in SPLITS) for i in range(nc)]
    nz = [c for c in counts if c]
    print(f"box-level imbalance: {max(nz) / min(nz):.1f}x (v12: 163.4x)")
    absent = [(names[i], s) for i in range(nc) for s in SPLITS if img[s][i] == 0]
    if absent:
        print("WARNING - classes with no image in a split:", absent)

    print(f"\n{'class':<34}{'tr_box':>7}{'va_box':>7}{'te_box':>7}{'tr_img':>7}{'va_img':>7}{'te_img':>7}")
    dist = []
    for i in range(nc):
        tb, vb, eb = box["train"][i], box["valid"][i], box["test"][i]
        tot = tb + vb + eb
        print(f"{names[i]:<34}{tb:>7}{vb:>7}{eb:>7}"
              f"{img['train'][i]:>7}{img['valid'][i]:>7}{img['test'][i]:>7}")
        dist.append([names[i], tb, vb, eb, round(100 * tb / tot, 1) if tot else 0,
                     img["train"][i], img["valid"][i], img["test"][i]])

    if dry_run:
        print("\n--dry-run: nothing written")
        return

    # ---------- 5. write v13 ----------
    if DST.exists():
        if not force:
            raise SystemExit(f"{DST} exists - pass --force to delete and rebuild it")
        shutil.rmtree(DST)
    for s in SPLITS:
        (DST / "images" / s).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / s).mkdir(parents=True, exist_ok=True)
    manifest = []
    for n, it in enumerate(sorted(reps, key=lambda it: (assign[id(it)], it["img"].name))):
        s = assign[id(it)]
        out = DST / "images" / s / it["img"].name
        shutil.copy2(it["img"], out)
        (DST / "labels" / s / (it["img"].stem + ".txt")).write_text(
            "\n".join(it["lines"]) + ("\n" if it["lines"] else ""))
        pc = primary(it)
        manifest.append([it["img"].name, s, it["split"],
                         hashlib.sha256(out.read_bytes()).hexdigest(), f"{dhash(out):016x}",
                         names[pc] if pc != "_empty" else "", len(it["lines"]), it["crop"],
                         it["group_size"]])
        if n % 2000 == 0:
            print(f"  wrote {n}/{len(reps)}", flush=True)
    with open(DST / "data.yaml", "w") as f:
        yaml.safe_dump({"path": str(DST), "train": "images/train", "val": "images/valid",
                        "test": "images/test", "nc": nc, "names": names}, f, sort_keys=False)

    with open(HERE / "v13_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "v12_split", "sha256", "dhash", "primary_class", "boxes",
                    "crop", "twin_group_size"])
        w.writerows(manifest)
    with open(HERE / "v13_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "train_boxes", "valid_boxes", "test_boxes",
                    "train_pct", "train_imgs", "valid_imgs", "test_imgs"])
        w.writerows(dist)
    print(f"\nwrote {DST}, v13_manifest.csv, v13_distribution.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print statistics, write nothing")
    ap.add_argument("--force", action="store_true", help="delete and rebuild an existing v13")
    ap.add_argument("--strict", action="store_true",
                    help="collapse only pixel-confirmed twins (sensitivity check)")
    a = ap.parse_args()
    main(a.dry_run, a.force, a.strict)
