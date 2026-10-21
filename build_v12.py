"""Build merged-dataset-v12: a leakage-free version of MV-LeafDet22.

v10 has 20%+ of its valid/test images byte-identical to training images (see
duplicate_detection.py). Any metric computed on v10 is inflated by memorisation.

v12 fixes this structurally:
  1. Group all images by near-duplicate identity (64-bit dHash, Hamming <= 3, union-find).
     This catches exact copies AND re-compressions/resizes that MD5 misses.
  2. Collapse each group to ONE representative image (the one with the most annotations;
     ties broken by filename for determinism). Every other member is dropped.
  3. Re-split the surviving unique images stratified by primary class at 70/20/10.

Because a duplicate group can no longer have members in two splits, cross-split
leakage is impossible by construction rather than by inspection.

Also reports label disagreement inside duplicate groups - identical pixels with
different annotations is an annotation-quality defect worth knowing about.
"""
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SRC = Path("f:/0BScDefense/05_Datasets/merged-dataset-v10")
DST = Path("f:/0BScDefense/05_Datasets/merged-dataset-v12")
REPORT = Path("f:/0BScDefense/04_Code/journal_analysis")

SPLITS = ["train", "valid", "test"]
RATIOS = (0.70, 0.20, 0.10)
HAMMING_THRESHOLD = 3
SEED = 42

NC = 22
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


def dhash(path, size=8):
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    except Exception:
        return None
    bits = 0
    for b in (a[:, 1:] > a[:, :-1]).flatten():
        bits = (bits << 1) | int(b)
    return bits


def read_label(p):
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(line)
    return out


def main():
    random.seed(SEED)

    # ---------- 1. inventory ----------
    items = []  # dict(img, lbl, split, boxes, classes)
    for s in SPLITS:
        for p in sorted((SRC / "images" / s).iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl = SRC / "labels" / s / (p.stem + ".txt")
            lines = read_label(lbl)
            cls = [int(l.split()[0]) for l in lines]
            items.append({"img": p, "lbl": lbl, "split": s, "lines": lines, "classes": cls})
    print(f"v10 inventory: {len(items)} images", flush=True)

    # ---------- 2. near-duplicate grouping ----------
    print("hashing...", flush=True)
    for i, it in enumerate(items):
        if i % 2000 == 0:
            print(f"  {i}/{len(items)}", flush=True)
        it["dh"] = dhash(it["img"])
    valid = [it for it in items if it["dh"] is not None]
    print(f"hashed ok: {len(valid)}", flush=True)

    by_dh = defaultdict(list)
    for it in valid:
        by_dh[it["dh"]].append(it)
    keys = list(by_dh.keys())

    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    karr = np.array(keys, dtype=np.uint64)
    print("near-dup scan...", flush=True)
    for idx in range(len(keys)):
        if idx % 2000 == 0 and idx:
            print(f"  {idx}/{len(keys)}", flush=True)
        xor = np.bitwise_xor(karr[idx + 1:], karr[idx])
        if xor.size == 0:
            continue
        pc = np.unpackbits(xor.view(np.uint8)).reshape(-1, 64).sum(axis=1)
        for j in np.nonzero(pc <= HAMMING_THRESHOLD)[0]:
            union(keys[idx], keys[idx + 1 + int(j)])

    groups = defaultdict(list)
    for k in keys:
        groups[find(k)].extend(by_dh[k])
    print(f"unique groups: {len(groups)}  (from {len(valid)} images)", flush=True)

    # ---------- 3. collapse each group to one representative ----------
    label_conflicts = 0
    reps = []
    for g in groups.values():
        sigs = {tuple(sorted(it["lines"])) for it in g}
        if len(sigs) > 1:
            label_conflicts += 1
        # representative = most annotations, then lexicographic filename (deterministic)
        rep = sorted(g, key=lambda it: (-len(it["lines"]), it["img"].name))[0]
        rep["group_size"] = len(g)
        reps.append(rep)

    print(f"\nrepresentatives kept : {len(reps)}")
    print(f"images dropped       : {len(valid) - len(reps)}")
    print(f"duplicate groups with CONFLICTING labels: {label_conflicts}")

    # ---------- 4. stratified re-split by primary class ----------
    by_primary = defaultdict(list)
    for it in reps:
        if not it["classes"]:
            by_primary["_empty"].append(it)
        else:
            by_primary[Counter(it["classes"]).most_common(1)[0][0]].append(it)

    assign = {}
    for cls, lst in by_primary.items():
        lst = sorted(lst, key=lambda it: it["img"].name)
        random.shuffle(lst)
        n = len(lst)
        n_tr = int(round(n * RATIOS[0]))
        n_va = int(round(n * RATIOS[1]))
        # guarantee every class is present in every split when it has >= 3 images
        if n >= 3:
            n_tr = max(1, min(n_tr, n - 2))
            n_va = max(1, min(n_va, n - n_tr - 1))
        for i, it in enumerate(lst):
            if i < n_tr:
                assign[it["img"]] = "train"
            elif i < n_tr + n_va:
                assign[it["img"]] = "valid"
            else:
                assign[it["img"]] = "test"

    # ---------- 5. write v12 ----------
    if DST.exists():
        shutil.rmtree(DST)
    for s in SPLITS:
        (DST / "images" / s).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / s).mkdir(parents=True, exist_ok=True)

    for it in reps:
        s = assign[it["img"]]
        shutil.copy(it["img"], DST / "images" / s / it["img"].name)
        with open(DST / "labels" / s / (it["img"].stem + ".txt"), "w") as f:
            f.write("\n".join(it["lines"]) + ("\n" if it["lines"] else ""))

    yaml.safe_dump(
        {"path": str(DST), "train": "images/train", "val": "images/valid",
         "test": "images/test", "nc": NC, "names": NAMES},
        open(DST / "data.yaml", "w"), sort_keys=False)

    # ---------- 6. report ----------
    box = {s: Counter() for s in SPLITS}
    img = {s: Counter() for s in SPLITS}
    for it in reps:
        s = assign[it["img"]]
        for c in it["classes"]:
            box[s][c] += 1
        for c in set(it["classes"]):
            img[s][c] += 1

    print("\n--- v12 split sizes ---")
    for s in SPLITS:
        n = len(list((DST / "images" / s).iterdir()))
        print(f"  {s:<6} {n:>6} images  {sum(box[s].values()):>6} boxes")

    print(f"\n{'class':<34}{'tr_box':>7}{'va_box':>7}{'te_box':>7}{'train%':>8}{'tr_img':>7}")
    print("-" * 74)
    rows = []
    for i in range(NC):
        tb, vb, eb = box["train"][i], box["valid"][i], box["test"][i]
        tot = tb + vb + eb
        pct = 100 * tb / tot if tot else 0
        print(f"{NAMES[i]:<34}{tb:>7}{vb:>7}{eb:>7}{pct:>7.1f}%{img['train'][i]:>7}")
        rows.append([NAMES[i], tb, vb, eb, round(pct, 1),
                     img["train"][i], img["valid"][i], img["test"][i]])

    with open(REPORT / "v12_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "train_boxes", "valid_boxes", "test_boxes",
                    "train_pct", "train_imgs", "valid_imgs", "test_imgs"])
        w.writerows(rows)

    tot_box = sum(sum(box[s].values()) for s in SPLITS)
    counts = [sum(box[s][i] for s in SPLITS) for i in range(NC)]
    nz = [c for c in counts if c]
    print(f"\ntotal boxes: {tot_box}")
    print(f"imbalance ratio: {max(nz) / min(nz):.1f}x  (v10 was 91.2x)")
    print(f"\nWROTE {DST}")


if __name__ == "__main__":
    main()
