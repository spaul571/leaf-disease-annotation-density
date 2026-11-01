"""Resampled v13 training sets for the class-imbalance arms (Shakib S-1, revision/PLAN_Shakib.md).

Ports the Environment 1 recipes to v13 without touching merged-dataset-v13:
  naive      - every primary class sampled to exactly TARGET images: undersampled without
               replacement above it, topped up with replacement below it (v10 exp1, v12 exp14,
               v12_kaggle/build_naive.py);
  oversample - every primary class below TARGET topped up to it with replacement; classes at or
               above TARGET keep all their images (v12 exp7, as the manuscript describes it).
Primary class = the class holding most of an image's boxes (Counter.most_common, as the
original). Images without boxes have no primary class and are left out, as in the original.
Sampling uses random.seed(42), as the original. Validation and test stay the v13 ones.

Images are copied, not hard-linked: Ultralytics rewrites a corrupt JPEG in place, and through a
hard link that would change merged-dataset-v13. A duplicate gets the suffix _dupN. Each set gets
data.yaml and resample_manifest.csv (image, primary class, copy number).

    qpy build_resampled_v13.py          # builds both; refuses to overwrite an existing set
"""
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from classes import NAMES

SRC = Path(r"F:\0BScDefense\05_Datasets\merged-dataset-v13")
TARGET = 400
SEED = 42
EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def primary_pool():
    """{class index: sorted image stems whose most frequent box class it is}, and skipped stems."""
    pool, empty = defaultdict(list), []
    for lf in sorted((SRC / "labels" / "train").glob("*.txt")):
        cnt = Counter(int(l.split()[0]) for l in lf.read_text().splitlines() if l.strip())
        if not cnt:
            empty.append(lf.stem)
            continue
        pool[cnt.most_common(1)[0][0]].append(lf.stem)
    return pool, empty


def image_of(stem):
    for e in EXTS:
        p = SRC / "images" / "train" / (stem + e)
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def choose(pool, mode):
    """[(stem, copy number)] in a fixed order; copy 0 is the original image."""
    random.seed(SEED)
    chosen = []
    for c in range(len(NAMES)):
        stems = pool.get(c, [])
        assert stems, f"class {NAMES[c]} has no primary-class training image"
        if len(stems) >= TARGET:
            chosen += [(s, 0) for s in (random.sample(stems, TARGET) if mode == "naive" else stems)]
        else:
            chosen += [(s, 0) for s in stems]
            chosen += [(s, j + 1) for j, s in enumerate(random.choices(stems, k=TARGET - len(stems)))]
    return chosen


def build(mode, pool):
    out = SRC.parent / f"merged-dataset-v13-{mode}"
    if out.exists():
        raise SystemExit(f"{out} exists - delete it first if it really should be rebuilt")
    (out / "images" / "train").mkdir(parents=True)
    (out / "labels" / "train").mkdir(parents=True)
    chosen, rows, seen = choose(pool, mode), [], set()
    primary = {s: c for c, stems in pool.items() for s in stems}
    for s, k in chosen:
        name = s if k == 0 else f"{s}_dup{k}"
        assert name not in seen, name
        seen.add(name)
        img = image_of(s)
        shutil.copy2(img, out / "images" / "train" / (name + img.suffix))
        shutil.copy2(SRC / "labels" / "train" / (s + ".txt"), out / "labels" / "train" / (name + ".txt"))
        rows.append([name, s, NAMES[primary[s]], k])
    with open(out / "resample_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "source_image", "primary_class", "copy"])
        w.writerows(rows)
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out), "train": "images/train",
         "val": str(SRC / "images" / "valid"), "test": str(SRC / "images" / "test"),
         "nc": len(NAMES), "names": NAMES}, sort_keys=False), encoding="utf-8")

    per = Counter(r[2] for r in rows)
    boxes = Counter()
    for lf in (out / "labels" / "train").glob("*.txt"):
        boxes.update(NAMES[int(l.split()[0])] for l in lf.read_text().splitlines() if l.strip())
    print(f"\n{mode}: {len(rows)} training images ({sum(r[3] > 0 for r in rows)} duplicates) -> {out}")
    for c in NAMES:
        print(f"  {c:<34} primary images {len(pool[NAMES.index(c)]):>5} -> {per[c]:>5}   boxes {boxes[c]:>6}")
    print(f"  box imbalance {max(boxes.values()) / min(boxes.values()):.1f}x")
    return out


if __name__ == "__main__":
    pool, empty = primary_pool()
    n = sum(len(v) for v in pool.values())
    print(f"v13 train: {n} images with a primary class, {len(empty)} without boxes (left out)")
    for mode in ("oversample", "naive"):
        build(mode, pool)
