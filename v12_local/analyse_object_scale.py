"""Object-scale profile of a dataset (v13 by default) in COCO terms.

Buckets every box by absolute pixel area using the COCO definitions
(small < 32^2, medium 32^2-96^2, large >= 96^2), plus the tiny-object bands
used in the P2-head literature (< 8^2, < 16^2). Areas are reported at native
resolution and at the training input sizes (640 and 1024, long side letterboxed),
because detector strides act on the resized image.

Potato labels are polygons; labels.py reads each as its bounding box, as Ultralytics does. The
published results/object_scale.csv predates that fix and read polygon vertices as box sizes.

Run: qpy -I analyse_object_scale.py [--dataset v12]
Writes: results/v13_object_scale.csv or results/v12_object_scale.csv (per class) and prints the
group summary.
"""
import argparse
import csv
import sys
from collections import defaultdict

from PIL import Image

import paper_data as P
from labels import parse_line

D = P.dataset(P.add_dataset_arg(argparse.ArgumentParser()).parse_args().dataset)
ROOT = D.root
PROTOCOL = D.protocol
OUT = P.RES / f"{D.tag}object_scale.csv"
SPLITS = ("train", "valid", "test")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
BANDS = (("lt8", 8 ** 2), ("lt16", 16 ** 2), ("small", 32 ** 2), ("medium", 96 ** 2))


def names():
    lines = (ROOT / "data.yaml").read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == "names:")
    return [ln.strip()[2:] for ln in lines[i + 1:] if ln.strip().startswith("- ")]


def bucket(area):
    out = {"lt8": area < 8 ** 2, "lt16": area < 16 ** 2,
           "small": area < 32 ** 2, "medium": 32 ** 2 <= area < 96 ** 2,
           "large": area >= 96 ** 2}
    return out


def main():
    cls_names = names()
    group = {}
    with PROTOCOL.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            group[row["class"]] = row["group"]

    # counts[scope][res][band] ; scope = class name / group / ALL
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    n_boxes = defaultdict(int)
    n_images = 0
    sizes = defaultdict(int)
    missing = 0

    for split in SPLITS:
        idir, ldir = ROOT / "images" / split, ROOT / "labels" / split
        stems = {p.stem: p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT}
        for lbl in ldir.glob("*.txt"):
            img = stems.get(lbl.stem)
            if img is None:
                missing += 1
                continue
            with Image.open(img) as im:
                W, H = im.size
            sizes[(W, H)] += 1
            n_images += 1
            for ln in lbl.read_text(encoding="utf-8").splitlines():
                r = parse_line(ln)
                if r is None:
                    continue
                c = cls_names[r[0]]
                w, h = r[3], r[4]
                native = w * W * h * H
                res_area = {"native": native}
                for r in (640, 1024):
                    s = r / max(W, H)
                    res_area[str(r)] = native * s * s
                for scope in (c, group.get(c, "?"), "ALL"):
                    n_boxes[scope] += 1
                    for r, a in res_area.items():
                        for band, hit in bucket(a).items():
                            counts[scope][r][band] += hit

    print(f"images={n_images} boxes={n_boxes['ALL']} missing_images={missing}")
    print(f"boxes/image={n_boxes['ALL'] / n_images:.2f}")
    print("top image sizes:", sorted(sizes.items(), key=lambda kv: -kv[1])[:5])
    hdr = ["scope", "n_boxes"] + [f"{r}_{b}" for r in ("native", "640", "1024")
                                  for b in ("lt8", "lt16", "small", "medium", "large")]
    rows = []
    for scope in ["ALL", "A_whole_leaf", "B_lesion"] + cls_names:
        n = n_boxes[scope]
        if not n:
            continue
        row = [scope, n] + [round(counts[scope][r][b] / n, 4) for r in ("native", "640", "1024")
                            for b in ("lt8", "lt16", "small", "medium", "large")]
        rows.append(row)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([hdr] + rows)
    for row in rows[:3]:
        d = dict(zip(hdr, row))
        print(f"{d['scope']:14s} n={d['n_boxes']:6d} | @640 small {d['640_small']:.3f} "
              f"medium {d['640_medium']:.3f} large {d['640_large']:.3f} <16px {d['640_lt16']:.3f} "
              f"<8px {d['640_lt8']:.3f} | native small {d['native_small']:.3f} "
              f"| @1024 small {d['1024_small']:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
