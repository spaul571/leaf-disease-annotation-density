"""Audit a YOLO dataset for rotated/flipped twins across splits.

The Cucumber, Potato and Tomato sources were exported from Roboflow with augmentation: three
versions of each training image, each with a random 90-degree rotation and/or flip (see each
source's README.roboflow.txt). v10 moved source-train images into valid/test, and build_v12.py
re-split everything after collapsing near-duplicates with a dHash, which is not invariant to
rotation or flipping. Two augmented versions of one photograph can therefore sit in different
splits of v12.

Method: for every image, compute the dHash of all 8 dihedral transforms (4 rotations x flip).
Image j is a candidate twin of image i when some transform of i is within Hamming distance 3 of
j's untransformed hash (found through a 4 x 16-bit pigeonhole index).

Two definitions are reported, because no pixel threshold separates twins from look-alikes cleanly
(checked by eye on contact sheets, 2026-10-21):
  strict   - candidate confirmed on pixels: grey-level MAD at 32 x 32 <= --mad-max, or normalised
             cross-correlation (NCC) at 64 x 64 >= --ncc-min. NCC catches the grayscale and
             brightness-shifted copies that MAD rejects. A lower bound on the leakage.
  lenient  - every candidate. About 3 in 4 of the candidates that fail the strict test were still
             twins on inspection; the false ones are mostly letterboxed frames and similar leaves on
             the same grey background. Leakage removal (build_v13.py, the twin-free test subset)
             uses this definition: a false twin only costs one clean image, a missed twin leaks.
Twins are grouped with union-find under each definition.

Writes, under results/ and paper/ (prefixed with --tag):
  dihedral_twins_pairs.csv             every candidate pair, with MAD, NCC and `confirmed` (strict)
  dihedral_twins_summary.json          counts, per-split and per-class leakage, both definitions
  audit_dihedral_twins_examples.png    12 random confirmed cross-split pairs
  audit_dihedral_twins_borderline.png  the 24 unconfirmed candidates with the highest NCC

    qpy audit_dihedral_twins.py                                        # v12
    qpy audit_dihedral_twins.py --data <...>/merged-dataset-v13 --tag v13_
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "05_Datasets" / "merged-dataset-v12"
RES = HERE / "results"
PAP = HERE / "paper"
SPLITS = ("train", "valid", "test")
HAMMING_MAX = 3
MAD_MAX = 12.0  # grey levels (0-255) at 32 x 32; same photo re-encoded sits far below this
NCC_MIN = 0.95  # at 64 x 64; insensitive to the grayscale/brightness changes that defeat MAD

Image.MAX_IMAGE_PIXELS = None


def _dhash(a):
    a = np.asarray(Image.fromarray(a).resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = 0
    for b in (a[:, 1:] > a[:, :-1]).flatten():
        bits = (bits << 1) | int(b)
    return bits


def _dihedral(a):
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, np.fliplr(r)]
    return out  # index 0 is the identity


def _load(path):
    with Image.open(path) as im:
        g = im.convert("L")
        a128 = np.asarray(g.resize((128, 128), Image.LANCZOS), dtype=np.uint8)
        a32 = np.asarray(g.resize((32, 32), Image.LANCZOS), dtype=np.uint8)
        a64 = np.asarray(g.resize((64, 64), Image.LANCZOS), dtype=np.uint8)
    return [_dhash(np.ascontiguousarray(t)) for t in _dihedral(a128)], a32, a64


def _ncc(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


def _primary_class(lbl):
    if not lbl.exists():
        return None
    ids = [int(l.split()[0]) for l in lbl.read_text().splitlines() if l.strip()]
    return Counter(ids).most_common(1)[0][0] if ids else None


def _sheet(items, pairs, path):
    tiles = []
    for i, j, m, t in pairs:
        a = Image.open(items[i]["img"]).convert("RGB").resize((160, 160))
        b = Image.open(items[j]["img"]).convert("RGB").resize((160, 160))
        row = Image.new("RGB", (330, 160), "white")
        row.paste(a, (0, 0))
        row.paste(b, (170, 0))
        tiles.append(row)
    sheet = Image.new("RGB", (330 * 3 + 20, 170 * ((len(tiles) + 2) // 3)), "white")
    for n, tile in enumerate(tiles):
        sheet.paste(tile, ((n % 3) * 340, (n // 3) * 170))
    sheet.save(path)
    print("wrote", path)


def _grouping(items, edges):
    """Union-find over twin edges -> (groups, valid/test images with a twin in train, #cross)."""
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(j)
    groups = defaultdict(set)
    for i, j in edges:
        groups[find(i)].update((i, j))
    leaked, cross = set(), 0
    for members in groups.values():
        splits = {items[m]["split"] for m in members}
        cross += len(splits) > 1
        if "train" in splits:
            leaked.update(m for m in members if items[m]["split"] != "train")
    return groups, leaked, cross


def main(data, tag, mad_max, ncc_min):
    names = yaml.safe_load((data / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    items = []
    for s in SPLITS:
        for p in sorted((data / "images" / s).iterdir()):
            items.append({"split": s, "img": p, "lbl": data / "labels" / s / (p.stem + ".txt")})
    print(f"{data.name}: {len(items)} images", flush=True)

    with ProcessPoolExecutor() as ex:
        loaded = list(ex.map(_load, [it["img"] for it in items], chunksize=64))
    H = [h for h, _, _ in loaded]
    A = np.stack([a for _, a, _ in loaded]).astype(np.float32)
    A64 = np.stack([a for _, _, a in loaded]).astype(np.float32)
    for it in items:
        c = _primary_class(it["lbl"])
        it["cls"] = names[c] if c is not None else "background"
        it["crop"] = it["img"].name.split("_")[0]
    print("hashed", flush=True)

    index = defaultdict(list)  # (block, value) -> image ids, on the identity hash
    for j, h in enumerate(H):
        for b in range(4):
            index[(b, (h[0] >> (16 * b)) & 0xFFFF)].append(j)

    best = {}  # unordered pair -> (mad, transform, ncc)
    for i, hs in enumerate(H):
        for t, h in enumerate(hs):
            cand = set()
            for b in range(4):
                cand.update(index[(b, (h >> (16 * b)) & 0xFFFF)])
            for j in cand:
                if j == i or bin(h ^ H[j][0]).count("1") > HAMMING_MAX:
                    continue
                ti = np.ascontiguousarray(_dihedral(A[i])[t])
                mad = float(np.abs(ti - A[j]).mean())
                key = (min(i, j), max(i, j))
                if key not in best or mad < best[key][0]:
                    ncc = _ncc(np.ascontiguousarray(_dihedral(A64[i])[t]), A64[j])
                    best[key] = (mad, t, ncc)

    def ok(m, c):
        return m <= mad_max or c >= ncc_min

    pairs = [(i, j, m, t) for (i, j), (m, t, c) in best.items() if ok(m, c)]
    print(f"candidate pairs: {len(best)}  confirmed (MAD <= {mad_max} or NCC >= {ncc_min}): "
          f"{len(pairs)}", flush=True)
    groups, leaked, cross_groups = _grouping(items, [(i, j) for i, j, _, _ in pairs])
    groups_l, leaked_l, cross_l = _grouping(items, list(best))

    RES.mkdir(exist_ok=True)
    PAP.mkdir(exist_ok=True)
    with open(RES / f"{tag}dihedral_twins_pairs.csv", "w", encoding="utf-8") as f:
        f.write("img_a,split_a,class_a,img_b,split_b,class_b,transform,mad,ncc,confirmed\n")
        for (i, j), (m, t, c) in sorted(best.items(), key=lambda kv: kv[1][0]):
            a, b = items[i], items[j]
            f.write(f"{a['img'].name},{a['split']},{a['cls']},{b['img'].name},{b['split']},"
                    f"{b['cls']},{t},{m:.2f},{c:.4f},{int(ok(m, c))}\n")

    per_split = {}
    for s in ("valid", "test"):
        ids = [k for k, it in enumerate(items) if it["split"] == s]
        by_crop, by_cls = Counter(), Counter()
        tot_crop = Counter(items[k]["crop"] for k in ids)
        tot_cls = Counter(items[k]["cls"] for k in ids)
        for k in ids:
            if k in leaked:
                by_crop[items[k]["crop"]] += 1
                by_cls[items[k]["cls"]] += 1
        per_split[s] = {
            "images": len(ids),
            "with_train_twin": sum(1 for k in ids if k in leaked),
            "with_train_twin_lenient": sum(1 for k in ids if k in leaked_l),
            "by_crop": {c: [by_crop[c], tot_crop[c]] for c in sorted(tot_crop)},
            "by_class": {c: [by_cls[c], tot_cls[c]] for c in sorted(tot_cls)},
        }
    summary = {
        "dataset": data.name, "images": len(items), "candidate_pairs": len(best),
        "confirmed_pairs": len(pairs), "hamming_max": HAMMING_MAX, "mad_max": mad_max,
        "ncc_min": ncc_min,
        "twin_groups": len(groups), "images_in_twin_groups": sum(len(g) for g in groups.values()),
        "groups_spanning_splits": cross_groups,
        "lenient": {"twin_groups": len(groups_l),
                    "images_in_twin_groups": sum(len(g) for g in groups_l.values()),
                    "groups_spanning_splits": cross_l},
        "transform_counts": dict(Counter(t for _, _, _, t in pairs)),
        "per_split": per_split,
    }
    (RES / f"{tag}dihedral_twins_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_split"}, indent=1))
    for s, d in per_split.items():
        print(f"\n{s}: {d['with_train_twin']} of {d['images']} images have a twin in train "
              f"(strict); {d['with_train_twin_lenient']} counting every candidate")
        for c, (n, tot) in d["by_crop"].items():
            print(f"   {c:12s} {n:5d} / {tot:5d}")

    cross = [(i, j, m, t) for i, j, m, t in pairs if items[i]["split"] != items[j]["split"]]
    random.seed(0)
    if cross:
        _sheet(items, random.sample(cross, min(12, len(cross))),
               PAP / f"{tag}audit_dihedral_twins_examples.png")
    near = [(i, j, m, t) for (i, j), (m, t, c) in
            sorted(best.items(), key=lambda kv: -kv[1][2]) if not ok(m, c)][:24]
    if near:
        _sheet(items, near, PAP / f"{tag}audit_dihedral_twins_borderline.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA, help="dataset root holding data.yaml")
    ap.add_argument("--tag", default="", help="prefix for output files, e.g. v13_")
    ap.add_argument("--mad-max", type=float, default=MAD_MAX)
    ap.add_argument("--ncc-min", type=float, default=NCC_MIN)
    a = ap.parse_args()
    main(a.data.resolve(), a.tag, a.mad_max, a.ncc_min)
