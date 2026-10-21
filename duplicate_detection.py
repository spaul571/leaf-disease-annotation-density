"""Duplicate / near-duplicate image detection for MV-LeafDet22 (merged-dataset-v10).

Addresses reviewer requests:
  R1: "Perform duplicate-image detection across sources."
  R2: "...duplicate picture removal ... potential overlap between Roboflow source datasets."

Reports three things, in order of severity:
  1. CROSS-SPLIT LEAKAGE  - identical/near-identical images in train AND val/test.
                            This inflates every reported metric and is the critical check.
  2. CROSS-CROP OVERLAP   - the same image present under two different source datasets.
  3. WITHIN-SPLIT DUPES   - redundancy (wastes capacity, biases class counts).

Exact duplicates : MD5 of raw file bytes.
Near-duplicates  : 64-bit dHash, grouped by exact hash match, then Hamming <= THRESHOLD
                   within each hash bucket's neighbourhood (BK-tree-free, bucketed scan).
"""
import csv
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BASE = Path("f:/0BScDefense/05_Datasets/merged-dataset-v10")
OUT_DIR = Path("f:/0BScDefense/04_Code/journal_analysis")
SPLITS = ["train", "valid", "test"]
HAMMING_THRESHOLD = 3  # <=3 bits of 64 differ => near-duplicate

CROPS = ["Bean", "Cauliflower", "Chilli", "Cucumber", "Potato", "Tomato"]


def crop_of(name: str) -> str:
    low = name.lower()
    for c in CROPS:
        if low.startswith(c.lower()):
            return c
    return "Unknown"


def dhash(path: Path, size: int = 8) -> int | None:
    """64-bit difference hash: resize to (size+1) x size grayscale, compare adjacent pixels."""
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
            a = np.asarray(im, dtype=np.int16)
    except Exception:
        return None
    diff = a[:, 1:] > a[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def md5(path: Path, chunk: int = 1 << 20) -> str | None:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except Exception:
        return None
    return h.hexdigest()


def main() -> None:
    files = []  # (split, path)
    for s in SPLITS:
        d = BASE / "images" / s
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                files.append((s, p))
    print(f"scanning {len(files)} images from {BASE}", flush=True)

    md5s, dhs = {}, {}
    for i, (s, p) in enumerate(files):
        if i % 2000 == 0:
            print(f"  {i}/{len(files)}", flush=True)
        md5s[p] = md5(p)
        dhs[p] = dhash(p)
    print("hashing done", flush=True)

    split_of = {p: s for s, p in files}

    # ---------- group by exact md5 ----------
    by_md5 = defaultdict(list)
    for p, h in md5s.items():
        if h:
            by_md5[h].append(p)
    exact_groups = [g for g in by_md5.values() if len(g) > 1]

    # ---------- group by dhash (near-dup) ----------
    by_dh = defaultdict(list)
    for p, h in dhs.items():
        if h is not None:
            by_dh[h].append(p)

    # bucket-scan for Hamming <= threshold between distinct dhash buckets
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
    for idx in range(len(keys)):
        if idx % 2000 == 0 and idx:
            print(f"  near-dup scan {idx}/{len(keys)}", flush=True)
        x = karr[idx]
        xor = np.bitwise_xor(karr[idx + 1:], x)
        if xor.size == 0:
            continue
        # popcount of uint64 via view as uint8
        pc = np.unpackbits(xor.view(np.uint8)).reshape(-1, 64).sum(axis=1)
        close = np.nonzero(pc <= HAMMING_THRESHOLD)[0]
        for j in close:
            union(keys[idx], keys[idx + 1 + int(j)])

    clusters = defaultdict(list)
    for k in keys:
        clusters[find(k)].extend(by_dh[k])
    near_groups = [g for g in clusters.values() if len(g) > 1]

    # ---------- classify ----------
    def analyse(groups, label):
        cross_split, cross_crop, within = [], [], []
        for g in groups:
            splits = {split_of[p] for p in g}
            crops = {crop_of(p.name) for p in g}
            if len(splits) > 1:
                cross_split.append(g)
            else:
                within.append(g)
            if len(crops) > 1:
                cross_crop.append(g)
        print(f"\n--- {label} ---")
        print(f"  groups                : {len(groups)}")
        print(f"  images involved       : {sum(len(g) for g in groups)}")
        print(f"  CROSS-SPLIT (leakage) : {len(cross_split)} groups"
              f"  [{sum(len(g) for g in cross_split)} images]")
        print(f"  cross-crop (source overlap): {len(cross_crop)} groups")
        print(f"  within-split only     : {len(within)} groups")
        if cross_split:
            print(f"  !! leaked pairs by split combo:")
            combo = defaultdict(int)
            for g in cross_split:
                combo[tuple(sorted({split_of[p] for p in g}))] += 1
            for k, v in sorted(combo.items(), key=lambda x: -x[1]):
                print(f"       {'+'.join(k):<20} {v} groups")
            print("  examples:")
            for g in cross_split[:5]:
                print("       " + " | ".join(f"{split_of[p]}:{p.name}" for p in g[:3]))
        return cross_split, cross_crop, within

    ex_cs, ex_cc, ex_w = analyse(exact_groups, f"EXACT duplicates (md5)")
    nr_cs, nr_cc, nr_w = analyse(near_groups, f"NEAR duplicates (dHash, Hamming<={HAMMING_THRESHOLD})")

    # ---------- per-crop ----------
    print("\n--- near-duplicate images per crop ---")
    per_crop = defaultdict(int)
    for g in near_groups:
        for p in g:
            per_crop[crop_of(p.name)] += 1
    total_per_crop = defaultdict(int)
    for _s, p in files:
        total_per_crop[crop_of(p.name)] += 1
    for c in CROPS + ["Unknown"]:
        if total_per_crop[c]:
            print(f"  {c:<12} {per_crop[c]:>6} / {total_per_crop[c]:>6}"
                  f"  ({100*per_crop[c]/total_per_crop[c]:.1f}%)")

    # ---------- CSV ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "duplicates.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "kind", "cross_split", "cross_crop", "split", "crop", "file"])
        gid = 0
        for kind, groups in (("exact", exact_groups), ("near", near_groups)):
            for g in groups:
                splits = {split_of[p] for p in g}
                crops = {crop_of(p.name) for p in g}
                for p in g:
                    w.writerow([gid, kind, len(splits) > 1, len(crops) > 1,
                                split_of[p], crop_of(p.name), p.name])
                gid += 1
    print(f"\nSAVED {out}")


if __name__ == "__main__":
    main()
