"""Two further manuscript figures.

fig0_annotation_examples.png
    What annotation density looks like. Three whole-leaf-annotated classes on the top row,
    three lesion-annotated classes on the bottom, with the released boxes drawn on the image
    and the class's mean boxes-per-image printed. This is the figure that makes the paper's
    central variable legible without reading a number.

fig4_test_result.png
    The held-out test result: baseline against the selected system, mean of three seeds with
    the seed standard deviation as the error bar, and the Welch p-value over each pair.

Both read only released label files and the finished result tables, so neither can drift from
the analysis. Run after build_paper_assets.py.

Potato labels are polygons and are drawn as their bounding box (labels.py), which is what the
detector was trained on. Before 2026-10-21 the potato panels read polygon vertices as box sizes.

    qpy build_extra_figs.py [--dataset v12]   # v13 -> paper/, v12 -> paper_v12/
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import paper_data as P
from classes import NAMES
from labels import read_boxes

D = P.dataset(P.add_dataset_arg(argparse.ArgumentParser()).parse_args().dataset)
OUT = D.out
OUT.mkdir(exist_ok=True)
DATA = D.root
IDX = {n: i for i, n in enumerate(NAMES)}
FIG_DPI = 600

# top row: whole-leaf convention; bottom row: lesion convention
WHOLE = ["tomato_curl_virus", "chilli_Healthy-Leafs", "cauliflower_downy_mildew"]
LESION = ["potato_blackleg", "cucumber_Bacterial Wilt", "potato_early_blight"]


def load_labels(stem):
    return read_boxes(DATA / "labels" / "train" / f"{stem}.txt")


def pick(cls, target, median_area):
    """Pick a training image for `cls` whose box count is closest to the class mean.

    Representativeness matters here: choosing the most heavily annotated image would
    overstate the class's density and misrepresent what the figure claims to show. Among images
    with that count, prefer the one whose median box area is closest to the class median
    (added 2026-10-21: by count alone, potato_early_blight's panel showed 4 near-leaf-sized boxes
    for a class whose median box covers 1.3% of the image). Remaining ties are broken towards
    the larger, less padded image so the panel is legible.
    """
    ci = IDX[cls]
    best, best_key = None, None
    for lab in sorted((DATA / "labels" / "train").glob("*.txt")):
        rows = load_labels(lab.stem)
        if not rows or any(r[0] != ci for r in rows):
            continue
        img = next((DATA / "images" / "train").glob(lab.stem + ".*"), None)
        if img is None:
            continue
        with Image.open(img) as im:
            w, h = im.size
            small = im.convert("L").resize((32, 32))
        # Several source datasets ship letterboxed images. Padding bars survive into the
        # figure and read as part of the leaf, so prefer near-square, unpadded frames.
        px = list(small.getdata())  # noqa: PIL<14
        border = [px[i] for i in range(32)] + [px[992 + i] for i in range(32)] + \
                 [px[r * 32] for r in range(32)] + [px[r * 32 + 31] for r in range(32)]
        letterbox = sum(1 for v in border if v < 18) / len(border)
        squareness = min(w, h) / max(w, h)
        # log ratio, so closeness means the same for 1%-of-image lesions and whole leaves
        img_med = max(float(np.median([r[3] * r[4] for r in rows])), 1e-6)
        area_fit = -round(abs(np.log(img_med / median_area)), 1)
        key = (-abs(len(rows) - target), area_fit, -round(letterbox, 2), round(squareness, 2),
               w * h)
        if best_key is None or key > best_key:
            best, best_key = (lab.stem, rows), key
    return best


def deletterbox(im):
    """Crop uniform black padding bars added by the source dataset's export.

    Returns the cropped image and the (x0, y0) pixel offset, so annotation boxes -- which are
    expressed relative to the padded frame -- can be shifted to match. Only padding is removed;
    no image content is cropped.
    """
    g = np.asarray(im.convert("L"))
    rows = np.where(g.max(axis=1) > 25)[0]
    cols = np.where(g.max(axis=0) > 25)[0]
    if rows.size == 0 or cols.size == 0:
        return im, 0, 0
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    if (x1 - x0) * (y1 - y0) < 0.25 * im.size[0] * im.size[1]:
        return im, 0, 0
    return im.crop((x0, y0, x1, y1)), x0, y0


def squareify(im, pad=(255, 255, 255)):
    """Pad to a square canvas so every panel shares one aspect ratio without distortion."""
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), pad)
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas, (side - w) // 2, (side - h) // 2


def panel(ax, cls, dens, median_area, lesion):
    got = pick(cls, target=round(dens), median_area=median_area)
    if got is None:
        ax.axis("off")
        return
    stem, rows = got
    img_path = next((DATA / "images" / "train").glob(stem + ".*"))
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    im, cx, cy = deletterbox(im)
    sq, ox, oy = squareify(im)
    ax.imshow(sq)
    colour = "#d62728" if lesion else "#1f77b4"
    for _, xc, yc, w, h in rows:
        ax.add_patch(mpatches.Rectangle(((xc - w / 2) * W - cx + ox,
                                         (yc - h / 2) * H - cy + oy),
                                        w * W, h * H,
                                        fill=False, edgecolor=colour, linewidth=1.4))
    n = len(rows)
    ax.set_title(f"{cls.replace('_', ' ')}\n{n} box{'' if n == 1 else 'es'}"
                 f"  (class mean {dens:.2f} per image)", fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(colour); sp.set_linewidth(2)


def annotation_examples():
    prot = pd.read_csv(D.protocol).set_index("class")
    dens, med = prot["boxes_per_img"], prot["median_area"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.6))
    for j, c in enumerate(WHOLE):
        panel(axes[0, j], c, dens[c], med[c], lesion=False)
    for j, c in enumerate(LESION):
        panel(axes[1, j], c, dens[c], med[c], lesion=True)
    axes[0, 0].set_ylabel("whole-leaf convention", fontsize=9, color="#1f77b4")
    axes[1, 0].set_ylabel("lesion convention", fontsize=9, color="#d62728")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"fig0_annotation_examples.{ext}",
                    dpi=FIG_DPI if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("wrote fig0_annotation_examples")


def test_result():
    t = pd.read_csv(OUT / "table3_test.csv")
    val = t[~t.system.str.contains("Welch|Holm")].set_index("system")
    pv = t[t.system.str.contains("Welch")].set_index("system")
    cols = ["ALL", "A", "B", "recall", "mAP50_95"]
    labels = ["class mean\nmAP@50", "Group A\nmAP@50", "Group B\nmAP@50", "recall", "mAP@50-95"]

    def split(s):
        m, sd = s.split("+/-")
        return float(m), float(sd)

    base = [split(val.loc["baseline", c]) for c in cols]
    sysm = [split(val.loc["s@1024 + TTA", c]) for c in cols]
    p = [float(pv.loc["s@1024 + TTA: Welch p vs baseline", c]) for c in cols]
    hp = t[t.system.str.contains("Holm")].set_index("system")
    holm = [float(hp.loc["s@1024 + TTA: Holm p", c]) for c in cols]

    x = np.arange(len(cols))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - 0.19, [b[0] for b in base], 0.38, yerr=[b[1] for b in base], capsize=3,
           label="baseline (YOLOv11n, 640 px)", color="#9e9e9e")
    ax.bar(x + 0.19, [s[0] for s in sysm], 0.38, yerr=[s[1] for s in sysm], capsize=3,
           label="YOLOv11s, 1024 px + TTA", color="#2e7d32")
    # The star follows the Holm-adjusted p (five metrics), the paper's significance criterion.
    for i, (pp, hh) in enumerate(zip(p, holm)):
        top = max(base[i][0] + base[i][1], sysm[i][0] + sysm[i][1])
        ax.text(x[i], top + 0.018, f"p = {pp:.3f}\nHolm {hh:.3f}" + ("*" if hh < 0.05 else ""),
                ha="center", fontsize=7,
                fontweight="bold" if hh < 0.05 else "normal")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("score on the held-out test split")
    ax.set_ylim(0, 1.16)   # room for the two-line p labels
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.3, axis="y")
    ax.set_title("Held-out test split: mean of three seeds, error bars are seed sd", fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(OUT / f"fig4_test_result.{ext}",
                    dpi=FIG_DPI if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print("wrote fig4_test_result   (* marks Holm-adjusted p < 0.05)")


if __name__ == "__main__":
    annotation_examples()
    test_result()
