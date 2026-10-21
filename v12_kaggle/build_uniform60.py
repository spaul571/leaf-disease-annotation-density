"""Build the CONTROLLED YOLOv11 n/s/m model-size ablation on clean v12.

Why this exists: exp6b/exp8/exp9 ended up at 100/122/60 epochs respectively, because
`time=` does NOT act as a ceiling - Ultralytics REPLACES the epoch schedule with a
wall-clock schedule and rescales in BOTH directions (s expanded past its 100 target to
122; m truncated to 60). Any model-size comparison across those runs is confounded.

Controlled design here:
  * epochs=60 for all three - the largest uniform budget yolo11m can finish inside a
    single 12h Kaggle session (measured: m = 11.0 min/ep -> 60 ep = ~11.0h).
  * NO time= parameter. At all. It is the confound.
  * patience=100 (> epochs) so early stopping can NEVER fire. Without this a model could
    stop at ~45 epochs and reintroduce unequal budgets. All three run exactly 60.
    best.pt still tracks the best epoch, so this costs nothing but compute.
  * everything else identical: imgsz 640, batch 16, seed 0, amp, no resampling.

Stated limitation for the paper: yolo11s peaked at epoch 108 in the (confounded) 122-epoch
run, so 60 epochs likely undertrains n and s. The comparison is fair (equal budgets) but is
a fixed-budget comparison, not a train-to-convergence comparison.

Projected: n ~3.7h, s ~5.2h, m ~11.0h.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

SETUP = '''# === Setup (T4 expected; guard for P100 sm_60) ===
import subprocess, sys
gpu = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print("Assigned GPU:", gpu, flush=True)
if "P100" in gpu:
    print("P100 -> installing sm_60-compatible torch (cu121)...", flush=True)
    subprocess.run([sys.executable,"-m","pip","install","-q",
        "torch==2.5.1","torchvision==0.20.1",
        "--index-url","https://download.pytorch.org/whl/cu121"], check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","ultralytics"], check=False)

import os, glob, csv, hashlib
from pathlib import Path
import yaml, torch
from ultralytics import YOLO
_x = torch.zeros(1, device="cuda"); assert float((_x+1).cpu()) == 1.0
print("Torch:", torch.__version__, "| CUDA OK on", torch.cuda.get_device_name(0), flush=True)

NC = 22
NAMES = [
    "bean_angular_leaf_spot","bean_bean_rust","bean_healthy",
    "cauliflower_bacterial_spot_rot","cauliflower_black_rot",
    "cauliflower_downy_mildew","cauliflower_no_disease",
    "chilli_Curling-Leafs","chilli_Healthy-Leafs","chilli_Leaf Spot-Leafs",
    "chilli_Leveillula taurica-Leafs","chilli_Mozaik Leaf",
    "cucumber_Antraknosa","cucumber_Bacterial Wilt","cucumber_Downy Mildew",
    "potato_healthy","potato_blackleg","potato_late_blight","potato_early_blight",
    "tomato_bacterial_spot","tomato_curl_virus","tomato_two_spotted_spider_mite",
]
GROUP_B_LESION = {
    "potato_healthy","cucumber_Antraknosa","cucumber_Bacterial Wilt",
    "potato_late_blight","potato_early_blight","bean_bean_rust",
    "potato_blackleg","bean_angular_leaf_spot",
}
group_of = lambda c: "B_lesion" if c in GROUP_B_LESION else "A_whole_leaf"
'''

LOCATE = '''# === Locate v12 and rewrite data.yaml for the Kaggle mount ===
cands = [p for p in glob.glob("/kaggle/input/**/data.yaml", recursive=True) if "v12" in p]
assert cands, "merged-dataset-v12 not found under /kaggle/input"
V12 = str(Path(cands[0]).parent); print("V12 root:", V12)
v12_yaml = "/kaggle/working/data_v12.yaml"
yaml.safe_dump({"path": V12, "train": "images/train", "val": "images/valid",
                "test": "images/test", "nc": NC, "names": NAMES},
               open(v12_yaml, "w"), sort_keys=False)
for s in ["train","valid","test"]:
    print(f"  {s:<6} {len(glob.glob(f'{V12}/images/{s}/*')):>6} images")
'''

MD5 = '''# === Leakage guard ===
def md5(p, chunk=1<<20):
    h = hashlib.md5()
    with open(p,"rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()
def hashes_of(d): return {md5(p) for p in glob.glob(d + "/*")}
tr = hashes_of(f"{V12}/images/train")
lv = len(tr & hashes_of(f"{V12}/images/valid"))
lt = len(tr & hashes_of(f"{V12}/images/test"))
print(f"leak train<->valid: {lv} | train<->test: {lt}")
assert lv == 0 and lt == 0, "LEAKAGE - stop"
print("v12 leakage-free.")
'''

VALFN = '''# === Per-class validation helper ===
def val_perclass(weights, data_yaml, tag, split="val"):
    b = YOLO(weights).val(data=data_yaml, split=split, imgsz=640, verbose=False).box
    rows = []
    for i, ci in enumerate(b.ap_class_index):
        ci = int(ci); name = NAMES[ci] if ci < len(NAMES) else str(ci)
        rows.append([tag, name, group_of(name),
                     round(float(b.p[i]),4), round(float(b.r[i]),4),
                     round(float(b.ap50[i]),4), round(float(b.ap[i]),4)])
    rows.append([tag, "ALL", "ALL", round(float(b.mp),4), round(float(b.mr),4),
                 round(float(b.map50),4), round(float(b.map),4)])
    return rows
'''


def train_cell(model_pt, run):
    return f'''# === Train {model_pt} on clean v12 - CONTROLLED 60-epoch budget ===
# NO time= parameter: it replaces the epoch schedule and rescales in both directions
# (that is what pushed yolo11s to 122 ep and cut yolo11m to 60 ep in the earlier runs).
# patience=100 > epochs=60 so early stopping can never fire -> all three models get an
# IDENTICAL 60-epoch budget. best.pt still tracks the best epoch.
RUN = "{run}"
model = YOLO("{model_pt}")
model.train(data=v12_yaml, epochs=60, imgsz=640, batch=16, device=0,
            amp=True, patience=100, seed=0,
            project="/kaggle/working/runs", name=RUN)
'''


def val_cell(csv_name, tag):
    return f'''# === Validate on clean v12 val set ===
rows = val_perclass(f"/kaggle/working/runs/{{RUN}}/weights/best.pt", v12_yaml, "{tag}")
out_csv = "/kaggle/working/{csv_name}"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm","class","group","precision","recall","mAP50","mAP50_95"])
    w.writerows(rows)
print("SAVED", out_csv)
'''

REPORT = '''# === Report ===
import pandas as pd
df = pd.read_csv(f"/kaggle/working/runs/{RUN}/results.csv")
df.columns = [c.strip() for c in df.columns]
n_ep = len(df)
wall = float(df["time"].iloc[-1]) / 3600 if "time" in df.columns else float("nan")
mcol = [c for c in df.columns if "mAP50(B)" in c and "95" not in c][0]
print(f"epochs completed: {n_ep}/60  wall: {wall:.2f}h")
assert n_ep == 60, f"EXPECTED EXACTLY 60 EPOCHS, GOT {n_ep} - budget not controlled"
print(f"best mAP50 during training: {df[mcol].max():.4f} @ epoch {int(df[mcol].idxmax())+1}")
for g in ("A_whole_leaf","B_lesion"):
    sel = [r for r in rows if r[2] == g]
    if sel:
        print(f"{g}: n={len(sel)} mAP50={sum(r[5] for r in sel)/len(sel):.3f} "
              f"recall={sum(r[4] for r in sel)/len(sel):.3f}")
allr = [r for r in rows if r[1]=="ALL"][0]
print(f"ALL: mAP50={allr[5]:.3f} recall={allr[4]:.3f}")
print()
print("CONFOUNDED earlier runs (unequal budgets - do NOT compare against these):")
print("  n 100ep=0.8975 | s 122ep=0.9135 | m 60ep(time-capped)=0.9035")
print()
for r in rows: print(r)
'''


def nb(cells):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": c.splitlines(keepends=True)} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11.0"}},
            "nbformat": 4, "nbformat_minor": 5}


def meta(slug, title, fname):
    return {"id": f"shrikanta571/{slug}", "title": title, "code_file": fname,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "keywords": ["gpu"], "dataset_sources": ["shrikanta571/merged-dataset-v12"],
            "kernel_sources": [], "competition_sources": [], "model_sources": []}


# slug MUST equal slugify(title) or Kaggle silently creates a second kernel.
SPECS = [
    ("vld-exp10-v12-nano-60ep", "VLD Exp10 v12 Nano 60ep", "yolo11n.pt",
     "nano60", "perclass_v12_nano60.csv", "v12_nano_60ep"),
    ("vld-exp11-v12-small-60ep", "VLD Exp11 v12 Small 60ep", "yolo11s.pt",
     "small60", "perclass_v12_small60.csv", "v12_small_60ep"),
    ("vld-exp12-v12-medium-60ep", "VLD Exp12 v12 Medium 60ep", "yolo11m.pt",
     "medium60", "perclass_v12_medium60.csv", "v12_medium_60ep"),
]

for slug, title, model_pt, run, csv_name, tag in SPECS:
    cells = [SETUP, LOCATE, MD5, VALFN,
             train_cell(model_pt, run), val_cell(csv_name, tag), REPORT]
    d = HERE / slug
    d.mkdir(exist_ok=True)
    fname = f"{slug}.ipynb"
    json.dump(nb(cells), open(d / fname, "w", encoding="utf-8"), indent=1)
    json.dump(meta(slug, title, fname), open(d / "kernel-metadata.json", "w",
                                             encoding="utf-8"), indent=2)
    print("built", d / fname)
