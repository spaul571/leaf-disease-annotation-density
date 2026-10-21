"""Build vld-exp13-v12-focal: focal-loss YOLOv11n on clean v12.

Port of the v10 exp1c (vld-exp1c-focal-nano) onto the exp6b v12 harness. Isolates the
LOSS-level imbalance fix (focal BCE, gamma=1.5 alpha=0.25) with NO data resampling, so it
sits directly beside exp7 (the DATA-level oversample fix) and exp6b (the plain baseline),
all three on identical clean v12 at 100 epochs.

Config matches exp6b exactly: epochs=100, imgsz=640, batch=16, seed=0, patience=30, NO time=.
nano @ 100 ep ~= 6.1h, fits one session.

Point of the run: if focal ALSO fails to lift Group B (as oversample did), the paper closes
"imbalance is not the bottleneck" at BOTH the data and loss level.
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
import torch.nn as nn
import torch.nn.functional as F
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

PATCH = '''# === Patch Ultralytics classification loss: BCE -> Focal (element-wise, drop-in) ===
# v8DetectionLoss does: loss_cls = self.bce(pred_scores, target_scores).sum()/target_scores_sum
# so the replacement MUST return an element-wise tensor of the same shape (reduction="none").
GAMMA, ALPHA = 1.5, 0.25

class FocalBCE(nn.Module):
    def __init__(self, gamma=GAMMA, alpha=ALPHA):
        super().__init__(); self.gamma = gamma; self.alpha = alpha
    def forward(self, pred, target):
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p = pred.sigmoid()
        p_t = target * p + (1 - target) * (1 - p)
        loss = loss * (1.0 - p_t) ** self.gamma
        if self.alpha > 0:
            a_t = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss = loss * a_t
        return loss

import ultralytics.utils.loss as ul
_orig_init = ul.v8DetectionLoss.__init__
def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    self.bce = FocalBCE()
    print(f"[patch] cls loss -> FocalBCE(gamma={GAMMA}, alpha={ALPHA})", flush=True)
ul.v8DetectionLoss.__init__ = _patched_init
print("Focal-loss patch installed.")
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

TRAIN = '''# === Train focal-loss YOLOv11n on ORIGINAL (unresampled) clean v12 ===
# Same config as exp6b baseline; the ONLY difference is the focal cls-loss patch above.
# NO time= (it rescales epochs). 100 ep nano ~= 6.1h.
RUN = "nano_focal_v12"
model = YOLO("yolo11n.pt")
model.train(data=v12_yaml, epochs=100, imgsz=640, batch=16, device=0,
            amp=True, patience=30, seed=0,
            project="/kaggle/working/runs", name=RUN)
'''

VAL = '''# === Validate on clean v12 val set ===
rows = val_perclass(f"/kaggle/working/runs/{RUN}/weights/best.pt", v12_yaml, "v12_focal_100ep")
out_csv = "/kaggle/working/perclass_v12_focal.csv"
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
print(f"epochs completed: {len(df)}/100 (patience=30 may early-stop)")
for g in ("A_whole_leaf","B_lesion"):
    sel = [r for r in rows if r[2] == g]
    if sel:
        print(f"{g}: n={len(sel)} mAP50={sum(r[5] for r in sel)/len(sel):.3f} "
              f"recall={sum(r[4] for r in sel)/len(sel):.3f}")
allr = [r for r in rows if r[1]=="ALL"][0]
print(f"ALL: mAP50={allr[5]:.3f} recall={allr[4]:.3f}")
print()
print("COMPARE (all clean v12, nano, 100 ep):")
print("  exp6b baseline        : ALL mAP50=0.8975 recall=0.8697 | A=0.9695 | B=0.7716")
print("  exp7  oversample-only : ALL mAP50=0.9023 recall=0.8652 | (data-level fix, a wash)")
print("  this  focal           : ALL mAP50=? (loss-level fix)")
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


slug, title = "vld-exp13-v12-focal", "VLD Exp13 v12 Focal"
fname = f"{slug}.ipynb"
cells = [SETUP, PATCH, LOCATE, MD5, VALFN, TRAIN, VAL, REPORT]
d = HERE / slug
d.mkdir(exist_ok=True)
json.dump(nb(cells), open(d / fname, "w", encoding="utf-8"), indent=1)
json.dump({"id": f"shrikanta571/{slug}", "title": title, "code_file": fname,
           "language": "python", "kernel_type": "notebook", "is_private": True,
           "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "keywords": ["gpu"], "dataset_sources": ["shrikanta571/merged-dataset-v12"],
           "kernel_sources": [], "competition_sources": [], "model_sources": []},
          open(d / "kernel-metadata.json", "w", encoding="utf-8"), indent=2)
print("built", d / fname)
