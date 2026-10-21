"""Build vld-exp15-v12-rtdetr: RT-DETR-l transformer baseline on clean v12.

Port of v10 exp2 (vld-exp2-rt-detr-baseline), with the time= confound removed.

HARDWARE REALITY: RT-DETR-l is ~32M params and SLOW - the v10 run managed only 24 epochs
in 10.5h (26.3 min/ep). On v12 (~20% less data) that is ~20.5 min/ep. To finish cleanly
inside one 12h Kaggle session (a hard kill DISCARDS /working output), the budget is capped:

    epochs = 30   (~10.25h at the v12 rate, comfortable margin), NO time=, patience=100.

DISCLOSURE the paper MUST make: 30 epochs is compute-limited, not convergence. RT-DETR
typically needs 72+ epochs, so this is "RT-DETR-l under a fixed ~10h budget" and is NOT a
fair train-to-convergence comparison against the 60-100 epoch YOLO runs. Frame it as a
budget-matched baseline or drop it - do not claim RT-DETR is simply worse.
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
from ultralytics import RTDETR
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

TRAIN = '''# === Train RT-DETR-l on clean v12 - compute-limited 30-epoch budget ===
# NO time= (it rescales epochs). RT-DETR-l ~= 20.5 min/ep on v12 -> 30 ep ~= 10.25h, fits a
# 12h session with margin. batch=8 (RT-DETR-l needs more VRAM than YOLO at 640). patience=100
# so no early stop. 30 ep is compute-limited, NOT converged - disclose in the paper.
RUN = "rtdetr_l_v12"
model = RTDETR("rtdetr-l.pt")
model.train(data=v12_yaml, epochs=30, imgsz=640, batch=8, device=0,
            amp=True, patience=100, seed=0,
            project="/kaggle/working/runs", name=RUN)
'''

VAL = '''# === Per-class validation on clean v12 val set ===
def val_perclass(weights, data_yaml, tag, split="val"):
    b = RTDETR(weights).val(data=data_yaml, split=split, imgsz=640, verbose=False).box
    rows = []
    for i, ci in enumerate(b.ap_class_index):
        ci = int(ci); name = NAMES[ci] if ci < len(NAMES) else str(ci)
        rows.append([tag, name, group_of(name),
                     round(float(b.p[i]),4), round(float(b.r[i]),4),
                     round(float(b.ap50[i]),4), round(float(b.ap[i]),4)])
    rows.append([tag, "ALL", "ALL", round(float(b.mp),4), round(float(b.mr),4),
                 round(float(b.map50),4), round(float(b.map),4)])
    return rows

rows = val_perclass(f"/kaggle/working/runs/{RUN}/weights/best.pt", v12_yaml, "v12_rtdetr_30ep")
out_csv = "/kaggle/working/perclass_v12_rtdetr.csv"
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
wall = float(df["time"].iloc[-1]) / 3600 if "time" in df.columns else float("nan")
print(f"epochs completed: {len(df)}/30  wall: {wall:.2f}h  (COMPUTE-LIMITED, not converged)")
for g in ("A_whole_leaf","B_lesion"):
    sel = [r for r in rows if r[2] == g]
    if sel:
        print(f"{g}: n={len(sel)} mAP50={sum(r[5] for r in sel)/len(sel):.3f} "
              f"recall={sum(r[4] for r in sel)/len(sel):.3f}")
allr = [r for r in rows if r[1]=="ALL"][0]
print(f"ALL: mAP50={allr[5]:.3f} recall={allr[4]:.3f}")
print()
print("COMPARE (clean v12): YOLOv11n 100ep=0.8975 | n/s/m 60ep = 0.881/0.904/0.907")
print("NOTE: RT-DETR here is 30 ep (compute-limited) vs YOLO 60-100 ep - NOT convergence-matched")
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


slug, title = "vld-exp15-v12-rtdetr", "VLD Exp15 v12 RTDETR"
fname = f"{slug}.ipynb"
cells = [SETUP, LOCATE, MD5, TRAIN, VAL, REPORT]
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
