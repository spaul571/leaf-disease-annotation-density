"""Build vld-exp14-v12-naive: naive resample (oversample rare + UNDERSAMPLE heavy) on v12.

Port of v10 exp1 (vld-exp1-imbalance-nano). Every primary class sampled to TARGET=400:
undersample if it has more, oversample-with-replacement if fewer. This is the arm that was
net-negative on v10 (undersampling threw away majority cucumber data). It completes the
4-arm imbalance table on clean v12:

  exp6b baseline (no resampling)      0.8975
  exp7  oversample-only (keep major)  0.9023   <- the good data-level arm
  exp14 naive over+under (THIS)       ?
  exp13 focal (loss-level)            ?

Config identical to exp6b: 100 ep, imgsz 640, batch 16, seed 0, patience 30, NO time=.
valid/test come from clean v12, untouched. Leakage assert on the built train set too.
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

import os, glob, csv, random, shutil, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import yaml, torch
from ultralytics import YOLO
_x = torch.zeros(1, device="cuda"); assert float((_x+1).cpu()) == 1.0
print("Torch:", torch.__version__, "| CUDA OK on", torch.cuda.get_device_name(0), flush=True)
random.seed(42)

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

LOCATE = '''# === Locate v12 ===
cands = [p for p in glob.glob("/kaggle/input/**/data.yaml", recursive=True) if "v12" in p]
assert cands, "merged-dataset-v12 not found"
V12 = str(Path(cands[0]).parent); print("V12 root:", V12)
'''

MD5 = '''# === Leakage helpers ===
def md5(p, chunk=1<<20):
    h = hashlib.md5()
    with open(p,"rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()
def hashes_of(d): return {md5(p) for p in glob.glob(d + "/*")}
VAL_H = hashes_of(f"{V12}/images/valid"); TEST_H = hashes_of(f"{V12}/images/test")
def assert_clean(train_dir, tag):
    tr = hashes_of(train_dir)
    lv, lt = len(tr & VAL_H), len(tr & TEST_H)
    print(f"[{tag}] leak train<->valid: {lv} | train<->test: {lt}")
    assert lv == 0 and lt == 0, f"LEAKAGE in {tag}"
assert_clean(f"{V12}/images/train", "v12_original")
'''

BUILD = '''# === Build NAIVE balanced train set: every primary class -> TARGET=400 (under OR over) ===
TARGET_IMAGES = 400
TRAIN_LBL = Path(V12)/"labels"/"train"
TRAIN_IMG = Path(V12)/"images"/"train"
OUT = Path("/kaggle/working/balanced")
if OUT.exists(): shutil.rmtree(OUT)
(OUT/"images"/"train").mkdir(parents=True, exist_ok=True)
(OUT/"labels"/"train").mkdir(parents=True, exist_ok=True)

def img_for(stem):
    for e in (".jpg",".jpeg",".png",".JPG",".PNG"):
        p = TRAIN_IMG/(stem+e)
        if p.exists(): return p
    return None

pool = defaultdict(list)
for lf in TRAIN_LBL.glob("*.txt"):
    cnt = Counter()
    with open(lf) as f:
        for line in f:
            line = line.strip()
            if line: cnt[int(line.split()[0])] += 1
    if not cnt or img_for(lf.stem) is None:
        continue
    pool[cnt.most_common(1)[0][0]].append(lf.stem)

random.seed(42)
chosen = []
print(f"{'class':<38}{'orig':>6}{'->':>4}{'target':>7}")
for c in range(NC):
    stems = sorted(pool.get(c, []))
    if not stems:
        print("WARN class", c, NAMES[c], "no primary-class images"); continue
    if len(stems) >= TARGET_IMAGES:
        chosen += [(s,0) for s in random.sample(stems, TARGET_IMAGES)]   # UNDERSAMPLE
    else:
        chosen += [(s,0) for s in stems]                                 # keep all
        chosen += [(s,j+1) for j,s in enumerate(random.choices(stems, k=TARGET_IMAGES-len(stems)))]
    print(f"{NAMES[c]:<38}{len(stems):>6}{'->':>4}{TARGET_IMAGES:>7}")

seen = set()
for s,d in chosen:
    ip = img_for(s); lp = TRAIN_LBL/(s+".txt")
    stem = s if d==0 else f"{s}_dup{d}"
    while stem in seen: stem += "x"
    seen.add(stem)
    shutil.copy(ip, OUT/"images"/"train"/(stem+ip.suffix))
    shutil.copy(lp, OUT/"labels"/"train"/(stem+".txt"))

bc = Counter()
for lf in (OUT/"labels"/"train").glob("*.txt"):
    with open(lf) as f:
        for line in f:
            if line.strip(): bc[int(line.split()[0])] += 1
print("naive-balanced train images:", len(list((OUT/'images'/'train').glob('*'))))
print("annotation imbalance ratio:", round(max(bc.values())/max(1,min(bc.values())),1), "x")
assert_clean(str(OUT/"images"/"train"), "naive_balanced")

bal_yaml = "/kaggle/working/data_naive.yaml"
yaml.safe_dump({"path": str(OUT), "train": "images/train",
                "val": str(Path(V12)/"images"/"valid"),
                "test": str(Path(V12)/"images"/"test"),
                "nc": NC, "names": NAMES},
               open(bal_yaml, "w"), sort_keys=False)
print("wrote", bal_yaml)
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

TRAIN = '''# === Train YOLOv11n on the naive-balanced set (same config as exp6b) ===
RUN = "nano_naive_v12"
model = YOLO("yolo11n.pt")
model.train(data=bal_yaml, epochs=100, imgsz=640, batch=16, device=0,
            amp=True, patience=30, seed=0,
            project="/kaggle/working/runs", name=RUN)
'''

VAL = '''# === Validate on clean v12 val set ===
rows = val_perclass(f"/kaggle/working/runs/{RUN}/weights/best.pt", bal_yaml, "v12_naive_100ep")
out_csv = "/kaggle/working/perclass_v12_naive.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm","class","group","precision","recall","mAP50","mAP50_95"])
    w.writerows(rows)
print("SAVED", out_csv)
'''

REPORT = '''# === Report ===
for g in ("A_whole_leaf","B_lesion"):
    sel = [r for r in rows if r[2] == g]
    if sel:
        print(f"{g}: n={len(sel)} mAP50={sum(r[5] for r in sel)/len(sel):.3f} "
              f"recall={sum(r[4] for r in sel)/len(sel):.3f}")
allr = [r for r in rows if r[1]=="ALL"][0]
print(f"ALL: mAP50={allr[5]:.3f} recall={allr[4]:.3f}")
print()
print("4-arm imbalance table (clean v12, nano, 100 ep):")
print("  baseline        0.8975 / 0.8697")
print("  oversample-only 0.9023 / 0.8652")
print("  naive over+under  THIS RUN (v10 analogue was net-negative)")
print("  focal (exp13)   see that kernel")
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


slug, title = "vld-exp14-v12-naive", "VLD Exp14 v12 Naive Resample"
fname = f"{slug}.ipynb"
cells = [SETUP, LOCATE, MD5, BUILD, VALFN, TRAIN, VAL, REPORT]
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
