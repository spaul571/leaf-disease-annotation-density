"""Build the ARCHITECTURE-NOVELTY runs on clean v12, targeting the diagnosed bottleneck.

Diagnosis from the whole v12 campaign: mAP50 is pinned ~0.90 across model size, architecture
family, imbalance handling, CBAM and WIoU. The deficit lives entirely in the Group B LESION
classes (~0.77) - small, multi-instance lesions. That is a SMALL-OBJECT DETECTION problem,
which architecture CAN address (unlike the annotation-quality component, which it cannot).

So the evidence-based novelty is NOT more attention - it is a P2 detection head:

  exp18  YOLOv11n-P2       : add a stride-4 (P2/4) detection head so Detect runs on
                             {P2,P3,P4,P5} instead of {P3,P4,P5}. High-res head = small lesions.
                             Validated locally: 2.73M params (nano 2.62M), 11.2 GFLOPs, trains.
  exp19  YOLOv11n-P2-WIoU  : P2 head + WIoU box loss (WIoU was the ONE thing that improved
                             localization, +0.024 mAP50-95). Combined shot at the lesion classes.

Config = exp6b baseline (100 ep, imgsz 640, batch 16, seed 0, patience 30, NO time=, no extra
aug) so the ONLY change vs baseline is the architecture. P2 is ~1.7x GFLOPs -> ~10h/100ep,
fits a 12h session. Report per Group A/B so the lesion-class effect is visible.

HONEST EXPECTATION: P2 should help Group B lesion classes (small-object head). It will NOT fix
the bad-label component (1735 conflicting-label groups). If Group B rises materially, that is a
real, mechanistically-motivated novelty. If it does not, that is itself the paper's strongest
statement: the ceiling is annotation, immovable even by a bottleneck-targeted architecture.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# stock yolo11 format (nc:22 + scales:n, values scale DOWN by width 0.25 - NOT pre-scaled,
# so no double-scaling like the abandoned CBAM yaml). 4-head P2 variant.
P2_YAML = '''nc: 22
scales:
  n: [0.50, 0.25, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[19, 22, 25, 28], 1, Detect, [nc]]
'''

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

WIOU = '''# === WIoU v3: replace ONLY the IoU term in the current BboxLoss.forward (8.4.x-safe) ===
import torch.nn.functional as F
import ultralytics.utils.loss as L
from ultralytics.utils.tal import bbox2dist
class WIoULoss(nn.Module):
    def __init__(self, eps=1e-7, momentum=0.02):
        super().__init__(); self.eps=eps; self.momentum=momentum
        self.register_buffer("iou_mean", torch.tensor(0.5))
    @staticmethod
    def _area(b): return (b[...,2]-b[...,0]).clamp(0)*(b[...,3]-b[...,1]).clamp(0)
    def _iou(self, p, t):
        x1=torch.max(p[...,0],t[...,0]); y1=torch.max(p[...,1],t[...,1])
        x2=torch.min(p[...,2],t[...,2]); y2=torch.min(p[...,3],t[...,3])
        inter=(x2-x1).clamp(0)*(y2-y1).clamp(0)
        return inter/(self._area(p)+self._area(t)-inter+self.eps)
_wiou = WIoULoss()
def _fwd(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
         target_scores, target_scores_sum, fg_mask, imgsz, stride):
    weight = target_scores[fg_mask].sum(-1)
    pfg = pred_bboxes[fg_mask]; tfg = target_bboxes[fg_mask]
    _wiou.to(pfg.device)
    if pfg.numel() == 0:
        loss_iou = torch.tensor(0.0, device=pred_bboxes.device)
    else:
        iou = _wiou._iou(pfg, tfg).clamp(0, 1)
        with torch.no_grad():
            _wiou.iou_mean.copy_((1-_wiou.momentum)*_wiou.iou_mean + _wiou.momentum*iou.detach().mean())
        beta = (iou.detach()/(_wiou.iou_mean.detach()+_wiou.eps)).clamp(1e-4,1e4); beta = beta/beta.mean()
        loss_iou = (beta*(1.0-iou)*weight).sum()/target_scores_sum
    weight_kd = target_scores[fg_mask].sum(-1, keepdim=True)
    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max-1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask])*weight_kd
        loss_dfl = loss_dfl.sum()/target_scores_sum
    else:
        target_ltrb = bbox2dist(anchor_points, target_bboxes)
        target_ltrb = target_ltrb*stride
        target_ltrb[...,0::2] /= imgsz[1]; target_ltrb[...,1::2] /= imgsz[0]
        pred_dist = pred_dist*stride
        pred_dist[...,0::2] /= imgsz[1]; pred_dist[...,1::2] /= imgsz[0]
        loss_dfl = (F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)*weight_kd)
        loss_dfl = loss_dfl.sum()/target_scores_sum
    return loss_iou, loss_dfl
L.BboxLoss.forward = _fwd
print("[patch] BboxLoss IoU term -> WIoU v3 (DFL kept verbatim)")
'''

LOCATE = '''# === Locate v12 + write P2 yaml + data yaml ===
cands = [p for p in glob.glob("/kaggle/input/**/data.yaml", recursive=True) if "v12" in p]
assert cands, "merged-dataset-v12 not found"
V12 = str(Path(cands[0]).parent); print("V12 root:", V12)
v12_yaml = "/kaggle/working/data_v12.yaml"
yaml.safe_dump({"path": V12, "train": "images/train", "val": "images/valid",
                "test": "images/test", "nc": NC, "names": NAMES},
               open(v12_yaml, "w"), sort_keys=False)
p2_yaml_path = "/kaggle/working/yolo11n_p2.yaml"
Path(p2_yaml_path).write_text(P2_YAML)
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
lv = len(tr & hashes_of(f"{V12}/images/valid")); lt = len(tr & hashes_of(f"{V12}/images/test"))
print(f"leak train<->valid: {lv} | train<->test: {lt}")
assert lv == 0 and lt == 0, "LEAKAGE - stop"
print("v12 leakage-free.")
'''

BUILD_SMOKE = '''# === Build YOLOv11n-P2 + transfer stock nano weights + SMOKE TEST ===
model = YOLO(p2_yaml_path)
pre = YOLO("yolo11n.pt").model.state_dict(); st = model.model.state_dict(); tx = 0
for k, v in pre.items():
    if k in st and st[k].shape == v.shape:
        st[k] = v; tx += 1
model.model.load_state_dict(st, strict=False)
tot = sum(p.numel() for p in model.model.parameters())
print(f"P2 built | params {tot/1e6:.3f}M (nano 2.62M) | transferred {tx}")
# NOTE: do NOT smoke-test via len(eval-mode forward) - in eval Detect returns (preds, feats),
# i.e. ALWAYS length 2 regardless of head count. Check Detect.nl / strides, and use a
# TRAIN-mode forward, which returns one tensor per detection scale.
det = model.model.model[-1]
strides = [int(s) for s in det.stride]
print(f"Detect nl={det.nl} strides={strides} (expect nl=4, strides [4,8,16,32])")
assert det.nl == 4, f"P2 head missing: Detect.nl={det.nl}"
assert strides == [4, 8, 16, 32], f"unexpected strides {strides}"
model.model.train()
with torch.no_grad():
    out = model.model(torch.zeros(1, 3, 640, 640))
# ultralytics 8.4.x Detect returns a dict {boxes, scores, feats} in train mode.
feats = out["feats"] if isinstance(out, dict) else out
assert len(feats) == 4, f"train-mode forward gave {len(feats)} scales"
assert tuple(feats[0].shape[-2:]) == (160, 160), f"finest feat {tuple(feats[0].shape)} != stride-4"
print("SMOKE TEST passed:", [tuple(f.shape) for f in feats])
'''

def train_cell(run):
    return f'''# === Train at exp6b baseline config (only change vs baseline = the P2 architecture) ===
RUN = "{run}"
model.train(data=v12_yaml, epochs=100, imgsz=640, batch=16, device=0,
            amp=True, patience=30, seed=0,
            project="/kaggle/working/runs", name=RUN)
'''

def val_cell(csv_name, tag):
    return f'''# === Validate on clean v12 val set ===
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
rows = val_perclass(f"/kaggle/working/runs/{{RUN}}/weights/best.pt", v12_yaml, "{tag}")
out_csv = "/kaggle/working/{csv_name}"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm","class","group","precision","recall","mAP50","mAP50_95"]); w.writerows(rows)
print("SAVED", out_csv)
'''

REPORT = '''# === Report (Group B lesion is the target metric) ===
import pandas as pd
df = pd.read_csv(f"/kaggle/working/runs/{RUN}/results.csv"); df.columns=[c.strip() for c in df.columns]
print(f"epochs completed: {len(df)}/100")
for g in ("A_whole_leaf","B_lesion"):
    sel=[r for r in rows if r[2]==g]
    if sel: print(f"{g}: n={len(sel)} mAP50={sum(r[5] for r in sel)/len(sel):.3f} "
                  f"mAP50-95={sum(r[6] for r in sel)/len(sel):.3f} recall={sum(r[4] for r in sel)/len(sel):.3f}")
allr=[r for r in rows if r[1]=="ALL"][0]
print(f"ALL: mAP50={allr[5]:.3f} recall={allr[4]:.3f} mAP50-95={allr[6]:.3f}")
print()
print("TARGET to beat (clean v12, nano, 100 ep baseline):")
print("  baseline   : ALL 0.8975 | A 0.9695 | B_lesion 0.7716 | B recall ~0.687")
print("  best so far: m@60 0.9066 ; CBAM+WIoU 0.8982 (m50-95 0.7322)")
print("  KEY: did Group B lesion mAP50 rise above 0.7716? that is the whole point of P2.")
print()
for r in rows: print(r)
'''

def nb(cells):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": c.splitlines(keepends=True)} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11.0"}},
            "nbformat": 4, "nbformat_minor": 5}

def meta(slug, title, fname):
    return {"id": f"shrikanta571/{slug}", "title": title, "code_file": fname,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
            "dataset_sources": ["shrikanta571/merged-dataset-v12"],
            "kernel_sources": [], "competition_sources": [], "model_sources": []}

# P2_YAML is referenced inside LOCATE via a module-level string injected into the kernel.
P2_DEF = 'P2_YAML = """' + P2_YAML.replace('"', '\\"') + '"""\n'

SPECS = [
    ("vld-exp18-v12-p2", "VLD Exp18 v12 P2", "p2_v12", "perclass_v12_p2.csv", "v12_p2_100ep", False),
    ("vld-exp19-v12-p2-wiou", "VLD Exp19 v12 P2 WIoU", "p2_wiou_v12",
     "perclass_v12_p2_wiou.csv", "v12_p2_wiou_100ep", True),
]

def build_all():
  for slug, title, run, csv_name, tag, with_wiou in SPECS:
    cells = [SETUP + "\n" + P2_DEF]
    if with_wiou:
        cells.append(WIOU)
    cells += [LOCATE, MD5, BUILD_SMOKE, train_cell(run), val_cell(csv_name, tag), REPORT]
    d = HERE / slug
    d.mkdir(exist_ok=True)
    fname = f"{slug}.ipynb"
    json.dump(nb(cells), open(d / fname, "w", encoding="utf-8"), indent=1)
    m = meta(slug, title, fname)
    mpath = d / "kernel-metadata.json"
    if mpath.exists():  # keep id_no - re-pushing an EXISTING kernel needs it
        old = json.load(open(mpath, encoding="utf-8"))
        if "id_no" in old:
            m["id_no"] = old["id_no"]
    json.dump(m, open(mpath, "w", encoding="utf-8"), indent=2)
    print("built", d / fname)

if __name__ == "__main__":  # guarded so P2_YAML/WIOU can be imported without rewriting kernels
    build_all()
