"""Build the novelty ladder on clean v12: YOLOv11n-CBAM (exp16) and +WIoU (exp17).

Ported from 04_Code/yolo11n_cbam_train.py and yolo11n_cbam_wiou_train.py, which were NEVER
run anywhere - so they are treated as unvalidated and hardened here:

 1. CBAM is made LAZY (channel count inferred on first forward). The original YAML passes
    CBAM [256] but YOLOv11n's width=0.25 makes the real tensor 64ch - a custom-module /
    channel-scaling mismatch that would crash training. Lazy init sidesteps it entirely and,
    because a SMOKE-TEST forward runs BEFORE train(), the lazily-created conv params exist
    before Ultralytics builds the optimizer, so CBAM actually trains.
 2. A dummy forward asserts the model builds and runs BEFORE the ~6h training - a broken
    YAML fails in seconds, not hours.
 3. Config is IDENTICAL to the exp6b baseline (100 ep, imgsz 640, batch 16, seed 0,
    patience 30, NO time=, NO extra augmentation). The original scripts added mixup /
    copy_paste / label_smoothing / 150 ep, which would confound the attention effect. This
    gives a clean isolation ladder: baseline (exp6b) -> +CBAM (exp16) -> +CBAM+WIoU (exp17).

WIoU (exp17) additionally monkey-patches Ultralytics BboxLoss to replace CIoU with the
author's Wise-IoU v3 (faithful port, including its simplified unweighted form).
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

CBAM = '''# === CBAM module (lazy channel inference) ===
# NOTE: the original scripts inserted CBAM via a hand-authored YAML that double-scaled the
# channels (hardcoded nano widths + scales:n) and crashed C2PSA with key_dim=0. That YAML is
# abandoned. Instead we insert CBAM PROGRAMMATICALLY into the known-good stock yolo11n.pt,
# wrapping the C2PSA layer (index 10) output - no channel bookkeeping, no parser edge cases.
class ChannelAttention(nn.Module):
    def __init__(self, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = None                     # built lazily on first forward
        self.sigmoid = nn.Sigmoid()
    def _build(self, c, device):
        mid = max(1, c // self.reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=False)).to(device)
    def forward(self, x):
        if self.fc is None:
            self._build(x.shape[1], x.device)
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x))

class C2PSA_CBAM(nn.Module):
    """Wraps the stock C2PSA layer and appends CBAM to its output (shape-preserving)."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.cbam = CBAM()
    def forward(self, x):
        return self.cbam(self.base(x))
print("CBAM defined (programmatic C2PSA-wrap insertion).")
'''

WIOU = '''# === WIoU v3: replace CIoU inside Ultralytics BboxLoss (faithful port) ===
class WIoULoss(nn.Module):
    def __init__(self, eps=1e-7, momentum=0.02, wiou_scale=1.0):
        super().__init__()
        self.eps = eps; self.momentum = momentum; self.wiou_scale = wiou_scale
        self.register_buffer("iou_mean", torch.tensor(0.5))
    @staticmethod
    def _area(b): return (b[...,2]-b[...,0]).clamp(0)*(b[...,3]-b[...,1]).clamp(0)
    def _iou(self, p, t):
        x1=torch.max(p[...,0],t[...,0]); y1=torch.max(p[...,1],t[...,1])
        x2=torch.min(p[...,2],t[...,2]); y2=torch.min(p[...,3],t[...,3])
        inter=(x2-x1).clamp(0)*(y2-y1).clamp(0)
        union=self._area(p)+self._area(t)-inter+self.eps
        return inter/union
    def forward(self, pred, target):
        iou=self._iou(pred,target).clamp(0,1)
        with torch.no_grad():
            self.iou_mean.copy_((1-self.momentum)*self.iou_mean+self.momentum*iou.detach().mean())
        beta=(iou.detach()/(self.iou_mean.detach()+self.eps)).clamp(1e-4,1e4)
        beta=beta/beta.mean()
        return (beta*(1.0-iou)).mean()*self.wiou_scale

def _patch_wiou():
    # Replace ONLY the IoU term in the CURRENT BboxLoss.forward with WIoU, keeping this
    # ultralytics version's exact DFL / imgsz-stride handling verbatim. Reimplementing the
    # whole forward broke on the 8.4.x signature change (9 args, self.dfl_loss, imgsz/stride).
    import torch.nn.functional as F
    import ultralytics.utils.loss as L
    from ultralytics.utils.tal import bbox2dist
    _wiou = WIoULoss(momentum=0.02, wiou_scale=1.0)
    def _fwd(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
             target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores[fg_mask].sum(-1)                 # (N,)
        pfg = pred_bboxes[fg_mask]; tfg = target_bboxes[fg_mask]
        _wiou.to(pfg.device)
        if pfg.numel() == 0:
            loss_iou = torch.tensor(0.0, device=pred_bboxes.device)
        else:
            iou = _wiou._iou(pfg, tfg).clamp(0, 1)              # standard IoU (N,)
            with torch.no_grad():
                _wiou.iou_mean.copy_((1 - _wiou.momentum) * _wiou.iou_mean
                                     + _wiou.momentum * iou.detach().mean())
            beta = (iou.detach() / (_wiou.iou_mean.detach() + _wiou.eps)).clamp(1e-4, 1e4)
            beta = beta / beta.mean()                           # WIoU v3 dynamic focus, mean~1
            loss_iou = (beta * (1.0 - iou) * weight).sum() / target_scores_sum
        # ---- DFL block copied verbatim from BboxLoss.forward (8.4.x) ----
        weight_kd = target_scores[fg_mask].sum(-1, keepdim=True)
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                     target_ltrb[fg_mask]) * weight_kd
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]; target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]; pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask],
                                  reduction="none").mean(-1, keepdim=True) * weight_kd)
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl
    L.BboxLoss.forward = _fwd
    print("[patch] BboxLoss IoU term -> WIoU v3 (DFL kept verbatim)")
_patch_wiou()
'''

LOCATE = '''# === Locate v12 ===
cands = [p for p in glob.glob("/kaggle/input/**/data.yaml", recursive=True) if "v12" in p]
assert cands, "merged-dataset-v12 not found"
V12 = str(Path(cands[0]).parent); print("V12 root:", V12)
v12_yaml = "/kaggle/working/data_v12.yaml"
yaml.safe_dump({"path": V12, "train": "images/train", "val": "images/valid",
                "test": "images/test", "nc": NC, "names": NAMES},
               open(v12_yaml, "w"), sort_keys=False)
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

BUILD_SMOKE = '''# === Insert CBAM into stock yolo11n.pt (after C2PSA) + SMOKE TEST ===
model = YOLO("yolo11n.pt")            # known-good pretrained nano architecture
layers = model.model.model
c2psa = layers[10]
assert type(c2psa).__name__ == "C2PSA", f"layer 10 is {type(c2psa).__name__}, expected C2PSA"
wrap = C2PSA_CBAM(c2psa)
# preserve Ultralytics routing attributes so BaseModel.forward still works
wrap.f, wrap.i = c2psa.f, c2psa.i
wrap.type = type(c2psa).__name__ + "+CBAM"
wrap.np = sum(p.numel() for p in wrap.parameters())
layers[10] = wrap
print("CBAM inserted after C2PSA (layer 10).")

# SMOKE TEST: forward a dummy batch BEFORE the ~6h train. Proves the model builds and runs,
# and triggers CBAM's lazy conv creation so those params exist before Ultralytics builds the
# optimizer -> CBAM actually gets trained.
model.model.eval()
with torch.no_grad():
    _ = model.model(torch.zeros(1, 3, 640, 640))
n_cbam = sum(p.numel() for n_, p in model.model.named_parameters() if "cbam" in n_)
print(f"SMOKE TEST passed. CBAM params present: {n_cbam}")
assert n_cbam > 0, "CBAM params missing after smoke forward - lazy build failed"
model.model.train()
'''

def train_cell(run):
    return f'''# === Train at EXACT exp6b baseline config (isolates the architecture change) ===
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

REPORT = '''# === Report ===
import pandas as pd
df = pd.read_csv(f"/kaggle/working/runs/{RUN}/results.csv"); df.columns=[c.strip() for c in df.columns]
print(f"epochs completed: {len(df)}/100")
for g in ("A_whole_leaf","B_lesion"):
    sel=[r for r in rows if r[2]==g]
    if sel: print(f"{g}: n={len(sel)} mAP50={sum(r[5] for r in sel)/len(sel):.3f} recall={sum(r[4] for r in sel)/len(sel):.3f}")
allr=[r for r in rows if r[1]=="ALL"][0]
print(f"ALL: mAP50={allr[5]:.3f} recall={allr[4]:.3f}")
print()
print("NOVELTY LADDER (clean v12, nano, 100 ep, identical config):")
print("  baseline   (exp6b) : ALL mAP50=0.8975 recall=0.8697 | A=0.9695 B=0.7716")
print("  +CBAM      (exp16) : this kernel or see it")
print("  +CBAM+WIoU (exp17) : this kernel or see it")
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


SPECS = [
    ("vld-exp16-v12-cbam", "VLD Exp16 v12 CBAM", "cbam_v12", "perclass_v12_cbam.csv",
     "v12_cbam_100ep", False),
    ("vld-exp17-v12-cbam-wiou", "VLD Exp17 v12 CBAM WIoU", "cbam_wiou_v12",
     "perclass_v12_cbam_wiou.csv", "v12_cbam_wiou_100ep", True),
]

for slug, title, run, csv_name, tag, with_wiou in SPECS:
    cells = [SETUP, CBAM]
    if with_wiou:
        cells.append(WIOU)
    cells += [LOCATE, MD5, BUILD_SMOKE, train_cell(run), val_cell(csv_name, tag), REPORT]
    d = HERE / slug
    d.mkdir(exist_ok=True)
    fname = f"{slug}.ipynb"
    json.dump(nb(cells), open(d / fname, "w", encoding="utf-8"), indent=1)
    json.dump(meta(slug, title, fname), open(d / "kernel-metadata.json", "w", encoding="utf-8"), indent=2)
    print("built", d / fname)
