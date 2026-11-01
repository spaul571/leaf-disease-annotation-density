"""Generate a VS Code notebook for a local v12 training arm.

Every YOLO run is delivered as an .ipynb the user runs themselves in VS Code (live tqdm bars,
visible metrics), not as a backgrounded script. The notebook pins the `qml` kernel so no manual
kernel picking is needed.

    qpy build_nb.py --arm baseline --seed 0
    qpy build_nb.py --arm p2 --seed 1 --imgsz 1024 --batch 8
    qpy build_nb.py --dataset v13 --arm small --seed 0 --imgsz 1024 --workers 2
    code -r nb/local_baseline_s0_640.ipynb

--dataset v13 trains on merged-dataset-v13 (v12 without its rotated/flipped twins, see
../TWIN_LEAKAGE_PLAN.md). Its runs, notebooks and result files carry the prefix v13_ instead of
local_, so they can never overwrite the v12 results that build_paper_assets.py reads.

Training config defaults to exp6b EXACTLY (100 ep / 640 / batch 16 / patience 30 / amp) so local
numbers stay comparable to the Kaggle T4 results. Changing imgsz or batch breaks that
comparability on purpose - use it only for declared separate arms.
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent

HEADER = '''# {title}

**Arm:** `{arm}` | **seed** {seed} | **{epochs} ep** | **imgsz {imgsz}** | **batch {batch}** | **workers {workers}**

Config matches Kaggle `exp6b` (the controlled baseline) unless imgsz/batch were overridden.

Reference to beat (clean v12, YOLOv11n, 100 ep, Kaggle T4):

| metric | value |
|---|---|
| ALL mAP@50 | 0.8975 |
| Group A whole-leaf | 0.9695 |
| **Group B lesion** | **0.7716** |
| mAP@50-95 | 0.7268 |
| recall | 0.8697 |

Results so far: P2 head **hurt** Group B (0.7424), P2+WIoU hurt more (0.7102). The open question
is whether local hardware reproduces the T4 baseline within +/-0.005 - if not, local arms must be
compared only against a local baseline.

Run all cells. Progress bars are live; the RAM watchdog frees the image cache if memory spikes.
'''

ENV = '''# === Environment check ===
import os, sys, gc, csv, json, time
from pathlib import Path
import torch, psutil
# Plain tqdm, NOT tqdm.auto: inside a notebook kernel tqdm.auto resolves to the ipywidgets
# frontend, and VS Code then blocks on a "Widgets require us to download supporting files from a
# 3rd party website" dialog (@jupyter-widgets/base). The text bar needs no CDN and no network.
from tqdm import tqdm

print("python     :", sys.version.split()[0])
print("torch      :", torch.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "no CUDA - qml venv needs torch cu130"
print("gpu        :", torch.cuda.get_device_name(0),
      "| cc", torch.cuda.get_device_capability(0),
      "| %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
vm = psutil.virtual_memory()
print("system RAM : %.1f GB total, %.1f GB available (%.0f%% used)"
      % (vm.total/1e9, vm.available/1e9, vm.percent))
'''

CONFIG = '''# === Config (exp6b-identical unless overridden) ===
ARM        = "{arm}"
SEED       = {seed}
EPOCHS     = {epochs}
IMGSZ      = {imgsz}
BATCH      = {batch}
WORKERS    = {workers}
PATIENCE   = 30          # exp6b value - do not change, it defines the early-stop budget
CACHE      = {cache!r}   # False | "ram" | "disk"; RAM cache is watchdogged below
RAM_LIMIT  = {ram_limit} # percent of system RAM; above this the image cache is dropped

HERE   = Path(r"{here}")
DATA   = Path(r"{data}")
os.chdir(HERE)   # keep weight downloads / stray ultralytics dirs out of nb/
# workers MUST appear in RUN and TAG whenever it is not the 8 that every published arm used.
# Without it the R1-3 worker-control re-run of baseline/seed-0 lands on
# runs/local_baseline_s0_640/ and on the perclass+summary files that build_paper_assets.py
# turns into the paper tables, destroying the measured baseline it is supposed to be compared to.
_W     = "" if WORKERS == 8 else f"_W{{WORKERS}}"
RUN    = f"{prefix}{{ARM}}_s{{SEED}}_{{IMGSZ}}{{_W}}"
# imgsz MUST appear in TAG for non-640 runs: without it a 1024 arm overwrites the 640
# results CSV of the same arm/seed and pollutes the noise_floor glob.
_SZ    = "" if IMGSZ == 640 else f"_{{IMGSZ}}"
TAG    = f"{prefix}{{ARM}}_{{EPOCHS}}ep_s{{SEED}}{{_SZ}}{{_W}}"
assert DATA.exists(), DATA

NAMES = [
    "bean_angular_leaf_spot", "bean_bean_rust", "bean_healthy",
    "cauliflower_bacterial_spot_rot", "cauliflower_black_rot",
    "cauliflower_downy_mildew", "cauliflower_no_disease",
    "chilli_Curling-Leafs", "chilli_Healthy-Leafs", "chilli_Leaf Spot-Leafs",
    "chilli_Leveillula taurica-Leafs", "chilli_Mozaik Leaf",
    "cucumber_Antraknosa", "cucumber_Bacterial Wilt", "cucumber_Downy Mildew",
    "potato_healthy", "potato_blackleg", "potato_late_blight", "potato_early_blight",
    "tomato_bacterial_spot", "tomato_curl_virus", "tomato_two_spotted_spider_mite",
]
GROUP_B_LESION = {{
    "potato_healthy", "cucumber_Antraknosa", "cucumber_Bacterial Wilt",
    "potato_late_blight", "potato_early_blight", "bean_bean_rust",
    "potato_blackleg", "bean_angular_leaf_spot",
}}
group_of = lambda c: "B_lesion" if c in GROUP_B_LESION else "A_whole_leaf"
REF = {{"ALL": 0.8975, "A_whole_leaf": 0.9695, "B_lesion": 0.7716,
       "recall": 0.8697, "mAP50_95": 0.7268}}   # 0.7268 = exp6b 100ep. 0.7083 is exp10 @60ep.
print("run:", RUN)
'''

WATCHDOG = '''# === tqdm epoch bar + RAM cache watchdog ===
# Ultralytics prints a per-batch bar of its own; this adds an OVERALL epoch bar carrying the
# live metrics, and frees the dataset image cache if system RAM crosses RAM_LIMIT.

_bar = {"h": None, "t0": None}

def _ram():
    return psutil.virtual_memory().percent

def _drop_image_cache(trainer, why):
    """Free the ultralytics in-RAM decoded-image cache (dataset.ims) and collect."""
    freed = 0
    for loader in (getattr(trainer, "train_loader", None), getattr(trainer, "test_loader", None)):
        ds = getattr(loader, "dataset", None)
        ims = getattr(ds, "ims", None)
        if ims:
            freed += sum(1 for x in ims if x is not None)
            ds.ims = [None] * len(ims)
            for attr in ("im_hw0", "im_hw"):
                if hasattr(ds, attr):
                    setattr(ds, attr, [None] * len(ims))
            ds.cache = False          # stop it refilling
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\\n[ram-watchdog] {why}: dropped {freed} cached images -> {_ram():.0f}% used", flush=True)

def on_train_start(trainer):
    _bar["t0"] = time.time()
    # file=sys.stdout keeps the bar out of the stderr stream, which VS Code paints as an error.
    _bar["h"] = tqdm(total=EPOCHS, desc=f"{ARM} s{SEED}", unit="ep", dynamic_ncols=True,
                     file=sys.stdout)
    if _ram() > RAM_LIMIT:
        _drop_image_cache(trainer, f"RAM {_ram():.0f}% > {RAM_LIMIT}% at start")

def on_fit_epoch_end(trainer):
    m = trainer.metrics or {}
    get = lambda k: m.get(k, float("nan"))
    b = _bar["h"]
    if b is not None:
        b.update(1)
        b.set_postfix({
            "mAP50": f"{get('metrics/mAP50(B)'):.4f}",
            "mAP50-95": f"{get('metrics/mAP50-95(B)'):.4f}",
            "recall": f"{get('metrics/recall(B)'):.4f}",
            "RAM%": f"{_ram():.0f}",
            "VRAM": f"{torch.cuda.memory_reserved()/1e9:.1f}G",
        })
    if _ram() > RAM_LIMIT:
        _drop_image_cache(trainer, f"RAM {_ram():.0f}% > {RAM_LIMIT}%")

def on_train_end(trainer):
    if _bar["h"] is not None:
        _bar["h"].close()
    mins = (time.time() - _bar["t0"]) / 60
    print(f"train wall-clock: {mins:.1f} min ({mins/max(EPOCHS,1):.2f} min/ep)")

print(f"watchdog armed at {RAM_LIMIT}% RAM (now {_ram():.0f}%)")
'''

WIOU_CELL = '''# === WIoU v3 patch (same code as the exp19 Kaggle kernel) ===
import torch.nn as nn
sys.path.insert(0, r"__KAGGLE_DIR__")
from build_p2 import WIOU
exec(WIOU)
'''

WIOU_VERIFY = '''# === Post-train verification: was WIoU v3 the box loss? ===
import ultralytics.utils.loss as L
_crit = getattr(model.trainer.model, "criterion", None)
assert _crit is not None, "no criterion on the trained model"
assert type(_crit.bbox_loss).forward.__name__ == "_fwd", "BboxLoss was not the WIoU v3 patch"
print("WIoU v3 was the box loss during training")
'''

# Verbatim from the exp13 Kaggle kernel (v12_kaggle/build_focal.py, PATCH). Copied, not imported:
# build_focal.py writes its Kaggle notebooks at import time.
FOCAL_CELL = '''# === Focal loss patch: classification BCE -> focal BCE (gamma 1.5, alpha 0.25) ===
# v8DetectionLoss does: bce_loss = self.bce(pred_scores, target_scores); loss[1] = bce_loss.sum()/...
# so the replacement MUST return an element-wise tensor of the same shape (reduction="none").
import torch.nn as nn
import torch.nn.functional as F
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

FOCAL_VERIFY = '''# === Post-train verification: was the classification loss focal? ===
_crit = getattr(model.trainer.model, "criterion", None)
assert _crit is not None, "no criterion on the trained model"
assert type(_crit.bce).__name__ == "FocalBCE", f"cls loss was {type(_crit.bce).__name__}, not focal"
print("focal loss was the classification loss during training")
'''

MODEL_RTDETR = '''# === Model: RT-DETR-l (detector family) ===
from ultralytics import RTDETR, YOLO
model = RTDETR("rtdetr-l.pt")
print("params %.3fM" % (sum(p.numel() for p in model.model.parameters())/1e6))
'''

MODEL_P2_RM = '''# === Model: YOLOv11n-P2 (stride-4 head), pretrained weights mapped BY ROLE (Shakib S-6) ===
# Found 2026-11-01: YOLO(<yaml>).train() rebuilds the network from the YAML with fresh weights,
# because Model.train() reuses in-memory weights only for a model loaded from a checkpoint. Every
# earlier P2 run (MODEL_P2 below) therefore trained from scratch. Here yolo11n's tensors are mapped
# by role (448 of 593; only the new P2 layers keep a fresh init), the model is saved as a
# checkpoint and trained from it, and a callback checks inside training that the weights arrived.
from ultralytics import YOLO
sys.path.insert(0, r"__KAGGLE_DIR__")
from build_p2 import P2_YAML

p2_path = HERE / "yolo11n_p2.yaml"
p2_path.write_text(P2_YAML)
LAYER = {**{i: i for i in range(17)}, 23: 17, 24: 18, 25: 19, 26: 20, 27: 21, 28: 22, 29: 23}
BRANCH = {1: 0, 2: 1, 3: 2}     # Detect branches: P2 model (P2, P3, P4, P5) -> yolo11n (P3, P4, P5)

def src_key(k):
    p = k.split("."); i = int(p[1])
    if i not in LAYER:
        return None
    p[1] = str(LAYER[i])
    if i == 29 and p[2] in ("cv2", "cv3"):
        if int(p[3]) not in BRANCH:
            return None
        p[3] = str(BRANCH[int(p[3])])
    return ".".join(p)

torch.manual_seed(SEED)              # the new layers' initialisation follows the run's seed
fresh = YOLO(str(p2_path))
PRE = YOLO("yolo11n.pt").model.state_dict()
st = fresh.model.state_dict(); tx = 0
for k in st:
    s = src_key(k)
    if s and s in PRE and PRE[s].shape == st[k].shape:
        st[k] = PRE[s].clone(); tx += 1
assert tx == 448 and len(st) == 593, f"role mapping changed: {tx}/{len(st)}"
fresh.model.load_state_dict(st)
init_pt = HERE / "runs" / f"{RUN}_init.pt"
init_pt.parent.mkdir(exist_ok=True)
fresh.save(str(init_pt))
model = YOLO(str(init_pt))           # a checkpoint, so train() keeps these weights
assert model.ckpt, "no checkpoint: train() would rebuild the model with random weights"
print(f"params {sum(p.numel() for p in model.model.parameters())/1e6:.3f}M | "
      f"mapped {tx}/{len(st)} tensors -> {init_pt.name}")

det = model.model.model[-1]
strides = [int(s) for s in det.stride]
assert det.nl == 4 and strides == [4, 8, 16, 32], f"P2 head wrong: nl={det.nl} {strides}"

PROBES = {"model.0.conv.weight": "model.0.conv.weight",
          "model.25.cv1.conv.weight": "model.19.cv1.conv.weight",
          "model.29.cv2.1.0.conv.weight": "model.23.cv2.0.0.conv.weight"}

def check_pretrained(trainer):
    """Inside training: the network being trained must hold yolo11n's tensors, by role."""
    sd = trainer.model.state_dict()
    for k, s in PROBES.items():
        assert torch.allclose(sd[k].float().cpu(), PRE[s].float(), atol=1e-3), (
            f"{k} does not hold yolo11n {s}: the pretrained weights did not reach training")
    print("pretrained weights verified inside training:", ", ".join(PROBES), flush=True)

model.add_callback("on_train_start", check_pretrained)
'''

MODEL_BASE = '''# === Model: stock YOLO ===
from ultralytics import YOLO
model = YOLO("__WEIGHTS__")
print("params %.3fM" % (sum(p.numel() for p in model.model.parameters())/1e6))
'''

MODEL_UADH = '''# === Model: YOLOv11n + UADH (uncertainty-aware detection head) ===
# The contribution. P2 failed because it added CAPACITY against noisy labels; UADH adds
# ROBUSTNESS instead: a third head branch predicts a per-anchor log-variance s, and the box loss
# becomes a heteroscedastic NLL  exp(-s)*(1-IoU) + 0.5*s  so the model can discount boxes whose
# annotation is unreliable while paying a price for claiming uncertainty.
from ultralytics import YOLO
sys.path.insert(0, r"__LOCAL_DIR__")
import uadh

uadh.patch_uadh()          # patches Detect.__init__/forward_head and BboxLoss.forward
model = YOLO("yolo11n.pt")
uadh.attach_uadh(model)

det = model.model.model[-1]
with torch.no_grad():
    model.model.train()
    _o = model.model(torch.zeros(2, 3, 640, 640))
assert "unc" in _o, "uncertainty output missing"
assert _o["unc"].abs().max() < 1e-6, "s must be 0 at init (zero-init final conv)"
print("UADH smoke OK | s == 0 at init, so step 0 is exactly the baseline objective")
'''

UADH_VERIFY = '''# === Post-train verification: did the branch SURVIVE and LEARN? ===
# YOLO.train() rebuilds the model from yaml, which silently discards instance-attached modules
# (verified: a branch attached before .train() is gone after). Checking only before training is
# how an arm ends up secretly being a plain baseline - so verify AFTER.
uadh.assert_uadh_alive(model, "after train")
_m = model.trainer.model if hasattr(model, "trainer") else model.model
_m.train()
with torch.no_grad():
    _o = _m(torch.rand(4, 3, 640, 640, device="cuda"))
_s = _o["unc"]
print(f"learned s: mean {_s.mean():+.4f} | sd {_s.std():.4f} | "
      f"range [{_s.min():+.3f}, {_s.max():+.3f}]")
assert _s.std() > 1e-3, "s has no spread - the branch did not learn, arm is a no-op baseline"
print("UADH is alive and learned a non-trivial uncertainty field")
'''

MODEL_P2 = '''# === Model: YOLOv11n-P2 (stride-4 head) + weight transfer + smoke test ===
from ultralytics import YOLO
sys.path.insert(0, r"__KAGGLE_DIR__")
from build_p2 import P2_YAML

p2_path = HERE / "yolo11n_p2.yaml"
p2_path.write_text(P2_YAML)
model = YOLO(str(p2_path))

pre = YOLO("yolo11n.pt").model.state_dict(); st = model.model.state_dict(); tx = 0
for k, v in pre.items():
    if k in st and st[k].shape == v.shape:
        st[k] = v; tx += 1
model.model.load_state_dict(st, strict=False)
print("params %.3fM | transferred %d" % (sum(p.numel() for p in model.model.parameters())/1e6, tx))

# NOTE: eval-mode Detect returns (preds, feats) = ALWAYS len 2, and 8.4.x train-mode returns a
# dict {boxes, scores, feats}. Checking len(forward) is what broke exp18 v1 on Kaggle.
det = model.model.model[-1]
strides = [int(s) for s in det.stride]
assert det.nl == 4 and strides == [4, 8, 16, 32], f"P2 head wrong: nl={det.nl} {strides}"
model.model.train()
with torch.no_grad():
    out = model.model(torch.zeros(1, 3, 640, 640))
feats = out["feats"] if isinstance(out, dict) else out
assert len(feats) == 4 and tuple(feats[0].shape[-2:]) == (160, 160), "P2 smoke failed"
print("P2 smoke OK:", [tuple(f.shape) for f in feats])
'''

TRAIN = '''# === Train ===
for ev, fn in (("on_train_start", on_train_start),
               ("on_fit_epoch_end", on_fit_epoch_end),
               ("on_train_end", on_train_end)):
    model.add_callback(ev, fn)

results = model.train(
    data=str(DATA), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, device=0, amp=True,
    patience=PATIENCE, seed=SEED, workers=WORKERS, cache=CACHE,
    project=str(HERE / "runs"), name=RUN, exist_ok=True,
)
'''

VAL = '''# === Per-class validation on the clean v12 val split ===
best = HERE / "runs" / RUN / "weights" / "best.pt"
b = YOLO(str(best)).val(data=str(DATA), split="val", imgsz=IMGSZ, verbose=False,
                        project=str(HERE / "runs"), name=f"{RUN}_val", exist_ok=True).box

rows = []
for i, ci in enumerate(tqdm(b.ap_class_index, desc="per-class", unit="cls", file=sys.stdout)):
    ci = int(ci); name = NAMES[ci] if ci < len(NAMES) else str(ci)
    rows.append([TAG, name, group_of(name), round(float(b.p[i]), 4), round(float(b.r[i]), 4),
                 round(float(b.ap50[i]), 4), round(float(b.ap[i]), 4)])
rows.append([TAG, "ALL", "ALL", round(float(b.mp), 4), round(float(b.mr), 4),
             round(float(b.map50), 4), round(float(b.map), 4)])

res_dir = HERE / "results"; res_dir.mkdir(exist_ok=True)
csv_path = res_dir / f"perclass_{TAG}.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "class", "group", "precision", "recall", "mAP50", "mAP50_95"])
    w.writerows(rows)
print("SAVED", csv_path)
'''

REPORT = '''# === Report vs the Kaggle T4 reference ===
# Reads the CSV rather than the in-memory `rows`, so this cell survives a kernel restart and can
# be re-run on its own without redoing the 2.5h train.
import pandas as pd
_csv = HERE / "results" / f"perclass_{TAG}.csv"
_d = pd.read_csv(_csv)
df   = _d[_d["class"] != "ALL"].reset_index(drop=True)
allr = _d[_d["class"] == "ALL"].iloc[0].tolist()
g = df.groupby("group")[["mAP50","mAP50_95","recall"]].mean()

summary = {"ALL": allr[5], "recall": allr[4], "mAP50_95": allr[6],
           "A_whole_leaf": float(g.loc["A_whole_leaf","mAP50"]),
           "B_lesion": float(g.loc["B_lesion","mAP50"])}
for k in ("ALL", "A_whole_leaf", "B_lesion", "recall", "mAP50_95"):
    print(f"{k:14s} {summary[k]:.4f}   ({summary[k]-REF[k]:+.4f} vs T4)")
print(f"{'A-B gap':14s} {summary['A_whole_leaf']-summary['B_lesion']:.4f}   (T4 0.1979)")

d = abs(summary["ALL"] - REF["ALL"])
if WORKERS != 8:
    # R1-3 worker control. The reference is the 3-seed LOCAL workers=8 baseline, not the T4 run,
    # so the deltas printed above are env drift + seed noise + worker effect mixed together.
    print(f"\\nworker-control replicate (workers={WORKERS}) - judge it in nb/worker_control.ipynb "
          f"against the 3-seed local workers=8 baseline, NOT against the T4 deltas above")
elif ARM == "baseline" and SEED == 0 and EPOCHS >= 100:
    print("\\nCALIBRATION:", "PASS - local and Kaggle numbers may share a table"
          if d <= 0.005 else
          f"FAIL (|delta|={d:.4f} > 0.005) - local arms must use a LOCAL baseline as reference")
elif ARM == "baseline" and SEED != 0:
    # seed != 0 is a VARIANCE replicate, not a calibration - the T4 deltas above are
    # env drift + seed noise combined. Pool the seeds in nb/noise_floor.ipynb.
    print(f"\\nvariance replicate (seed {SEED}) - run nb/noise_floor.ipynb once seeds 0-2 exist")
elif ARM == "baseline":
    print(f"\\n(short run, {EPOCHS} ep - not a calibration; needs the full 100 to compare to T4)")

(HERE / "results" / f"summary_{TAG}.json").write_text(json.dumps(summary, indent=2))
display(df.sort_values("mAP50").reset_index(drop=True))
'''

CURVE = '''# === Training curve ===
import pandas as pd, matplotlib.pyplot as plt
h = pd.read_csv(HERE / "runs" / RUN / "results.csv")
h.columns = [c.strip() for c in h.columns]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h["epoch"], h["metrics/mAP50(B)"], label="mAP@50")
ax[0].plot(h["epoch"], h["metrics/mAP50-95(B)"], label="mAP@50-95")
ax[0].axhline(REF["ALL"], ls="--", c="grey", label="T4 baseline 0.8975")
ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title(TAG); ax[0].grid(alpha=.3)
for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
    if k in h: ax[1].plot(h["epoch"], h[k], label=k.split("/")[-1])
ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("losses"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

peak = h["metrics/mAP50(B)"].idxmax()
print(f"peak mAP50 {h['metrics/mAP50(B)'][peak]:.4f} @ epoch {int(h['epoch'][peak])}/{EPOCHS}"
      f" | last-10 mean {h['metrics/mAP50(B)'].tail(10).mean():.4f}")
'''


_ids = iter(f"cell{i:02d}" for i in range(100))


def md(src):
    return {"cell_type": "markdown", "id": next(_ids), "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "id": next(_ids), "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


DATASETS = {   # dataset -> (data.yaml, run/notebook/result prefix)
    "v12": (r"f:\0BScDefense\05_Datasets\merged-dataset-v12\data.yaml", "local_"),
    "v13": (r"f:\0BScDefense\05_Datasets\merged-dataset-v13\data.yaml", "v13_"),
}

HEADER_V13 = '''# {title}

**Dataset:** merged-dataset-v13 (v12 without its rotated/flipped twins) | **Arm:** `{arm}` | **seed** {seed} | **{epochs} ep** | **imgsz {imgsz}** | **batch {batch}** | **workers {workers}**

Same training recipe as the published v12 runs. v13 numbers are **not** comparable to v12 or
Kaggle numbers, because the validation and test images differ. Compare v13 arms only with v13 arms.

Run all cells. Progress bars are live; the RAM watchdog frees the image cache if memory spikes.
'''

REPORT_V13 = '''# === Report ===
# Reads the CSV rather than the in-memory `rows`, so this cell survives a kernel restart.
import pandas as pd
_d = pd.read_csv(HERE / "results" / f"perclass_{TAG}.csv")
df   = _d[_d["class"] != "ALL"].reset_index(drop=True)
allr = _d[_d["class"] == "ALL"].iloc[0].tolist()
g = df.groupby("group")[["mAP50", "mAP50_95", "recall"]].mean()
summary = {"ALL": allr[5], "recall": allr[4], "mAP50_95": allr[6],
           "A_whole_leaf": float(g.loc["A_whole_leaf", "mAP50"]),
           "B_lesion": float(g.loc["B_lesion", "mAP50"])}
for k, v in summary.items():
    print(f"{k:14s} {v:.4f}")
print(f"{'A-B gap':14s} {summary['A_whole_leaf'] - summary['B_lesion']:.4f}")
(HERE / "results" / f"summary_{TAG}.json").write_text(json.dumps(summary, indent=2))
display(df.sort_values("mAP50").reset_index(drop=True))
'''


TITLES = {"baseline": "v12 baseline (local calibration)",
          "p2": "v12 + P2 small-lesion head",
          "p2_wiou": "v12 + P2 head + WIoU",
          "uadh": "v12 + UADH (uncertainty-aware detection head)",
          "small": "v12 YOLOv11s (capacity x resolution)",
          # Shakib S-1 and S-6 (revision/PLAN_Shakib.md), v13 only
          "medium": "v12 YOLOv11m (model size)",
          "rtdetr": "v12 RT-DETR-l (detector family)",
          "focal": "v12 + focal loss",
          "wiou": "v12 + WIoU v3 box loss",
          "oversample": "v12, training set oversampled to 400 images per class",
          "naive": "v12, training set resampled to exactly 400 images per class",
          "p2_rm": "v12 + P2 head, pretrained weights mapped by role",
          "p2_wiou_rm": "v12 + P2 head (weights mapped by role) + WIoU"}
V13_ONLY = ("medium", "rtdetr", "focal", "wiou", "oversample", "naive", "p2_rm", "p2_wiou_rm")
WEIGHTS = {"baseline": "yolo11n.pt", "small": "yolo11s.pt", "medium": "yolo11m.pt",
           "focal": "yolo11n.pt", "wiou": "yolo11n.pt", "oversample": "yolo11n.pt",
           "naive": "yolo11n.pt"}
RESAMPLED = {   # built by build_resampled_v13.py; validation and test are the v13 ones
    "oversample": r"f:\0BScDefense\05_Datasets\merged-dataset-v13-oversample\data.yaml",
    "naive": r"f:\0BScDefense\05_Datasets\merged-dataset-v13-naive\data.yaml"}


def build(arm, seed, epochs, imgsz, batch, workers, cache, ram_limit, dataset="v12"):
    kaggle_dir = str((HERE.parent / "v12_kaggle").resolve())
    data, prefix = DATASETS[dataset]
    if arm in V13_ONLY and dataset != "v13":
        raise SystemExit(f"arm {arm} exists for v13 only")
    data = RESAMPLED.get(arm, data)
    title = TITLES[arm]
    if dataset != "v12":
        title = title.replace("v12", dataset)

    head = HEADER_V13 if dataset == "v13" else HEADER
    cells = [
        md(head.format(title=title, arm=arm, seed=seed, epochs=epochs, imgsz=imgsz,
                       batch=batch, workers=workers)),
        code(ENV),
        code(CONFIG.format(arm=arm, seed=seed, epochs=epochs, imgsz=imgsz, batch=batch,
                           workers=workers, cache=cache, ram_limit=ram_limit,
                           here=str(HERE), data=data, prefix=prefix)),
        code(WATCHDOG),
    ]
    if arm in ("p2_wiou", "p2_wiou_rm", "wiou"):
        cells.append(code(WIOU_CELL.replace("__KAGGLE_DIR__", kaggle_dir)))
    if arm == "focal":
        cells.append(code(FOCAL_CELL))
    if arm == "uadh":
        cells.append(code(MODEL_UADH.replace("__LOCAL_DIR__", str(HERE))))
    elif arm in ("p2_rm", "p2_wiou_rm"):
        cells.append(code(MODEL_P2_RM.replace("__KAGGLE_DIR__", kaggle_dir)))
    elif arm == "rtdetr":
        cells.append(code(MODEL_RTDETR))
    elif arm in WEIGHTS:
        cells.append(code(MODEL_BASE.replace("__WEIGHTS__", WEIGHTS[arm])))
    else:
        cells.append(code(MODEL_P2.replace("__KAGGLE_DIR__", kaggle_dir)))
    cells.append(code(TRAIN))
    if arm == "uadh":
        cells.append(code(UADH_VERIFY))
    if arm == "focal":
        cells.append(code(FOCAL_VERIFY))
    if arm in ("p2_wiou_rm", "wiou"):
        cells.append(code(WIOU_VERIFY))
    val = VAL.replace("the clean v12 val split", "the v13 val split")
    if arm == "rtdetr":   # an RT-DETR checkpoint must be validated through the RTDETR class
        val = val.replace("b = YOLO(str(best))", "b = RTDETR(str(best))")
    if dataset == "v13":
        cells += [code(val),
                  code(REPORT_V13),
                  code(CURVE.replace('ax[0].axhline(REF["ALL"], ls="--", c="grey", '
                                     'label="T4 baseline 0.8975")\n', ""))]
    else:
        cells += [code(VAL), code(REPORT), code(CURVE)]

    nbjson = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (QML)", "language": "python", "name": "qml"},
            "language_info": {"name": "python", "version": "3.12.13"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    nb_dir = HERE / "nb"; nb_dir.mkdir(exist_ok=True)
    _wk = "" if workers == 8 else f"_W{workers}"   # mirrors _W in the CONFIG cell
    path = nb_dir / f"{prefix}{arm}_s{seed}_{imgsz}{_wk}.ipynb"
    # never silently clobber a notebook that already holds executed results
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if any(c.get("outputs") for c in old.get("cells", [])):
            raise SystemExit(
                f"REFUSING to overwrite {path.name} - it contains executed outputs.\n"
                f"Delete it or pass --force if you really want to regenerate it.")
    path.write_text(json.dumps(nbjson, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="v12")
    ap.add_argument("--arm", choices=sorted(TITLES), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="False", help="False | ram | disk")
    ap.add_argument("--ram-limit", type=int, default=85, help="%% RAM before cache is dropped")
    ap.add_argument("--force", action="store_true", help="overwrite a notebook holding outputs")
    a = ap.parse_args()
    if a.force:
        _wk = "" if a.workers == 8 else f"_W{a.workers}"
        p = HERE / "nb" / f"{DATASETS[a.dataset][1]}{a.arm}_s{a.seed}_{a.imgsz}{_wk}.ipynb"
        p.unlink(missing_ok=True)
    cache = False if a.cache in ("False", "false", "0", "") else a.cache
    p = build(a.arm, a.seed, a.epochs, a.imgsz, a.batch, a.workers, cache, a.ram_limit,
              a.dataset)
    print("built", p)
    print(f"open with:  code -r \"{p}\"")
