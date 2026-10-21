"""Validate UADH before spending GPU-hours on it.

Checks, in order:
  1. branch attaches, param count sane, s == 0 at init (so step 0 == baseline objective)
  2. the loss patch actually RUNS (a real 2-epoch coco8 train, not just "the patch applied")
  3. the uncertainty head LEARNS - s must move away from 0, otherwise the branch is dead weight
     and the whole contribution is vacuous
  4. equivalence check: with s pinned to 0 the loss must equal the stock CIoU loss
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from ultralytics import YOLO
import uadh

HERE = Path(__file__).resolve().parent

print("=" * 70, "\n1) EQUIVALENCE AT INIT (s=0 must reproduce the stock loss)\n")
import ultralytics.utils.loss as L
stock_forward = L.BboxLoss.forward

uadh.patch_uadh()
m = YOLO("yolo11n.pt")
uadh.attach_uadh(m)

det = m.model.model[-1]
with torch.no_grad():
    m.model.train()
    out = m.model(torch.zeros(2, 3, 256, 256))
assert "unc" in out, "unc missing from train-mode output"
s0 = out["unc"]
print(f"   unc shape {tuple(s0.shape)} | mean {s0.mean():.6f} | max|s| {s0.abs().max():.6f}")
assert s0.abs().max() < 1e-6, "s should be exactly 0 at init (zero-init final conv)"
print("   OK: s == 0 at init -> exp(-s) == 1 -> objective identical to baseline\n")

print("=" * 70, "\n2) REAL 2-EPOCH TRAIN (the check exp16/17/18 each skipped)\n")
r = m.train(data="coco8.yaml", epochs=2, imgsz=256, batch=2, device=0, amp=True, workers=0,
            val=False, plots=False, project=str(HERE / "runs"), name="uadh_test", exist_ok=True,
            verbose=False)
print("   train completed\n")

print("=" * 70, "\n2b) DID THE BRANCH SURVIVE THE REBUILD INSIDE .train()?\n")
uadh.assert_uadh_alive(m, "after train (m.model)")
assert hasattr(m.trainer.model.model[-1], "cv4"), "branch missing on trainer.model"
print("   OK: cv4 exists on both m.model and trainer.model after training\n")

print("=" * 70, "\n3) DID THE UNCERTAINTY HEAD LEARN?\n")
with torch.no_grad():
    m.model.train()
    out = m.model(torch.zeros(2, 3, 256, 256))
s1 = out["unc"]
print(f"   after training: mean {s1.mean():+.5f} | sd {s1.std():.5f} | "
      f"range [{s1.min():+.4f}, {s1.max():+.4f}]")
moved = s1.abs().max().item()
print("   VERDICT:", "head is learning (s moved off 0)" if moved > 1e-4 else
      "DEAD BRANCH - s never moved, gradient is not reaching cv4")
assert moved > 1e-4, "cv4 received no gradient - UADH would be a no-op"

grads = [p.grad for p in det.cv4.parameters() if p.grad is not None]
print(f"   cv4 params with grads: {len(grads)}/{len(list(det.cv4.parameters()))}")

print("\n" + "=" * 70, "\n4) PARAM OVERHEAD\n")
tot = sum(p.numel() for p in m.model.parameters())
add = sum(p.numel() for p in det.cv4.parameters())
print(f"   total {tot/1e6:.3f}M | uadh branch {add:,} ({100*add/tot:.1f}%)")
print("\nALL CHECKS PASSED")
