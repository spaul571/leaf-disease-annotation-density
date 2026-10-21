"""GPU smoke test: verify the cu130 wheel actually trains on sm_120 (Blackwell).

An "imports fine" torch can still die with `no kernel image is available for execution on the
device`. This runs a real 2-epoch train of the heaviest arm (P2 + WIoU) so both the extra
stride-4 head and the monkey-patched loss are exercised on the GPU.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from train_local import build_model, apply_wiou, HERE

print("torch", torch.__version__, "|", torch.cuda.get_device_name(0),
      "| cc", torch.cuda.get_device_capability(0))
apply_wiou()
model = build_model("p2")
model.train(data="coco8.yaml", epochs=2, imgsz=640, batch=8, device=0, amp=True,
            workers=0, val=True, plots=False, project=str(HERE / "runs"),
            name="gpu_smoke", exist_ok=True)
print("peak VRAM %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
print("GPU SMOKE OK")
