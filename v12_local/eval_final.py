"""Stack the free (no-retraining) gains on the best model: TTA, NMS tuning, ensembling."""
import json
from pathlib import Path
import torch
from ultralytics import YOLO
from eval_ensemble import EnsembleX, build, report

HERE = Path(__file__).resolve().parent
DATA = Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v12\data.yaml")
R = HERE / "runs"
S1024 = R / "local_small_s0_1024/weights/best.pt"
N1024 = R / "local_baseline_s0_1024/weights/best.pt"


def ev(w, imgsz, tag, augment=False, iou=0.7, ens=None):
    m = YOLO(str(w))
    if ens:
        m.model = build(ens).cuda()
    b = m.val(data=str(DATA), split="val", imgsz=imgsz, verbose=False, augment=augment,
              iou=iou, project=str(HERE / "runs"), name="fin_tmp", exist_ok=True).box
    return report(b, tag)


if __name__ == "__main__":
    out = []
    out.append(ev(S1024, 1024, "s@1024 (reference)"))
    out.append(ev(S1024, 1024, "s@1024 + TTA", augment=True))
    out.append(ev(S1024, 1024, "s@1024 + NMS iou=0.5", iou=0.5))
    out.append(ev(S1024, 1024, "s@1024 + n@1024 ensemble", ens=[S1024, N1024]))
    out.append(ev(S1024, 1024, "s@1024 + n@1024 ens + TTA", augment=True, ens=[S1024, N1024]))
    (HERE / "results" / "opt_final.json").write_text(json.dumps(out, indent=2))
