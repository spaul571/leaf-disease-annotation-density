"""Did the cold-start NMS warning change the first evaluation job?

eval_v13.ipynb and eval_twinfree_v12.ipynb both logged "NMS time limit 2.050s exceeded" on their
first job (test_full | baseline s0). When that happens Ultralytics stops NMS for the rest of the
batch, so those images count as having no detections. This script re-runs that job in one process,
first cold and then warm, scores it exactly as the EVAL cell does, and compares the result with the
stored value. Nothing is written to results/.

    qpy check_nms_warmup.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_ensemble import GROUP_B, NAMES  # noqa: E402

DS = Path(r"f:\0BScDefense\05_Datasets")
CASES = [  # (label, weights, data.yaml, stored eval json)
    ("v13 cold", HERE / "runs/v13_baseline_s0_640_W2/weights/best.pt",
     DS / "merged-dataset-v13/data.yaml", HERE / "results/eval_v13_eval.json"),
    ("v13 warm", HERE / "runs/v13_baseline_s0_640_W2/weights/best.pt",
     DS / "merged-dataset-v13/data.yaml", HERE / "results/eval_v13_eval.json"),
    ("v12 warm", HERE / "runs/local_baseline_s0_640/weights/best.pt",
     DS / "merged-dataset-v12/data.yaml", HERE / "results/twinfree_v12_eval.json"),
]
KEYS = ("ALL", "A", "B", "recall", "mAP50_95")


class NmsFlag(logging.Handler):
    hit = False

    def emit(self, record):
        if "NMS time limit" in record.getMessage():
            NmsFlag.hit = True


logging.getLogger("ultralytics").addHandler(NmsFlag())


def score(weights, yml):
    b = YOLO(str(weights)).val(
        data=str(yml), split="test", imgsz=640, augment=False, workers=0, plots=False,
        verbose=False, project=str(HERE / "runs"), name="eval_tmp", exist_ok=True).box
    ap50 = {NAMES[int(c)]: float(b.ap50[i]) for i, c in enumerate(b.ap_class_index)}
    ga = [v for k, v in ap50.items() if k not in GROUP_B]
    gb = [v for k, v in ap50.items() if k in GROUP_B]
    return {"ALL": float(b.map50), "A": float(np.mean(ga)), "B": float(np.mean(gb)),
            "recall": float(b.mr), "mAP50_95": float(b.map)}


rows = []
for label, weights, yml, stored_json in CASES:
    NmsFlag.hit = False
    got = score(weights, yml)
    stored = next(r for r in json.loads(stored_json.read_text())
                  if r["subset"] == "test_full" and r["system"] == "baseline" and r["seed"] == 0)
    rows.append((label, NmsFlag.hit, got, stored))

print("\n=== test_full | baseline s0 ===")
for label, hit, got, stored in rows:
    print(f"{label}: NMS warning {'YES' if hit else 'no'}")
    for k in KEYS:
        print(f"  {k:<9} now {got[k]:.6f}   stored {stored[k]:.6f}   diff {got[k] - stored[k]:+.6f}")
