"""Ensemble the already-trained models and score on the same clean v12 val split.

"Merge/mix" with zero extra training: concatenate detections from several independently trained
models and let NMS fuse them. Diversity comes from seeds (s0/s1/s2) and, optionally, resolution.
Scored with the stock validator so numbers stay comparable to every other arm.
"""
import argparse, json
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import Ensemble

HERE = Path(__file__).resolve().parent
DATA = Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v12\data.yaml")
# Defined once in classes.py; other scripts import NAMES and GROUP_B from here, so keep these names.
from classes import GROUP_B, NAMES  # noqa: E402


def report(b, tag):
    per = {NAMES[int(c)]: float(b.ap50[i]) for i, c in enumerate(b.ap_class_index)}
    gb = [v for k, v in per.items() if k in GROUP_B]
    ga = [v for k, v in per.items() if k not in GROUP_B]
    print(f"{tag:34s} ALL {float(b.map50):.4f} | A {sum(ga)/len(ga):.4f} | "
          f"B {sum(gb)/len(gb):.4f} | R {float(b.mr):.4f} | m50-95 {float(b.map):.4f}", flush=True)
    return {"tag": tag, "ALL": round(float(b.map50), 4), "A": round(sum(ga)/len(ga), 4),
            "B": round(sum(gb)/len(gb), 4), "recall": round(float(b.mr), 4),
            "mAP50_95": round(float(b.map), 4)}


class EnsembleX(Ensemble):
    """Ensemble that tolerates the kwargs the 8.4.x validator passes (embed, visualize).

    Stock Ensemble.forward has signature (x, augment, profile) only, so the validator's
    `embed=` kwarg raises TypeError. Body is the stock NMS-ensemble: concatenate each model's
    detections along the anchor axis and let NMS fuse them.
    """

    def forward(self, x, augment=False, profile=False, visualize=False, embed=None):
        import torch as _t
        y = [module(x, augment, profile)[0] for module in self]
        return _t.cat(y, 2), None


def build(paths):
    ens = EnsembleX()
    for p in paths:
        m = YOLO(str(p)).model
        m.float().eval()
        ens.append(m)
    ref = ens[-1]
    for a in ("names", "nc", "yaml", "stride", "args", "pt_path", "task"):
        if hasattr(ref, a):
            setattr(ens, a, getattr(ref, a))
    ens.eval()
    return ens


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou", type=float, default=0.6)
    a = ap.parse_args()
    R = HERE / "runs"
    sets = {
        "seeds s0+s1 (640)":       [R/"local_baseline_s0_640/weights/best.pt",
                                    R/"local_baseline_s1_640/weights/best.pt"],
        "seeds s0+s1+s2 (640)":    [R/"local_baseline_s0_640/weights/best.pt",
                                    R/"local_baseline_s1_640/weights/best.pt",
                                    R/"local_baseline_s2_640/weights/best.pt"],
        # architecture diversity: p2/uadh are WORSE alone, but ensembles reward decorrelated
        # errors, so a weak-but-different model can still add value
        "3 seeds + P2":            [R/"local_baseline_s0_640/weights/best.pt",
                                    R/"local_baseline_s1_640/weights/best.pt",
                                    R/"local_baseline_s2_640/weights/best.pt",
                                    R/"local_p2_s0_640/weights/best.pt"],
        "3 seeds + P2 + UADH":     [R/"local_baseline_s0_640/weights/best.pt",
                                    R/"local_baseline_s1_640/weights/best.pt",
                                    R/"local_baseline_s2_640/weights/best.pt",
                                    R/"local_p2_s0_640/weights/best.pt",
                                    R/"local_uadh_s0_640/weights/best.pt"],
        "3 seeds + 1024 + P2":     [R/"local_baseline_s0_640/weights/best.pt",
                                    R/"local_baseline_s1_640/weights/best.pt",
                                    R/"local_baseline_s2_640/weights/best.pt",
                                    R/"local_baseline_s0_1024/weights/best.pt",
                                    R/"local_p2_s0_640/weights/best.pt"],
        "3 seeds + 1024":          [R/"local_baseline_s0_640/weights/best.pt",
                                    R/"local_baseline_s1_640/weights/best.pt",
                                    R/"local_baseline_s2_640/weights/best.pt",
                                    R/"local_baseline_s0_1024/weights/best.pt"],
    }
    out = []
    for tag, paths in sets.items():
        try:
            wrapper = YOLO(str(paths[0]))
            wrapper.model = build(paths).cuda()
            b = wrapper.val(data=str(DATA), split="val", imgsz=a.imgsz, verbose=False,
                            iou=a.iou, project=str(HERE/"runs"), name="ens_tmp",
                            exist_ok=True).box
            out.append(report(b, f"{tag} @{a.imgsz}"))
        except Exception as e:
            print(f"{tag:34s} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
    if out:
        (HERE/"results"/f"opt_ensemble_{a.imgsz}.json").write_text(json.dumps(out, indent=2))
