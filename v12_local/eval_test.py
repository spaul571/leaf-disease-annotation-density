"""FINAL held-out TEST evaluation.

Everything reported so far was measured on val, and the winning configuration (resolution, TTA,
NMS threshold, ensemble membership) was SELECTED by looking at val. Those numbers are therefore
optimistically biased. This re-scores the shortlist on the untouched test split, and derives the
noise bar from the three baseline seeds ON TEST so the verdicts are not imported from val.
"""
import json
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from eval_ensemble import build, report

HERE = Path(__file__).resolve().parent
DATA = Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v12\data.yaml")
R = HERE / "runs"


def ev(w, imgsz, tag, augment=False, iou=0.7, ens=None, split="test"):
    m = YOLO(str(w))
    if ens:
        m.model = build(ens).cuda()
    b = m.val(data=str(DATA), split=split, imgsz=imgsz, verbose=False, augment=augment, iou=iou,
              project=str(HERE / "runs"), name="test_tmp", exist_ok=True).box
    return report(b, tag)


if __name__ == "__main__":
    S1024 = R / "local_small_s0_1024/weights/best.pt"
    N1024 = R / "local_baseline_s0_1024/weights/best.pt"
    out = []
    for s in (0, 1, 2):
        out.append(ev(R / f"local_baseline_s{s}_640/weights/best.pt", 640, f"640 baseline seed{s}"))
    out.append(ev(N1024, 1024, "n@1024"))
    out.append(ev(S1024, 1024, "s@1024"))
    out.append(ev(S1024, 1024, "s@1024 + TTA", augment=True))
    out.append(ev(S1024, 1024, "s@1024 + n@1024 ensemble", ens=[S1024, N1024]))
    (HERE / "results" / "test_final.json").write_text(json.dumps(out, indent=2))

    base = [r for r in out if r["tag"].startswith("640 baseline")]
    print("\n=== TEST-derived noise bar (3 baseline seeds, 1.96*sd*sqrt2) ===")
    bar = {}
    for k in ("ALL", "A", "B", "recall", "mAP50_95"):
        vals = np.array([r[k] for r in base])
        bar[k] = float(1.96 * np.sqrt(2) * vals.std(ddof=1))
        print(f"  {k:10s} mean {vals.mean():.4f} sd {vals.std(ddof=1):.5f} -> bar {bar[k]:.4f}")
    print("\n=== best configs vs the 3-seed TEST baseline mean ===")
    bm = {k: float(np.mean([r[k] for r in base])) for k in ("ALL", "A", "B", "recall", "mAP50_95")}
    for r in out:
        if r["tag"].startswith("640 baseline"):
            continue
        print(f"\n{r['tag']}")
        for k in ("ALL", "A", "B", "recall", "mAP50_95"):
            d = r[k] - bm[k]
            ratio = abs(d) / bar[k] if bar[k] else float("inf")
            v = "REAL" if ratio >= 2 else "MARGINAL" if ratio >= 1 else "NOISE"
            print(f"   {k:10s} {bm[k]:.4f} -> {r[k]:.4f}  {d:+.4f}  {ratio:4.1f}x  {v}")
    (HERE / "results" / "test_bar.json").write_text(json.dumps({"bar": bar, "baseline_mean": bm}, indent=2))
    # Force exit: ultralytics' val() can leave DataLoader workers alive after the script body
    # finishes. Those workers held ~10 GB and starved the NEXT run into
    # "WinError 1455: paging file too small" while loading cublas.
    import os, sys
    sys.stdout.flush(); os._exit(0)
