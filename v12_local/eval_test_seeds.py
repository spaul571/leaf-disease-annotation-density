"""Final held-out TEST comparison: 3 baseline seeds vs 3 s@1024 seeds, mean vs mean.

With n=3 per arm the comparison is no longer single-run-vs-single-run, so the bar tightens from
1.96*sd*sqrt(2) to 1.96*sd_pooled*sqrt(1/3+1/3). Also reports Welch's t-test, which does not
assume equal variances (baseline and arm scatter differently on the small test split).
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats
from eval_ensemble import report
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
DATA = Path(r"f:\0BScDefense\05_Datasets\merged-dataset-v12\data.yaml")
R = HERE / "runs"
KEYS = ("ALL", "A", "B", "recall", "mAP50_95")


def ev(w, imgsz, tag, augment=False):
    b = YOLO(str(w)).val(data=str(DATA), split="test", imgsz=imgsz, verbose=False,
                         augment=augment, project=str(HERE / "runs"), name="ts_tmp",
                         exist_ok=True).box
    return report(b, tag)


if __name__ == "__main__":
    arm = [ev(R / f"local_small_s{s}_1024/weights/best.pt", 1024, f"s@1024 seed{s}")
           for s in (0, 1, 2)]
    armt = [ev(R / f"local_small_s{s}_1024/weights/best.pt", 1024, f"s@1024+TTA seed{s}",
               augment=True) for s in (0, 1, 2)]
    prev = json.loads((HERE / "results" / "test_final.json").read_text())
    base = [r for r in prev if r["tag"].startswith("640 baseline")]

    print("\n=== TEST: mean +/- sd (n=3 per arm) ===")
    for name, grp in (("640 baseline", base), ("s@1024", arm), ("s@1024+TTA", armt)):
        print(f"  {name:14s} " + "  ".join(
            f"{k} {np.mean([r[k] for r in grp]):.4f}+/-{np.std([r[k] for r in grp], ddof=1):.4f}"
            for k in KEYS))

    out = {}
    for name, grp in (("s@1024", arm), ("s@1024+TTA", armt)):
        print(f"\n=== {name} vs baseline on TEST (mean-vs-mean, n=3 each) ===")
        out[name] = {}
        for k in KEYS:
            a = np.array([r[k] for r in grp]); b = np.array([r[k] for r in base])
            d = a.mean() - b.mean()
            sp = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2)
            bar = 1.96 * sp * np.sqrt(2 / 3)
            t, p = stats.ttest_ind(a, b, equal_var=False)
            ratio = abs(d) / bar if bar else np.inf
            v = "REAL" if ratio >= 2 else "MARGINAL" if ratio >= 1 else "NOISE"
            sig = "SIGNIFICANT" if p < 0.05 else "n.s."
            print(f"  {k:10s} {b.mean():.4f} -> {a.mean():.4f}  {d:+.4f}  bar {bar:.4f}  "
                  f"{ratio:4.1f}x  {v:8s} | Welch p={p:.4f} {sig}")
            out[name][k] = {"delta": float(d), "bar": float(bar), "ratio": float(ratio),
                            "p": float(p), "verdict": v}
    (HERE / "results" / "test_seeds_final.json").write_text(
        json.dumps({"arm": arm, "arm_tta": armt, "stats": out}, indent=2))
    import os, sys
    sys.stdout.flush(); os._exit(0)
