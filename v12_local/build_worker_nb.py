"""Generate nb/worker_control.ipynb - the R1-3 dataloader-worker control.

Reviewer 1 comment 3: the 1024 px arms used workers=2 while the 640 px baseline used workers=8,
so the resolution comparison carries one uncontrolled difference. Worker count does not enter the
model or the loss, but it does change the per-worker augmentation RNG streams, so it can only be
answered by measurement.

The experiment matches DOWNWARD, which is the cheap direction: re-run the 640 px baseline at
workers=2, three seeds (~2.4 h each), and ask whether it lands inside the already-measured seed
noise band of the workers=8 baseline. If it does, worker count is shown not to carry the effect
and the published 640-vs-1024 comparison stands. Matching upward instead (1024 px at workers=8)
costs ~4.7 h per seed and risks host RAM at 1024 px, which is why workers=2 was chosen there.

Run AFTER the three worker-control notebooks have executed:

    qpy build_nb.py --arm baseline --seed 0 --workers 2
    qpy build_nb.py --arm baseline --seed 1 --workers 2
    qpy build_nb.py --arm baseline --seed 2 --workers 2
    qpy build_worker_nb.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

HEADER = '''# R1-3 worker control: does `workers` move the result?

**Question.** The published 640 px baseline trained with `workers: 8`; the 1024 px arms trained
with `workers: 2`. Worker count touches neither the model nor the loss, but each worker owns its
own augmentation RNG stream, so the 640-to-1024 comparison in the paper carries one uncontrolled
difference alongside the intended one (Limitations, seventh item).

**Design.** Re-run the *640 px baseline* at `workers: 2`, three seeds, everything else identical
(YOLOv11n, 100 ep, batch 16, patience 30, amp, same data.yaml). This isolates the worker effect:
the only difference from the published arm is the thing under test.

**Decision rule, fixed before looking.** The comparison is a 3-seed mean against a 3-seed mean, so
the bar is `1.96 * sd_pooled * sqrt(2/3)`, the same rule used for every other mean-vs-mean claim in
the paper. Two outcomes, both publishable:

| outcome | reading | what the paper says |
|---|---|---|
| inside the bar | worker count carries no detectable effect | the 640-vs-1024 comparison stands as reported; limitation seven narrows to "controlled, no detectable effect" |
| outside the bar | worker count moves the metric | the resolution effect must be re-stated against a workers-matched baseline, and the size of the worker effect is reported alongside it |

A null here is the useful result, and it is reported either way. Do not re-run seeds until the
answer changes.

**Prerequisite.** `results/summary_local_baseline_100ep_s{0,1,2}_W2.json` must exist, i.e. the
three `nb/local_baseline_s*_640_W2.ipynb` notebooks have been executed.
'''

LOAD = '''# === Load both arms ===
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(r"{here}")
RES  = HERE / "results"
METRICS = ["ALL", "A_whole_leaf", "B_lesion", "recall", "mAP50_95"]
SEEDS   = (0, 1, 2)

def load(suffix, label):
    rows = {{}}
    for s in SEEDS:
        p = RES / f"summary_local_baseline_100ep_s{{s}}{{suffix}}.json"
        if p.exists():
            rows[s] = json.loads(p.read_text())
    assert rows, f"no {{label}} summaries found - expected {{RES}}/summary_local_baseline_100ep_s0{{suffix}}.json"
    return pd.DataFrame(rows).T[METRICS].sort_index()

W8 = load("",    "workers=8 (published)")
W2 = load("_W2", "workers=2 (control)")

print("workers=8 seeds:", list(W8.index), "| workers=2 seeds:", list(W2.index))
# n=3 vs n=3 is the rule the paper uses. With fewer seeds the bar below is not the paper's bar.
if len(W2) < 3:
    print(f"NOTE: only {{len(W2)}} control seed(s) present. The bar below assumes 3 per arm; "
          f"treat anything computed now as provisional and re-run this notebook when all three exist.")
display(pd.concat({{"workers=8": W8, "workers=2": W2}}, names=["arm", "seed"]).round(4))
'''

STATS = '''# === Per-metric comparison against the mean-vs-mean bar ===
# Bar = 1.96 * sd_pooled * sqrt(2/3): the 95% band on a difference of two 3-seed means, the same
# rule used for every mean-vs-mean claim in the paper. Welch's t-test is reported beside it
# because that is what Table 5 reports; with n=3 per arm it has 2-4 df and little power, so the
# bar is the primary read and p is the secondary one.
rows = []
for m in METRICS:
    a, b = W8[m].astype(float).values, W2[m].astype(float).values
    sd_pool = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    bar = 1.96 * sd_pool * np.sqrt(2 / 3)
    delta = b.mean() - a.mean()
    t, p = stats.ttest_ind(b, a, equal_var=False)
    rows.append({
        "metric": m,
        "workers=8": round(a.mean(), 4), "sd8": round(a.std(ddof=1), 5),
        "workers=2": round(b.mean(), 4), "sd2": round(b.std(ddof=1), 5),
        "delta": round(delta, 4), "bar": round(bar, 4),
        "ratio": round(abs(delta) / bar, 2) if bar else np.inf,
        "welch_p": round(float(p), 4),
        "verdict": "OUTSIDE bar" if abs(delta) > bar else "inside bar",
    })
C = pd.DataFrame(rows).set_index("metric")
display(C)

outside = C[C["verdict"] == "OUTSIDE bar"]
if outside.empty:
    print("\\nRESULT: no metric moves beyond the 3-vs-3 bar.")
    print("Worker count carries no detectable effect at this power. The published 640-vs-1024")
    print("comparison stands; limitation seven becomes a controlled null rather than an open")
    print("confound. State the bar and the power in the text - this is 'no detectable effect at")
    print("n=3', not 'no effect'.")
else:
    print("\\nRESULT: worker count moves", ", ".join(outside.index) + ".")
    print("The resolution effect must be restated against a workers-matched baseline, and the")
    print("size of the worker effect reported beside it. Do NOT fold it silently into the")
    print("existing numbers.")
'''

COST = '''# === Computational cost (also answers the second half of R1-3) ===
# Wall-clock comes from the `time` column of each run's results.csv, so it is measured, not
# estimated. Published arms were timed on the same machine, so the ratio is meaningful.
import pandas as pd
from pathlib import Path

RUNS = HERE / "runs"
ARMS = {
    "baseline 640 (workers=8, published)": [f"local_baseline_s{s}_640" for s in SEEDS],
    "baseline 640 (workers=2, control)":   [f"local_baseline_s{s}_640_W2" for s in SEEDS],
    "YOLOv11s 1024 (workers=2, selected)": [f"local_small_s{s}_1024" for s in SEEDS],
}
rows = []
for label, runs in ARMS.items():
    for r in runs:
        f = RUNS / r / "results.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f); h.columns = [c.strip() for c in h.columns]
        rows.append({"arm": label, "run": r, "epochs": int(h["epoch"].iloc[-1]),
                     "seconds": float(h["time"].iloc[-1]),
                     "hours": round(float(h["time"].iloc[-1]) / 3600, 2)})
T = pd.DataFrame(rows)
display(T)

if not T.empty:
    g = T.groupby("arm")["hours"].agg(["mean", "count"]).round(2)
    display(g)
    base = "baseline 640 (workers=8, published)"
    if base in g.index:
        for arm in g.index:
            print(f"{arm:38s} {g.loc[arm,'mean']:.2f} h/run  "
                  f"({g.loc[arm,'mean']/g.loc[base,'mean']:.2f}x baseline, n={int(g.loc[arm,'count'])})")
    print("\\nFor the manuscript: report training hours per arm, the GPU, and the TTA inference")
    print("multiplier separately - TTA costs inference time, not training time.")
'''

SAVE = '''# === Persist for the manuscript ===
out = {
    "experiment": "R1-3 dataloader worker control",
    "design": "640px YOLOv11n baseline, 100ep, batch16, patience30, amp; workers 8 vs 2; 3 seeds each",
    "rule": "1.96 * sd_pooled * sqrt(2/3) on a difference of two 3-seed means",
    "per_seed": {"workers_8": W8.astype(float).round(6).to_dict(),
                 "workers_2": W2.astype(float).round(6).to_dict()},
    "comparison": C.reset_index().to_dict(orient="records"),
    "cost_hours": (T.to_dict(orient="records") if not T.empty else []),
}
p = HERE / "results" / "worker_control.json"
p.write_text(json.dumps(out, indent=2))
print("SAVED", p)

# Paste-ready sentence for Section 3.4 / the Limitations rewrite. Numbers are filled from the run,
# so it cannot drift from the measurement.
_d = C.loc["ALL"]; _b = C.loc["B_lesion"]
verdict = "no metric cleared" if (C["verdict"] == "inside bar").all() else "at least one metric cleared"
print(f"""
DRAFT SENTENCE ({verdict} the bar):
  A worker-count control was run to isolate this difference: the 640 px baseline was retrained at
  workers = 2, three seeds, with every other setting unchanged. Class-mean mAP@50 changed by
  {_d['delta']:+.4f} against a 3-versus-3 significance bar of {_d['bar']:.4f} (Welch p = {_d['welch_p']}),
  and lesion-group mAP@50 by {_b['delta']:+.4f} against a bar of {_b['bar']:.4f} (p = {_b['welch_p']}).
""")
'''


def md(src, i):
    return {"cell_type": "markdown", "id": f"wc{i:02d}", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src, i):
    return {"cell_type": "code", "id": f"wc{i:02d}", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


if __name__ == "__main__":
    cells = [md(HEADER, 0), code(LOAD.format(here=str(HERE)), 1), code(STATS, 2),
             code(COST, 3), code(SAVE, 4)]
    nbjson = {"cells": cells,
              "metadata": {"kernelspec": {"display_name": "Python (QML)", "language": "python",
                                          "name": "qml"},
                           "language_info": {"name": "python", "version": "3.12.13"}},
              "nbformat": 4, "nbformat_minor": 5}
    out = HERE / "nb" / "worker_control.ipynb"
    out.parent.mkdir(exist_ok=True)
    if out.exists() and any(c.get("outputs")
                            for c in json.loads(out.read_text(encoding="utf-8"))["cells"]):
        raise SystemExit(f"REFUSING to overwrite {out.name} - it holds executed outputs.")
    out.write_text(json.dumps(nbjson, indent=1), encoding="utf-8")
    print("built", out)
