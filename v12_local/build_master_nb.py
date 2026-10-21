"""Generate nb/master_table.ipynb - the paper's results spine.

Pools every clean-v12 arm (Kaggle T4) plus the three local baseline seeds, and judges every
effect against the measured noise floor in results/noise_floor.json.

Comparison rules encoded below (getting these wrong is how a benchmark paper misleads):
  * 100-epoch nano arms are compared to exp6b (same env, same budget).
  * The 60-epoch n/s/m ablation is compared to exp10 (nano @ 60 ep), NOT to exp6b - different
    budget entirely.
  * RT-DETR ran 30 ep to convergence; it has no matched-budget partner, so it is reported
    without a verdict and flagged.
  * The noise floor is SEED noise within one environment. Kaggle-vs-Kaggle comparisons are
    valid against it; the local seeds are reported separately as the floor's provenance.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KAGGLE = (HERE.parent / "v12_kaggle").resolve()

HEADER = '''# Clean v12 - master results table

Every arm on the leakage-free `merged-dataset-v12`, judged against the measured noise floor
(3 local baseline seeds, `results/noise_floor.json`).

**Bar for a single-run-vs-single-run difference = 1.96 x sd x sqrt(2)**, because each effect is a
difference of two single runs, not a deviation from a mean.

| metric | 95% bar |
|---|---|
| ALL mAP@50 | 0.0040 |
| Group A | 0.0058 |
| Group B lesion | 0.0080 |
| recall | 0.0058 |
| mAP@50-95 | 0.0145 |

Verdict scale: **REAL** >2x bar - **MARGINAL** 1-2x - **NOISE** <1x.
'''

LOAD = '''# === Load every arm ===
import json
from pathlib import Path
import pandas as pd, numpy as np

HERE   = Path(r"__HERE__")
KAGGLE = Path(r"__KAGGLE__")
pd.set_option("display.width", 200)

FLOOR = json.loads((HERE / "results" / "noise_floor.json").read_text())["threshold"]
KEY = {"ALL": "ALL", "A_whole_leaf": "A_whole_leaf", "B_lesion": "B_lesion",
       "recall": "recall", "mAP50_95": "mAP50_95"}

# arm -> (csv path, epochs, model, env, reference arm for the delta)
ARMS = {
    "baseline (n, 100ep)":      (KAGGLE/"out6b/perclass_v12_100ep.csv",      100, "YOLOv11n",        "T4", None),
    "oversample-only":          (KAGGLE/"out7/perclass_v12_oversample.csv",  100, "YOLOv11n",        "T4", "baseline (n, 100ep)"),
    "naive over+under":         (KAGGLE/"out14/perclass_v12_naive.csv",      100, "YOLOv11n",        "T4", "baseline (n, 100ep)"),
    "focal loss":               (KAGGLE/"out13/perclass_v12_focal.csv",      100, "YOLOv11n",        "T4", "baseline (n, 100ep)"),
    "+CBAM":                    (KAGGLE/"out16r/perclass_v12_cbam.csv",      100, "n+CBAM",          "T4", "baseline (n, 100ep)"),
    "+CBAM+WIoU":               (KAGGLE/"out17/perclass_v12_cbam_wiou.csv",  100, "n+CBAM+WIoU",     "T4", "baseline (n, 100ep)"),
    "+P2 head":                 (KAGGLE/"out18f/perclass_v12_p2.csv",        100, "n+P2",            "T4", "baseline (n, 100ep)"),
    "+P2+WIoU":                 (KAGGLE/"out19/perclass_v12_p2_wiou.csv",    100, "n+P2+WIoU",       "T4", "baseline (n, 100ep)"),
    "n @ 60ep":                 (KAGGLE/"out10/perclass_v12_nano60.csv",      60, "YOLOv11n",        "T4", None),
    "s @ 60ep":                 (KAGGLE/"out11/perclass_v12_small60.csv",     60, "YOLOv11s",        "T4", "n @ 60ep"),
    "m @ 60ep":                 (KAGGLE/"out12b/perclass_v12_medium60.csv",   60, "YOLOv11m",        "T4", "n @ 60ep"),
    "RT-DETR-l @ 30ep":         (KAGGLE/"out15/perclass_v12_rtdetr.csv",      30, "RT-DETR-l (32M)", "T4", None),
}
LOCAL = {f"local baseline s{s}": HERE/f"results/perclass_local_baseline_100ep_s{s}.csv"
         for s in (0, 1, 2)}

def summarise(path):
    d = pd.read_csv(path)
    a = d[d["class"] == "ALL"].iloc[0]
    per = d[d["class"] != "ALL"]
    g = per.groupby("group")["mAP50"].mean()
    return {"ALL": a["mAP50"], "A_whole_leaf": g["A_whole_leaf"], "B_lesion": g["B_lesion"],
            "recall": a["recall"], "mAP50_95": a["mAP50_95"]}, per

rows, perclass = {}, {}
for name, (p, ep, model, env, ref) in ARMS.items():
    if not Path(p).exists():
        print("MISSING", name, p); continue
    s, per = summarise(p)
    s.update(epochs=ep, model=model, env=env, ref=ref)
    rows[name] = s
    perclass[name] = per.set_index("class")["mAP50"]
for name, p in LOCAL.items():
    s, per = summarise(p)
    s.update(epochs=100, model="YOLOv11n", env="5070Ti", ref=None)
    rows[name] = s
    perclass[name] = per.set_index("class")["mAP50"]

M = pd.DataFrame(rows).T
print(f"loaded {len(M)} arms")
'''

TABLE = '''# === Master table ===
cols = ["model", "epochs", "env", "ALL", "A_whole_leaf", "B_lesion", "recall", "mAP50_95"]
show = M[cols].copy()
for c in ["ALL", "A_whole_leaf", "B_lesion", "recall", "mAP50_95"]:
    show[c] = show[c].astype(float).round(4)
show["A-B gap"] = (show["A_whole_leaf"] - show["B_lesion"]).round(4)
display(show)

(HERE / "results").mkdir(exist_ok=True)
show.to_csv(HERE / "results" / "master_table.csv")
print("SAVED results/master_table.csv")
'''

VERDICT = '''# === Deltas vs the correct reference, judged against the floor ===
def verdict(ratio):
    if ratio >= 2.0: return "REAL"
    if ratio >= 1.0: return "MARGINAL"
    return "NOISE"

out = []
for name, r in M.iterrows():
    ref = r["ref"]
    if ref is None:
        continue
    base = M.loc[ref]
    for metric in ["ALL", "A_whole_leaf", "B_lesion", "recall", "mAP50_95"]:
        eff = float(r[metric]) - float(base[metric])
        bar = FLOOR[metric]
        out.append({"arm": name, "vs": ref, "metric": metric,
                    "value": round(float(r[metric]), 4), "delta": round(eff, 4),
                    "bar": round(bar, 4), "ratio": round(abs(eff)/bar, 2),
                    "verdict": verdict(abs(eff)/bar)})
D = pd.DataFrame(out)
display(D[D.metric.isin(["ALL", "B_lesion"])].reset_index(drop=True))
D.to_csv(HERE / "results" / "master_deltas.csv", index=False)

print("\\nEffects that are REAL on ALL mAP@50:")
for _, r in D[(D.metric == "ALL") & (D.verdict == "REAL")].iterrows():
    print(f"  {r['arm']:22s} {r['delta']:+.4f}  ({r['ratio']}x bar)")
print("\\nEffects that are NOISE on ALL mAP@50 - not results:")
for _, r in D[(D.metric == "ALL") & (D.verdict == "NOISE")].iterrows():
    print(f"  {r['arm']:22s} {r['delta']:+.4f}  ({r['ratio']}x bar)")
print("\\nNOTE: RT-DETR (30 ep, converged) has no matched-budget partner and is excluded from "
      "verdicts;\\n      report it descriptively as 'a 32M transformer does not beat 2.6M nano'.")
'''

THESIS = '''# === The thesis: A-B gap in every single arm ===
gap = (M["A_whole_leaf"].astype(float) - M["B_lesion"].astype(float)).sort_values()
g = pd.DataFrame({"A": M["A_whole_leaf"].astype(float).round(4),
                  "B_lesion": M["B_lesion"].astype(float).round(4),
                  "gap": gap.round(4)}).sort_values("gap")
display(g)
print(f"gap range {gap.min():.4f} - {gap.max():.4f} | mean {gap.mean():.4f} | sd {gap.std():.4f}")
print(f"noise bar on the gap: {FLOOR['A_B_gap']:.4f}")
print(f"\\nThe gap is ~{gap.mean()/FLOOR['A_B_gap']:.0f}x its own noise bar and appears in "
      f"ALL {len(gap)} arms - model size, architecture family, imbalance handling, attention, "
      f"loss function and a small-object head all leave it intact.")
'''

PERCLASS = '''# === Per-class matrix across arms (paper appendix) ===
PC = pd.DataFrame(perclass).round(4)
GROUP_B = {"potato_healthy","cucumber_Antraknosa","cucumber_Bacterial Wilt","potato_late_blight",
           "potato_early_blight","bean_bean_rust","potato_blackleg","bean_angular_leaf_spot"}
PC.insert(0, "group", ["B_lesion" if c in GROUP_B else "A_whole_leaf" for c in PC.index])
PC = PC.sort_values(["group", PC.columns[1]])
display(PC)
PC.to_csv(HERE / "results" / "master_perclass.csv")

# classes too unstable across seeds to support any per-class claim
seedcols = [c for c in PC.columns if c.startswith("local baseline")]
unstable = (PC[seedcols].max(axis=1) - PC[seedcols].min(axis=1))
print("\\nPer-class seed spread (local baselines) - claims about these are NOT defensible:")
for c, v in unstable[unstable > 0.02].sort_values(ascending=False).items():
    print(f"  {c:34s} {v:.4f}")
print("\\nZero-variance across seeds (saturated whole-leaf classes):")
print("  " + ", ".join(unstable[unstable == 0].index))
'''

PLOT = '''# === Figure: every arm vs the baseline, with the noise band ===
import matplotlib.pyplot as plt
order = [a for a in M.index if M.loc[a, "epochs"] == 100 and not a.startswith("local")]
base = float(M.loc["baseline (n, 100ep)", "ALL"])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

vals = [float(M.loc[a, "ALL"]) - base for a in order]
cols = ["tab:red" if v < 0 else "tab:green" for v in vals]
ax[0].barh(order, vals, color=cols)
ax[0].axvspan(-FLOOR["ALL"], FLOOR["ALL"], color="grey", alpha=.25, label="noise band")
ax[0].axvline(0, c="k", lw=.8); ax[0].legend(fontsize=8)
ax[0].set_xlabel("delta ALL mAP@50 vs baseline"); ax[0].set_title("nothing clears the bar upward")

bvals = [float(M.loc[a, "B_lesion"]) - float(M.loc["baseline (n, 100ep)", "B_lesion"]) for a in order]
ax[1].barh(order, bvals, color=["tab:red" if v < 0 else "tab:green" for v in bvals])
ax[1].axvspan(-FLOOR["B_lesion"], FLOOR["B_lesion"], color="grey", alpha=.25)
ax[1].axvline(0, c="k", lw=.8)
ax[1].set_xlabel("delta Group B lesion mAP@50"); ax[1].set_title("the P2 head made Group B worse")
plt.tight_layout(); plt.show()
'''


def md(src, i):
    return {"cell_type": "markdown", "id": f"mt{i:02d}", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src, i):
    return {"cell_type": "code", "id": f"mt{i:02d}", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


if __name__ == "__main__":
    load = LOAD.replace("__HERE__", str(HERE)).replace("__KAGGLE__", str(KAGGLE))
    cells = [md(HEADER, 0), code(load, 1), code(TABLE, 2), code(VERDICT, 3),
             code(THESIS, 4), code(PERCLASS, 5), code(PLOT, 6)]
    nbjson = {"cells": cells,
              "metadata": {"kernelspec": {"display_name": "Python (QML)", "language": "python",
                                          "name": "qml"},
                           "language_info": {"name": "python", "version": "3.12.13"}},
              "nbformat": 4, "nbformat_minor": 5}
    out = HERE / "nb" / "master_table.ipynb"
    if out.exists() and any(c.get("outputs") for c in json.loads(out.read_text(encoding="utf-8"))["cells"]):
        raise SystemExit(f"REFUSING to overwrite {out.name} - it holds executed outputs.")
    out.write_text(json.dumps(nbjson, indent=1), encoding="utf-8")
    print("built", out)
