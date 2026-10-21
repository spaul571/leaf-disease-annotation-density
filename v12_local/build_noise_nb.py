"""Generate nb/noise_floor.ipynb - pools local baseline seeds into a defensible noise floor.

Run AFTER the baseline seed notebooks have executed. It reads whatever
results/perclass_local_baseline_100ep_s*.csv files exist, so it works with 2 seeds and gets
tighter with 3.

What it answers: how big must an effect be before it counts? Every claim in the paper gets
re-checked against that number - which is what killed the exp17 CBAM+WIoU result.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

HEADER = '''# Noise floor from local baseline seeds

Same config, same data, **different seeds** - so the spread here is pure run-to-run variance.
Any reported effect smaller than this band is not a result.

Context: the seed-0 local run reproduced the Kaggle T4 baseline on ALL mAP@50 to **+0.0003** and
on mAP@50-95 to **+0.0029**, but drifted **+0.0314** on `potato_blackleg` at the *same* seed.
This notebook turns those one-off drift numbers into a proper variance estimate.

Effects currently on the table, to be judged against the floor:

| effect | size | metric |
|---|---|---|
| oversample-only | +0.005 | ALL mAP@50 |
| CBAM+WIoU | +0.0007 | ALL mAP@50 |
| CBAM+WIoU | +0.024 | mAP@50-95 |
| P2 head | −0.017 | ALL mAP@50 |
| **P2 head** | **−0.029** | **Group B lesion** |
| P2+WIoU | −0.037 | ALL mAP@50 |
| focal loss | −0.032 | ALL mAP@50 |
'''

LOAD = '''# === Load every available local baseline seed ===
import json, re
from pathlib import Path
import pandas as pd, numpy as np

HERE = Path(r"{here}")
RES  = HERE / "results"
# Match ONLY plain seed files: an imgsz-suffixed arm (e.g. ..._s0_1024.csv) also matches a
# naive s*.csv glob and must never be mixed into a 640 noise floor.
_pat = re.compile(r"^perclass_local_baseline_100ep_s(\d+)\.csv$")
files = [f for f in sorted(RES.glob("perclass_local_baseline_100ep_s*.csv"))
         if _pat.match(f.name)]
assert files, "no baseline seed CSVs found - run the seed notebooks first"

frames = {{}}
for f in files:
    s = int(_pat.match(f.name).group(1))
    frames[s] = pd.read_csv(f)
seeds = sorted(frames)
print("seeds found:", seeds)
# HARD STOP: with one seed every spread is 0, the floor collapses to 0, and every effect below
# would be stamped "REAL". Refuse rather than emit a flattering lie.
assert len(seeds) >= 2, (
    f"only seed(s) {{seeds}} present - a variance estimate needs >=2 (3 preferred). "
    f"Run nb/local_baseline_s1_640.ipynb and nb/local_baseline_s2_640.ipynb first.")
if len(seeds) == 2:
    print("NOTE: n=2 - spread is a floor estimate, not a real sd. Treat it as a lower bound.")

GROUP_B_LESION = {{
    "potato_healthy", "cucumber_Antraknosa", "cucumber_Bacterial Wilt",
    "potato_late_blight", "potato_early_blight", "bean_bean_rust",
    "potato_blackleg", "bean_angular_leaf_spot",
}}
T4 = {{"ALL": 0.8975, "A_whole_leaf": 0.9695, "B_lesion": 0.7716,
      "recall": 0.8697, "mAP50_95": 0.7268}}   # 0.7268 = exp6b 100ep. 0.7083 is exp10 @60ep.
'''

AGG = '''# === Per-seed summary metrics ===
rows = []
for s, df in frames.items():
    allr = df[df["class"] == "ALL"].iloc[0]
    per  = df[df["class"] != "ALL"]
    g = per.groupby("group")["mAP50"].mean()
    rows.append({
        "seed": s,
        "ALL": allr["mAP50"],
        "A_whole_leaf": g["A_whole_leaf"],
        "B_lesion": g["B_lesion"],
        "recall": allr["recall"],
        "mAP50_95": allr["mAP50_95"],
        "A_B_gap": g["A_whole_leaf"] - g["B_lesion"],
    })
S = pd.DataFrame(rows).set_index("seed").sort_index()
display(S.round(4))
'''

FLOOR = '''# === The noise floor ===
# spread = max-min across seeds (honest for n=2-3, where a std is barely defined)
METRICS = ["ALL", "A_whole_leaf", "B_lesion", "recall", "mAP50_95", "A_B_gap"]
floor = pd.DataFrame({
    "mean":   S[METRICS].mean(),
    "std":    S[METRICS].std(ddof=1),
    "min":    S[METRICS].min(),
    "max":    S[METRICS].max(),
    "spread": S[METRICS].max() - S[METRICS].min(),
})
floor["+/-2sd"] = 2 * floor["std"]
# Every reported effect is a DIFFERENCE between two single runs (arm vs baseline), and the sd of
# a difference of two independent runs is sd*sqrt(2) - so the bar is 1.96*sd*sqrt(2), NOT 2*sd.
# Using 2*sd here would be too lenient and would wave marginal effects through as real.
floor["diff_95"] = 1.96 * np.sqrt(2) * floor["std"]
display(floor.round(5))

THRESH = {m: float(floor.loc[m, "diff_95"]) for m in METRICS}
print("\\nMinimum |effect| for a single-run-vs-single-run comparison to clear 95%:")
for m, v in THRESH.items():
    print(f"  {m:14s} > {v:.4f}   (sd {floor.loc[m,'std']:.5f}, n={len(S)})")
print("\\nNOTE: this covers SEED noise only, so it applies to comparisons made WITHIN one\\n"
      "environment (e.g. a Kaggle arm vs the Kaggle baseline). Aggregate metrics port across\\n"
      "environments well (ALL +0.0003, mAP50-95 +0.0029 at a fixed seed), but per-class ones do\\n"
      "NOT: potato_blackleg drifted +0.0314.")
'''

VERDICT = '''# === Re-judge every reported effect against the floor ===
EFFECTS = [
    ("oversample-only",  "ALL",       +0.0050),
    ("CBAM+WIoU",        "ALL",       +0.0007),
    ("CBAM+WIoU",        "mAP50_95",  +0.0240),
    ("P2 head",          "ALL",       -0.0169),
    ("P2 head",          "B_lesion",  -0.0292),
    ("P2+WIoU",          "ALL",       -0.0370),
    ("P2+WIoU",          "B_lesion",  -0.0614),
    ("focal loss",       "ALL",       -0.0324),
]
def verdict(ratio):
    # >2x the 95% difference bar = comfortable; 1-2x = clears the bar but only just, and with
    # n=3 the sd itself is loosely estimated - do not build a headline on it.
    if ratio >= 2.0: return "REAL"
    if ratio >= 1.0: return "MARGINAL"
    return "NOISE"

out = []
for name, metric, eff in EFFECTS:
    t = THRESH[metric]
    ratio = abs(eff) / t if t else np.inf
    out.append({"effect": name, "metric": metric, "size": eff, "floor": round(t, 4),
                "ratio": round(ratio, 2), "verdict": verdict(ratio)})
V = pd.DataFrame(out)
display(V)

for v, label in (("REAL", "Solid - safe to report as findings"),
                 ("MARGINAL", "Marginal - clears the bar but only just; report with the floor"),
                 ("NOISE", "Not distinguishable from seed noise - do NOT report as results")):
    sel = [r for r in out if r["verdict"] == v]
    if sel:
        print(f"\\n{label}:")
        for r in sel:
            print(f"  {r['effect']:16s} {r['metric']:10s} {r['size']:+.4f} "
                  f"({r['ratio']}x floor {r['floor']})")
'''

PERCLASS = '''# === Per-class variance: which classes are measurable at all? ===
piv = pd.concat([f.assign(seed=s) for s, f in frames.items()])
piv = piv[piv["class"] != "ALL"]
pc = piv.pivot_table(index="class", columns="seed", values="mAP50")
pc["spread"] = pc.max(axis=1) - pc.min(axis=1)
pc["group"] = [("B_lesion" if c in GROUP_B_LESION else "A_whole_leaf") for c in pc.index]
display(pc.sort_values("spread", ascending=False).round(4))

unstable = pc[pc["spread"] > 0.02]
print(f"\\n{len(unstable)}/{len(pc)} classes swing >0.02 between seeds - per-class claims about "
      f"these are not defensible:")
for c, r in unstable.iterrows():
    print(f"  {c:34s} spread {r['spread']:.4f}  ({r['group']})")
sat = list(pc[pc["spread"] == 0].index)
print("\\nSaturated / zero-variance classes (identical across seeds) - supports the "
      "whole-leaf-is-trivial thesis:")
print("  " + (", ".join(sat) if sat else "none"))
'''

PLOT = '''# === Visual: seed spread vs the effects being claimed ===
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

for m in ["ALL", "A_whole_leaf", "B_lesion"]:
    ax[0].plot(S.index, S[m], "o-", label=m)
    ax[0].axhline(T4[m], ls=":", alpha=.5)
ax[0].set_xlabel("seed"); ax[0].set_xticks(S.index)
ax[0].set_ylabel("mAP@50"); ax[0].legend(fontsize=8)
ax[0].set_title("seed spread (dotted = Kaggle T4)"); ax[0].grid(alpha=.3)

lab = [f"{n}\\n{m}" for n, m, _ in EFFECTS]
val = [abs(e) for _, _, e in EFFECTS]
flr = [THRESH[m] for _, m, _ in EFFECTS]
x = np.arange(len(lab))
ax[1].bar(x - .2, val, .4, label="|effect|")
ax[1].bar(x + .2, flr, .4, label="noise floor", color="grey")
ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=6, rotation=45, ha="right")
ax[1].legend(fontsize=8); ax[1].set_title("effect vs floor"); ax[1].grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()

(HERE / "results" / "noise_floor.json").write_text(json.dumps(
    {"seeds": [int(s) for s in S.index], "threshold": {k: float(v) for k, v in THRESH.items()},
     "per_seed": S.round(6).to_dict()}, indent=2))
print("SAVED results/noise_floor.json")
'''


def md(src, i):
    return {"cell_type": "markdown", "id": f"nf{i:02d}", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src, i):
    return {"cell_type": "code", "id": f"nf{i:02d}", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


if __name__ == "__main__":
    cells = [md(HEADER, 0), code(LOAD.format(here=str(HERE)), 1), code(AGG, 2),
             code(FLOOR, 3), code(VERDICT, 4), code(PERCLASS, 5), code(PLOT, 6)]
    nbjson = {"cells": cells,
              "metadata": {"kernelspec": {"display_name": "Python (QML)", "language": "python",
                                          "name": "qml"},
                           "language_info": {"name": "python", "version": "3.12.13"}},
              "nbformat": 4, "nbformat_minor": 5}
    out = HERE / "nb" / "noise_floor.ipynb"
    out.parent.mkdir(exist_ok=True)
    if out.exists() and any(c.get("outputs") for c in json.loads(out.read_text(encoding="utf-8"))["cells"]):
        raise SystemExit(f"REFUSING to overwrite {out.name} - it holds executed outputs.")
    out.write_text(json.dumps(nbjson, indent=1), encoding="utf-8")
    print("built", out)
