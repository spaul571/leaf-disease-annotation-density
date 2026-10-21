"""Final paper tables + figures.

RULE ON SOURCES: where an arm was reproduced locally, the local (multi-seed) result is used.
Arms with no local replicate keep their original numbers.
CRITICAL: an arm is always compared against the baseline measured in the SAME run environment.
Mixing an arm from one environment against a baseline from another would fold environment drift
into the reported effect, so every arm carries its own `ref`.
Platform/hardware is deliberately absent from all outputs - it is an implementation detail.

Datasets (paper_data.py). With --dataset v13, the default, the local arms come from v13 and are
judged against the v13 noise floor. The Environment 1 arms exist on v12 only, so they go to their
own tables (table1_arms_env1_v12.csv, table2_deltas_env1_v12.csv, fig2b_*) and are judged against
the v12 floor. With --dataset v12 every arm is on v12 and in the same tables, as in the published
paper. Group means are recomputed with classes.py (potato_healthy is Group A since 2026-10-21).
Arms whose runs have not finished are skipped with a message.

    qpy build_paper_assets.py                 # v13 -> paper/
    qpy build_paper_assets.py --dataset v12   # v12 -> paper_v12/
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import paper_data as P
from classes import group_of

D = P.dataset(P.add_dataset_arg(argparse.ArgumentParser()).parse_args().dataset)
OUT = D.out
OUT.mkdir(exist_ok=True)

# Elsevier wants >=300 dpi for halftone and 500-1000 dpi for line art; these are line art.
# Vector PDF is what production actually prefers, so emit both and let the submission pick.
FIG_DPI = 600


def save_fig(name):
    """Write <name>.png at FIG_DPI and <name>.pdf as vector, both tightly cropped."""
    plt.savefig(OUT / f"{name}.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(OUT / f"{name}.pdf", bbox_inches="tight")


def local_arms(ds):
    """Environment 2 arms of one dataset: means over the seeds that exist."""
    rows, perclass = {}, {}
    for arm, files in ds.val_arms.items():
        have = [f for f in files if f.exists()]
        if not have:
            print(f"skip {arm}: {files[0].name} not written yet")
            continue
        ss = [P.summarise(f) for f in have]
        rows[arm] = {k: float(np.mean([s[0][k] for s in ss])) for k in P.METRICS}
        rows[arm].update(n_seeds=len(ss), ref=None if arm == "baseline" else "baseline")
        perclass[arm] = pd.concat([s[1] for s in ss], axis=1).mean(axis=1)   # mean over seeds
    return rows, perclass


def env1_arms():
    rows, perclass = {}, {}
    for arm, (f, ref) in P.ENV1.items():
        if f.exists():
            s, pc = P.summarise(f)
            rows[arm] = dict(s, n_seeds=1, ref=ref)
            perclass[arm] = pc
    return rows, perclass


def arm_tables(rows, floor, suffix):
    M = pd.DataFrame(rows).T
    M["A_B_gap"] = M["A"] - M["B"]
    cols = ["n_seeds", "ALL", "A", "B", "recall", "mAP50_95", "A_B_gap"]
    M[cols].astype(float).round(4).to_csv(OUT / f"table1_arms{suffix}.csv")
    print(M[cols].astype(float).round(4).to_string())
    d_rows = []
    for arm, r in M.iterrows():
        if pd.isna(r["ref"]) or not r["ref"]:   # None becomes NaN in a DataFrame
            continue
        b = M.loc[r["ref"]]
        for k in P.METRICS:
            eff = float(r[k]) - float(b[k])
            ratio = abs(eff) / floor[k]
            d_rows.append({"arm": arm, "vs": r["ref"], "metric": k,
                           "value": round(float(r[k]), 4), "delta": round(eff, 4),
                           "bar": round(floor[k], 4), "ratio": round(ratio, 2),
                           "verdict": "REAL" if ratio >= 2 else "MARGINAL" if ratio >= 1 else "NOISE"})
    Dl = pd.DataFrame(d_rows)
    Dl.to_csv(OUT / f"table2_deltas{suffix}.csv", index=False)
    if len(Dl):
        print("\n" + Dl[Dl.metric.isin(["ALL", "B"])].to_string(index=False))
    return M, Dl


def effects_axes(ax, Dl, floor, title):
    sub = Dl[Dl.metric == "ALL"].sort_values("delta")
    ax.barh(sub.arm, sub.delta, color=["tab:red" if v < 0 else "tab:green" for v in sub.delta])
    ax.axvspan(-floor["ALL"], floor["ALL"], color="grey", alpha=.3, label="noise band")
    ax.axvline(0, c="k", lw=.8)
    ax.set_xlabel("change in overall mAP@50 vs reference")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="x")


def effects_figure(Dl, floor, name, title):
    if Dl[Dl.metric == "ALL"].empty:
        return
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(Dl[Dl.metric == "ALL"]) + 1.6))
    effects_axes(ax, Dl, floor, title)
    plt.tight_layout()
    save_fig(name)
    plt.close()


def effects_panels(panels, name):
    """One figure, one panel per (deltas, floor, title): the journal takes one file per figure."""
    n = [len(d[d.metric == "ALL"]) for d, _, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 0.45 * sum(n) + 1.8 * len(panels)),
                             gridspec_kw={"height_ratios": n})
    for ax, (d, f, t) in zip(axes, panels):
        effects_axes(ax, d, f, t)
    plt.tight_layout()
    save_fig(name)
    plt.close()


# ---------------- Tables 1 and 2: arms on validation, effects against the noise floor ----------
floor = P.noise_floor(D)
rows, perclass = local_arms(D)
if D.name == "v12":
    r1, p1 = env1_arms()
    rows.update(r1); perclass.update(p1)
M, Dl = arm_tables(rows, floor, "")
if D.name != "v12":
    print(f"\n--- Environment 1 arms (v12 only, judged against the v12 floor) ---")
    floor_v12 = P.noise_floor(P.dataset("v12"))
    r1, _ = env1_arms()
    M1, Dl1 = arm_tables(r1, floor_v12, "_env1_v12")

# ---------------- noise table (manuscript Table 2) ----------------
T = P.test_per_seed(D)
tbar = P.test_bar(T)
noise = pd.DataFrame({"val_sd": {k: floor[k] / (1.96 * np.sqrt(2)) for k in P.METRICS},
                      "val_bar": {k: floor[k] for k in P.METRICS},
                      "test_sd": {k: tbar[k] / (1.96 * np.sqrt(2)) for k in P.METRICS},
                      "test_bar": tbar})
if D.name != "v12":   # the bars the Environment 1 arms are judged against
    noise["v12_val_sd"] = {k: floor_v12[k] / (1.96 * np.sqrt(2)) for k in P.METRICS}
    noise["v12_val_bar"] = {k: floor_v12[k] for k in P.METRICS}
noise.index.name = "metric"
noise.round(6).to_csv(OUT / "table_noise.csv")

# ---------------- Table 3: held-out test, three seeds per system ----------------
# Six decimals, not four: Table 5 and Fig. 5 round these again to three, and rounding a
# 4-decimal value a second time goes wrong at the midpoint. It turned p = 0.00547 into 0.0055
# and then 0.006, and the sd 0.00548 into 0.006; both are 0.005.
W = P.welch_vs_baseline(T)
LABEL = {"baseline": "baseline", "s@1024": "s@1024", "s@1024+TTA": "s@1024 + TTA"}
tt = [dict(system=LABEL[s], **{k: f"{T[s][k].mean():.6f} +/- {T[s][k].std(ddof=1):.6f}"
                               for k in P.METRICS}) for s in P.TEST_SYSTEMS]
for s in P.TEST_SYSTEMS[1:]:
    tt.append(dict(system=f"{LABEL[s]}: Welch p vs baseline",
                   **{k: f"{W[s][k]['p']:.6f}" for k in P.METRICS}))
for s in P.TEST_SYSTEMS[1:]:   # Holm across the five metrics of each comparison
    tt.append(dict(system=f"{LABEL[s]}: Holm p",
                   **{k: f"{W[s][k]['p_holm']:.6f}" for k in P.METRICS}))
pd.DataFrame(tt).to_csv(OUT / "table3_test.csv", index=False)
print("\n" + pd.DataFrame(tt).to_string(index=False))

# ---------------- figures ----------------
prot = pd.read_csv(D.protocol)
bpc = pd.concat([P.perclass(f)[1] for f in D.val_arms["baseline"]])
bpc = bpc.groupby("class", as_index=False)["mAP50"].mean()
dd = prot.merge(bpc, on="class")
r_p, p_p = stats.pearsonr(dd.boxes_per_img, dd.mAP50)

fig, ax = plt.subplots(figsize=(6, 4.2))
for g, c, lab in (("A_whole_leaf", "tab:blue", "whole-leaf classes"),
                  ("B_lesion", "tab:red", "lesion classes")):
    s = dd[dd.group == g]
    ax.scatter(s.boxes_per_img, s.mAP50, c=c, label=lab, s=45, alpha=.85)
z = np.polyfit(dd.boxes_per_img, dd.mAP50, 1)
xs = np.linspace(dd.boxes_per_img.min(), dd.boxes_per_img.max(), 50)
ax.plot(xs, np.polyval(z, xs), "k--", lw=1)
ax.set_xlabel("annotated boxes per image")
ax.set_ylabel("class mAP@50")
ax.set_title(f"Annotation density predicts accuracy (r = {r_p:.2f}, "
             + ("p < 1e-4)" if p_p < 1e-4 else f"p = {p_p:.4f})"))
ax.legend(fontsize=8)
ax.grid(alpha=.3)
plt.tight_layout()
save_fig("fig1_density_law")
plt.close()
print(f"\ndensity law ({D.name}, validation, {len(dd)} classes): r = {r_p:.3f}, p = {p_p:.2e}")

effects_figure(Dl, floor, "fig2_effects_vs_floor", "Interventions against the measured noise floor")
if D.name != "v12":
    # "earlier split" is the manuscript's name for v12 (Section 3.2)
    effects_figure(Dl1, floor_v12, "fig2b_env1_v12_effects_vs_floor",
                   "Environment 1 interventions on the earlier split, against its noise floor")
    # the manuscript's Fig. 3: both of the above as panels (a) and (b) of one file
    effects_panels([(Dl, floor, "(a) Rebuilt split: interventions against its noise floor"),
                    (Dl1, floor_v12, "(b) Earlier split: Environment 1 interventions")],
                   "fig2_effects_panels")

prog = [(a, M.loc[a, "B"]) for a in ("baseline", "n @1024", "s @1024") if a in M.index]
base_b = float(M.loc["baseline", "B"])
fig, ax = plt.subplots(figsize=(5.2, 4))
# The band from Fig. 2 and Table 2. The title used to claim "Resolution and capacity lift the
# bottleneck classes": true on v12, false on v13, where both gains sit inside the band (2026-10-21).
ax.axhspan(base_b - floor["B"], base_b + floor["B"], color="grey", alpha=.3, label="noise band")
ax.plot([p[0] for p in prog], [float(p[1]) for p in prog], "o-", lw=2, ms=8, color="tab:green")
ax.axhline(base_b, ls=":", c="grey")
ax.set_ylabel("lesion-class mAP@50")
ax.set_title("Lesion-class mAP@50 at 640 and 1024 px")
ax.legend(fontsize=8)
ax.grid(alpha=.3)
plt.tight_layout()
save_fig("fig3_groupB_progression")
plt.close()

PC = pd.DataFrame(perclass).round(3)
PC.insert(0, "group", [group_of(c) for c in PC.index])
PC.sort_values(["group"]).to_csv(OUT / "table4_perclass.csv")
print("\nwrote:", ", ".join(sorted(p.name for p in OUT.iterdir() if p.is_file())))
