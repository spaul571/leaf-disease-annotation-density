"""Render the manuscript tables (Markdown) from the generated CSV/JSON assets.

Reads only files produced by `build_paper_assets.py`, `build_table_dataset.py` and
`build_reviewer_stats.py`, so the manuscript tables cannot drift from the analysis. Writes
<D.out>/tables_generated.md: paper/ for v13 (the default), paper_v12/ for v12.

Since 2026-10-21 the manuscript's 02_Journal/CEA_manuscript/30_tables.md has hand-written
captions and its own column names ("Variant", "Convention"), so this script must never write
there. Copy changed data rows from tables_generated.md into it by hand.

Manuscript numbering differs from the asset filenames:
  Table 1 <- table_dataset.csv
  Table 2 <- table_noise.csv
  Table 3 <- table1_arms.csv            (v13: Table 3b <- table1_arms_env1_v12.csv)
  Table 4 <- table2_deltas.csv, pivoted (v13: Table 4b <- table2_deltas_env1_v12.csv)
  Table 5 <- table3_test.csv
  Table 6 <- results/<tag>reviewer_stats.json, R1-1
  Table 7 <- results/<tag>reviewer_stats.json, R1-2
  Table 8 <- results/<tag>reviewer_stats.json, R1-3 (worker control, run on v12)

    qpy build_paper_tables_md.py [--dataset v12]
"""
import argparse
import json
import textwrap
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

import paper_data as P

D = P.dataset(P.add_dataset_arg(argparse.ArgumentParser()).parse_args().dataset)
PAP = D.out
OUT = PAP / "tables_generated.md"
V13 = D.name != "v12"
N_IMG = P.images_on_disk(D)
TOTAL = sum(N_IMG.values())


def md(df, align=None):
    cols = list(df.columns)
    align = align or ["---"] * len(cols)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "| " + " | ".join(align) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(out)


parts = [f"# Tables ({D.root.name})"]

# ---------------- Table 1 ----------------
t1 = pd.read_csv(PAP / "table_dataset.csv")
t1 = t1.rename(columns={
    "class": "Class", "crop": "Crop", "group": "Protocol group", "images": "Images",
    "boxes": "Boxes", "boxes_per_img": "Boxes/image", "median_area": "Median box area",
    "mixed_frac": "Mixing", "mAP50": "mAP@50", "recall": "Recall"})
t1["Protocol group"] = t1["Protocol group"].map({"A_whole_leaf": "A (whole-leaf)",
                                                 "B_lesion": "B (lesion)"})
t1["Class"] = t1["Class"].str.replace("_", " ", regex=False)
for c in ("Median box area", "Mixing", "mAP@50", "Recall"):
    t1[c] = t1[c].map(lambda v: f"{v:.3f}")
t1["Boxes/image"] = t1["Boxes/image"].map(lambda v: f"{v:.2f}")
# Crop is recoverable from the class-name prefix and per-class recall is never discussed, so
# both are dropped: a 10-column table does not fit the page and is unreadable when it does.
t1 = t1.drop(columns=["Crop", "Recall"]).rename(columns={
    "Protocol group": "Protocol", "Boxes/image": "Boxes per image", "Median box area": "Box area",
    "Mixing": "Mixed"})
parts += [
    "\n## Table 1\n",
    "Per-class composition and annotation statistics of the benchmark, with baseline accuracy "
    "for reference. Protocol A is the whole-leaf convention and B the lesion convention: a class "
    "is in B when its median box covers less than half the image. "
    "`Images` counts images containing at least one box of that class, so the column sums to "
    f"more than the {TOTAL:,} images in the benchmark because a few images carry boxes of two "
    "classes. `Box area` is the median annotated box area as a fraction of image area. `Mixed` "
    "is the fraction of a class's boxes drawn in the opposite convention from its own majority, "
    "and is a lower bound (Section 3.3). `mAP@50` is the mean of three baseline replicates on "
    "the validation split. Rows are ordered by protocol group, then by descending annotation "
    "density.\n",
    md(t1), ""]

# ---------------- Table 2 ----------------
nz = pd.read_csv(PAP / "table_noise.csv", index_col="metric")
KEY = [("Class mean mAP@50", "ALL"), ("Group A mAP@50", "A"), ("Group B mAP@50", "B"),
       ("Recall", "recall"), ("mAP@50–95", "mAP50_95")]
rows = []
for label, k in KEY:
    r = nz.loc[k]
    row = {"Metric": label, "Validation sd": f"{r.val_sd:.4f}", "Validation bar": f"{r.val_bar:.4f}",
           "Test sd": f"{r.test_sd:.4f}", "Test bar (single run)": f"{r.test_bar:.4f}"}
    if V13:
        row["v12 validation bar"] = f"{r.v12_val_bar:.4f}"
    rows.append(row)
parts += [
    "\n## Table 2\n",
    "The measured noise floor. Standard deviations are computed across three baseline replicates "
    "differing only in random seed. The significance bar for a difference between two single runs "
    "is 1.96·sd·√2. Comparisons of means at three seeds per arm on the test split are "
    "made with Welch's t-test instead (Table 5), because the pooled standard deviation differs "
    "for each pair of arms and no single bar applies. The lesion group (B) is far noisier on "
    "the test split than on validation because the test split contains few lesion-class "
    "instances (Section 4.2)."
    + (" The last column is the v12 validation bar, against which the Environment 1 arms, run on "
       "v12 only, are judged (Tables 3b and 4b)." if V13 else "") + "\n",
    md(pd.DataFrame(rows)), ""]


# ---------------- Tables 3 and 4 (and 3b, 4b on v13) ----------------
def arms_table(path):
    t3 = pd.read_csv(path, index_col=0)
    t3.index.name = "Arm"
    t3 = t3.reset_index().rename(columns={
        "n_seeds": "Seeds", "ALL": "Class mean mAP@50", "A": "Group A", "B": "Group B",
        "recall": "Recall", "mAP50_95": "mAP@50–95", "A_B_gap": "A−B gap"})
    t3["Seeds"] = t3["Seeds"].astype(int)
    for c in t3.columns[2:]:
        t3[c] = t3[c].map(lambda v: f"{v:.4f}")
    return t3


_MARK = {"baseline": "a", "baseline (Environment 1)": "b", "n @60ep": "c"}


def deltas_table(path, mark=True):
    # Each cell carried the delta, the ratio and a verdict word, which made the row 118
    # characters wide and unreadable at page width. Report the standardised effect instead: the
    # change divided by that metric's own significance bar, signed. The bar is metric-specific
    # (Table 2) and the absolute value of every arm is in Table 3, so the raw delta remains
    # recoverable and nothing is lost. The reference column becomes a footnote marker.
    d = pd.read_csv(path)
    d["signed"] = np.sign(d["delta"]) * d["ratio"]
    d["cell"] = d["signed"].map(lambda v: f"**{v:+.1f}**" if abs(v) >= 2 else f"{v:+.1f}")
    piv = d.pivot(index=["arm", "vs"], columns="metric", values="cell").reset_index()
    piv = piv[["arm", "vs", "ALL", "A", "B", "recall", "mAP50_95"]]
    if mark:   # off for v13 Table 4: one reference, named in its caption
        piv["arm"] = [f"{a} ({_MARK.get(v, '?')})" for a, v in zip(piv["arm"], piv["vs"])]
    piv = piv.drop(columns=["vs"])
    piv.columns = ["Arm", "Class mean mAP@50", "Group A", "Group B", "Recall", "mAP@50–95"]
    return piv


STD_EFFECT = ("Values are standardised effects: the change in that metric divided by its own "
              "significance bar from Table 2, carrying the sign of the change. A magnitude of 2 "
              "or more is a real effect and is shown in bold, 1 to 2 is marginal, and below 1 is "
              "indistinguishable from seed noise.")
t3 = arms_table(PAP / "table1_arms.csv")
piv = deltas_table(PAP / "table2_deltas.csv", mark=not V13)
if V13:
    parts += [
        "\n## Table 3\n",
        f"All Environment 2 arms on the {D.name} validation split. `baseline` is the three-seed "
        "mean; every other arm is compared with it (Table 4). The Environment 1 arms were run on "
        "v12 only and are reported separately (Table 3b).\n",
        md(t3), "",
        "\n## Table 4\n",
        "Every Environment 2 effect against the three-seed baseline, on the validation split. "
        + STD_EFFECT + " Absolute values for every arm are given in Table 3.\n",
        md(piv), "",
        "\n## Table 3b\n",
        "Environment 1 arms, run on v12 only. Their validation split shares rotated and flipped "
        "twins with training (Section 3.2), so these results carry that caveat. `baseline "
        "(Environment 1)` is the single-seed reference for the 100-epoch arms and `n @60ep` for "
        "the 60-epoch size sweep.\n",
        md(arms_table(PAP / "table1_arms_env1_v12.csv")), "",
        "\n## Table 4b\n",
        "Environment 1 effects against their matched references, judged against the v12 "
        "validation bar (Table 2, last column). " + STD_EFFECT + " Reference arms: (b) the "
        "single-seed baseline in Environment 1, (c) the 60-epoch nano arm. Arms with no matched "
        "reference at their own epoch budget carry no row.\n",
        md(deltas_table(PAP / "table2_deltas_env1_v12.csv")), ""]
else:
    parts += [
        "\n## Table 3\n",
        "All arms on the validation split. `baseline` is the three-seed mean in Environment 2; "
        "`baseline (Environment 1)` is the single-seed reference for the arms run in that "
        "environment. Arms are only comparable to a reference sharing their epoch budget and run "
        "environment; Table 4 pairs each arm with its correct reference.\n",
        md(t3), "",
        "\n## Table 4\n",
        "Every effect against its correctly matched reference, on the validation split. "
        + STD_EFFECT + " Absolute values for every arm are given in Table 3, so raw differences "
        "remain recoverable. Reference arms: (a) the three-seed baseline in Environment 2, (b) "
        "the single-seed baseline in Environment 1, (c) the 60-epoch nano arm. Arms with no "
        "matched reference at their own epoch budget (`n @60ep`, `RT-DETR-l @30ep`, and the two "
        "baselines) carry no row.\n",
        md(piv), ""]

# ---------------- Table 5 ----------------
t5 = pd.read_csv(PAP / "table3_test.csv")
t5 = t5.rename(columns={"system": "System", "ALL": "Class mean mAP@50", "A": "Group A",
                        "B": "Group B", "recall": "Recall", "mAP50_95": "mAP@50–95"})


def _r3(x):
    """Round half-up to three decimals.

    table3_test.csv carries six decimals, so this is in effect a single rounding of the raw
    value. When it carried four, rounding again made the mAP@50-95 p-value 0.006 (raw 0.00547)
    and one sd 0.006 (raw 0.00548). Both are 0.005.
    """
    return str(Decimal(str(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _tidy(v):
    """Four decimals on a mean and its sd overstates precision and doubles the column width."""
    v = v.strip()
    if "+/-" in v:
        m, sd = (x.strip() for x in v.split("+/-"))
        return f"{_r3(m)} ± {_r3(sd)}"
    return _r3(v)


t5["System"] = (t5["System"].str.replace("s@1024 + TTA", "YOLOv11s 1024 + TTA", regex=False)
                            .str.replace("s@1024", "YOLOv11s 1024", regex=False)
                            .str.replace(": Welch p vs baseline", ", Welch p", regex=False)
                            .str.replace(": Holm p", ", Holm p", regex=False))
for c in t5.columns[1:]:
    t5[c] = t5[c].map(_tidy)
parts += [
    "\n## Table 5\n",
    "Held-out test split, the result of record. Three seeds per arm, reported as mean ± "
    "standard deviation, with Welch's two-sample *t*-test (unequal variances) against the "
    "baseline for every metric, and the Holm-adjusted p-value across the five metrics of each "
    "comparison. The configuration was selected on the validation split "
    "(Section 3.6, Section 4.5) and measured once here.\n",
    md(t5), ""]

# ---------------- Table 6: confound controls (Reviewer 1, comment 1) ----------------
rs = json.loads((P.RES / f"{D.tag}reviewer_stats.json").read_text())
c = rs["R1_1_confounds"]


_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _p(v):
    """Format a p-value: three decimals above 0.001, else m x 10^-k with real superscripts."""
    if v is None or not np.isfinite(v):
        return "n/a"
    if v >= 1e-3:
        return f"{v:.3f}"
    mant, exp = f"{v:.1e}".split("e")
    return f"{mant} × 10⁻{str(abs(int(exp))).translate(_SUP)}"


rows6 = []
for lab, k in (("boxes per image (density)", "boxes_per_img"),
               ("median box area", "median_area"),
               ("class frequency (log boxes)", "log_n_boxes"),
               ("within-class protocol mixing", "mixed_frac")):
    u = c["univariate"][k]
    rows6.append({"Analysis": f"Univariate: {lab}", "Estimate": f"r = {u['pearson_r']:+.3f}",
                  "p": _p(u["pearson_p"]), "Note": f"Spearman {u['spearman_r']:+.3f}"})
for lab, k in (("controlling median box area", "control_median_area"),
               ("controlling class frequency", "control_log_n_boxes"),
               ("controlling both", "control_both")):
    pc = c["partial"][k]
    rows6.append({"Analysis": f"Partial correlation, density vs mAP@50, {lab}",
                  "Estimate": f"r = {pc['r']:+.3f}", "p": _p(pc["p"]), "Note": f"df = {pc['df']}"})
mr = c["multiple_regression"]
for k, lab in (("boxes_per_img", "density"), ("median_area", "median box area"),
               ("log_n_boxes", "class frequency")):
    co = mr["coef"][k]
    rows6.append({"Analysis": f"Multiple regression: {lab}",
                  "Estimate": f"β = {co['beta']:+.3f} ± {co['se']:.3f}",
                  "p": _p(co["p"]), "Note": f"VIF {mr['vif'][k]:.2f}"})
fe, mm = c["crop_fixed_effects"], c["mixed_model"]
rows6.append({"Analysis": "Density with crop (= source dataset) as fixed effect",
              "Estimate": f"β = {fe['beta']:+.3f} ± {fe['se']:.3f}", "p": _p(fe["p"]),
              "Note": f"R² = {fe['r2']:.3f}"})
if "beta" in mm:
    ok = mm.get("converged", True) and np.isfinite(mm["se"])
    rows6.append({"Analysis": "Density with crop as random intercept",
                  "Estimate": f"β = {mm['beta']:+.3f}" + (f" ± {mm['se']:.3f}" if ok else ""),
                  "p": _p(mm["p"]),
                  "Note": (f"crop variance {mm['group_var']:.3f} ({mm['n_groups']} crops)"
                           if ok else "did not converge; standard error not estimable")})
parts += [
    "\n## Table 6\n",
    "Does annotation density survive controls for object size, class frequency and source "
    "dataset? All rows use per-class mAP@50 averaged over the three baseline replicates on the "
    "validation split, n = 22 classes. The multiple regression attains R² = "
    f"{mr['r2']:.3f} (adjusted {mr['adj_r2']:.3f}). Each of the six source datasets contributed "
    "exactly one crop, so crop and source dataset are the same factor in this benchmark and "
    "cannot be separated; conditioning on crop is therefore the available source control. A "
    "crop variance estimated at zero is a singular fit, meaning crop explains no residual "
    "variance once density is in the model. These are controls on an observational design, not "
    "evidence of causation (Section 4.1).\n",
    md(pd.DataFrame(rows6)), ""]

# ---------------- Table 7: per-seed results (Reviewer 1, comment 2) ----------------
s = rs["R1_2_seeds_and_intervals"]
rows7 = []
for seed, vals in sorted(s["val_per_seed"].items()):
    rows7.append({"Split": "validation", "Arm": "baseline (640 px)", "Seed": seed,
                  "Class mean mAP@50": f"{vals['ALL']:.4f}", "Group A": f"{vals['A']:.4f}",
                  "Group B": f"{vals['B']:.4f}", "Recall": f"{vals['recall']:.4f}",
                  "mAP@50–95": f"{vals['mAP50_95']:.4f}"})
_LBL = {"baseline": "baseline (640 px)", "s@1024": "YOLOv11s 1024",
        "s@1024+TTA": "YOLOv11s 1024 + TTA"}
for arm in ("baseline", "s@1024", "s@1024+TTA"):
    for vals in s["test_per_seed"].get(arm, []):
        rows7.append({"Split": "test", "Arm": _LBL[arm], "Seed": str(vals["seed"]),
                      "Class mean mAP@50": f"{vals['ALL']:.4f}", "Group A": f"{vals['A']:.4f}",
                      "Group B": f"{vals['B']:.4f}", "Recall": f"{vals['recall']:.4f}",
                      "mAP@50–95": f"{vals['mAP50_95']:.4f}"})
parts += [
    "\n## Table 7\n",
    "Individual seed results behind every multi-seed mean reported in this paper. Arms differ "
    "only in the random seed passed to the training framework. The validation rows are the three "
    "baseline replicates that define the noise floor of Table 2; the test rows are the three "
    "arms of Table 5. At three seeds per arm a single-arm 95% interval uses t(2) = 4.303, so "
    "intervals are reported on differences between arms rather than on individual means "
    "(Section 3.8).\n",
    md(pd.DataFrame(rows7)), ""]

# ---------------- Table 8: worker-count control (Reviewer 1, comment 3), run on v12 ----------
wc = rs["R1_3_worker_control_v12"]
_WL = {"ALL": "Class mean mAP@50", "A": "Group A", "B": "Group B", "recall": "Recall",
       "mAP50_95": "mAP@50–95"}
rows8 = [{"Metric": _WL[r["metric"]],
          "8 workers": f"{r['workers=8']:.4f} ± {r['sd8']:.4f}",
          "2 workers": f"{r['workers=2']:.4f} ± {r['sd2']:.4f}",
          "Difference": f"{r['delta']:+.4f}", "95% bar": f"{r['bar']:.4f}",
          "Ratio": f"{r['ratio']:.2f}", "Welch p": f"{r['welch_p']:.3f}"} for r in wc["comparison"]]
cost = ""
wc_path = P.RES / "worker_control.json"
if wc_path.exists() and json.loads(wc_path.read_text()).get("cost_hours"):
    hrs = pd.DataFrame(json.loads(wc_path.read_text())["cost_hours"]).groupby("arm")["hours"].mean()
    cost = (" Mean wall-clock per 100-epoch run on the same machine: "
            + "; ".join(f"{a}, {h:.2f} h" for a, h in hrs.items()) + ".")
parts += [
    "\n## Table 8\n",
    "Dataloader worker-count control"
    + (", run on v12 before the v13 rebuild; every v13 run uses two workers" if V13 else "")
    + ". The 640 px baseline was retrained with two worker processes instead of eight, three "
    "seeds, every other setting held fixed, and compared against the three-seed eight-worker "
    "baseline. The bar is 1.96·sd_pooled·√(2/3), "
    "the same mean-versus-mean rule used elsewhere in this paper, and the ratio is the "
    "difference divided by that bar. No metric reaches its bar, so worker count carries no "
    "effect detectable at three seeds per arm. The two sets of runs were measured two months "
    "apart on the same machine, so this null also covers environment drift over that "
    "interval." + cost + "\n",
    md(pd.DataFrame(rows8)), ""]


def _wrap(part):
    """Wrap caption prose to 95 columns, preserving the surrounding blank lines.

    The leading/trailing newlines are structural: Markdown only recognises a table when a blank
    line precedes it, so stripping them turns every table into a paragraph of literal pipes.
    """
    if not part.strip() or part.lstrip().startswith(("|", "#")):
        return part
    lead = part[:len(part) - len(part.lstrip("\n"))]
    trail = part[len(part.rstrip("\n")):]
    body = textwrap.fill(" ".join(part.split()), width=95,
                         break_long_words=False, break_on_hyphens=False)
    return lead + body + trail


OUT.write_text("\n".join(_wrap(p) for p in parts), encoding="utf-8")
print("wrote", OUT)
print(f"Table 1: {len(t1)} rows | Table 3: {len(t3)} rows | Table 4: {len(piv)} rows | "
      f"Table 5: {len(t5)} rows")
