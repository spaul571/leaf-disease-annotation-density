"""LaTeX rows for the manuscript's appendix tables (Shakib review, revision/PLAN_Shakib.md).

  Table A1 (tab:robustness) <- results/v13_reviewer_stats.json, key S2_robustness (S-2)
  Table A2 (tab:splits)     <- paper/tableA2_splits.csv, written by build_table_dataset.py (S-5)
  Table A3 (tab:subset699)  <- results/eval_v13_not_v12valid_stats.json (S-3)

Every value is rounded once, from the unrounded number, half up. p-values follow the manuscript:
below 0.001 as a × 10^-b, below 0.1 to three decimals, otherwise to two. The rows are written to
paper/appendix_rows.tex and copied into latex/manuscript.tex by hand, like every other table.

    qpy build_appendix_tables.py
"""
import json
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

import paper_data as P

SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
D = P.dataset("v13")


def rnd(x, places):
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def num(x, places=3):
    s = f"{rnd(x, places):f}"
    return s.replace("-", "−") if not s.startswith("-0.000") else s[1:]


def pval(p):
    if p < 0.001:
        m, e = f"{p:.1e}".split("e")
        return f"{m} × 10{str(int(e)).translate(SUP)}"
    return f"{rnd(p, 3 if p < 0.1 else 2):f}"


def corr(d):
    return f"{num(d['pearson_r'])}, p = {pval(d['pearson_p'])}"


out = []
R = json.loads((P.RES / f"{D.tag}reviewer_stats.json").read_text())["S2_robustness"]
A, B = R["all"], R["without_potato_blackleg"]
rows = [
    ("Pearson r, density (R²)", lambda c: f"{num(c['density']['pearson_r'])} ({num(c['density']['r2'], 2)}), p = {pval(c['density']['pearson_p'])}"),
    ("Spearman ρ, density", lambda c: f"{num(c['density']['spearman_r'])}, p = {pval(c['density']['spearman_p'])}"),
    ("Pearson r, log density (R²)", lambda c: f"{num(c['log_density']['pearson_r'])} ({num(c['log_density']['r2'], 2)}), p = {pval(c['log_density']['pearson_p'])}"),
    ("Pearson r, log density vs logit mAP@50", lambda c: corr(c["log_density_vs_logit_map"])),
    ("Beta regression on log density, coefficient", lambda c: f"{num(c['beta_regression_log_density']['coef'], 2)}, p = {pval(c['beta_regression_log_density']['p'])}"),
    ("Partial r, density, controlling box area", lambda c: f"{num(c['partial']['density|area']['r'])}, p = {pval(c['partial']['density|area']['p'])}"),
    ("Partial r, density, controlling log frequency", lambda c: f"{num(c['partial']['density|freq']['r'])}, p = {pval(c['partial']['density|freq']['p'])}"),
    ("Partial r, density, controlling both", lambda c: f"{num(c['partial']['density|area+freq']['r'])}, p = {pval(c['partial']['density|area+freq']['p'])}"),
    ("Partial r, log density, controlling box area", lambda c: f"{num(c['partial']['log_density|area']['r'])}, p = {pval(c['partial']['log_density|area']['p'])}"),
    ("Partial Spearman, density, controlling box area", lambda c: f"{num(c['partial']['spearman density|area']['r'])}, p = {pval(c['partial']['spearman density|area']['p'])}"),
    ("Partial Spearman, density, controlling both", lambda c: f"{num(c['partial']['spearman density|area+freq']['r'])}, p = {pval(c['partial']['spearman density|area+freq']['p'])}"),
    ("Multiple regression, p of density; of box area", lambda c: f"{pval(c['multiple_regression']['p_density'])}; {pval(c['multiple_regression']['p_median_area'])}"),
    ("Crop fixed effect, p of density; of log density", lambda c: f"{pval(c['crop_fixed_effect_density']['p'])}; {pval(c['crop_fixed_effect_log_density']['p'])}"),
    ("Density added to the A/B grouping, p (linear; log)", lambda c: f"{pval(c['group_plus_density']['p'])}; {pval(c['group_plus_log_density']['p'])}"),
]
out.append("% Table A1 rows: analysis & all 22 classes & without potato_blackleg")
for name, f in rows:
    out.append(f"{name} & {f(A)} & {f(B)} \\\\")
S = R["subsets"]
out.append(f"Within Group A (15 classes) & {corr(S['group_A'])} & unchanged \\\\")
out.append(f"Within Group B (7; 6 without) & {corr(S['group_B'])} & {corr(S['group_B_without_potato_blackleg'])} \\\\")
out.append(f"Classes below 0.98 (15; 14 without) & {corr(S['nonsaturated'])} & {corr(S['nonsaturated_without_potato_blackleg'])} \\\\")
T = R["test_split"]
out.append(f"Test split, all classes (22; 21 without) & {corr(T['all'])} & {corr(T['all_without_potato_blackleg'])} \\\\")
out.append(f"Test split, at least five test boxes (18; 17 without) & {corr(T['ge5'])} & {corr(T['ge5_without_potato_blackleg'])} \\\\")
lo = R["leave_one_out"]["potato_blackleg"]
out.append(f"% note: potato_blackleg leverage {num(lo['leverage'], 2)} (mean {num(lo['mean_leverage'], 2)}), "
           f"Cook's distance {num(lo['cooks_d'], 2)}; weakest leave-one-out r {num(R['leave_one_out']['weakest'][1])}")

sp = pd.read_csv(D.out / "tableA2_splits.csv", index_col=0)
out.append("\n% Table A2 rows: class & convention & train/valid/test images & train/valid/test boxes")
for c, r in sp.iterrows():
    conv = "A" if r.group == "A_whole_leaf" else "B"
    out.append(f"{c.replace('_', ' ')} & {conv} & {r.train_imgs} & {r.valid_imgs} & {r.test_imgs} & "
               f"{r.train_boxes} & {r.valid_boxes} & {r.test_boxes} \\\\")

st = json.loads((P.RES / "eval_v13_not_v12valid_stats.json").read_text())["test_not_v12valid"]
K = ("ALL", "A", "B", "recall", "mAP50_95")
out.append("\n% Table A3 rows: system & class mean & A & B & recall & mAP@50-95")
pm = lambda m, s: f"{num(m)} ± {num(s)}"
out.append("baseline & " + " & ".join(pm(st["s@1024"][k]["baseline"], st["s@1024"][k]["baseline_sd"]) for k in K) + " \\\\")
for sysname, lab in (("s@1024", "YOLOv11s 1024"), ("s@1024+TTA", "YOLOv11s 1024 + TTA")):
    out.append(f"{lab} & " + " & ".join(pm(st[sysname][k]["system"], st[sysname][k]["system_sd"]) for k in K) + " \\\\")
for sysname, lab in (("s@1024", "YOLOv11s 1024"), ("s@1024+TTA", "YOLOv11s 1024 + TTA")):
    out.append(f"{lab}, Welch p & " + " & ".join(pval(st[sysname][k]["p"]) for k in K) + " \\\\")
for sysname, lab in (("s@1024", "YOLOv11s 1024"), ("s@1024+TTA", "YOLOv11s 1024 + TTA")):
    out.append(f"{lab}, Holm p & " + " & ".join(pval(st[sysname][k]["p_holm"]) for k in K) + " \\\\")

text = "\n".join(out) + "\n"
(D.out / "appendix_rows.tex").write_text(text, encoding="utf-8")
print(text)
