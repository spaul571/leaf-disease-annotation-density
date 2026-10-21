"""Statistics requested by Reviewer 1 comments 1 and 2. No training; reads finished runs only.

R1-1 asks whether the density-accuracy correlation survives controls for dataset source, object
size and class frequency. R1-2 asks for per-seed results, confidence intervals, and a statement of
what the three-seed design supports.

Everything here is computed from files already on disk, through paper_data.py:
  <dataset>_protocol_analysis.csv  - per-class annotation statistics
  the baseline per-class CSVs      - per-class validation accuracy, three seeds
  the dataset's eval JSON          - per-seed held-out test results with per-class AP, three arms

Source dataset and crop are the same factor in this benchmark: each of the six public sources
contributed exactly one crop, so "crop" below IS the source control. That identification limit is
reported, not papered over.

Group means are recomputed with classes.py, and the JSON stores unrounded values: rounding happens
once, when a table is rendered.

    qpy build_reviewer_stats.py                 # v13 -> results/v13_reviewer_stats.json
    qpy build_reviewer_stats.py --dataset v12   # v12 -> results/v12_reviewer_stats.json
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import paper_data as P
from paper_data import welch_ci  # noqa: F401 - the eval notebooks import welch_ci from here


# ---------------------------------------------------------------- helpers

def partial_corr(x, y, covars):
    """Pearson correlation of x and y after linearly removing every column of covars.

    Returns (r, p, df). df = n - 2 - k, so the test is not credited with information the
    covariates already consumed.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = sm.add_constant(np.column_stack(covars))
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = len(x) - 2 - (Z.shape[1] - 1)
    t = r * np.sqrt(df / max(1 - r ** 2, 1e-12))
    return r, float(2 * stats.t.sf(abs(t), df)), df


def vif(X):
    """Variance inflation factor per column of a design matrix (no constant column)."""
    out = {}
    for i, name in enumerate(X.columns):
        others = sm.add_constant(X.drop(columns=[name]).values)
        r2 = sm.OLS(X[name].values, others).fit().rsquared
        out[name] = float(1 / max(1 - r2, 1e-12))
    return out


# ---------------------------------------------------------------- R1-1

def r1_1(D):
    prot = pd.read_csv(D.protocol)
    files = D.val_arms["baseline"]
    assert all(f.exists() for f in files), f"missing baseline seed CSVs: {files}"
    acc = pd.concat([P.perclass(f)[1] for f in files])
    acc = acc.groupby("class")["mAP50"].mean().rename("mAP50")

    d = prot.merge(acc, left_on="class", right_index=True, how="inner")
    assert len(d) == 22, f"expected 22 classes after merge, got {len(d)}"
    d["crop"] = d["class"].str.split("_").str[0]
    d["log_n_boxes"] = np.log10(d["n_boxes"])

    y = d["mAP50"].values
    dens = d["boxes_per_img"].values
    area = d["median_area"].values
    freq = d["log_n_boxes"].values

    uni = {}
    for name, v in (("boxes_per_img", dens), ("median_area", area), ("log_n_boxes", freq),
                    ("mixed_frac", d["mixed_frac"].values)):
        pr, pp = stats.pearsonr(v, y)
        sr, sp = stats.spearmanr(v, y)
        uni[name] = {"pearson_r": round(float(pr), 4), "pearson_p": float(pp),
                     "spearman_r": round(float(sr), 4), "spearman_p": float(sp)}

    # density vs accuracy, holding each competing explanation fixed in turn
    par = {}
    for label, cov in (("control_median_area", [area]),
                       ("control_log_n_boxes", [freq]),
                       ("control_both", [area, freq])):
        r, p, df = partial_corr(dens, y, cov)
        par[label] = {"r": round(r, 4), "p": float(p), "df": int(df)}

    X = pd.DataFrame({"boxes_per_img": dens, "median_area": area, "log_n_boxes": freq})
    m = sm.OLS(y, sm.add_constant(X)).fit()
    multi = {
        "n": int(len(y)), "r2": round(float(m.rsquared), 4),
        "adj_r2": round(float(m.rsquared_adj), 4), "f_p": float(m.f_pvalue),
        "coef": {k: {"beta": round(float(m.params[k]), 5), "se": round(float(m.bse[k]), 5),
                     "t": round(float(m.tvalues[k]), 3), "p": float(m.pvalues[k])}
                 for k in ["const", "boxes_per_img", "median_area", "log_n_boxes"]},
        "vif": {k: round(v, 2) for k, v in vif(X).items()},
    }

    # crop == source dataset in this benchmark, so these are the source controls
    crops = pd.get_dummies(d["crop"], drop_first=True).astype(float)
    Xfe = pd.concat([pd.Series(dens, name="boxes_per_img", index=d.index), crops], axis=1)
    mfe = sm.OLS(y, sm.add_constant(Xfe)).fit()
    fe = {"beta": round(float(mfe.params["boxes_per_img"]), 5),
          "se": round(float(mfe.bse["boxes_per_img"]), 5),
          "p": float(mfe.pvalues["boxes_per_img"]), "r2": round(float(mfe.rsquared), 4),
          "df_resid": int(mfe.df_resid)}

    try:
        mm = sm.MixedLM(y, sm.add_constant(pd.DataFrame({"boxes_per_img": dens})),
                        groups=d["crop"].values).fit(reml=True)
        mixed = {"beta": round(float(mm.params["boxes_per_img"]), 5),
                 "se": round(float(mm.bse["boxes_per_img"]), 5),
                 "p": float(mm.pvalues["boxes_per_img"]),
                 "group_var": round(float(mm.cov_re.iloc[0, 0]), 6),
                 "converged": bool(mm.converged), "n_groups": int(d["crop"].nunique())}
    except Exception as e:                                    # noqa: BLE001
        mixed = {"error": f"{type(e).__name__}: {e}"}

    per_crop = {}
    for c, g in d.groupby("crop"):
        if len(g) >= 3:
            sr, sp = stats.spearmanr(g["boxes_per_img"], g["mAP50"])
            per_crop[c] = {"n": int(len(g)), "spearman_r": round(float(sr), 3), "p": float(sp)}
        else:
            per_crop[c] = {"n": int(len(g)), "spearman_r": None, "p": None}

    return {"univariate": uni, "partial": par, "multiple_regression": multi,
            "crop_fixed_effects": fe, "mixed_model": mixed, "per_crop_spearman": per_crop,
            "per_class": d[["class", "group", "crop", "boxes_per_img", "median_area",
                            "n_boxes", "mAP50"]].round(4).to_dict(orient="records")}


# ---------------------------------------------------------------- R1-2

def r1_2(D):
    V = P.val_per_seed(D)[list(P.METRICS)]
    T = P.test_per_seed(D)
    W = P.welch_vs_baseline(T)   # Welch CI and p per metric; Holm across the five, unrounded
    return {"val_per_seed": V.to_dict(orient="index"),
            "val_summary": {c: {"mean": float(V[c].mean()), "sd": float(V[c].std(ddof=1))}
                            for c in V.columns},
            "test_per_seed": {k: v.reset_index().to_dict(orient="records") for k, v in T.items()},
            "test_ci": {s: {m: {k: v for k, v in r.items() if k != "p_holm"} for m, r in res.items()}
                        for s, res in W.items()},
            "holm_adjusted_p": {s: {m: r["p_holm"] for m, r in res.items()} for s, res in W.items()},
            "note": ("n=3 per arm; a single-arm 95% interval uses t(2)=4.303, so intervals are "
                     "wide by construction. The CI is on the difference of means (Welch).")}


# ---------------------------------------------------------------- R1-3

def r1_3():
    """Worker-count control, run on v12: three seeds at 8 workers against three at 2.

    Same rule as build_worker_nb.py (bar = 1.96 * sd_pooled * sqrt(2/3), Welch p beside it), but
    with A and B regrouped by classes.py. The run times stay in results/worker_control.json.
    """
    w8 = pd.DataFrame([P.summarise(f)[0] for f in P.dataset("v12").val_arms["baseline"]])
    w2 = pd.DataFrame([P.summarise(P.RES / f"perclass_local_baseline_100ep_s{s}_W2.csv")[0]
                       for s in (0, 1, 2)])
    rows = []
    for m in P.METRICS:
        a, b = w8[m].values, w2[m].values
        sd_pool = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        bar = 1.96 * sd_pool * np.sqrt(2 / 3)
        delta = b.mean() - a.mean()
        rows.append({"metric": m, "workers=8": a.mean(), "sd8": a.std(ddof=1),
                     "workers=2": b.mean(), "sd2": b.std(ddof=1), "delta": delta, "bar": bar,
                     "ratio": abs(delta) / bar,
                     "welch_p": float(stats.ttest_ind(b, a, equal_var=False)[1])})
    return {"dataset": "v12", "rule": "1.96 * sd_pooled * sqrt(2/3)", "comparison": rows}


if __name__ == "__main__":
    D = P.dataset(P.add_dataset_arg(argparse.ArgumentParser()).parse_args().dataset)
    OUT = P.RES / f"{D.tag}reviewer_stats.json"
    out = {"dataset": D.name, "R1_1_confounds": r1_1(D), "R1_2_seeds_and_intervals": r1_2(D),
           "R1_3_worker_control_v12": r1_3()}
    OUT.write_text(json.dumps(out, indent=2))
    print("SAVED", OUT)

    a = out["R1_1_confounds"]
    print("\n--- R1-1 univariate ---")
    for k, v in a["univariate"].items():
        print(f"  {k:16s} Pearson {v['pearson_r']:+.4f} (p={v['pearson_p']:.2e})  "
              f"Spearman {v['spearman_r']:+.4f} (p={v['spearman_p']:.2e})")
    print("--- R1-1 partial (density vs mAP50) ---")
    for k, v in a["partial"].items():
        print(f"  {k:22s} r={v['r']:+.4f}  p={v['p']:.2e}  df={v['df']}")
    print("--- R1-1 multiple regression ---")
    mr = a["multiple_regression"]
    print(f"  R2={mr['r2']} adj={mr['adj_r2']} F p={mr['f_p']:.2e} n={mr['n']}")
    for k, v in mr["coef"].items():
        print(f"    {k:16s} beta={v['beta']:+.5f} se={v['se']:.5f} t={v['t']:+.2f} p={v['p']:.4f}")
    print(f"  VIF: {mr['vif']}")
    print("--- R1-1 crop (= source) controls ---")
    print(f"  fixed effects: beta={a['crop_fixed_effects']['beta']:+.5f} "
          f"se={a['crop_fixed_effects']['se']:.5f} p={a['crop_fixed_effects']['p']:.2e} "
          f"R2={a['crop_fixed_effects']['r2']}")
    print(f"  mixed model  : {a['mixed_model']}")
    for c, v in a["per_crop_spearman"].items():
        print(f"    {c:12s} n={v['n']} spearman={v['spearman_r']} p={v['p']}")

    b = out["R1_2_seeds_and_intervals"]
    print("\n--- R1-2 validation per seed ---")
    print(pd.DataFrame(b["val_per_seed"]).T.astype(float).round(4))
    print("--- R1-2 test per seed ---")
    for k, v in b["test_per_seed"].items():
        print(f"  {k}:")
        print(pd.DataFrame(v).round(4).to_string(index=False))
    print("--- R1-2 Welch 95% CI on the difference vs baseline ---")
    for label, per_metric in b["test_ci"].items():
        print(f"  {label}:")
        for m, v in per_metric.items():
            print(f"    {m:10s} delta={v['delta']:+.4f} CI95=[{v['ci95'][0]:+.4f},"
                  f"{v['ci95'][1]:+.4f}] p={v['p']:.4f} df={v['welch_df']:.2f} "
                  f"{'excludes 0' if v['excludes_zero'] else 'includes 0'}")
    print("--- R1-2 Holm-adjusted p across the five metrics ---")
    for label, adj in b["holm_adjusted_p"].items():
        print(f"  {label}: " + ", ".join(f"{m} {q:.4f}" for m, q in adj.items()))
    print("--- R1-3 worker control (v12, 8 vs 2 workers) ---")
    for r in out["R1_3_worker_control_v12"]["comparison"]:
        print(f"  {r['metric']:10s} {r['workers=8']:.4f} -> {r['workers=2']:.4f} "
              f"delta {r['delta']:+.4f} bar {r['bar']:.4f} ratio {r['ratio']:.2f} p {r['welch_p']:.3f}")
