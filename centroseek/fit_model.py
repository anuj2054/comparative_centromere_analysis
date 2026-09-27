"""Refit the CentroSeek coefficients from the deposited human array table.

    python -m centroseek.fit_model [--label censat|cenpa] [--out model.json]

The shipped model.json is the --label censat fit, which is what produced every
CentroSeek number in the manuscript. --label cenpa refits on the CENP-A-defined
active array instead. The two labels pick the same top array on all 19 chromosomes
but differ at the array level, 27 CenSat-active against 19 CENP-A-active, so the
coefficients are not identical. Use this to check the shipped constants or to
explore the alternative; the paper's results correspond to the shipped ones.
"""
import argparse
import json
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TABLE = os.path.join(ROOT, "results", "tables", "array_level_19chrom.csv")
FEATS = ["bz2", "H11"]


def fit(label="censat", table=TABLE):
    d = pd.read_csv(table).dropna(subset=FEATS)
    if label == "censat":
        y = d.censat_active.astype(int).values
    elif label == "cenpa":
        y = (d.groupby("chrom").cenpa.transform("max") == d.cenpa).astype(int).values
    else:
        raise ValueError(f"unknown label {label!r}")
    X = d[FEATS].values
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(sc.transform(X), y)
    return dict(mean=[round(float(v), 6) for v in sc.mean_],
                scale=[round(float(v), 6) for v in sc.scale_],
                coef=[round(float(v), 6) for v in clf.coef_[0]],
                intercept=round(float(clf.intercept_[0]), 6),
                n_arrays=int(len(d)), n_positive=int(y.sum()),
                n_chromosomes=int(d.chrom.nunique()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", choices=["censat", "cenpa"], default="censat")
    ap.add_argument("--out", help="write a model.json here instead of reporting")
    a = ap.parse_args()

    f = fit(a.label)
    print(f"label={a.label}  n={f['n_arrays']} arrays, {f['n_positive']} positive, "
          f"{f['n_chromosomes']} chromosomes")
    print(f"  mean      {f['mean']}\n  scale     {f['scale']}")
    print(f"  coef      {f['coef']}\n  intercept {f['intercept']}")

    shipped = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "model.json")))
    same = (f["mean"] == shipped["standardisation"]["mean"]
            and f["scale"] == shipped["standardisation"]["scale"]
            and f["coef"] == shipped["coefficients"]
            and f["intercept"] == shipped["intercept"])
    print(f"  {'matches' if same else 'DIFFERS from'} the shipped model.json")

    if a.out:
        shipped["standardisation"] = {"mean": f["mean"], "scale": f["scale"]}
        shipped["coefficients"], shipped["intercept"] = f["coef"], f["intercept"]
        with open(a.out, "w") as fh:
            json.dump(shipped, fh, indent=2)
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
