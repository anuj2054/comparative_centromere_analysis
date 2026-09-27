"""End-to-end analysis on real T2T-CHM13 sequence.

Produces (all from real data, no fabricated numbers):
  results/tables/class_features.csv     per-fragment features by class
  results/tables/class_summary.csv      mean +/- sd per class
  results/tables/h1_nested_models.csv   nested-model AUROC (the H1 test)
  results/tables/map_windows.csv        windowed metrics across the chr21 region
  results/figures/*.png                 figures
A run log with all headline numbers is written to results/run_log.txt.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from lib_metrics import clean, dinucleotide_shuffle  # noqa: E402
from lib_windows import WindowConfig, fragment_metrics, window_metrics  # noqa: E402
from lib_plotting import violin_box  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "results", "figures")
LOG_LINES: list[str] = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))


def load_fasta_seq(path):
    return "".join(l.strip() for l in open(path) if not l.startswith(">"))


def load_cds(path):
    recs, name, seq = [], None, []
    for line in open(path):
        if line.startswith(">"):
            if name:
                recs.append((name, "".join(seq)))
            name, seq = line[1:].strip(), []
        else:
            seq.append(line.strip())
    if name:
        recs.append((name, "".join(seq)))
    return recs


# --------------------------------------------------------------------------- #
# Part 1 — labeled class fragments + the H1 nested-model test
# --------------------------------------------------------------------------- #
def build_class_table(cfg):
    log("== Part 1: labeled class fragments ==")
    frames = []

    sat = load_fasta_seq(os.path.join(DATA, "satellite.fasta"))
    frames.append(fragment_metrics(sat, cfg, "satellite", max_frags=120))

    uniq = load_fasta_seq(os.path.join(DATA, "unique.fasta"))
    frames.append(fragment_metrics(uniq, cfg, "unique", max_frags=120))

    # Coding: concatenate CDS, then fragment to the SAME window size for a fair
    # length-matched comparison (k-mer counts depend on length).
    cds = load_cds(os.path.join(DATA, "coding_cds.fasta"))
    coding_seq = "".join(clean(s) for _, s in cds)
    frames.append(fragment_metrics(coding_seq, cfg, "coding", max_frags=120))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(TAB, "class_features.csv"), index=False)
    log(f"fragments per class:\n{df['label'].value_counts().to_string()}")
    return df


def summarize_classes(df, cfg):
    feats = ["gc", "H1", f"H{cfg.k_mid}", f"H{cfg.k_high}", "lz_norm",
             "gzip_bpb", "bz2_bpb", "R_uniform", "R_shuffle"]
    summ = df.groupby("label")[feats].agg(["mean", "std"])
    summ.to_csv(os.path.join(TAB, "class_summary.csv"))
    log("\n== class means ==")
    log(df.groupby("label")[feats].mean().round(4).to_string())
    return feats


def nested_auroc(df, cfg, pos="satellite", neg="unique", seed=0):
    """The H1 test: can complexity metrics separate two region classes that
    classic low-order Shannon entropy reports as near-identical? Compare nested
    logistic models by cross-validated AUROC on held-out fragments."""
    log(f"\n== H1 nested-model test: {pos} vs {neg} ==")
    sub = df[df["label"].isin([pos, neg])].copy()
    y = (sub["label"] == pos).astype(int).values

    models = {
        "GC only": ["gc"],
        "GC + H1": ["gc", "H1"],
        f"GC + H1 + H{cfg.k_mid}": ["gc", "H1", f"H{cfg.k_mid}"],
        f"GC + H1 + R_shuffle": ["gc", "H1", "R_shuffle"],
        "all complexity": ["gc", "H1", f"H{cfg.k_mid}", f"H{cfg.k_high}",
                           "lz_norm", "gzip_bpb", "bz2_bpb", "R_shuffle"],
    }
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, feats in models.items():
        X = sub[feats].values
        aucs = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000)
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        rows.append({"model": name, "features": ",".join(feats),
                     "cv_auroc_mean": np.mean(aucs), "cv_auroc_sd": np.std(aucs)})
        log(f"  {name:28s} AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # single-feature AUROC (which metric carries the signal?)
    log("  -- single-feature AUROC --")
    single = []
    for f in ["gc", "H1", f"H{cfg.k_mid}", f"H{cfg.k_high}", "lz_norm",
              "gzip_bpb", "bz2_bpb", "R_uniform", "R_shuffle"]:
        x = sub[f].values
        a = roc_auc_score(y, x)
        a = max(a, 1 - a)  # direction-agnostic separability
        single.append({"model": f"[single] {f}", "features": f,
                       "cv_auroc_mean": a, "cv_auroc_sd": 0.0})
        log(f"     {f:12s} AUROC = {a:.3f}")

    out = pd.DataFrame(rows + single)
    out.to_csv(os.path.join(TAB, "h1_nested_models.csv"), index=False)
    return out


def shuffle_control(cfg, seed=0):
    """Control: dinucleotide-shuffle the satellite fragments (destroys higher-
    order repetition, preserves composition). R_shuffle and H_high should move
    toward the unique-sequence regime, confirming the signal is repetition, not
    base composition."""
    log("\n== control: dinucleotide-shuffled satellite ==")
    rng = np.random.default_rng(seed)
    sat = clean(load_fasta_seq(os.path.join(DATA, "satellite.fasta")))
    n = min(40, len(sat) // cfg.window)
    rows = []
    for i in range(n):
        frag = sat[i * cfg.window : (i + 1) * cfg.window]
        shuf = dinucleotide_shuffle(frag, rng)
        from lib_entropy_rank import entropy_rank_ratio, kmer_entropy
        rows.append({
            "Hhigh_real": kmer_entropy(frag, cfg.k_high),
            "Hhigh_shuf": kmer_entropy(shuf, cfg.k_high),
            "Rshuffle_real": entropy_rank_ratio(frag, k=cfg.k_high, null="shuffle", n_mc=cfg.r_n_mc),
            "Rshuffle_shuf": entropy_rank_ratio(shuf, k=cfg.k_high, null="shuffle", n_mc=cfg.r_n_mc),
        })
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(TAB, "shuffle_control.csv"), index=False)
    log(c.mean().round(4).to_string())
    return c


# --------------------------------------------------------------------------- #
# Part 2 — descriptive genome map across the chr21 region
# --------------------------------------------------------------------------- #
def build_map(cfg_map):
    log("\n== Part 2: chr21 complexity map ==")
    seq = load_fasta_seq(os.path.join(DATA, "chr21_map.fasta"))
    # offset = region start from manifest
    import json
    man = json.load(open(os.path.join(DATA, "manifest.json")))
    offset = man["map"]["region"][0]
    df = window_metrics(seq, cfg_map, start_offset=offset)
    df.to_csv(os.path.join(TAB, "map_windows.csv"), index=False)
    log(f"map windows: {len(df)} of {cfg_map.window} bp (step {cfg_map.step})")
    return df


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_class_box(df, cfg):
    feats = ["H1", f"H{cfg.k_high}", "bz2_bpb", "R_shuffle"]
    titles = ["Shannon H₁ (per-base)", f"Shannon H_{cfg.k_high} (high-order)",
              "bzip2 bits/base", "Entropy-Rank Ratio R (shuffle null)"]
    order = ["coding", "unique", "satellite"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, f, t in zip(axes, feats, titles):
        violin_box(ax, [df[df.label == o][f].dropna().values for o in order], order, colors)
        ax.set_title(t, fontsize=10)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Complexity metrics by region class (real T2T-CHM13 chr21)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "class_boxplots.png"), dpi=130)
    plt.close(fig)


def fig_h1_vs_r(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"coding": "#1f77b4", "unique": "#2ca02c", "satellite": "#d62728"}
    for lab, g in df.groupby("label"):
        ax.scatter(g["H1"], g["R_shuffle"], s=14, alpha=0.6,
                   label=lab, color=colors.get(lab))
    ax.set_xlabel("Shannon H₁ (per-base entropy) — saturates")
    ax.set_ylabel("Entropy-Rank Ratio R (shuffle null) — resolves structure")
    ax.set_title("H₁ cannot separate classes; R does")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "h1_vs_R.png"), dpi=130)
    plt.close(fig)


def fig_map(df):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    x = (df["start"] + df["end"]) / 2 / 1e6
    sk = dict(s=5, alpha=0.5, edgecolors="none")
    axes[0].scatter(x, df["H1"], color="#555", **sk)
    axes[0].set_ylabel("H₁")
    axes[0].set_title("Information-topology map across T2T-CHM13 chr21 (real sequence)")
    axes[1].scatter(x, df["bz2_bpb"], color="#9467bd", **sk)
    axes[1].set_ylabel("bz2 bits/base")
    axes[2].scatter(x, df["R_shuffle"], color="#d62728", **sk)
    axes[2].set_ylabel("R (shuffle null)")
    axes[2].set_xlabel("chr21 position (Mb)")
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS05_chr21_map.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    cfg = WindowConfig(window=2000, step=2000, k_high=11, r_n_mc=60)
    cfg_map = WindowConfig(window=2000, step=4000, k_high=11, r_n_mc=40)

    df = build_class_table(cfg)
    summarize_classes(df, cfg)
    nested_auroc(df, cfg, pos="satellite", neg="unique")
    nested_auroc(df, cfg, pos="satellite", neg="coding", seed=1)
    shuffle_control(cfg)
    fig_class_box(df, cfg)
    fig_h1_vs_r(df)

    mp = build_map(cfg_map)
    fig_map(mp)

    with open(os.path.join(ROOT, "results", "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES) + "\n")
    log("\nAll outputs written to results/.")


if __name__ == "__main__":
    main()
