"""Census of every ape chromosome carrying both active_hor and dhor.

The published cross-species result rests on 15 chromosomes (chimp 6, gorilla 4,
bonobo 5) that were hardcoded in the earlier per-species scripts. This
script re-derives the eligible set from each species' CenSat annotation, so we can
say how many chromosomes were available and whether the tested subset is
representative (Reviewer 1, comment 5; Reviewer 2, comment 7).

It also tests the longest-array rule directly from the annotation -- no sequence
needed -- which is exactly the claim Reviewer 1 disputes for bonobo.

Output: results/tables/ape_censat_census.csv ; results/ape_census_log.txt
"""
import os
import pandas as pd
import pyBigWig

ROOT = os.path.join(os.path.dirname(__file__), "..")
TAB = os.path.join(ROOT, "results", "tables")
GA = "https://genomeark.s3.amazonaws.com/species/"
SPECIES = {
    "chimp":   (GA + "Pan_troglodytes/mPanTro3/assembly_curated/repeats/mPanTro3_v2.0_CenSat_v1.2.bb",
                ["chr11_hap1_hsa9", "chr21_hap1_hsa20", "chr2_hap1_hsa3",
                 "chr3_hap1_hsa4", "chr5_hap1_hsa6", "chr7_hap1_hsa8"]),
    "gorilla": (GA + "Gorilla_gorilla/mGorGor1/assembly_curated/repeats/mGorGor1_v2.0_CenSat_v1.2.bb",
                ["chr2_mat_hsa3", "chr5_mat_hsa6", "chr7_mat_hsa8", "chr22_mat_hsa21"]),
    "bonobo":  (GA + "Pan_paniscus/mPanPan1/assembly_curated/repeats/mPanPan1_v2.0_CenSat_v1.2.bb",
                ["chr11_mat_hsa9", "chr15_mat_hsa14", "chr18_mat_hsa16",
                 "chr21_mat_hsa20", "chr23_mat_hsa22"]),
}
LOG = []


def log(m=""):
    print(m); LOG.append(str(m))


def arrays(bb, ch):
    out = []
    for s, e, r in (bb.entries(ch, 0, bb.chroms()[ch]) or []):
        n = r.split("\t")[0].lower()
        if n.startswith("active_hor"):
            out.append({"start": s, "end": e, "span": e - s, "active": True})
        elif n.startswith("dhor"):
            out.append({"start": s, "end": e, "span": e - s, "active": False})
    return out


def main():
    rows = []
    for sp, (cen, used) in SPECIES.items():
        bb = pyBigWig.open(cen)
        chroms = [c for c in bb.chroms() if "_hsa" in c or c.startswith("chr")]
        log(f"\n=== {sp}: {len(chroms)} sequences in CenSat ===")
        for ch in sorted(chroms):
            a = arrays(bb, ch)
            act = [x for x in a if x["active"]]
            ina = [x for x in a if not x["active"]]
            if not (act and ina):
                continue
            longest = max(a, key=lambda x: x["span"])
            rows.append({
                "species": sp, "chrom": ch, "in_published_set": ch in used,
                "n_active": len(act), "n_inactive": len(ina),
                "active_bp_max": max(x["span"] for x in act),
                "inactive_bp_max": max(x["span"] for x in ina),
                "longest_is_active": longest["active"],
            })
        bb.close()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "ape_censat_census.csv"), index=False)

    log("\n" + "=" * 78)
    log("Eligible chromosomes (both active_hor and dhor present)")
    log("=" * 78)
    for sp, g in df.groupby("species"):
        used = g.in_published_set.sum()
        log(f"  {sp:8s} eligible {len(g):2d}   used in paper {used:2d}   "
            f"never tested {len(g)-used:2d}")
    log(f"  {'TOTAL':8s} eligible {len(df):2d}   used in paper "
        f"{df.in_published_set.sum():2d}")

    log("\n" + "=" * 78)
    log("Longest-array rule, from CenSat spans alone (R1#5)")
    log("=" * 78)
    for sp, g in df.groupby("species"):
        pub, all_ = g[g.in_published_set], g
        log(f"  {sp:8s} published subset {pub.longest_is_active.sum()}/{len(pub)}"
            f"   all eligible {all_.longest_is_active.sum()}/{len(all_)}")
    log(f"  {'TOTAL':8s} published subset "
        f"{df[df.in_published_set].longest_is_active.sum()}/{df.in_published_set.sum()}"
        f"   all eligible {df.longest_is_active.sum()}/{len(df)}")

    bad = df[~df.longest_is_active]
    log(f"\nChromosomes where the active array is NOT the longest: {len(bad)}")
    for _, r in bad.iterrows():
        log(f"  {r.species:8s} {r.chrom:22s} active max {r.active_bp_max/1e3:8.0f} kb  "
            f"vs relict max {r.inactive_bp_max/1e3:8.0f} kb  "
            f"{'(in published set)' if r.in_published_set else '(never tested)'}")

    with open(os.path.join(ROOT, "results", "ape_census_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
