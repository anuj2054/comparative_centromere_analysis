"""Shared plotting helper: a violin with an inner box-and-whisker.

Gives the distribution shape (kernel density) plus the familiar box summary
(interquartile-range box, 1.5xIQR whiskers, median line, white-dot mean) in one
mark, used across the manuscript figures for any panel showing a distribution.
"""


def violin_box(ax, data, labels, colors, widths=0.78):
    pos = list(range(1, len(data) + 1))
    vp = ax.violinplot(data, positions=pos, showextrema=False, widths=widths)
    for body, c in zip(vp["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.45); body.set_edgecolor(c); body.set_linewidth(0.9)
    ax.boxplot(data, positions=pos, widths=0.11, showfliers=False, showmeans=True,
               patch_artist=True, medianprops=dict(color="white", lw=1.3),
               meanprops=dict(marker="o", mfc="white", mec="#222222", ms=3.5, mew=0.7),
               boxprops=dict(facecolor="#3a3a3a", edgecolor="#3a3a3a"),
               whiskerprops=dict(color="#3a3a3a", lw=1.0), capprops=dict(color="#3a3a3a", lw=1.0))
    ax.set_xticks(pos); ax.set_xticklabels(labels)
