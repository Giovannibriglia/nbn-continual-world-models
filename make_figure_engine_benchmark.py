"""Figure 3 of the paper (Appendix C; commented out in the current version).

  (a) plan quality over the timeline (results_v11/continual_v11.json): success
      of the planned action relative to optimal, for the strategies that
      adapt; pooled refit leaves the axis after the drift (value annotated);
      dashed: the same planner with the true dynamics;
  (b) plan score vs particles per state on the 200 evaluation states, for the
      planner in the NBN, Pyro and NumPyro on CPU, mean +- 1 s.d. over seeds,
      with the same planner on the true dynamics as reference
      (results_v11/engine_accuracy.json, results_v11/planning_cost.json).
Also prints the numbers quoted in the text.
Writes Images/engine_benchmark.pdf (+ .png).

Run:  python make_figure_engine_benchmark.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.gridspec import GridSpec

import make_figures_v11_mdn as F2

ACC = json.load(open("results_v11/engine_accuracy.json"))
TRUE = json.load(open("results_v11/planning_cost.json"))["score"]["true"]
P = np.array(ACC["meta"]["particles"])
S_USED = 2048                                     # particles per state in Sec. 4
INK, MUTED = "#333333", "#888888"
# Colour = engine (blue = NBN, as in Figure 2).
ENGINE = {"nbn": ("#2a78d6", "NBN"), "pyro-vec": ("#eb6834", "Pyro"), "numpyro-vec": ("#1baf7a", "NumPyro")}
LINE = dict(ls="-", marker="o", ms=3, lw=1.4)


def band(ax, x, a, color):
    """Shaded +- 1 s.d. over seeds (axis 1); the caller draws the mean."""
    m, sd = a.mean(1), a.std(1)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.12, lw=0)


def plan_timeline(ax):
    F2.curve(ax, "plan")
    ax.set_ylim(0.80, 1.0)
    ax.axhline(F2.PLAN_CEIL, color="#444444", lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.text(F2.NPH - 0.65, F2.PLAN_CEIL + 0.004, "true dynamics", fontsize=6.8, color="#444444",
            ha="right", va="bottom")
    pp = F2.R["summary"][F2.FAM]["pooled"]["plan"]["mean"][F2.DRIFT]
    ax.text(F2.DRIFT - 0.8, 0.818, f"pooled: {pp:.0%}", fontsize=6.8, color=F2.STYLE["pooled"]["c"],
            ha="right", va="center")
    ax.set_xlabel("phase")
    ax.set_ylabel("success / optimal")
    ax.set_title("(a) Is the plan right?", fontsize=8, pad=3)


def accuracy(ax):
    t = np.array([TRUE[str(p)] for p in P])
    ax.plot(P, t.mean(1), color=INK, ls="--", marker="o", ms=3, lw=1.4, label="true dynamics")
    band(ax, P, t, INK)
    print(f"  (b) {'true dynamics':16s} " + "  ".join(f"S={p}: {m:.3f}±{s:.3f}"
                                                      for p, m, s in zip(P, t.mean(1), t.std(1))))
    for k in ("nbn/cpu", "pyro-vec/cpu", "numpyro-vec/cpu"):
        c, name = ENGINE[k.split("/")[0]]
        a = np.array([ACC["score"][k][str(p)] for p in P])                          # (P, seeds)
        ax.plot(P, a.mean(1), color=c, label=name, **LINE)
        band(ax, P, a, c)
        print(f"  (b) {k:16s} " + "  ".join(f"S={p}: {m:.3f}±{s:.3f}" for p, m, s in zip(P, a.mean(1), a.std(1))))
    ax.axvline(S_USED, color=MUTED, lw=0.7, ls=":")
    ax.text(S_USED * 0.92, 0.80, "used in\nSec. 4", color=MUTED, fontsize=6.5, ha="right", va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xticks(P)
    ax.set_xticklabels([f"{p:,}" if p in (64, 256, 1024, 4096) else "" for p in P])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("particles per state")
    ax.set_ylabel("success / optimal")
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_title("(b) Accuracy vs particles (CPU)", fontsize=8, pad=3)


def main():
    plt.rcParams.update({"font.size": 7.8, "axes.linewidth": 0.6,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig = plt.figure(figsize=(6.0, 2.25), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.3, 1], hspace=0.2, wspace=0.32)
    leg_a, leg_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for ax in (leg_a, leg_b):
        ax.axis("off")
    aa, ab = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    plan_timeline(aa)
    accuracy(ab)
    short = {"pooled": "pooled", "window": "window", "raw-restart": "raw lik. + restart",
             "pipeline": "NBN pipeline"}
    h, _ = aa.get_legend_handles_labels()
    leg_a.legend(h, [short[k] for k in F2.STYLE], loc="center", ncol=2, frameon=False, fontsize=6.9,
                 handlelength=2.0, columnspacing=0.8, labelspacing=0.2, handletextpad=0.4, borderaxespad=0.0)
    h, l = ab.get_legend_handles_labels()
    leg_b.legend(h, l, loc="center", ncol=2, frameon=False, fontsize=6.9, handlelength=2.0,
                 columnspacing=0.8, labelspacing=0.2, handletextpad=0.4, borderaxespad=0.0)
    for ax in (aa, ab):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig("Images/engine_benchmark.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("Images/engine_benchmark.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote Images/engine_benchmark.pdf/.png")


if __name__ == "__main__":
    main()
