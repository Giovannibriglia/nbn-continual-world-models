"""Figure 2 of the paper, from results_v11/continual_v11.json.

One row of three panels on the same phase axis:
  (a) flags over time: for each detector and mechanism, how many of the runs
      raised a flag at each phase; the one real change is boxed;
  (b) model error (W1) and (c) plan quality, zoomed on the strategies that
      adapt; pooled refit leaves the axis after the drift (value annotated);
      dashed lines: true model (W1 floor) and planner with the true dynamics.
Also prints the excess over the whole timeline (phases 1-7) used in the text.
Each legend sits directly above the panel(s) it describes.
Writes Images/continual_v11_mdn.pdf (+ .png).

Run:  python make_figures_v11_mdn.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

FAM = "mdn"
R = json.load(open("results_v11/continual_v11.json"))
SHIFT, DRIFT = R["meta"]["shift_at"], R["meta"]["drift_at"]
NPH = R["meta"]["n_phases"]
NSEED = len(R["meta"]["seeds"])
STYLE = {
    "pooled":      dict(c="#1baf7a", ls="-",  m="s", lbl="pooled refit"),
    "window":      dict(c="#6250d6", ls="--", m="v", lbl="sliding window"),
    "raw-restart": dict(c="#eb6834", ls="-",  m="^", lbl="raw likelihood + restart"),
    "pipeline":    dict(c="#2a78d6", ls="-",  m="D", lbl="NBN pipeline"),
}
GOOD, BAD, MUTED = "#1baf7a", "#e34948", "#888888"
# Reference levels measured by test_v11.py with the TRUE mechanisms plugged in:
W1_FLOOR = 0.070      # W1 of the true model to itself (sampling noise only)
PLAN_CEIL = 0.964     # plan score of the same planner with the true dynamics


def excess():
    """Sum over phases 1..7 of the gap to the true model / true-dynamics planner."""
    summ = R["summary"][FAM]
    out = {}
    for s in STYLE:
        w = np.array(summ[s]["joint"]["mean"][1:]); p = np.array(summ[s]["plan"]["mean"][1:])
        out[s] = (float((w - W1_FLOOR).sum()), float((PLAN_CEIL - p).sum()))
    base = out["pipeline"]
    for s, (w, p) in out.items():
        print(f"  {s:12s} excess W1 {w:.3f} ({w / base[0] - 1:+.0%})   "
              f"excess plan loss {p:.3f} ({p / base[1] - 1:+.0%})")
    return out


def events(ax):
    for t in (SHIFT, DRIFT):
        ax.axvline(t - 0.5, color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_xlim(-0.4, NPH - 0.6)
    ax.set_xticks(range(NPH))


def curve(ax, key):
    summ = R["summary"][FAM]
    x = np.arange(NPH)
    for s, st in STYLE.items():
        m = np.array(summ[s][key]["mean"]); sd = np.array(summ[s][key]["sd"])
        ax.plot(x, m, color=st["c"], ls=st["ls"], marker=st["m"], ms=3, lw=1.4,
                label=st["lbl"], zorder=4 if s == "pipeline" else 3)
        ax.fill_between(x, m - sd, m + sd, color=st["c"], alpha=0.12, lw=0)
    events(ax)
    ax.grid(alpha=0.25, lw=0.4)


def flags(ax):
    """Rows: detector x mechanism; columns: phases 1..7; cell = runs flagged."""
    fc = R["flag_counts"][FAM]
    rows = [("pipeline", "S1n"), ("pipeline", "S2n"), ("raw-restart", "S1n"), ("raw-restart", "S2n")]
    lab = {"S1n": "$S_1'$", "S2n": "$S_2'$"}
    for r, (det, node) in enumerate(rows):
        counts = fc[det][node]                              # phases 1..NPH-1
        for t in range(1, NPH):
            k = counts[t - 1]
            real = node == "S2n" and t == DRIFT
            if real:
                ax.add_patch(Rectangle((t - 0.42, r - 0.42), 0.84, 0.84, fill=False,
                                       ec="#222222", lw=1.0, zorder=2))
            if k == 0:
                ax.plot(t, r, marker="o", ms=2.2, color="#bbbbbb", zorder=3)
            else:
                col = GOOD if real else BAD
                ax.plot(t, r, marker="o", ms=11.5, color=col, alpha=0.9, zorder=3, mec="white", mew=0.8)
                ax.text(t, r, str(k), ha="center", va="center", fontsize=7.6, color="white",
                        fontweight="bold", zorder=4)
    ax.set_ylim(len(rows) - 0.5, -1.25)
    for t, txt in ((SHIFT, "behavior\nshift"), (DRIFT, "drift")):
        ax.text(t - 0.4, -0.65, txt, fontsize=7.0, color="#555555", va="bottom", ha="left",
                linespacing=0.95)
    ax.set_yticks(range(len(rows)))
    short = {"pipeline": "test", "raw-restart": "raw lik."}
    ax.set_yticklabels([f"{short[d]} {lab[n]}" for d, n in rows])
    ax.axhline(1.5, color="#cccccc", lw=0.6)
    events(ax)
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("phase")
    ax.set_title("(a) Which mechanism changed?", fontsize=8, pad=3)


def main():
    plt.rcParams.update({"font.size": 7.8, "axes.linewidth": 0.6,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig = plt.figure(figsize=(6.9, 2.25), dpi=300)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.3, 1], width_ratios=[1.05, 1, 1],
                  hspace=0.2, wspace=0.42)
    leg_a = fig.add_subplot(gs[0, 0]); leg_a.axis("off")
    leg_bc = fig.add_subplot(gs[0, 1:]); leg_bc.axis("off")
    aa, ab, ac = (fig.add_subplot(gs[1, j]) for j in range(3))
    excess()
    flags(aa)
    curve(ab, "joint"); curve(ac, "plan")
    ab.set_ylabel("model error ($W_1$)"); ac.set_ylabel("success / optimal")
    ab.set_ylim(0.04, 0.26); ac.set_ylim(0.80, 1.0)
    ab.axhline(W1_FLOOR, color="#444444", lw=0.8, ls=(0, (4, 2)), zorder=2)
    ab.text(NPH - 0.65, W1_FLOOR - 0.005, "true model", fontsize=6.8, color="#444444",
            ha="right", va="top")
    ac.axhline(PLAN_CEIL, color="#444444", lw=0.8, ls=(0, (4, 2)), zorder=2)
    ac.text(NPH - 0.65, PLAN_CEIL + 0.004, "true dynamics", fontsize=6.8, color="#444444",
            ha="right", va="bottom")
    pw = R["summary"][FAM]["pooled"]["joint"]["mean"][DRIFT]
    pp = R["summary"][FAM]["pooled"]["plan"]["mean"][DRIFT]
    c = STYLE["pooled"]["c"]
    ab.text(DRIFT - 0.8, 0.238, f"pooled: {pw:.2f}", fontsize=6.8, color=c,
            ha="right", va="center")
    ac.text(DRIFT - 0.8, 0.818, f"pooled: {pp:.0%}", fontsize=6.8, color=c,
            ha="right", va="center")
    for ax in (ab, ac):
        ax.set_xlabel("phase")
    ab.set_title("(b) Is the model right?", fontsize=8, pad=3)
    ac.set_title("(c) Is the plan right?", fontsize=8, pad=3)
    h, l = ab.get_legend_handles_labels()
    leg_bc.legend(h, l, loc="center", ncol=2, frameon=False, fontsize=7.6, handlelength=2.4,
                  columnspacing=1.2, labelspacing=0.2, borderaxespad=0.0)
    proxies = [Line2D([], [], marker="o", ls="", ms=7, color=GOOD, label="correct flag"),
               Line2D([], [], marker="o", ls="", ms=7, color=BAD, label="false alarm"),
               Line2D([], [], marker="s", ls="", ms=7, mfc="none", mec="#222222", label="real change"),
               Line2D([], [], marker="o", ls="", ms=2.2, color="#bbbbbb", label="no flag")]
    leg_a.legend(handles=proxies, loc="center", ncol=2, frameon=False, fontsize=7.6,
                 handletextpad=0.3, columnspacing=1.0, labelspacing=0.2, borderaxespad=0.0)
    fig.savefig("Images/continual_v11_mdn.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("Images/continual_v11_mdn.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("written Images/continual_v11_mdn.pdf")


if __name__ == "__main__":
    main()
