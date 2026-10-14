"""Figure 3 of the paper (Appendix C), from results_v11/planning_cost.json.

  (a) plan score vs number of particles per state, for the planner with the
      true dynamics and with the MDN NBN fitted on phase 0 (mean +- 1 s.d.
      over seeds); dotted line: the 2,048 particles used in Sec. 4;
  (b) wall-clock time per planned state vs number of states B, one batched
      call (solid) vs B calls of one state (dashed), per device (colour and
      marker); median over repetitions, mean over seeds.
Also prints the numbers quoted in Appendix C.
Writes Images/planning_cost.pdf (+ .png).

Run:  python make_figure_planning_cost.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

R = json.load(open("results_v11/planning_cost.json"))
S = np.array(R["meta"]["particles"])
B = np.array(R["meta"]["batches"])
S_USED = R["meta"]["s_timing"]
MUTED = "#888888"
# Colours as in Figure 2 (blue = NBN); one marker per device as a second cue.
DEV_STYLE = {"cpu_1thread": dict(c="#2a78d6", m="o", lbl="CPU, 1 thread"),
             "mps": dict(c="#1baf7a", m="s", lbl="Apple GPU (MPS)")}
MULTI = dict(c="#eb6834", m="^")


def dev_style(label):
    if label in DEV_STYLE:
        return DEV_STYLE[label]
    n = label.split("_")[1].replace("threads", "")
    return dict(MULTI, lbl=f"CPU, {n} threads")


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.35), gridspec_kw=dict(wspace=0.32))

    # (a) accuracy vs particles
    for name, c, ls, lbl in (("true", "#333333", "--", "true dynamics"),
                             ("mdn", "#2a78d6", "-", "MDN NBN (phase 0)")):
        a = np.array([R["score"][name][str(s)] for s in S])          # (S, seeds)
        m, sd = a.mean(1), a.std(1)
        ax1.plot(S, m, color=c, ls=ls, marker="o", ms=3, lw=1.4, label=lbl)
        ax1.fill_between(S, m - sd, m + sd, color=c, alpha=0.12, lw=0)
        print(f"  score {name}: " + "  ".join(f"S={s}: {v:.3f}±{e:.3f}" for s, v, e in zip(S, m, sd)))
    ax1.axvline(S_USED, color=MUTED, lw=0.7, ls=":")
    ax1.text(S_USED * 0.92, 0.80, "used in\nSec. 4", color=MUTED, fontsize=6.5, ha="right", va="bottom")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(S)
    ax1.set_xticklabels([str(s) if s in (64, 256, 1024, 4096) else "" for s in S], fontsize=7)
    ax1.set_xlabel("particles per state", fontsize=8)
    ax1.set_ylabel("success / optimal", fontsize=8)
    ax1.set_title("(a) Accuracy vs particles", fontsize=8.5)
    ax1.legend(fontsize=6.5, frameon=False, loc="upper left")

    # (b) cost of batching
    for label, per_b in R["time"].items():
        st = dev_style(label)
        for mode, ls in (("batched", "-"), ("one_at_a_time", "--")):
            ms = np.array([1e3 * np.mean(per_b[str(b)][mode]) for b in B])
            ax2.plot(B, ms, color=st["c"], ls=ls, marker=st["m"], ms=3.5, lw=1.4)
            print(f"  time {label:14s} {mode:13s}: " + "  ".join(f"B={b}: {v:.3f} ms" for b, v in zip(B, ms)))
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(B)
    ax2.set_xticklabels([str(b) for b in B], fontsize=7)
    yt = [0.4, 0.6, 1, 2, 3, 5]
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_yticks(yt)
    ax2.set_yticklabels([f"{v:g}" for v in yt])
    ax2.set_ylim(0.33, 6)
    ax2.set_xlabel("states planned together, $B$", fontsize=8)
    ax2.set_ylabel("ms per state", fontsize=8)
    ax2.set_title(f"(b) Cost per state ({S_USED} particles)", fontsize=8.5)
    handles = [plt.Line2D([], [], color=dev_style(l)["c"], marker=dev_style(l)["m"], ms=3.5, lw=1.4,
                          label=dev_style(l)["lbl"]) for l in R["time"]]
    handles += [plt.Line2D([], [], color="#333333", ls="-", lw=1.2, label="one batched call"),
                plt.Line2D([], [], color="#333333", ls="--", lw=1.2, label="one state per call")]
    ax2.legend(handles=handles, fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))

    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7)
    fig.savefig("Images/planning_cost.pdf", bbox_inches="tight")
    fig.savefig("Images/planning_cost.png", dpi=200, bbox_inches="tight")
    print("wrote Images/planning_cost.pdf/.png")


if __name__ == "__main__":
    main()
