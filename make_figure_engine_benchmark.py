"""Figure 3 of the paper (Appendix C), from the results of engine_benchmark.py.

  (a) speed relative to vectorized Pyro on the same device (dashed, 1x), for a
      fixed workload of 10,240 queries (2,048 particles each) answered in
      calls of B states; top: GPU, bottom: CPU.  For NumPyro, since it is on
      JAX, the compilation time is counted: JAX compiles once per batch size,
      and the compile part of the first call (first call minus one
      steady-state call) is added to the time of the workload;
      (results_v11/engine_speed.json, results_v11/engine_speed_numpyro_cuda.json)
  (b) plan score vs particles per state on the 200 evaluation states, CPU
      engines, mean +- 1 s.d. over seeds, with the same planner on the true
      dynamics as reference (planning_cost.py)
      (results_v11/engine_accuracy.json, results_v11/planning_cost.json).
Also prints the numbers quoted in Appendix C.
Writes Images/engine_benchmark.pdf (+ .png).

Run:  python make_figure_engine_benchmark.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

SPEED = [p for p in ("results_v11/engine_speed.json", "results_v11/engine_speed_numpyro_cuda.json")
         if os.path.exists(p)]
ACC = json.load(open("results_v11/engine_accuracy.json"))
TRUE = json.load(open("results_v11/planning_cost.json"))["score"]["true"]
INK, MUTED = "#333333", "#888888"
# Colour = engine (blue = NBN, as in Figure 2).
ENGINE = {"nbn": ("#2a78d6", "NBN"), "pyro-vec": ("#eb6834", "Pyro"), "numpyro-vec": ("#1baf7a", "NumPyro")}
DEVICE_NAME = {"cpu": "CPU", "cuda": "GPU"}
LINE = dict(ls="-", marker="o", ms=3.5, lw=1.4)


def load_speed():
    """Time of the whole workload per engine and B, NumPyro with its compilation."""
    total, meta = {}, None
    for path in SPEED:
        X = json.load(open(path))
        meta = meta or X["meta"]
        n = X["meta"]["workload"]
        for k, per_b in X["total_s"].items():
            total[k] = {}
            for b, t in per_b.items():
                if k.startswith("numpyro"):
                    compile_s = X["compile_s"][b] - t / (n // int(b))
                    print(f"  NumPyro {k.split('/')[1]:4s} B={b:>4s}: compile {compile_s:.2f} s, "
                          f"workload {t:.2f} s -> {t + compile_s:.2f} s")
                    t += compile_s
                total[k][b] = t
    return total, meta


TOTAL, META = load_speed()
B = np.array(META["batches"])
P = np.array(ACC["meta"]["particles"])
S_USED = META["s_timing"]
GPU = next((k.split("/")[1] for k in TOTAL if k.split("/")[1] != "cpu"), None)


def band(ax, x, a, color):
    """Shaded +- 1 s.d. over seeds (axis 1); the caller draws the mean."""
    if a.shape[1] > 1:
        m, sd = a.mean(1), a.std(1)
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.12, lw=0)


def speed_panel(ax, dev, engines):
    base = f"pyro-vec/{dev}"
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.text(B[-1], 0.95, f"Pyro (vectorized), {DEVICE_NAME[dev]}", color=MUTED, fontsize=6, ha="right", va="top")
    lo, hi = 1.0, 1.0
    for k in engines:
        if k not in TOTAL:
            continue
        c = ENGINE[k.split("/")[0]][0]
        r = np.array([TOTAL[base][str(b)] / TOTAL[k][str(b)] for b in B])
        ax.plot(B, r, color=c, **LINE)
        lo, hi = min(lo, r.min()), max(hi, r.max())
        name = ENGINE[k.split("/")[0]][1] + (" (incl. compile)" if k.startswith("numpyro") else "")
        ax.annotate(f"{name} {r[-1]:.1f}$\\times$", (B[-1], r[-1]), textcoords="offset points",
                    xytext=(3, 5), ha="right", fontsize=6, color=c)
        print(f"  (a) {k:16s} / {base:13s}: " + "  ".join(
            f"B={b}: {v:.2f}x ({TOTAL[k][str(b)]:.2f} s)" for b, v in zip(B, r)))
    ax.set_xscale("log")
    ax.set_yscale("log", base=2)
    ax.set_xticks(B)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ylim = (lo / 1.9, hi * 1.45)                     # room for the labels above the lines and below 1x
    yt = [v for v in (0.25, 0.5, 1, 2, 4, 8) if ylim[0] <= v <= ylim[1]]
    if len(yt) < 3:                                   # narrow range: finer ticks
        yt = [v for v in (0.5, 0.75, 1, 1.25, 1.5, 2) if ylim[0] <= v <= ylim[1]]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:g}$\\times$" for v in yt])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(*ylim)
    ax.text(0.02, 0.05, DEVICE_NAME[dev], transform=ax.transAxes, fontsize=7.5, fontweight="bold",
            color=INK, va="bottom")


def main():
    fig = plt.figure(figsize=(6.6, 2.9))
    gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.12)
    ax_gpu = fig.add_subplot(gs[0, 0])
    ax_cpu = fig.add_subplot(gs[1, 0], sharex=ax_gpu)
    ax2 = fig.add_subplot(gs[:, 1])

    # (a) speed relative to vectorized Pyro, same device
    speed_panel(ax_gpu, GPU, [f"nbn/{GPU}", f"numpyro-vec/{GPU}"])
    speed_panel(ax_cpu, "cpu", ["nbn/cpu", "numpyro-vec/cpu"])
    plt.setp(ax_gpu.get_xticklabels(), visible=False)
    ax_cpu.set_xticklabels([str(b) for b in B])
    ax_cpu.set_xlabel("states planned in one call, $B$", fontsize=8)
    fig.text(0.045, 0.47, "speed relative to Pyro", rotation=90, fontsize=8, ha="center", va="center")
    ax_gpu.set_title(f"(a) Speed vs Pyro ({META['workload']:,} queries, {S_USED:,} particles)", fontsize=8.5)

    # (b) accuracy vs particles (CPU engines), with the true-dynamics planner
    t = np.array([TRUE[str(p)] for p in P])
    ax2.plot(P, t.mean(1), color=INK, ls="--", marker="o", ms=3, lw=1.4, label="true dynamics")
    band(ax2, P, t, INK)
    print(f"  (b) {'true dynamics':16s} " + "  ".join(f"S={p}: {m:.3f}±{s:.3f}"
                                                      for p, m, s in zip(P, t.mean(1), t.std(1))))
    for k in ("nbn/cpu", "pyro-vec/cpu", "numpyro-vec/cpu"):
        c, name = ENGINE[k.split("/")[0]]
        a = np.array([ACC["score"][k][str(p)] for p in P])                          # (P, seeds)
        ax2.plot(P, a.mean(1), color=c, label=name, **LINE)
        band(ax2, P, a, c)
        print(f"  (b) {k:16s} " + "  ".join(f"S={p}: {m:.3f}±{s:.3f}" for p, m, s in zip(P, a.mean(1), a.std(1))))
    ax2.axvline(S_USED, color=MUTED, lw=0.7, ls=":")
    ax2.text(S_USED * 0.92, 0.80, "used in\nSec. 4", color=MUTED, fontsize=6.5, ha="right", va="bottom")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(P)
    ax2.set_xticklabels([f"{p:,}" if p in (64, 256, 1024, 4096) else "" for p in P])
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("particles per state", fontsize=8)
    ax2.set_ylabel("success / optimal", fontsize=8)
    ax2.set_title("(b) Accuracy vs particles (CPU)", fontsize=8.5)
    ax2.legend(fontsize=6.5, frameon=False, loc="upper left")

    for ax in (ax_gpu, ax_cpu, ax2):
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    print(f"  accuracy seeds: {len(ACC['meta']['seeds'])}")
    fig.savefig("Images/engine_benchmark.pdf", bbox_inches="tight")
    fig.savefig("Images/engine_benchmark.png", dpi=200, bbox_inches="tight")
    print("wrote Images/engine_benchmark.pdf/.png")


if __name__ == "__main__":
    main()
