"""Figure (draft): NBN against Pyro and NumPyro on the planning query,
from results_v11/engine_benchmark_threadsdefault.json (engine_benchmark.py)
and, for the true-dynamics reference, results_v11/planning_cost.json.

  (a) speed relative to vectorized Pyro on the same device, vs number of
      states B answered by one batched call, at 2,048 particles, for a fixed
      workload of 10,240 queries (one call per B states); top: GPU engines
      over Pyro on GPU; bottom: CPU engines over Pyro on CPU; the dashed grey
      line at 1x is Pyro itself; NumPyro (JAX) includes its compilation time,
      paid once per batch size;
  (b) plan score vs particles per state on the 200 evaluation states (CPU
      engines), as in Figure 3a, with the same planner run on the true
      dynamics (Appendix C) as reference.
Mean +- 1 s.d. over seeds (the seed fixes the MDN fit and the particles).
Also prints the numbers.  Writes Images/engine_benchmark.pdf (+ .png).

Run:  python make_figure_engine_benchmark.py [speed json [extra speed json ...]]
      default: the fixed-workload server runs for (a), the 5-seed server sweep for (b).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# Panel (a): the fixed-workload server runs (CPU + CUDA, and NumPyro on CUDA);
# panel (b): the particle sweep of the 5-seed server run.
SPEED = ["results_v11/server/engine_benchmark_threadsdefault_workload10240.json",
         "results_v11/server/engine_benchmark_threadsdefault_jaxcuda_numpyro_workload10240.json"]
ACC = "results_v11/server/engine_benchmark_threadsdefault.json"
PATHS = sys.argv[1:] or [p for p in SPEED if os.path.exists(p)]


def with_compile(X):
    """NumPyro runs on JAX, which compiles once per input shape: add that cost.

    The first call of each batch size B is timed in X["compile_s"] (compile +
    one call); its compile part, first call minus one steady-state call, is
    added to the total time of the fixed workload, and the per-query time is
    recomputed from it.  NBN and Pyro do not compile, so they are unchanged.
    """
    if "workload_total_s" not in X:
        return X
    n = X["meta"]["workload"]
    for k, per_b in X["batch"].items():
        if not k.startswith("numpyro"):
            continue
        for b in per_b:
            total = X["workload_total_s"][k][b][0]
            compile_s = X["compile_s"][b][0] - total / (n // int(b))
            per_b[b] = [(total + compile_s) / n]
            print(f"  NumPyro {k.split('/')[1]:4s} B={b:>4s}: compile {compile_s:.2f} s, "
                  f"workload {total:.2f} s -> {total + compile_s:.2f} s")
    return X


R = with_compile(json.load(open(PATHS[0])))
for extra in PATHS[1:]:
    X = with_compile(json.load(open(extra)))
    for part in ("batch", "loop", "sweep"):
        R[part].update(X.get(part, {}))
R["sweep"] = json.load(open(ACC))["sweep"]
GPU = next((k.split("/")[1] for k in R["batch"] if k.split("/")[1] in ("cuda", "mps")), None)
GPU_NAME = R["meta"].get("gpu_name") or "GPU"
WORKLOAD = R["meta"].get("workload")
TRUE = json.load(open("results_v11/planning_cost.json"))["score"]["true"]
B = np.array(R["meta"]["batches"])
P = np.array(json.load(open(ACC))["meta"]["particles"])
S_USED = R["meta"]["s_timing"]
N_SEEDS = len(next(iter(R["batch"].values()))[str(B[0])])
INK, MUTED = "#333333", "#888888"
# Colour = engine (blue = NBN, as in Figures 2-3).
ENGINE = {"nbn": ("#2a78d6", "NBN"), "pyro-vec": ("#eb6834", "Pyro"), "numpyro-vec": ("#1baf7a", "NumPyro")}
DEVICE_NAME = {"cpu": "CPU", "mps": "GPU", "cuda": "GPU"}
LINE = dict(ls="-", marker="o", ms=3.5, lw=1.4)


def band(ax, x, a, color):
    """Shaded +- 1 s.d. over seeds (axis 1); the caller draws the mean."""
    if a.shape[1] > 1:
        m, sd = a.mean(1), a.std(1)
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.12, lw=0)


def speed_panel(ax, dev, engines):
    base = f"pyro-vec/{dev}"
    lo, hi = 1.0, 1.0
    ax.axhline(1.0, color=MUTED, lw=1.2, ls="--")
    ax.text(B[-1], 0.95, f"Pyro (vectorized), {DEVICE_NAME[dev]}", color=MUTED, fontsize=6, ha="right", va="top")
    for k in engines:
        if k not in R["batch"]:
            continue
        c = ENGINE[k.split("/")[0]][0]
        r = np.array([np.array(R["batch"][base][str(b)]) / np.array(R["batch"][k][str(b)]) for b in B])  # (B, seeds)
        ax.plot(B, r.mean(1), color=c, **LINE)
        band(ax, B, r, c)
        lo, hi = min(lo, (r.mean(1) - r.std(1)).min()), max(hi, (r.mean(1) + r.std(1)).max())
        name = ENGINE[k.split("/")[0]][1] + (" (incl. compile)" if k.startswith("numpyro") else "")
        ax.annotate(f"{name} {r.mean(1)[-1]:.1f}$\\times$", (B[-1], r.mean(1)[-1]), textcoords="offset points",
                    xytext=(3, 5), ha="right", fontsize=6, color=c)
        print(f"  (a) {k:16s} / {base:13s}: " + "  ".join(
            f"B={b}: {m:.2f}x±{s:.2f}" for b, m, s in zip(B, r.mean(1), r.std(1))))
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
    if f"numpyro-vec/{GPU}" not in R["batch"]:
        ax_gpu.text(0.98, 0.05, "NumPyro on GPU: not run", transform=ax_gpu.transAxes,
                    fontsize=5.5, color=MUTED, ha="right", va="bottom")
    speed_panel(ax_cpu, "cpu", ["nbn/cpu", "numpyro-vec/cpu"])
    plt.setp(ax_gpu.get_xticklabels(), visible=False)
    ax_cpu.set_xticklabels([str(b) for b in B])
    ax_cpu.set_xlabel("states planned in one call, $B$", fontsize=8)
    fig.text(0.045, 0.47, "speed relative to Pyro", rotation=90, fontsize=8, ha="center", va="center")
    ax_gpu.set_title(f"(a) Speed vs Pyro ({WORKLOAD:,} queries, {S_USED:,} particles)" if WORKLOAD
                     else f"(a) Speed vs Pyro ({S_USED:,} particles)", fontsize=8.5)

    # (b) accuracy vs particles (CPU engines), with the true-dynamics planner
    t = np.array([TRUE[str(p)] for p in P])
    ax2.plot(P, t.mean(1), color=INK, ls="--", marker="o", ms=3, lw=1.4, label="true dynamics")
    band(ax2, P, t, INK)
    print(f"  (b) {'true dynamics':16s} " + "  ".join(f"S={p}: {m:.3f}±{s:.3f}"
                                                      for p, m, s in zip(P, t.mean(1), t.std(1))))
    for k in ("nbn/cpu", "pyro-vec/cpu", "numpyro-vec/cpu"):
        c, name = ENGINE[k.split("/")[0]]
        a = np.array([R["sweep"][k][str(p)]["score"] for p in P])                    # (P, seeds)
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
    print(f"  seeds: {N_SEEDS}")
    fig.savefig("Images/engine_benchmark.pdf", bbox_inches="tight")
    fig.savefig("Images/engine_benchmark.png", dpi=200, bbox_inches="tight")
    print("wrote Images/engine_benchmark.pdf/.png")


if __name__ == "__main__":
    main()
