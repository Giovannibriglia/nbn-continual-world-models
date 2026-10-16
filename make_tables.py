"""Tables 2-4 of the paper (Appendix A), printed as LaTeX, from the stored results.

  Table 2  model error (W1) and plan score per strategy: at the behavior shift
           (phase 2), right after the drift (phase 5), and summed over phases
           1-7 (results_v11/continual_v11.json; MDN, and the linear-Gaussian
           pipeline);
  Table 3  time to answer 10,240 planning queries in calls of B states, NBN,
           Pyro and NumPyro on GPU and CPU, NumPyro with its compilation
           (results_v11/engine_speed*.json);
  Table 4  plan score vs particles per state for the three engines (CPU) and
           with the true dynamics (results_v11/engine_accuracy.json,
           results_v11/planning_cost.json).
Bold marks the best value at the printed precision; a tie goes to the value with
the lower std over seeds (all tied values stay bold if their std is equal too): in Table 2 per
column among the five strategies (lower W1, excess and loss; higher plan score),
in Table 3 per row and device (shortest time, NumPyro with compilation), in
Table 4 per row among the three engines (highest score).  Reference rows and
columns (true dynamics, linear-Gaussian pipeline) are not ranked.
Table 1 (checks of the test) is printed by test_v11.py.

Run:  python make_tables.py
"""
import json
import statistics as st

W1_FLOOR, PLAN_CEIL = 0.070, 0.964    # true model / true-dynamics planner (test_v11.py)
NAMES = {"frozen": "frozen", "pooled": "pooled", "window": "sliding window",
         "raw-restart": "raw likelihood + restart", "pipeline": "NBN pipeline"}


def cell(text, bold):
    return f"$\\mathbf{{{text}}}$" if bold else f"${text}$"


def model_values(summary, strategy):
    s = summary[strategy]
    w, p = s["joint"]["mean"], s["plan"]["mean"]
    excess = sum(x - W1_FLOOR for x in w[1:])
    loss = sum(PLAN_CEIL - x for x in p[1:])
    return [f"{v:.2f}" for v in (w[2], w[5], excess, p[2], p[5], loss)]


def model_stds(summary, strategy):
    """Std over seeds of the shift/drift cells (the summed columns have none)."""
    s = summary[strategy]
    return [s["joint"]["sd"][2], s["joint"]["sd"][5], 0.0, s["plan"]["sd"][2], s["plan"]["sd"][5], 0.0]


def winners(values, stds, higher):
    """Indices of the best printed value; a tie goes to the lower std."""
    best = (max if higher else min)(float(v) for v in values)
    tied = [i for i, v in enumerate(values) if float(v) == best]
    low = min(stds[i] for i in tied)
    return {i for i in tied if stds[i] == low}


def table_model():
    d = json.load(open("results_v11/continual_v11.json"))["summary"]
    keys = list(NAMES)
    rows = {k: model_values(d["mdn"], k) for k in keys}
    stds = {k: model_stds(d["mdn"], k) for k in keys}
    higher = (False, False, False, True, True, False)            # plan scores: higher is better
    bold = [winners([rows[k][j] for k in keys], [stds[k][j] for k in keys], hi) for j, hi in enumerate(higher)]
    print("% Table 2: rows (W1 shift, drift, excess 1-7 | plan shift, drift, loss 1-7)")
    for i, (k, lab) in enumerate(NAMES.items()):
        cells = [cell(v, i in bold[j]) for j, v in enumerate(rows[k])]
        print(f"    {lab} & " + " & ".join(cells) + " \\\\")
    print(f"    true dynamics & ${W1_FLOOR:.2f}$ & ${W1_FLOOR:.2f}$ & -- & ${PLAN_CEIL:.2f}$ & ${PLAN_CEIL:.2f}$ & -- \\\\")
    lg = model_values(d["lg"], "pipeline")
    print("    NBN pipeline, linear-Gaussian & " + " & ".join(f"${v}$" for v in lg) + " \\\\")


def table_speed():
    main = json.load(open("results_v11/engine_speed.json"))
    gpu_np = json.load(open("results_v11/engine_speed_numpyro_cuda.json"))
    n = main["meta"]["workload"]

    def total(X, k, b):
        t = X["total_s"][k][str(b)]
        if not k.startswith("numpyro"):
            return t, None
        compile_s = X["compile_s"][str(b)] - t / (n // b)       # first call minus one steady call
        return t + compile_s, compile_s

    fmt = lambda x: f"{x:.2f}" if x < 10 else f"{x:.1f}"

    def device(ts, compile_s):
        """Cells for NBN, Pyro, NumPyro on one device; the shortest printed time in bold."""
        shown = [fmt(t) for t in ts]
        best = min(float(x) for x in shown)
        out = [cell(x, float(x) == best) for x in shown]
        out[2] += f" (${compile_s:.1f}$)"
        return out

    print("% Table 3: B & GPU: NBN, Pyro, NumPyro (compile) & CPU: NBN, Pyro, NumPyro (compile)")
    for b in main["meta"]["batches"]:
        gn, cn = total(gpu_np, "numpyro-vec/cuda", b), total(main, "numpyro-vec/cpu", b)
        g = device([total(main, "nbn/cuda", b)[0], total(main, "pyro-vec/cuda", b)[0], gn[0]], gn[1])
        c = device([total(main, "nbn/cpu", b)[0], total(main, "pyro-vec/cpu", b)[0], cn[0]], cn[1])
        print(f"    {b} & " + " & ".join(g + c) + " \\\\")


def table_accuracy():
    acc = json.load(open("results_v11/engine_accuracy.json"))
    true = json.load(open("results_v11/planning_cost.json"))["score"]["true"]
    print("% Table 4: particles & NBN & Pyro & NumPyro & true dynamics (mean +- std over seeds)")
    for p in acc["meta"]["particles"]:
        eng = [acc["score"][k][str(p)] for k in ("nbn/cpu", "pyro-vec/cpu", "numpyro-vec/cpu")]
        means = [f"{st.mean(v):.3f}" for v in eng]
        bold = winners(means, [st.pstdev(v) for v in eng], higher=True)
        cells = [(f"$\\mathbf{{{m}}} \\pm {st.pstdev(v):.3f}$" if i in bold else f"${m} \\pm {st.pstdev(v):.3f}$")
                 for i, (m, v) in enumerate(zip(means, eng))]
        t = true[str(p)]
        cells.append(f"${st.mean(t):.3f} \\pm {st.pstdev(t):.3f}$")
        print(f"    {p:,}".replace(",", "{,}") + " & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    table_model()
    table_speed()
    table_accuracy()
