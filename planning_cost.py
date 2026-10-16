"""Planning by inference with the true dynamics (paper, Appendix A and Figure 3b),
and the planner shared by engine_benchmark.py and test_v11.py.

The planner of Sec. 3 (4) and Sec. 4: a*(s) = mode of p(a | s, R = 1), computed
by nbn's batched likelihood weighting (uniform action prior; particles weighted
by P(R = 1 | S2')), with the mode read from a weighted Gaussian kernel
(bandwidth 0.05) on the 201-point action grid, in chunks of states so that
S = 8192 particles fit in memory (test_v11.py checks that it returns the same
actions as v11_common.plan_actions).

main() runs the planner with the TRUE mechanisms on the 200 evaluation states
for S in {64, ..., 8192} particles per state, seeds 0..n_seeds-1 (the seed
fixes the particles): the true-dynamics reference of Appendix A and Figure 3b.

Output: results_v11/planning_cost.json
Run:    python planning_cost.py [n_seeds=5]
"""
import json
import os
import sys

import numpy as np
import torch
from torch.distributions import Normal, Uniform

import nbn
import v11_common as C
from nbn.inference.likelihood_weighting import LikelihoodWeightingEngine
from test_v11 import oracle_model

PARTICLES = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
CHUNK = 32                         # states per chunk in the mode-finding step
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v11", "planning_cost.json")


class DeviceRoot(C.FixedRoot):
    """C.FixedRoot with its distribution built on the model's device (for MPS)."""

    def __init__(self, kind, device):
        super().__init__(kind)
        self.dev = torch.device(device)

    def forward(self, parents):
        one = torch.ones(1, device=self.dev)
        if self.kind == "uniform":
            return Uniform(-one, one, validate_args=False)
        return Normal(0.0 * one, one)


def to_device(model, device):
    m = C.clone(model).to(device)
    for n, kind in (("S1", "normal"), ("S2", "normal"), ("A", "uniform")):
        m.set_mechanism(n, DeviceRoot(kind, device))
    return m


def plan(model, s1, s2, n_particles, seed, bw=0.05):
    """Same planner as C.plan_actions, with S particles and a chunked mode step."""
    dev = model.device
    torch.manual_seed(seed)
    eng = LikelihoodWeightingEngine(n_samples=n_particles)
    ev = {"S1": torch.as_tensor(s1, dtype=torch.float32, device=dev).unsqueeze(-1),
          "S2": torch.as_tensor(s2, dtype=torch.float32, device=dev).unsqueeze(-1),
          "R": torch.ones(len(s1), 1, device=dev)}
    w, samp = eng.query(model, ["A"], ev)
    a = samp[..., 0]
    dt = torch.float64 if dev.type == "cpu" else torch.float32   # MPS has no float64
    grid = torch.as_tensor(C.A_GRID, dtype=dt, device=dev)
    out = []
    for i in range(0, len(s1), CHUNK):
        ai, wi = a[i:i + CHUNK].to(dt), w[i:i + CHUNK].to(dt)
        kern = torch.exp(-0.5 * ((grid[None, None, :] - ai[..., None]) / bw) ** 2)
        out.append((wi[..., None] * kern).sum(1).argmax(1))
    return C.A_GRID[torch.cat(out).cpu().numpy()]


def score(actions, s1, s2, drifted=False):
    got = C.true_success(s1, s2, actions, drifted)
    best = C.true_success(s1[:, None], s2[:, None], C.A_GRID[None, :], drifted).max(1)
    return float(got.mean() / best.mean())


def fit_phase0(seed):
    rng = np.random.default_rng(seed)
    data0 = C.simulate(rng, C.N_PER_PHASE, *C.phase_flags(0))   # = phase 0 of continual_v11
    torch.manual_seed(seed)
    m = C.new_model("mdn")
    for n in C.TRANSITION_NODES:
        C.fit_node(m, "mdn", n, data0)
    return m


def main():
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    s1, s2 = C.eval_states()
    torch.set_num_threads(1)
    res = {"meta": {"particles": PARTICLES, "seeds": list(seeds), "nbn": nbn.__version__,
                    "torch": torch.__version__},
           "score": {"true": {}}}
    oracle = oracle_model(False)
    for seed in seeds:
        for S in PARTICLES:
            sc = score(plan(oracle, s1, s2, S, seed), s1, s2)
            res["score"]["true"].setdefault(str(S), []).append(sc)
            print(f"seed {seed} S={S:5d}: true dynamics {sc:.3f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
