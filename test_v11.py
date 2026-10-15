"""Checks for the experiments of the paper (v11_common).

1. Test calibration: false-flag rate of the matched-pair test with no change and
   under a behavior shift; power under the mechanism drift.
2. Planner: with the TRUE mechanisms plugged into the NBN, the LW goal query
   must give near-optimal actions (checks LW + mode extraction, not learning).
3. Raw score: with the TRUE mechanisms, a behavior shift alone lowers the
   mean log-likelihood of S2' (Eq. 1: the entropy term moves with the parents).
4. W1 metric: the true model has (near) zero error.

Run:  python test_v11.py
"""
import numpy as np
import torch

import v11_common as C
from nbn.mechanisms.base import Mechanism
from nbn.utils.batching import pack_parents


class TrueMech(Mechanism):
    """The true mechanism of S1' or S2' (parents in the DAG's order)."""

    def __init__(self, node, parent_names, drifted=False):
        super().__init__()
        self.node, self.pn, self.drifted = node, list(parent_names), drifted

    def _split(self, pa):
        pa = pa.double().numpy().reshape(-1, len(self.pn))
        d = {n: pa[:, i] for i, n in enumerate(self.pn)}
        z = np.zeros(len(pa))
        return d.get("S1", z), d.get("S2", z), d["A"]

    def forward(self, parents):
        raise NotImplementedError

    def log_prob(self, x, parents):
        s1, s2, a = self._split(parents)
        lp = C.true_logpdf(self.node, x.double().numpy().reshape(-1), s1, s2, a, self.drifted)
        return torch.as_tensor(lp, dtype=torch.float32)

    def sample(self, parents, n=1):
        s1, s2, a = (np.repeat(v, n) for v in self._split(parents))
        rng = np.random.default_rng(0)
        x1, x2 = C.sample_children(rng, s1, s2, a, self.drifted)
        x = x1 if self.node == "S1n" else x2
        return torch.as_tensor(x, dtype=torch.float32).reshape(-1, n, 1)

    def fit_local(self, x, parents, **kw):
        return {}


def oracle_model(drifted=False):
    m = C.new_model("lg")
    for n in C.TRANSITION_NODES:
        m.mechanisms[n] = TrueMech(n, m.dag.parents(n), drifted)
    return m


def test_calibration_and_power(reps=200):
    rng = np.random.default_rng(2026)
    res = {}
    for name, sh, dr in (("none", False, False), ("shift", True, False), ("drift", False, True)):
        p = np.array([[C.test_node(n, C.simulate(rng, 2000, False, False),
                                   C.simulate(rng, 2000, sh, dr), rng)[0]
                       for n in C.TRANSITION_NODES] for _ in range(reps if name != "drift" else 50)])
        res[name] = (p < 0.05).mean(0), (p < 0.005).mean(0)
        print(f"  {name:5s} flag rate S1', S2' at 0.05: {res[name][0]}  at 0.005: {res[name][1]}")
    assert res["none"][0].max() <= 0.09 and res["shift"][0].max() <= 0.10
    assert res["drift"][1][1] == 1.0 and res["drift"][0][0] <= 0.12


def test_planner_oracle():
    for dr in (False, True):
        score = C.plan_score(oracle_model(dr), dr)
        print(f"  oracle plan score (drifted={dr}): {score:.3f}")
        assert score > 0.95


def test_raw_score_moves_under_behavior_shift():
    rng = np.random.default_rng(1)
    m = oracle_model(False)
    ll0 = C.node_logprob(m, "S2n", C.simulate(rng, 4000, False, False)).mean()
    ll1 = C.node_logprob(m, "S2n", C.simulate(rng, 4000, True, False)).mean()
    print(f"  true-model mean log-lik of S2': original behavior {ll0:.3f}, shifted {ll1:.3f}")
    assert ll1 < ll0 - 0.1


def test_w1_metric():
    w = C.w1_error(oracle_model(False), False)["joint"]
    print(f"  W1 of the true model to itself (sampling noise only): {w:.3f}")
    assert w < 0.08


def test_planning_cost_planner():
    """planning_cost.plan = C.plan_actions; one-at-a-time and MPS plans score alike."""
    import planning_cost as P
    s1, s2 = C.eval_states()
    m = oracle_model(False)
    a_ref = C.plan_actions(m, s1, s2, seed=0)
    a_new = P.plan(m, s1, s2, C.LW_PARTICLES, seed=0)
    print(f"  chunked planner = plan_actions: {np.array_equal(a_ref, a_new)}")
    assert np.array_equal(a_ref, a_new)
    sc_b = P.score(a_new, s1, s2)
    a_one = np.concatenate([P.plan(m, s1[j:j + 1], s2[j:j + 1], C.LW_PARTICLES, seed=j) for j in range(len(s1))])
    sc_1 = P.score(a_one, s1, s2)
    print(f"  plan score batched {sc_b:.3f}, one state at a time {sc_1:.3f}")
    assert abs(sc_b - sc_1) < 0.02
    if torch.backends.mps.is_available():
        rng = np.random.default_rng(4)
        lg = C.new_model("lg")
        d = C.simulate(rng, 2000, False, False)
        for n in C.TRANSITION_NODES:
            C.fit_node(lg, "lg", n, d)
        sc_cpu = P.score(P.plan(lg, s1, s2, C.LW_PARTICLES, seed=0), s1, s2)
        sc_mps = P.score(P.plan(P.to_device(lg, "mps"), s1, s2, C.LW_PARTICLES, seed=0), s1, s2)
        print(f"  linear-Gaussian NBN plan score CPU {sc_cpu:.3f}, MPS {sc_mps:.3f}")
        assert abs(sc_cpu - sc_mps) < 0.02


if __name__ == "__main__":
    for f in (test_calibration_and_power, test_planner_oracle,
              test_raw_score_moves_under_behavior_shift, test_w1_metric,
              test_planning_cost_planner):
        print(f.__name__)
        f()
    print("all checks passed")
