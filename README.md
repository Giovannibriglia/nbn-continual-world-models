# Neural Bayesian Networks as Continual World Models

Code for the NeurIPS 2026 workshop paper *Neural Bayesian Networks as Continual World Models for Reinforcement Learning* (Workshop on Continual World Models).

## The pipeline

A Neural Bayesian Network (NBN) is a Bayesian network whose mechanisms, one per variable, are learned conditional densities, such as mixture density networks [(Briglia et al., 2026)](https://arxiv.org/abs/2610.16902). The paper builds a pipeline with an NBN as its backbone, so that the NBN serves as a world model that keeps learning while the agent plans with it.

The world model is a two-slice NBN over $(S_t, A_t, S_{t+1}, R_t)$: one learned mechanism per next-state variable, a behavior policy into the action, and a goal node $R$. On every new batch $W$ of experience, the pipeline runs three steps:

1. **Test: which mechanism changed?** For each mechanism $i$, test $H_0: X_i \perp E \mid \mathrm{Pa}(i)$, where $E$ tells the stored data from the new batch. The test is a matched-pair sign-flip permutation test: each new transition is paired with a stored one whose parents are close, and the test asks whether the parents predict the difference of the two children. It does not use the model, and a Bonferroni correction is applied over the mechanisms.

   Why not the per-mechanism log-likelihood? Its expectation is $-\mathbb{E}_q[H(p^\star_i)] - \mathbb{E}_q[\mathrm{KL}(p^\star_i \Vert p_{\phi_i})]$, where $q$ is the distribution of the parents. The entropy term moves whenever the agent's behavior moves $q$, even if the mechanism is unchanged (Eq. 1 of the paper). The test compares conditionals, so it is not fooled by a change of behavior.
2. **Update: change only what changed.** Each mechanism keeps the data $\mathcal{D}_i$ seen since its last restart. If the test flags it, it restarts from $W$. Otherwise $\mathcal{D}_i \gets \mathcal{D}_i \cup W$ and it is refit (warm start for MDNs).
3. **Plan: ask the network which action succeeds.** With $R=1$ when the next state reaches the goal and a uniform prior on $A$, $p(a \mid s, R=1) \propto P(R=1 \mid s, do(a))$. The posterior is computed for all states in one batched likelihood-weighting call (2,048 particles per state), and the planned action is its mode.

## The experiment (Section 4 and Appendix A of the paper)

The world is a toy MDP with two continuous states and an action $A \in [-1, 1]$:
- $S_1'$ is linear in $(S_1, A)$, with skewed noise;
- $S_2'$ is nonlinear in $A$, with bimodal noise whose spread grows with $S_1$;
- the goal is $|S_2'| < 0.2$.

Experience arrives in 8 phases of 2,000 transitions, and the timeline has two events:
- at phase 2 the **behavior** changes: other states are visited, and the policy depends on $S_1$; no mechanism changes;
- at phase 5 the **mechanism** of $S_2'$ changes: the effect of the action is reversed.

The pipeline (MDN mechanisms; a linear-Gaussian NBN is the reference) is compared with four other ways to maintain the same NBN:
- frozen after phase 0;
- pooled refit on all data;
- a sliding window (the last phase only);
- *raw likelihood + restart*: the pipeline with the test replaced by a z-test for a drop of the log-likelihood.

The metrics are the Wasserstein-1 distance of the model to the current dynamics, and the success of the planned action relative to the optimal one.

## Cost of planning, and other engines (Appendix C of the paper)

- **Figure 3:** the planner alone: plan score against particles per state, and the time per state when $B$ states are answered by one batched call or one call per state (Apple M5 laptop: CPU and MPS GPU).
- **Figure 4:** the same planner in three engines: the NBN, Pyro (vectorized with plates) and NumPyro (compiled with `jax.jit`). The Pyro and NumPyro models load the trained weights of the NBN's MDNs, so the three engines run the same model with the same estimator.
  - (a) speed relative to Pyro on the same device, for 10,240 queries answered in calls of $B$ states (Linux server, NVIDIA TITAN X (Pascal) GPU). For NumPyro, since it is on JAX, the compilation time is counted: JAX compiles once per batch size.
  - (b) plan score against particles per state (5 seeds).

## Install

Python 3.10 or later; tested with Python 3.12 on macOS and Python 3.10 on Linux.

```bash
git clone https://github.com/Giovannibriglia/nbn-continual-world-models.git
cd nbn-continual-world-models
pip install -r requirements.txt      # nbn==0.20.2 (with PyTorch), scipy, matplotlib, pyro-ppl, numpyro, jax
```

Pyro, NumPyro and JAX are needed only for Figure 4 (`engine_benchmark.py`). For its GPU runs, install a CUDA build of PyTorch and `pip install "jax[cuda12]"`. The paper's server used PyTorch 2.6.0 with CUDA 12.4 (its driver supports CUDA 12.4 at most), JAX 0.5.3, NumPyro 0.19.0 and Pyro 1.9.2.

[`nbn`](https://github.com/Giovannibriglia/NeuralBayesianNetworks) is the NBN library ([PyPI](https://pypi.org/project/nbn/)). It is pinned to the version used for the paper.

## Run

Run every command from the repository root. Figures 2 and 3 need only a laptop CPU; the speed panel of Figure 4 uses a CUDA GPU when one is available.

| Step | Command | What it reproduces | Output | Time |
|---|---|---|---|---|
| 1 | `python test_v11.py` | Sanity checks (see below) | printed | ~1 min |
| 2 | `python continual_v11.py [n_seeds]` | Section 4: the whole timeline for MDN and linear-Gaussian NBNs under the five strategies | `results_v11/continual_v11.json` | ~1 min per seed |
| 3 | `python make_figures_v11_mdn.py` | Figure 2, and the whole-timeline summary quoted in Section 4 | `Images/continual_v11_mdn.pdf/.png` | seconds |
| 4 | `python planning_cost.py [n_seeds]` | Appendix C, Figure 3: plan score vs particles; time per planned state, batched vs one state per call, on CPU (1 thread and PyTorch's default thread count) and on the Apple GPU (MPS) if available | `results_v11/planning_cost.json` | ~4 min per seed |
| 5 | `python make_figure_planning_cost.py` | Figure 3, and the numbers quoted in Appendix C | `Images/planning_cost.pdf/.png` | seconds |
| 6 | `python engine_benchmark.py accuracy` | Appendix C, Figure 4b: plan score vs particles for the NBN, Pyro and NumPyro on CPU, 5 seeds | `results_v11/engine_accuracy.json` | a few minutes per seed |
| 7 | `python engine_benchmark.py speed` | Appendix C, Figure 4a: time to answer 10,240 queries in calls of $B$ states, for the NBN and Pyro on CPU and CUDA GPU, and NumPyro on CPU | `results_v11/engine_speed.json` | ~15 min |
| 8 | `python engine_benchmark.py speed --jax-platform cuda --only-numpyro` | Appendix C, Figure 4a: the same for NumPyro on the GPU (a separate process: JAX and PyTorch share a GPU badly) | `results_v11/engine_speed_numpyro_cuda.json` | ~2 min |
| 9 | `python make_figure_engine_benchmark.py` | Figure 4, and the numbers quoted in Appendix C (NumPyro's compilation time included) | `Images/engine_benchmark.pdf/.png` | seconds |

- **Seeds:** the paper uses 5 seeds, which is the default. Use at least 2, because the summaries report standard deviations. The speed runs use one seed: each point is already the total time of 10,240 queries.
- **Precomputed results:** the results used in the paper are in `results_v11/`, so steps 3, 5 and 9 rebuild the figures without rerunning the experiments. Rerunning an experiment overwrites its file.
- **Timings:** they depend on the machine. Figure 3's are from an Apple M5 laptop. Figure 4's are from a Linux server with an NVIDIA TITAN X (Pascal) GPU and a CPU shared with other jobs, so its CPU times are less reliable than its GPU times. Plan scores can differ slightly across machines and library versions, because the random streams do.

**What the checks in `test_v11.py` cover:**
- calibration and power of the drift test, with and without a behavior shift;
- the planner with the true dynamics (plan score 0.964);
- the raw log-likelihood moving under a behavior shift even with the true mechanism;
- the $W_1$ floor of the true model (0.070);
- the Appendix C planner: it returns the same actions as the paper's planner, and batched, one-state-at-a-time and (when available) MPS plans score alike.

**Key numbers to expect (paper, 5 seeds):**
- the test flags exactly the 5 real changes out of 70 tests, and nothing at the behavior shift; the raw likelihood flags $S_2'$ at the behavior shift in every run;
- right after the drift, the pipeline plans at 92% of optimal, against 96% with the true dynamics and 25% for pooling.

## Files

| File | Content | Paper |
|---|---|---|
| `v11_common.py` | the toy MDP, the NBN world models (MDN, linear-Gaussian), the drift test, the raw-likelihood detector, the planner (likelihood weighting), the metrics | Sec. 3, 4; App. A |
| `continual_v11.py` | the experiment: the timeline and the five maintenance strategies | Sec. 4 |
| `make_figures_v11_mdn.py` | Figure 2 and the whole-timeline summary | Sec. 4 |
| `planning_cost.py` | cost and accuracy of planning by inference | App. C |
| `make_figure_planning_cost.py` | Figure 3 | App. C |
| `engine_benchmark.py` | the planner in the NBN, Pyro and NumPyro: accuracy and speed | App. C |
| `make_figure_engine_benchmark.py` | Figure 4 | App. C |
| `test_v11.py` | the checks | — |
| `results_v11/` | the results used in the paper (JSON, logs) | — |
| `Images/` | the figures | — |

## Citation

```bibtex
@inproceedings{briglia2026nbn,
  title     = {Neural Bayesian Networks as Continual World Models for Reinforcement Learning},
  author    = {Briglia, Giovanni and Lippi, Marco and Mariani, Stefano and Zambonelli, Franco},
  booktitle = {NeurIPS 2026 Workshop on Continual World Models},
  year      = {2026}
}

@article{briglia2026neural,
  title   = {Neural {B}ayesian Networks for Modular and Scalable Decision-Making Under Uncertainty},
  author  = {Briglia, Giovanni and Lippi, Marco and Mariani, Stefano and Zambonelli, Franco},
  journal = {arXiv preprint arXiv:2610.16902},
  year    = {2026}
}
```

## License

Apache License 2.0 (see `LICENSE`).
