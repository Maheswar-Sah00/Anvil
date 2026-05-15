# P-04 technical writeup

## Problem and contribution

Precision-controlled associative memory (PCAM) stores a codebook of patterns and retrieves them by gradient flow on a fixed energy landscape. The agent sees only a corrupted query and must return a positive precision vector that scales each coordinate of the gradient during integration. Our contribution is a hybrid agent: offline, class-conditional templates from the Hessian (with Ruiz equilibration) and from pattern variance; online, a softmax posterior over classes mixes those templates, then confidence gating pulls precision toward identity when the posterior is ambiguous. The goal is to improve retrieval over the Pi=I baseline without violating per-seed penalty gates on accuracy drop or spread reduction.

## The math

The frozen bench energy (see `bench-p04-pcam/pcam_model.py`) is

\[
E(a) = \tfrac{1}{2} a^\top R a - \frac{\eta}{\beta} \log \sum_i e^{\beta x_i^\top a}.
\]

At state \(a\), with \(s = \mathrm{softmax}(\beta X a)\),

\[
H(a) = R - \eta \beta \, X^\top (\mathrm{diag}(s) - s s^\top) X.
\]

Dynamics apply precision element-wise to the gradient: \(a_{t+1} = a_t + \Delta t\,(-\pi \odot \nabla E(a_t) + u)\) during the input window. For offline templates we evaluate \(H\) at each stored pattern \(X_k\), regularize with \(10^{-3} I\), and run symmetric Ruiz scaling: repeatedly \(D = \mathrm{diag}(1/\sqrt{\max_j |H_{ij}|})\), \(H \leftarrow D H D\), accumulating diagonal scales \(d\); the precision template is \(\pi_i = d_i^2\). Query-time \(\pi\) is a geometric mix of Hessian and class templates, modulated by softmax weights and confidence exponent on \(w_{\max}\).

## Architecture decisions

**Why hybrid.** Test 2 measures anisotropy (ratio of largest to smallest eigenvalue of the symmetrized contraction operator under \(\pi\)). Hessian-based Ruiz templates directly target ill-conditioning of the local quadratic model. Class-variance templates capture which coordinates differ across stored patterns at low cost and need no per-query linear algebra. Neither alone covers both geometry and discriminative structure.

**Why geometric mean.** Hessian and class precisions are positive scales on different axes. A weighted arithmetic mean can be dominated by outliers; \(\pi_{\mathrm{combined}} \propto \pi_{\mathrm{hess}}^{h_w} \pi_{\mathrm{class}}^{1-h_w}\) blends in log-space, is stable, and equals the two templates when \(h_w \in \{0,1\}\).

**Why confidence gating.** After mixing templates, we raise \(\pi\) to the power \(c = w_{\max}^{\gamma}\) with \(\gamma =\) `confidence_exp`. When the class posterior is flat, \(w_{\max}\) is small and \(c \to 0\), so \(\pi \to 1\) after normalization—reducing harmful over-modulation on hard queries. When one class dominates, \(c \to 1\) and the full template applies. This defends Test 1 (per-seed \(\Delta\) vs identity) on ambiguous inputs while allowing strong precision when the query is informative.

## Robustness arguments

**Test 2 (anisotropy).** For each class, Ruiz \(\pi\) is accepted only if eigenvalue spread strictly decreases versus the raw Hessian; otherwise we fall back to inverse diagonal magnitudes. Combined templates are clipped to harness bounds \([0.1, 10]\). The offline safety net guarantees we never deploy a Hessian template that worsens local conditioning on the stored pattern. In stress runs, vanilla configs show median spread ratios \(\approx 0.53\)–\(0.55\) relative to the dummy baseline (below the \(1.0\times\) gate), so the stress script flags `penalty=yes` on spread even when delta is nonnegative. High-\(K\), high-\(N\), and PCA-MNIST rows pass the spread gate (\(\approx 1.05\)–\(1.22\)) but retrieval accuracy is \(0\%\)—scale stress, not a retrieval win.

**Test 1 (retrieval).** Confidence gating limits deviation from \(\pi = 1\) when softmax mass is diffuse, which caps regression on seeds where aggressive \(\pi\) would steer dynamics into wrong basins. A 36-config sweep (`docs/SWEEP_RESULTS.md`) ranks `cls_beta_mult=1.0`, `confidence_exp=0.5`, `hessian_weight=0.5` first with mean \(\Delta = +0.0027\) and min \(\Delta = +0.0000\) on fast eval. Stress still shows occasional negative per-seed deltas (e.g. vanilla-6 at \(-0.06\)), so the gate is a design target, not a theorem for every seed.

## Results

Stress table (`docs/STRESS_RESULTS.md`, fast mode): on \(K=16, N=64\) vanilla seeds, test1 accuracy ranges \(0.70\)–\(0.88\); most deltas are \(0\) or small positive, with penalties driven mainly by spread \(< 1.0\). Scaling tests (high-\(K\), high-\(N\), PCA-MNIST) show spread above the gate but zero retrieval accuracy under the current agent.

Sweep: best fast configuration uses geometric mix weight \(0.5\), confidence exponent \(0.5\), and class softmax multiplier \(1.0\), with nonnegative worst-seed delta.

Figures in `notebooks/figures/`: `plot_1.png` eigenvalue spectrum before/after \(\Pi\) on the worst-conditioned class; `plot_2.png` PCA trajectories showing Pi=I vs agent basins; `plot_4.png` distribution of query-time \(\pi\) over 1000 corrupted queries (non-trivial spread, not all 1.0).

## References

- Ramsauer et al. (2020). Modern Hopfield networks and attention as associative memory.
- Krotov and Hopfield (2016). Dense associative memory for pattern storage and retrieval.
- PCAM predecessor (OpenReview, 2023). Precision-controlled associative memory (lineage to the NeurIPS 2026 PCAM submission in the bench docstring).
- GRACE and related work on curvature-aware / Hessian-informed preconditioning for gradient-based optimization.
