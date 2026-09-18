# score-SA for the direct route's pair step

Matt -- this branch is your `etadist` tip (`1566cc9d`) plus one short stack of
commits.  It adds a second way of moving a declared pair's thetas in
`etaDistQ2PairStep`, keeps yours byte-identical behind the same knob, and
records three things I ran into in the branch on the way.  Nothing here touches
the E-step, the objective, or any model without a `dist()` pair.

Hidde van de Beek, September 2026.

## The knob

```r
saemControl(etaDistQ2Rule = c("hybrid", "argmax", "score", "newton"))
```

All four run inside `etaDistQ2PairStep`, on the same sample, the same objective
(`rxEtaDistPairLogD`), the same coordinates (thetas, atanh rho), the same
`pas(kiter)` damping and the same accept-on-improvement gate.  They differ only
in how the proposal is produced from the sample:

| rule | proposal | cost per iteration |
|---|---|---|
| `argmax` | your n1qn1 run to convergence (unchanged) | ~200 objective evaluations |
| `hybrid` (default here) | one step `theta + alpha H^-1 g`, `H` = outer product of the SA-accumulated per-subject scores, alpha from 1 halved by sqrt(2) until the sampled objective improves | 1 gradient + ~1 evaluation |
| `score` | the same step, alpha = 1 | 1 gradient + 1 evaluation |
| `newton` | one step with the finite-difference Hessian of the sampled objective (reference; Fisher fallback where not PD) | ~60 evaluations |

`g` and the per-subject scores come from your `rxEtaDistPairLoglikGrad`, called
one record at a time, so the step is on your objective and nothing is
reimplemented.

## Where it comes from

This is the likelihood-score stochastic approximation of Delyon, Lavielle and
Moulines (1999, section 8.2) that I use in saemix for a non-Gaussian FREM, put
into the shape of NONMEM's non-mu theta route (NONMEM 7 Technical Guide,
eqs. 1.47-1.52 and the alpha test after 1.46):

| | NONMEM | this branch |
|---|---|---|
| gradient | conditional-mean score of the whole joint density | score of the eta-scale prior only, closed form -- possible because on the direct route the declared thetas appear nowhere else |
| curvature | `sum_i g_i g_i'` from this iteration's chains (1.51) | the same form from scores accumulated across iterations (an estimate of the same conditional mean) |
| step | one Newton step, alpha = 1 then /sqrt(2) until Lc improves | identical |
| across iterations | full step, then Polyak-average the estimates (1.152) | `pas(kiter)` damping, as everywhere else in saem |

The guide's sentence "no shortcut evaluation can be made by maximizing just the
parameter density portion" is exactly what the direct route makes false, and
that is the whole reason the gradient is cheap.  The accumulated `sum g_i g_i'`
is returned on the fit as `.etaDistQ2InfoFit` (it is NONMEM's 1.153, the
Appendix C standard-error quantity); at the optimum it agreed with a
finite-difference Hessian of the sampled objective to 3-10%.

## Three things in the branch

All three are measured, and fixed in this stack where a fix was in scope.

1. **`etaDistParam="auto"` cannot run on published dependencies.**  It resolves
   to `direct` for a declared pair under saem, and `rxode2@etadist`'s
   `.rxEtaDistAnchors()` refuses unless `lotri::lotriEtaDists()` carries a
   `roles` column -- which no lotri branch on GitHub has, `etadist` included
   (`git grep roles 76bc359 -- R/` is empty).  Every fit died with
   `needs the argument roles for 'eta.v'`.  I shimmed it locally with
   `roles = parNames`, which is what your own examples (`rxEdA.eta.cl.rate`)
   imply, but the real column presumably sits in your working copy.

2. **`covMethod="sa"` (the default) appends `nSaCov = 500` iterations at gain
   zero, and both M-step rules keep firing through them.**  n1qn1 runs to
   convergence 500 more times for nothing; every "500-iteration" fit was 1000
   and every timing I first took was dominated by it.  Estimates are unaffected.
   I did not change this; the study runs with `covMethod=""`.

3. **The stochastic phi0 update random-walks a declared pair's thetas for the
   first `niter/2` iterations.**  `mprior_phi0 = COV0*MCOV0` (saem.cpp, the
   `skipStochPhi0` block) sets every non-mu theta to its sampled pseudo-eta
   mean.  On the direct route the observation likelihood is flat in the
   declared thetas (the anchors are unread), so that pseudo-eta MCMC is a pure
   random walk under its Gaussian pseudo-prior.  Measured with the pair step
   disabled (`NLMIXR2_ETADIST_FREEZE=1`): the CL shape wandered from 2.9 to
   7.9 over 200 iterations and stopped exactly when `skipStochPhi0` switched
   the update off.  Fix in `d8f6dcf1`: the Q2 columns and the copula column are
   restored after the assignment, so the pair step alone owns them from the
   first iteration.  This helped your rule too (shCL 2.96 -> 2.88 on the
   dataset below), and it was what made the score step look like it drifted
   before the fix (7.28 against an MLE of 6.79 at 1100 iterations; 6.76 after).

Two smaller ones, not fixed: `rho0 = etaDistRho(k)` is read from the LOWER
member's slot, which holds 0 until the first move writes both, so iteration 1
of the pair step always starts at rho = 0; and the pair step is 2x2 only (so is
mine).

## What it shows so far

Two correlated Gamma random effects (V ~ Gamma(6, 0.30), CL ~ Gamma(2.5,
0.714), latent rho 0.6, prop 0.12, N = 200, 8 samples), started at shape 8/4,
rho 0.05, 300 + 200 iterations, `nmc = 1`, one seed, scored against the
observed-data MLE by adaptive Gauss-Hermite quadrature (q16 and q24 agree to 4
digits):

| | shV | rtV | shCL | rtCL | rho | fit time |
|---|---|---|---|---|---|---|
| MLE (AGQ) | 6.785 | 0.3331 | 2.877 | 0.8075 | 0.528 | -- |
| `argmax` | 6.845 | 0.3346 | 2.869 | 0.8015 | 0.532 | 46 s |
| `hybrid` | 6.803 | 0.3331 | 2.875 | 0.8048 | 0.529 | 28 s |
| `newton` | 6.77 | 0.332 | 2.87 | -- | -- | -- |

Run to 1100 iterations all three sit within 0.02 of the MLE on every parameter.
NONMEM-form variants of the step (fresh per-subject outer product; full step
with Polyak averaging; `NLMIXR2_Q2_FRESH_H`, `NLMIXR2_Q2_POLYAK`) all land
within 1% of the MLE on this dataset.  The line search fired once in ~550
firings (alpha 0.71); from this start it is a safeguard, not an estimator.

Replicates (rho in {0, 0.3, 0.6, 0.85}, five datasets each, all three rules,
gold per dataset) are running; the table will replace this section.  Until
then the honest claim is: same fixed point as your rule, at roughly 60% of the
wall-clock, with the information matrix as a by-product -- not better accuracy.

## Caveats I would want a reviewer to know

* The outer product of scores is a poor curvature far from the optimum (8-15x
  too stiff on a coordinate whose shape sat at 7.9 against an MLE of 2.9).  That
  is BHHH's known weakness and NONMEM's route inherits it; the line search can
  only shorten a step, so it does not help there.  `newton` exists to expose it.
* At `nmc = 1` the per-subject score is one draw, not NONMEM's conditional mean
  over `ISAMPLE` chains; the accumulation across iterations is what stands in
  for that.  `NLMIXR2_Q2_FRESH_H` averages over chains first when `nmc > 1`.
* Pair-only, like yours.

## Running it

```r
# deps: rxode2@etadist, lotri@etadist + the roles shim
# (github.com/hiddevandebeek/lotri, branch roles-shim), rxode2ll
devtools::load_all(".", helpers = FALSE)
fit <- nlmixr2(model, data, est = "saem",
               control = saemControl(etaDistQ2Rule = "hybrid", covMethod = ""))
get(".etaDistQ2LsFit", envir = fit$env)    # firings / shortened / failed / minAlpha
get(".etaDistQ2InfoFit", envir = fit$env)  # accumulated score information
```

Diagnostics, all under environment variables and all off by default:
`NLMIXR2_ETADIST_OPT` (per-firing line-search trace, and the FD-Hessian check
at the last iteration), `NLMIXR2_ETADIST_MOVE` (the proposed move per firing),
`NLMIXR2_ETADIST_FREEZE` (compute the step, never apply it).

Commits, oldest first: `de7ac7b8` the knob; `cfcf2c91` counters and the
information check; `d8f6dcf1` the phi0 fix and the `newton` rule; `296bb6aa`
the NONMEM-form variants.
