#ifndef __SCORESA_H__
#define __SCORESA_H__
// One preconditioned step along the complete-data score, as an alternative to
// SAEM's finite-statistic M-step for the Gaussian population block.
//
// SAEM smooths sufficient statistics and then MAXIMIZES: the mu-referenced
// thetas come out of a GLS solve against statphi11, and Omega out of the second
// moment of the deviations.  That is exact for this model, and it is what makes
// saem fast.  The alternative -- Delyon, Lavielle and Moulines section 8.2,
// equation 74 -- never maximizes.  It takes ONE step along the complete-data
// score, preconditioned by an estimate of the Fisher information accumulated
// along the recursion:
//
//   H_k       = (1/N) sum_i s_i^k
//   Delta_i^k = (1 - gamma_k) Delta_i^{k-1} + gamma_k s_i^k
//   I_k       = (1/N) sum_i Delta_i^k (Delta_i^k)'
//   theta_k   = theta_{k-1} + gamma_k (I_k + r_k I)^-1 H_k
//
// By Fisher's identity the conditional expectation of the complete-data score
// is the observed-data score, so the recursion targets the observed-data
// likelihood without ever forming it.  The information estimate is Delattre and
// Kuhn's, averaged along the recursion rather than over one batch, following
// Baey, Delattre, Kuhn, Leger and Lemler (2023), Algorithm 1.
//
// WHY IT IS HERE.  For a Gaussian population block this cannot beat the exact
// M-step -- it solves the same stationary equations by a slower route.  It is
// here to be MEASURED against it on a standard fit, and because the step
// generalizes to population models whose M-step is not closed-form (declared
// non-Gaussian families, copula blocks) where saem must otherwise fall back on
// a derivative-free search over an ODE-solving objective.
//
// The parameterization is unconstrained so the step cannot leave the model:
// mu-referenced thetas as they are, Omega through its Cholesky factor with a
// log diagonal.  `covstruct1` is respected -- an off-diagonal the model does
// not estimate carries no coordinate.

#include <RcppArmadillo.h>

// The step's OWN gain, not the host's.
//
// SAEM's `pas` is 1.0 throughout burn-in, which is right for a maximizer -- an
// exact M-step cannot overshoot, so a full step is free.  A preconditioned
// score step is a NEWTON step, and its natural length is one only once the
// information estimate is worth trusting.  Measured on a plain Gaussian fit
// (200 subjects, lognormal V and CL): with 300 full-gain iterations the
// location block ran to CL 22.8 against a truth of 3.5; with 40 it reached
// 4.6; Omega was unharmed either way, because its score is far better
// conditioned than the location block's.
//
// So the step gets the three-phase schedule of Baey, Delattre, Kuhn, Leger and
// Lemler (2023) section 3.4.1, which is what the saemix implementation uses:
//
//   pre-heating   gamma_k = gamma0^(1 - k/K),  rising from gamma0 to 1
//   heating       gamma_k = 1
//   decreasing    gamma_k = (k - K)^-alpha
//
// The pre-heating phase is the one that matters here: it lets the information
// accumulate before any full step is taken.
static inline double scoreSaGain(unsigned int k, unsigned int preheat,
                                 double start, double alpha,
                                 unsigned int heatEnd) {
  if (k <= preheat && preheat > 0)
    return std::pow(start, 1.0 - (double)k / (double)preheat);
  if (k <= heatEnd) return 1.0;
  return std::pow((double)(k - heatEnd), -alpha);
}

struct scoreSaState {
  arma::mat deltaSubject;          // N x p, the smoothed per-subject scores
  arma::mat inverseInformation;    // p x p
  unsigned int iteration = 0;
  unsigned int metricUpdates = 0;
  bool ready = false;
  // Polyak average of the iterate over the decreasing-gain phase.
  //
  // The score here is driven by ONE MCMC draw per subject, where SAEM's M-step
  // maximizes a SMOOTHED sufficient statistic.  The recursion therefore
  // oscillates around its fixed point rather than settling on it, and reporting
  // the last iterate samples that oscillation -- measured on a plain Gaussian
  // fit, the location block came out high at 500 iterations and low at 1800,
  // with Omega (whose score is far better conditioned) unaffected either way.
  // Averaging the iterate over the decreasing phase is what the saemix
  // implementation reports (copulaScoreSa.R:1191-1209), and it is the standard
  // Polyak-Ruppert remedy for exactly this.
  arma::vec average;
  unsigned int averageCount = 0;
  bool averaging = false;
  // Expanding truncation sets, Fort, Moulines, Schreck and Vihola (2016).
  // Projected stochastic approximation converges to a boundary point of the
  // projected ODE, which is NOT a stationary point of the likelihood, so a
  // FIXED box is not safe.  Each hit doubles the half-width and returns the
  // iterate to the centre; under the coercivity the convergence argument
  // already assumes, the boundary is hit finitely often almost surely.
  arma::vec centre;
  arma::vec width;
  unsigned int expansions = 0;
  unsigned int backtracks = 0;
};

static inline void scoreSaOmegaIndex(const arma::mat &covstruct1,
                                     const arma::uvec &flatPhi1,
                                     std::vector<std::pair<unsigned int,
                                     unsigned int> > &index);

// Write a packed coordinate vector back onto the model's parameters.
static inline void scoreSaUnpack(const arma::vec &coords, unsigned int nBeta,
                                 const std::vector<std::pair<unsigned int,
                                 unsigned int> > &omegaIndex,
                                 arma::vec &Plambda1, arma::mat &Gamma2) {
  for (unsigned int l = 0; l < nBeta && l < Plambda1.n_elem; ++l)
    Plambda1(l) = coords(l);
  for (size_t k = 0; k < omegaIndex.size(); ++k) {
    unsigned int a = omegaIndex[k].first, b = omegaIndex[k].second;
    if (a == b) Gamma2(a, a) = std::exp(coords(nBeta + k));
    else { Gamma2(a, b) = coords(nBeta + k); Gamma2(b, a) = Gamma2(a, b); }
  }
}

// Report the averaged iterate, if one has accumulated.  Called once, at the
// last iteration.
static inline bool scoreSaReportAverage(scoreSaState &state,
                                        const arma::mat &covstruct1,
                                        const arma::uvec &flatPhi1,
                                        arma::vec &Plambda1,
                                        arma::mat &Gamma2) {
  if (!state.averaging || state.averageCount < 2 || !state.average.is_finite())
    return false;
  std::vector<std::pair<unsigned int, unsigned int> > omegaIndex;
  scoreSaOmegaIndex(covstruct1, flatPhi1, omegaIndex);
  if (omegaIndex.empty()) return false;
  unsigned int nBeta = state.average.n_elem - (unsigned int)omegaIndex.size();
  arma::mat candidate = Gamma2;
  arma::vec lambda = Plambda1;
  scoreSaUnpack(state.average, nBeta, omegaIndex, lambda, candidate);
  arma::mat probe;
  if (!arma::inv_sympd(probe, candidate)) return false;
  Plambda1 = lambda; Gamma2 = candidate;
  return true;
}

// Free coordinates of Omega: the log-diagonal always, plus every off-diagonal
// the covariance structure estimates.  Returned as (row, col) pairs; a pair
// with row == col is a log-diagonal coordinate.
static inline void scoreSaOmegaIndex(const arma::mat &covstruct1,
                                     const arma::uvec &flatPhi1,
                                     std::vector<std::pair<unsigned int,
                                     unsigned int> > &index) {
  index.clear();
  unsigned int n = covstruct1.n_rows;
  arma::uvec flat(n, arma::fill::zeros);
  for (unsigned int f = 0; f < flatPhi1.n_elem; ++f)
    if (flatPhi1(f) < n) flat(flatPhi1(f)) = 1;
  for (unsigned int a = 0; a < n; ++a) {
    if (flat(a)) continue;                 // no Omega of its own
    index.push_back(std::make_pair(a, a));
  }
  for (unsigned int a = 1; a < n; ++a) {
    if (flat(a)) continue;
    for (unsigned int b = 0; b < a; ++b) {
      if (flat(b)) continue;
      if (covstruct1(a, b) != 0) index.push_back(std::make_pair(a, b));
    }
  }
}

// The complete-data score of one subject, in the free coordinates.
//
//   d/dbeta   log p(phi_i | mu_i, Omega) = X_i' Omega^-1 (phi_i - mu_i)
//   d/dOmega  log p(phi_i | mu_i, Omega) = -1/2 (Omega^-1 - Omega^-1 e e' Omega^-1)
//
// carried to the log-diagonal by the chain rule (a factor of Omega_aa) and to
// an off-diagonal by its symmetry (a factor of two).
static inline arma::vec scoreSaSubject(const arma::rowvec &e,
                                       const arma::rowvec &design,
                                       const arma::mat &omegaInverse,
                                       const arma::mat &omega,
                                       const arma::umat &lambdaMap,
                                       const std::vector<std::pair<unsigned int,
                                       unsigned int> > &omegaIndex) {
  unsigned int nBeta = lambdaMap.n_rows;
  arma::vec answer(nBeta + omegaIndex.size(), arma::fill::zeros);
  arma::vec oe = omegaInverse * e.t();
  for (unsigned int l = 0; l < nBeta; ++l) {
    unsigned int column = lambdaMap(l, 0);
    unsigned int covariate = lambdaMap(l, 1);
    answer(l) = design(covariate) * oe(column);
  }
  arma::mat outer = oe * oe.t();
  for (size_t k = 0; k < omegaIndex.size(); ++k) {
    unsigned int a = omegaIndex[k].first, b = omegaIndex[k].second;
    double value = -0.5 * (omegaInverse(a, b) - outer(a, b));
    if (a == b) value *= omega(a, a);            // log-diagonal
    else value *= 2.0;                           // symmetry
    answer(nBeta + k) = value;
  }
  return answer;
}

// One step.  `Plambda1` and `Gamma2` are updated in place; `false` means the
// step declined and the caller should keep whatever the exact M-step produced.
static inline bool scoreSaStep(scoreSaState &state,
                               const arma::mat &phi1,
                               const arma::mat &mprior,
                               const arma::mat &COV1,
                               const arma::umat &lambdaMap,
                               const arma::mat &covstruct1,
                               const arma::uvec &flatPhi1,
                               const arma::uvec &fixedIx1,
                               double gain, double ridge, bool averagePhase,
                               double boxWidth,
                               arma::vec &Plambda1, arma::mat &Gamma2) {
  unsigned int N = phi1.n_rows;
  if (N < 2 || !phi1.is_finite() || !mprior.is_finite()) return false;
  arma::mat omega = Gamma2;
  std::vector<std::pair<unsigned int, unsigned int> > omegaIndex;
  scoreSaOmegaIndex(covstruct1, flatPhi1, omegaIndex);
  if (omegaIndex.empty()) return false;
  arma::mat omegaInverse;
  if (!arma::inv_sympd(omegaInverse, omega)) return false;
  unsigned int nBeta = lambdaMap.n_rows;
  unsigned int p = nBeta + (unsigned int)omegaIndex.size();

  arma::mat scores(N, p);
  for (unsigned int i = 0; i < N; ++i)
    scores.row(i) = scoreSaSubject(phi1.row(i) - mprior.row(i), COV1.row(i),
                                   omegaInverse, omega, lambdaMap,
                                   omegaIndex).t();
  if (!scores.is_finite()) return false;
  arma::vec meanScore = arma::mean(scores, 0).t();

  // Delta follows the parameter gain, as in Baey et al.: the information must
  // not lag an iterate that is still moving, or a full step leaves the region.
  if (state.deltaSubject.n_rows != N || state.deltaSubject.n_cols != p)
    state.deltaSubject = scores;
  else
    state.deltaSubject = (1.0 - gain) * state.deltaSubject + gain * scores;
  arma::mat information = state.deltaSubject.t() * state.deltaSubject / (double)N;
  information = 0.5 * (information + information.t());
  // Both floors: a relative ridge is equivariant but needs tr(I) > 0, an
  // absolute one is unconditional but not equivariant.  The max of the two is
  // unconditional AND equivariant wherever there is a scale to be equivariant
  // to.
  double relative = ridge * arma::trace(information) / (double)p;
  double floorValue = std::max(1e-10, relative);
  information.diag() += floorValue;
  arma::mat inverse;
  if (arma::inv_sympd(inverse, information)) {
    state.inverseInformation = inverse;
    state.metricUpdates++;
    state.ready = true;
  } else if (!state.ready) {
    state.inverseInformation = arma::eye(p, p);
  }
  if (state.inverseInformation.n_rows != p) state.inverseInformation = arma::eye(p, p);

  arma::vec direction = state.inverseInformation * meanScore;
  if (!direction.is_finite()) return false;

  // Assemble the current point in the free coordinates, step, and read back.
  arma::vec current(p, arma::fill::zeros);
  for (unsigned int l = 0; l < nBeta; ++l) current(l) = Plambda1(l);
  for (size_t k = 0; k < omegaIndex.size(); ++k) {
    unsigned int a = omegaIndex[k].first, b = omegaIndex[k].second;
    current(nBeta + k) = (a == b) ? std::log(std::max(omega(a, a), 1e-300))
      : omega(a, b);
  }
  arma::vec proposal = current + gain * direction;
  if (!proposal.is_finite()) return false;
  // Truncation box.  Half-width `boxWidth` around the starting point, doubling
  // and re-initialising on every hit.
  if (state.centre.n_elem != p) {
    state.centre = current;
    state.width = arma::vec(p, arma::fill::value(boxWidth));
    state.expansions = 0;
  }
  {
    arma::vec lower = state.centre - state.width;
    arma::vec upper = state.centre + state.width;
    arma::vec projected = arma::min(upper, arma::max(lower, proposal));
    if (arma::any(arma::abs(projected - proposal) > 0)) {
      state.width *= 2.0;
      state.expansions++;
      // re-initialise, as Algorithm 2 of Fort et al. requires
      proposal = state.centre;
      state.deltaSubject.reset();
      state.average.reset(); state.averageCount = 0; state.averaging = false;
    } else {
      proposal = projected;
    }
  }
  // A theta the model fixes takes no step.
  for (unsigned int f = 0; f < fixedIx1.n_elem; ++f)
    if (fixedIx1(f) < nBeta) proposal(fixedIx1(f)) = current(fixedIx1(f));

  arma::mat candidate = omega;
  for (size_t k = 0; k < omegaIndex.size(); ++k) {
    unsigned int a = omegaIndex[k].first, b = omegaIndex[k].second;
    if (a == b) candidate(a, a) = std::exp(proposal(nBeta + k));
    else { candidate(a, b) = proposal(nBeta + k); candidate(b, a) = candidate(a, b); }
  }
  // Step halving until the candidate Omega is usable.  The log-diagonal keeps
  // the variances positive; an off-diagonal step can still leave the positive
  // definite cone, and clipping it would move the fixed point.
  double scale = 1.0;
  arma::mat probe;
  while (!arma::inv_sympd(probe, candidate) && scale > 1.0 / 4096.0) {
    scale *= 0.5;
    arma::vec halved = current + scale * gain * direction;
    for (unsigned int f = 0; f < fixedIx1.n_elem; ++f)
      if (fixedIx1(f) < nBeta) halved(fixedIx1(f)) = current(fixedIx1(f));
    candidate = omega;
    for (size_t k = 0; k < omegaIndex.size(); ++k) {
      unsigned int a = omegaIndex[k].first, b = omegaIndex[k].second;
      if (a == b) candidate(a, a) = std::exp(halved(nBeta + k));
      else { candidate(a, b) = halved(nBeta + k); candidate(b, a) = candidate(a, b); }
    }
    proposal = halved;
  }
  if (!arma::inv_sympd(probe, candidate)) return false;

  for (unsigned int l = 0; l < nBeta; ++l) Plambda1(l) = proposal(l);
  Gamma2 = candidate;
  // Average only over the decreasing-gain phase: averaging a constant-gain
  // random walk would average the exploration, not the estimate.
  if (averagePhase) {
    if (!state.averaging || state.average.n_elem != proposal.n_elem) {
      state.average = proposal; state.averageCount = 1; state.averaging = true;
    } else {
      state.averageCount++;
      state.average += (proposal - state.average) / (double)state.averageCount;
    }
  }
  state.iteration++;
  return true;
}


// ===========================================================================
// DECLARED-FAMILY BLOCK: the score for dist()-declared margins under a
// Gaussian copula, in ETA space.
//
// This is the case with no closed-form M-step, and the reason the score route
// is worth having.  After rxEtaDistExpand() the family parameters sit in the
// DATA likelihood, so their score there would need differentiating through the
// ODE solve.  In eta space they are prior-only again:
//
//   log p(eta; a, R) = log c_R(xi) + sum_j log f_j(eta_j; a_j),
//   xi_j = Phi^-1{F_j(eta_j; a_j)}
//
//   d/da_j log p = d/da_j log f_j  +  [-(R^-1 - I) xi]_j * dxi_j/da_j
//   dxi_j/da_j   = (dF_j/da_j) / phi(xi_j)
//
// The second term is the one a marginal-only M-step cannot see.  Verified
// against finite differences of the complete-data log density to 2.4e-9 on two
// correlated gammas (analysis/gammaEtaScore.R).
//
// dF/da has no elementary closed form for a Gamma's shape, so it is taken by
// central difference -- a scalar per draw, negligible against the solve.

// CDF, mirroring rxEtaDistQ()'s dispatch.  Local to this header on purpose.
static inline double scoreSaCdf(int fam, double x, const double *a) {
  switch (fam) {
  case RXETADIST_NORM:      return R::pnorm(x, a[0], a[1], 1, 0);
  case RXETADIST_STDNORMAL: return R::pnorm(x, 0.0, 1.0, 1, 0);
  case RXETADIST_CAUCHY:    return R::pcauchy(x, a[0], a[1], 1, 0);
  case RXETADIST_LOGIS:     return R::plogis(x, a[0], a[1], 1, 0);
  case RXETADIST_LNORM:     return R::plnorm(x, a[0], a[1], 1, 0);
  case RXETADIST_CHISQ:     return R::pchisq(x, a[0], 1, 0);
  case RXETADIST_EXP:       return R::pexp(x, 1.0/a[0], 1, 0);
  case RXETADIST_GAMMA:     return R::pgamma(x, a[0], 1.0/a[1], 1, 0);
  case RXETADIST_WEIBULL:   return R::pweibull(x, a[0], a[1], 1, 0);
  case RXETADIST_BETA:      return R::pbeta(x, a[0], a[1], 1, 0);
  case RXETADIST_UNIF:      return R::punif(x, a[0], a[1], 1, 0);
  default:                  return NA_REAL;
  }
}

// Exact d(logD)/d(native parameter), defined in etaDistFam.cpp.  Declared here
// rather than in etaDistFam.h so the header is untouched.
bool rxEtaDistGradD(int fam, double x, const double *a, double *ll, double *g);

// d(log f)/da EXACTLY where rxode2ll provides it, dF/da by central difference.  Using the same
// scheme for each keeps the two consistent to the same order, which matters
// because they are added.
static inline void scoreSaFamilyGrad(int fam, double x, const double *a,
                                     int na, double *dLogf, double *dCdf) {
  // The analytic path halves the work and is exact; only the CDF derivative
  // genuinely needs differencing (the incomplete gamma has no elementary
  // derivative in its shape).
  double ll = 0.0;
  bool exact = rxEtaDistGradD(fam, x, a, &ll, dLogf);
  for (int j = 0; j < na; ++j) {
    double h = 1e-6 * std::max(1.0, std::fabs(a[j]));
    double ap[4], am[4];
    for (int i = 0; i < na; ++i) { ap[i] = a[i]; am[i] = a[i]; }
    ap[j] += h; am[j] -= h;
    if (!exact)
      dLogf[j] = (rxEtaDistLogD(fam, x, ap) - rxEtaDistLogD(fam, x, am)) / (2*h);
    dCdf[j] = (scoreSaCdf(fam, x, ap) - scoreSaCdf(fam, x, am)) / (2*h);
  }
}

// One preconditioned score step for a declared PAIR (family parameters of both
// margins plus the correlation, in the atanh coordinate the expansion carries).
// Returns false to decline, leaving the caller's parameters untouched.
static inline bool scoreSaEtaDistStep(scoreSaState &state,
                                      const std::vector<double> &eta0,
                                      const std::vector<double> &eta1,
                                      int fam0, int fam1,
                                      double *a0, double *a1, double &rho,
                                      double gain, double ridge,
                                      bool averagePhase, double boxWidth) {
  int na0 = rxEtaDistNarg(fam0), na1 = rxEtaDistNarg(fam1);
  if (na0 <= 0 || na1 <= 0) return false;
  size_t n = std::min(eta0.size(), eta1.size());
  if (n < 10) return false;
  if (!std::isfinite(rho) || std::fabs(rho) >= 0.999) return false;
  unsigned int p = (unsigned int)(na0 + na1 + 1);
  double det = 1.0 - rho*rho;
  // excess = R^-1 - I for a pair
  double e00 = rho*rho/det, e01 = -rho/det, e11 = rho*rho/det;
  arma::mat scores((unsigned int)n, p, arma::fill::zeros);
  double dLogf[4], dCdf[4];
  for (size_t i = 0; i < n; ++i) {
    double u0 = scoreSaCdf(fam0, eta0[i], a0);
    double u1 = scoreSaCdf(fam1, eta1[i], a1);
    if (!std::isfinite(u0) || !std::isfinite(u1)) return false;
    u0 = std::min(std::max(u0, 1e-15), 1.0 - 1e-15);
    u1 = std::min(std::max(u1, 1e-15), 1.0 - 1e-15);
    double x0 = R::qnorm(u0, 0.0, 1.0, 1, 0), x1 = R::qnorm(u1, 0.0, 1.0, 1, 0);
    if (!std::isfinite(x0) || !std::isfinite(x1)) return false;
    // influence = -(R^-1 - I) xi
    double inf0 = -(e00*x0 + e01*x1), inf1 = -(e01*x0 + e11*x1);
    double d0 = R::dnorm(x0, 0.0, 1.0, 0), d1 = R::dnorm(x1, 0.0, 1.0, 0);
    if (!(d0 > 0) || !(d1 > 0)) return false;
    // Scores are formed in the SAME coordinates the step is taken in -- log
    // for a positive parameter, atanh for rho.  Forming them natively and
    // rescaling the DIRECTION afterwards is wrong: the information is then
    // native while the step is not, and the correct log step A^-1 I^-1 s comes
    // out as A I^-1 s -- an error of a^2, which for a shape of 8 is a step 64
    // times too long.  Measured before this fix: the recursion drove a shape
    // of truth 6 to 147.
    int m0 = rxEtaDistPosMask(fam0), m1 = rxEtaDistPosMask(fam1);
    scoreSaFamilyGrad(fam0, eta0[i], a0, na0, dLogf, dCdf);
    for (int j = 0; j < na0; ++j) {
      double v = dLogf[j] + inf0 * dCdf[j] / d0;
      scores(i, j) = (m0 & (1 << j)) ? v * a0[j] : v;
    }
    scoreSaFamilyGrad(fam1, eta1[i], a1, na1, dLogf, dCdf);
    for (int j = 0; j < na1; ++j) {
      double v = dLogf[j] + inf1 * dCdf[j] / d1;
      scores(i, na0 + j) = (m1 & (1 << j)) ? v * a1[j] : v;
    }
    // d/drho of log c, then chain rule to atanh(rho)
    double dLogDet = -2.0*rho/det;
    double q = (x0*x0 + x1*x1)*(2.0*rho*det + 2.0*rho*rho*rho)/(det*det)
      - 2.0*x0*x1*(det + 2.0*rho*rho)/(det*det);
    scores(i, p - 1) = -0.5*(dLogDet + q) * det;
  }
  if (!scores.is_finite()) return false;
  arma::vec meanScore = arma::mean(scores, 0).t();
  if (state.deltaSubject.n_rows != (unsigned int)n ||
      state.deltaSubject.n_cols != p) state.deltaSubject = scores;
  else state.deltaSubject = (1.0 - gain)*state.deltaSubject + gain*scores;
  arma::mat information = state.deltaSubject.t()*state.deltaSubject/(double)n;
  information = 0.5*(information + information.t());
  information.diag() += std::max(1e-10,
    ridge*arma::trace(information)/(double)p);
  arma::mat inverse;
  if (arma::inv_sympd(inverse, information)) {
    state.inverseInformation = inverse; state.ready = true;
  } else if (state.inverseInformation.n_rows != p) {
    state.inverseInformation = arma::eye(p, p);
  }
  arma::vec direction = state.inverseInformation * meanScore;
  if (!direction.is_finite()) return false;
  // unconstrained coordinates: log for a positive parameter, atanh for rho
  arma::vec current(p);
  int mask0 = rxEtaDistPosMask(fam0), mask1 = rxEtaDistPosMask(fam1);
  for (int j = 0; j < na0; ++j)
    current(j) = (mask0 & (1 << j)) ? std::log(a0[j]) : a0[j];
  for (int j = 0; j < na1; ++j)
    current(na0 + j) = (mask1 & (1 << j)) ? std::log(a1[j]) : a1[j];
  current(p - 1) = std::atanh(rho);
  if (!current.is_finite()) return false;
  // direction is already in the unconstrained coordinates: the scores were
  // formed there, so the information and the step agree.
  arma::vec proposal = current + gain*direction;
  if (!proposal.is_finite()) return false;
  if (state.centre.n_elem != p) {
    state.centre = current;
    state.width = arma::vec(p, arma::fill::value(boxWidth));
  }
  {
    arma::vec lo = state.centre - state.width, hi = state.centre + state.width;
    arma::vec pr = arma::min(hi, arma::max(lo, proposal));
    if (arma::any(arma::abs(pr - proposal) > 0)) {
      state.width *= 2.0; state.expansions++;
      proposal = state.centre; state.deltaSubject.reset();
      state.average.reset(); state.averageCount = 0; state.averaging = false;
    } else proposal = pr;
  }
  if (averagePhase) {
    if (!state.averaging || state.average.n_elem != p) {
      state.average = proposal; state.averageCount = 1; state.averaging = true;
    } else {
      state.averageCount++;
      state.average += (proposal - state.average)/(double)state.averageCount;
    }
  }
  for (int j = 0; j < na0; ++j)
    a0[j] = (mask0 & (1 << j)) ? std::exp(proposal(j)) : proposal(j);
  for (int j = 0; j < na1; ++j)
    a1[j] = (mask1 & (1 << j)) ? std::exp(proposal(na0+j)) : proposal(na0+j);
  rho = std::tanh(proposal(p - 1));
  state.iteration++;
  return true;
}

#endif // __SCORESA_H__
