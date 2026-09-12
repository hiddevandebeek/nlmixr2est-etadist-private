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

#endif // __SCORESA_H__
