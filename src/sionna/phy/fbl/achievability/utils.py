#
# SPDX-License-Identifier: Apache-2.0
#
"""Real-AWGN achievability mathematics (Spectre Ch. 2 and PPV, 2010).

These are equation-based implementations. In particular, no modified MATLAB
noncentral-t implementation is included; special functions come from SciPy.
"""

import math
from collections.abc import Callable

import numpy as np
from scipy import integrate, optimize, special, stats

from ..utils import achievable_integer_log2, awgn_log_beta


def _maximize_unit_interval(
    objective: Callable[[float], float], rtol: float,
) -> float:
    """Keep the best feasible grid/refinement value; global optimality unclaimed."""
    grid = np.linspace(0.01, 0.99, 17)
    values = [objective(float(value)) for value in grid]
    index = int(np.argmax(values))
    lower = float(grid[max(0, index - 1)])
    upper = float(grid[min(len(grid) - 1, index + 1)])
    result = optimize.minimize_scalar(
        lambda value: -objective(value), bounds=(lower, upper),
        method="bounded", options={"xatol": rtol},
    )
    if not result.success:
        raise RuntimeError("achievability optimization did not converge")
    return max(max(values), -float(result.fun))


# Real AWGN: kappa-beta

def awgn_kappa(tau: float, n: int, snr: float, rtol: float) -> float:
    """Radial Neyman-Pearson test for finite-n kappa, not its CLT limit.

    P_R = chi2_n(n*snr) and Q_R = (1+snr)*chi2_n. The likelihood ratio is
    unimodal. Write its acceptance interval as [F_P^-1(u), F_P^-1(u+tau)]
    and match likelihood ratios at the two boundaries. This also handles an
    interval touching zero, unlike a fixed radius grid.
    """
    if snr == 0:
        return tau
    radial_p = stats.ncx2(n, n * snr)
    radial_q = stats.chi2(n, scale=1 + snr)

    def interval(u: float) -> tuple[float, float]:
        return float(radial_p.ppf(u)), float(radial_p.ppf(u + tau))

    def log_ratio(radius: float) -> float:
        if radius == 0:
            return n / 2 * (math.log1p(snr) - snr)
        if math.isinf(radius):
            return -math.inf
        return float(radial_p.logpdf(radius) - radial_q.logpdf(radius))

    def difference(u: float) -> float:
        left, right = interval(u)
        return log_ratio(left) - log_ratio(right)

    if difference(0.0) >= 0:
        lower_mass = 0.0
    else:
        lower_mass = optimize.brentq(
            difference, 0.0, 1 - tau, xtol=1e-14,
            rtol=max(4 * np.finfo(float).eps, rtol * 0.01),
        )
    left, right = interval(lower_mass)
    if left == 0:
        return float(radial_q.cdf(right))
    # Integrate in an affine coordinate to avoid subtracting almost equal
    # CDFs for small tau. The interval is short precisely in that regime.
    width = right - left
    result, error = integrate.quad(
        lambda u: float(radial_q.pdf(left + width * u)) * width,
        0, 1, epsabs=0, epsrel=rtol, limit=200,
    )
    if result <= 0 or not math.isfinite(result) or error > rtol * result * 10:
        raise RuntimeError("finite-n kappa quadrature failed to converge")
    return float(result)


def awgn_kappa_beta_log2_codebook_size(
    n: int, epsilon: float, snr: float, kappa_mode: str, rtol: float,
) -> float:
    """Optimize kappa_tau / beta_(1-epsilon+tau), then round down."""
    if snr == 0:
        return 0.0

    def objective(fraction: float) -> float:
        tau = epsilon * fraction
        if kappa_mode == "exact":
            kappa = awgn_kappa(tau, n, snr, rtol)
        else:
            # erf/erfinv avoid cancellation in 2*Phi(x)-1 at small tau.
            scale = math.sqrt(1 + 2 * snr) / (1 + snr)
            kappa = float(special.erf(scale * special.erfinv(tau)))
        if kappa <= 0 or not math.isfinite(kappa):
            raise RuntimeError("kappa could not be resolved numerically")
        return (math.log(kappa) - awgn_log_beta(epsilon - tau, n, snr)) / math.log(2)

    return achievable_integer_log2(_maximize_unit_interval(objective, rtol))


# Real AWGN: Gallager's cost-constrained random coding

def awgn_gallager_log_error(n: int, rate: float, snr: float) -> float:
    """Gallager's maximal-error bound, natural logarithm, rate in nats/use.

    PPV Eq. (44) has the squared prefactor (2*exp(s*delta)/mu)^2.
    Spectre's gallager_ach.m omits this square. Use the theorem's conservative
    prefactor and Spectre's feasible delta=min(1/s,n) choice.
    """
    critical_beta = 0.5 * (1 + snr / 2 + math.hypot(1, snr / 2))
    critical_rate = 0.5 * math.log(critical_beta)
    beta = math.exp(2 * max(rate, critical_rate))
    if rate >= critical_rate:
        root = math.sqrt(1 + 4 * beta / (snr * (beta - 1)))
        rho = snr / (2 * beta) * (1 + root) - 1
        # Rationalizing root-1 avoids cancellation at large SNR.
        exponent = (snr / (2 * beta) - 1 / (root + 1)
                    + 0.5 * math.log(beta * (1 - 2 / (root + 1))))
    else:
        rho = 1.0
        # beta-P/2 is evaluated without subtraction of large close numbers.
        beta_minus_half_power = 0.5 * (1 + 1 / (math.hypot(1, snr / 2) + snr / 2))
        exponent = (1 - beta + snr / 2
                    + 0.5 * math.log(beta_minus_half_power)
                    + 0.5 * math.log(beta) - rate)
    s = max(0.0, rho) * snr / (2 * (1 + rho)**2 * beta)
    delta = min(n, 1 / s) if s > 0 else float(n)
    mu = float(stats.chi2.cdf(n, n) - stats.chi2.cdf(n - delta, n))
    if mu <= 0:
        raise RuntimeError("Gallager shell probability could not be resolved")
    return 2 * (math.log(2) + s * delta - math.log(mu)) - n * exponent


def awgn_gallager_log2_codebook_size(
    n: int, epsilon: float, snr: float, rtol: float,
) -> float:
    """Invert Gallager's error bound, retaining the feasible rate endpoint."""
    if snr == 0:
        return 0.0
    target = math.log(epsilon)
    if awgn_gallager_log_error(n, 0.0, snr) > target:
        return 0.0
    lower, upper = 0.0, 0.5 * math.log1p(snr)
    for _ in range(200):
        middle = (lower + upper) / 2
        if awgn_gallager_log_error(n, middle, snr) <= target:
            lower = middle
        else:
            upper = middle
        if n * (upper - lower) <= rtol:
            return achievable_integer_log2(n * lower / math.log(2))
    raise RuntimeError("Gallager rate inversion did not converge")


# Real AWGN: Shannon's cone-packing bound

def awgn_shannon_cone(
    n: int, snr: float, theta: float, rtol: float, error_scale: float,
) -> tuple[float, float]:
    """Return average-error bound and log spherical cap fraction (PPV Eq. 42).

    Substitute u=(sin(phi)/sin(theta))^(n-1), so small cap areas never need
    to be represented in probability space. The error is a ratio of scaled
    integrals. q_n(phi) is a noncentral-t CDF with n-1 degrees of freedom.
    """
    sine = math.sin(theta)

    def weight(u: float) -> tuple[float, float]:
        sin_phi = sine * u**(1 / (n - 1))
        cos_phi = math.sqrt(max(np.finfo(float).tiny, 1 - sin_phi**2))
        return sin_phi, cos_phi

    denominator, denominator_error = integrate.quad(
        lambda u: 1 / weight(u)[1], 0, 1,
        epsabs=0, epsrel=rtol, limit=200,
    )

    def integrand(u: float) -> float:
        sin_phi, cos_phi = weight(u)
        threshold = math.sqrt(n - 1) * cos_phi / sin_phi
        return float(stats.nct.cdf(threshold, n - 1, math.sqrt(n * snr))) / cos_phi

    numerator, error = integrate.quad(
        integrand, 0, 1, epsabs=error_scale * rtol * 0.1,
        epsrel=rtol, limit=300,
    )
    if (denominator_error > 10 * rtol * denominator
            or error > 10 * rtol * max(numerator, error_scale)):
        raise RuntimeError("Shannon cone quadrature did not converge")
    log_cap = (special.gammaln(n / 2) - special.gammaln((n - 1) / 2)
               - 0.5 * math.log(math.pi) + (n - 1) * math.log(sine)
               - math.log(n - 1) + math.log(denominator))
    return numerator / denominator, float(log_cap)


def awgn_shannon_log2_codebook_size(
    n: int, epsilon: float, snr: float, rtol: float,
) -> float:
    """Shannon cone bound with average-to-maximal-error expurgation."""
    if snr == 0:
        return 0.0

    def objective(fraction: float) -> float:
        target = fraction * epsilon
        lower, upper = 1e-12, math.pi / 2 - 1e-7
        error, _ = awgn_shannon_cone(n, snr, upper, rtol, target)
        if error > target:
            return 0.0
        log_cap = 0.0
        for _ in range(100):
            theta = (lower + upper) / 2
            error, candidate = awgn_shannon_cone(n, snr, theta, rtol, target)
            if error <= target:
                upper, log_cap = theta, candidate
            else:
                lower = theta
            if upper - lower <= rtol * 0.01 / n:
                break
        else:
            raise RuntimeError("Shannon angle inversion did not converge")
        # First round the average-error code size, then expurgate. Using
        # epsilon here instead of fraction*epsilon would invalidate this step.
        average_log_size = achievable_integer_log2(-log_cap / math.log(2))
        return average_log_size + math.log1p(-fraction) / math.log(2)

    return achievable_integer_log2(_maximize_unit_interval(objective, rtol))


def awgn_gallager_error_probability(
    payload_bits: int, n: int, snr: float,
) -> float:
    """Upper-bound BLER at a fixed payload using Gallager's random-coding law.

    Inputs are the number of information bits, real channel uses, and linear
    real-AWGN SNR. The output is a probability in [0,1]. This is an achievable
    benchmark for unrestricted real-AWGN codes, rather than a guarantee for
    a particular modulation or decoder. Public evaluators validate inputs.
    """
    if payload_bits == 0:
        return 0.0
    if snr == 0:
        return 1.0
    rate = payload_bits * math.log(2) / n
    if rate >= 0.5 * math.log1p(snr):
        return 1.0
    return math.exp(min(0.0, awgn_gallager_log_error(n, rate, snr)))


def _awgn_cone_angle_for_codebook_size(
    n: int, codebook_size: int, rtol: float,
) -> float:
    """Invert a spherical cap fraction without evaluating the channel law."""
    # The cap fraction is 0.5*I_(sin(theta)^2)((n-1)/2, 1/2).
    # Invert directly whenever its probability is representable. Moving the
    # fraction and angle down makes the codebook guarantee conservative.
    probability = float(2 / codebook_size)
    if probability > 0:
        probability = float(np.nextafter(probability, 0.0))
        square_sine = float(special.betaincinv((n - 1) / 2, 0.5, probability))
        if 0 < square_sine < 1:
            theta = math.asin(math.sqrt(square_sine))
            return float(np.nextafter(theta, 0.0))

    # For an underflowed cap area, factor out sin(theta)^(n-1) and use the
    # hypergeometric representation of the incomplete beta integral in logs.
    target = -math.log(codebook_size)
    normalization = (special.gammaln(n / 2) - special.gammaln((n - 1) / 2)
                     - 0.5 * math.log(math.pi) - math.log(n - 1))

    def log_cap(theta: float) -> float:
        sine = math.sin(theta)
        factor = float(special.hyp2f1(
            (n - 1) / 2, 0.5, (n + 1) / 2, sine**2))
        if factor <= 0 or not math.isfinite(factor):
            raise RuntimeError("spherical-cap special function did not converge")
        return float(normalization + (n - 1) * math.log(sine) + math.log(factor))

    lower, upper = 0.0, math.pi / 2
    for _ in range(200):
        middle = (lower + upper) / 2
        if log_cap(middle) <= target:
            lower = middle
        else:
            upper = middle
        if lower > 0 and upper - lower <= rtol * 0.01 * lower:
            return lower
    raise RuntimeError("spherical-cap angle inversion did not converge")


def awgn_shannon_error_probability(
    payload_bits: int, n: int, snr: float, rtol: float,
) -> float:
    """Upper-bound BLER with Shannon's fixed-M cone bound and expurgation.

    For an expurgation fraction tau, choose an average-error code with
    M_avg=ceil(2**payload_bits/(1-tau)) messages. Evaluate its cone bound at
    cap fraction 1/M_avg, then divide its average-error bound by tau. A grid
    and local refinement minimize this feasible maximal-error guarantee.
    Inputs use real channel uses and linear SNR; the output lies in [0,1].
    This directly evaluates the fixed-message theorem and avoids repeatedly
    inverting the much slower log-codebook-size evaluator. Public evaluators
    validate inputs; n must be at least two.
    """
    if payload_bits == 0:
        return 0.0
    if snr == 0:
        return 1.0
    messages = 1 << payload_bits

    def objective(fraction: float) -> float:
        # Integer arithmetic keeps the ceiling exact even above 53 bits.
        numerator, denominator = fraction.as_integer_ratio()
        retained_numerator = denominator - numerator
        average_messages = (messages * denominator + retained_numerator - 1) // retained_numerator
        theta = _awgn_cone_angle_for_codebook_size(n, average_messages, rtol)
        error, _ = awgn_shannon_cone(
            n, snr, theta, rtol, np.finfo(float).tiny)
        if not math.isfinite(error) or error < 0:
            raise RuntimeError("Shannon error quadrature produced an invalid probability")
        return -min(1.0, error / fraction)

    return -_maximize_unit_interval(objective, rtol)
