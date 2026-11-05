#
# SPDX-License-Identifier: Apache-2.0
#
"""Channel-specific converse kernels."""

import math

from scipy import optimize, stats

from ..utils import awgn_log_beta, ncx2_logcdf


def awgn_metaconverse_log2_codebook_size(
    blocklength: int,
    error_probability: float,
    snr: float,
    power_constraint: str,
) -> float:
    """Evaluate the Gaussian auxiliary-output metaconverse (PPV Eq. 218)."""
    # Spectre's n+1 embedding converts a maximal-power code into an
    # equal-energy code. Keeping P unchanged is valid, though slightly loose.
    dimension = blocklength + (power_constraint == "maximal")
    return -awgn_log_beta(error_probability, dimension, snr) / math.log(2)


def awgn_metaconverse_error_probability(
    payload_bits: int, blocklength: int, snr: float, power_constraint: str,
) -> float:
    """Invert beta directly at M=2**payload_bits to lower-bound optimal BLER."""
    if snr == 0:
        return -math.expm1(-payload_bits * math.log(2))
    dimension = blocklength + (power_constraint == "maximal")
    noncentrality = dimension * (1 + 1 / snr)
    target = -payload_bits * math.log(2)
    threshold = float(stats.ncx2.ppf(math.exp(target), dimension, noncentrality))
    if threshold == 0 or not math.isfinite(threshold):
        # Invert log CDFs when 1/M cannot be represented as a float.
        upper = math.log(dimension + noncentrality)
        lower = upper - 1
        while ncx2_logcdf(math.exp(lower), dimension, noncentrality) > target:
            lower -= 1
        threshold = math.exp(optimize.brentq(
            lambda value: ncx2_logcdf(math.exp(value), dimension, noncentrality)
            - target, lower, upper,
        ))
    result = float(stats.ncx2.sf((1 + snr) * threshold,
                                dimension, dimension / snr))
    if not math.isfinite(result):
        raise RuntimeError("metaconverse BLER evaluation failed")
    return result
