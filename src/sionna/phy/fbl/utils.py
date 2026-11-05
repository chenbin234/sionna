#
# SPDX-License-Identifier: Apache-2.0
#
"""Shared finite-blocklength statistics, validation, and numerical kernels."""

import math
from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from scipy import special, stats

from sionna._validation import check_tensor

__all__ = ["real_awgn_capacity_dispersion"]

TensorLike = int | float | torch.Tensor


def prepare_inputs(
    blocklength: TensorLike,
    error_probability: TensorLike,
    snr: TensorLike,
    dtype: torch.dtype,
    device: str,
    *,
    minimum_blocklength: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate numerical inputs and broadcast parameter batches."""
    values = []
    for name, value in zip(
        ("blocklength", "error_probability", "snr"),
        (blocklength, error_probability, snr),
    ):
        if not isinstance(value, (int, float, torch.Tensor)):
            raise TypeError(f"{name} must be a real scalar or torch.Tensor")
        tensor = torch.as_tensor(value, device=device)
        if tensor.is_complex() or tensor.dtype == torch.bool:
            raise ValueError(f"{name} must have a real, non-boolean dtype")
        tensor = tensor.to(dtype=dtype)
        check_tensor(torch.all(torch.isfinite(tensor)), f"{name} must be finite")
        values.append(tensor)
    n, epsilon, power = values
    check_tensor(
        torch.all((n >= minimum_blocklength) & (n == torch.floor(n))),
        f"blocklength must contain integers >= {minimum_blocklength}",
    )
    check_tensor(
        torch.all((epsilon > 0) & (epsilon < 1)),
        "error_probability must lie strictly between zero and one",
    )
    check_tensor(torch.all(power >= 0), "snr must be nonnegative (linear units)")
    try:
        n, epsilon, power = torch.broadcast_tensors(n, epsilon, power)
    except RuntimeError as exc:
        raise ValueError("input shapes must be broadcastable") from exc
    return n, epsilon, power


def real_awgn_capacity_dispersion(
    snr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return capacity and dispersion for real AWGN with unit noise variance.

    The constraint is :math:`\|x^n\|^2 \le n\,\mathrm{snr}`. The formulas
    are :math:`C=\log_2(1+P)/2` and
    :math:`V=P(P+2)/(2(P+1)^2)(\log_2 e)^2` (PPV, 2010).

    :param snr: Nonnegative finite floating tensor, in linear units.
    :output capacity: Same shape, dtype, and device; bits per real channel use.
    :output dispersion: Same shape, dtype, and device; bits squared per use.

    Supports autograd for positive SNR and PyTorch tensor execution.
    """
    if not isinstance(snr, torch.Tensor):
        raise TypeError("snr must be a torch.Tensor")
    if not snr.is_floating_point():
        raise ValueError("snr must have a floating dtype")
    check_tensor(torch.all(torch.isfinite(snr) & (snr >= 0)),
                 "snr must be finite and nonnegative")
    capacity = 0.5 * torch.log1p(snr) / math.log(2)
    # The factored expression avoids overflow in P*(P+2).
    dispersion = (0.5 * (snr / (1 + snr)) * (1 + 1 / (1 + snr))
                  / math.log(2)**2)
    return capacity, dispersion


def evaluate_cpu(
    kernel: Callable[[int, float, float], float],
    n: torch.Tensor,
    epsilon: torch.Tensor,
    snr: torch.Tensor,
) -> torch.Tensor:
    """Apply a double-precision SciPy kernel without detaching gradients."""
    for value in (n, epsilon, snr):
        if value.device.type != "cpu":
            raise ValueError("this numerical bound requires device='cpu'")
        if value.requires_grad:
            raise ValueError("SciPy bounds do not support autograd")
    result = [kernel(int(ni), float(ei), float(pi))
              for ni, ei, pi in zip(n.reshape(-1), epsilon.reshape(-1),
                                   snr.reshape(-1))]
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("bound evaluation produced a non-finite result")
    return torch.tensor(result, dtype=n.dtype, device=n.device).reshape(n.shape)


def validate_cpu_device(device: str) -> None:
    """Reject unsupported numerical backends at construction time."""
    if torch.device(device).type != "cpu":
        raise ValueError("this SciPy reference bound requires device='cpu'")


def validate_tolerance(value: float) -> None:
    """Check a relative numerical tolerance."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("rtol must be a real number")
    if not math.isfinite(value) or not 1e-12 <= value <= 1e-2:
        raise ValueError("rtol must lie in [1e-12, 1e-2]")


def _log_regularized_gamma(
    a: NDArray[np.float64], x: float,
) -> NDArray[np.float64]:
    """Evaluate log P(a,x), including lower tails below floating-point range."""
    with np.errstate(divide="ignore"):
        result = np.asarray(np.log(special.gammainc(a, x)), dtype=np.float64)
    mask = ~np.isfinite(result)
    if np.any(mask):
        shape = a[mask]
        # Positive lower incomplete-gamma series, used only in the lower tail.
        term = np.ones_like(shape)
        total = term.copy()
        for index in range(1, 100000):
            term *= x / (shape + index)
            total += term
            if np.all(term <= total * 2e-15):
                break
        else:
            raise RuntimeError("lower incomplete-gamma series did not converge")
        result[mask] = (shape * math.log(x) - x
                        - special.gammaln(shape + 1) + np.log(total))
    return result


def ncx2_logcdf(x: float, df: int, noncentrality: float) -> float:
    """Noncentral chi-square log CDF with a positive Poisson-series fallback.

    SciPy's logcdf can underflow before taking the logarithm. Sum the mixture
    in the log domain around its largest terms in that case. Truncation uses
    a relative tail threshold, not an asymptotic replacement of the bound.
    """
    value = float(stats.ncx2.logcdf(x, df, noncentrality))
    if math.isfinite(value):
        return value
    if x <= 0 or not math.isfinite(x):
        raise RuntimeError("invalid noncentral chi-square threshold")
    if noncentrality == 0:
        return float(_log_regularized_gamma(np.array([df / 2]), x / 2)[0])
    mu, half_x, shape = noncentrality / 2, x / 2, df / 2
    center = int(2 * mu * half_x
                 / (shape + math.sqrt(shape**2 + 4 * mu * half_x)))
    width = max(32, int(12 * math.sqrt(center + 1)))
    for _ in range(20):
        lo, hi = max(0, center - width), center + width
        indices = np.arange(lo, hi + 1, dtype=np.float64)
        terms = (-mu + indices * math.log(mu) - special.gammaln(indices + 1)
                 + _log_regularized_gamma(shape + indices, half_x))
        peak = float(np.max(terms))
        if (lo == 0 or terms[0] < peak - 40) and terms[-1] < peak - 40:
            return float(special.logsumexp(terms))
        width *= 2
        if width > 1000000:
            break
    raise RuntimeError("noncentral chi-square log-CDF series did not converge")


def awgn_log_beta(error_probability: float, n: int, snr: float) -> float:
    r"""Return ln beta_(1-epsilon) for a shell word versus N(0,1+P)^n."""
    if snr == 0:
        return math.log1p(-error_probability)
    # isf avoids rounding 1-epsilon to one for rare errors.
    threshold = float(stats.ncx2.isf(error_probability, n, n / snr))
    if not math.isfinite(threshold):
        raise RuntimeError("noncentral chi-square quantile did not converge")
    return ncx2_logcdf(threshold / (1 + snr), n, n * (1 + 1 / snr))


def achievable_integer_log2(log_size: float) -> float:
    """Round a continuous codebook guarantee down, with M=1 as a fallback."""
    if log_size <= 0:
        return 0.0
    if log_size < 52:
        return math.log2(max(1, math.floor(2**log_size)))
    # At large sizes the subtraction of one message is below one output ulp.
    return float(np.nextafter(log_size, -math.inf))
