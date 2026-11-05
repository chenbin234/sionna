#
# SPDX-License-Identifier: Apache-2.0
#
"""Evaluate analytical BLER benchmarks alongside Monte Carlo coding models."""

import math
import time
from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import ArrayLike

from sionna._validation import check_tensor
from sionna.phy import Block
from sionna.phy.utils import PlotBER, ebnodb2no

from .achievability import AWGNGallager, AWGNKappaBeta, AWGNShannon
from .achievability import utils as ach_utils
from .converse import AWGNMetaConverse
from .converse import utils as conv_utils
from .normapx import AWGNNormalApproximation
from .utils import TensorLike, real_awgn_capacity_dispersion

__all__ = ["AWGNBoundBLER", "FBLPlotBER"]


class AWGNBoundBLER(Block):
    r"""BLER benchmark at fixed payload for the notebook's unit-energy QPSK.

    Wraps a real-AWGN log2(M) evaluator. With k payload bits, n coded bits,
    and QPSK, there are n/2 complex symbols and n real channel uses. The real
    SNR is ``2*(k/n)*10**(ebno_db/10)``, or ``1/no`` with Sionna's
    ``ebnodb2no(ebno_db, 2, k/n)``. CRC bits belong to coding overhead and
    are not added to the payload k. No fading, pilots, or HARQ is included.

    :param k: Number of information bits, a positive integer <= n.
    :param n: Transmitted coded bits, a positive even integer.
    :param bound: One of the five AWGN evaluators, configured with
        precision="double" and device="cpu". Defaults to the equal-energy
        metaconverse, which applies to QPSK codewords.
    :param num_bits_per_symbol: Must be 2 (QPSK). Other constellations require
        their own channel-use and power accounting.
    :param error_probability_floor: Smallest resolved kappa-beta upper bound;
        default 1e-10. If already feasible there, return this conservative
        upper bound. This is a numerical resolution limit, not a plot limit.
    :param inversion_rtol: Relative epsilon tolerance for kappa-beta inversion,
        default 1e-4. Bisection in log epsilon keeps the feasible upper endpoint.

    :input ebno_db: Scalar or Tensor [...], finite Eb/N0 in dB.
    :output bler: Same shape, float64 CPU Tensor. An optimal-BLER lower bound
        for the converse, upper bound for achievability, or approximation.

    Shannon, Gallager, and normal approximation have direct fixed-M paths;
    the metaconverse inverts the Gaussian hypothesis test. Kappa-beta inverts
    log2(M) numerically. This CPU reference adapter does not support autograd
    or torch.compile. It does not produce transmitted/decoded bits or BER.

    The converse bounds apply to QPSK. Achievability concerns unrestricted
    real/spherical AWGN codebooks; it benchmarks the AWGN optimum and does
    not guarantee a QPSK coding scheme with the displayed BLER.
    """

    def __init__(
        self,
        k: int,
        n: int,
        bound: Block | None = None,
        num_bits_per_symbol: int = 2,
        error_probability_floor: float = 1e-10,
        inversion_rtol: float = 1e-4,
    ) -> None:
        super().__init__(precision="double", device="cpu")
        for name, value in (("k", k), ("n", n),
                            ("num_bits_per_symbol", num_bits_per_symbol)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 0 < k <= n or n % 2:
            raise ValueError("require 0 < k <= n and even n for QPSK")
        if num_bits_per_symbol != 2:
            raise ValueError("AWGNBoundBLER currently supports QPSK only")
        for name, option_value in (("error_probability_floor", error_probability_floor),
                                   ("inversion_rtol", inversion_rtol)):
            if isinstance(option_value, bool) or not isinstance(option_value, (int, float)):
                raise TypeError(f"{name} must be real")
        if not 1e-10 <= error_probability_floor < .1:
            raise ValueError("error_probability_floor must lie in [1e-10, 0.1)")
        if not 1e-8 <= inversion_rtol <= 1e-2:
            raise ValueError("inversion_rtol must lie in [1e-8, 1e-2]")
        if bound is None:
            bound = AWGNMetaConverse(power_constraint="equal", precision="double",
                                    device="cpu")
        if type(bound) not in (AWGNMetaConverse, AWGNNormalApproximation,
                               AWGNGallager, AWGNKappaBeta, AWGNShannon):
            raise TypeError("bound must be a supported real-AWGN evaluator")
        if bound.device != "cpu" or bound.precision != "double":
            raise ValueError("bound must use precision='double', device='cpu'")
        self._bound = bound
        self._k, self._n = k, n
        self._floor = error_probability_floor
        self._inversion_rtol = inversion_rtol

    @property
    def result_kind(self) -> str:
        """Direction of the result for optimal real-AWGN BLER."""
        if isinstance(self._bound, AWGNMetaConverse):
            return "lower bound"
        if isinstance(self._bound, AWGNNormalApproximation) or (
            isinstance(self._bound, AWGNKappaBeta)
            and self._bound.kappa_mode == "asymptotic"
        ):
            return "approximation"
        return "upper bound (AWGN)"

    def _kappa_beta_bler(self, snr: float) -> float:
        """Invert the configured size evaluator, preserving achievability."""
        def size(epsilon: float) -> float:
            return float(self._bound(
                self._n, torch.tensor(epsilon, dtype=self.dtype),
                torch.tensor(snr, dtype=self.dtype),
            ))

        lower, upper = math.log(self._floor), math.log1p(-1e-7)
        if size(math.exp(lower)) >= self._k:
            return self._floor
        if size(math.exp(upper)) < self._k:
            return 1.0
        for _ in range(100):
            middle = (lower + upper) / 2
            if size(math.exp(middle)) >= self._k:
                upper = middle
            else:
                lower = middle
            if upper - lower <= math.log1p(self._inversion_rtol):
                return math.exp(upper)
        raise RuntimeError("kappa-beta BLER inversion did not converge")

    def call(self, ebno_db: TensorLike) -> torch.Tensor:
        """Evaluate a deterministic BLER curve without Monte Carlo sampling."""
        value = torch.as_tensor(ebno_db, device=self.device)
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError("ebno_db must be real and non-boolean")
        if value.requires_grad:
            raise ValueError("AWGNBoundBLER does not support autograd")
        value = value.to(dtype=self.dtype)
        check_tensor(torch.all(torch.isfinite(value)), "ebno_db must be finite")
        power = 1 / ebnodb2no(value, num_bits_per_symbol=2,
                             coderate=self._k / self._n,
                             precision=self.precision, device=self.device)
        check_tensor(torch.all(torch.isfinite(power) & (power > 0)),
                     "Eb/N0 is outside the supported numerical SNR range")
        if isinstance(self._bound, AWGNNormalApproximation):
            capacity, dispersion = real_awgn_capacity_dispersion(power)
            correction = .5 * math.log2(self._n) if self._bound.third_order else 0
            standardized = ((self._k - self._n * capacity - correction)
                            / torch.sqrt(self._n * dispersion))
            # erfc preserves small lower-tail probabilities that ndtr's
            # implementation can lose through subtraction from one.
            probability: torch.Tensor = .5 * torch.special.erfc(-standardized / math.sqrt(2))
            return probability
        results = []
        for snr in power.reshape(-1):
            power_scalar = float(snr)
            if isinstance(self._bound, AWGNMetaConverse):
                result = conv_utils.awgn_metaconverse_error_probability(
                    self._k, self._n, power_scalar, self._bound.power_constraint)
            elif isinstance(self._bound, AWGNGallager):
                result = ach_utils.awgn_gallager_error_probability(
                    self._k, self._n, power_scalar)
            elif isinstance(self._bound, AWGNShannon):
                result = ach_utils.awgn_shannon_error_probability(
                    self._k, self._n, power_scalar, self._bound.rtol)
            else:
                result = self._kappa_beta_bler(power_scalar)
            if not math.isfinite(result) or not 0 <= result <= 1:
                raise RuntimeError("BLER evaluation produced an invalid probability")
            results.append(result)
        return torch.tensor(results, dtype=self.dtype).reshape(value.shape)


class FBLPlotBER(PlotBER):
    """PlotBER with analytical BLER models accepted by the same simulate method.

    Coding models delegate to PlotBER.simulate unchanged. AWGNBoundBLER models
    are evaluated once per SNR and added with is_bler=True. Monte Carlo/compile
    settings are accepted for a common calling pattern but unused for these
    deterministic models. Their BER and error/sample counts print as ``--``.
    They return ``(nan_ber, bler)``, with an undefined BER tensor filled with NaN.
    Coding models return the usual ``(ber, bler)`` tensors.
    """

    def simulate(
        self,
        mc_fun: Callable,
        ebno_dbs: ArrayLike | torch.Tensor,
        batch_size: int = 1,
        max_mc_iter: int = 1,
        legend: str = "",
        add_ber: bool | None = None,
        add_bler: bool | None = None,
        soft_estimates: bool = False,
        num_target_bit_errors: int | None = None,
        num_target_block_errors: int | None = None,
        target_ber: float | None = None,
        target_bler: float | None = None,
        early_stop: bool = True,
        compile_mode: str | None = None,
        add_results: bool = True,
        forward_keyboard_interrupt: bool = True,
        show_fig: bool = True,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate a coding model or evaluate a fixed-payload BLER benchmark.

        Keywords follow PlotBER.simulate. For analytical models, add_ber
        defaults to False and add_bler to True; requesting BER raises
        ValueError. ``verbose``, ``show_fig``, and ``add_results`` retain their
        ordinary meaning. Monte Carlo stopping/compile settings are unused.
        Re-evaluating the same analytical legend replaces its previous curve.
        The first returned tensor contains NaN for undefined analytical BER.
        """
        if not isinstance(mc_fun, AWGNBoundBLER):
            return super().simulate(
                mc_fun, ebno_dbs=torch.as_tensor(ebno_dbs), batch_size=batch_size,
                max_mc_iter=max_mc_iter, legend=legend,
                add_ber=True if add_ber is None else add_ber,
                add_bler=False if add_bler is None else add_bler,
                soft_estimates=soft_estimates,
                num_target_bit_errors=num_target_bit_errors,
                num_target_block_errors=num_target_block_errors,
                target_ber=target_ber, target_bler=target_bler,
                early_stop=early_stop, compile_mode=compile_mode,
                add_results=add_results,
                forward_keyboard_interrupt=forward_keyboard_interrupt,
                show_fig=show_fig, verbose=verbose,
            )
        if add_ber:
            raise ValueError("finite-blocklength benchmarks provide BLER only")
        if not isinstance(legend, str):
            raise TypeError("legend must be str")
        if isinstance(ebno_dbs, torch.Tensor):
            if ebno_dbs.device.type != "cpu" or ebno_dbs.requires_grad:
                raise ValueError("analytical ebno_dbs must be a CPU tensor without gradients")
            ebno_dbs = ebno_dbs.numpy()
        points = np.asarray(ebno_dbs)
        if points.ndim != 1 or not len(points):
            raise ValueError("ebno_dbs must be a nonempty one-dimensional sequence")
        if points.dtype.kind not in "iuf" or not np.all(np.isfinite(points)):
            raise ValueError("ebno_dbs must contain finite real values")
        points = points.astype(np.float64)
        row_format = ("{:>9} | {:>10} | {:>10} | {:>11} | {:>11} | "
                      "{:>12} | {:>11} | {:>11} | {}")
        if verbose:
            print(row_format.format("EbNo [dB]", "BER", "BLER", "bit errors",
                                    "num bits", "block errors", "num blocks",
                                    "runtime [s]", "status"))
            print("-" * 151)
        results = []
        for point in points:
            start = time.perf_counter()
            bler = float(mc_fun(torch.tensor(point, dtype=torch.float64)))
            results.append(bler)
            if verbose:
                print(row_format.format(f"{point:.1f}", "--", f"{bler:.4e}",
                                        "--", "--", "--", "--",
                                        f"{time.perf_counter() - start:.3f}",
                                        mc_fun.result_kind))
        result = torch.tensor(results, dtype=torch.float64)
        if add_results and add_bler is not False:
            label = legend + " (BLER)"
            for index in reversed(range(len(self.legend))):
                if self.legend[index] == label and self.is_bler[index]:
                    self.remove(index)
            self.add(points, result.numpy(), is_bler=True, legend=label)
        if show_fig:
            self(show_ber=False)
        return torch.full_like(result, math.nan), result
