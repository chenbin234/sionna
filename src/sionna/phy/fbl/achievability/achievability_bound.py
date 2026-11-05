#
# SPDX-License-Identifier: Apache-2.0
#
"""Public finite-blocklength achievability evaluators."""


import torch

from sionna.phy import Block
from sionna.phy.config import Precision

from .. import utils as shared_utils
from . import utils as ach_utils

__all__ = ["AWGNGallager", "AWGNKappaBeta", "AWGNShannon"]


class AWGNKappaBeta(Block):
    r"""Kappa-beta lower bound on log2(M) for real AWGN and maximal error.

    Implements Spectre Ch. 2 ``kappabeta_ach`` / PPV (2010), Eq. (218), with
    Q=N(0,1+snr)^n. Codewords lie on the sphere ||x||^2=n*snr, hence also
    satisfy the per-codeword power inequality. Noise is iid N(0,1); no fading
    or CSI. Finite-n kappa uses a radial Neyman-Pearson test.

    :param kappa_mode: "exact" (default) uses finite-n kappa. "asymptotic"
        uses Spectre's kappa_inf and is an approximation, not a proven bound.
    :param rtol: Relative quadrature/optimization tolerance, default 1e-7.
        Tau uses a feasible grid followed by local refinement; the global
        optimum is not guaranteed, which can only loosen exact achievability.
    :param precision: Output precision; None uses Sionna's configured default.
    :param device: Must resolve to "cpu"; None uses the Sionna default.

    :input blocklength: [...], real scalar or Tensor. Positive integers.
    :input error_probability: [...], real scalar or Tensor. Strictly in (0,1).
    :input snr: [...], real scalar or Tensor. Nonnegative linear SNR.
    :output log2_codebook_size: Broadcast input shape, Tensor, lower bound in
        bits. Divide by blocklength for bits per real use. Round down to an
        integer codebook size, with M=1 as a fallback (also at zero SNR).

    SciPy reference computation uses float64 on CPU. No autograd or
    torch.compile support. Use double input tensors and precision="double"
    for reference comparisons. Numerical quadrature is not interval-certified.
    """

    def __init__(
        self,
        kappa_mode: str = "exact",
        rtol: float = 1e-7,
        precision: Precision | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        shared_utils.validate_cpu_device(self.device)
        shared_utils.validate_tolerance(rtol)
        if kappa_mode not in ("exact", "asymptotic"):
            raise ValueError("kappa_mode must be 'exact' or 'asymptotic'")
        self._kappa_mode = kappa_mode
        self._rtol = rtol

    @property
    def kappa_mode(self) -> str:
        """Finite-n or asymptotic kappa evaluation."""
        return self._kappa_mode

    def call(
        self,
        blocklength: shared_utils.TensorLike,
        error_probability: shared_utils.TensorLike,
        snr: shared_utils.TensorLike,
    ) -> torch.Tensor:
        """Evaluate a codebook-size lower bound or the selected approximation."""
        n, epsilon, power = shared_utils.prepare_inputs(
            blocklength, error_probability, snr, self.dtype, self.device)
        return shared_utils.evaluate_cpu(
            lambda ni, ei, pi: ach_utils.awgn_kappa_beta_log2_codebook_size(
                ni, ei, pi, self._kappa_mode, self._rtol), n, epsilon, power)


class AWGNGallager(Block):
    r"""Gallager lower bound on log2(M) for real AWGN and maximal error.

    Uses the cost-constrained Gaussian random-coding exponent in Spectre
    Ch. 2 and PPV (2010), Eq. (44), with the theorem's squared prefactor.
    Noise is iid N(0,1) and every codeword satisfies ||x||^2 <= n*snr.
    The input ensemble is Gaussian conditioned on a narrow energy shell.
    No fading or CSI. See README.md for the discrepancy with Spectre's
    unsquared prefactor.

    :param rtol: Absolute stopping tolerance in n*rate (nats), default 1e-7.
        Inversion retains the feasible lower endpoint. Shell width uses the
        feasible choice min(1/s,n), rather than a global optimization.
    :param precision: Output precision; None uses the Sionna default.
    :param device: Must resolve to "cpu"; None uses the Sionna default.

    :input blocklength: [...], real scalar or Tensor. Positive integers.
    :input error_probability: [...], real scalar or Tensor. Strictly in (0,1).
    :input snr: [...], real scalar or Tensor. Nonnegative linear SNR.
    :output log2_codebook_size: Broadcast input shape, Tensor, lower bound in
        bits. Divide by blocklength for bits per real use. Rounded down to an
        integer codebook size, with M=1 as a fallback (also at zero SNR).

    Float64 SciPy reference computation on CPU; no autograd or torch.compile.
    This evaluates a theorem numerically, without interval certification.
    """

    def __init__(
        self,
        rtol: float = 1e-7,
        precision: Precision | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        shared_utils.validate_cpu_device(self.device)
        shared_utils.validate_tolerance(rtol)
        self._rtol = rtol

    def call(
        self,
        blocklength: shared_utils.TensorLike,
        error_probability: shared_utils.TensorLike,
        snr: shared_utils.TensorLike,
    ) -> torch.Tensor:
        """Invert the maximal-error bound for each broadcast parameter tuple."""
        n, epsilon, power = shared_utils.prepare_inputs(
            blocklength, error_probability, snr, self.dtype, self.device)
        return shared_utils.evaluate_cpu(
            lambda ni, ei, pi: ach_utils.awgn_gallager_log2_codebook_size(
                ni, ei, pi, self._rtol), n, epsilon, power)


class AWGNShannon(Block):
    r"""Shannon cone-packing lower bound on log2(M), with maximal error.

    Uses the spherical ensemble in PPV (2010), Eqs. (41)-(42), and expurgates
    an average-error code at tau*epsilon to maximal error epsilon, Eq. (220).
    Noise is iid N(0,1), codewords are real and have energy exactly n*snr.
    They also satisfy the per-codeword power inequality. No fading or CSI.
    Corresponds to Spectre Ch. 2 ``shannon_ach2``, using exact cap integrals
    and a corrected expurgation target rather than its cap estimates.

    :param rtol: Relative quadrature/optimization tolerance, default 1e-6.
        A grid and local refinement choose a feasible expurgation fraction.
        The angle inversion retains the feasible endpoint. No claim of a
        global optimum or interval certification is made.
    :param precision: Output precision; None uses the Sionna default.
    :param device: Must resolve to "cpu"; None uses the Sionna default.

    :input blocklength: [...], real scalar or Tensor. Integers >= 2.
    :input error_probability: [...], real scalar or Tensor. Strictly in (0,1).
    :input snr: [...], real scalar or Tensor. Nonnegative linear SNR.
    :output log2_codebook_size: Broadcast input shape, Tensor, lower bound in
        bits. Divide by blocklength for bits per real use. Rounded down to an
        integer codebook size, with M=1 as a fallback (also at zero SNR).

    Float64 SciPy quadrature on CPU; no autograd or torch.compile support.
    This is the slowest evaluator. Batched inputs run independent scalar solves.
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        precision: Precision | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        shared_utils.validate_cpu_device(self.device)
        shared_utils.validate_tolerance(rtol)
        self._rtol = rtol

    @property
    def rtol(self) -> float:
        """Relative quadrature and optimization tolerance."""
        return self._rtol

    def call(
        self,
        blocklength: shared_utils.TensorLike,
        error_probability: shared_utils.TensorLike,
        snr: shared_utils.TensorLike,
    ) -> torch.Tensor:
        """Evaluate Shannon's cone bound with maximal-error expurgation."""
        n, epsilon, power = shared_utils.prepare_inputs(
            blocklength, error_probability, snr, self.dtype, self.device,
            minimum_blocklength=2)
        return shared_utils.evaluate_cpu(
            lambda ni, ei, pi: ach_utils.awgn_shannon_log2_codebook_size(
                ni, ei, pi, self._rtol), n, epsilon, power)
