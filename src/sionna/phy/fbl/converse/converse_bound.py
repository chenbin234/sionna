#
# SPDX-License-Identifier: Apache-2.0
#
"""Public finite-blocklength converse evaluators."""


import torch

from sionna.phy import Block
from sionna.phy.config import Precision

from .. import utils as shared_utils
from . import utils as conv_utils

__all__ = ["AWGNMetaConverse"]


class AWGNMetaConverse(Block):
    r"""Upper bound on log2(M) for real AWGN, for average or maximal error.

    Implements the Gaussian auxiliary-output hypothesis test in Spectre Ch. 2
    ``converse`` and PPV (2010), Eq. (218). Noise is iid N(0,1), and
    Q=N(0,1+snr)^d. ``maximal`` means a per-codeword energy inequality
    ||x||^2 <= n*snr and uses d=n+1 as in Spectre. ``equal`` requires every
    word to have energy exactly n*snr and uses d=n. The latter is not the
    converse for arbitrary words inside the power ball. No fading or CSI.

    :param power_constraint: "maximal" (default) or "equal".
    :param precision: Sionna output precision; None uses the configured default.
    :param device: Must resolve to "cpu"; None uses the Sionna default.

    :input blocklength: [...], real scalar or Tensor. Positive integers.
    :input error_probability: [...], real scalar or Tensor. Strictly in (0,1).
    :input snr: [...], real scalar or Tensor. Nonnegative linear SNR.
    :output log2_codebook_size: Broadcast input shape, Tensor, upper bound in
        bits. Divide by the original blocklength for bits per real use.

    SciPy computes in float64 on CPU, including a log-domain tail series.
    No autograd or torch.compile support. This is a numerical evaluation of a
    theorem, not an interval-certified bound. At zero SNR it gives
    -log2(1-epsilon). Output is not rounded to an integer codebook size.
    """

    def __init__(
        self,
        power_constraint: str = "maximal",
        precision: Precision | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        shared_utils.validate_cpu_device(self.device)
        if power_constraint not in ("maximal", "equal"):
            raise ValueError("power_constraint must be 'maximal' or 'equal'")
        self._power_constraint = power_constraint

    @property
    def power_constraint(self) -> str:
        """Per-codeword power constraint used by the hypothesis test."""
        return self._power_constraint

    def call(
        self,
        blocklength: shared_utils.TensorLike,
        error_probability: shared_utils.TensorLike,
        snr: shared_utils.TensorLike,
    ) -> torch.Tensor:
        """Evaluate the rate converse in log2 codebook-size units."""
        n, epsilon, power = shared_utils.prepare_inputs(
            blocklength, error_probability, snr, self.dtype, self.device)
        return shared_utils.evaluate_cpu(
            lambda ni, ei, pi: conv_utils.awgn_metaconverse_log2_codebook_size(
                ni, ei, pi, self._power_constraint), n, epsilon, power)
