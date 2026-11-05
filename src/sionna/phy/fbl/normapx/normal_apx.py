#
# SPDX-License-Identifier: Apache-2.0
#
"""Public finite-blocklength normal-approximation evaluators."""


import torch

from sionna.phy import Block
from sionna.phy.config import Precision

from .. import utils as shared_utils
from . import utils as norm_utils

__all__ = ["AWGNNormalApproximation"]


class AWGNNormalApproximation(Block):
    r"""Normal approximation to log2(M) on the real AWGN channel.

    Uses Spectre Ch. 2 ``normapx_awgn`` / Polyanskiy, Poor, and Verdu (2010):
    :math:`nC-\sqrt{nV}Q^{-1}(\epsilon)+\frac12\log_2 n`.
    Noise is iid N(0,1); each real codeword has energy at most n*snr.
    This is an approximation, not an upper or lower bound. Negative values
    are retained. At zero SNR the output is defined as zero. No CSI or fading
    parameters are involved.

    :param third_order: Include the AWGN log2(n)/2 correction (default True).
    :param precision: Sionna precision; None uses the configured default.
    :param device: Sionna device; None uses the configured default.

    :input blocklength: [...], real scalar or Tensor. Positive integers.
    :input error_probability: [...], real scalar or Tensor. Strictly in (0,1).
    :input snr: [...], real scalar or Tensor. Nonnegative linear SNR.
    :output log2_codebook_size: Broadcast input shape, floating Tensor in bits.
        Divide by blocklength for bits per real channel use.

    Supports CPU/CUDA, autograd at positive SNR, and torch.compile. Use double
    precision for small error probabilities. Inputs follow Block's casting
    convention; double tensors preserve all supplied double-precision digits.
    """

    def __init__(
        self,
        third_order: bool = True,
        precision: Precision | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(precision=precision, device=device)
        if not isinstance(third_order, bool):
            raise TypeError("third_order must be bool")
        self._third_order = third_order

    @property
    def third_order(self) -> bool:
        """Whether the AWGN logarithmic correction is included."""
        return self._third_order

    def call(
        self,
        blocklength: shared_utils.TensorLike,
        error_probability: shared_utils.TensorLike,
        snr: shared_utils.TensorLike,
    ) -> torch.Tensor:
        """Evaluate log2(M), broadcasting all three numerical inputs."""
        n, epsilon, power = shared_utils.prepare_inputs(
            blocklength, error_probability, snr, self.dtype, self.device)
        return norm_utils.awgn_normal_log2_codebook_size(
            n, epsilon, power, self._third_order)
