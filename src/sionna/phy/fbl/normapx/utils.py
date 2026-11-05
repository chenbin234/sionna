#
# SPDX-License-Identifier: Apache-2.0
#
"""Channel-specific normal approximations."""

import torch

from ..utils import real_awgn_capacity_dispersion


def awgn_normal_log2_codebook_size(
    blocklength: torch.Tensor,
    error_probability: torch.Tensor,
    snr: torch.Tensor,
    third_order: bool,
) -> torch.Tensor:
    """Real AWGN: n*C + sqrt(n*V)*Phi^-1(epsilon), optionally +log2(n)/2."""
    capacity, dispersion = real_awgn_capacity_dispersion(snr)
    result = (blocklength * capacity
              + torch.sqrt(blocklength * dispersion)
              * torch.special.ndtri(error_probability))
    if third_order:
        result = result + 0.5 * torch.log2(blocklength)
    # The third-order asymptotics require positive dispersion. At zero power
    # return the zero-rate limit, rather than the spurious log(n)/2 term.
    return torch.where(snr == 0, torch.zeros_like(result), result)
