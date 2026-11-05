#
# SPDX-License-Identifier: Apache-2.0
#
"""Example real-AWGN comparison; run with python -m sionna.phy.fbl.examples."""

import torch

from sionna.phy.fbl import (
    AWGNGallager,
    AWGNKappaBeta,
    AWGNMetaConverse,
    AWGNNormalApproximation,
)


def main() -> None:
    """Print rates in bits per real channel use at 0 dB and BLER 0.001."""
    blocklength = torch.tensor([100, 300, 1000])
    epsilon = torch.tensor(.001, dtype=torch.float64)
    snr = torch.tensor(1., dtype=torch.float64)
    print("Real AWGN, unit noise variance, linear SNR=1, BLER=0.001")
    print("n:", blocklength.tolist())
    for evaluator in (AWGNGallager, AWGNKappaBeta, AWGNNormalApproximation,
                      AWGNMetaConverse):
        block = evaluator(precision="double", device="cpu")
        rate = block(blocklength, epsilon, snr) / blocklength
        print(f"{evaluator.__name__:25s}", [round(value, 6) for value in rate.tolist()])


if __name__ == "__main__":
    main()
