#
# SPDX-License-Identifier: Apache-2.0
#
"""FBL regression tests; run explicitly with pytest on this file.

MATLAB fixtures: MATLAB R2024a, Spectre commit
f34e409e9714e93b195cae348656421f05777a64, generated 2026-11-04.
Inputs: real AWGN, linear P, epsilon=0.001, outputs log2(M).
The kappa-beta MATLAB fixtures use hack=1 (asymptotic kappa).
"""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy import integrate, special, stats

from sionna.phy.fbl import (
    AWGNGallager,
    AWGNKappaBeta,
    AWGNMetaConverse,
    AWGNNormalApproximation,
    AWGNShannon,
    real_awgn_capacity_dispersion,
)
from sionna.phy.fbl.achievability.utils import (
    awgn_gallager_log_error,
    awgn_kappa,
    awgn_shannon_cone,
)
from sionna.phy.fbl.utils import awgn_log_beta, ncx2_logcdf

EVALUATORS = (AWGNGallager, AWGNKappaBeta, AWGNMetaConverse,
              AWGNNormalApproximation, AWGNShannon)
# n, P, normapx_awgn, converse, kappabeta_ach(hack=1), gallager_ach
MATLAB = [
    (20, 1., -0.04849153060539058, 4.00636904391836,
     -7.563712890309015, 0.),
    (20, 10., 22.71539346389564, 27.32323583272336,
     13.06255364890071, 14.86685670582593),
    (100, 1., 26.02075545410565, 30.73832506379052,
     17.91672776098086, 17.46063232421875),
    (100, 10., 144.8993678128089, 149.9041873240969,
     134.478162746987, 130.9058299416906),
    (300, 1., 106.8273912252046, 111.7799116409276,
     98.10530738722899, 87.94956207275391),
    (300, 10., 468.6529044983006, 473.8131829306799,
     457.581920102636, 442.3109594777414),
]


def scalar(value: float) -> torch.Tensor:
    """Preserve double precision through Sionna Block conversion."""
    return torch.tensor(value, dtype=torch.float64)


@pytest.mark.parametrize("row", MATLAB)
def test_matlab_references(row: tuple[float, ...]) -> None:
    """Compare unchanged formulas and account for deliberate port differences."""
    n, power, normal, converse, kappa, gallager = row
    epsilon = scalar(.001)
    for cls, reference in ((AWGNNormalApproximation, normal),
                           (AWGNMetaConverse, converse)):
        result = cls(precision="double", device="cpu")(n, epsilon, scalar(power))
        np.testing.assert_allclose(result, reference, atol=1e-9, rtol=1e-10)
    result = AWGNKappaBeta(kappa_mode="asymptotic", precision="double",
                          device="cpu")(n, epsilon, scalar(power)).item()
    # Python refines tau beyond Spectre's fixed grid and floors M, so allow
    # 0.01 bit for grid optimization and integer rounding in these fixtures.
    np.testing.assert_allclose(result, max(0, kappa), atol=.01, rtol=0)
    corrected = AWGNGallager(precision="double", device="cpu")(
        n, epsilon, scalar(power)).item()
    np.testing.assert_equal(0 <= corrected <= gallager + 1e-3, True)


def test_capacity_dispersion_and_normal_second_order() -> None:
    """Check units, zero dispersion, and the separately selectable correction."""
    c, v = real_awgn_capacity_dispersion(torch.tensor([0., 1.], dtype=torch.float64))
    np.testing.assert_allclose(c, [0., .5])
    np.testing.assert_allclose(v, [0., 3 / (8 * math.log(2)**2)])
    second = AWGNNormalApproximation(third_order=False, precision="double",
                                     device="cpu")(100, .5, 1.)
    np.testing.assert_allclose(second, 50.)
    third = AWGNNormalApproximation(precision="double", device="cpu")(100, .5, 1.)
    np.testing.assert_allclose(third - second, .5 * math.log2(100))


@pytest.mark.parametrize("cls", EVALUATORS)
@pytest.mark.parametrize("precision,dtype", [("single", torch.float32),
                                             ("double", torch.float64)])
def test_broadcast_dtype_and_zero_power(cls: type, precision: str,
                                        dtype: torch.dtype) -> None:
    """Cheap zero-power cases also exercise shape and output precision."""
    result = cls(precision=precision, device="cpu")(
        torch.tensor([[10], [100]]), torch.tensor([.01, .1, .5]), 0.)
    np.testing.assert_equal(tuple(result.shape), (2, 3))
    np.testing.assert_equal(result.dtype, dtype)
    np.testing.assert_equal(result.device.type, "cpu")
    expected = (-torch.log2(1 - torch.tensor([.01, .1, .5], dtype=dtype))
                if cls is AWGNMetaConverse else torch.zeros(3, dtype=dtype))
    torch.testing.assert_close(result, expected.expand(2, 3), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("values", [
    (0, .1, 1.), (2.5, .1, 1.), (10, 0., 1.), (10, 1., 1.),
    (10, .1, -1.), (10, float("nan"), 1.), (10, .1, float("inf")),
    (True, .1, 1.), (10, .1, 1j),
])
@pytest.mark.parametrize("cls", EVALUATORS)
def test_invalid_inputs(cls: type, values: tuple) -> None:
    """Validation must reject unphysical values before numerical solving."""
    with pytest.raises(ValueError):
        cls(device="cpu")(*values)


def test_invalid_shapes_and_configuration() -> None:
    """Reject incompatible batches and misspelled algorithm choices."""
    with pytest.raises(ValueError, match="broadcastable"):
        AWGNNormalApproximation(device="cpu")(torch.ones(2), torch.ones(3) / 2, 1.)
    with pytest.raises(ValueError):
        AWGNKappaBeta(kappa_mode="approx", device="cpu")
    with pytest.raises(ValueError):
        AWGNMetaConverse(power_constraint="average", device="cpu")
    with pytest.raises(ValueError):
        AWGNShannon(device="cpu")(1, .1, 1.)
    with pytest.raises(ValueError):
        AWGNGallager(rtol=0, device="cpu")
    with pytest.raises(TypeError):
        AWGNNormalApproximation(third_order=1, device="cpu")  # type: ignore[arg-type]


@pytest.mark.parametrize("cls", (AWGNKappaBeta, AWGNMetaConverse,
                                 AWGNGallager, AWGNShannon))
def test_scipy_rejects_autograd(cls: type) -> None:
    """Never silently detach a differentiable input."""
    with pytest.raises(ValueError, match="autograd"):
        cls(device="cpu")(100, .001, torch.tensor(1., requires_grad=True))


def test_normal_autograd_and_fullgraph() -> None:
    """Tensor validation and the approximation must support fullgraph capture."""
    block = AWGNNormalApproximation(precision="double", device="cpu")
    snr = scalar(1.).requires_grad_()
    np.testing.assert_equal(torch.autograd.gradcheck(
        lambda power: block(100, scalar(.001), power), (snr,)), True)
    compiled = torch.compile(block, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(100, scalar(.001), snr),
                               block(100, scalar(.001), snr))


def test_beta_one_dimension_independent_gaussian_integral() -> None:
    """In one dimension the likelihood-ratio test accepts a plain interval."""
    power, epsilon = 2., .01
    threshold = stats.ncx2.isf(epsilon, 1, 1 / power)
    center = (1 + power) / math.sqrt(power)
    radius = math.sqrt(threshold)
    p_mass = stats.norm.cdf(center + radius - math.sqrt(power)) - stats.norm.cdf(
        center - radius - math.sqrt(power))
    q_mass = stats.norm.cdf((center + radius) / math.sqrt(1 + power)) - stats.norm.cdf(
        (center - radius) / math.sqrt(1 + power))
    np.testing.assert_allclose(p_mass, 1 - epsilon, atol=1e-13)
    np.testing.assert_allclose(math.exp(awgn_log_beta(epsilon, 1, power)), q_mass,
                               rtol=1e-12)


def test_log_cdf_underflow_and_fallback() -> None:
    """df=1 has an independent Gaussian-tail expression even below exp(-4900)."""
    expected = special.log_ndtr(-99.)
    expected += math.log1p(-math.exp(special.log_ndtr(-101.) - expected))
    np.testing.assert_allclose(ncx2_logcdf(1., 1, 10000.), expected, rtol=1e-12)
    reference = stats.ncx2.logcdf(40., 100, 20.)
    with patch("sionna.phy.fbl.utils.stats.ncx2.logcdf", return_value=-math.inf):
        np.testing.assert_allclose(ncx2_logcdf(40., 100, 20.), reference, rtol=1e-12)
    upper = AWGNMetaConverse(precision="double", device="cpu")(10000, scalar(.001), 10.)
    np.testing.assert_equal(torch.isfinite(upper).item(), True)


def test_kappa_against_independent_radial_grid() -> None:
    """Sort radial likelihood ratios to approximate the optimal randomized test."""
    n, power, tau = 10, 1., .1
    # Uniform Q-quantiles give cells of equal Q probability. This independent
    # discretization converges to the radial Neyman-Pearson test.
    count = 200000
    quantiles = (np.arange(count) + .5) / count
    radii = stats.chi2.ppf(quantiles, n) * (1 + power)
    ratios = stats.ncx2.pdf(radii, n, n * power) / (
        stats.chi2.pdf(radii / (1 + power), n) / (1 + power))
    ordered = np.sort(ratios)[::-1]
    cumulative = np.cumsum(ordered) / count
    index = int(np.searchsorted(cumulative, tau))
    prior = cumulative[index - 1] if index else 0.
    reference = index / count + (tau - prior) / ordered[index]
    np.testing.assert_allclose(awgn_kappa(tau, n, power, 1e-9), reference, rtol=2e-6)


def test_gallager_inversion_is_feasible() -> None:
    """Check the returned integer code against its error bound."""
    n, epsilon, power = 100, .001, 1.
    size = AWGNGallager(precision="double", device="cpu")(n, scalar(epsilon), power).item()
    log_error = awgn_gallager_log_error(n, size * math.log(2) / n, power)
    np.testing.assert_equal(log_error <= math.log(epsilon) + 1e-10, True)
    # Increasing by one bit must cross the optimized error threshold.
    np.testing.assert_equal(awgn_gallager_log_error(
        n, (size + 1) * math.log(2) / n, power) > math.log(epsilon), True)


def test_shannon_cone_against_direct_angle_integration() -> None:
    """For n=2, cap=theta/pi; compute cone escape using independent Gaussians."""
    theta, power = .8, 2.

    def escape(angle: float) -> float:
        return float(integrate.quad(
            lambda z: 2 * stats.norm.pdf(z) * stats.norm.cdf(
                z / math.tan(angle) - math.sqrt(2 * power)),
            0, np.inf, epsabs=1e-10,
        )[0])

    reference = integrate.quad(escape, 0, theta, epsabs=1e-8)[0] / theta
    error, log_cap = awgn_shannon_cone(2, power, theta, 1e-8, .1)
    np.testing.assert_allclose(math.exp(log_cap), theta / math.pi, rtol=1e-9)
    np.testing.assert_allclose(error, reference, rtol=1e-7)


def test_positive_snr_bounds_ordering() -> None:
    """Compare all methods under the same maximal power/error conventions."""
    n, power, epsilon = 100, scalar(1.), scalar(.001)
    upper = AWGNMetaConverse(precision="double", device="cpu")(n, epsilon, power)
    for cls in (AWGNKappaBeta, AWGNGallager, AWGNShannon):
        lower = cls(precision="double", device="cpu")(n, epsilon, power)
        np.testing.assert_equal((0 < lower).item(), True)
        np.testing.assert_equal((lower < upper).item(), True)
    equal = AWGNMetaConverse(power_constraint="equal", precision="double",
                            device="cpu")(n, epsilon, power)
    np.testing.assert_equal((equal < upper).item(), True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_normal_approximation() -> None:
    """The tensor path has the same units and output on CUDA."""
    cpu = AWGNNormalApproximation(precision="double", device="cpu")
    gpu = AWGNNormalApproximation(precision="double", device="cuda:0")
    torch.testing.assert_close(gpu(100, .001, 1.).cpu(), cpu(100, .001, 1.))
    with pytest.raises(ValueError, match="cpu"):
        AWGNMetaConverse(device="cuda:0")
