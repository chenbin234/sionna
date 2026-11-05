#
# SPDX-License-Identifier: Apache-2.0
#
"""BLER benchmarking, QPSK normalization, and PlotBER integration tests."""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sionna.phy import config
from sionna.phy.fbl import (
    AWGNBoundBLER,
    AWGNGallager,
    AWGNKappaBeta,
    AWGNMetaConverse,
    AWGNNormalApproximation,
    AWGNShannon,
    FBLPlotBER,
)
from sionna.phy.utils import PlotBER


def tensor(value: float | list[float]) -> torch.Tensor:
    """Construct double inputs before Sionna's automatic input casting."""
    return torch.tensor(value, dtype=torch.float64)


@pytest.mark.parametrize("k,n", [(64, 128), (32, 128)])
@pytest.mark.parametrize("third_order", [True, False])
def test_qpsk_normalization_and_normal_roundtrip(
    k: int, n: int, third_order: bool,
) -> None:
    """QPSK maps n coded bits to n real dimensions, with P=2*rate*EbNo."""
    bound = AWGNNormalApproximation(third_order=third_order,
                                    precision="double", device="cpu")
    model = AWGNBoundBLER(k=k, n=n, bound=bound)
    ebno = tensor([-1., 0., 1.])
    power = 2 * (k / n) * torch.pow(10., ebno / 10)
    bler = model(ebno)
    recovered = bound(n, bler, power)
    torch.testing.assert_close(recovered, torch.full_like(ebno, k),
                               atol=1e-9, rtol=1e-10)
    np.testing.assert_equal(tuple(bler.shape), (3,))
    np.testing.assert_equal(bler.dtype, torch.float64)
    np.testing.assert_equal(bler.device.type, "cpu")
    np.testing.assert_equal(model.result_kind, "approximation")


def test_normal_rare_tail_and_global_precision_independence() -> None:
    """The CPU float64 adapter preserves rare tails and its own precision."""
    model = AWGNBoundBLER(k=128, n=256,
                          bound=AWGNNormalApproximation(precision="double", device="cpu"))
    original_precision = config.precision
    try:
        config.precision = "single"
        single_default = model(tensor(4.5))
        config.precision = "double"
        double_default = model(tensor(4.5))
    finally:
        config.precision = original_precision
    torch.testing.assert_close(single_default, double_default, atol=0, rtol=0)
    np.testing.assert_equal((0 < single_default < 1e-10).item(), True)


@pytest.mark.parametrize("power_constraint", ["equal", "maximal"])
def test_metaconverse_roundtrip(power_constraint: str) -> None:
    """The direct BLER hypothesis test inverts the same log-size converse."""
    bound = AWGNMetaConverse(power_constraint=power_constraint,
                             precision="double", device="cpu")
    model = AWGNBoundBLER(k=16, n=32, bound=bound)
    ebno = tensor([0., 1., 2.])
    bler = model(ebno)
    recovered = bound(32, bler, torch.pow(10., ebno / 10))
    torch.testing.assert_close(recovered, torch.full_like(ebno, 16),
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_equal(model.result_kind, "lower bound")


@pytest.mark.parametrize("bound_type", [AWGNGallager, AWGNShannon])
def test_fixed_message_achievability_order_and_monotonicity(bound_type: type) -> None:
    """Upper and lower optimal-BLER bounds retain their order under conversion."""
    bound = bound_type(precision="double", device="cpu")
    model = AWGNBoundBLER(k=8, n=32, bound=bound)
    ebno = tensor([3., 6., 9.])
    upper = model(ebno)
    lower = AWGNBoundBLER(k=8, n=32)(ebno)
    np.testing.assert_equal(torch.all((upper >= 0) & (upper <= 1)).item(), True)
    np.testing.assert_equal(torch.all(upper[:-1] >= upper[1:]).item(), True)
    np.testing.assert_equal(torch.all(lower <= upper).item(), True)
    np.testing.assert_equal(model.result_kind, "upper bound (AWGN)")


def test_gallager_log_size_feasible_roundtrip() -> None:
    """A slightly relaxed direct upper BLER guarantees the requested payload."""
    bound = AWGNGallager(precision="double", device="cpu")
    model = AWGNBoundBLER(k=64, n=128, bound=bound)
    ebno = 10 * math.log10(2.5)
    bler = model(tensor(ebno))
    np.testing.assert_equal((0 < bler < 1).item(), True)
    recovered = bound(128, bler * 1.00001, tensor(2.5))
    np.testing.assert_equal((recovered >= 64).item(), True)


def test_kappa_beta_inversion_returns_feasible_endpoint() -> None:
    """The returned upper error endpoint must achieve at least k bits."""
    bound = AWGNKappaBeta(precision="double", device="cpu")
    model = AWGNBoundBLER(k=8, n=32, bound=bound, inversion_rtol=1e-3)
    bler = model(tensor(10 * math.log10(4.)))
    # Rate=1/4, so real P=0.5*EbNo=2 at this Eb/N0.
    recovered = bound(32, bler, tensor(2.))
    np.testing.assert_equal((0 < bler < 1).item(), True)
    np.testing.assert_equal((recovered >= 8).item(), True)


def test_bler_table_and_repeated_curve_replacement(capsys: pytest.CaptureFixture[str]) -> None:
    """Analytical rows contain BLER/runtime and placeholders for sample counts."""
    plot = FBLPlotBER("Analytical test")
    model = AWGNBoundBLER(k=64, n=128,
                          bound=AWGNNormalApproximation(precision="double", device="cpu"))
    ber, bler = plot.simulate(model, ebno_dbs=[0., 1.], legend="normal",
                              batch_size=10000, max_mc_iter=100,
                              compile_mode="default", show_fig=False)
    np.testing.assert_equal(torch.all(torch.isnan(ber)).item(), True)
    np.testing.assert_equal(tuple(bler.shape), (2,))
    output = capsys.readouterr().out
    np.testing.assert_equal("EbNo [dB]" in output, True)
    np.testing.assert_equal("runtime [s]" in output, True)
    fields = [field.strip() for field in output.splitlines()[2].split("|")]
    np.testing.assert_equal(fields[1], "--")
    np.testing.assert_equal(fields[3:7], ["--"] * 4)
    np.testing.assert_equal(fields[-1], "approximation")
    np.testing.assert_allclose(float(fields[2]), float(bler[0]), rtol=1e-4)
    np.testing.assert_equal(plot.legend, ["normal (BLER)"])
    np.testing.assert_equal(plot.is_bler, [True])
    plot.simulate(model, ebno_dbs=[2., 3.], legend="normal",
                  show_fig=False, verbose=False)
    np.testing.assert_equal(plot.legend, ["normal (BLER)"])
    np.testing.assert_equal(plot.is_bler, [True])
    np.testing.assert_allclose(plot.snr[0], [2., 3.])


def test_add_results_false_and_bler_disabled() -> None:
    """Analytical evaluations can return results without modifying plot data."""
    plot = FBLPlotBER()
    model = AWGNBoundBLER(k=64, n=128)
    for store_results, store_bler in ((False, True), (True, False)):
        _, bler = plot.simulate(model, ebno_dbs=[0., 1.], legend="converse",
                                show_fig=False, verbose=False,
                                add_results=store_results, add_bler=store_bler)
        np.testing.assert_equal(tuple(bler.shape), (2,))
        np.testing.assert_equal(plot.legend, [])
        np.testing.assert_equal(plot.ber, [])


def test_normal_coding_model_delegates_to_plotber() -> None:
    """Coding models retain the existing simulation arguments and return pair."""
    plot = FBLPlotBER()
    expected = (tensor([.1]), tensor([.2]))

    def coding_model(batch_size: int, ebno_db: torch.Tensor) -> tuple:
        """Placeholder: delegation is mocked before Monte Carlo execution."""
        bits = torch.zeros((batch_size, 4), device=ebno_db.device)
        return bits, bits

    with patch.object(PlotBER, "simulate", return_value=expected) as simulate:
        result = plot.simulate(coding_model, ebno_dbs=[0.], batch_size=4,
                               max_mc_iter=3, legend="code", add_bler=True,
                               show_fig=False, compile_mode="default")
        simulate.assert_called_once()
        np.testing.assert_equal(simulate.call_args.args[0] is coding_model, True)
        options = simulate.call_args.kwargs
        torch.testing.assert_close(options["ebno_dbs"], torch.tensor([0.]))
        for key, value in {"batch_size": 4, "max_mc_iter": 3, "legend": "code",
                           "add_ber": True, "add_bler": True, "show_fig": False,
                           "compile_mode": "default"}.items():
            np.testing.assert_equal(options[key], value)
    torch.testing.assert_close(result[0], expected[0])
    torch.testing.assert_close(result[1], expected[1])


@pytest.mark.parametrize("kwargs,error", [
    ({"k": 0, "n": 128}, ValueError),
    ({"k": 129, "n": 128}, ValueError),
    ({"k": 64, "n": 127}, ValueError),
    ({"k": True, "n": 128}, TypeError),
    ({"k": 64., "n": 128}, TypeError),
    ({"k": 64, "n": 128, "num_bits_per_symbol": 4}, ValueError),
    ({"k": 64, "n": 128, "bound": AWGNNormalApproximation(precision="single", device="cpu")}, ValueError),
    ({"k": 64, "n": 128, "error_probability_floor": 0}, ValueError),
    ({"k": 64, "n": 128, "inversion_rtol": 0}, ValueError),
])
def test_invalid_configuration(kwargs: dict, error: type[Exception]) -> None:
    """Reject unsupported payload/channel assumptions before evaluating bounds."""
    with pytest.raises(error):
        AWGNBoundBLER(**kwargs)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, 1j])
def test_invalid_ebno(value: float) -> None:
    """Physical Eb/N0 values must be finite and real."""
    with pytest.raises(ValueError):
        AWGNBoundBLER(k=64, n=128)(value)
    with pytest.raises(ValueError, match="autograd"):
        AWGNBoundBLER(k=64, n=128)(tensor(1.).requires_grad_())


@pytest.mark.parametrize("points", [[], [[0., 1.]], [float("nan")], [True]])
def test_invalid_plot_points(points: list) -> None:
    """Curve evaluation requires a nonempty real one-dimensional sweep."""
    with pytest.raises(ValueError):
        FBLPlotBER().simulate(AWGNBoundBLER(k=64, n=128), ebno_dbs=points,
                             show_fig=False, verbose=False)


def test_analytical_ber_and_unknown_keywords_rejected() -> None:
    """BLER benchmarks cannot be disguised as bit-error simulations."""
    plot, model = FBLPlotBER(), AWGNBoundBLER(k=64, n=128)
    with pytest.raises(ValueError, match="BLER only"):
        plot.simulate(model, ebno_dbs=[0.], add_ber=True, show_fig=False)
    with pytest.raises(TypeError, match="unexpected"):
        # Deliberately pass an unsupported keyword to check the runtime API.
        plot.simulate(model, ebno_dbs=[0.], typo=True, show_fig=False)  # type: ignore[call-arg]
