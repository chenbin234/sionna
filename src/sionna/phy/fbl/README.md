# Real-AWGN finite-blocklength bounds

Implemented following the module layout in [README_fbl.md](README_fbl.md),
Spectre's supplied manual, Chapter 2 (PDF pages 7-9), and its `awgn/` routines.
All implementation, examples, and tests are inside this directory. The parent
PHY package and project dependencies are unchanged. Validated with Python 3.13.

## Public API

```python
import torch
from sionna.phy.fbl import (
    AWGNKappaBeta,
    AWGNGallager,
    AWGNShannon,
    AWGNMetaConverse,
    AWGNNormalApproximation,
    real_awgn_capacity_dispersion,
)

n = torch.tensor([100, 300, 1000])
epsilon = torch.tensor(1e-3, dtype=torch.float64)
snr = torch.tensor(1.0, dtype=torch.float64)  # linear SNR: 0 dB

achievable = AWGNKappaBeta(precision="double", device="cpu")(n, epsilon, snr)
converse = AWGNMetaConverse(precision="double", device="cpu")(n, epsilon, snr)
normal = AWGNNormalApproximation(precision="double", device="cpu")(n, epsilon, snr)
rate_lower = achievable / n  # bits per real channel use
```

Every evaluator is a Sionna `Block`. All accept
`(blocklength, error_probability, snr)` and return a floating tensor named
`log2_codebook_size`, in **bits**, with the broadcast shape of the inputs.
Scalar inputs produce a zero-dimensional tensor. An axis is a parameter-batch
axis, not a time, antenna, or Monte Carlo axis. For a Cartesian sweep use, for
example, `blocklength[:, None]` and `snr[None, :]`.

- `blocklength`: positive integer scalar/tensor; Shannon requires at least 2.
- `error_probability`: finite and strictly between 0 and 1; block error, not BER.
- `snr`: finite, nonnegative, **linear** SNR. Convert dB explicitly with
  `sionna.phy.utils.db_to_lin`.
- Inputs must be broadcastable. Invalid values, shapes, or dtypes raise
  `ValueError`; validation remains active under optimized Python.
- `precision=None` and `device=None` follow Sionna configuration. Use
  `precision="double"` and double input tensors for reference calculations.
  In the current Sionna `Block`, Python floats are initially converted using
  PyTorch's default dtype; supplying float64 tensors avoids that initial rounding.

The common quantity and named evaluators can be reused for other channels.
Add each channel's public classes to the corresponding main evaluator file,
its numerical kernels to that category's `utils.py`, and only shared mathematics
to the root `utils.py`. Channel-specific assumptions and corrections stay with
that channel; there is no universal channel selector or extra base-bound class.

## Model and result semantics

The channel is `Y = X + Z`, with independent **real** `Z ~ N(0,1)` and
per-codeword energy `sum(X**2) <= n*snr`. These are not complex channel uses:
Sionna's simulation `AWGN(no)` uses complex noise variance `no`, whose real
component has variance `no/2`.

| Evaluator | Result | Implementation |
| --- | --- | --- |
| `AWGNKappaBeta(kappa_mode="exact")` | Achievability lower bound, maximal error | Finite-n radial Neyman-Pearson kappa and Gaussian beta; grid plus local tau optimization |
| `AWGNKappaBeta(kappa_mode="asymptotic")` | Achievability **approximation** | Substitutes Spectre's limiting `kappa_inf`; not a rigorous finite-n bound |
| `AWGNGallager()` | Achievability lower bound, maximal error | Cost-constrained Gaussian random coding, theorem's squared prefactor, feasible shell width, rate inversion |
| `AWGNShannon()` | Achievability lower bound, maximal error | Exact spherical-cap integration and expurgation from average error |
| `AWGNMetaConverse(power_constraint="maximal")` | Converse upper bound, average or maximal error | Gaussian auxiliary output, dimension `n+1`, matching Spectre's power-ball embedding |
| `AWGNMetaConverse(power_constraint="equal")` | Converse upper bound for equal-energy codes | Dimension `n`; requires every codeword to have exactly energy `n*snr` |
| `AWGNNormalApproximation(third_order=True)` | Approximation, no bound direction | `n*C + sqrt(n*V)*Phi_inv(epsilon) + log2(n)/2` |

`real_awgn_capacity_dispersion(snr_tensor)` returns `(C, V)` in bits/use and
bits squared/use, with the input tensor's shape, dtype, and device:

```text
C = log2(1 + snr) / 2
V = snr*(snr + 2) / (2*(1 + snr)**2) * log2(e)**2
```

Achievability returns a nonnegative value and conservatively rounds down to an
integer codebook size, with the one-message zero-error code as a fallback.
For sizes beyond 52 bits the implementation moves the result down one float64
ULP instead of materializing the enormous codebook size. The converse keeps
the continuous upper bound. The normal approximation keeps negative values;
`third_order=False` omits the logarithmic term. At zero SNR, achievability and
the normal approximation return zero; the converse returns
`-log2(1-error_probability)`. The latter is an average-error upper bound, not a
promise that multiple zero-power codewords are distinguishable.

## Numerical behavior

The normal approximation and capacity/dispersion use PyTorch, support
CPU/CUDA and autograd at positive SNR, and support full-graph capture. The test
suite checks gradients and `torch.compile(..., backend="eager", fullgraph=True)`.
Other compile backends are not certified by these tests.

The four numerical bound classes use **CPU SciPy**, computing internally in
float64 and returning the selected Sionna dtype. Their configured device must
be CPU; autograd inputs are rejected. They do not support `torch.compile`.
Batches run scalar solves, so Shannon in particular can take seconds per point.
Defaults are `rtol=1e-7` for kappa-beta/Gallager and `rtol=1e-6` for Shannon.
Gallager interprets this tolerance as an absolute tolerance in `n*rate` in nats;
the quadrature methods use it relatively. Float32 output can round away a
smaller internal tolerance.

Gaussian beta uses a noncentral chi-square survival inverse without forming
`1-epsilon`. Its CDF is evaluated in the log domain, with a positive
Poisson/incomplete-gamma series when SciPy underflows. No large-deviation or
normal approximation is silently substituted. Finite-n kappa solves the
radial likelihood-ratio test; a short-interval quadrature avoids CDF subtraction.
Shannon rescales its angular integrals so tiny cap areas are stored as logs.

These are numerical evaluations of bounds, **not interval-certified results**.
Local optimization may leave a looser achievable result. Under-resolved special
functions or nonconvergent solvers raise `RuntimeError`; extremely small error
probabilities and extreme powers can exceed float64 numerical resolution.
There is no hidden cache, RNG use, plotting, or filesystem access in evaluators.

## Differences from the supplied MATLAB source

Reference checkout: `f34e409e9714e93b195cae348656421f05777a64`.

1. **Kappa-beta:** the default is finite-n kappa, whereas Spectre defaults to
   the asymptotic approximation (`hack=1`). Python refines tau instead of only
   trying Spectre's fixed grid. Spectre's finite-n `kappa.m` overwrites its scalar
   `P` with a two-element CDF vector and subsequently crashes in `kappa_inf`.
   Python solves the radial test without that variable reuse or fixed radius grid.
2. **Shannon:** `shannon_ach2.m` initializes with `tau*epsilon` but bisects
   against `epsilon`, then still expurgates by `1-tau`. Python consistently uses
   `tau*epsilon`. It evaluates the cap integral in PPV Eq. (42) instead of
   Spectre's cap estimates, and rounds the average-error code size before
   expurgating. Numerical results need not match the uncorrected MATLAB routine.
3. **Gallager:** PPV Eq. (44) uses `(2*exp(s*delta)/mu)**2`; Spectre uses the
   unsquared multiplier. Python retains the squared prefactor, giving a more
   conservative result. It keeps Spectre's feasible `delta=min(1/s,n)` choice.
4. **Converse:** Python evaluates beta directly, including a log-domain series,
   instead of switching to Spectre's Wolfowitz approximation when tails underflow.
5. The achievability one-message fallback and explicit integer rounding differ
   from Spectre's possibly negative continuous `log2(M)` output.

The Python code is implemented from the probability laws and equations.
Spectre serves as a mathematical and numerical reference; its MATLAB function
bodies and embedded MathWorks noncentral-t code were not incorporated.

## Validation and examples

From the repository root:

```sh
.venv/bin/python -m pytest src/sionna/phy/fbl/test_fbl.py
.venv/bin/python -m sionna.phy.fbl.examples
ruff check src/sionna/phy/fbl
mypy --python-version 3.13 --follow-imports silent --ignore-missing-imports \
  --disallow-untyped-defs --warn-return-any src/sionna/phy/fbl
```

The tests contain double-precision MATLAB R2024a fixtures generated from the
supplied Spectre checkout on 2026-11-04. They use `n in {20,100,300}`,
`P in {1,10}`, and `epsilon=0.001`, with `kappabeta_ach(...,1)` for the
asymptotic option. Normal/metaconverse agreement uses approximately `1e-9`
absolute tolerance; kappa-beta allows 0.01 bit for tau refinement and rounding.
Gallager is checked to be more conservative than the source's unsquared result.
Finite-n kappa is checked by an independent discretized radial hypothesis test,
not against the crashing MATLAB implementation. Further checks cover a direct
one-dimensional Gaussian beta integral, log tails below `exp(-4900)`, an
independent two-dimensional Shannon calculation, inversion feasibility,
broadcasting, dtype, invalid inputs, zero SNR, gradient checking, and bound order.
The CUDA test is conditional on hardware availability.

MATLAB fixtures can be regenerated using the original `awgn` folder:

```matlab
for n = [20,100,300]
    for P = [1,10]
        normal = normapx_awgn(n,0.001,P);
        upper = converse(n,0.001,P);
        approx = kappabeta_ach(n,0.001,P,1);
        gallager = gallager_ach(n,0.001,P);
        fprintf('%d %.1f %.16g %.16g %.16g %.16g\n', ...
                n,P,normal,upper,approx,gallager);
    end
end
```

Mathematical reference: Y. Polyanskiy, H. V. Poor, and S. Verdu,
*Channel Coding Rate in the Finite Blocklength Regime*, IEEE Transactions on
Information Theory 56(5), 2010. Equation numbers above follow the
[author-hosted manuscript](https://people.lids.mit.edu/yp/homepage/data/finite_block.pdf),
also cited by Spectre: Eqs. (41)-(44), (215)-(218), and (220).

## BLER plots alongside QPSK coding schemes

`AWGNBoundBLER` converts a configured evaluator into a fixed-payload BLER
benchmark. `FBLPlotBER` extends `PlotBER.simulate` to accept these benchmarks
as well as ordinary coding models. The notebook
`tutorials/phy/5G_Channel_Coding_vs_Finite_Blocklength_Bounds.ipynb` demonstrates
all five methods and includes them in both short-code BLER comparison plots.

```python
from sionna.phy.fbl import AWGNBoundBLER, AWGNMetaConverse, FBLPlotBER

plot = FBLPlotBER("Coding schemes and AWGN benchmarks")
benchmark = AWGNBoundBLER(
    k=64, n=128,
    bound=AWGNMetaConverse(power_constraint="equal", precision="double", device="cpu"),
)

# Coding models and analytical models share this calling pattern.
models_under_test = [
    # (System_Model(k=64, n=128, num_bits_per_symbol=2,
    #               encoder=encoder, decoder=decoder), "5G LDPC BP-20"),
    (benchmark, "FBL metaconverse (lower, QPSK-valid)"),
]
for model, name in models_under_test:
    ber, bler = plot.simulate(
        model, ebno_dbs=[0., .5, 1., 1.5, 2.], legend=name,
        batch_size=10000, max_mc_iter=100, num_target_block_errors=1000,
        add_bler=True, compile_mode="default", show_fig=False,
    )
plot(ylim=(1e-5, 1), show_ber=False)
```

Coding models use the original Monte Carlo simulation. Analytical models
are evaluated once at every SNR and print the same table columns, with `--`
for BER and Monte Carlo counts, numerical BLER/runtime, and a bound-direction
status. Their returned BER tensor contains `NaN` (undefined); only the BLER
curve is stored. Explicitly requesting `add_ber=True` for a benchmark raises
`ValueError`. Batch-size, iteration, stopping, and compile options do not affect
analytical evaluation. Repeating an analytical legend replaces its old curve.

This adapter currently supports **unit-energy QPSK**, with `n` transmitted coded
bits and `k` information bits. CRC overhead is included in the coding operation,
not added to `k`. There are `n/2` complex symbols and `n` real channel uses:

```text
real_snr = 2*(k/n)*10**(ebno_db/10) = 1/ebnodb2no(ebno_db, 2, k/n)
```

Thus `k=64,n=128` gives `real_snr=10**(ebno_db/10)`. The equal-energy
metaconverse applies to every QPSK codeword. AWGN achievability and normal
approximation use the larger class of unrestricted real-AWGN codebooks and
are labeled accordingly. The achievability curves bound optimal AWGN BLER
from above; they do not promise a QPSK encoder/decoder with that error rate.
These quantities are block errors and are compared with **BLER**, not BER.

`AWGNBoundBLER` uses float64 CPU computation. Normal approximation, Gallager,
Shannon, and metaconverse have direct fixed-message evaluations; Shannon
optimizes its expurgation fraction after choosing an integer average-error
codebook size. Kappa-beta uses a feasible upper endpoint from bisection in
log error probability. Its default relative inversion tolerance is `1e-4`;
values below its default numerical resolution `1e-10` are conservatively
reported as `1e-10`. This floor affects kappa-beta only. All returned analytical
probabilities remain unclipped by plotting limits. The adapter supports neither
autograd nor `torch.compile` and checks its underlying evaluator is configured
with `precision="double",device="cpu"`.

Additional regression tests:

```sh
.venv/bin/python -m pytest src/sionna/phy/fbl/test_benchmark.py
```
