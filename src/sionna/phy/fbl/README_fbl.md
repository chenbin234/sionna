# Finite-blocklength bounds: architecture and Spectre porting plan

This is a design for `src/sionna/phy/fbl`, based on the local Sionna 2.2.0
checkout (commit `c4f11da561a8adb23d21e46f9725852af3713c2e`) and the supplied
Spectre v2.0 manual and MATLAB source. The checkout uses **PyTorch**, not
TensorFlow. The six existing Python files in `fbl` are empty placeholders.
The tree and APIs below are **proposed implementation targets**; this document
does not implement bounds, create those modules, or rename the placeholders.

The main recommendation is to keep achievability, converse, and normal
approximation as the three main categories. Each category has one main Python
file containing its public evaluators and one `utils.py` containing the
implementations and helpers for all supported channel models. Organize those
functions into clearly labeled sections inside the file. Keep the existing
`normapx/normal_apx.py` naming and add only one root `utils.py` for mathematics
shared across categories.

## 1. How FBL fits into Sionna PHY

The existing PHY layout separates infrastructure, simulation components, and
coding algorithms:

| Existing location under `src/sionna/phy` | Responsibility | Relationship to FBL |
| --- | --- | --- |
| `config.py`, `object.py`, `block.py` | Precision, device, RNG, common object and callable block behavior | Reuse these foundations directly; they are PHY-wide, not specific to FEC. |
| `utils/` | Tensor operations, numerical solving, random sampling, conversions, metrics | Reuse applicable public functions instead of duplicating them. |
| `channel/` | Stochastic channel simulation, CIR generation, channel application | Useful for sampling and validation; a simulator does not by itself provide the analytic channel laws required by a bound. |
| `mapping.py` | Constellations, mapping, demapping, symbol generation | Reuse alphabets and sampling for constrained-input bounds. |
| `fec/` | Concrete encoders, decoders, code construction and diagnostics | Use for comparisons with actual codes; theoretical bounds should not depend on an LDPC/polar decoder. |
| `mimo/`, `ofdm/`, `nr/` | Spatial processing, waveform and standard-specific systems | Relevant to later benchmarks when their assumptions match a bound. |
| `fbl/` | Proposed theoretical limits and approximations | A PHY sibling of `fec`, with a separate mathematical role. |

### Choosing `Block`, `Object`, or a function

**Use `Block` for a public tensor evaluator** when users should get Sionna's
precision/device behavior and batched invocation. Import it with
`from sionna.phy import Block`, call
`super().__init__(precision=precision, device=device)`, and implement `call()`.
Do not replace its `__call__()` or manually invoke `build()`.

`Block.__call__()` recursively converts inputs, builds once when needed, then
delegates through `forward()` to `call()`. Floating tensors are cast to the
block's dtype and device. Python floats become tensors; Python integers remain
integers. Keep shape parameters such as sample counts or antenna counts as
explicit integer configuration where appropriate. An integer tensor remains an
integer tensor: validate and normalize numerical parameters deliberately.

**Use `Object` for a stateful helper** that needs `dtype`, `cdtype`, `device`, or
RNG access but defines its own methods, such as a channel-law object with
`log_prob()` and `capacity_dispersion()`. `Object` is already a
`torch.nn.Module`; it does not provide `Block`'s automatic invocation-time input
conversion.

**Use ordinary PyTorch functions for stateless kernels**, such as log-binomial
coefficients, hypothesis-test calculations, or a capacity formula. They do not
need a separate block or a new inheritance hierarchy. Existing PHY utilities
already follow this pattern. A new abstract `Bound` base class is unnecessary
until multiple implementations demonstrably share behavior beyond `Block`.

Objects capture configured precision and device at construction. Accept
`precision=None` and `device=None` consistently; use explicit double precision
for reference validation. Do not set global `config.precision` or reset
`config.seed` inside a bound evaluator. A block does not automatically make a
numerical optimizer differentiable, GPU-capable, or compatible with
`torch.compile`.

### Concrete modules to reuse

The following symbols were checked in this checkout. Paths are relative to
`src/sionna/phy`.

| Module / symbols | Suitable reuse | Important limit |
| --- | --- | --- |
| `config.py`: `config`, `dtypes`, `Precision`; `Object.dtype`, `cdtype`, `np_dtype`, `torch_rng`, `np_rng` | Existing dtype/device defaults and controlled randomness | No second FBL config singleton. The configured devices in this checkout are CPU/CUDA. |
| `utils/tensors.py`: `expand_to_rank`, `insert_dims`, `flatten_dims`, `split_dim` | Broadcasting parameter batches and separating sample/block/antenna axes | Define the statistical meaning of each axis yourself. |
| `utils/random.py`: `rand`, `randint`, `normal`, `complex_normal` | Gaussian matrices, random messages, Monte Carlo samples | Use Sionna's RNG integration. Eager and compiled paths need not generate identical samples. |
| `utils/numerics.py`: `bisection_method`, `expand_bound` | Batched inversions of monotone rate/error/SNR relationships | Bisection assumes a **decreasing** residual. Set the sign and physical bracket correctly; disable automatic expansion when it could leave the domain. It is not a general constrained optimizer. |
| `utils/misc.py`: `db_to_lin`, `lin_to_db`, `ebnodb2no` | Explicit conversion between SNR conventions | `ebnodb2no` uses constellation bits/symbol and encoder code rate, with optional OFDM overhead. An FBL rate in bits/channel-use is not automatically an encoder code rate. |
| `utils/linalg.py`: `inv_cholesky`, `matrix_pinv`; `torch.linalg` | Existing matrix operations when appropriate | Use solves and stable log determinants for likelihoods; do not form inverses unnecessarily. |
| `channel/awgn.py`: `AWGN` | Complex AWGN sampling and code comparisons | `no` is variance per complex dimension, with `no/2` per real component. Spectre Ch. 2 instead uses real noise variance 1. |
| `channel/discrete_channel.py`: `BinarySymmetricChannel`, `BinaryErasureChannel` | Existing BSC/BEC samplers | They do not replace the exact binomial sums or hypothesis tests used in analytical bounds. |
| `channel/rayleigh_block_fading.py`: `RayleighBlockFading`; `channel/flat_fading_channel.py` | Fading sampling when the law and coherence match | `RayleighBlockFading` keeps the same coefficient over every time step of a batch example. A codeword spanning several independent coherence blocks needs explicitly independent coefficients. |
| `mapping.py`: `Constellation`, `Mapper`, `SymbolSource`, `QAMSource` | Reuse finite alphabets and symbol generation | A Gaussian-input AWGN bound and a binary-input AWGN bound are different problems. |
| `mapping.py`: `Demapper` | Comparison with a demapper or matched binary likelihood calculation | Its bit LLR is `log P(1)/P(0)`. Bit LLRs are not a general symbol information-density interface. |
| `utils/metrics.py`: `compute_bler`, `count_block_errors`; `utils/misc.py`: `sim_ber` | Empirical comparison with practical coding schemes | Match block error probability, channel normalization, rate, and decoder assumptions. |

`fec.utils.llr2mi`, `j_fun`, `j_fun_inv`, and `GaussianPriorSource` are useful
diagnostics under their LLR assumptions. They are not general replacements for
channel capacity, dispersion, information density, or hypothesis-testing bounds.
In particular, `llr2mi` has symmetry/sign assumptions and clipping, while the
J-functions use a Gaussian LLR approximation.

What FBL still needs to implement is the theorem-specific mathematics:
information densities, auxiliary distributions, randomized hypothesis tests,
capacity/dispersion where unavailable, stable tail probabilities, quadrature,
and constrained optimization. Sionna's foundations supply infrastructure rather
than these information-theoretic quantities.

## 2. Simplified folder structure

Keep the existing three categories. Each has the same three-file pattern:
`__init__.py`, one main evaluator file, and `utils.py`. Different channel models
are functions or sections inside `utils.py`; they do not need separate files.
All paths below are inside `src/sionna/phy/fbl`.

```text
fbl/
|-- README.md
|-- __init__.py
|-- utils.py
|-- achievability/
|   |-- __init__.py
|   |-- achievability_bound.py
|   `-- utils.py
|-- converse/
|   |-- __init__.py
|   |-- converse_bound.py
|   `-- utils.py
`-- normapx/
    |-- __init__.py
    |-- normal_apx.py
    `-- utils.py
```

This is the proposed implementation layout. The existing six Python files
remain in place; initializers and root `utils.py` can be added when their
implementations are ready.

### Which file does what

| File | Responsibility |
| --- | --- |
| `README.md` | Structure, Sionna integration, API conventions, and Spectre coverage. |
| Root `__init__.py` | Export the implemented public API explicitly. |
| Root `utils.py` | Functions needed by more than one category: shared capacity/dispersion or information-density calculations, stable probability/HT kernels, validation, and numerical inversion. Reuse existing PHY utilities first. Capacity/outage references also fit here. |
| `achievability/__init__.py` | Export public evaluators from `achievability_bound.py`. |
| `achievability/achievability_bound.py` | Define all public achievability classes/functions, their parameters and output contracts, and their calls into local helpers. Use `Block` for tensor evaluators where appropriate. |
| `achievability/utils.py` | Implement channel-specific achievability mathematics and supporting operations for AWGN, BSC, BEC, block fading, quasi-static fading, and long-term power. Group them by channel within this one file. |
| `converse/__init__.py` | Export public evaluators from `converse_bound.py`. |
| `converse/converse_bound.py` | Define all public converse classes/functions and their input/output contracts; call the corresponding local helpers. |
| `converse/utils.py` | Implement all channel-specific converse calculations, auxiliary-distribution evaluations, and optimization helpers. Group them by channel within this one file. |
| `normapx/__init__.py` | Export public approximation evaluators from `normal_apx.py`. |
| `normapx/normal_apx.py` | Define all public normal-approximation classes/functions and their input/output contracts; call local helpers and shared statistics. |
| `normapx/utils.py` | Implement approximations for all supported channel models, including each model's correction terms and fading expectations. |

For example, `achievability/utils.py` can have BSC, BEC, AWGN, block-fading,
quasi-static, and long-term-power sections. Use descriptive names such as
`bsc_dt_error_probability`, `awgn_kappa_beta_log2_codebook_size`, and
`quasi_static_simo_csir_rate` to distinguish implementations without creating
extra files. These are illustrative names, not existing API symbols.

The main evaluator file owns the public API; local `utils.py` owns its
channel-specific mathematical implementations. A main evaluator validates its
inputs, selects or calls the appropriate helper, and formats its documented
output. Its role is more than re-exporting symbols. Use explicit exports in
`__init__.py` so that internal helpers remain implementation details.

Keep genuinely shared calculations in root `fbl/utils.py`. For example, BSC
capacity/dispersion or a discrete beta function used by several categories
belongs there. An achievability-only USTM helper stays in
`achievability/utils.py`. Small local validation and diagnostic helpers can stay
in these same files; they do not require their own modules.

### Extending the same pattern to the full manual

The three categories above are enough for the initial channel-bound work. When
porting the manual's energy, source-coding, and JSCC topics, use the same compact
pattern for each additional category:

```text
energy/
|-- __init__.py
|-- energy_bound.py
`-- utils.py

source_coding/
|-- __init__.py
|-- source_coding_bound.py
`-- utils.py

joint_source_channel/
|-- __init__.py
|-- joint_source_channel_bound.py
`-- utils.py
```

These are future additions as those topics are implemented. All models within
each category share its one main file and one `utils.py`. The main file defines
named public achievability, converse, approximation, or exact-quantity
evaluators; its `utils.py` contains their model-specific formulas and solvers.
For example, binary and Gaussian lossy-source functions belong to sections of
`source_coding/utils.py`, while their public evaluators belong to
`source_coding_bound.py`. Source coding also includes exact quantities and
distortion inversions, so the API must identify the kind of result explicitly
even though the main filename uses the word `bound`.

Comparison examples and numerical tests can be added when actual methods are
available. They do not need a separate directory for every model.

### Imports and package registration

Keep dependencies straightforward:

```text
category main evaluator file
            |
            v
category utils.py
            |
            v
root fbl/utils.py
            |
            v
existing Sionna PHY utilities / PyTorch
```

Each file may also import existing PHY components directly. Helpers used across
categories belong in root `utils.py`; root utilities do not import category
modules. Avoid importing one category's `utils.py` from another category.
Use explicit aliases when both utility files are needed, for example within
`achievability/achievability_bound.py`:

```python
from . import utils as ach_utils
from .. import utils as shared_utils
```

Once `fbl/__init__.py` is added, users can explicitly import
`sionna.phy.fbl` or `from sionna.phy.fbl import ...` without editing
`src/sionna/phy/__init__.py`. The current package discovery includes `sionna*`.
The parent currently does not eagerly import this new package, so use an
explicit FBL import.

## 3. Spectre manual-to-file coverage

The manual includes substantially more than finite-blocklength channel coding.
The mapping below covers its implemented topics and explicitly identifies
unfinished or restricted ones. Page numbers refer to the PDF body pages; its
table of contents is offset/stale. MATLAB names are given for traceability,
not as a requirement to preserve opaque Python naming or overloaded signatures.

| Manual chapter / pages | MATLAB methods to account for | Proposed destination and interpretation |
| --- | --- | --- |
| Ch. 2, pp. 7-9: real AWGN | `shannon_ach2`, `kappabeta_ach`, `gallager_ach` | `achievability/achievability_bound.py` + `achievability/utils.py`, AWGN section: cone-packing, kappa-beta, Gallager. Expose exact versus asymptotic kappa explicitly instead of an opaque `hack` argument. |
| Ch. 2 | `converse` | `converse/converse_bound.py` + `converse/utils.py`, AWGN section: metaconverse with the specified Gaussian auxiliary output and power constraint. A converse lower bound on error induces an upper bound on maximal rate. |
| Ch. 2 | `normapx_awgn`, `normapx_biawgn`; `cap_awgn`, `biawgn_stats` | `normapx/normal_apx.py` + `normapx/utils.py`, AWGN/BI-AWGN sections; shared statistics in root `utils.py`. |
| Ch. 2 | `plot_v3`, `universe/plot_universe` | A later comparison example using the public evaluators; empirical code data are comparison assets. |
| Ch. 3, pp. 11-14: block-memoryless no CSI | `DT_USTM`; `DT_USTM_Alamouti` | `achievability/achievability_bound.py` + `achievability/utils.py`, block-fading section; retain USTM and Alamouti inner-code assumptions and supported antenna ranges. |
| Ch. 3 | `MC_USTM_2x2`, `MC_USTM_4x4`, `compute_MC_2x2_telatar`; `MC_USTM_Alamouti` | `converse/converse_bound.py` + `converse/utils.py`, block-fading section. Separate fixed-covariance evaluations from optimization. Label the generalized-Telatar-conjecture restriction; do not present it as an unrestricted proven converse evaluation. |
| Ch. 3 | `outage_mi`, `outage_alamouti`, `outage_2x2_telatar`, `ergodic_USTM` | Root `utils.py`, capacity/outage section. `ergodic_USTM` is an input-specific MI lower bound on no-CSI capacity; preserve covariance and inner-code restrictions. |
| Ch. 4, pp. 15-17: quasi-static fading | `ach_simo_csir`, `ach_simo_nocsi`, `mimo_iso_ach` | `achievability/achievability_bound.py` + `achievability/utils.py`, quasi-static section: SIMO CSIR kappa-beta, SIMO no-CSI, isotropic-codeword MIMO. |
| Ch. 4 | `converse_simo`, `mimo_iso_conv` | `converse/converse_bound.py` + `converse/utils.py`, quasi-static section: retain the converse's CSIRT and codeword-constraint assumptions. |
| Ch. 4 | `normapprox_simo`, `normapprox_mimo_iso`; capacity/summary plotting | `normapx/normal_apx.py` + `normapx/utils.py`, quasi-static section; shared capacity/outage references in root `utils.py`. The approximation solves a fading-averaged Gaussian error equation, not a universal AWGN formula. |
| Ch. 5, pp. 19-20: long-term power | `awgn_ach_lt`, `awgn_conv_lt`, `awgn_na_lt` | The three main evaluator files and their local `utils.py` files, long-term-power sections. Preserve average-over-codewords power and the complex AWGN convention. |
| Ch. 5 section 5.2 | Quasi-static fading with perfect CSIT | Explicitly **under construction** in the manual. Reserve for later research; there is no implemented manual routine to port here. |
| Ch. 6, pp. 21-23: block-memoryless CSIR | `MIMO_achievability_CSIR` | `achievability/achievability_bound.py` + `achievability/utils.py`, CSIR block-fading section: beta-beta-type achievability with the normalized Gaussian/shell input construction and selected auxiliary output. |
| Ch. 6 | `MIMO_csir_normapx`, `capacity_and_dispersion`, `capacity_and_dispersion_mc` | `normapx/normal_apx.py` + `normapx/utils.py`, CSIR section; capacity/dispersion in root `utils.py`: second-order approximation; quadrature for the supported complex case, Monte Carlo for real/complex cases. |
| Ch. 7, pp. 25-27: energy/payload | `energy_awgn_ach`, `energy_awgn_conv`, `energy_awgn_normapx` | `energy/energy_bound.py` + `energy/utils.py`, AWGN section. Achievability may return required total energy for a payload; converse/approximation may return payload for a total energy. Name these quantities separately. |
| Ch. 7 | `energy_csir_conv`; `energy_nocsi_ach_ht`, `energy_nocsi_ach_ml`, `energy_nocsi_conv`, `energy_nocsi_normapx` | `energy/energy_bound.py` + `energy/utils.py`, Rayleigh section. Reuse AWGN achievability/approximation for the CSIR result only under the manual's stated result. Distinguish HT and ML no-CSI constructions. |
| Ch. 8, pp. 29-33: binary lossy source | `sc/BMS/Rstar`, `Dstar`: Shannon achievability, Gaussian approximation, sphere-covering/HT converse, general d-tilted converse, random-sphere and permutation achievability, Marton converse | `source_coding/source_coding_bound.py` + `source_coding/utils.py`, binary lossy section. Expose meaningful method names and rate/distortion evaluation; preserve which methods each original entry point actually supports. |
| Ch. 8: Gaussian lossy source | `sc/GMS/Rstar`, `Dstar`: Shannon achievability, Gaussian approximation, sphere-covering converse/achievability; `Rstar` additionally general converse and Rogers achievability | `source_coding/source_coding_bound.py` + `source_coding/utils.py`, Gaussian lossy section. Preserve squared-error distortion and source variance conventions. |
| Ch. 8: erased binary observation | `sc/BES/Rstar`: sphere-covering achievability/converse/Gaussian and `aeq`, `ceq`, `normaleq` | `source_coding/source_coding_bound.py` + `source_coding/utils.py`, erased-binary section. Distinguish the erased-observation problem from its surrogate problem. |
| Ch. 9, pp. 35-36 | `sc/varrate/Rstar`: `approx`, `exact`, `exacti`, `ie` | `source_coding/source_coding_bound.py` + `source_coding/utils.py`, variable-length section: approximation, exact minimum average rate, Erokhin function, and expected cutoff information. |
| Ch. 10, pp. 37-40: binary source/BSC | `jscc/BMS-BSC/Rstar`, `Dstar`: Gaussian, list/HT converse, joint coding, separation; `Rstar` additionally general converse, separation approximation, almost-lossless RCU; `Dstar` additionally fair-source random coding and uncoded transmission | `joint_source_channel/joint_source_channel_bound.py` + its `utils.py`, binary-source/BSC section. Distinguish rigorous bound, approximation, and achievable reference scheme. |
| Ch. 10: Gaussian source/AWGN | `jscc/GMS-AWGN/Rstar`, `Dstar`: approximation, list/HT converse, joint coding, separation; `Rstar` additionally general converse and separation approximation; `Dstar` additionally uncoded transmission and approximation | `joint_source_channel/joint_source_channel_bound.py` + its `utils.py`, Gaussian-source/AWGN section. Capacity, rate-distortion, and likelihood calculations use natural logarithms; the returned transmission ratio `k/n` is source symbols/channel-use, not nats. |
| Ch. 11, pp. 41-43: BSC | `dt_ach`, `rcu_ach`, `gallager_ach`; `converse`; `normapx` | The three main evaluator files and their local `utils.py` files, BSC sections: DT, RCU, Gallager, sphere-packing/equivalent metaconverse, and approximation. |
| Ch. 12, pp. 45-47: BEC | `dt_ach`, `gallager_ach`; `converse`; `normapx` | The three main evaluator files and their local `utils.py` files, BEC sections: DT/Gallager, BEC-specific metaconverse, approximation. The manual does not evaluate RCU for BEC; do not imply it does. |
| Ch. 13, pp. 49-52: BI-AWGN | `converse_mc` with `normal`, `On2`, `On3`, `full`, and forward-rate/inverse-SNR modes | `converse/converse_bound.py` + `converse/utils.py`, BI-AWGN section with explicit numerical mode; share applicable kernels through root `utils.py` with `normapx`. Asymptotic evaluations are approximations to a converse, not certified bounds by default. |
| Ch. 13 | `example_speff`, `example_per` | A later comparison example using public evaluators: rate, error-probability, and SNR-inversion examples with explicit quantities. |
| Ch. 14, pp. 53-56: lossless source | `p2p_rcu`, `p2p_han_converse`, `p2p_ht_converse`, `p2p_optimum` | `source_coding/source_coding_bound.py` + `source_coding/utils.py`, lossless section: achievability, converses, exact optimum. |
| Ch. 14: Slepian-Wolf | `sw_rcu`, `sw_han_converse`, `sw_ht_converse`, `gaussian_approx` | `source_coding/source_coding_bound.py` + `source_coding/utils.py`, Slepian-Wolf section: retain the supplied binary source family and symmetric sum-rate operating point; this is not an implementation of an arbitrary full rate region. |

For both Ch. 14 groups of functions, the supplied joint binary-source family is
`P00=a`, `P01=P10=P11=(1-a)/3`, with `0.25 <= a < 1`. The routines evaluate
the symmetric sum-rate point, with MATLAB rates in nats per source pair;
the example figure converts them to bits. This family should not be confused
with a doubly symmetric binary source with uniform marginals.

The surrounding Spectre checkout also has `block-fading-PAT-SNN/` routines
including RCUs/saddlepoint methods. They are not described by this attached
manual and should be tracked as an optional extension, not counted as completed
manual coverage.

## 4. API and mathematical conventions

### Quantities, units, and direction

Use explicit names such as `blocklength`, `error_probability`, `snr`,
`snr_db`, `log2_codebook_size`, `rate`, `coherence_length`, `num_blocks`,
`distortion`, and `energy_over_n0`. Avoid a single overloaded scalar return
whose meaning changes silently with the method.

For channel coding, prefer a documented canonical quantity such as
`log2_codebook_size`, with `rate = log2_codebook_size / blocklength` as an
explicit conversion. At fixed blocklength and target error probability,
achievability gives a lower bound and converse gives an upper bound on the
optimal codebook size/rate. The actual codebook size `M` is an integer; any
rounding after inversion must preserve the bound's direction. At fixed
codebook size, those orientations become
an upper and lower bound on error probability, respectively.

Source compression has a different optimization direction: an achievable
compression rate bounds the required minimum rate from above; a converse bounds
it from below. JSCC must distinguish source symbols from channel uses. Ch. 7
has no finite channel-use constraint at all. Keep these APIs separate rather
than force every result into `(n, epsilon, snr)`.

In particular, Gaussian JSCC's transmission rate `k/n` is dimensionless
(source symbols/channel-use), even though its capacity and rate-distortion
formulas use nats internally. Do not apply a nats-to-bits factor to that ratio.

Internal log-likelihoods may use natural logarithms; explicitly convert outputs
to bits when that is the public contract. Dispersion units must match capacity
units: nats squared versus bits squared differ by a squared conversion factor.
Do not assume all Spectre `Rstar` or `logM` outputs use the same units.

Expose named entry points such as `BSCDependenceTesting` or
`bsc_dt_log2_codebook_size`, not a universal `bound(channel, type, ...)`
function. The naming convention should be chosen once before implementation.
These names are illustrative proposals, not available API symbols.

### Model assumptions belong in every public docstring

Each evaluator must identify its theorem/reference, optimized quantity and
bound direction; real/complex channel and noise variance; input distribution;
per-symbol, per-codeword, per-coherence-block, or long-term power constraint;
CSI at each endpoint; antenna/coherence restrictions; logarithm/rate units;
supported shapes; numerical method/tolerances; and whether the returned value
is a theorem-bound evaluation, approximation, or statistical estimate.

For fading, define total channel uses as `blocklength = num_blocks *
coherence_length`. Spectre Ch. 6's MATLAB `n` means the **number of coherence
blocks**, unlike other chapters. Translate it at the boundary rather than reuse
that ambiguous name. Some no-CSI models also constrain power separately within
each block.

For real AWGN in Ch. 2, `Z ~ N(0,1)` and `snr = P`. Sionna's complex `AWGN`
uses `Z ~ CN(0,no)`. For real-noise validation, sampling with `normal()` is
clearer; alternatively the real component of complex noise with `no=2` has
variance 1. Do not carry a factor-of-two mismatch into capacities or dispersions.

### Approximation and numerical status

A universal helper can evaluate the second-order expression
`n*C - sqrt(n*V)*Qinv(epsilon)`, but higher-order corrections must be owned by
the model. Spectre uses a `+0.5*log2(n)` term for AWGN/BSC, omits it for BEC,
and uses a second-order expression for CSIR block fading. Quasi-static fading
uses an expectation/inversion over fading. Lossless source coding has a
different sign/orientation and correction. Handle zero-dispersion limits and
boundary probabilities with explicit limiting formulas or domain restrictions.

Use log-domain calculations (`logsumexp`, stable log differences, tail/survival
functions) for tiny error probabilities. A formula that is a bound in exact
arithmetic is not automatically a certified bound after Monte Carlo sampling,
asymptotic substitution, numerical integration, or unfinished optimization.
Return optional convergence/sample-count/uncertainty diagnostics and label the
method honestly. Increasing sample counts cannot by itself prove a rare-event
tail is accurate.

Discrete hypothesis tests often require randomization at a threshold to satisfy
the required acceptance probability. A continuous root finder without boundary
mass handling is insufficient. Numerical inversions should retain feasible
brackets and conservative endpoints where a bound direction matters.

Reuse PyTorch and Sionna utilities for the main tensor path. SciPy is already a
dependency in this checkout, so a CPU reference/fallback path need not introduce
a new dependency. Make such a path explicit: NumPy/SciPy conversion does not
preserve GPU execution, autograd, or `torch.compile`. Document supported
capabilities per implementation rather than claiming all bounds are
differentiable. Do not silently move tensors to CPU.

Keep sample generation and numerical calculation separate from plotting and
disk persistence. Spectre routines with `filename` sample caches or `persistent`
solver warm starts should become explicit caller-managed cache/state options,
if needed, so results do not depend on hidden call history.

## 5. Manual/source discrepancies to resolve while porting

The manual is a mathematical/source reference, not execution instructions.
Its example commands and performance suggestions are not requirements for this
Python package. Verify equations and MATLAB source together:

1. Ch. 13 calls the numerical `full` BI-AWGN mode forthcoming, but the local
   `bi-awgn/converse_mc.m` implements it. Its other modes remain asymptotic
   approximations.
2. The Ch. 8 binary-source selector printed as `martona` is `marton` in source
   and denotes a converse.
3. Section 4.6 prints an achievability function name in the converse signature;
   the actual source is `mimo_iso_conv.m`.
4. Ch. 3 headings retain obsolete USTM function suffixes. The local files are
   `DT_USTM.m`, `MC_USTM_2x2.m`, and `MC_USTM_4x4.m`.
5. Ch. 6's `MIMO_csir_normapx` source signature/return values differ from the
   manual example. Normalize the intended mathematical quantities in Python.
6. Several PDF citations render as `[?]`. Recover references from
   `documentation/manual.tex`, `documentation/refs.bib`, and source headers;
   do not propagate unresolved citation markers into Python docstrings.
7. Spectre `lib/logC.m` sometimes computes analytic bounds on a combinatorial
   quantity. Keep these distinct from an exact log-binomial implementation.

Record paper references and numerical-source provenance for each port. No
repository-level `LICENSE`/`COPYING` file was found in the inspected Spectre
checkout; some functions contain their own attribution/copyright notices,
including modified noncentral-t code inside `awgn/shannon_ach2.m` and the
Erseghe BI-AWGN implementation. An equation-based Python implementation and a
copy of MATLAB implementation bodies are different provenance cases. Preserve
the applicable notices and resolve code provenance before incorporating bodies.

## 6. Implementation sequence using the existing files

Retain all six existing paths and fill them according to the simple split:

| Existing file | Implementation responsibility |
| --- | --- |
| `achievability/achievability_bound.py` | All public achievability evaluators across channel models. |
| `achievability/utils.py` | All channel-specific achievability formulas and supporting operations, grouped into sections. |
| `converse/converse_bound.py` | All public converse evaluators across channel models. |
| `converse/utils.py` | All channel-specific converse formulas and supporting operations, grouped into sections. |
| `normapx/normal_apx.py` | All public normal-approximation evaluators across channel models. |
| `normapx/utils.py` | All model-specific approximation formulas and correction terms, grouped into sections. |

Add the initializers and root `utils.py` as implementations become available.
Keep shared code in root `utils.py` only when it is needed across categories.

Suggested implementation order:

1. Start with BSC/BEC. Implement their formulas in each category's `utils.py`
   and their public evaluators in the corresponding main file. Put reusable
   probability/HT calculations and channel statistics in root `utils.py`.
2. Add AWGN and BI-AWGN as sections in those same files. Keep special-function
   and CPU-reference paths explicit and validate the noise conventions.
3. Add block-fading, quasi-static, and long-term-power sections in the same
   category files, retaining their CSI, antenna, covariance, and power
   assumptions.
4. Add the energy, source-coding, and JSCC categories when those topics are
   implemented, following the same three-file pattern for each category.
5. Add numerical reference comparisons, tests, and examples alongside the
   implemented methods. Document fixture inputs, units, options, source
   revision, precision, tolerances, and random seeds.

The repository normally uses `test/unit/`. Under the current `fbl`-only scope,
tests can initially be kept in a single `fbl/test_fbl.py` and run explicitly
with `python -m pytest src/sionna/phy/fbl/test_fbl.py` once implemented. A single
`fbl/examples.py` can hold initial comparison examples. Neither file needs to be
split by channel model. Top-level test/documentation integration can be handled
in a later separately scoped change.

### Validation expected for each implementation

- Check a small deterministic case by enumeration or independent calculation,
  not only by comparing two implementations of the same formula.
- Store a few MATLAB reference values with exact inputs, units, method options,
  source revision, and tolerance. For Monte Carlo use uncertainty-aware
  comparisons rather than identical random-stream expectations.
- Check boundary/limiting regimes and only the monotonicities actually implied
  by the theorem. Compare compatible achievability/converse results under the
  same channel, CSI, power, and code constraints.
- Check scalar/batched broadcasting, dtype, CPU/device behavior, and seeded
  reproducibility. Test CUDA/compile/autograd only where supported; explicitly
  identify CPU-only numerical reference methods.
- Verify inversion round trips and feasible brackets. Check discrete
  hypothesis tests at boundary masses, zero-dispersion cases, and rare-event
  stability.

## 7. Evidence and references

The local checkout and supplied MATLAB/PDF files are the primary evidence for
this design. Useful Sionna source entry points are `block.py`, `object.py`,
`config.py`, `utils/numerics.py`, `utils/random.py`, `utils/tensors.py`,
`channel/awgn.py`, `channel/rayleigh_block_fading.py`, `mapping.py`, and
`fec/utils.py`, all under `src/sionna/phy`.

Official Sionna documentation corroborates the
[Block/Object integration conventions](https://nvlabs.github.io/sionna/phy/developer/dev_sionna_block.html)
and [configuration API](https://nvlabs.github.io/sionna/phy/api/config.html).
The local 2.2.0 source is authoritative when a documentation version differs.

Spectre references used here: the supplied `documentation/manual.pdf`, its
`manual.tex` and `refs.bib`, the MATLAB functions named in the coverage table,
and `lossless-sc/README.md`. This guide describes a proposed architecture and
coverage plan; numerical equivalence and bound correctness remain to be
established as each implementation is added.
