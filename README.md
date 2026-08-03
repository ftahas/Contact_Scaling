# Contact_Scaling

Code accompanying "Canonical-ensemble scaling of Tan's contact in the trapped
Tonks-Girardeau gas" (F. T. Sant'Ana), Phys. Rev. A, manuscript AT12679.
Preprint: [arXiv:2605.15994](https://arxiv.org/abs/2605.15994)
[cond-mat.quant-gas].

The canonical contact of `N` hard-core bosons in a harmonic trap at reduced
temperature `tau = T/(N hbar omega)` obeys

```
C_N(tau) = A(tau) N^(5/2) + B(tau) N^(3/2) + O(N^(1/2))
```

with `A` and `B` universal functions of `tau` alone. This repository computes
`C_N` from the canonical contour-integral representation, evaluates `A` and `B`
from their integral representations, and reproduces the figures of the paper.

## Layout

| path | contents |
|---|---|
| `contact_scaling.ipynb` | main notebook: contact from the contour integral, `A`/`B`, Padé approximants, Figs. 1-5 |
| `contact_scaling_high_tau.ipynb` | high-temperature (`tau` in [5,10]) verification, Fig. 3 |
| `scripts/` | standalone scripts that generated the published figures and the numerical checks |

`scripts/` writes data to `data/` and figures to `figures/`; override with the
`CONTACT_DATA` and `CONTACT_FIGS` environment variables. Requires numpy, scipy,
matplotlib, and (for `verify_bedge.py`) mpmath.

Typical order:

```
python scripts/abgrid.py          # A(tau), B(tau) on a dense grid -> data/AB_grid.npz
python scripts/figdata.py         # canonical contact, Figs. 2 and 3 grids
python scripts/figdata3_scaled.py # high-tau contact, log-scaled recurrence
python scripts/figdata_full.py    # full-range C_N(tau) for Figs. 4 and 5
python scripts/pade_fit.py        # constrained minimax fit of the Padé approximants
python scripts/render_figs.py     # Figs. 2 and 3
python scripts/render_figs456.py  # Figs. 4 and 5
```

`scripts/verify_bedge.py` and `scripts/lowtau_coeffs.py` are checks rather than
figure producers and can be run on their own.

## Numerical notes

### Hermite recurrence

A recurrence seeded directly on `phi_0(x) = pi^(-1/4) exp(-x^2/2)` underflows
to zero in float64 for `|x| > 38.6`, which would zero every `phi_n` beyond that
radius, including high-lying levels whose classically allowed region extends
much further. `ho_wavefunctions` therefore carries an explicit per-point log
scale, so the orbitals remain accurate out to arbitrary radius; norms and
`<p^2>` agree with their analytic values to `~1e-12` relative at `M = 8600`.
The spatial grid uses `12M` points to resolve the highest orbitals.

### Low-temperature law for B

The `tau^2` edge term is zero: the inner (`y>0`) and outer (`y<0`) halves of
each Fermi-surface boundary layer contribute `+-(sqrt2/pi^3) P'(0)` and cancel,
so the low-temperature law is purely linear,

```
B(tau) = -(16 sqrt2 / 3 pi^3) tau + O(tau^3)
```

`scripts/verify_bedge.py` checks this three ways: analytic telescoping of the
edge integral, a high-precision fit at small `tau`, and direct evaluation of
the edge functions.

### Truncation safety factor

Plotted points are filtered on `M_MAX >= 1.2 * M_full(N, tau)`, so the
single-particle basis is comfortably larger than the occupied set at every
`(N, tau)` shown and truncation bias is negligible. The collapse is 0.002% to
0.06% for `N` from 10 to 50 over `tau` in [5,10].

### Padé approximants

`A_pade` and `B_pade` are rational functions of `sigma = sqrt(tau)` built so
that every asymptotic property derived analytically is imposed rather than
fitted (values and Sommerfeld slopes at `tau = 0`, absence of half-integer
powers at low `tau`, and the Boltzmann asymptotes), with only the crossover
region fitted by constrained minimax:

```
qA = [0.07615092, 1.33428628, 2.12047377, 1.31842142]
dB = [0.40773691, 0.43707782, 2.44381631, 2.95067127, 7.28346255]
```

Maximum relative error 0.24% for `A` on `tau` in [0,10] and 0.26% for `B` on
[0.01,10]; both pole-free on `tau >= 0` and asymptotically correct.

Notebook outputs are not stored in the repository. Re-run to regenerate.
