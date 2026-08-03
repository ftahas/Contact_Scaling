"""Constrained minimax fit of the Pade approximants A_P(tau), B_P(tau).

All known analytic structure is imposed:
  A: A(0)=a0; no odd/half powers below tau^2; tau^2 coeff = a2; A -> sqrt(tau)/pi^{3/2}.
  B: slope b1; NO tau^{3/2}, tau^2, tau^{5/2} terms (the tau^2 term vanishes by the
     edge cancellation, c2 = 0); B -> -sqrt(tau)/pi^{3/2}.
Free parameters are fitted by minimax (Nelder-Mead on the max relative error).
"""
import numpy as np
from scipy.optimize import minimize

from paths import DATA

a0 = 128*np.sqrt(2)/(45*np.pi**3)
a2 = 4*np.sqrt(2)/(9*np.pi)
b1 = -16*np.sqrt(2)/(3*np.pi**3)
A_INF = 1/np.pi**1.5          # A ~ A_INF*sqrt(tau)
a1B = -np.pi**1.5*b1          # = 16*sqrt(2)/(3*pi^{3/2})

d = np.load(DATA + '/AB_grid.npz')
tau, Anum, Bnum = d['tau'], d['A'], d['B']
selA = tau <= 10.0
selB = (tau >= 0.01) & (tau <= 10.0)

# ---------------------------------------------------------------- A
# A_P = N(s)/D(s), s = sqrt(tau); N deg 5, D deg 4 (monic constant 1)
# series targets: c0=a0, c1=c2=c3=0, c4=a2 (i.e. a0 + a2*tau^2)
# asymptote: p5/q4 = A_INF
def A_coeffs(q):
    q1, q2, q3, q4 = q
    c = [a0, 0.0, 0.0, 0.0, a2]
    qs = [q1, q2, q3, q4]
    p = []
    for k in range(5):
        pk = c[k] + sum(qs[j-1]*c[k-j] for j in range(1, min(k, 4)+1))
        p.append(pk)
    p.append(q4*A_INF)          # p5 fixed by the large-tau asymptote
    return np.array(p), np.array([1.0, q1, q2, q3, q4])

def A_pade_fit(t, q):
    p, dd = A_coeffs(q)
    s = np.sqrt(np.atleast_1d(t))
    num = sum(p[k]*s**k for k in range(6))
    den = sum(dd[k]*s**k for k in range(5))
    return num/den

def A_obj(q):
    if np.any(np.array(q) < 0):        # keep D(s) > 0 on s>=0
        return 1e3
    val = A_pade_fit(tau[selA], q)
    if np.any(val <= 0):
        return 1e3
    return np.max(np.abs(val/Anum[selA] - 1))

best = None
rng = np.random.default_rng(0)
for trial in range(60):
    q0 = 10**rng.uniform(-1.2, 1.2, 4)
    r = minimize(A_obj, q0, method='Nelder-Mead',
                 options=dict(maxiter=6000, xatol=1e-11, fatol=1e-13))
    if best is None or r.fun < best.fun:
        best = r
qA = best.x
print(f"A_P fit: max rel err on [0,10] = {100*best.fun:.4f}%")
print(f"   q = {np.array2string(qA, precision=6)}")
p_, d_ = A_coeffs(qA)
print(f"   p = {np.array2string(p_, precision=6)}")
print(f"   checks: A_P(0)={A_pade_fit(1e-12, qA)[0]:.8f} vs a0={a0:.8f}")
for t in (0.05, 0.2, 1.0, 5.0, 10.0, 100.0):
    ap = A_pade_fit(t, qA)[0]
    print(f"   tau={t:7.2f}: A_P={ap:.6f}   asympt sqrt(tau)/pi^1.5={np.sqrt(t)/np.pi**1.5:.6f}"
          + (f"   num={Anum[np.argmin(abs(tau-t))]:.6f}" if t <= 12 else ""))

# ---------------------------------------------------------------- B
# B_P = -(1/pi^{3/2}) * s * N(s)/D(s), N deg 6 (no constant), D deg 5
# G = s*N/D must expand as a1B*s^2 + 0*s^3 + 0*s^4 + 0*s^5 + free*s^6
# asymptote: leading of s*N/D -> (a6/d5)*s^2?? -> we need B ~ -sqrt(tau)/pi^1.5 = -(s)/pi^1.5
#   so s*N/D -> s  =>  N/D -> 1  =>  deg N = deg D and ratio of leading coeffs = 1.
def B_coeffs(dd):
    d1, d2, d3, d4, d5 = dd
    # e_k = coefficients of N/D ; need e0=0, e1=a1B, e2=e3=e4=0, e5 free-> set by asymptote
    e = [0.0, a1B, 0.0, 0.0, 0.0]
    ds = [d1, d2, d3, d4, d5]
    a = []
    for k in range(5):
        ak = e[k] + sum(ds[j-1]*e[k-j] for j in range(1, min(k, 5)+1))
        a.append(ak)
    a.append(d5)      # a5 = d5  => N/D -> 1 at large s (both deg 5)
    return np.array(a), np.array([1.0, d1, d2, d3, d4, d5])

def B_pade_fit(t, dd):
    a, D = B_coeffs(dd)
    s = np.sqrt(np.atleast_1d(t))
    num = sum(a[k]*s**k for k in range(6))
    den = sum(D[k]*s**k for k in range(6))
    return -(s/np.pi**1.5)*num/den

def B_obj(dd):
    if np.any(np.array(dd) < 0):
        return 1e3
    val = B_pade_fit(tau[selB], dd)
    if np.any(val >= 0):
        return 1e3
    return np.max(np.abs(val/Bnum[selB] - 1))

bestB = None
for trial in range(60):
    d0 = 10**rng.uniform(-1.2, 1.2, 5)
    r = minimize(B_obj, d0, method='Nelder-Mead',
                 options=dict(maxiter=8000, xatol=1e-11, fatol=1e-13))
    if bestB is None or r.fun < bestB.fun:
        bestB = r
dB = bestB.x
print(f"\nB_P fit: max rel err on [0.01,10] = {100*bestB.fun:.4f}%")
print(f"   d = {np.array2string(dB, precision=6)}")
aB, DB = B_coeffs(dB)
print(f"   a = {np.array2string(aB, precision=6)}")
for t in (0.01, 0.1, 1.0, 5.0, 10.0, 100.0):
    bp = B_pade_fit(t, dB)[0]
    print(f"   tau={t:7.2f}: B_P={bp:+.6f}   asympt -sqrt(tau)/pi^1.5={-np.sqrt(t)/np.pi**1.5:+.6f}"
          + (f"   num={Bnum[np.argmin(abs(tau-t))]:+.6f}" if t <= 12 else ""))
print(f"   low-tau: B_P/tau at tau=1e-4 -> {B_pade_fit(1e-4, dB)[0]/1e-4:.8f}  (b1={b1:.8f})")
# verify the tau^2 coefficient vanishes
for t in (1e-3, 3e-3):
    print(f"   (B_P - b1*tau)/tau^2 at tau={t}: {(B_pade_fit(t,dB)[0]-b1*t)/t**2:+.5f}  (should ~ 0)")

np.savez(DATA + '/pade_coeffs.npz', qA=qA, dB=dB, errA=best.fun, errB=bestB.fun)
print("\nsaved pade_coeffs.npz")
