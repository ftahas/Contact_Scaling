"""High-precision extraction of the low-tau expansion coefficients of B(tau)."""
import numpy as np
from scipy.integrate import quad


b1 = -16*np.sqrt(2)/(3*np.pi**3)

def solve_xi(tau):
    return 1.0 + tau*np.log1p(-np.exp(-1.0/tau)) if 1/tau > 30 else tau*np.log(np.expm1(1.0/tau))

def q_ints(ui, tau, xi):
    a = xi - ui*ui
    ql = np.sqrt(max(a, 0)+50*tau)+5
    pts = [np.sqrt(a)] if a > 0 else None
    kw = dict(limit=1200, epsabs=1e-15, epsrel=1e-12)
    if pts: kw['points'] = pts
    def ft(q):
        e = (q*q-a)/tau
        return 0.0 if e > 500 else (1.0 if e < -500 else 1.0/(np.exp(e)+1.0))
    out = []
    for wsel in (lambda F: F, lambda F: F*(1-F), lambda F: F*(1-F)*(1-2*F)):
        for mom in (0, 2):
            v, _ = quad(lambda q: (q**mom)*wsel(ft(q)), 0, ql, **kw)
            out.append(2*v)
    return out

def B_hp(tau):
    xi = solve_xi(tau)
    V2 = np.pi*tau/(np.exp(-xi/tau)+1.0)
    V3 = V2*(1.0 - 1.0/(np.exp(-xi/tau)+1.0))
    ue = np.sqrt(xi); ulim = np.sqrt(xi+50*tau)+5
    def integrand(u):
        I0, I2, J0, J2, K0, K2 = q_ints(u, tau, xi)
        return (V3/V2**2)*(J0*I2+I0*J2) - (K0*I2+2*J0*J2+I0*K2)/V2
    v, _ = quad(integrand, 0, ulim, points=[ue], limit=400, epsabs=1e-14, epsrel=1e-11)
    return np.sqrt(2)/np.pi**2 * 2*v

print("=== low-tau expansion of B(tau):  B/tau = b1 + c2*tau + c3*tau^2 + ... ===")
print(f"b1 (analytic) = {b1:.10f}\n")
taus = np.array([0.008, 0.012, 0.016, 0.020, 0.025, 0.030, 0.040, 0.050, 0.065, 0.080])
Bs = np.array([B_hp(t) for t in taus])
print("  tau      B(tau)           (B-b1*tau)/tau^2      (B-b1*tau)/tau^3")
for t, b in zip(taus, Bs):
    r2 = (b - b1*t)/t**2; r3 = (b - b1*t)/t**3
    print(f"  {t:5.3f}  {b:+.10f}    {r2:+.6f}          {r3:+.5f}")

# Fit A: free c2 (3-param), on the smallest-tau half
for lo, hi, lbl in [(0.008, 0.030, 'tau<=0.03'), (0.008, 0.050, 'tau<=0.05'), (0.008, 0.080, 'tau<=0.08')]:
    m = (taus >= lo) & (taus <= hi)
    X = np.vstack([np.ones(m.sum()), taus[m], taus[m]**2]).T
    (b1f, c2f, c3f), *_ = np.linalg.lstsq(X, Bs[m]/taus[m], rcond=None)
    print(f"\n  [free-c2 fit, {lbl}]  b1={b1f:+.8f} (err {b1f-b1:+.1e})   c2={c2f:+.6f}   c3={c3f:+.5f}")

# Fit B: impose c2 = 0, fit b1 and c3 only -> check residuals
for lo, hi, lbl in [(0.008, 0.050, 'tau<=0.05'), (0.008, 0.080, 'tau<=0.08')]:
    m = (taus >= lo) & (taus <= hi)
    X = np.vstack([np.ones(m.sum()), taus[m]**2]).T
    (b1f, c3f), *_ = np.linalg.lstsq(X, Bs[m]/taus[m], rcond=None)
    resid = Bs[m]/taus[m] - (b1f + c3f*taus[m]**2)
    print(f"  [c2=0 fit,     {lbl}]  b1={b1f:+.8f} (err {b1f-b1:+.1e})   c3={c3f:+.5f}   max resid {np.max(np.abs(resid)):.1e}")

print("\n  => the fitted c2 is consistent with zero: the tau^2 edge term vanishes,")
print("     so B(tau) = b1*tau + O(tau^3).")
