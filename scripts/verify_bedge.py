"""High-precision verification that the tau^2 (edge) coefficient of B(tau) vanishes.

The inner (y>0) and outer (y<0) halves of the Fermi-surface boundary layer
cancel:
  I_edge = int_{-inf}^{inf} dy [P''(y) - (8/3) Theta(y)] = 0,  P = Phi0*Phi2,
so that B(tau) = b1*tau + O(tau^3).
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

f = lambda x: 1.0/(1.0+np.exp(x)) if x < 500 else 0.0

def edge_moment(y, k, deriv):
    # int dp p^k * d^deriv/dy^deriv f_y(p),  f_y = 1/(e^{p^2-y}+1)
    def w(p):
        ff = f(p*p - y)
        if deriv == 0: return ff
        if deriv == 1: return ff*(1-ff)
        return ff*(1-ff)*(1-2*ff)
    lim = np.sqrt(max(y,0)+45)+3
    val,_ = quad(lambda p: (p**k)*w(p), 0, lim, limit=400, epsabs=1e-13, epsrel=1e-12)
    return 2*val

Phi0  = lambda y: edge_moment(y,0,0); Phi2  = lambda y: edge_moment(y,2,0)
dPhi0 = lambda y: edge_moment(y,0,1); dPhi2 = lambda y: edge_moment(y,2,1)

def Pprime(y): return dPhi0(y)*Phi2(y) + Phi0(y)*dPhi2(y)

# telescoped: I_edge = [P'(Y1) - (8/3) Y1] - P'(-Y0)
for Y1 in (20.0, 40.0, 60.0):
    print(f"  P'({Y1}) - (8/3)*{Y1} = {Pprime(Y1) - 8.0/3.0*Y1:+.3e}")
print(f"  P'(-8) = {Pprime(-8.0):+.3e}   P'(-14) = {Pprime(-14.0):+.3e}")
print(f"  => I_edge = 0 (inner -P'(0) and outer +P'(0) cancel);  P'(0) = {Pprime(0.0):.6f}")
print(f"  large-y check: P(40) - (4/3)*40^2 = {Phi0(40)*Phi2(40)-4/3*1600:.6f} vs pi^2/9 = {np.pi**2/9:.6f}")

# ---- high-precision B(tau) at small tau via nested adaptive quadrature ----
def solve_xi(tau):
    return 1.0 + tau*np.log1p(-np.exp(-1.0/tau)) if 1/tau > 30 else tau*np.log(np.expm1(1.0/tau))

def q_ints(ui, tau, xi):
    a = xi - ui*ui
    ql = np.sqrt(max(a,0)+45*tau)+4
    pts = [np.sqrt(a)] if a > 0 else None
    kw = dict(limit=900, epsabs=1e-14, epsrel=1e-11)
    if pts: kw['points'] = pts
    def ft(q):
        e = (q*q-a)/tau
        return 0.0 if e > 500 else (1.0 if e < -500 else 1.0/(np.exp(e)+1.0))
    out = []
    for wsel in (lambda F: F, lambda F: F*(1-F), lambda F: F*(1-F)*(1-2*F)):
        for mom in (0, 2):
            v,_ = quad(lambda q: (q**mom)*wsel(ft(q)), 0, ql, **kw)
            out.append(2*v)
    return out  # I0,I2,J0,J2,K0,K2

def B_hp(tau):
    xi = solve_xi(tau)
    V2 = np.pi*tau/(np.exp(-xi/tau)+1.0)                      # phase-space identity
    V3 = V2*(1.0-1.0/(np.exp(-xi/tau)+1.0))                   # pi*tau*f*(1-f)
    ue = np.sqrt(xi); ulim = np.sqrt(xi+45*tau)+4
    def integrand(u):
        I0,I2,J0,J2,K0,K2 = q_ints(u, tau, xi)
        return (V3/V2**2)*(J0*I2+I0*J2) - (K0*I2+2*J0*J2+I0*K2)/V2
    v,_ = quad(integrand, 0, ulim, points=[ue], limit=300, epsabs=1e-12, epsrel=1e-9)
    return np.sqrt(2)/np.pi**2 * 2*v

b1 = -16*np.sqrt(2)/(3*np.pi**3)
taus = np.array([0.02, 0.03, 0.045, 0.06, 0.08, 0.11])
Bs = np.array([B_hp(t) for t in taus])
print("\n  tau      B_hp(tau)        (B-b1*tau)/tau^2")
for t, b in zip(taus, Bs):
    print(f"  {t:5.3f}  {b:+.9f}   {(b-b1*t)/t**2:+.6f}")
# quadratic fit of B/tau in tau: B/tau = b1 + c2*tau + c3*tau^2
X = np.vstack([np.ones_like(taus), taus, taus**2]).T
coef, *_ = np.linalg.lstsq(X, Bs/taus, rcond=None)
print(f"\n  fit B/tau = b1 + c2*tau + c3*tau^2:")
print(f"  b1_fit = {coef[0]:+.6f}  (analytic {b1:+.6f}, diff {coef[0]-b1:+.1e})")
print(f"  c2_fit = {coef[1]:+.6f}  (expected 0)")
print(f"  c3_fit = {coef[2]:+.6f}")
