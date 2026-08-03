"""Universal scaling functions A(tau), B(tau) + validations + experimental tables.
Independent re-implementation following the paper's integral representations,
with edge-refined composite quadrature (improves on the repo's plain GL-200).
"""
import numpy as np, json, time
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import zeta

from paths import DATA

# ---------- scaled chemical potential: tau*log(1+e^{xi/tau}) = 1 ----------
def solve_xi(tau):
    if tau < 1e-10: return 1.0
    # closed form: xi = tau*log(e^{1/tau}-1), overflow-safe
    if 1.0/tau > 30:
        return 1.0 + tau*np.log1p(-np.exp(-1.0/tau))
    return tau*np.log(np.expm1(1.0/tau))

def fermi(a):
    # 1/(e^a+1), overflow-safe, vectorized
    out = np.empty_like(a)
    pos = a > 0
    out[pos] = np.exp(-a[pos])/(1.0+np.exp(-a[pos]))
    out[~pos] = 1.0/(1.0+np.exp(a[~pos]))
    return out

def q_integrals(ui, tau, xi):
    """I0,I2,J0,J2,K0,K2 at scaled position u=ui (adaptive quad, edge breakpoints)."""
    a = xi - ui*ui
    ql = np.sqrt(max(a, 0.0) + 40*tau) + 4.0
    pts = [np.sqrt(a)] if a > 0 else None
    def ft(q):
        e = (q*q - a)/tau
        if e > 500: return 0.0
        if e < -500: return 1.0
        return 1.0/(np.exp(e)+1.0)
    kw = dict(limit=800)
    if pts: kw['points'] = pts
    res = []
    for w in (lambda f: f, lambda f: f*(1-f), lambda f: f*(1-f)*(1-2*f)):
        for mom in (lambda q: 1.0, lambda q: q*q):
            val, _ = quad(lambda q: mom(q)*w(ft(q)), 0, ql, **kw)
            res.append(2.0*val)   # even integrand
    I0, I2, J0, J2, K0, K2 = res
    return I0, I2, J0, J2, K0, K2

def u_nodes(tau, xi, n=160):
    """Composite GL nodes on [0, ulim] with refinement panels around the edge u_e."""
    ulim = np.sqrt(max(xi, 0) + 40*tau) + 4.0
    gl_x, gl_w = np.polynomial.legendre.leggauss(n)
    panels = []
    if xi > 0:
        ue = np.sqrt(xi); d = min(0.45*ue, max(12*tau/ (2*ue), 6*tau))
        edges = [0.0, max(ue-d, 0.0), min(ue+d, ulim), ulim]
        edges = sorted(set(edges))
    else:
        edges = [0.0, ulim]
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 1e-14: continue
        panels.append((0.5*(b-a)*gl_x + 0.5*(a+b), 0.5*(b-a)*gl_w))
    u = np.concatenate([p[0] for p in panels]); w = np.concatenate([p[1] for p in panels])
    return u, w

def compute_AB(tau, n=160):
    xi = solve_xi(tau)
    u, wu = u_nodes(tau, xi, n)
    vals = np.array([q_integrals(ui, tau, xi) for ui in u])  # (nu, 6)
    I0, I2, J0, J2, K0, K2 = vals.T
    # symmetric in u: integrals over full line = 2 * [0, ulim]
    A  = 2*np.sqrt(2)/np.pi**3 * 2*np.sum(I0*I2*wu)
    V2 = 2*np.sum(J0*wu); V3 = 2*np.sum(K0*wu)
    H  = 0.5*((K0*I2 + 2*J0*J2 + I0*K2)/V2 - V3/V2**2*(J0*I2 + I0*J2))
    B  = -2*np.sqrt(2)/np.pi**2 * 2*np.sum(H*wu)
    return A, B, V2, V3

# ---------- closed-form asymptotics (as in manuscript) ----------
A0   = 128*np.sqrt(2)/(45*np.pi**3)
a2   = 4*np.sqrt(2)/(9*np.pi)
b1   = -16*np.sqrt(2)/(3*np.pi**3)
bedge = -np.sqrt(2)/(4*np.pi**3)*((1-2*np.sqrt(2))*(2-np.sqrt(2))*zeta(-0.5)*zeta(1.5)
        - 2*(1-np.sqrt(2))**2*zeta(0.5)**2)
A_low  = lambda t: A0 + a2*t**2
B_low  = lambda t: b1*t + bedge*t**2
A_high = lambda t: np.sqrt(t)/np.pi**1.5*(1 + (2-np.sqrt(3))/(2*t))
B_high = lambda t: -np.sqrt(t)/np.pi**1.5*(1 + (5-3*np.sqrt(3))/(2*t))

if __name__ == '__main__':
    t0 = time.time()
    # ---- validation of edge coefficient from universal edge functions ----
    def Phi0(y): return quad(lambda p: 1/(np.exp(p*p-y)+1), -12, 12, limit=400)[0]
    def Phi2(y): return quad(lambda p: p*p/(np.exp(p*p-y)+1), -14, 14, limit=400)[0]
    h = 1e-5
    P  = lambda y: Phi0(y)*Phi2(y)
    Pp0 = (P(h)-P(-h))/(2*h)
    bedge_num = np.sqrt(2)/np.pi**3*Pp0
    print(f"[edge] P'(0) numeric = {Pp0:.8f};  b_edge numeric = {bedge_num:.8f}; closed form = {bedge:.8f}; rel.diff = {abs(bedge_num/bedge-1):.2e}")

    # ---- tau grids ----
    tau_fig2 = np.linspace(0.005, 0.1, 25)
    tau_fig3 = np.linspace(5.0, 10.0, 25)
    tau_dense = np.unique(np.concatenate([
        tau_fig2, np.linspace(0.12, 2.0, 30), np.linspace(2.2, 4.8, 14),
        tau_fig3, np.linspace(10.5, 12, 4),
        [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]]))
    A = np.zeros_like(tau_dense); B = np.zeros_like(tau_dense)
    V2 = np.zeros_like(tau_dense); V3 = np.zeros_like(tau_dense)
    for i, t in enumerate(tau_dense):
        A[i], B[i], V2[i], V3[i] = compute_AB(t)
    print(f"[grid] {len(tau_dense)} tau points in {time.time()-t0:.1f}s")

    # ---- validations ----
    for t in (0.005, 0.02, 0.05):
        i = np.argmin(abs(tau_dense-t))
        print(f"[low ] tau={tau_dense[i]:.3f}: A={A[i]:.7f} (Somm {A_low(tau_dense[i]):.7f}, rel {abs(A[i]/A_low(tau_dense[i])-1):.1e}) "
              f"B={B[i]:.7f} (Somm {B_low(tau_dense[i]):.7f}, rel {abs(B[i]/B_low(tau_dense[i])-1):.1e})")
    # b_edge extraction from numerics
    for t in (0.02, 0.04, 0.06):
        i = np.argmin(abs(tau_dense-t))
        be_x = (B[i]-b1*tau_dense[i])/tau_dense[i]**2
        print(f"[bedge-extract] tau={tau_dense[i]:.3f}: (B - b1*tau)/tau^2 = {be_x:.6f}  vs closed form {bedge:.6f}")
    for t in (5.0, 10.0):
        i = np.argmin(abs(tau_dense-t))
        print(f"[high] tau={t}: A={A[i]:.6f} (virial {A_high(t):.6f}, rel {abs(A[i]/A_high(t)-1):.1e}) "
              f"B={B[i]:.6f} (virial {B_high(t):.6f}, rel {abs(B[i]/B_high(t)-1):.1e}) V2={V2[i]:.5f} (-> pi={np.pi:.5f})")
    # convergence: refine
    for t in (0.01, 0.5, 5.0):
        Ar, Br, _, _ = compute_AB(t, n=260)
        i = np.argmin(abs(tau_dense-t))
        print(f"[conv] tau={t}: dA={abs(Ar/A[i]-1):.1e} dB={abs(Br/B[i]-1):.1e}")
    # ---- experimental tables ----
    Ns = [10, 25, 50, 100]
    taus_tab = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    ratio = {}
    print("\n[table] relative ensemble correction |B|N^{3/2}/(A N^{5/2}) in %:")
    print("        " + "".join(f"  N={N:<4d}" for N in Ns))
    for t in taus_tab:
        i = np.argmin(abs(tau_dense-t)); r = abs(B[i]/A[i])
        ratio[t] = r
        print(f"  tau={t:<4}: " + "".join(f" {100*r/N:6.2f}" for N in Ns))
    print("\n[B/A] tau, B/A:")
    for t in [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        i = np.argmin(abs(tau_dense-t)); print(f"   {t:5}: {B[i]/A[i]:+.4f}")

    # detectability: N*(tau) = |B/A|/eps for eps=5%, 2%
    print("\n[detect] N* such that correction = eps:")
    for t in [0.25, 0.5, 1.0, 2.0, 5.0]:
        i = np.argmin(abs(tau_dense-t)); r = abs(B[i]/A[i])
        print(f"   tau={t:4}: eps=5%: N*={r/0.05:5.1f}   eps=2%: N*={r/0.02:6.1f}")

    # thermometry bias: solve A(tau_fit) = A(tau) + B(tau)/N  for tau_fit
    from scipy.interpolate import interp1d
    Ai = interp1d(tau_dense, A, kind='cubic')
    print("\n[thermo] contact-thermometry bias delta_tau = tau_fit - tau (GCE/LDA fit ignoring B):")
    for N in (10, 25, 50):
        row = []
        for t in (0.15, 0.2, 0.3, 0.5, 1.0):
            i = np.argmin(abs(tau_dense-t))
            target = A[i] + B[i]/N
            try:
                tf = brentq(lambda x: Ai(x)-target, 0.005, 11.9)
                row.append((t, tf-t))
            except Exception:
                row.append((t, np.nan))
        print(f"   N={N:3d}: " + "  ".join(f"tau={t}: {d:+.4f}" for t, d in row))
    print(f"   (Sommerfeld estimate: delta_tau ~ -0.608/N -> N=10: -0.061, N=25: -0.024, N=50: -0.012)")

    np.savez(DATA + '/AB_grid.npz', tau=tau_dense, A=A, B=B, V2=V2, V3=V3,
             tau_fig2=tau_fig2, tau_fig3=tau_fig3)
    print(f"\nSaved AB_grid.npz. Total {time.time()-t0:.1f}s")
