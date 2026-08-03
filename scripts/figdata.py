"""Canonical contact C_N(tau) via contour integration (paper's method)
for Figs. 2 and 3. All plotted points satisfy M_R >= 1.2*M_full, so truncation
bias is negligible.
Run in background: writes progress to stdout, data to C_data.npz.
"""
import numpy as np, time, sys
from scipy.optimize import brentq

from paths import DATA

def ho_chunk(M, x):
    """phi_n(x), phi_n'(x) for n=0..M-1 on grid chunk x."""
    Nx = len(x); phi_all = np.zeros((M+1, Nx))
    phi_all[0] = np.pi**(-0.25)*np.exp(-x**2/2.0)
    if M >= 1: phi_all[1] = np.sqrt(2.0)*x*phi_all[0]
    for n in range(1, M):
        phi_all[n+1] = np.sqrt(2.0/(n+1))*x*phi_all[n] - np.sqrt(n/(n+1))*phi_all[n-1]
    phi = phi_all[:M]
    dphi = np.zeros((M, Nx))
    for n in range(M):
        tm = np.sqrt(n/2.0)*phi_all[n-1] if n > 0 else 0.0
        dphi[n] = tm - np.sqrt((n+1)/2.0)*phi_all[n+1]
    return phi, dphi

def setup_R(M_max, chunk=3000):
    L = np.sqrt(2.0*M_max)+6.0; Nx = max(4000, 6*M_max)
    x = np.linspace(-L, L, Nx); dx = x[1]-x[0]
    P = np.zeros((M_max, M_max)); Q = np.zeros((M_max, M_max))
    t0 = time.time()
    for s in range(0, Nx, chunk):
        xs = x[s:s+chunk]
        w = np.full(len(xs), dx)
        if s == 0: w[0] = dx/2
        if s+chunk >= Nx: w[-1] = dx/2
        phi, dphi = ho_chunk(M_max, xs)
        p2 = phi**2; d2 = dphi**2; v = phi*dphi
        P += (p2*w[None, :]) @ d2.T
        Q += (v*w[None, :]) @ v.T
        print(f"  R chunk {s//chunk+1}/{(Nx+chunk-1)//chunk} ({time.time()-t0:.0f}s)", flush=True)
    return P - Q

def contact_CE(N, tau, R, N_theta=None, safety=1.3):
    if N_theta is None: N_theta = max(256, 2*N+64)
    M_R = R.shape[0]; M_full = int(N+14*tau*N+20)
    M = max(M_R, M_full)
    beta = 1.0/(tau*N); q = np.exp(-np.arange(M)*beta)
    def eq(lr):
        rq = np.exp(lr)*q
        return np.sum(rq/(1.0+rq)) - N
    lr = brentq(eq, -max(200, (N-0.5)*beta*2+50), max(200, (N-0.5)*beta*2+50))
    r = np.exp(lr)
    th = 2.0*np.pi*np.arange(N_theta)/N_theta
    z = r*np.exp(1j*th)
    zq = z[:, None]*q[None, :]
    logXi = np.sum(np.log1p(zq), axis=1)
    w = np.exp(logXi - logXi[0].real)
    M_use = min(M, M_R, int(np.ceil(safety*M_full)))
    nb = (zq/(1.0+zq))[:, :M_use]
    G = np.sum(nb*(nb @ R[:M_use, :M_use]), axis=1)
    ph = np.exp(-1j*N*th)
    return (2.0/np.pi)*(np.mean(w*G*ph)/np.mean(w*ph)).real, M_full

if __name__ == '__main__':
    T0 = time.time()
    # ---------- Fig. 2 (low tau) ----------
    tau2 = np.linspace(0.005, 0.1, 25)
    N2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    print("Building R (M=400) for low-tau ...", flush=True)
    R_low = setup_R(400)
    C2 = np.zeros((len(N2), len(tau2)))
    for i, N in enumerate(N2):
        for j, t in enumerate(tau2):
            C2[i, j], Mf = contact_CE(N, t, R_low)
            assert 1.2*Mf <= 400, (N, t, Mf)
        print(f"fig2 N={N} done ({time.time()-T0:.0f}s)", flush=True)
    np.savez(DATA + '/C_fig2.npz', tau=tau2, N=N2, C=C2)
    print("fig2 data saved.", flush=True)

    # N_theta convergence spot check
    for (N, t) in [(100, 0.005), (100, 0.1), (10, 0.005)]:
        c1, _ = contact_CE(N, t, R_low)
        c2c, _ = contact_CE(N, t, R_low, N_theta=1024)
        print(f"[Ntheta] N={N} tau={t}: rel change 256->1024: {abs(c2c/c1-1):.2e}", flush=True)

    # ---------- Fig. 3 (high tau) ----------
    MBIG = 8600
    print(f"Building R (M={MBIG}) for high-tau ... (several minutes)", flush=True)
    R_big = setup_R(MBIG)
    print(f"R_big done at {time.time()-T0:.0f}s", flush=True)
    tau3 = np.linspace(5.0, 10.0, 25)
    N3 = [10, 20, 30, 40, 50]
    C3 = np.zeros((len(N3), len(tau3)))
    for i, N in enumerate(N3):
        tr = time.time()
        for j, t in enumerate(tau3):
            C3[i, j], Mf = contact_CE(N, t, R_big)
            assert 1.2*Mf <= MBIG, (N, t, Mf)
        print(f"fig3 N={N} done in {time.time()-tr:.0f}s ({time.time()-T0:.0f}s total)", flush=True)
    np.savez(DATA + '/C_fig3.npz', tau=tau3, N=N3, C=C3)

    # benchmark against manuscript appendix numbers
    c, Mf = contact_CE(20, 10.0, R_big)
    print(f"[bench] C(N=20, tau=10) = {c:.2f}  (manuscript: converged 979.10, scaling 979.27; M_full={Mf})", flush=True)
    for (N, t) in [(50, 10.0), (20, 10.0)]:
        c1, _ = contact_CE(N, t, R_big)
        c2c, _ = contact_CE(N, t, R_big, N_theta=768)
        print(f"[Ntheta] N={N} tau={t}: rel change: {abs(c2c/c1-1):.2e}", flush=True)
    print(f"ALL DONE in {time.time()-T0:.0f}s", flush=True)
