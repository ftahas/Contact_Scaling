"""Fig. 3 data with underflow-safe (log-scaled) Hermite recurrence.
Carrying a per-point log scale keeps phi_n(x) accurate for |x| > 38.6, where a
directly seeded float64 recurrence would underflow to zero.
"""
import numpy as np, time
import figdata

from paths import DATA

def ho_chunk_scaled(M, x):
    """phi_n(x), phi_n'(x), n=0..M-1, underflow-safe via per-point log scaling."""
    Nx = len(x)
    ell = -x*x/2.0                      # log-scale; phi_hat_0 = pi^{-1/4}
    a = np.zeros(Nx)                    # phi_hat_{n-1}
    b = np.full(Nx, np.pi**-0.25)       # phi_hat_n
    phi_all = np.zeros((M+1, Nx))
    with np.errstate(under='ignore'):
        phi_all[0] = b*np.exp(ell)
        for n in range(0, M):
            if n == 0:
                c = np.sqrt(2.0)*x*b
            else:
                c = np.sqrt(2.0/(n+1))*x*b - np.sqrt(n/(n+1))*a
            m = np.maximum(np.abs(b), np.abs(c))
            big = m > 1e50
            small = (m > 0) & (m < 1e-50)
            resc = big | small
            if np.any(resc):
                s = np.where(resc, m, 1.0)
                ell = ell + np.log(s)
                b = b/s; c = c/s
            a, b = b, c
            phi_all[n+1] = b*np.exp(ell)
    phi = phi_all[:M]
    dphi = np.zeros((M, Nx))
    for n in range(M):
        tm = np.sqrt(n/2.0)*phi_all[n-1] if n > 0 else 0.0
        dphi[n] = tm - np.sqrt((n+1)/2.0)*phi_all[n+1]
    return phi, dphi

def setup_R_scaled(M_max, fac=12, chunk=2000):
    L = np.sqrt(2.0*M_max)+6.0; Nx = fac*M_max
    x = np.linspace(-L, L, Nx); dx = x[1]-x[0]
    P = np.zeros((M_max, M_max)); Q = np.zeros((M_max, M_max))
    t0 = time.time(); nch = (Nx+chunk-1)//chunk
    for k, s in enumerate(range(0, Nx, chunk)):
        xs = x[s:s+chunk]; w = np.full(len(xs), dx)
        if s == 0: w[0] = dx/2
        if s+chunk >= Nx: w[-1] = dx/2
        phi, dphi = ho_chunk_scaled(M_max, xs)
        P += (phi**2*w[None, :]) @ (dphi**2).T
        Q += ((phi*dphi)*w[None, :]) @ (phi*dphi).T
        if k % 8 == 0: print(f"  chunk {k+1}/{nch} ({time.time()-t0:.0f}s)", flush=True)
    return P - Q

if __name__ == '__main__':
    # quick validation of the scaled recurrence
    x = np.array([5.0, 30.0, 50.0, 70.0, 100.0])
    p1, d1 = figdata.ho_chunk(3200, x)
    p2, d2 = ho_chunk_scaled(3200, x)
    print("phi_3000 at x=5,30,50,70,100:")
    print("  plain :", p1[3000]); print("  scaled:", p2[3000], flush=True)
    # norm check: int phi_n^2 = 1 on a grid
    xg = np.linspace(-90, 90, 40000); dxg = xg[1]-xg[0]
    pg, _ = ho_chunk_scaled(3200, xg)
    for n in (100, 1000, 3000):
        print(f"  norm phi_{n}: {np.sum(pg[n]**2)*dxg:.10f}", flush=True)
    del pg

    T0 = time.time()
    MBIG = 8600
    print(f"Building underflow-safe R (M={MBIG}, Nx=12M) ...", flush=True)
    R = setup_R_scaled(MBIG)
    print(f"R done at {time.time()-T0:.0f}s", flush=True)
    np.save(DATA + '/R_scaled_diag.npy', np.diagonal(R)[:8:1])
    tau3 = np.linspace(5.0, 10.0, 25)
    N3 = [10, 20, 30, 40, 50]
    C3 = np.zeros((len(N3), len(tau3)))
    for i, N in enumerate(N3):
        tr = time.time()
        for j, t in enumerate(tau3):
            C3[i, j], Mf = figdata.contact_CE(N, t, R)
            assert 1.2*Mf <= MBIG
        print(f"N={N} done in {time.time()-tr:.0f}s", flush=True)
    np.savez(DATA + '/C_fig3_scaled.npz', tau=tau3, N=N3, C=C3)
    c, mf = figdata.contact_CE(20, 10.0, R)
    print(f"[bench] C(20,10) = {c:.3f} (scaling-law prediction 979.27)", flush=True)
    print(f"ALL DONE {time.time()-T0:.0f}s", flush=True)
