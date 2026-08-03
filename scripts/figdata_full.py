"""Full-range C_N(tau) for Figs. 4 and 5 (underflow-safe recurrence, Nx=12M),
M_MAX = 5000 as documented in Appendix E, with the (N,tau) safety mask applied."""
import numpy as np, time
import figdata
from figdata3_scaled import setup_R_scaled

from paths import DATA

if __name__ == '__main__':
    T0 = time.time()
    M_MAX = 5000
    print(f"Building underflow-safe R (M={M_MAX}, Nx=12M) ...", flush=True)
    R = setup_R_scaled(M_MAX, fac=12, chunk=2500)
    print(f"R done at {time.time()-T0:.0f}s", flush=True)

    N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    tau_full = np.linspace(0.05, 10.0, 50)
    SAFETY = 1.2
    C = np.full((len(N_values), len(tau_full)), np.nan)
    mask = np.zeros_like(C, dtype=bool)
    for i, N in enumerate(N_values):
        tr = time.time()
        for j, t in enumerate(tau_full):
            M_full = int(N + 14*t*N + 20)
            if M_MAX < SAFETY*M_full:
                continue
            C[i, j], _ = figdata.contact_CE(N, t, R)
            mask[i, j] = True
        print(f"  N={N:3d}: {mask[i].sum()}/{len(tau_full)} points, {time.time()-tr:.0f}s "
              f"(total {time.time()-T0:.0f}s)", flush=True)
    np.savez(DATA + '/C_full.npz', tau=tau_full, N=N_values, C=C, mask=mask)
    print(f"ALL DONE {time.time()-T0:.0f}s", flush=True)
