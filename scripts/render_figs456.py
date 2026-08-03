"""Render Figs. 4, 5 (full-tau verification) and 6 (universal functions)
from the contact data and the analytically constrained Pade approximants."""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d

from paths import DATA, FIGS

mpl.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm', 'axes.labelsize': 13, 'legend.fontsize': 9.5,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'lines.linewidth': 1.4,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.framealpha': 0.9, 'legend.edgecolor': '0.8',
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})
OUT = FIGS
COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

a0 = 128*np.sqrt(2)/(45*np.pi**3); a2 = 4*np.sqrt(2)/(9*np.pi)
b1 = -16*np.sqrt(2)/(3*np.pi**3); A_INF = 1/np.pi**1.5
a1B = -np.pi**1.5*b1

pn = np.load(DATA + '/pade_coeffs.npz'); qA, dB = pn['qA'], pn['dB']

def A_coeffs(q):
    c = [a0, 0.0, 0.0, 0.0, a2]; qs = list(q)
    p = [c[k] + sum(qs[j-1]*c[k-j] for j in range(1, min(k, 4)+1)) for k in range(5)]
    p.append(q[3]*A_INF)
    return np.array(p), np.array([1.0, *q])

def B_coeffs(dd):
    e = [0.0, a1B, 0.0, 0.0, 0.0]; ds = list(dd)
    a = [e[k] + sum(ds[j-1]*e[k-j] for j in range(1, min(k, 5)+1)) for k in range(5)]
    a.append(dd[4])
    return np.array(a), np.array([1.0, *dd])

pA, qAv = A_coeffs(qA); aB, dBv = B_coeffs(dB)

def A_pade(t):
    s = np.sqrt(np.asarray(t, dtype=float))
    return sum(pA[k]*s**k for k in range(6))/sum(qAv[k]*s**k for k in range(5))

def B_pade(t):
    s = np.sqrt(np.asarray(t, dtype=float))
    return -(s/np.pi**1.5)*sum(aB[k]*s**k for k in range(6))/sum(dBv[k]*s**k for k in range(6))

A_low = lambda t: a0 + a2*t**2
B_low = lambda t: b1*t                                   # no tau^2 term
A_high = lambda t: np.sqrt(t)/np.pi**1.5*(1 + (2-np.sqrt(3))/(2*t))
B_high = lambda t: -np.sqrt(t)/np.pi**1.5*(1 + (5-3*np.sqrt(3))/(2*t))

ab = np.load(DATA + '/AB_grid.npz')
tg, Ag, Bg = ab['tau'], ab['A'], ab['B']
Ai = interp1d(tg, Ag, kind='cubic'); Bi = interp1d(tg, Bg, kind='cubic')

d = np.load(DATA + '/C_full.npz')
tau, Ns, C, mask = d['tau'], list(d['N']), d['C'], d['mask']
MARK = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']

def ratio_fig(fname, use_pade, ylab):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    stats = []
    for i, N in enumerate(Ns):
        m = mask[i]
        if not m.any(): continue
        tg_ = tau[m]
        den = (A_pade(tg_)*N**2.5 + B_pade(tg_)*N**1.5) if use_pade \
              else (Ai(tg_)*N**2.5 + Bi(tg_)*N**1.5)
        R = C[i][m]/den
        col = cm.viridis(0.05 + 0.87*i/(len(Ns)-1))
        ax.plot(tg_, R, '-', color=col, marker=MARK[i], ms=3.2, mfc='white',
                mew=0.8, markevery=3, label=f'$N={N}$')
        stats.append((N, np.max(np.abs(R-1))))
    ax.axhline(1.0, color='k', ls='--', alpha=0.5, lw=0.8)
    ax.set_xlabel(r'$\tau$'); ax.set_ylabel(ylab)
    ax.set_xlim(0, 10); ax.legend(ncol=2, loc='best')
    fig.tight_layout(); fig.savefig(f'{OUT}/{fname}')
    return stats

s4 = ratio_fig('plot4_full_tau.pdf', False, r'$\mathcal{R}_N(\tau)$')
s5 = ratio_fig('plot5_pade.pdf', True, r'$\mathcal{R}_N^{(\mathrm{P})}(\tau)$')
print("Fig 4 (numerical A,B):  max|R-1| per N:")
for N, e in s4: print(f"   N={N:3d}: {100*e:.3f}%")
print(f"  -> worst overall {100*max(e for _, e in s4):.3f}%")
print("Fig 5 (Pade A,B):       max|R-1| per N:")
for N, e in s5: print(f"   N={N:3d}: {100*e:.3f}%")
print(f"  -> worst overall {100*max(e for _, e in s5):.3f}%")

# ---- Fig. 6 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
tl = tg[tg <= 0.5]; th = tg[(tg >= 3.0) & (tg <= 10)]
td = np.linspace(0.005, 10, 600)
for ax, data, lowfn, highfn, padefn, ylab in [
    (axes[0], Ag, A_low, A_high, A_pade, r'$A(\tau)$'),
    (axes[1], Bg, B_low, B_high, B_pade, r'$B(\tau)$')]:
    ax.plot(tg, data, '-', color=COLORS[0], lw=1.7, label='numerical')
    ax.plot(td, padefn(td), '--', color='k', lw=1.2, label='Pad\u00e9')
    ax.plot(tl, lowfn(tl), ':', color=COLORS[1], lw=1.7, label=r'$\tau\ll 1$')
    ax.plot(th, highfn(th), ':', color=COLORS[2], lw=1.7, label=r'$\tau\gg 1$')
    ax.set_xlabel(r'$\tau$'); ax.set_ylabel(ylab)
    ax.legend(loc='best'); ax.set_xlim(0, 10)
axes[0].text(0.03, 0.92, '(a)', transform=axes[0].transAxes, fontsize=12)
axes[1].text(0.03, 0.92, '(b)', transform=axes[1].transAxes, fontsize=12)
fig.tight_layout(w_pad=1.5)
fig.savefig(f'{OUT}/plot6_AB.pdf')
print("\nFig 6 rendered.")
selA = tg <= 10; selB = (tg >= 0.01) & (tg <= 10)
print(f"  A_P max rel err on [0,10]     = {100*np.max(np.abs(A_pade(tg[selA])/Ag[selA]-1)):.3f}%")
print(f"  B_P max rel err on [0.01,10]  = {100*np.max(np.abs(B_pade(tg[selB])/Bg[selB]-1)):.3f}%")
print(f"  A_P(100)={A_pade(100.):.4f} vs {np.sqrt(100)/np.pi**1.5:.4f};  "
      f"B_P(100)={B_pade(100.):.4f} vs {-np.sqrt(100)/np.pi**1.5:.4f}")
print("\n  numerator coeffs A:", np.array2string(pA, precision=6))
print("  denominator coeffs A:", np.array2string(qAv, precision=6))
print("  numerator coeffs B:", np.array2string(aB, precision=6))
print("  denominator coeffs B:", np.array2string(dBv, precision=6))
