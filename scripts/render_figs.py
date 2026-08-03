"""Render Figs. 2 and 3: two panels, distinct markers."""
import numpy as np, sys, os
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
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'lines.linewidth': 1.3,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'legend.framealpha': 0.9, 'legend.edgecolor': '0.8',
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.04,
})

ab = np.load(DATA + '/AB_grid.npz')
Ai = interp1d(ab['tau'], ab['A'], kind='cubic')
Bi = interp1d(ab['tau'], ab['B'], kind='cubic')
MARKERS = ['o', 's', '^', 'v', 'D']
OUT = FIGS

def panel_plot(ax, tau, Ns, C, all_N, label_prefix='', mark_stride=2):
    for k, N in enumerate(Ns):
        i = list(all_N).index(N)
        R = C[i]/(Ai(tau)*N**2.5 + Bi(tau)*N**1.5)
        col = cm.viridis(0.05 + 0.87*i/(len(all_N)-1))
        ax.plot(tau, R, '-', color=col, marker=MARKERS[k % 5], ms=3.6,
                mfc='white', mew=0.9, markevery=(k % mark_stride if mark_stride > 1 else 0, mark_stride),
                label=f'$N={N}$')
    ax.axhline(1.0, color='k', ls='--', alpha=0.5, lw=0.8)
    ax.set_xlabel(r'$\tau$')

def render_fig2():
    d = np.load(DATA + '/C_fig2.npz')
    tau, Ns, C = d['tau'], list(d['N']), d['C']
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    panel_plot(axes[0], tau, [10, 20, 30, 40, 50], C, Ns)
    panel_plot(axes[1], tau, [60, 70, 80, 90, 100], C, Ns)
    axes[0].set_ylabel(r'$\mathcal{R}_N(\tau)$')
    axes[0].set_xlim(0, 0.1); axes[1].set_xlim(0, 0.1)
    axes[0].set_ylim(0.980, 1.004)
    axes[1].set_ylim(0.9990, 1.0006)
    axes[1].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))
    axes[0].legend(loc='lower right', ncol=2, handlelength=1.6, columnspacing=0.9)
    axes[1].legend(loc='lower right', ncol=2, handlelength=1.6, columnspacing=0.9)
    axes[0].text(0.03, 0.93, '(a)', transform=axes[0].transAxes, fontsize=12)
    axes[1].text(0.03, 0.93, '(b)', transform=axes[1].transAxes, fontsize=12)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(f'{OUT}/plot2_low_tau.pdf')
    print('fig2 rendered')
    for i, N in enumerate(Ns):
        R = C[i]/(Ai(tau)*N**2.5 + Bi(tau)*N**1.5)
        print(f'  N={N:3d}: max|R-1|={100*np.max(np.abs(R-1)):.3f}%')

def render_fig3():
    f = DATA + '/C_fig3_scaled.npz'
    if not os.path.exists(f):
        print('fig3 data not ready'); return False
    d = np.load(f)
    tau, Ns, C = d['tau'], list(d['N']), d['C']
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    panel_plot(axes[0], tau, [10, 20], C, Ns)
    panel_plot(axes[1], tau, [30, 40, 50], C, Ns)
    axes[0].set_ylabel(r'$\mathcal{R}_N(\tau)$')
    axes[0].set_xlim(5, 10); axes[1].set_xlim(5, 10)
    axes[0].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.4f'))
    axes[1].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.5f'))
    axes[1].yaxis.set_major_locator(mpl.ticker.MaxNLocator(6))
    axes[0].legend(loc='best', handlelength=1.6)
    axes[1].legend(loc='best', ncol=1, handlelength=1.6)
    axes[0].text(0.03, 0.93, '(a)', transform=axes[0].transAxes, fontsize=12)
    axes[1].text(0.03, 0.93, '(b)', transform=axes[1].transAxes, fontsize=12)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(f'{OUT}/plot3_high_tau.pdf')
    print('fig3 rendered')
    for i, N in enumerate(Ns):
        R = C[i]/(Ai(tau)*N**2.5 + Bi(tau)*N**1.5)
        print(f'  N={N:3d}: max|R-1|={100*np.max(np.abs(R-1)):.4f}%  mean={100*np.mean(np.abs(R-1)):.4f}%')
    return True

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('2', 'both'): render_fig2()
    if which in ('3', 'both'): render_fig3()
