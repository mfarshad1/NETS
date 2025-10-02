#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 21:05:48 2025

@author: mfarshad
"""

# ============ Single import block ============
import re
import csv
import glob
import os
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg
from random import seed
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy import integrate
from scipy.integrate import quad
from scipy.integrate import cumulative_trapezoid as cumtrapz
from sympy import S, symbols, printing
# ============================================


# ----------------- Analytical force and potential energy solution for system 1(Figure 2).
energy = [1, 2, 3, 4, 5, 6, 7, 10]

fv = []
Analytical_free_energy = []
VW1 = []

names = [
    '$k_{B}T$', '$2\\ k_{B}T$', '$3\\ k_{B}T$', '$4\\ k_{B}T$',
    '$5\\ k_{B}T$', '$6\\ k_{B}T$', '$7\\ k_{B}T$', '$10\\ k_{B}T$',
]

for i in range(len(energy)):
    def WV(x):
        return -1 * ( ((x > -6) * (x < -5) * -energy[i]) +
                      ((x > 5) * (x < 6) *  energy[i]) + 0.1 ) if abs(x) > 1e-10 else x

    WV = np.vectorize(WV)

    X = np.linspace(-13, 13, 1_000_000)
    fv.append(WV(X))
    ww = WV(X)

    VW1.append(cumtrapz(ww, x=X, initial=0))
    # Store the most recent potential (avoids invalid list-array arithmetic)
    Analytical_free_energy.append(VW1[-1])

# ---------------- Figure 2 ----------------
position_z = np.linspace(-13, 13, 60)
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for j in (0, 1):
    for side in ("bottom", "left", "right", "top"):
        ax[j].spines[side].set_linewidth(0.5)

ax[0].set_xlabel('Z (σ)', fontstyle='italic')
ax[1].set_xlabel('Z (σ)', fontstyle='italic')
ax[0].set_ylabel('Force ($k_{B}T/σ$)')
ax[1].set_ylabel('F ($k_{B}T$)')

ax[0].yaxis.tick_left()
ax[1].yaxis.tick_left()

ax[0].xaxis.label.set_size(32)
ax[1].xaxis.label.set_size(32)
ax[0].yaxis.label.set_size(32)
ax[1].yaxis.label.set_size(32)
ax[0].tick_params(axis='both', which='major', labelsize=32)
ax[1].tick_params(axis='both', which='major', labelsize=32)

# Force curves (negative of fv)
ax[0].plot(X, -fv[0], label=names[0], lw=3)
ax[0].plot(X, -fv[1], label=names[1], lw=3)
ax[0].plot(X, -fv[2], label=names[2], lw=3)
ax[0].plot(X, -fv[3], label=names[3], lw=3)
ax[0].plot(X, -fv[4], label=names[4], lw=3)
ax[0].plot(X, -fv[5], label=names[5], lw=3)
ax[0].plot(X, -fv[6], label=names[6], lw=3)

# Free-energy curves
ax[1].plot(X, VW1[0], label=names[0], lw=3)
ax[1].plot(X, VW1[1], label=names[1], lw=3)
ax[1].plot(X, VW1[2], label=names[2], lw=3)
ax[1].plot(X, VW1[3], label=names[3], lw=3)
ax[1].plot(X, VW1[4], label=names[4], lw=3)
ax[1].plot(X, VW1[5], label=names[5], lw=3)
ax[1].plot(X, VW1[6], label=names[6], lw=3)
# ax[1].plot(X, VW1[7], label=names[7])  # available if desired

ax[0].legend(loc='lower right', fontsize=19.5, frameon=False,
             labelspacing=0.05, handlelength=1, ncol=2, columnspacing=0.9)

pp = PdfPages('Figure 2.pdf')
plt.tight_layout()
plt.show()
pp.savefig(fig, bbox_inches='tight')
pp.close()


# =========================
# System 1 (constant SPS and Simulation time length)
# =========================

F = []    # NETS free energy
M = []    # Analytical model free energy
Std = []  # Standard deviation

n = 60
m = 1000
k = 10000

# Load base datasets
for i in range(1, 8):
    F.append(np.loadtxt(f'v{i}_025p_sps_freeenergy'))
    M.append(np.loadtxt(f'v{i}_025p_sps_modelprofile'))
    Std.append(np.loadtxt(f'v{i}_025p_ps_std'))

# Helper for analytical profiles (2kBT and 3kBT)
def make_VW(barrier_kbt: float, num_points: int):
    Xv = np.linspace(-13, 13, num_points)
    ww = -1 * ( ((Xv > -6) & (Xv < -5)) * (-barrier_kbt)
                + ((Xv > 5) & (Xv < 6)) * (barrier_kbt)
                + 0.1 )
    VW = cumtrapz(ww, x=Xv, initial=0)
    return Xv, ww, VW

X2, ww2, VW1sys = make_VW(2.0, 10000)  # 2 kBT
X3, ww3, VW2sys = make_VW(3.0, 1000)   # 3 kBT

# ---------------- Figure 3 (formerly your first figure block) ----------------
position_z = np.linspace(-13, 13, 60)
Xplot = np.linspace(-13, 13, 1000)

fig, ax = plt.subplots(1, 1, figsize=(7, 5.7))
for side in ("bottom", "left", "right", "top"):
    ax.spines[side].set_linewidth(0.5)

ax.set_xlabel('Z (σ)', fontstyle='italic')
ax.set_ylabel('F ($k_{B}T$)', fontstyle='italic')
ax.xaxis.label.set_size(32)
ax.yaxis.label.set_size(32)
ax.yaxis.tick_left()
ax.tick_params(axis='both', which='major', labelsize=24)

# 1 kBT
ax.plot(Xplot, M[0].reshape([m, ]), color='green', linewidth=1.0)
ax.scatter(position_z, F[0].reshape([n, ]), color='green', marker='*', label='$k_{B}T$')
xc_2 = F[0].reshape([n, ]) + 2 * Std[0].reshape([n, ])
xv_2 = F[0].reshape([n, ]) - 2 * Std[0].reshape([n, ])
ax.fill_between(position_z, xv_2, xc_2, alpha=0.5, color='green')

# 2 kBT
ax.plot(X2, VW1sys.reshape([k, ]), color='pink', linewidth=1.0)
ax.scatter(position_z, F[1].reshape([n, ]), color='pink', marker='*', label='$2\\ k_{B}T$')
xc_2 = F[1].reshape([n, ]) + 2 * Std[1].reshape([n, ])
xv_2 = F[1].reshape([n, ]) - 2 * Std[1].reshape([n, ])
ax.fill_between(position_z, xv_2, xc_2, alpha=0.5, color='pink')

# 3 kBT
ax.plot(X3, VW2sys.reshape([m, ]), color='violet', linewidth=1.0)
ax.scatter(position_z, F[2].reshape([n, ]), color='violet', marker='*', label='$3\\ k_{B}T$')
xc_2 = F[2].reshape([n, ]) + 2 * Std[2].reshape([n, ])
xv_2 = F[2].reshape([n, ]) - 2 * Std[2].reshape([n, ])
ax.fill_between(position_z, xv_2, xc_2, alpha=0.5, color='violet')

# 4–7 kBT
colors = ['blue', 'black', 'red', 'purple']
labels = [r'$4\ k_{B}T$', r'$5\ k_{B}T$', r'$6\ k_{B}T$', r'$7\ k_{B}T$']
for idx, (ci, lab) in enumerate(zip(colors, labels), start=3):
    ax.plot(Xplot, M[idx].reshape([m, ]), color=ci, linewidth=1.0)
    ax.scatter(position_z, F[idx].reshape([n, ]), color=ci, marker='*', label=lab)
    xc = F[idx].reshape([n, ]) + 2 * Std[idx].reshape([n, ])
    xv = F[idx].reshape([n, ]) - 2 * Std[idx].reshape([n, ])
    ax.fill_between(position_z, xv, xc, alpha=0.5, color=ci)

ax.legend(loc='best', fontsize=18, frameon=False)

pp = PdfPages('Figure 3.pdf')
plt.tight_layout()
pp.savefig(fig, bbox_inches='tight')
pp.close()
plt.show()

# =========================
# SPS grids for various counts (Figure 5)
# =========================
F_5, Std_5 = [], []
F_1, Std_1 = [], []
F_2, Std_2 = [], []
F_50, Std_50 = [], []
F_8, Std_8 = [], []

for i in range(1, 3 + 1):
    F_5.append(np.loadtxt(f'v{i}_025p_500sps_freeenergy'))
    Std_5.append(np.loadtxt(f'v{i}_025p_500sps_std'))
    F_1.append(np.loadtxt(f'v{i}_025p_1000sps_freeenergy'))
    Std_1.append(np.loadtxt(f'v{i}_025p_1000sps_std'))
    F_2.append(np.loadtxt(f'v{i}_025p_2500sps_freeenergy'))
    Std_2.append(np.loadtxt(f'v{i}_025p_2500sps_std'))
    F_50.append(np.loadtxt(f'v{i}_025p_5000sps_freeenergy'))
    Std_50.append(np.loadtxt(f'v{i}_025p_5000sps_std'))
    F_8.append(np.loadtxt(f'v{i}_025p_8000sps_freeenergy'))
    Std_8.append(np.loadtxt(f'v{i}_025p_8000sps_std'))

# ---------------- Figure 5 ----------------
position_z = np.linspace(-13, 13, 60)
fig, ax = plt.subplots(3, 2, figsize=(10.8, 10))
fig.subplots_adjust(wspace=0)

# cosmetic spines
for r in range(3):
    for c in range(2):
        for side in ("bottom", "left", "right", "top"):
            ax[r][c].spines[side].set_linewidth(0.5)

# axis labels and ticks
ax[0][0].set_xlabel('Z (σ)')
ax[0][1].set_xlabel('Z (σ)')
ax[1][0].set_xlabel('Z (σ)')
ax[1][1].set_xlabel('Z (σ)')
ax[2][0].set_xlabel('$Z$ ($σ$)')
ax[2][1].set_xlabel('$Z$ ($σ$)')

ax[0][1].set_yticklabels([])
ax[1][0].set_ylabel('$F$ ($k_{B}T$)')
ax[1][1].set_yticklabels([])
ax[2][1].set_yticklabels([])

for r in range(3):
    for c in range(2):
        ax[r][c].yaxis.tick_left()
        ax[r][c].xaxis.label.set_size(32)
        ax[r][c].yaxis.label.set_size(32)
        ax[r][c].tick_params(axis='both', which='major', labelsize=24)

names_pan = ['$k_{B}T$', '$2\\ k_{B}T$', '$3\\ k_{B}T$']

def plot_panel(ax_handle, Fset, Sset):
    ax_handle.scatter(position_z, Fset[0].reshape([n, ]), color='green', marker='*', label=names_pan[0])
    ax_handle.scatter(position_z, Fset[1].reshape([n, ]), color='brown', marker='*', label=names_pan[1])
    ax_handle.scatter(position_z, Fset[2].reshape([n, ]), color='red', marker='*', label=names_pan[2])

    for Fi, Si, col in [(Fset[0], Sset[0], 'green'),
                        (Fset[1], Sset[1], 'yellow'),
                        (Fset[2], Sset[2], 'red')]:
        xc = Fi.reshape([n, ]) + 2 * Si.reshape([n, ])
        xv = Fi.reshape([n, ]) - 2 * Si.reshape([n, ])
        ax_handle.fill_between(position_z, xv, xc, alpha=0.5, color=col)

plot_panel(ax[0][0], F_5,  Std_5)   # 500
plot_panel(ax[0][1], F_1,  Std_1)   # 1000
plot_panel(ax[1][0], F_2,  Std_2)   # 2500
plot_panel(ax[1][1], F_50, Std_50)  # 5000
plot_panel(ax[2][0], F_8,  Std_8)   # 8000

# 10000 SPS panel (uses original F/Std for kBT, 2kBT, 3kBT)
ax[2][1].scatter(position_z, F[0].reshape([n, ]), color='green', marker='*', label=names_pan[0])
ax[2][1].scatter(position_z, F[1].reshape([n, ]), color='brown', marker='*', label=names_pan[1])
ax[2][1].scatter(position_z, F[2].reshape([n, ]), color='red', marker='*', label=names_pan[2])
for Fi, Si, col in [(F[0], Std[0], 'green'), (F[1], Std[1], 'yellow'), (F[2], Std[2], 'red')]:
    xc = Fi.reshape([n, ]) + 2 * Si.reshape([n, ])
    xv = Fi.reshape([n, ]) - 2 * Si.reshape([n, ])
    ax[2][1].fill_between(position_z, xv, xc, alpha=0.5, color=col)

ax[0][0].legend(loc='best', fontsize=19, frameon=False,
                labelspacing=0.05, handlelength=0.3, handletextpad=0.2, ncol=1, columnspacing=0.9)

# SPS captions
ax[0][0].text(0.05, 0.07, '500 SPS',   transform=ax[0][0].transAxes, fontsize=16, fontweight='bold', va='bottom')
ax[0][1].text(0.05, 0.07, '1000 SPS',  transform=ax[0][1].transAxes, fontsize=16, fontweight='bold', va='bottom')
ax[1][0].text(0.05, 0.07, '2500 SPS',  transform=ax[1][0].transAxes, fontsize=16, fontweight='bold', va='bottom')
ax[1][1].text(0.05, 0.07, '5000 SPS',  transform=ax[1][1].transAxes, fontsize=16, fontweight='bold', va='bottom')
ax[2][0].text(0.05, 0.07, '8000 SPS',  transform=ax[2][0].transAxes, fontsize=16, fontweight='bold', va='bottom')
ax[2][1].text(0.05, 0.07, '10000 SPS', transform=ax[2][1].transAxes, fontsize=16, fontweight='bold', va='bottom')

plt.subplots_adjust(hspace=0)
pp = PdfPages('Figure 5.pdf')
pp.savefig(fig, bbox_inches='tight')
pp.close()
plt.show()

# =========================
# System 2 (simulation time) — Figure 9
# =========================
M1 = np.loadtxt('noneq_large_freeenergy_new'); N1 = np.loadtxt('noneq_large_std')
A1 = np.loadtxt('noneq_small_freeenergy');     B1 = np.loadtxt('noneq_small_std')
M2 = np.loadtxt('noneq_large2_freeenergy');    N2 = np.loadtxt('noneq_large2_std')
A2 = np.loadtxt('noneq_small2_freeenergy');    B2 = np.loadtxt('noneq_small2_std')
M3 = np.loadtxt('noneq_large3_freeenergy');    N3 = np.loadtxt('noneq_large3_std')
A3 = np.loadtxt('noneq_small3_freeenergy');    a3 = np.loadtxt('noneq_small3_free'); B3 = np.loadtxt('noneq_small3_std')
M4 = np.loadtxt('noneq_large3_pdf');           A4 = np.loadtxt('noneq_small3_pdf')
M5 = np.loadtxt('noneq_large_bruteforce');     A5 = np.loadtxt('noneq_small_bruteforce')

# ---------------- Figure 10 ----------------
position_2 = np.linspace(-10.58, 10.58, 60)
y_min_F, y_max_F = 1.0, 3.0
x_lines = [-5, -4, 4, 5]

fig, ax = plt.subplots(1, 1, figsize=(7, 4))
for side in ("bottom", "left", "right", "top"):
    ax.spines[side].set_linewidth(0.5)

ax.set_xlabel('y (σ)', fontstyle='italic')
ax.set_ylabel('F ($k_{B}T$)', fontstyle='italic')
ax.xaxis.label.set_size(32)
ax.yaxis.label.set_size(32)
ax.yaxis.tick_left()
ax.tick_params(axis='both', which='major', labelsize=18)

# Filled regions
for x_line in x_lines:
    ax.axvline(x_line, color='none', linewidth=1, zorder=1)
ax.fill_betweenx([y_min_F, y_max_F], x_lines[0], x_lines[1], color='red',  alpha=1, zorder=0)
ax.fill_betweenx([y_min_F, y_max_F], x_lines[2], x_lines[3], color='blue', alpha=1, zorder=0)

# Curves and scatter
ax.scatter(position_2, M5.reshape([n, ]), color='green', marker='*', linewidth=1.0)
ax.plot(position_2, M1.reshape([n, ]), color='green', label='Heavier particle (NETS)')

ax.scatter(position_2, A5.reshape([n, ]), color='black', marker='*', linewidth=1.0)
ax.plot(position_2, A1.reshape([n, ]), color='black', label='lighter particle (NETS)')

# 95% CI bands
xc_2 = M1.reshape([n, ]) + 2 * N1.reshape([n, ])
xv_2 = M1.reshape([n, ]) - 2 * N1.reshape([n, ])
ax.fill_between(position_2, xv_2, xc_2, alpha=0.5, color='green')

xc_2_ = A1.reshape([n, ]) + 2 * B1.reshape([n, ])
xv_2_ = A1.reshape([n, ]) - 2 * B1.reshape([n, ])
ax.fill_between(position_2, xv_2_, xc_2_, alpha=0.5, color='black')

ax.set_ylim(1.83, 2.8)

pp = PdfPages('Figure 9.pdf')
plt.tight_layout()
pp.savefig(fig, bbox_inches='tight')
pp.close()
plt.show()
