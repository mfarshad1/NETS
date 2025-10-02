#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 23:50:41 2025

@author: mfarshad
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc

# ---------------- Styling ----------------
rc('xtick', labelsize='16')
rc('ytick', labelsize='16')

# ----------------- System 2 (Figure 6).

# ---------------- Helpers ----------------
def load_blocks(filename: str, block_size: int = 60, max_firstcol: float = 60.0) -> np.ndarray:
    """
    Read a whitespace-delimited file, skipping comment lines.
    Keep rows with values[0] <= max_firstcol.
    Return a (num_blocks, block_size, num_cols) array.
    """
    rows = []
    with open(filename, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = [float(x) for x in s.split()]
            if vals[0] > max_firstcol:
                continue
            rows.append(vals)
    rows = np.asarray(rows, dtype=float)
    n_full = (rows.shape[0] // block_size) * block_size
    rows = rows[:n_full]  # drop incomplete tail
    if n_full == 0:
        raise ValueError(f"No full {block_size}-row blocks found in {filename}.")
    num_blocks = n_full // block_size
    return rows.reshape(num_blocks, block_size, -1)

def average_profile(blocks: np.ndarray) -> np.ndarray:
    """Average over time blocks -> (block_size, num_cols)."""
    return blocks.mean(axis=0)

def extract_time_series(blocks: np.ndarray, hot_idx: int = 14, cold_idx: int = 43, value_col: int = 3):
    """Extract per-block series at 'hot' and 'cold' spatial bins, and spatial average per block."""
    hot = blocks[:, hot_idx, value_col]
    cold = blocks[:, cold_idx, value_col]
    spatial_avg = blocks[:, :, value_col].mean(axis=1)
    return hot, cold, spatial_avg

# ---------------- File paths ----------------
base = "/afs/crc/group/whitmer/Data-MF-01/ansah/neq/binary_mixture_1000_mass"
fn_temp  = f"{base}/temp.dat"
fn_pe_L  = f"{base}/potential_avg_large.dat"
fn_ke_L  = f"{base}/kinetic_avg_large.dat"
fn_pe_S  = f"{base}/potential_avg_small.dat"
fn_ke_S  = f"{base}/kinetic_avg_small.dat"

# ---------------- Load and process ----------------
# Temperature (for profiles and hot/cold time series)
blocks_T   = load_blocks(fn_temp, block_size=60, max_firstcol=60.0)
prof_T     = average_profile(blocks_T)                    # (60, 4)
hot_T, cold_T, _ = extract_time_series(blocks_T)

# Potential/Kinetic (Large)
blocks_peL = load_blocks(fn_pe_L, block_size=60, max_firstcol=60.0)
blocks_keL = load_blocks(fn_ke_L, block_size=60, max_firstcol=60.0)
prof_peL   = average_profile(blocks_peL)                  # (60, 4)
prof_keL   = average_profile(blocks_keL)                  # (60, 4)
hot_peL, cold_peL, ave_peL = extract_time_series(blocks_peL)
hot_keL, cold_keL, ave_keL = extract_time_series(blocks_keL)

# Potential/Kinetic (Small)
blocks_peS = load_blocks(fn_pe_S, block_size=60, max_firstcol=60.0)
blocks_keS = load_blocks(fn_ke_S, block_size=60, max_firstcol=60.0)
prof_peS   = average_profile(blocks_peS)
prof_keS   = average_profile(blocks_keS)
hot_peS, cold_peS, ave_peS = extract_time_series(blocks_peS)
hot_keS, cold_keS, ave_keS = extract_time_series(blocks_keS)

# ---------------- Derived quantities ----------------
# Time axis: one sample per block (adjust scale as appropriate)
num_blocks = blocks_T.shape[0]
t = np.arange(num_blocks) * 1000  # scaling to mimic original

# Spatial coordinate (assume column 1 is y-midpoint)
y_mid = prof_T[:, 1]

# Total energy profiles (averaged over time) at each y-bin
E_L_profile = prof_peL[:, 3] + prof_keL[:, 3]
E_S_profile = prof_peS[:, 3] + prof_keS[:, 3]
E_tot_profile = E_L_profile + E_S_profile

# Total energy time series (spatially averaged per block)
E_L_time = ave_peL + ave_keL
E_S_time = ave_peS + ave_keS
E_tot_time = E_L_time + E_S_time

# ---------------- Plotting ----------------
pp = PdfPages('Figure 6.pdf')  # <- renamed for Figure 6
fig, axes = plt.subplots(2, 2, figsize=(8, 4))

# (0,0) Temperature profile (avg over time)
axes[0, 0].plot(prof_T[:, 1], prof_T[:, 3], '-', c='black')
axes[0, 0].set_xticks(np.arange(-10, 11, 5))
axes[0, 0].axvline(x=-5, color='red',  linestyle='--')
axes[0, 0].axvline(x=-4, color='red',  linestyle='--')
axes[0, 0].axvline(x= 4, color='blue', linestyle='--')
axes[0, 0].axvline(x= 5, color='blue', linestyle='--')
axes[0, 0].set_ylim(0.25, 1.31)
axes[0, 0].set_ylabel(r'$T$', rotation=90, labelpad=15, fontsize=28)
axes[0, 0].set_xticklabels([])

# (0,1) Hot/Cold temperature vs time
axes[0, 1].plot(t, hot_T,  '-', c='red')
axes[0, 1].plot(t, cold_T, '-', c='blue')
axes[0, 1].set_ylim(0.25, 1.31)
axes[0, 1].set_xticklabels([])

# (1,0) Total energy profile (avg over time)
axes[1, 0].plot(y_mid, E_tot_profile, '-', c='black')
axes[1, 0].set_xticks(np.arange(-10, 11, 5))
axes[1, 0].axvline(x=-5, color='red',  linestyle='--')
axes[1, 0].axvline(x=-4, color='red',  linestyle='--')
axes[1, 0].axvline(x= 4, color='blue', linestyle='--')
axes[1, 0].axvline(x= 5, color='blue', linestyle='--')
axes[1, 0].set_ylim(-13.5, -4.8)
axes[1, 0].set_xlabel(r'$y$ ($\sigma$)', rotation=0, labelpad=5, fontsize=28)
axes[1, 0].set_ylabel(r'$E$ ($\epsilon$)', rotation=90, labelpad=0, fontsize=28)
axes[1, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 1))

# (1,1) Total energy vs time (spatial average)
axes[1, 1].plot(t, E_tot_time, '-', c='black')
axes[1, 1].set_xlabel(r'Time Step', rotation=0, labelpad=10, fontsize=28)
axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 1))
axes[1, 1].ticklabel_format(style='sci', axis='x', scilimits=(0, 1))
axes[1, 1].set_ylim(-13.5, -4.8)

# Ticks for all panels
for ax in axes.flat:
    ax.tick_params(axis='both', which='major', labelsize=18)

plt.tight_layout()
pp.savefig(fig, bbox_inches='tight')
pp.close()
plt.show()
