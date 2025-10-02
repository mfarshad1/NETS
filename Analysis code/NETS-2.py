import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.stats import scoreatpercentile
from collections import defaultdict
from numba import njit
import os
import time
from matplotlib import rc
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

# ----------------- System 2 (Figure 7 and 8).

# Plotting Setup
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
rc('text', usetex=False)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize=28)
rc('xtick', labelsize=24)
rc('ytick', labelsize=24)

# Parameters
num_bins = 60
bin_min = -10.58
bin_max = 10.58
bin_edges = np.linspace(bin_min, bin_max, num_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
skip_frames = 480
transitions_per_bin = 4000

@njit
def build_transition_matrix_with_indices(traj_data, traj_lengths, num_bins, bin_min, bin_max, skip_frames, sample_indices_by_id):
    transition_counts = np.zeros((num_bins, num_bins), dtype=np.float32)
    bin_counts = np.zeros(num_bins, dtype=np.int32)
    bin_size = (bin_max - bin_min) / num_bins
    for p in range(len(traj_data)):
        traj = traj_data[p]
        indices = sample_indices_by_id[p]
        for i in indices:
            y1 = traj[i]
            y2 = traj[i + skip_frames]
            bin_start = int((y1 - bin_min) / bin_size)
            bin_end = int((y2 - bin_min) / bin_size)
            if bin_start < 0 or bin_start >= num_bins or bin_end < 0 or bin_end >= num_bins:
                continue
            transition_counts[bin_start, bin_end] += 1
            bin_counts[bin_start] += 1
    return transition_counts, bin_counts

def read_lammps_trajectory(filename):
    positions_by_id = defaultdict(list)
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("ITEM: TIMESTEP"):
                _ = f.readline()
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                num_atoms = int(f.readline())
            elif line.startswith("ITEM: BOX BOUNDS"):
                _ = [f.readline() for _ in range(3)]
            elif line.startswith("ITEM: ATOMS"):
                headers = line.strip().split()[2:]
                id_idx = headers.index('id') if 'id' in headers else 0
                y_idx = headers.index('y') if 'y' in headers else 2
                for _ in range(num_atoms):
                    atom_line = f.readline().strip().split()
                    if len(atom_line) > max(id_idx, y_idx):
                        try:
                            atom_id = int(atom_line[id_idx])
                            y_pos = float(atom_line[y_idx])
                            positions_by_id[atom_id].append(y_pos)
                        except:
                            continue
    return {k: np.array(v, dtype=np.float32) for k, v in positions_by_id.items()}

def generate_sample_indices_balanced(traj_data, traj_lengths, skip_frames, num_bins, bin_min, bin_max, transitions_per_bin):
    bin_size = (bin_max - bin_min) / num_bins
    bin_to_indices = [[] for _ in range(num_bins)]
    all_transitions = []
    for traj_id, (traj, length) in enumerate(zip(traj_data, traj_lengths)):
        start_idx = max(0, length - 100000)
        for i in range(start_idx, length - skip_frames):
            y1 = traj[i]
            bin_start = int((y1 - bin_min) / bin_size)
            if 0 <= bin_start < num_bins:
                bin_to_indices[bin_start].append((traj_id, i))
    sample_indices_by_id = [list() for _ in traj_data]
    with open("transitions.txt", "w") as f:
        for bin_start in range(num_bins):
            candidates = bin_to_indices[bin_start]
            if len(candidates) < transitions_per_bin:
                chosen = candidates
            else:
                chosen = np.random.choice(len(candidates), transitions_per_bin, replace=False)
                chosen = [candidates[i] for i in chosen]
            for traj_id, idx in chosen:
                sample_indices_by_id[traj_id].append(idx)
                y1 = traj_data[traj_id][idx]
                y2 = traj_data[traj_id][idx + skip_frames]
                bin_end = int((y2 - bin_min) / bin_size)
                f.write(f"{bin_start} {bin_end}\n")
                all_transitions.append((bin_start, bin_end))
    for p in range(len(sample_indices_by_id)):
        sample_indices_by_id[p] = np.array(sample_indices_by_id[p], dtype=np.int32)
    return sample_indices_by_id, np.array(all_transitions, dtype=np.int32)

def compute_steady_state_power(T):
    eigvals, eigvecs = eig(T.T)
    idx = np.argmin(np.abs(eigvals - 1))
    ss = np.real(eigvecs[:, idx])
    ss = np.abs(ss)
    return ss / ss.sum()

def compute_histogram_distribution(positions_by_id, bin_edges):
    all_positions = np.concatenate(list(positions_by_id.values()))
    hist, _ = np.histogram(all_positions, bins=bin_edges)
    return 0.5 * (bin_edges[:-1] + bin_edges[1:]), hist / np.sum(hist)

def bootstrap_steady_state_CI(transitions, num_bins, n_bootstrap=200, confidence_level=95):
    alpha = (100 - confidence_level) / 2
    all_ss = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(transitions), size=len(transitions), replace=True)
        sampled = transitions[sample_idx]
        T_boot = np.zeros((num_bins, num_bins), dtype=np.float64)
        for start, end in sampled:
            T_boot[start, end] += 1
        row_sums = T_boot.sum(axis=1, keepdims=True)
        T_boot = np.nan_to_num(T_boot / row_sums)
        ss_boot = compute_steady_state_power(T_boot)
        all_ss.append(ss_boot)
    all_ss = np.array(all_ss)
    return (
        np.mean(all_ss, axis=0),
        scoreatpercentile(all_ss, alpha, axis=0),
        scoreatpercentile(all_ss, 100 - alpha, axis=0)
    )

def analyze_file_with_debug(input_file):
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return None
    print("\n📥 Reading trajectory...")
    t0 = time.time()
    positions_by_id = read_lammps_trajectory(input_file)
    print(f"⏱️ Trajectory read in {time.time() - t0:.2f} s")
    traj_data = list(positions_by_id.values())
    traj_lengths = np.array([len(t) for t in traj_data])
    print("\n🎯 Generating sample indices...")
    t0 = time.time()
    sample_indices_by_id, all_transitions = generate_sample_indices_balanced(
        traj_data, traj_lengths, skip_frames, num_bins, bin_min, bin_max, transitions_per_bin
    )
    print(f"⏱️ Sampling completed in {time.time() - t0:.2f} s")
    print("\n🔄 Building transition matrix...")
    t0 = time.time()
    T, bin_counts = build_transition_matrix_with_indices(
        traj_data, traj_lengths, num_bins, bin_min, bin_max, skip_frames, sample_indices_by_id
    )
    print(f"⏱️ Matrix built in {time.time() - t0:.2f} s")
    print(f"\n✅ Total transitions recorded: {int(np.sum(T))}")
    col_sums = T.sum(axis=0, keepdims=True)
    T = np.nan_to_num(T / col_sums)
    print("\n📈 Computing steady state...")
    t0 = time.time()
    ss = compute_steady_state_power(T)
    print(f"⏱️ Steady state computed in {time.time() - t0:.2f} s")
    print("\n📊 Computing histogram...")
    t0 = time.time()
    h_centers, h_dist = compute_histogram_distribution(positions_by_id, bin_edges)
    print(f"⏱️ Histogram computed in {time.time() - t0:.2f} s")
    return T, bin_centers, ss, h_centers, h_dist, all_transitions

# ------------------------- Main -------------------------
if __name__ == "__main__":
    file_large = "/afs/crc/group/whitmer/Data-MF-01/ansah/neq/binary_mixture_1000_mass/large.dat"
    file_small = "/afs/crc/group/whitmer/Data-MF-01/ansah/neq/binary_mixture_1000_mass/small.dat"

    result_large = analyze_file_with_debug(file_large)
    result_small = analyze_file_with_debug(file_small)

    if result_large and result_small:
        T_large, centers, ss_large, hc_large, hd_large, transitions_large = result_large
        T_small, _, ss_small, hc_small, hd_small, transitions_small = result_small

        # plot_results_updated() will be extended and replaced with new composite plot later

        # Plot 1: Side-by-side Transition Matrices
        pp = PdfPages("Figure 7.pdf")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5)) 
        plt.subplots_adjust(wspace=0.2)  # horizontal space between the two subplots
        for ax, T, label in zip(axes, [T_large, T_small], ["Large", "Small"]):
            im = ax.imshow(T, origin='lower', cmap='magma', extent=[0, num_bins, 0, num_bins])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label("Probability", fontsize=40)
            cbar.ax.tick_params(labelsize=24)
            ax.set_xlabel('Starting bin', fontsize=40)
            ax.set_ylabel('Ending bin', fontsize=40)
            ax.set_xticks(np.arange(0, 65, 10))
            ax.set_yticks(np.arange(60, -5, -10))
            ax.invert_yaxis()  # flips y-axis so 0 at bottom, 60 at top

            # ax.set_title(f'Transition Matrix ({label})', fontsize=24)
        plt.tight_layout()
        plt.show()
        pp.savefig(fig, bbox_inches='tight')
        pp.close()

        # Plot 2: Probability Distributions with Confidence Interval
        mean_large, low_large, high_large = bootstrap_steady_state_CI(transitions_large, num_bins)
        mean_small, low_small, high_small = bootstrap_steady_state_CI(transitions_small, num_bins)

        pp = PdfPages("Figure 8.pdf")
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        axes[0].axvspan(-5, -4, color='red', alpha=1.0)
        axes[0].axvspan(4, 5, color='blue', alpha=1.0)

        axes[0].plot(bin_centers, mean_large, label="Large-TM", color='green')
        axes[0].fill_between(bin_centers, low_large, high_large, color='green', alpha=0.5)
        axes[0].plot(hc_large, hd_large, '--', label="Large-BF", color='green')

        axes[0].plot(bin_centers, mean_small, label="Small-TM", color='black')
        axes[0].fill_between(bin_centers, low_small, high_small, color='black', alpha=0.5)
        axes[0].plot(hc_small, hd_small, '--', label="Small-BF", color='black')

        axes[0].set_ylabel(r"$P_{\mathrm{ss}}$", fontsize=40)
        # plt.title("Steady-State vs Histogram (Large and Small)")
        # plt.legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='16', loc='upper left', handlelength=0.4)


        # Plot 3: -0.74 log P Comparison
        def safe_log(x):
            return np.log(np.clip(x, 1e-12, None))

        axes[1].axvspan(-5, -4, color='red', alpha=1.0)
        axes[1].axvspan(4, 5, color='blue', alpha=1.0)
        
        axes[1].plot(bin_centers, -0.74 * safe_log(mean_large), label="Large-TM", color='green')
        axes[1].fill_between(bin_centers, -0.74 * safe_log(low_large), -0.74 * safe_log(high_large), color='green', alpha=0.5)
        axes[1].plot(bin_centers, -0.74 * safe_log(hd_large), '--', label="Large-BF", color='green')

        axes[1].plot(bin_centers, -0.74 * safe_log(mean_small), label="Small-TM", color='black')
        axes[1].fill_between(bin_centers, -0.74 * safe_log(low_small), -0.74 * safe_log(high_small), color='black', alpha=0.5)
        axes[1].plot(bin_centers, -0.74 * safe_log(hd_small), '--', label="Small-BF", color='black')

        axes[1].set_xlabel(r"$y\ (\sigma)$", fontsize=40)
        axes[1].set_ylabel(r"$F\ (k_{\mathrm{B}} T)$", fontsize=40, labelpad=33)
        # plt.title("Effective Free Energy (Large and Small)")
        # plt.legend(frameon=False, borderpad=0.1, labelspacing=0.2, columnspacing=0.2, borderaxespad=0.4, handletextpad=0.4, fontsize='16', loc='lower left', handlelength=0.4)
        # plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        pp.savefig(fig, bbox_inches='tight')
        pp.close()
