# Non-equilibrium Trajectory Sampling (NETS) Method for Generating Free-energy Landscapes and Steady-state Distributions

This repository contains all code and data supporting the paper:

**"Non-equilibrium Trajectory Sampling (NETS) Method for Generating Free-energy Landscapes and Steady-state Distributions"**  
Mohsen Farshad*, Akwasi Nana Prempeh Ansah-Antwi*, Pedro H. Amorim Valença, Jonathan K. Whitmer  
*These authors contributed equally.

## 🧠 Overview

This repository, alongside the main text, introduces a sparse-sampling workflow to reconstruct **steady-state distributions** and **effective free-energy landscapes** from short trajectory segments. We build transition counts \( C_{ij} \) at lag \( \delta t \), normalize to a column-stochastic matrix \( T_{ij} = C_{ij}/\sum_k C_{kj} \), obtain \( P_ss \) as the right eigenvector of \( T \), and compute \( F(i) = -k_{\mathrm{B}}T \ln P_ss(i) \). The workflow supports both equilibrium and non-equilibrium steady states.

### 📈 Method Flow:

1. Discretize configurations into states (choose CVs, bins)  
2. Generate short trajectories per state (SPS); select lag time \( \delta t \)  
3. Accumulate transition counts \( C_{ij}(\delta t) \)  
4. Normalize to \( T \) and obtain \( P_ss \) (right eigenvector, eigenvalue 1)  
5. Compute \( F = -k_{\mathrm{B}}T \ln P_ss \) and quantify uncertainty (bootstrap)

## 📁 Files

| File | Description |
|------|-------------|
| `Analysis code/NETS-1.py` | Produces Figures **2**, **3**, **5**, **9** (NETS analysis). |
| `Analysis code/data-NETS-1.zip` | **Data for `NETS-1.py`** (ready-to-run dataset for Figures 2, 3, 5, 9). |
| `Analysis code/BF.py` | Produces Figure **6** (brute-force baseline, non-equilibrium). |
| `Analysis code/NETS-2.py` | Produces Figures **7** and **8** (additional NETS analysis). |
| `LAMMPS code/system1-NETS.in` | **System 1 (equilibrium):** example input for NETS short-trajectory simulations. |
| `LAMMPS code/system2-NETS.in` | **System 2 (non-equilibrium):** example input for NETS short-trajectory simulations. |
| `LAMMPS code/system2-BF.in` | **System 2 (non-equilibrium):** brute-force long simulation input. **Use this to reproduce data for `BF.py` and `NETS-2.py`.** |

