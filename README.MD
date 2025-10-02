# Non-equilibrium Trajectory Sampling (NETS) method for generating free-energy landscapes and steady-state distributions

This repository contains all code and data supporting the paper:

**"Non-equilibrium Trajectory Sampling (NETS) method for generating free-energy landscapes and steady-state distributions"**  
Mohsen Farshad*, Akwasi Nana Prempeh Ansah-Antwi*, Pedro H. Amorim Valença, Jonathan K. Whitmer  
*These authors contributed equally.*

## 🧠 Overview

This repository, alongside the main text, introduces a sparse-sampling workflow to reconstruct **steady-state distributions** and **effective free-energy landscapes** from short trajectory segments. We build transition counts \( C_{ij} \) at lag \( \delta t \), normalize to a column-stochastic matrix \( T_{ij} = C_{ij}/\sum_k C_{kj} \), obtain \( P \) as the right eigenvector of \( T \), and compute \( F(i) = -k_{\mathrm{B}}T \ln P(i) \). The workflow supports both equilibrium and non-equilibrium steady states.

### 📈 Method Flow:

1. Discretize configurations into states (choose CVs, bins)  
2. Generate short trajectories per state (SPS); select lag time \( \delta t \)  
3. Accumulate transition counts \( C_{ij}(\delta t) \)  
4. Normalize to \( T \) and obtain \( P \) (right eigenvector, eigenvalue 1)  
5. Compute \( F = -k_{\mathrm{B}}T \ln P \) and quantify uncertainty (bootstrap)

## 📁 Files

| File | Description |
|------|-------------|
| `Analysis code/NETS1.py` | Produces Figures **2**, **3**, **5**, **9** (NETS analysis). |
| `Analysis code/BF.py` | Produces Figure **6** (brute-force baseline, non-equilibrium). |
| `Analysis code/NETS2.py` | Produces Figures **7** and **8** (additional NETS analysis). |
| `LAMMPS code/system1-NETS.in` | **System 1 (equilibrium):** example input for NETS short-trajectory simulations. |
| `LAMMPS code/system2-NETS.in` | **System 2 (non-equilibrium):** example input for NETS short-trajectory simulations. |
| `LAMMPS code/system2-BF.in` | **System 2 (non-equilibrium):** example input for brute-force long simulations. |
