## 📁 Experimental Code Overview

This repository contains the Python implementations used to reproduce the numerical experiments in our study on random feature approximations of ReLU kernels. Each script corresponds to a specific experimental task described in the paper.

---

### 🔹 Task I: Empirical MSE vs. Theoretical Upper Bound

**File:** `mse_bound_verify_case_I.py`

This script verifies the finite-sample error bound derived for the i.i.d. Gaussian random feature estimator.  
It compares the empirical mean squared error (MSE) of the Monte Carlo kernel estimator with the theoretical upper bound as the number of random features varies.

---

### 🔹 Task II: i.i.d. Gaussian vs. Orthogonal Random Features

**File:** `IID_ORF_mse_comp.py`

This script compares the empirical MSE performance of:
- i.i.d. Gaussian random features, and  
- orthogonal random feature (ORF) ensembles.

The experiment highlights the variance reduction effect brought by orthogonalization under identical feature dimensions and sampling budgets.

---

### 🔹 Task III: Comparison of Different Random Feature Constructions

**File:** `diff_RF_comp.py`

This script evaluates and compares the empirical MSE of four different random feature constructions:
- i.i.d. Gaussian,
- i.i.d. Rademacher,
- dense orthogonal random features (Dense ORF),
- structured orthogonal random features (Structured ORF).

The goal is to assess both accuracy and robustness across different projection families.

---

### 🔹 Task IV: Computational Complexity and Runtime Comparison

**File:** `diff_ORF_time_comp.py`

This script compares the computational efficiency of dense ORF and structured ORF kernels, including:
- **Setup time:** random matrix generation and preprocessing;
- **Online runtime:** per-sample kernel evaluation time.

The experiment empirically validates the sub-quadratic complexity advantage of structured ORF constructions.

