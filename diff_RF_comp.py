import numpy as np
import math
import matplotlib.pyplot as plt


# ============================================================
# 1. Closed-form ReLU kernel (first-order arc-cosine kernel)
# ============================================================
def relu_kernel_closed_form(x, y):
    """
    Compute the exact kernel value k(x, y) induced by ReLU random features.

    k(x, y) = ||x|| ||y|| / (2*pi) * [ sin(theta) + (pi - theta) * cos(theta) ],
    where theta = arccos( <x, y> / (||x|| ||y||) ).

    Parameters
    ----------
    x, y : 1D numpy arrays of shape (d,)

    Returns
    -------
    k : float
        Exact kernel value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0

    rho = float(np.dot(x, y) / (norm_x * norm_y))
    rho = np.clip(rho, -1.0, 1.0)  # numerical safety

    theta = math.acos(rho)

    return norm_x * norm_y / (2.0 * math.pi) * (
        math.sin(theta) + (math.pi - theta) * math.cos(theta)
    )


# ============================================================
# 2. Given a projection matrix G, compute \hat{k}_m(x, y)
# ============================================================
def relu_kernel_from_projection(G, x, y):
    """
    Compute the Monte Carlo kernel estimate given a matrix of projection vectors G.

    Each row g_i of G is one projection vector and we use:
        hat{k}_m(x, y) = (1/m) sum_i ReLU(g_i^T x) ReLU(g_i^T y).

    Parameters
    ----------
    G : 2D numpy array of shape (m, d)
    x, y : 1D numpy arrays of shape (d,)

    Returns
    -------
    k_hat : float
    """
    proj_x = G @ x
    proj_y = G @ y

    relu_x = np.maximum(0.0, proj_x)
    relu_y = np.maximum(0.0, proj_y)

    return float(np.mean(relu_x * relu_y))


# ============================================================
# 3. Projection families
#    3.1 i.i.d. Gaussian
#    3.2 i.i.d. Rademacher
#    3.3 Dense ORF (Gaussian + QR)
#    3.4 Structured ORF (Hadamard-based)
# ============================================================

def sample_gaussian_projection(m, d, rng=None):
    """
    Sample a Gaussian projection matrix G in R^{m x d},
    with i.i.d. standard normal entries N(0, 1).
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal(size=(m, d))


def sample_rademacher_projection(m, d, rng=None):
    """
    Sample a Rademacher projection matrix G in R^{m x d},
    each entry is +1 or -1 with probability 1/2.
    Variance of each entry is 1, so Cov(g) ≈ I_d.
    """
    if rng is None:
        rng = np.random.default_rng()
    # rng.integers(0, 2) gives {0, 1}; map to {-1, +1}
    G = rng.integers(0, 2, size=(m, d))
    G = 2.0 * G - 1.0
    return G.astype(float)


def sample_dense_orf_projection(m, d, rng=None):
    """
    Sample an ORF (dense) projection matrix G in R^{m x d} using
    a Gaussian matrix followed by QR factorization.

    Steps:
      - Sample a Gaussian matrix A in R^{d x d}.
      - Compute QR: A = Q R, where Q is orthogonal (Q^T Q = I).
      - Scale rows of Q by sqrt(d) to match the typical norm of Gaussian rows.
      - If m > d, repeat the process with multiple blocks and stack them.

    Now supports m > d.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Calculate number of blocks needed
    num_blocks = math.ceil(m / d)
    blocks = []

    for _ in range(num_blocks):
        A = rng.standard_normal(size=(d, d))
        Q, _ = np.linalg.qr(A)          # Q is orthonormal
        Q = math.sqrt(d) * Q            # each row now has norm ~ sqrt(d)
        blocks.append(Q)

    # Stack all blocks and take first m rows
    G_full = np.vstack(blocks)
    G_orf = G_full[:m, :]
    return G_orf


# ---------- Hadamard construction ----------

def hadamard_matrix(d):
    """
    Construct the d x d Hadamard matrix (Sylvester construction).

    d must be a power of 2. Entries are +/- 1.

    H_1 = [1]
    H_{2n} = [[H_n,  H_n],
              [H_n, -H_n]]
    """
    if d == 1:
        return np.array([[1.0]])

    if (d & (d - 1)) != 0:
        raise ValueError("d must be a power of 2 for Hadamard matrix.")

    H = np.array([[1.0]])
    n = 1
    while n < d:
        # Block construction
        top = np.concatenate([H, H], axis=1)
        bottom = np.concatenate([H, -H], axis=1)
        H = np.concatenate([top, bottom], axis=0)
        n *= 2
    return H


def sample_structured_orf_projection(m, d, rng=None):
    """
    Sample a structured ORF projection matrix G in R^{m x d}
    using a Hadamard-based construction:

      U = H / sqrt(d)  (orthonormal Hadamard)
      D1, D2, D3 are diagonal sign matrices (Rademacher on the diagonal)
      W = D3 * U * D2 * U * D1 * U

    Finally we scale rows by sqrt(d) so that row norms are ~ sqrt(d).

    Now supports m > d by creating multiple independent blocks.
    d must be a power of 2.
    """
    if rng is None:
        rng = np.random.default_rng()

    if (d & (d - 1)) != 0:
        raise ValueError("d must be a power of 2 for structured ORF.")

    # Calculate number of blocks needed
    num_blocks = math.ceil(m / d)
    blocks = []

    for _ in range(num_blocks):
        # Hadamard matrix with entries +/-1
        H = hadamard_matrix(d)
        # Orthonormal version
        U = H / math.sqrt(d)

        # Three random diagonal sign matrices (different for each block)
        s1 = rng.integers(0, 2, size=d)
        s2 = rng.integers(0, 2, size=d)
        s3 = rng.integers(0, 2, size=d)

        s1 = 2.0 * s1 - 1.0
        s2 = 2.0 * s2 - 1.0
        s3 = 2.0 * s3 - 1.0

        D1 = np.diag(s1)
        D2 = np.diag(s2)
        D3 = np.diag(s3)

        # W = D3 * U * D2 * U * D1 * U
        W = D3 @ U @ D2 @ U @ D1 @ U

        # Scale rows by sqrt(d) so that row norm ~ sqrt(d),
        # comparable to Gaussian and dense ORF rows.
        W = math.sqrt(d) * W
        blocks.append(W)

    # Stack all blocks and take first m rows
    G_full = np.vstack(blocks)
    G_struct = G_full[:m, :]
    return G_struct


# ============================================================
# 4. Estimate mean and bias for a given family & m
# ============================================================
def estimate_mean_and_bias(family_name, m, d, x, y, n_trials=200, seed=0):
    """
    For a given projection family and m, estimate the mean of hat{k}_m(x, y)
    and its bias w.r.t. the true kernel.

    Parameters
    ----------
    family_name : str
        One of {"gaussian", "rademacher", "orf_dense", "orf_struct"}.
    m, d       : int
    x, y       : 1D arrays of length d
    n_trials   : int, number of Monte Carlo repetitions
    seed       : int, random seed

    Returns
    -------
    mean_k_hat : float
        Empirical mean of hat{k}_m over trials.
    abs_bias   : float
        Absolute bias |mean_k_hat - k_true|.
    mse        : float
        Empirical mean squared error E[(hat{k}_m - k_true)^2].
    """
    rng = np.random.default_rng(seed)
    k_true = relu_kernel_closed_form(x, y)

    estimates = []

    for t in range(n_trials):
        if family_name == "gaussian":
            G = sample_gaussian_projection(m, d, rng)
        elif family_name == "rademacher":
            G = sample_rademacher_projection(m, d, rng)
        elif family_name == "orf_dense":
            G = sample_dense_orf_projection(m, d, rng)
        elif family_name == "orf_struct":
            G = sample_structured_orf_projection(m, d, rng)
        else:
            raise ValueError(f"Unknown family: {family_name}")

        k_hat = relu_kernel_from_projection(G, x, y)
        estimates.append(k_hat)

    estimates = np.array(estimates, dtype=float)
    mean_k_hat = float(np.mean(estimates))
    abs_bias = float(abs(mean_k_hat - k_true))
    mse = float(np.mean((estimates - k_true) ** 2))

    return mean_k_hat, abs_bias, mse


# ============================================================
# 5. Main experiment
# ============================================================
def main():
    # ----- Global parameters -----
    d = 256                       # dimension (must be a power of 2 for structured ORF)
    m_values = np.arange(1000, 5001, 500)  # number of random features (m <= d)
    n_trials = 1000                # Monte Carlo repetitions per configuration

    # ----- Fix a pair (x, y) -----
    rng_xy = np.random.default_rng(42)
    x = rng_xy.standard_normal(size=d)
    y = rng_xy.standard_normal(size=d)

    # Normalize to unit norm to keep things simple
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    k_true = relu_kernel_closed_form(x, y)
    print(f"||x||^2 = {np.linalg.norm(x)**2:.4f}, ||y||^2 = {np.linalg.norm(y)**2:.4f}")
    print(f"True kernel value k(x, y) = {k_true:.6f}")

    families = ["gaussian", "rademacher", "orf_dense", "orf_struct"]
    family_labels = {
        "gaussian": "i.i.d. Gaussian",
        "rademacher": "i.i.d. Rademacher",
        "orf_dense": "Dense ORF (Gaussian + QR)",
        "orf_struct": "Structured ORF (Hadamard)"
    }

    # Store abs-bias and MSE for plotting
    abs_bias_results = {f: [] for f in families}
    mse_results = {f: [] for f in families}

    # ----- Sweep m and all families -----
    for m in m_values:
        print(f"\n=== m = {m} ===")
        for fam in families:
            mean_k, abs_bias, mse = estimate_mean_and_bias(
                fam, m, d, x, y, n_trials=n_trials, seed=1000 + m
            )
            abs_bias_results[fam].append(abs_bias)
            mse_results[fam].append(mse)

            print(f"{family_labels[fam]:>30s} | "
                  f"mean(k_hat)={mean_k:.6f}, |bias|={abs_bias:.3e}, MSE={mse:.3e}")

    # Convert to arrays for plotting
    for fam in families:
        abs_bias_results[fam] = np.array(abs_bias_results[fam])
        mse_results[fam] = np.array(mse_results[fam])

    # ----- Plot absolute bias vs m -----
    plt.figure(figsize=(7, 5))
    for fam in families:
        plt.plot(m_values, abs_bias_results[fam], marker='o', label=family_labels[fam])
    plt.yscale('log')
    plt.xlabel("Number of random features m")
    plt.ylabel("|E[hat{k}_m] - k(x, y)| (log scale)")
    plt.title("Kernel consistency across different projection families\n(abs bias vs m)")
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- (Optional) Plot MSE vs m -----
    plt.figure(figsize=(7, 5))
    for fam in families:
        plt.plot(m_values, mse_results[fam], marker='s', label=family_labels[fam])
    plt.yscale('log')
    plt.xlabel("Number of random features m")
    plt.ylabel("MSE (log scale)")
    plt.title("MSE of ReLU kernel estimate for different projection families")
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
