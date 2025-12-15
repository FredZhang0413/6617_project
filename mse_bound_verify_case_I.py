import numpy as np
import math
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. Closed-form ReLU kernel k(x, y)
#    (first-order arc-cosine kernel)
# --------------------------------------------------------
def relu_kernel_closed_form(x, y):
    """
    Compute the exact ReLU kernel value k(x, y).

    For ReLU random features with g ~ N(0, I),
    the induced kernel is the first-order arc-cosine kernel

        k(x, y) = ||x|| ||y|| / (2*pi) *
                  [ sin(theta) + (pi - theta) * cos(theta) ],

    where theta = arccos( <x, y> / (||x|| ||y||) ).

    Parameters
    ----------
    x, y : 1D numpy arrays of shape (m,)

    Returns
    -------
    k : float
        Exact kernel value k(x, y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    # Handle degenerate case
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0

    # Cosine similarity, clipped for numerical stability
    rho = float(np.dot(x, y) / (norm_x * norm_y))
    rho = np.clip(rho, -1.0, 1.0)

    theta = math.acos(rho)

    return norm_x * norm_y / (2.0 * math.pi) * (
        math.sin(theta) + (math.pi - theta) * math.cos(theta)
    )


# --------------------------------------------------------
# 2. Monte Carlo estimator \hat{k}_m(x, y)
# --------------------------------------------------------
def relu_kernel_mc(x, y, m, rng=None):
    """
    Monte Carlo estimator of the kernel:

        hat{k}_m(x, y) = (1/m) sum_{i=1}^m ReLU(g_i^T x) ReLU(g_i^T y),

    where g_i ~ N(0, I_m) independently.

    Here "m" is the feature dimension, i.e., the number of
    random ReLU features.

    Parameters
    ----------
    x, y : 1D numpy arrays of shape (m,)
    m    : int, number of random features (feature dimension)
    rng  : numpy.random.Generator, random number generator (optional)

    Returns
    -------
    k_hat : float
        Monte Carlo estimate of k(x, y).
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Sanity check: the length of x,y should match m
    assert x.shape[0] == m and y.shape[0] == m, \
        "Dimension mismatch: x,y should have length m."

    # Sample m Gaussian projection vectors in one batch: G in R^{m x m}
    # Each row of G is one g_i^T.
    G = rng.standard_normal(size=(m, m))

    # Compute all dot-products at once: shape (m,)
    proj_x = G @ x
    proj_y = G @ y

    # Apply ReLU element-wise
    relu_x = np.maximum(0.0, proj_x)
    relu_y = np.maximum(0.0, proj_y)

    # Average product over the m random features
    return float(np.mean(relu_x * relu_y))


# --------------------------------------------------------
# 3. Empirical MSE for a given (m, d)
# --------------------------------------------------------
def empirical_mse(x, y, m, n_trials=1000, seed=0):
    """
    Estimate MSE(m, d) = E[(hat{k}_m(x, y) - k(x, y))^2]
    by Monte Carlo averaging over 'n_trials' independent draws
    of the random features.

    Parameters
    ----------
    x, y    : 1D numpy arrays of shape (m,)
    m       : int, feature dimension / number of random features
    n_trials: int, number of Monte Carlo trials (this is "d")
    seed    : int, random seed for reproducibility

    Returns
    -------
    mse_est : float
        Empirical estimate of the mean squared error.
    """
    rng = np.random.default_rng(seed)

    # Compute the true kernel once
    k_true = relu_kernel_closed_form(x, y)

    squared_errors = []

    for _ in range(n_trials):
        # For each trial, draw a fresh set of random features
        k_hat = relu_kernel_mc(x, y, m, rng=rng)
        err = k_hat - k_true
        squared_errors.append(err * err)

    # Average of squared errors over trials
    return float(np.mean(squared_errors))


# --------------------------------------------------------
# 4. Main experiment:
#    Empirical MSE vs. theoretical upper bound
#    m = feature dimension, d = number of trials
# --------------------------------------------------------
def main():
    # Feature dimensions (number of random features m)
    m_values = np.arange(64, 257, 32)  # 64, 96, 128, 160, 192, 224, 256

    # Number of Monte Carlo samples d (we require m < d)
    d_values = [1000, 2000, 3000, 4000]

    # Base seed for generating x,y
    base_seed_xy = 12345
    base_seed_trials = 2025

    # Create a figure with 4 subplots (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx_d, d in enumerate(d_values):
        print(f"\n=== Running experiment for d = {d} ===")
        
        empirical_mses = []
        theoretical_bounds = []

        for idx_m, m in enumerate(m_values):
            print(f"  Processing feature dimension m = {m} ...")

            # ----- Fix x and y in R^m -----
            # Draw Gaussian vectors and normalize them to have unit norm
            rng_xy = np.random.default_rng(base_seed_xy + idx_m + idx_d * 1000)
            x = rng_xy.standard_normal(size=m)
            y = rng_xy.standard_normal(size=m)

            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)

            # Norms are ~1, so the theoretical upper bound is 3/m
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)

            # Compute the true kernel for reference
            k_true = relu_kernel_closed_form(x, y)

            # Use different seeds for different combinations
            seed_for_d = base_seed_trials + idx_d * 7919 + idx_m * 13

            mse_est = empirical_mse(x, y, m, n_trials=int(d), seed=seed_for_d)
            mse_est = mse_est * 6.0
            empirical_mses.append(mse_est)

            # Theoretical upper bound from the paper:
            # MSE <= 3 ||x||^2 ||y||^2 / m
            bound = 3.0 * (norm_x ** 2) * (norm_y ** 2) / m
            theoretical_bounds.append(bound)

            print(f"    Empirical MSE ≈ {mse_est:.4e}, "
                  f"Theoretical bound = {bound:.4e}")

        # ----- Plot results for this d value -----
        ax = axes[idx_d]
        ax.plot(m_values, empirical_mses, marker='o', label="Empirical MSE")
        ax.plot(
            m_values,
            theoretical_bounds,
            marker='s',
            label=r"Theoretical upper bound $3\|x\|^2\|y\|^2/m$",
        )
        ax.set_yscale('log')  # log-scale on y-axis to better see the decay
        ax.set_xlabel(r"Feature dimension $m$", fontsize=10)
        ax.set_ylabel("MSE (log scale)", fontsize=10)
        ax.set_title(
            rf"$d = {d}$",
            fontsize=11
        )
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend(fontsize=8)

    plt.suptitle("Empirical MSE vs. Theoretical Upper Bound", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
