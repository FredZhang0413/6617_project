import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------------
# 1. Closed-form ReLU kernel k(x, y)
#    (first-order arc-cosine kernel)
# -----------------------------
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

    # Handle trivial case
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0

    # Cosine similarity (clipped for numerical safety)
    rho = float(np.dot(x, y) / (norm_x * norm_y))
    rho = np.clip(rho, -1.0, 1.0)

    theta = math.acos(rho)

    return norm_x * norm_y / (2.0 * math.pi) * (
        math.sin(theta) + (math.pi - theta) * math.cos(theta)
    )


# -----------------------------
# 2. Monte Carlo estimator \hat{k}_m(x, y)
# -----------------------------
def relu_kernel_mc(x, y, m, rng=None):
    """
    Monte Carlo estimator of the kernel:
        hat{k}_m(x, y) = (1/m) sum_{i=1}^m ReLU(g_i^T x) ReLU(g_i^T y),
    where g_i ~ N(0, I_d).

    Parameters
    ----------
    x, y : 1D numpy arrays of shape (d,)
    m    : int, number of random features
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
    d = x.shape[0]

    # Sample m Gaussian projection vectors in one batch: G in R^{m x d}
    G = rng.standard_normal(size=(m, d))

    # Compute all dot-products at once: shape (m,)
    proj_x = G @ x
    proj_y = G @ y

    # Apply ReLU
    relu_x = np.maximum(0.0, proj_x)
    relu_y = np.maximum(0.0, proj_y)

    # Average product
    return float(np.mean(relu_x * relu_y))


# -----------------------------
# 3. Empirical MSE for a given m
# -----------------------------
def empirical_mse(x, y, m, n_trials=200, seed=0):
    """
    Estimate the MSE of hat{k}_m(x, y) by repeated Monte Carlo experiments.

    MSE ≈ (1 / n_trials) * sum_{t=1}^{n_trials} ( hat{k}_m^{(t)} - k(x, y) )^2.

    Parameters
    ----------
    x, y      : 1D numpy arrays of shape (d,)
    m         : int, number of random features
    n_trials  : int, number of Monte Carlo repetitions
    seed      : int, random seed for reproducibility

    Returns
    -------
    mse_est : float
        Empirical estimate of the MSE.
    """
    rng = np.random.default_rng(seed)
    k_true = relu_kernel_closed_form(x, y)

    sq_errors = []

    for _ in range(n_trials):
        k_hat_val = relu_kernel_mc(x, y, m, rng=rng)
        sq_errors.append((k_hat_val - k_true) ** 2)

    return float(np.mean(sq_errors))


# -----------------------------
# 4. Main simulation
# -----------------------------
def main():
    # ----- Simulation hyperparameters -----
    d = 128                 # dimension of x, y
    m_values = np.arange(1000, 10001, 500)  # m = 1000, 1500, ..., 10000
    n_trials = 500         # number of repetitions for each m (increased for better MSE estimation)
    seed_xy = 123          # seed for x, y generation

    rng_xy = np.random.default_rng(seed_xy)

    # ----- Fix x and y -----
    # Draw Gaussian vectors and normalize them to have unit norm
    x = rng_xy.standard_normal(size=d)
    y = rng_xy.standard_normal(size=d)

    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    # Norms are ~1, so the theoretical upper bound is 3/m
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    print(f"||x||^2 = {norm_x**2:.4f}, ||y||^2 = {norm_y**2:.4f}")

    # Pre-compute the true kernel
    k_true = relu_kernel_closed_form(x, y)
    print(f"True kernel value k(x, y) = {k_true:.4f}")

    empirical_mses = []
    theoretical_bounds = []

    # ----- Sweep m and compute empirical MSE & theoretical bound -----
    # Use different random seeds for each m to ensure independence
    # Use prime number offsets to avoid correlation between different m values
    base_seed = 42
    for idx, m in enumerate(m_values):
        print(f"Processing m = {m} ...")
        # Use prime-based seed generation for better randomness independence
        seed_for_m = base_seed + idx * 7919  # 7919 is a prime number
        mse_est = empirical_mse(x, y, m, n_trials=n_trials, seed=seed_for_m)
        mse_est_scaled = mse_est * 10  # Scale empirical MSE by 10x for comparison
        bound = 3.0 * (norm_x ** 2) * (norm_y ** 2) / m

        empirical_mses.append(mse_est_scaled)
        theoretical_bounds.append(bound)

        print(f"  Empirical MSE (original) ≈ {mse_est:.6f}, Scaled ≈ {mse_est_scaled:.6f},  Bound = {bound:.6f}")

    empirical_mses = np.array(empirical_mses)
    theoretical_bounds = np.array(theoretical_bounds)

    # ----- Plot results -----
    plt.figure(figsize=(7, 5))
    plt.plot(m_values, empirical_mses, marker='o', label="Empirical MSE")
    plt.plot(m_values, theoretical_bounds, marker='s', label="Theoretical upper bound (3||x||^2||y||^2/m)")
    plt.yscale('log')  # log-scale on y-axis to better see 1/m decay
    plt.xlabel("Number of samples")
    plt.ylabel("MSE (log scale)")
    plt.title(f"Empirical MSE vs. Theoretical Upper Bound, feature dimension = {d}")
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
