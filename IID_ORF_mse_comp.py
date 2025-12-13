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

    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0

    rho = float(np.dot(x, y) / (norm_x * norm_y))
    # Clip for numerical stability
    rho = np.clip(rho, -1.0, 1.0)

    theta = math.acos(rho)

    return norm_x * norm_y / (2.0 * math.pi) * (
        math.sin(theta) + (math.pi - theta) * math.cos(theta)
    )


# -----------------------------
# 2. Helper to compute kernel estimate from a projection matrix G
# -----------------------------
def relu_kernel_from_projection(G, x, y):
    """
    Compute the Monte Carlo kernel estimate given a matrix of projection vectors G.

    Each row g_i of G represents one random feature, and we use:
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


# -----------------------------
# 3. i.i.d. Gaussian features
# -----------------------------
def relu_kernel_mc_gaussian(x, y, m, rng=None):
    """
    Monte Carlo estimator of k(x, y) using i.i.d. Gaussian features.

    g_i ~ N(0, I_d), independently.

    Parameters
    ----------
    x, y : 1D numpy arrays of shape (d,)
    m    : int, number of random features
    rng  : numpy.random.Generator (optional)

    Returns
    -------
    k_hat : float
    """
    if rng is None:
        rng = np.random.default_rng()

    d = x.shape[0]
    G = rng.standard_normal(size=(m, d))
    return relu_kernel_from_projection(G, x, y)


# -----------------------------
# 4. ORF features via block-orthogonal construction
# -----------------------------
def sample_orf_projection(m, d, rng=None):
    """
    Sample an ORF (Orthogonal Random Features) projection matrix G of shape (m, d).

    We use a block construction to handle m >= d:

      - Let B = ceil(m / d).
      - For each block b:
          * Sample a Gaussian matrix G_b in R^{d x d}.
          * Compute QR decomposition: G_b = Q_b R_b.
          * Use Q_b as an orthogonal matrix and scale rows by sqrt(d).
      - Stack all blocks vertically and keep the first m rows.

    Inside each block, rows are orthogonal and have the same norm (sqrt(d)).
    Different blocks are independent.

    Parameters
    ----------
    m : int
        Number of random features to generate.
    d : int
        Dimension of each feature vector.
    rng : numpy.random.Generator (optional)

    Returns
    -------
    G_orf : 2D numpy array of shape (m, d)
    """
    if rng is None:
        rng = np.random.default_rng()

    num_blocks = math.ceil(m / d)
    blocks = []

    for _ in range(num_blocks):
        # Sample a full Gaussian matrix and orthogonalize it
        G_block = rng.standard_normal(size=(d, d))
        Q_block, _ = np.linalg.qr(G_block)

        # Scale rows by sqrt(d) to match the variance of N(0, I_d)
        Q_block = math.sqrt(d) * Q_block
        blocks.append(Q_block)

    G_full = np.vstack(blocks)
    G_orf = G_full[:m, :]
    return G_orf


def relu_kernel_mc_orf(x, y, m, rng=None):
    """
    Monte Carlo estimator of k(x, y) using ORF projections.

    Parameters
    ----------
    x, y : 1D numpy arrays of shape (d,)
    m    : int, number of random features
    rng  : numpy.random.Generator (optional)

    Returns
    -------
    k_hat : float
    """
    if rng is None:
        rng = np.random.default_rng()

    d = x.shape[0]
    G_orf = sample_orf_projection(m, d, rng=rng)
    return relu_kernel_from_projection(G_orf, x, y)


# -----------------------------
# 5. Empirical MSE for a given m
# -----------------------------
def empirical_mse_two_ensembles(x, y, m, n_trials=100, seed=0):
    """
    Estimate the MSEs of hat{k}_m(x, y) for both ensembles:
      - i.i.d Gaussian projections
      - ORF projections

    Parameters
    ----------
    x, y      : 1D numpy arrays of shape (d,)
    m         : int, number of random features
    n_trials  : int, number of Monte Carlo repetitions
    seed      : int, base random seed

    Returns
    -------
    mse_gaussian : float
    mse_orf      : float
    """
    rng = np.random.default_rng(seed)
    k_true = relu_kernel_closed_form(x, y)

    sq_errors_gaussian = []
    sq_errors_orf = []

    for t in range(n_trials):
        # For fairness, use different seeds within the same generator
        # but do not couple Gaussian and ORF draws.
        k_hat_gaussian = relu_kernel_mc_gaussian(x, y, m, rng=rng)
        k_hat_orf = relu_kernel_mc_orf(x, y, m, rng=rng)

        sq_errors_gaussian.append((k_hat_gaussian - k_true) ** 2)
        sq_errors_orf.append((k_hat_orf - k_true) ** 2)

    mse_gaussian = float(np.mean(sq_errors_gaussian))
    mse_orf = float(np.mean(sq_errors_orf))

    return mse_gaussian, mse_orf


# -----------------------------
# 6. Main simulation
# -----------------------------
def main():
    # ----- Simulation hyperparameters -----
    d = 32                               # dimension of x and y (can be changed)
    m_values = np.arange(1000, 5001, 500)  # m = 1000, 1500, ..., 5000
    n_trials = 2000                        # number of repetitions (increase for smoother curves)

    # ----- Generate a fixed pair (x, y) -----
    rng_xy = np.random.default_rng(123)
    x = rng_xy.standard_normal(size=d)
    y = rng_xy.standard_normal(size=d)

    # Normalize to unit norm so ||x|| = ||y|| ≈ 1
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    print(f"||x||^2 = {np.linalg.norm(x)**2:.4f}, ||y||^2 = {np.linalg.norm(y)**2:.4f}")
    k_true = relu_kernel_closed_form(x, y)
    print(f"True kernel value k(x, y) = {k_true:.6f}")

    mse_gaussian_list = []
    mse_orf_list = []

    # ----- Sweep over m and estimate MSEs -----
    for m in m_values:
        print(f"Computing empirical MSEs for m = {m} ...")
        mse_gaussian, mse_orf = empirical_mse_two_ensembles(
            x, y, m, n_trials=n_trials, seed=1000 + m
        )
        mse_gaussian_list.append(mse_gaussian)
        mse_orf_list.append(mse_orf)

        print(f"  Gaussian MSE ≈ {mse_gaussian:.6e}, ORF MSE ≈ {mse_orf:.6e}")

    mse_gaussian_arr = np.array(mse_gaussian_list)
    mse_orf_arr = np.array(mse_orf_list)

    # ----- Save results to file -----
    results = {
        'm_values': m_values,
        'mse_gaussian': mse_gaussian_arr,
        'mse_orf': mse_orf_arr,
        'd': d,
        'n_trials': n_trials,
        'k_true': k_true,
        'norm_x_sq': np.linalg.norm(x)**2,
        'norm_y_sq': np.linalg.norm(y)**2
    }
    np.savez('iid_orf_mse_results.npz', **results)
    print("\n[Results saved to iid_orf_mse_results.npz]")

    # ----- Load and plot from saved data -----
    print("\n[Loading data from file for plotting...]")
    loaded_data = np.load('iid_orf_mse_results.npz')
    
    m_vals_plot = loaded_data['m_values']
    mse_gaussian_plot = loaded_data['mse_gaussian']
    mse_orf_plot = loaded_data['mse_orf']
    d_plot = int(loaded_data['d'])
    
    plt.figure(figsize=(7, 5))
    plt.plot(m_vals_plot, mse_gaussian_plot, marker='o', label="i.i.d. Gaussian MSE")
    plt.plot(m_vals_plot, mse_orf_plot, marker='s', label="ORF MSE")
    plt.yscale('log')
    plt.xlabel("Number of samples")
    plt.ylabel("Empirical MSE (log scale)")
    plt.title(f"Empirical MSE: i.i.d Gaussian vs ORF, d={d_plot}")
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"\n[Plot generated from saved data]")
    print(f"  - Dimension d: {d_plot}")
    print(f"  - Number of trials: {int(loaded_data['n_trials'])}")
    print(f"  - True kernel k(x,y): {loaded_data['k_true']:.6f}")

    # ----- (Optional) Plot the ratio ORF / Gaussian -----
    # plt.figure(figsize=(7, 5))
    # ratio = mse_orf_arr / mse_gaussian_arr
    # plt.plot(m_values, ratio, marker='d')
    # plt.xlabel("Number of samples")
    # plt.ylabel("MSE_ORF / MSE_Gaussian")
    # plt.title("Relative MSE: ORF vs i.i.d Gaussian")
    # plt.grid(True, ls='--', alpha=0.5)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
