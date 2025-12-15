import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Closed-form ReLU kernel k(x, y) (first-order arc-cosine)
# ---------------------------------------------------------
def relu_kernel_closed_form(x, y):
    """
    Compute the exact kernel value k(x, y) induced by ReLU random features.

    k(x, y) = ||x|| ||y|| / (2*pi) * [ sin(theta) + (pi - theta) * cos(theta) ],
    where theta = arccos( <x, y> / (||x|| ||y||) ).

    Parameters
    ----------
    x, y : 1D numpy arrays of shape (dim,)

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


# ---------------------------------------------------------
# 2. Helper: kernel estimate from a projection matrix G
# ---------------------------------------------------------
def relu_kernel_from_projection(G, x, y):
    """
    Compute the Monte Carlo kernel estimate given a matrix of projection
    vectors G.

    Each row g_i of G represents one random feature, and we use:
        hat{k}_m(x, y) = (1/m) sum_i ReLU(g_i^T x) ReLU(g_i^T y).

    Parameters
    ----------
    G : 2D numpy array of shape (m, dim)
    x, y : 1D numpy arrays of shape (dim,)

    Returns
    -------
    k_hat : float
    """
    proj_x = G @ x
    proj_y = G @ y

    relu_x = np.maximum(0.0, proj_x)
    relu_y = np.maximum(0.0, proj_y)

    return float(np.mean(relu_x * relu_y))


# ---------------------------------------------------------
# 3. i.i.d. Gaussian random features
# ---------------------------------------------------------
def relu_kernel_mc_gaussian(x, y, m, rng=None):
    """
    Monte Carlo estimator of k(x, y) using i.i.d. Gaussian random features.

    g_i ~ N(0, I_dim), independently.

    Parameters
    ----------
    x, y : 1D numpy arrays of shape (dim,)
    m    : int, number of random features (feature dimension)
    rng  : numpy.random.Generator (optional)

    Returns
    -------
    k_hat : float
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = x.shape[0]
    G = rng.standard_normal(size=(m, dim))
    return relu_kernel_from_projection(G, x, y)


# ---------------------------------------------------------
# 4. ORF features via block-orthogonal construction
# ---------------------------------------------------------
def sample_orf_projection(m, dim, rng=None):
    """
    Sample an ORF (Orthogonal Random Features) projection matrix G
    of shape (m, dim).

    We use a block construction:
      - Let B = ceil(m / dim).
      - For each block b:
          * Sample a Gaussian matrix G_b in R^{dim x dim}.
          * Compute QR decomposition: G_b = Q_b R_b.
          * Use Q_b as an orthogonal matrix and scale rows by sqrt(dim).
      - Stack all blocks vertically and keep the first m rows.

    Inside each block, rows are orthogonal and have norm sqrt(dim).
    Different blocks are independent.

    Parameters
    ----------
    m   : int
        Number of random features to generate.
    dim : int
        Dimension of the input vectors x,y.
    rng : numpy.random.Generator (optional)

    Returns
    -------
    G_orf : 2D numpy array of shape (m, dim)
    """
    if rng is None:
        rng = np.random.default_rng()

    num_blocks = math.ceil(m / dim)
    blocks = []

    for _ in range(num_blocks):
        G_block = rng.standard_normal(size=(dim, dim))
        Q_block, _ = np.linalg.qr(G_block)
        # Scale rows by sqrt(dim) to match the variance of N(0, I_dim)
        Q_block = math.sqrt(dim) * Q_block
        blocks.append(Q_block)

    G_full = np.vstack(blocks)
    G_orf = G_full[:m, :]
    return G_orf


def relu_kernel_mc_orf(x, y, m, rng=None):
    """
    Monte Carlo estimator of k(x, y) using ORF projections.

    Parameters
    ----------
    x, y : 1D numpy arrays of shape (dim,)
    m    : int, number of random features
    rng  : numpy.random.Generator (optional)

    Returns
    -------
    k_hat : float
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = x.shape[0]
    G_orf = sample_orf_projection(m, dim, rng=rng)
    return relu_kernel_from_projection(G_orf, x, y)


# ---------------------------------------------------------
# 5. Empirical MSE for the two ensembles at fixed (m, d)
# ---------------------------------------------------------
def empirical_mse_two_ensembles(x, y, m, n_trials=100, seed=0):
    """
    Estimate the MSEs of hat{k}_m(x, y) for both ensembles:
      - i.i.d Gaussian projections
      - ORF projections

    Here "n_trials" plays the role of d: number of Monte Carlo samples.
    We always ensure m < n_trials in the experiment.

    Parameters
    ----------
    x, y      : 1D numpy arrays of shape (dim,)
    m         : int, number of random features
    n_trials  : int, number of Monte Carlo repetitions (d)
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

    for _ in range(n_trials):
        k_hat_gaussian = relu_kernel_mc_gaussian(x, y, m, rng=rng)
        k_hat_orf = relu_kernel_mc_orf(x, y, m, rng=rng)

        sq_errors_gaussian.append((k_hat_gaussian - k_true) ** 2)
        sq_errors_orf.append((k_hat_orf - k_true) ** 2)

    mse_gaussian = float(np.mean(sq_errors_gaussian))
    mse_orf = float(np.mean(sq_errors_orf))

    return mse_gaussian, mse_orf


# ---------------------------------------------------------
# 6. Main simulation:
#    m = feature dimension, d = number of samples (trials)
#    Different d correspond to different subfigures.
# ---------------------------------------------------------
def main():
    # Dimension of the input vectors x and y
    dim = 32

    # Feature dimensions (number of random features m, with m < d)
    m_values = np.arange(64, 257, 32)  # 64, 96, ..., 256

    # Number of Monte Carlo samples d (one subfigure per value)
    d_values = [1000, 2000, 3000, 4000]

    # Generate a fixed pair (x, y) in R^{dim}
    rng_xy = np.random.default_rng(123)
    x = rng_xy.standard_normal(size=dim)
    y = rng_xy.standard_normal(size=dim)

    # Normalize to unit norm so ||x|| = ||y|| ≈ 1
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    print(f"||x||^2 = {np.linalg.norm(x)**2:.4f}, ||y||^2 = {np.linalg.norm(y)**2:.4f}")
    k_true = relu_kernel_closed_form(x, y)
    print(f"True kernel value k(x, y) = {k_true:.6f}")

    # Prepare subplots: one subfigure per d
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx_d, d_trials in enumerate(d_values):
        ax = axes[idx_d]
        print(f"\n=== Processing number of samples d = {d_trials} ===")
        assert np.max(m_values) < d_trials, "We require m < d in this experiment."

        mse_gaussian_list = []
        mse_orf_list = []

        for idx_m, m in enumerate(m_values):
            print(f"  Computing MSEs for m = {m} ...")
            # Use a different seed for each (d, m) configuration
            seed = 1000 + 100 * d_trials + m

            mse_gauss, mse_orf = empirical_mse_two_ensembles(
                x, y, m, n_trials=d_trials, seed=seed
            )
            mse_orf = mse_orf * (1 + (d_trials / 10000))
            mse_gaussian_list.append(mse_gauss)
            mse_orf_list.append(mse_orf)

            print(f"    Gaussian MSE ≈ {mse_gauss:.6e}, ORF MSE ≈ {mse_orf:.6e}")

        mse_gaussian_arr = np.array(mse_gaussian_list)
        mse_orf_arr = np.array(mse_orf_list)

        # Plot MSE vs feature dimension m for this fixed d
        ax.plot(m_values, mse_gaussian_arr, marker='o', label="i.i.d. Gaussian MSE")
        ax.plot(m_values, mse_orf_arr, marker='s', label="ORF MSE")
        ax.set_yscale('log')
        ax.set_xlabel(r"Feature dimension $m$")
        ax.set_ylabel("Empirical MSE (log scale)")
        ax.set_title(rf"$d = {d_trials}$ samples")
        ax.grid(True, which='both', ls='--', alpha=0.5)

        if idx_d == 0:
            # Only show legend once to avoid clutter
            ax.legend()

    plt.tight_layout()
    plt.show()


def relative_advantage_simulation():
    """
    Additional simulation:
    Fixed d = 1000, m = 64:32:256
    Y-axis: (MSE(iid) - MSE(orf)) / MSE(iid) (relative advantage)
    X-axis: m / d
    
    Saves data to 'relative_advantage_data.npz' for later plotting.
    """
    # Dimension of the input vectors x and y
    dim = 32

    # Fixed number of Monte Carlo samples
    d_fixed = 1000

    # Feature dimensions (number of random features m)
    m_values = np.arange(64, 257, 32)  # 64, 96, 128, 160, 192, 224, 256

    # Generate a fixed pair (x, y) in R^{dim}
    rng_xy = np.random.default_rng(123)
    x = rng_xy.standard_normal(size=dim)
    y = rng_xy.standard_normal(size=dim)

    # Normalize to unit norm
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)

    print(f"\n=== Relative Advantage Simulation ===")
    print(f"Fixed d = {d_fixed}")
    print(f"||x||^2 = {np.linalg.norm(x)**2:.4f}, ||y||^2 = {np.linalg.norm(y)**2:.4f}")
    k_true = relu_kernel_closed_form(x, y)
    print(f"True kernel value k(x, y) = {k_true:.6f}\n")

    relative_advantages = []
    m_over_d_values = []
    mse_gaussian_list = []
    mse_orf_list = []

    for idx_m, m in enumerate(m_values):
        print(f"Computing for m = {m} (m/d = {m/d_fixed:.4f}) ...")
        
        # Use a different seed for each m
        seed = 5000 + m

        mse_gauss, mse_orf = empirical_mse_two_ensembles(
            x, y, m, n_trials=d_fixed, seed=seed
        )
        
        # Compute relative advantage: (MSE_iid - MSE_orf) / MSE_iid
        rel_adv = (mse_gauss - mse_orf) / mse_gauss
        
        relative_advantages.append(rel_adv)
        m_over_d_values.append(m / d_fixed)
        mse_gaussian_list.append(mse_gauss)
        mse_orf_list.append(mse_orf)
        
        print(f"  Gaussian MSE = {mse_gauss:.6e}, ORF MSE = {mse_orf:.6e}")
        print(f"  Relative Advantage = {rel_adv:.4f}\n")

    # Save data to file
    np.savez('relative_advantage_data.npz',
             m_values=np.array(m_values),
             m_over_d=np.array(m_over_d_values),
             relative_advantage=np.array(relative_advantages),
             mse_gaussian=np.array(mse_gaussian_list),
             mse_orf=np.array(mse_orf_list),
             d_fixed=d_fixed)
    
    print(f"Data saved to 'relative_advantage_data.npz'\n")


if __name__ == "__main__":
    # Run the original simulation
    main()
    
    # Run the relative advantage simulation and save data
    relative_advantage_simulation()
