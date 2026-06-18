from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "source" / "img" / "gaussian-process"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = ["#1677ff", "#ff6b35", "#00a878", "#7c3aed", "#e11d48"]


def configure_plot_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "axes.titlecolor": "#0f172a",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
            "grid.color": "#e2e8f0",
            "font.size": 11,
        }
    )


def gaussian_density(points, covariance):
    inverse = np.linalg.inv(covariance)
    exponent = np.einsum("...i,ij,...j->...", points, inverse, points)
    scale = 2 * np.pi * np.sqrt(np.linalg.det(covariance))
    return np.exp(-0.5 * exponent) / scale


def plot_multivariate_gaussian():
    rng = np.random.default_rng(17)
    covariances = [
        ("Independent", np.array([[1.0, 0.0], [0.0, 1.0]])),
        ("Positive correlation", np.array([[1.3, 0.95], [0.95, 1.0]])),
        ("Negative correlation", np.array([[1.3, -0.95], [-0.95, 1.0]])),
    ]
    axis = np.linspace(-3.5, 3.5, 240)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.stack([xx, yy], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for index, (title, covariance) in enumerate(covariances):
        density = gaussian_density(grid, covariance)
        samples = rng.multivariate_normal([0, 0], covariance, size=180)
        axes[index].contour(xx, yy, density, levels=7, colors=COLORS[index], linewidths=1.3)
        axes[index].scatter(
            samples[:, 0], samples[:, 1], s=13, color=COLORS[index], alpha=0.28, edgecolors="none"
        )
        axes[index].axhline(0, color="#cbd5e1", linewidth=0.8)
        axes[index].axvline(0, color="#cbd5e1", linewidth=0.8)
        axes[index].set(title=title, xlabel="$x_1$", ylabel="$x_2$", xlim=(-3.5, 3.5), ylim=(-3.5, 3.5))
        axes[index].set_aspect("equal")
        axes[index].grid(alpha=0.5)

    fig.suptitle("Geometry of multivariate Gaussian distributions", fontsize=16, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "multivariate-gaussian.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    squared_distance = (x1[:, None] - x2[None, :]) ** 2
    return variance * np.exp(-0.5 * squared_distance / length_scale**2)


def plot_gp_prior_samples():
    rng = np.random.default_rng(23)
    x = np.linspace(-3, 3, 260)
    settings = [(0.35, "Short length scale: $\\ell=0.35$"), (1.2, "Long length scale: $\\ell=1.20$")]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True, constrained_layout=True)
    for axis, (length_scale, title) in zip(axes, settings):
        covariance = rbf_kernel(x, x, length_scale)
        chol = np.linalg.cholesky(covariance + 1e-9 * np.eye(len(x)))
        samples = chol @ rng.standard_normal((len(x), 5))
        for sample_index in range(samples.shape[1]):
            axis.plot(x, samples[:, sample_index], color=COLORS[sample_index], linewidth=1.6, alpha=0.9)
        axis.axhline(0, color="#94a3b8", linewidth=1, linestyle="--")
        axis.set(title=title, xlabel="$x$", xlim=(-3, 3), ylim=(-3.4, 3.4))
        axis.grid(alpha=0.55)
    axes[0].set_ylabel("$f(x)$")
    fig.suptitle("Samples from Gaussian process priors", fontsize=16, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "gp-prior-samples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gp_posterior():
    rng = np.random.default_rng(31)
    x_train = np.array([-4.0, -2.8, -1.5, -0.2, 1.1, 2.3, 3.8])
    y_train = np.sin(x_train) + 0.18 * rng.standard_normal(len(x_train))
    x_test = np.linspace(-5.5, 5.5, 420)
    noise = 0.16
    length_scale = 1.15

    k_train = rbf_kernel(x_train, x_train, length_scale) + noise**2 * np.eye(len(x_train))
    k_cross = rbf_kernel(x_train, x_test, length_scale)
    k_test = rbf_kernel(x_test, x_test, length_scale)

    chol = np.linalg.cholesky(k_train + 1e-9 * np.eye(len(x_train)))
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
    posterior_mean = k_cross.T @ alpha
    v = np.linalg.solve(chol, k_cross)
    posterior_covariance = k_test - v.T @ v
    posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2
    posterior_std = np.sqrt(np.clip(np.diag(posterior_covariance), 0, None))

    posterior_chol = np.linalg.cholesky(posterior_covariance + 1e-8 * np.eye(len(x_test)))
    posterior_samples = posterior_mean[:, None] + posterior_chol @ rng.standard_normal((len(x_test), 3))

    fig, axis = plt.subplots(figsize=(12.5, 5.8), constrained_layout=True)
    axis.fill_between(
        x_test,
        posterior_mean - 1.96 * posterior_std,
        posterior_mean + 1.96 * posterior_std,
        color="#60a5fa",
        alpha=0.24,
        label="95% credible interval",
    )
    for sample_index in range(posterior_samples.shape[1]):
        label = "Posterior samples" if sample_index == 0 else None
        axis.plot(
            x_test,
            posterior_samples[:, sample_index],
            color=COLORS[sample_index + 1],
            linewidth=1.15,
            alpha=0.72,
            label=label,
        )
    axis.plot(x_test, posterior_mean, color="#075985", linewidth=2.6, label="Posterior mean")
    axis.errorbar(
        x_train,
        y_train,
        yerr=noise,
        fmt="o",
        color="#111827",
        ecolor="#64748b",
        capsize=3,
        markersize=5.5,
        label="Observations",
        zorder=10,
    )
    axis.set(
        title="Gaussian process regression posterior",
        xlabel="$x$",
        ylabel="$f(x)$",
        xlim=(-5.5, 5.5),
        ylim=(-2.7, 2.7),
    )
    axis.grid(alpha=0.55)
    axis.legend(ncol=2, frameon=False, loc="upper center")
    fig.savefig(OUTPUT_DIR / "gp-posterior.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    configure_plot_style()
    plot_multivariate_gaussian()
    plot_gp_prior_samples()
    plot_gp_posterior()
    print(f"Generated figures in {OUTPUT_DIR}")
