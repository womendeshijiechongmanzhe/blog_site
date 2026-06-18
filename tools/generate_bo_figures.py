from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "source" / "img" / "bayesian-optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#1677ff"
NAVY = "#075985"
ORANGE = "#ff6b35"
GREEN = "#00a878"
PURPLE = "#7c3aed"
RED = "#e11d48"
SLATE = "#64748b"


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


def objective(x):
    return (
        0.55 * np.sin(1.5 * x - 0.4)
        + 0.28 * np.cos(3.2 * x)
        + 1.05 * np.exp(-0.5 * ((x - 4.7) / 0.55) ** 2)
        + 0.12 * x
    )


def rbf_kernel(x1, x2, length_scale=0.9, variance=1.0):
    distance2 = (x1[:, None] - x2[None, :]) ** 2
    return variance * np.exp(-0.5 * distance2 / length_scale**2)


def gp_posterior(x_train, y_train, x_test, noise=0.08):
    k_train = rbf_kernel(x_train, x_train) + noise**2 * np.eye(len(x_train))
    k_cross = rbf_kernel(x_train, x_test)
    k_test = rbf_kernel(x_test, x_test)

    chol = np.linalg.cholesky(k_train + 1e-9 * np.eye(len(x_train)))
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
    mean = k_cross.T @ alpha
    v = np.linalg.solve(chol, k_cross)
    covariance = k_test - v.T @ v
    std = np.sqrt(np.clip(np.diag(covariance), 0, None))
    return mean, std


def acquisition_values(mean, std, best, xi=0.04, kappa=1.8):
    safe_std = np.maximum(std, 1e-12)
    improvement = mean - best - xi
    z = improvement / safe_std
    pi = norm.cdf(z)
    ei = improvement * norm.cdf(z) + safe_std * norm.pdf(z)
    pi[std < 1e-10] = 0
    ei[std < 1e-10] = 0
    ucb = mean + kappa * std
    return pi, ei, ucb


def get_demo_state():
    x_grid = np.linspace(0, 6, 500)
    x_train = np.array([0.35, 1.55, 2.7, 4.05, 5.65])
    y_train = objective(x_train)
    mean, std = gp_posterior(x_train, y_train, x_grid)
    pi, ei, ucb = acquisition_values(mean, std, np.max(y_train))
    return x_grid, x_train, y_train, mean, std, pi, ei, ucb


def plot_bo_next_point():
    x_grid, x_train, y_train, mean, std, _, ei, _ = get_demo_state()
    next_index = int(np.argmax(ei))
    x_next = x_grid[next_index]

    fig, (posterior_axis, acquisition_axis) = plt.subplots(
        2,
        1,
        figsize=(12.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1]},
        constrained_layout=True,
    )
    posterior_axis.fill_between(
        x_grid,
        mean - 1.96 * std,
        mean + 1.96 * std,
        color="#60a5fa",
        alpha=0.24,
        label="95% credible interval",
    )
    posterior_axis.plot(x_grid, objective(x_grid), color="#111827", linewidth=1.8, linestyle="--", label="Black-box objective")
    posterior_axis.plot(x_grid, mean, color=NAVY, linewidth=2.5, label="GP posterior mean")
    posterior_axis.scatter(x_train, y_train, color=ORANGE, edgecolor="#ffffff", linewidth=1.2, s=58, zorder=5, label="Observations")
    posterior_axis.axvline(x_next, color=RED, linestyle=":", linewidth=2)
    posterior_axis.set(title="Surrogate posterior after five expensive evaluations", ylabel="$f(x)$")
    posterior_axis.grid(alpha=0.55)
    posterior_axis.legend(frameon=False, ncol=2, loc="upper left")

    acquisition_axis.fill_between(x_grid, 0, ei, color=PURPLE, alpha=0.22)
    acquisition_axis.plot(x_grid, ei, color=PURPLE, linewidth=2.2, label="Expected Improvement")
    acquisition_axis.scatter([x_next], [ei[next_index]], color=RED, s=48, zorder=5, label="Next evaluation")
    acquisition_axis.axvline(x_next, color=RED, linestyle=":", linewidth=2)
    acquisition_axis.set(xlabel="$x$", ylabel="EI", xlim=(0, 6))
    acquisition_axis.set_ylim(bottom=0)
    acquisition_axis.grid(alpha=0.55)
    acquisition_axis.legend(frameon=False, loc="upper left")

    fig.suptitle("One step of Bayesian optimization", fontsize=17, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "bo-next-point.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_acquisition_functions():
    x_grid, x_train, y_train, mean, std, pi, ei, ucb = get_demo_state()
    acquisitions = [
        ("Probability of Improvement", pi, BLUE, "PI"),
        ("Expected Improvement", ei, PURPLE, "EI"),
        ("Upper Confidence Bound", ucb, GREEN, "UCB"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.6), sharex=True, constrained_layout=True)
    for axis, (title, values, color, label) in zip(axes, acquisitions):
        next_index = int(np.argmax(values))
        axis.plot(x_grid, values, color=color, linewidth=2.2)
        baseline = min(0, float(np.min(values)))
        axis.fill_between(x_grid, baseline, values, color=color, alpha=0.16)
        axis.axvline(x_grid[next_index], color=RED, linestyle=":", linewidth=1.8)
        axis.scatter([x_grid[next_index]], [values[next_index]], color=RED, s=38, zorder=5)
        axis.set(title=title, ylabel=label)
        axis.grid(alpha=0.55)
    axes[-1].set(xlabel="$x$", xlim=(0, 6))

    for x_value in x_train:
        axes[-1].axvline(x_value, color=SLATE, linewidth=0.6, alpha=0.25)

    fig.suptitle("Acquisition functions choose different next evaluations", fontsize=17, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "acquisition-functions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def gaussian_kde(grid, samples, bandwidth):
    scaled = (grid[:, None] - samples[None, :]) / bandwidth
    return np.mean(norm.pdf(scaled) / bandwidth, axis=1)


def plot_tpe_density():
    rng = np.random.default_rng(2026)
    x_observed = np.sort(rng.uniform(0.1, 5.9, size=26))
    loss_observed = 1.65 - objective(x_observed) + rng.normal(0, 0.07, size=len(x_observed))
    gamma = 0.25
    threshold = float(np.quantile(loss_observed, gamma))
    good = loss_observed < threshold
    grid = np.linspace(0, 6, 500)
    l_density = gaussian_kde(grid, x_observed[good], bandwidth=0.33)
    g_density = gaussian_kde(grid, x_observed[~good], bandwidth=0.42)
    relative_good = l_density / np.maximum(l_density + g_density, 1e-12)
    next_index = int(np.argmax(relative_good))

    fig, (observation_axis, density_axis) = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    observation_axis.scatter(x_observed[~good], loss_observed[~good], color=SLATE, alpha=0.72, s=42, label="Bad group")
    observation_axis.scatter(x_observed[good], loss_observed[good], color=ORANGE, edgecolor="#ffffff", linewidth=1, s=56, label="Good group")
    observation_axis.axhline(threshold, color=RED, linestyle="--", linewidth=1.8, label=f"Quantile threshold ($\\gamma={gamma:.2f}$)")
    observation_axis.set(title="Split observations by a loss quantile", xlabel="$x$", ylabel="Observed loss", xlim=(0, 6))
    observation_axis.grid(alpha=0.55)
    observation_axis.legend(frameon=False)

    density_axis.fill_between(grid, 0, l_density, color=ORANGE, alpha=0.2)
    density_axis.fill_between(grid, 0, g_density, color=BLUE, alpha=0.14)
    density_axis.plot(grid, l_density, color=ORANGE, linewidth=2.3, label="Good density $l(x)$")
    density_axis.plot(grid, g_density, color=BLUE, linewidth=2.3, label="Bad density $g(x)$")
    score_axis = density_axis.twinx()
    score_axis.plot(grid, relative_good, color=GREEN, linewidth=1.8, linestyle="--", label="$l(x)/(l(x)+g(x))$")
    score_axis.scatter([grid[next_index]], [relative_good[next_index]], color=RED, s=46, zorder=5)
    score_axis.axvline(grid[next_index], color=RED, linestyle=":", linewidth=1.8)
    score_axis.set_ylabel("Relative good density", color=GREEN)
    score_axis.tick_params(axis="y", colors=GREEN)
    density_axis.set(title="Model good and bad parameter densities", xlabel="$x$", ylabel="Density", xlim=(0, 6))
    density_axis.set_ylim(bottom=0)
    density_axis.grid(alpha=0.55)

    handles_left, labels_left = density_axis.get_legend_handles_labels()
    handles_right, labels_right = score_axis.get_legend_handles_labels()
    density_axis.legend(handles_left + handles_right, labels_left + labels_right, frameon=False, loc="upper left")

    fig.suptitle("Tree-structured Parzen Estimator", fontsize=17, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "tpe-density.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    configure_plot_style()
    plot_bo_next_point()
    plot_acquisition_functions()
    plot_tpe_density()
    print(f"Generated figures in {OUTPUT_DIR}")
