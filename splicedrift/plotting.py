"""
Plotting functions for SpliceDrift.

Two main plot types:

1. **Scatter** — α' vs β' for all events within each group, coloured by
   significance (FDR-corrected).
2. **Event** — PSI vs age for a single event/group, with posterior mean
   μ(age) curve and 90 % credible band.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from splicedrift.model import fit_single, posterior_mu_curve


# ===================================================================
# plot_scatter — α' vs β' scatter panels (one per group)
# ===================================================================
def plot_scatter(
    summary: pd.DataFrame,
    *,
    groups: Sequence[str] | None = None,
    fdr_threshold: float = 0.05,
    ncols: int = 4,
    panel_size: tuple[float, float] = (4.0, 4.0),
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot α' vs β' scatter for every group (tissue).

    Each point is one splicing event.  Significance is determined by
    converting posterior probabilities to two-sided p-values and applying
    Benjamini–Hochberg FDR correction across all (event, group) pairs.

    Parameters
    ----------
    summary : DataFrame
        Output of :func:`splicedrift.fit`.
    groups : list of str, optional
        Subset of groups to plot.  ``None`` → all groups.
    fdr_threshold : float
        FDR-adjusted significance threshold (default ``0.05``).
    ncols : int
        Number of subplot columns (default ``4``).
    panel_size : tuple
        (width, height) per panel in inches (default ``(4, 4)``).
    save : str, optional
        If given, save figure to this path.
    dpi : int
        Resolution for saved figure (default ``300``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = summary.copy()

    # ---- FDR correction -------------------------------------------------
    # Two-sided "Bayesian p-value" from posterior probability
    alpha_pvals = 2 * np.minimum(
        df["P_alpha_prime_gt_0"], 1 - df["P_alpha_prime_gt_0"]
    )
    beta_pvals = 2 * np.minimum(
        df["P_beta_prime_lt_0"], 1 - df["P_beta_prime_lt_0"]
    )

    _, alpha_padj, _, _ = multipletests(alpha_pvals, method="fdr_bh")
    _, beta_padj, _, _ = multipletests(beta_pvals, method="fdr_bh")

    df["alpha_prime_padj"] = alpha_padj
    df["beta_prime_padj"] = beta_padj

    df["alpha_sig_pos"] = (
        (df["P_alpha_prime_gt_0"] > 0.5) & (df["alpha_prime_padj"] < fdr_threshold)
    )
    df["alpha_sig_neg"] = (
        (df["P_alpha_prime_gt_0"] <= 0.5) & (df["alpha_prime_padj"] < fdr_threshold)
    )
    df["beta_sig"] = df["beta_prime_padj"] < fdr_threshold

    # ---- Select groups --------------------------------------------------
    if groups is None:
        groups = sorted(df["group"].unique())
    n_groups = len(groups)
    if n_groups == 0:
        raise ValueError("No groups to plot.")

    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, grp in enumerate(groups):
        ax = axes_flat[idx]
        gd = df[df["group"] == grp]

        n_non_sig = len(gd[~gd["alpha_sig_pos"] & ~gd["alpha_sig_neg"]])
        n_pos = len(gd[gd["alpha_sig_pos"]])
        n_neg = len(gd[gd["alpha_sig_neg"]])
        n_beta = len(gd[gd["beta_sig"]])

        # Non-significant (gray)
        ns = gd[~gd["alpha_sig_pos"] & ~gd["alpha_sig_neg"]]
        ax.scatter(
            ns["alpha_prime_mean"],
            ns["beta_prime_mean"],
            alpha=0.3,
            s=30,
            color="gray",
            label=f"Not significant (n={n_non_sig})",
        )

        # α' > 0 significant (red)
        pos = gd[gd["alpha_sig_pos"]]
        ax.scatter(
            pos["alpha_prime_mean"],
            pos["beta_prime_mean"],
            alpha=0.7,
            s=30,
            color="red",
            edgecolors="darkred",
            linewidth=1.5,
            label=f"α' > 0 (sig, n={n_pos})",
        )

        # α' < 0 significant (blue)
        neg = gd[gd["alpha_sig_neg"]]
        ax.scatter(
            neg["alpha_prime_mean"],
            neg["beta_prime_mean"],
            alpha=0.7,
            s=30,
            color="blue",
            edgecolors="darkblue",
            linewidth=1.5,
            label=f"α' < 0 (sig, n={n_neg})",
        )

        # β' significant (orange ×)
        bs = gd[gd["beta_sig"]]
        ax.scatter(
            bs["alpha_prime_mean"],
            bs["beta_prime_mean"],
            alpha=0.8,
            s=30,
            marker="x",
            color="orange",
            linewidth=2.5,
            label=f"β' sig (n={n_beta})",
        )

        ax.axhline(0, color="gray", lw=1.8, ls="--", alpha=0.7)
        ax.axvline(0, color="gray", lw=1.8, ls="--", alpha=0.7)
        ax.set_xlabel("α' (effect on PSI mean)", fontsize=11)
        ax.set_ylabel("β' (effect on PSI variability)", fontsize=11)

        title = grp.split(" (")[0] if "(" in grp else grp
        ax.set_title(f"{title} (n={len(gd)})", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        legend = ax.legend(fontsize=8, loc="best", framealpha=0.8)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.8)

    # Hide unused panels
    for idx in range(n_groups, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout(pad=2)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    return fig


# ===================================================================
# plot_event — PSI vs age with posterior μ band
# ===================================================================
def plot_event(
    df: pd.DataFrame,
    *,
    event_id: str,
    group: str | None = None,
    fit_result: dict | None = None,
    col_k: str = "k",
    col_N: str = "N",
    col_age: str = "age",
    col_event: str = "event_id",
    col_group: str = "group",
    # passed to fit_single if fit_result is None
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    target_accept: float = 0.9,
    random_seed: int = 42,
    quiet: bool = True,
    save: str | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot PSI vs age for one event, with posterior mean μ curve.

    If *fit_result* is not provided the model is fitted on-the-fly
    (takes ~30 s for a typical dataset).

    Parameters
    ----------
    df : DataFrame
        Full input data (all events/groups).
    event_id : str
        Which event to plot.
    group : str, optional
        Which group to plot.  If ``None`` and data has only one group,
        that group is used.
    fit_result : dict, optional
        Pre-computed output of :func:`splicedrift.fit_single`.  Avoids
        re-fitting if you already have it.
    save : str, optional
        Save figure to this path.
    dpi : int
        Resolution for saved figure (default ``300``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    sub = df[df[col_event] == event_id].copy()
    if sub.empty:
        raise ValueError(f"No data found for event_id={event_id!r}.")

    if col_group not in sub.columns:
        sub[col_group] = "all"

    if group is not None:
        sub = sub[sub[col_group] == group]
    elif sub[col_group].nunique() > 1:
        raise ValueError(
            f"Multiple groups found for event {event_id}: "
            f"{sorted(sub[col_group].unique())}.  "
            "Please specify the ``group`` parameter."
        )

    if sub.empty:
        raise ValueError(
            f"No data for event_id={event_id!r}, group={group!r}."
        )

    grp_label = str(sub[col_group].iloc[0])

    # ---- Fit if needed --------------------------------------------------
    if fit_result is None:
        fit_result = fit_single(
            sub,
            col_k=col_k,
            col_N=col_N,
            col_age=col_age,
            col_event=col_event,
            col_group=col_group,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            quiet=quiet,
        )

    # ---- Compute PSI + μ band -------------------------------------------
    sub["PSI"] = sub[col_k] / sub[col_N]
    ages_grid = np.linspace(sub[col_age].min(), sub[col_age].max(), 80)
    band = posterior_mu_curve(fit_result, ages_grid)

    smry = fit_result["summary"]

    # ---- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(sub[col_age], sub["PSI"], alpha=0.5, s=25, zorder=3)
    ax.plot(band["ages"], band["mu_mean"], color="C1", lw=2, zorder=4)
    ax.fill_between(
        band["ages"],
        band["mu_lo"],
        band["mu_hi"],
        alpha=0.2,
        color="C1",
        zorder=2,
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("PSI (k / N)", fontsize=12)

    title = (
        f"{event_id} — {grp_label}\n"
        f"α'={smry['alpha_prime_mean']:.3f}  "
        f"(P>0={smry['P_alpha_prime_gt_0']:.2f})   "
        f"β'={smry['beta_prime_mean']:.3f}  "
        f"(P<0={smry['P_beta_prime_lt_0']:.2f})"
    )
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")

    return fig
