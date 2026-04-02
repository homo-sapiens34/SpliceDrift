"""
Core Bayesian Beta-Binomial regression model for age-related splicing
drift analysis.

Model
-----
For each (event, group) pair the observed inclusion counts are modelled as:

    k_i ~ BetaBinomial(N_i, mu_i * phi_i, (1 - mu_i) * phi_i)

    logit(mu_i)  = intercept_mu + alpha_prime * age_z_i
    log(phi_i)   = intercept_phi + beta_prime * age_z_i

where age_z is age standardised to zero mean and unit variance.

*  **alpha_prime (α')** — age effect on *mean* exon inclusion (PSI).
   α' > 0 means inclusion increases with age.
*  **beta_prime (β')** — age effect on splicing *variability* (precision).
   β' < 0 means precision drops, i.e. variability *increases* with age.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Silence chatty PyMC / PyTensor loggers by default
# ---------------------------------------------------------------------------
for _name in ("pytensor", "pymc", "pytensor.graph.rewriting"):
    logging.getLogger(_name).setLevel(logging.ERROR)


# ===================================================================
# fit_single — fit one (event, group) pair
# ===================================================================
def fit_single(
    df: pd.DataFrame,
    *,
    # Column names -------------------------------------------------------
    col_k: str = "k",
    col_N: str = "N",
    col_age: str = "age",
    col_event: str = "event_id",
    col_group: str = "group",
    # Data filtering -----------------------------------------------------
    min_total: int = 5,
    # MCMC parameters ----------------------------------------------------
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    target_accept: float = 0.9,
    random_seed: int = 42,
    # Convergence diagnostics --------------------------------------------
    rhat_max: float = 1.01,
    ess_min: int = 500,
    max_refits: int = 2,
    # Verbosity ----------------------------------------------------------
    quiet: bool = True,
) -> dict[str, Any]:
    """Fit a single (event, group) Beta-Binomial age-regression model.

    Parameters
    ----------
    df : DataFrame
        Input data.  Must contain at least the columns specified by
        *col_k*, *col_N*, and *col_age*.  Optionally *col_event* and
        *col_group* (used only for labelling output).  The DataFrame
        should contain data for **one** event and **one** group only.
    col_k : str
        Column with inclusion read counts (default ``"k"``).
    col_N : str
        Column with total informative read counts (default ``"N"``).
    col_age : str
        Column with the continuous covariate, typically age
        (default ``"age"``).
    col_event : str
        Column with the event identifier (default ``"event_id"``).
    col_group : str
        Column with the group/tissue identifier (default ``"group"``).
    min_total : int
        Minimum *N* for a sample to be included (default ``5``).
    draws : int
        Number of posterior draws per chain (default ``1000``).
    tune : int
        Number of tuning (warm-up) steps per chain (default ``1000``).
    chains : int
        Number of MCMC chains (default ``4``).
    cores : int
        Number of CPU cores for parallel chain sampling (default ``1``).
    target_accept : float
        Target acceptance probability for NUTS (default ``0.9``).
    random_seed : int
        Random seed for reproducibility (default ``42``).
    rhat_max : float
        Maximum acceptable R-hat for convergence (default ``1.01``).
    ess_min : int
        Minimum acceptable bulk ESS (default ``500``).
    max_refits : int
        Number of automatic refits if diagnostics fail (default ``2``).
    quiet : bool
        If *True*, suppress progress bars and print only warnings
        (default ``True``).

    Returns
    -------
    dict
        Keys:

        - ``"summary"`` — dict with α', β' posterior summaries
        - ``"trace"``   — ArviZ InferenceData
        - ``"model"``   — PyMC Model object
        - ``"df_used"`` — filtered DataFrame with ``age_z`` column added
        - ``"diagnostics"`` — convergence diagnostics dict
    """
    df = df.copy()

    # ---- Identify labels ------------------------------------------------
    event_id = (
        str(df[col_event].iloc[0])
        if col_event in df.columns and df[col_event].nunique() == 1
        else "NA"
    )
    group = (
        str(df[col_group].iloc[0])
        if col_group in df.columns and df[col_group].nunique() == 1
        else "NA"
    )

    if col_event in df.columns and df[col_event].nunique() > 1:
        raise ValueError("df must contain exactly one event_id.")
    if col_group in df.columns and df[col_group].nunique() > 1:
        raise ValueError(
            "df must contain exactly one group. Split by group first."
        )

    # ---- Coerce types ---------------------------------------------------
    df[col_k] = df[col_k].astype(int)
    df[col_N] = df[col_N].astype(int)
    df[col_age] = df[col_age].astype(float)

    # ---- Filter ---------------------------------------------------------
    df = df.dropna(subset=[col_k, col_N, col_age]).copy()
    df = df[
        (df[col_N] >= min_total)
        & (df[col_k] >= 0)
        & (df[col_k] <= df[col_N])
    ].copy()
    if df.empty:
        raise ValueError(
            "No rows left after filtering.  Check min_total and missing values."
        )

    # ---- Standardise age ------------------------------------------------
    age = df[col_age].to_numpy(dtype=float)
    age_mean = float(age.mean())
    age_std = float(age.std(ddof=0))
    if age_std == 0:
        raise ValueError(
            "Age has zero variance in this subset; cannot fit an age effect."
        )
    age_z = (age - age_mean) / age_std

    k_obs = df[col_k].to_numpy(dtype=int)
    N_obs = df[col_N].to_numpy(dtype=int)

    # ---- Internal helpers -----------------------------------------------
    def _fit_once(
        draws_: int, tune_: int, target_accept_: float
    ) -> tuple[pm.Model, az.InferenceData, dict]:
        with pm.Model() as model:
            # Mean PSI sub-model: logit(μ) = intercept_mu + α' · age_z
            intercept_mu = pm.Normal("intercept_mu", mu=0.0, sigma=1.5)
            alpha_prime = pm.Normal("alpha_prime", mu=0.0, sigma=1.0)

            # Precision sub-model: log(φ) = intercept_phi + β' · age_z
            intercept_phi = pm.Normal("intercept_phi", mu=0.0, sigma=1.0)
            beta_prime = pm.Normal("beta_prime", mu=0.0, sigma=1.0)

            eta = intercept_mu + alpha_prime * age_z
            mu = pm.Deterministic("mu", pm.math.sigmoid(eta))

            log_phi = intercept_phi + beta_prime * age_z
            phi = pm.Deterministic("phi", pm.math.exp(log_phi))

            pm.BetaBinomial(
                "k_obs",
                n=N_obs,
                alpha=mu * phi,
                beta=(1 - mu) * phi,
                observed=k_obs,
            )

            trace = pm.sample(
                draws=draws_,
                tune=tune_,
                chains=chains,
                cores=cores,
                target_accept=target_accept_,
                random_seed=random_seed,
                progressbar=not quiet,
                compute_convergence_checks=False,
            )

        diag = _diagnostics(trace)
        return model, trace, diag

    def _diagnostics(trace: az.InferenceData) -> dict:
        summ = az.summary(
            trace,
            var_names=["intercept_mu", "alpha_prime", "intercept_phi", "beta_prime"],
        )
        div = trace.sample_stats.get("diverging", None)
        n_div = int(div.values.sum()) if div is not None else 0

        rhat = summ["r_hat"].to_dict()
        ess_bulk = summ["ess_bulk"].to_dict()

        rhat_ok = all(
            (np.isnan(v) or v <= rhat_max) for v in rhat.values()
        )
        ess_ok = all(
            (np.isnan(v) or v >= ess_min) for v in ess_bulk.values()
        )
        div_ok = n_div == 0

        return {
            "summary_table": summ,
            "n_divergences": n_div,
            "rhat": rhat,
            "ess_bulk": ess_bulk,
            "rhat_ok": rhat_ok,
            "ess_ok": ess_ok,
            "div_ok": div_ok,
            "ok": rhat_ok and ess_ok and div_ok,
        }

    def _make_result(
        model: pm.Model, trace: az.InferenceData, diag: dict
    ) -> dict:
        post = trace.posterior
        ap = post["alpha_prime"].values.reshape(-1)
        bp = post["beta_prime"].values.reshape(-1)

        ap_hdi = az.hdi(ap, hdi_prob=0.95)
        bp_hdi = az.hdi(bp, hdi_prob=0.95)

        summary = {
            "event_id": event_id,
            "group": group,
            "n_samples": int(df.shape[0]),
            "age_mean": age_mean,
            "age_std": age_std,
            "alpha_prime_mean": float(ap.mean()),
            "alpha_prime_hdi_lo": float(ap_hdi[0]),
            "alpha_prime_hdi_hi": float(ap_hdi[1]),
            "P_alpha_prime_gt_0": float((ap > 0).mean()),
            "beta_prime_mean": float(bp.mean()),
            "beta_prime_hdi_lo": float(bp_hdi[0]),
            "beta_prime_hdi_hi": float(bp_hdi[1]),
            "P_beta_prime_lt_0": float((bp < 0).mean()),
            "diagnostics_ok": diag["ok"],
        }

        return {
            "model": model,
            "trace": trace,
            "df_used": df.assign(age_z=age_z),
            "summary": summary,
            "diagnostics": {
                "n_divergences": diag["n_divergences"],
                "rhat": diag["rhat"],
                "ess_bulk": diag["ess_bulk"],
                "ok": diag["ok"],
                "table": diag["summary_table"],
            },
        }

    # ---- Fit with automatic refits -------------------------------------
    current_draws = int(draws)
    current_tune = int(tune)
    current_ta = float(target_accept)

    last: tuple | None = None
    for attempt in range(max_refits + 1):
        model, trace, diag = _fit_once(
            current_draws, current_tune, current_ta
        )
        last = (model, trace, diag)

        if diag["ok"]:
            return _make_result(model, trace, diag)

        # Print only on failure
        logger.warning(
            "Diagnostics not OK for group=%s, event=%s "
            "(attempt %d/%d). divergences=%d, max r_hat=%.3f, "
            "min ess_bulk=%.0f — refitting…",
            group,
            event_id,
            attempt + 1,
            max_refits + 1,
            diag["n_divergences"],
            max(
                (v for v in diag["rhat"].values() if not np.isnan(v)),
                default=float("nan"),
            ),
            min(
                (v for v in diag["ess_bulk"].values() if not np.isnan(v)),
                default=float("nan"),
            ),
        )

        if diag["n_divergences"] > 0:
            current_ta = min(0.99, current_ta + 0.05)
        current_draws = int(np.ceil(current_draws * 1.5))
        current_tune = int(np.ceil(current_tune * 1.25))

    # Exhausted refits — return last attempt
    assert last is not None
    model, trace, diag = last
    logger.warning(
        "Returning fit despite failing diagnostics for "
        "group=%s, event=%s after %d attempts.",
        group,
        event_id,
        max_refits + 1,
    )
    return _make_result(model, trace, diag)


# ===================================================================
# fit — batch-fit all (event, group) pairs
# ===================================================================
def fit(
    df: pd.DataFrame,
    *,
    col_k: str = "k",
    col_N: str = "N",
    col_age: str = "age",
    col_event: str = "event_id",
    col_group: str = "group",
    min_total: int = 5,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    target_accept: float = 0.9,
    random_seed: int = 42,
    rhat_max: float = 1.01,
    ess_min: int = 500,
    max_refits: int = 2,
    quiet: bool = True,
) -> pd.DataFrame:
    """Fit the Beta-Binomial age model to every (event, group) pair.

    Parameters
    ----------
    df : DataFrame
        Tabular data with at least ``event_id``, ``group`` (optional),
        ``k``, ``N``, and ``age`` columns.  See :func:`fit_single` for
        the full list of tuneable parameters (all are forwarded).

    Returns
    -------
    DataFrame
        One row per (event, group) pair with columns:

        - ``event_id``, ``group``, ``n_samples``
        - ``alpha_prime_mean``, ``alpha_prime_hdi_lo``,
          ``alpha_prime_hdi_hi``, ``P_alpha_prime_gt_0``
        - ``beta_prime_mean``, ``beta_prime_hdi_lo``,
          ``beta_prime_hdi_hi``, ``P_beta_prime_lt_0``
        - ``diagnostics_ok``
    """
    if col_group not in df.columns:
        df = df.copy()
        df[col_group] = "all"

    groups = list(df.groupby([col_event, col_group]))
    rows: list[dict] = []

    common_kw = dict(
        col_k=col_k,
        col_N=col_N,
        col_age=col_age,
        col_event=col_event,
        col_group=col_group,
        min_total=min_total,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        target_accept=target_accept,
        random_seed=random_seed,
        rhat_max=rhat_max,
        ess_min=ess_min,
        max_refits=max_refits,
        quiet=quiet,
    )

    for (eid, grp), sub in tqdm(groups, desc="Fitting events", disable=quiet):
        try:
            result = fit_single(sub, **common_kw)
            rows.append(result["summary"])
        except Exception as exc:
            logger.warning("Skipping event=%s group=%s: %s", eid, grp, exc)
            continue

    return pd.DataFrame(rows)


# ===================================================================
# posterior_mu_curve — posterior mean PSI curve for plotting
# ===================================================================
def posterior_mu_curve(
    fit_result: dict,
    ages: np.ndarray,
    *,
    n_draws: int = 2000,
) -> dict[str, np.ndarray]:
    """Compute posterior mean-PSI curve over an age grid.

    Parameters
    ----------
    fit_result : dict
        Output of :func:`fit_single`.
    ages : array-like
        Age values at which to evaluate the curve.
    n_draws : int
        Maximum number of posterior draws to use (default ``2000``).

    Returns
    -------
    dict
        ``mu_mean``, ``mu_lo`` (5th percentile), ``mu_hi``
        (95th percentile) arrays, each of length ``len(ages)``.
    """
    trace = fit_result["trace"]
    df_used = fit_result["df_used"]

    age_mean = float(df_used["age"].mean())
    age_std = float(df_used["age"].std(ddof=0))

    ages = np.asarray(ages, dtype=float)
    age_z = (ages - age_mean) / age_std

    alpha = trace.posterior["intercept_mu"].values.reshape(-1)
    ap = trace.posterior["alpha_prime"].values.reshape(-1)

    if alpha.shape[0] > n_draws:
        rng = np.random.default_rng(0)
        idx = rng.choice(alpha.shape[0], size=n_draws, replace=False)
        alpha = alpha[idx]
        ap = ap[idx]

    eta = alpha[:, None] + ap[:, None] * age_z[None, :]
    mu = 1.0 / (1.0 + np.exp(-eta))

    return {
        "ages": ages,
        "mu_mean": mu.mean(axis=0),
        "mu_lo": np.quantile(mu, 0.05, axis=0),
        "mu_hi": np.quantile(mu, 0.95, axis=0),
    }
