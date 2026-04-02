"""
Command-line interface for SpliceDrift.

Usage
-----
::

    splicedrift fit          -i INPUT -o OUTPUT [options]
    splicedrift plot-scatter -i SUMMARY -o OUTPUT [options]
    splicedrift plot-event   -i INPUT -e EVENT [-g GROUP] -o OUTPUT [options]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_mcmc_args(parser: argparse.ArgumentParser) -> None:
    """Add shared MCMC tuneable arguments to a sub-parser."""
    g = parser.add_argument_group("MCMC parameters")
    g.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="Posterior draws per chain (default: 1000).",
    )
    g.add_argument(
        "--tune",
        type=int,
        default=1000,
        help="Tuning (warm-up) steps per chain (default: 1000).",
    )
    g.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4).",
    )
    g.add_argument(
        "--cores",
        type=int,
        default=1,
        help="CPU cores for parallel chain sampling (default: 1).",
    )
    g.add_argument(
        "--target-accept",
        type=float,
        default=0.9,
        help="NUTS target acceptance probability (default: 0.9).",
    )
    g.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    g.add_argument(
        "--min-total",
        type=int,
        default=5,
        help="Minimum total reads N per sample (default: 5).",
    )
    g.add_argument(
        "--rhat-max",
        type=float,
        default=1.01,
        help="Maximum acceptable R-hat (default: 1.01).",
    )
    g.add_argument(
        "--ess-min",
        type=int,
        default=500,
        help="Minimum acceptable bulk ESS (default: 500).",
    )
    g.add_argument(
        "--max-refits",
        type=int,
        default=2,
        help="Automatic refits on convergence failure (default: 2).",
    )


def _add_column_args(parser: argparse.ArgumentParser) -> None:
    """Add column-name override arguments."""
    g = parser.add_argument_group("Column names")
    g.add_argument("--col-k", default="k", help="Inclusion-count column (default: k).")
    g.add_argument(
        "--col-N", default="N", help="Total-count column (default: N)."
    )
    g.add_argument(
        "--col-age", default="age", help="Age / covariate column (default: age)."
    )
    g.add_argument(
        "--col-event",
        default="event_id",
        help="Event-ID column (default: event_id).",
    )
    g.add_argument(
        "--col-group",
        default="group",
        help="Group / tissue column (default: group).",
    )


# -------------------------------------------------------------------
# fit sub-command
# -------------------------------------------------------------------
def cmd_fit(args: argparse.Namespace) -> None:
    from splicedrift.io import load_data
    from splicedrift.model import fit

    print(f"[SpliceDrift] Loading data from {args.input} …")
    data = load_data(
        args.input,
        col_k=args.col_k,
        col_N=args.col_N,
        col_age=args.col_age,
        col_event=args.col_event,
        col_group=args.col_group,
    )

    n_events = data["event_id"].nunique()
    n_groups = data["group"].nunique()
    print(
        f"[SpliceDrift] Loaded {len(data)} rows — "
        f"{n_events} event(s) × {n_groups} group(s)."
    )

    results = fit(
        data,
        col_k="k",
        col_N="N",
        col_age="age",
        col_event="event_id",
        col_group="group",
        min_total=args.min_total,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
        rhat_max=args.rhat_max,
        ess_min=args.ess_min,
        max_refits=args.max_refits,
        quiet=not args.verbose,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, sep="\t", index=False)
    print(f"[SpliceDrift] Results written to {out} ({len(results)} rows).")


# -------------------------------------------------------------------
# plot-scatter sub-command
# -------------------------------------------------------------------
def cmd_plot_scatter(args: argparse.Namespace) -> None:
    import pandas as pd
    from splicedrift.plotting import plot_scatter

    print(f"[SpliceDrift] Loading summary from {args.input} …")
    summary = pd.read_csv(args.input, sep="\t")

    groups = args.groups.split(",") if args.groups else None

    fig = plot_scatter(
        summary,
        groups=groups,
        fdr_threshold=args.fdr,
        ncols=args.ncols,
        save=args.output,
        dpi=args.dpi,
    )
    print(f"[SpliceDrift] Scatter plot saved to {args.output}.")


# -------------------------------------------------------------------
# plot-event sub-command
# -------------------------------------------------------------------
def cmd_plot_event(args: argparse.Namespace) -> None:
    from splicedrift.io import load_data
    from splicedrift.plotting import plot_event

    print(f"[SpliceDrift] Loading data from {args.input} …")
    data = load_data(
        args.input,
        col_k=args.col_k,
        col_N=args.col_N,
        col_age=args.col_age,
        col_event=args.col_event,
        col_group=args.col_group,
    )

    fig = plot_event(
        data,
        event_id=args.event,
        group=args.group,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        random_seed=args.random_seed,
        quiet=not args.verbose,
        save=args.output,
        dpi=args.dpi,
    )
    print(f"[SpliceDrift] Event plot saved to {args.output}.")


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="splicedrift",
        description=(
            "SpliceDrift — Bayesian Beta-Binomial regression for "
            "age-related splicing drift analysis."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- fit ------------------------------------------------------------
    p_fit = sub.add_parser(
        "fit",
        help="Fit the Beta-Binomial model to all (event, group) pairs.",
    )
    p_fit.add_argument(
        "-i", "--input", required=True, help="Input TSV file."
    )
    p_fit.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output summary TSV file.",
    )
    p_fit.add_argument(
        "-v", "--verbose", action="store_true", help="Show progress bars."
    )
    _add_mcmc_args(p_fit)
    _add_column_args(p_fit)
    p_fit.set_defaults(func=cmd_fit)

    # ---- plot-scatter ---------------------------------------------------
    p_scat = sub.add_parser(
        "plot-scatter",
        help="Plot α' vs β' scatter panels from a summary file.",
    )
    p_scat.add_argument(
        "-i",
        "--input",
        required=True,
        help="Summary TSV (output of 'splicedrift fit').",
    )
    p_scat.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output image file (e.g. scatter.png).",
    )
    p_scat.add_argument(
        "--groups",
        default=None,
        help="Comma-separated list of groups to plot (default: all).",
    )
    p_scat.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        help="FDR-adjusted significance threshold (default: 0.05).",
    )
    p_scat.add_argument(
        "--ncols", type=int, default=4, help="Columns in subplot grid (default: 4)."
    )
    p_scat.add_argument(
        "--dpi", type=int, default=300, help="Image resolution (default: 300)."
    )
    p_scat.set_defaults(func=cmd_plot_scatter)

    # ---- plot-event -----------------------------------------------------
    p_ev = sub.add_parser(
        "plot-event",
        help="Plot PSI vs age for a single event with posterior μ band.",
    )
    p_ev.add_argument(
        "-i", "--input", required=True, help="Input TSV file (raw data)."
    )
    p_ev.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output image file (e.g. event.png).",
    )
    p_ev.add_argument(
        "-e", "--event", required=True, help="Event ID to plot."
    )
    p_ev.add_argument(
        "-g",
        "--group",
        default=None,
        help="Group / tissue to plot (required if data has multiple groups).",
    )
    p_ev.add_argument(
        "-v", "--verbose", action="store_true", help="Show progress bars."
    )
    p_ev.add_argument(
        "--dpi", type=int, default=300, help="Image resolution (default: 300)."
    )
    _add_mcmc_args(p_ev)
    _add_column_args(p_ev)
    p_ev.set_defaults(func=cmd_plot_event)

    # ---- Parse & dispatch -----------------------------------------------
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
