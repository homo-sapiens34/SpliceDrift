"""
Data loading and validation utilities for SpliceDrift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

# Columns that *must* be present in the standardised input
REQUIRED_COLUMNS = {"event_id", "sample_id", "k", "N", "age"}
# The ``group`` column is optional — if absent every row is assigned to
# the group ``"all"``.
OPTIONAL_COLUMNS = {"group"}


def load_data(
    path: str | Path,
    *,
    sep: str = "\t",
    col_event: str = "event_id",
    col_sample: str = "sample_id",
    col_k: str = "k",
    col_N: str = "N",
    col_age: str = "age",
    col_group: str = "group",
) -> pd.DataFrame:
    """Load and validate a SpliceDrift input file.

    The expected format is a **tab-separated** table with columns:

    ========== ============================================================
    Column     Description
    ========== ============================================================
    event_id   Splicing-event identifier (e.g. ``HsaEX0000244``).
    sample_id  Sample identifier.
    k          Inclusion junction read count.
    N          Total informative read count (``k + 2 * exclusion``).
    age        Continuous covariate (typically donor age).
    group      *(optional)* Tissue or condition label.  If absent, all
               rows are assigned to the group ``"all"``.
    ========== ============================================================

    Parameters
    ----------
    path : str or Path
        Path to a TSV (or CSV) file.
    sep : str
        Column separator (default ``"\\t"``).
    col_event, col_sample, col_k, col_N, col_age, col_group : str
        Override default column names if your file uses different headers.

    Returns
    -------
    DataFrame
        Validated data ready for :func:`splicedrift.fit`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing or data types are wrong.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path, sep=sep)

    # ----- Map user column names to canonical names ----------------------
    rename_map = {
        col_event: "event_id",
        col_sample: "sample_id",
        col_k: "k",
        col_N: "N",
        col_age: "age",
    }
    if col_group in df.columns:
        rename_map[col_group] = "group"

    df = df.rename(columns=rename_map)

    # ----- Check required columns ----------------------------------------
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}.  "
            f"Expected columns: {sorted(REQUIRED_COLUMNS)}."
        )

    # ----- Add default group if absent -----------------------------------
    if "group" not in df.columns:
        df["group"] = "all"

    # ----- Coerce types --------------------------------------------------
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["k", "N", "age"])
    n_after = len(df)
    if n_after < n_before:
        print(
            f"[SpliceDrift] Dropped {n_before - n_after} rows with "
            f"non-numeric k/N/age values."
        )

    df["k"] = df["k"].astype(int)
    df["N"] = df["N"].astype(int)

    return df


def load_vastdb(
    junction_file: str | Path,
    age_file: str | Path,
    *,
    sep: str = "\t",
    sample_id_parser: Callable[[str], str] | None = None,
    age_col: str = "AGE",
    subject_col: str = "SUBJID",
) -> pd.DataFrame:
    """Load data from VastDB junction-count format + a separate age file.

    This is a convenience wrapper for datasets in the format produced by
    ``vast-tools`` / GTEX pipelines, where each file has columns::

        event_id  tissue  sample_id  <junc1>  <junc2>  <exclusion>  PSI

    and age information lives in a separate metadata file.

    Parameters
    ----------
    junction_file : str or Path
        Path to the junction-count TSV.  The last four columns are
        assumed to be ``junc1, junc2, exclusion, PSI`` (positional).
    age_file : str or Path
        Path to a TSV with at least *subject_col* and *age_col*.
    sample_id_parser : callable, optional
        Function ``sample_id → subject_id``.  Default: extract the first
        two hyphen-separated tokens (``"GTEX-XXXX-…" → "GTEX-XXXX"``).
    age_col : str
        Column name for age in *age_file* (default ``"AGE"``).
    subject_col : str
        Column name for subject ID in *age_file* (default ``"SUBJID"``).

    Returns
    -------
    DataFrame
        Standardised table with columns ``event_id, group, sample_id,
        k, N, age`` ready for :func:`splicedrift.fit`.
    """
    if sample_id_parser is None:

        def sample_id_parser(sid: str) -> str:  # type: ignore[misc]
            parts = sid.split("-")
            return "-".join(parts[:2])

    jdf = pd.read_csv(Path(junction_file), sep=sep)

    # Positional columns: last 4 are junc1, junc2, exclusion, PSI
    col_junc1 = jdf.columns[-4]
    col_junc2 = jdf.columns[-3]
    col_excl = jdf.columns[-2]

    jdf["k"] = pd.to_numeric(jdf[col_junc1], errors="coerce") + pd.to_numeric(
        jdf[col_junc2], errors="coerce"
    )
    jdf["N"] = jdf["k"] + 2 * pd.to_numeric(jdf[col_excl], errors="coerce")

    # Parse subject ID
    jdf["_subj_id"] = jdf["sample_id"].astype(str).map(sample_id_parser)

    # Load age info
    adf = pd.read_csv(Path(age_file), sep=sep)
    id2age = dict(zip(adf[subject_col], adf[age_col]))
    jdf["age"] = jdf["_subj_id"].map(id2age)

    # Rename tissue → group
    if "tissue" in jdf.columns:
        jdf["group"] = jdf["tissue"].str.replace("_junctions", "", regex=False)
    elif "group" not in jdf.columns:
        jdf["group"] = "all"

    out = jdf[["event_id", "group", "sample_id", "k", "N", "age"]].copy()
    out = out.dropna(subset=["k", "N", "age"])
    out["k"] = out["k"].astype(int)
    out["N"] = out["N"].astype(int)
    return out
