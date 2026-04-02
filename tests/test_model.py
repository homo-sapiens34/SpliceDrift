"""Basic tests for SpliceDrift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from splicedrift.io import load_data


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _make_synthetic(
    n_samples: int = 60,
    n_events: int = 2,
    n_groups: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a small synthetic dataset for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for ev_idx in range(n_events):
        for g_idx in range(n_groups):
            ages = rng.uniform(20, 80, size=n_samples)
            for i in range(n_samples):
                N = rng.integers(10, 100)
                psi = 0.5 + 0.003 * (ages[i] - 50)  # slight trend
                psi = np.clip(psi, 0.01, 0.99)
                k = rng.binomial(N, psi)
                rows.append(
                    {
                        "event_id": f"EV{ev_idx:04d}",
                        "group": f"group_{g_idx}",
                        "sample_id": f"S{ev_idx}_{g_idx}_{i}",
                        "k": int(k),
                        "N": int(N),
                        "age": float(ages[i]),
                    }
                )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Tests — data loading
# -------------------------------------------------------------------
class TestLoadData:
    def test_load_valid(self, tmp_path):
        df = _make_synthetic()
        fpath = tmp_path / "data.tsv"
        df.to_csv(fpath, sep="\t", index=False)

        loaded = load_data(fpath)
        assert set(loaded.columns) >= {"event_id", "group", "k", "N", "age"}
        assert len(loaded) == len(df)

    def test_missing_column_raises(self, tmp_path):
        df = _make_synthetic().drop(columns=["k"])
        fpath = tmp_path / "bad.tsv"
        df.to_csv(fpath, sep="\t", index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_data(fpath)

    def test_no_group_gets_default(self, tmp_path):
        df = _make_synthetic().drop(columns=["group"])
        fpath = tmp_path / "nogroup.tsv"
        df.to_csv(fpath, sep="\t", index=False)

        loaded = load_data(fpath)
        assert (loaded["group"] == "all").all()


# -------------------------------------------------------------------
# Tests — model (smoke tests only — fast settings)
# -------------------------------------------------------------------
class TestFitSingle:
    def test_smoke(self):
        """Ensure fit_single runs without error on tiny data."""
        from splicedrift.model import fit_single

        df = _make_synthetic(n_samples=30, n_events=1, n_groups=1)
        result = fit_single(
            df,
            draws=50,
            tune=50,
            chains=1,
            cores=1,
            max_refits=0,
            quiet=True,
        )
        s = result["summary"]
        assert "alpha_prime_mean" in s
        assert "beta_prime_mean" in s
        assert "P_alpha_prime_gt_0" in s
        assert 0 <= s["P_alpha_prime_gt_0"] <= 1

    def test_multi_event_raises(self):
        from splicedrift.model import fit_single

        df = _make_synthetic(n_events=2, n_groups=1)
        with pytest.raises(ValueError, match="exactly one event_id"):
            fit_single(df, draws=10, tune=10, chains=1, max_refits=0)


class TestFit:
    def test_batch(self):
        from splicedrift.model import fit

        df = _make_synthetic(n_samples=30, n_events=2, n_groups=1)
        results = fit(
            df,
            draws=50,
            tune=50,
            chains=1,
            cores=1,
            max_refits=0,
            quiet=True,
        )
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2  # 2 events × 1 group
        assert "alpha_prime_mean" in results.columns
        assert "beta_prime_mean" in results.columns
