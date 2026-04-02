"""Generate a small synthetic example dataset for SpliceDrift.

Run this script to produce ``examples/example_input.tsv``::

    python examples/generate_example_data.py

The generated file can then be used with::

    splicedrift fit -i examples/example_input.tsv -o results.tsv -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate(
    *,
    n_samples: int = 20,
    n_events: int = 10,
    groups: list[str] | None = None,
    seed: int = 42,
    output: str | Path = "examples/example_input.tsv",
) -> pd.DataFrame:
    if groups is None:
        groups = ["Brain - Cortex", "Lung", "Heart - Left Ventricle", "Liver"]

    rng = np.random.default_rng(seed)
    rows = []

    for ev_idx in range(n_events):
        # Assign each event to a biological "archetype" to create realistic
        # clusters in the α'/β' plane:
        #   ~40% no effect (null)
        #   ~25% inclusion decreases + variability increases (aging decay)
        #   ~15% inclusion increases + variability increases
        #   ~10% inclusion decreases, variability stable
        #   ~10% strong effect on variability only
        archetype = rng.random()
        if archetype < 0.40:
            # Null — small noise
            true_alpha_prime = rng.normal(0, 0.08)
            true_beta_prime = rng.normal(0, 0.08)
        elif archetype < 0.65:
            # Aging decay: inclusion ↓, variability ↑
            true_alpha_prime = rng.normal(-0.45, 0.15)
            true_beta_prime = rng.normal(-0.50, 0.15)
        elif archetype < 0.80:
            # Inclusion ↑, variability ↑
            true_alpha_prime = rng.normal(0.40, 0.12)
            true_beta_prime = rng.normal(-0.35, 0.15)
        elif archetype < 0.90:
            # Inclusion ↓, variability stable
            true_alpha_prime = rng.normal(-0.35, 0.10)
            true_beta_prime = rng.normal(0.0, 0.08)
        else:
            # Variability only
            true_alpha_prime = rng.normal(0, 0.06)
            true_beta_prime = rng.normal(-0.60, 0.15)

        true_alpha = rng.normal(0, 0.5)      # intercept_mu (logit scale)
        true_phi0 = rng.normal(2.5, 0.5)      # intercept_phi (log scale)

        for grp in groups:
            # Add small tissue-specific perturbation
            tissue_shift_a = rng.normal(0, 0.05)
            tissue_shift_b = rng.normal(0, 0.05)

            ages = rng.uniform(20, 85, size=n_samples)
            for i in range(n_samples):
                age_z = (ages[i] - 52.5) / 18.0
                logit_mu = true_alpha + (true_alpha_prime + tissue_shift_a) * age_z
                mu = 1 / (1 + np.exp(-logit_mu))
                phi = np.exp(true_phi0 + (true_beta_prime + tissue_shift_b) * age_z)

                a = mu * phi
                b = (1 - mu) * phi

                # Clamp to avoid degenerate Beta
                a = max(a, 0.01)
                b = max(b, 0.01)

                N = rng.integers(10, 300)
                psi = rng.beta(a, b)
                k = rng.binomial(N, psi)

                rows.append(
                    {
                        "event_id": f"HsaEX{ev_idx:07d}",
                        "group": grp,
                        "sample_id": f"SAMPLE-{grp.replace(' ', '')}-{i:04d}",
                        "k": int(k),
                        "N": int(N),
                        "age": round(float(ages[i]), 1),
                    }
                )

    df = pd.DataFrame(rows)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {len(df)} rows ({n_events} events × {len(groups)} groups) → {out}")
    return df


if __name__ == "__main__":
    generate()
