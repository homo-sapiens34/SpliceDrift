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
    n_samples: int = 120,
    n_events: int = 5,
    groups: list[str] | None = None,
    seed: int = 42,
    output: str | Path = "examples/example_input.tsv",
) -> pd.DataFrame:
    if groups is None:
        groups = ["Tissue_A", "Tissue_B"]

    rng = np.random.default_rng(seed)
    rows = []

    for ev_idx in range(n_events):
        # Random "true" parameters for this event
        true_alpha = rng.normal(0, 0.3)      # intercept_mu (logit scale)
        true_alpha_prime = rng.normal(0, 0.5) # α' — age effect on mean
        true_phi0 = rng.normal(2.0, 0.5)      # intercept_phi (log scale)
        true_beta_prime = rng.normal(0, 0.3)   # β' — age effect on precision

        for grp in groups:
            ages = rng.uniform(20, 85, size=n_samples)
            for i in range(n_samples):
                age_z = (ages[i] - 52.5) / 18.0
                logit_mu = true_alpha + true_alpha_prime * age_z
                mu = 1 / (1 + np.exp(-logit_mu))
                phi = np.exp(true_phi0 + true_beta_prime * age_z)

                a = mu * phi
                b = (1 - mu) * phi

                # Clamp to avoid degenerate Beta
                a = max(a, 0.01)
                b = max(b, 0.01)

                N = rng.integers(15, 200)
                psi = rng.beta(a, b)
                k = rng.binomial(N, psi)

                rows.append(
                    {
                        "event_id": f"HsaEX{ev_idx:07d}",
                        "group": grp,
                        "sample_id": f"SAMPLE-{grp}-{i:04d}",
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
