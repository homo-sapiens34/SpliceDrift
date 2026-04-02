# SpliceDrift

**Bayesian Beta-Binomial regression for age-related splicing drift analysis.**

SpliceDrift fits a probabilistic model to RNA-seq junction counts in order to
detect age-associated changes in both the *mean* and *variability* of exon
inclusion (PSI) across tissues and species.

---

## Motivation

Most available RNA-seq datasets (< 50 million reads per sample) provide
limited coverage for individual splicing events.  Percent Spliced In (PSI)
estimates derived from junction counts can therefore be imprecise, and direct
correlation or regression analyses on PSI point estimates are often
insufficiently sensitive.

SpliceDrift addresses this limitation by adopting a **probabilistic modelling
framework** that explicitly accounts for read counts supporting each splice
junction, rather than collapsing them into a single PSI value.

## Model

For each exon-skipping event, splicing is quantified by the number of
inclusion reads **k** and the total number of informative reads **N**.
Inclusion counts are modelled with a **Beta-Binomial distribution**
parameterised by the mean exon inclusion level (PSI) **μ** and a precision
(inverse-dispersion) parameter **φ**, both modelled as functions of
normalised age:

```
k_i  ~  BetaBinomial(N_i,  μ_i · φ_i,  (1 − μ_i) · φ_i)

logit(μ_i) = α₀  + α′ · age_z_i
 log(φ_i)  = β₀  + β′ · age_z_i
```

where `age_z` is age standardised to zero mean and unit variance.

Age-related effects are captured through **two key parameters**:

| Parameter | Name in code    | Interpretation |
|-----------|-----------------|----------------|
| **α′**    | `alpha_prime`   | Whether mean PSI *increases* (α′ > 0) or *decreases* (α′ < 0) with age. |
| **β′**    | `beta_prime`    | Whether splicing variability *increases* (β′ < 0, precision drops) or *decreases* (β′ > 0, stabilises) with age. |

Together these parameters allow age to influence both the average exon
inclusion level and the stability of splicing regulation.

> Beta and beta-binomial formulations are standard in alternative splicing
> analysis (e.g. rMATS and MAJIQ); however, to our knowledge this is the
> first approach to explicitly introduce linear dependence on age for both
> the mean and dispersion components of the model.
>
> — *Chervontseva, Interim Report #1 (December 2025)*

The model is fitted using Markov Chain Monte Carlo (MCMC) via
[PyMC](https://www.pymc.io/).  Convergence is checked automatically
(R̂, ESS, divergences) and the sampler is re-run with more conservative
settings if diagnostics fail.

## Installation

```bash
pip install splicedrift
```

Or install from source:

```bash
git clone https://github.com/zchervontseva/SpliceDrift.git
cd SpliceDrift
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyMC ≥ 5.0, ArviZ ≥ 0.15, NumPy, Pandas, Matplotlib, Seaborn,
  statsmodels, tqdm

## Quick start

### Python API

```python
import splicedrift

# 1. Load data
data = splicedrift.load_data("input.tsv")

# 2. Fit all (event, group) pairs
results = splicedrift.fit(data)

# 3. Save results
results.to_csv("summary.tsv", sep="\t", index=False)

# 4. Plot α′ vs β′ scatter (one panel per group)
fig = splicedrift.plot_scatter(results, save="scatter.png")

# 5. Plot PSI vs age for a specific event
fig = splicedrift.plot_event(
    data,
    event_id="HsaEX0000244",
    group="Brain - Cortex",
    save="event_plot.png",
)
```

### Command line

```bash
# Fit the model
splicedrift fit -i input.tsv -o results/summary.tsv -v

# Scatter plot (α′ vs β′)
splicedrift plot-scatter -i results/summary.tsv -o scatter.png

# Single-event plot (PSI vs age with posterior band)
splicedrift plot-event -i input.tsv -e HsaEX0000244 -g "Brain - Cortex" -o event.png
```

## Input format

SpliceDrift expects a **tab-separated** file with the following columns:

| Column      | Required | Description |
|-------------|----------|-------------|
| `event_id`  | ✅        | Splicing event identifier (e.g. `HsaEX0000244`). |
| `sample_id` | ✅        | Sample identifier. |
| `k`         | ✅        | Inclusion junction read count. |
| `N`         | ✅        | Total informative read count (`k + 2 × exclusion_count`). |
| `age`       | ✅        | Continuous covariate (typically donor age in years). |
| `group`     | optional | Tissue / condition / species label.  If absent, all rows are treated as one group. |

Example:

```
event_id	group	sample_id	k	N	age
HsaEX0000244	Brain - Cortex	GTEX-1117F	8	21	66
HsaEX0000244	Brain - Cortex	GTEX-111CU	4	15	57
HsaEX0000244	Lung	GTEX-1117F	12	30	66
```

### Computing k and N from raw junction counts

If your data contains raw junction counts rather than pre-computed `k`/`N`:

```
k = inclusion_junction_1 + inclusion_junction_2
N = k + 2 × exclusion_junction
```

For **VastDB / GTEx** formatted files, use the convenience loader:

```python
data = splicedrift.load_vastdb(
    "counts_psi__HsaEX0000244.tsv",
    "age_info.tsv",
)
```

## Output format

`splicedrift fit` produces a TSV with one row per (event, group) pair:

| Column                | Description |
|-----------------------|-------------|
| `event_id`            | Event identifier. |
| `group`               | Group / tissue label. |
| `n_samples`           | Number of samples used in the fit. |
| `age_mean`            | Mean age (for de-standardising). |
| `age_std`             | Std of age (for de-standardising). |
| `alpha_prime_mean`    | Posterior mean of **α′** (age effect on mean PSI). |
| `alpha_prime_hdi_lo`  | Lower bound of 95% HDI for α′. |
| `alpha_prime_hdi_hi`  | Upper bound of 95% HDI for α′. |
| `P_alpha_prime_gt_0`  | Posterior probability that α′ > 0. |
| `beta_prime_mean`     | Posterior mean of **β′** (age effect on variability). |
| `beta_prime_hdi_lo`   | Lower bound of 95% HDI for β′. |
| `beta_prime_hdi_hi`   | Upper bound of 95% HDI for β′. |
| `P_beta_prime_lt_0`   | Posterior probability that β′ < 0 (increased variability). |
| `diagnostics_ok`      | Whether MCMC convergence diagnostics passed. |

## Parameters

All MCMC parameters can be tuned via the Python API or CLI flags:

| Parameter         | Default | CLI flag           | Description |
|-------------------|---------|--------------------|-------------|
| `draws`           | 1000    | `--draws`          | Posterior draws per chain. |
| `tune`            | 1000    | `--tune`           | Warm-up steps per chain. |
| `chains`          | 4       | `--chains`         | Number of MCMC chains. |
| `cores`           | 1       | `--cores`          | CPU cores for chain sampling. |
| `target_accept`   | 0.9     | `--target-accept`  | NUTS target acceptance probability. |
| `random_seed`     | 42      | `--random-seed`    | Random seed for reproducibility. |
| `min_total`       | 5       | `--min-total`      | Minimum *N* per sample. |
| `rhat_max`        | 1.01    | `--rhat-max`       | Maximum acceptable R̂. |
| `ess_min`         | 500     | `--ess-min`        | Minimum acceptable bulk ESS. |
| `max_refits`      | 2       | `--max-refits`     | Auto-refits on convergence failure. |

## Plots

### α′ vs β′ scatter

One panel per group.  Each point is a splicing event.  Significance is
determined by converting posterior probabilities to two-sided p-values and
applying Benjamini–Hochberg FDR correction.

- **Red** — α′ significantly > 0 (inclusion increases with age)
- **Blue** — α′ significantly < 0 (inclusion decreases with age)
- **Gray** — not significant
- **Orange ×** — β′ significant (variability changes with age)

### PSI vs age event plot

Observed PSI values (scatter) overlaid with the posterior mean μ(age) curve
and 90% credible band.

## Example

Generate a synthetic dataset and run the full pipeline:

```bash
cd SpliceDrift
python examples/generate_example_data.py
splicedrift fit -i examples/example_input.tsv -o results.tsv -v
splicedrift plot-scatter -i results.tsv -o scatter.png
```

## Citation

If you use SpliceDrift, please cite:

> Chervontseva Z. (2026). SpliceDrift: Bayesian Beta-Binomial regression
> for age-related splicing drift analysis.
> https://github.com/zchervontseva/SpliceDrift

## License

[MIT](LICENSE)
