# DFS Lifetime & Experimentation Platform

Interactive analytics dashboard for Daily Fantasy Sports player behavior, built with **Dash + Plotly**.

This project is positioned as a **product and behavioral analytics case study**: retention, experimentation, segmentation, and churn analysis on a real player cohort.

## What This Project Shows

- End-to-end analytical workflow: raw data ingestion, feature engineering, statistical analysis, and dashboard delivery
- Product analytics thinking: retention, engagement, monetization, experimentation, and player segmentation
- Reproducibility: modular code, reusable helpers, and `pytest` coverage

## What's Inside

| Module | Focus | Status |
|---|---|---|
| **Overview & EDA** | KPI cards, distributions, correlations, state-level breakdowns | ✅ Ready |
| **Retention Analysis** | Retention curves, churn timing, cohort retention, driver analysis | ✅ Ready |
| **A/B Testing** | Sample-size planning, simulated experiments, Bayesian uplift, sequential monitoring | ✅ Ready |
| **Segmentation** | Segment comparison, pairwise significance checks, segment footprint, behavioral profile views | ✅ Ready |
| **Churn Model** | Predictive churn scoring and driver interpretation | 🔲 Planned |

## Core Analytical Questions

- Which player segments retain longer and churn faster?
- Which segments generate disproportionate share of fees relative to their size?
- How should an experiment be sized for retention or conversion impact?
- Which segment differences are statistically meaningful versus noise?
- What behavioral patterns separate higher-risk and lower-risk users?

## Dataset

**"Patterns of Daily Fantasy Sport Play: Tackling the Issues"**

- **Source:** The Transparency Project, Division on Addiction, Cambridge Health Alliance
- **Publication:** Nelson, S. E., et al. (2019). *Journal of Gambling Studies*, 35(1), 181-204. [DOI](https://doi.org/10.1007/s10899-018-09817-w)
- **Cohort:** 10,385 DraftKings players who enrolled in Aug-Sep 2014 and entered at least one paid NFL contest
- **Period:** Aug 22, 2014 to Jan 25, 2015

### Getting the data

1. Request access from [thetransparencyproject.org](http://thetransparencyproject.org/Availabledataset)
2. Place the six CSV files into `data/raw/`

```text
data/raw/
├── TacklingData1Codes.csv
├── TacklingData2Cohort.csv
├── TacklingData3NFL.csv
├── TacklingData4Not.csv
├── TacklingData5All.csv
└── TacklingData6Play.csv
```

## Quick Start

### Local

```bash
git clone https://github.com/<YOUR_USERNAME>/dfs-lifetime-platform.git
cd dfs-lifetime-platform

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python -m app.components.data_loader
python app/app.py
```

Open [http://localhost:8050](http://localhost:8050)

### Run tests

```bash
pytest
```

### Docker

```bash
docker compose up --build
```

## Dashboard Modules

### 1. Overview & EDA

- Portfolio-level KPI summary
- Distribution analysis for fees, winnings, activity, and risk
- Correlation matrix for key player metrics
- Geographic and contest-type breakdowns

### 2. Retention Analysis

- Retention curves for player lifetime
- Group comparison for churn timing
- Driver analysis for churn risk
- Cohort retention heatmap
- Alternative milestone endpoint: time to Nth contest

### 3. A/B Testing

- Sample-size calculator for binary, continuous, and time-to-event metrics
- Simulated experiments on observed player segments
- Frequentist and Bayesian result views
- Bootstrap confidence intervals
- Sequential monitoring with stopping boundaries

### 4. Segmentation

- Segment median and IQR comparison
- Omnibus significance test plus pairwise post-hoc checks
- Segment footprint: player share vs fee share
- Segment profile heatmap relative to overall portfolio baseline

### 5. Churn Model

- Reserved for predictive churn scoring
- Planned additions: classification model, calibration, feature importance, what-if scoring

## Project Structure

```text
dfs-lifetime-platform/
├── app/
│   ├── app.py
│   ├── pages/
│   │   ├── overview.py
│   │   ├── survival.py
│   │   ├── ab_testing.py
│   │   ├── segmentation.py
│   │   └── churn_model.py
│   ├── components/
│   │   ├── data_loader.py
│   │   ├── plots.py
│   │   ├── stats.py
│   │   ├── survival_helpers.py
│   │   ├── ab_testing_helpers.py
│   │   ├── segmentation_helpers.py
│   │   └── layout_utils.py
│   └── assets/
│       └── custom.css
├── data/
│   ├── raw/
│   ├── processed/
│   └── codebook/
├── models/
├── notebooks/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Tech Stack

- **Python**
- **Dash**
- **Plotly**
- **Polars**
- **SciPy**
- **statsmodels**
- **scikit-learn**
- **pytest**

## Why This Is Relevant For A Data Analyst Role

- Turns a raw multi-table dataset into a usable analytical layer
- Frames business questions in terms of retention, monetization, and experimentation
- Balances descriptive analysis, statistical testing, and decision-oriented interpretation
- Demonstrates how to communicate results through a usable dashboard rather than notebooks alone

## License

Dataset usage is subject to the terms of [The Transparency Project](http://thetransparencyproject.org/).

Application code: MIT License.
