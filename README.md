# DFS Lifetime & Experimentation Platform

Interactive analytics dashboard for Daily Fantasy Sports player behavior, built with **Dash + Plotly** and powered by methods from **biostatistics** and **clinical trial design**.

## 📊 What's Inside

| Module | Methods | Status |
|---|---|---|
| **Overview & EDA** | Distributions, correlations, state map, KPI cards | ✅ Ready |
| **Survival Analysis** | Kaplan–Meier, Cox PH, log-rank, cohort retention | 🔲 Stub |
| **A/B Test Engine** | Sample size calc (Lehr, Schoenfeld), Bayesian, sequential | 🔲 Stub |
| **Segmentation** | Kruskal–Wallis, Dunn's test, two-way ANOVA | 🔲 Stub |
| **Churn Model** | LogReg, CatBoost, SHAP, calibration, what-if | 🔲 Stub |

## 📁 Dataset

**"Patterns of Daily Fantasy Sport Play: Tackling the Issues"**

- **Source:** The Transparency Project, Division on Addiction, Cambridge Health Alliance (Harvard Medical School)
- **Publication:** Nelson, S. E., et al. (2019). *Journal of Gambling Studies*, 35(1), 181–204. [DOI](https://doi.org/10.1007/s10899-018-09817-w)
- **Cohort:** 10,385 DraftKings players who enrolled Aug–Sep 2014 and entered ≥1 paid NFL contest
- **Period:** Aug 22, 2014 – Jan 25, 2015

### Getting the data

1. Request download from [thetransparencyproject.org](http://thetransparencyproject.org/Availabledataset)
2. Place the 6 CSV files into `data/raw/`:
   ```
   data/raw/
   ├── TacklingData1Codes.csv
   ├── TacklingData2Cohort.csv
   ├── TacklingData3NFL.csv
   ├── TacklingData4Not.csv
   ├── TacklingData5All.csv
   └── TacklingData6Play.csv
   ```

## 🚀 Quick Start

### Local (Python)

```bash
# Clone
git clone https://github.com/<YOUR_USERNAME>/dfs-lifetime-platform.git
cd dfs-lifetime-platform

# Virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# Validate data pipeline
python -m app.components.data_loader

# Run the app
python app/app.py
```

Open [http://localhost:8050](http://localhost:8050)

### Docker

```bash
docker compose up --build
```

## 🏗️ Project Structure

```
dfs-lifetime-platform/
├── app/
│   ├── app.py                  # Dash entrypoint + sidebar
│   ├── pages/
│   │   ├── overview.py         # Module 1 — EDA dashboard
│   │   ├── survival.py         # Module 2 — stub
│   │   ├── ab_testing.py       # Module 3 — stub
│   │   ├── segmentation.py     # Module 4 — stub
│   │   └── churn_model.py      # Module 5 — stub
│   ├── components/
│   │   ├── data_loader.py      # Data pipeline & feature engineering
│   │   ├── plots.py            # Reusable Plotly figure factories
│   │   ├── stats.py            # Statistical test wrappers
│   │   └── layout_utils.py     # Dash UI components
│   └── assets/
│       └── custom.css
├── data/
│   ├── raw/                    # Original CSVs (gitignored)
│   ├── processed/              # Cached parquet (gitignored)
│   └── codebook/               # PDF codebooks (gitignored)
├── models/                     # Trained model artifacts
├── notebooks/                  # Jupyter exploration
├── tests/                      # pytest suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🧬 Biostatistics ↔ iGaming Mapping

This project deliberately bridges clinical biostatistics and product analytics:

| Biostatistics | iGaming Application |
|---|---|
| Kaplan–Meier with right censoring | Player retention curves (77% censored) |
| Cox Proportional Hazards | Hazard ratios for churn risk factors |
| Phase III sample size (Schoenfeld) | A/B test design for time-to-event metrics |
| O'Brien–Fleming boundaries | Sequential testing / interim analysis |
| Shapiro–Wilk → Kruskal–Wallis | Non-parametric comparison of player segments |
| Logistic regression (OR + CI) | Churn prediction with interpretable coefficients |

## 📝 License

Dataset provided for research purposes by the Division on Addiction, Cambridge Health Alliance. See [The Transparency Project](http://thetransparencyproject.org/) for terms.

Application code: MIT License.
