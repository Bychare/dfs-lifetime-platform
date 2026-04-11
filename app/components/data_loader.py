"""
Data loader and feature engineering pipeline.

Reads the 6 raw CSV files from the Transparency Project DraftKings dataset,
joins them into a single analytical dataframe, and engineers features for
all downstream modules (survival, segmentation, churn model).

Reference:
    Nelson, S. E., et al. (2019). Patterns of Daily Fantasy Sport Play:
    Tackling the Issues. Journal of Gambling Studies, 35(1), 181-204.
"""

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"
PARQUET_PATH = PROCESSED_DIR / "players_features.parquet"
PICKLE_PATH = PROCESSED_DIR / "players_features.pkl"


# ---------------------------------------------------------------------------
# Raw loaders
# ---------------------------------------------------------------------------

def load_codes() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "TacklingData1Codes.csv")


def load_cohort() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "TacklingData2Cohort.csv")


def load_nfl() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "TacklingData3NFL.csv", parse_dates=["Date1st", "DateLst"])


def load_not_nfl() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "TacklingData4Not.csv", parse_dates=["Date1st", "DateLst"])


def load_all_activity() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "TacklingData5All.csv", parse_dates=["Date1st", "DateLst"])


def load_play_types() -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / "TacklingData6Play.csv")


# ---------------------------------------------------------------------------
# State / country name lookup
# ---------------------------------------------------------------------------

def build_code_lookup() -> tuple[dict, dict]:
    """Return (state_map, country_map) dicts from IDNumber -> Name."""
    codes = load_codes()
    state_map = dict(
        zip(
            codes.loc[codes["Level"] == "State", "IDNumber"],
            codes.loc[codes["Level"] == "State", "Name"],
        )
    )
    country_map = dict(
        zip(
            codes.loc[codes["Level"] == "Nation", "IDNumber"],
            codes.loc[codes["Level"] == "Nation", "Name"],
        )
    )
    return state_map, country_map


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

NFL_SEASON_END = pd.Timestamp("2015-01-25")

CONTEST_TYPE_NAMES = {
    "Cnt1": "50/50",
    "Cnt2": "Head-to-Head",
    "Cnt3": "Multiplier",
    "Cnt4": "League",
    "Cnt5": "Move Your Way Up",
    "Cnt6": "Other",
}


def _shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a distribution (natural log)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def build_features(force: bool = False) -> pd.DataFrame:
    """
    Join all tables and engineer features. Returns one row per player.

    Caches result to parquet. Pass force=True to rebuild.
    """
    if not force:
        if PARQUET_PATH.exists():
            try:
                return pd.read_parquet(PARQUET_PATH)
            except (ImportError, ValueError, OSError):
                pass
        if PICKLE_PATH.exists():
            return pd.read_pickle(PICKLE_PATH)

    # --- Load raw tables ---
    cohort = load_cohort()
    all_act = load_all_activity()
    nfl = load_nfl()
    play = load_play_types()
    state_map, country_map = build_code_lookup()

    # --- Start with all_activity (10 385 rows, one per player who played NFL) ---
    df = all_act.copy()

    # --- Survival variables ---
    df["is_churned"] = (df["DateLst"] < NFL_SEASON_END).astype(int)
    df["duration_days"] = (df["DateLst"] - df["Date1st"]).dt.days
    # Ensure minimum 1 day for survival models
    df["duration_days"] = df["duration_days"].clip(lower=1)

    # --- Financial features ---
    df["net_pnl"] = df["TotWinnings"] - df["TotFees"]
    df["net_pnl_pct"] = np.where(
        df["TotFees"] > 0,
        df["net_pnl"] / df["TotFees"] * 100,
        0.0,
    )
    df["log_total_fees"] = np.log1p(df["TotFees"])
    df["win_rate"] = np.where(
        df["nCont"] > 0, df["nUserUp"] / df["nCont"], 0.0
    )

    # --- Intensity features ---
    df["intensity"] = np.where(
        df["nDays"] > 0, df["nCont"] / df["nDays"], 0.0
    )
    df["entries_per_contest"] = np.where(
        df["nCont"] > 0, df["nEntries"] / df["nCont"], 0.0
    )
    df["lineups_per_contest"] = np.where(
        df["nCont"] > 0, df["nLineups"] / df["nCont"], 0.0
    )

    # --- Join cohort demographics ---
    cohort_cols = ["UserID", "RegStateID", "RegCountryID", "BirthYear", "RiskScore"]
    df = df.merge(cohort[cohort_cols], on="UserID", how="left")

    # Readable names
    df["state_name"] = df["RegStateID"].map(state_map).fillna("Unknown")
    df["country_name"] = df["RegCountryID"].map(country_map).fillna("Unknown")

    # Age (only for those with known birth year)
    df["has_age"] = (df["BirthYear"] > 1900).astype(int)
    df["age"] = np.where(df["has_age"] == 1, 2014 - df["BirthYear"], np.nan)

    # Age groups
    bins = [0, 25, 30, 35, 45, 100]
    labels = ["18-24", "25-29", "30-34", "35-44", "45+"]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # --- Join play types ---
    df = df.merge(play, on="UserID", how="left")

    # Multi-sport flag
    df["is_multisport"] = (df["DidNBA"] | df["DidOth"]).astype(int)
    df["n_sports"] = df["DidNFL"].astype(int) + df["DidNBA"].astype(int) + df["DidOth"].astype(int)

    # --- Contest type features ---
    cnt_cols = ["Cnt1", "Cnt2", "Cnt3", "Cnt4", "Cnt5", "Cnt6"]
    cnt_values = df[cnt_cols].values

    # Total contests across types
    df["total_type_contests"] = cnt_values.sum(axis=1)

    # Dominant contest type (>50% of activity)
    cnt_fractions = np.divide(
        cnt_values,
        df["total_type_contests"].values[:, None],
        out=np.zeros_like(cnt_values, dtype=float),
        where=df["total_type_contests"].values[:, None] > 0,
    )
    dominant_idx = np.argmax(cnt_fractions, axis=1)
    dominant_frac = np.max(cnt_fractions, axis=1)
    type_labels = list(CONTEST_TYPE_NAMES.values())
    df["dominant_type"] = [
        type_labels[i] if dominant_frac[j] >= 0.5 else "Mixed"
        for j, i in enumerate(dominant_idx)
    ]

    # Shannon entropy (diversity of contest types)
    df["type_diversity"] = [_shannon_entropy(row) for row in cnt_values]

    # --- RiskScore quartiles ---
    df["risk_quartile"] = pd.qcut(
        df["RiskScore"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    )

    # --- AvgBuyIn quartiles ---
    df["buyin_quartile"] = pd.qcut(
        df["AvgBuyIn"], q=4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    )

    # --- Save ---
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(PARQUET_PATH, index=False)
    except (ImportError, ValueError, OSError):
        df.to_pickle(PICKLE_PATH)

    return df


# ---------------------------------------------------------------------------
# Cached accessor used by Dash callbacks
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_players() -> pd.DataFrame:
    """Load or build the player features dataframe (cached in memory)."""
    return build_features()


# ---------------------------------------------------------------------------
# NFL-specific daily data (for cohort retention analysis)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_nfl_activity() -> pd.DataFrame:
    """NFL contest activity with dates."""
    return load_nfl()


# ---------------------------------------------------------------------------
# Quick validation helper
# ---------------------------------------------------------------------------

def validate_against_publication():
    """
    Reproduce key numbers from Nelson et al. (2019) as a sanity check.
    Run this as a script: python -m app.components.data_loader
    """
    df = build_features(force=True)

    print("=" * 60)
    print("VALIDATION vs. Nelson et al. (2019)")
    print("=" * 60)
    print(f"\nCohort size: {len(df)} (expected: 10,385)")
    print(f"DidNFL: {df['DidNFL'].sum()} (expected: 10,385)")
    print(f"DidNBA: {df['DidNBA'].sum()} (expected: ~3,065)")
    print(f"DidOth: {df['DidOth'].sum()} (expected: ~4,429)")

    print(f"\n--- All-contest metrics (Table 2 in paper) ---")
    for col in ["nDays", "nCont", "TotFees", "TotWinnings"]:
        s = df[col]
        print(f"{col}: mean={s.mean():.1f}, sd={s.std():.1f}, median={s.median():.1f}")

    print(f"\n--- Churn ---")
    print(f"Active at season end: {(~df['is_churned'].astype(bool)).sum()} ({(~df['is_churned'].astype(bool)).mean()*100:.1f}%)")
    print(f"Churned before end:   {df['is_churned'].sum()} ({df['is_churned'].mean()*100:.1f}%)")

    print(f"\n--- Financial ---")
    winners = (df["net_pnl"] > 0).sum()
    losers = (df["net_pnl"] < 0).sum()
    print(f"Winners: {winners} ({winners/len(df)*100:.1f}%)")
    print(f"Losers:  {losers} ({losers/len(df)*100:.1f}%)")
    print(f"Median net P&L: ${df['net_pnl'].median():.2f}")


if __name__ == "__main__":
    validate_against_publication()
