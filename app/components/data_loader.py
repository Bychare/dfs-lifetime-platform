"""
Data loader and feature engineering pipeline.

Reads the 6 raw CSV files from the Transparency Project DraftKings dataset,
joins them into a single analytical frame, and engineers features for
downstream modules.
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"
PARQUET_PATH = PROCESSED_DIR / "players_features.parquet"


def load_codes() -> pl.DataFrame:
    return pl.read_csv(RAW_DIR / "TacklingData1Codes.csv")


def load_cohort() -> pl.DataFrame:
    return pl.read_csv(RAW_DIR / "TacklingData2Cohort.csv")


def load_nfl() -> pl.DataFrame:
    return pl.read_csv(RAW_DIR / "TacklingData3NFL.csv", try_parse_dates=True)


def load_not_nfl() -> pl.DataFrame:
    return pl.read_csv(RAW_DIR / "TacklingData4Not.csv", try_parse_dates=True)


def load_all_activity() -> pl.DataFrame:
    return pl.read_csv(RAW_DIR / "TacklingData5All.csv", try_parse_dates=True)


def load_play_types() -> pl.DataFrame:
    return pl.read_csv(RAW_DIR / "TacklingData6Play.csv")


def build_code_lookup() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return state and country lookup frames."""
    codes = load_codes()
    state_lookup = (
        codes.filter(pl.col("Level") == "State")
        .select(
            pl.col("IDNumber").alias("RegStateID"),
            pl.col("Name").alias("state_name"),
        )
    )
    country_lookup = (
        codes.filter(pl.col("Level") == "Nation")
        .select(
            pl.col("IDNumber").alias("RegCountryID"),
            pl.col("Name").alias("country_name"),
        )
    )
    return state_lookup, country_lookup


NFL_SEASON_END = datetime(2015, 1, 25)

CONTEST_TYPE_NAMES = {
    "Cnt1": "50/50",
    "Cnt2": "Head-to-Head",
    "Cnt3": "Multiplier",
    "Cnt4": "League",
    "Cnt5": "Move Your Way Up",
    "Cnt6": "Other",
}


def _shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _quartile_expr(column: str, q1: float, q2: float, q3: float) -> pl.Expr:
    return (
        pl.when(pl.col(column).is_null()).then(None)
        .when(pl.col(column) <= q1).then(pl.lit("Q1 (Low)"))
        .when(pl.col(column) <= q2).then(pl.lit("Q2"))
        .when(pl.col(column) <= q3).then(pl.lit("Q3"))
        .otherwise(pl.lit("Q4 (High)"))
    )


def _dominant_type_expr() -> pl.Expr:
    return (
        pl.when(pl.col("total_type_contests") == 0).then(pl.lit("Mixed"))
        .when((pl.col("_max_type_count") / pl.col("total_type_contests")) < 0.5).then(pl.lit("Mixed"))
        .when(pl.col("_dominant_idx") == 0).then(pl.lit("50/50"))
        .when(pl.col("_dominant_idx") == 1).then(pl.lit("Head-to-Head"))
        .when(pl.col("_dominant_idx") == 2).then(pl.lit("Multiplier"))
        .when(pl.col("_dominant_idx") == 3).then(pl.lit("League"))
        .when(pl.col("_dominant_idx") == 4).then(pl.lit("Move Your Way Up"))
        .otherwise(pl.lit("Other"))
    )


def build_features(force: bool = False) -> pl.DataFrame:
    """Join all raw tables and engineer one row per player."""
    if not force and PARQUET_PATH.exists():
        return pl.read_parquet(PARQUET_PATH)

    cohort = load_cohort()
    all_act = load_all_activity()
    play = load_play_types()
    state_lookup, country_lookup = build_code_lookup()

    df = (
        all_act
        .join(
            cohort.select(["UserID", "RegStateID", "RegCountryID", "BirthYear", "RiskScore"]),
            on="UserID",
            how="left",
        )
        .join(play, on="UserID", how="left")
        .join(state_lookup, on="RegStateID", how="left")
        .join(country_lookup, on="RegCountryID", how="left")
        .with_columns([
            pl.col("state_name").fill_null("Unknown"),
            pl.col("country_name").fill_null("Unknown"),
            (pl.col("DateLst") < pl.lit(NFL_SEASON_END)).cast(pl.Int8).alias("is_churned"),
            (pl.col("DateLst") - pl.col("Date1st")).dt.total_days().clip(lower_bound=1).alias("duration_days"),
            (pl.col("TotWinnings") - pl.col("TotFees")).alias("net_pnl"),
            pl.when(pl.col("TotFees") > 0)
            .then((pl.col("TotWinnings") - pl.col("TotFees")) / pl.col("TotFees") * 100)
            .otherwise(0.0)
            .alias("net_pnl_pct"),
            pl.col("TotFees").log1p().alias("log_total_fees"),
            pl.when(pl.col("nCont") > 0)
            .then(pl.col("nUserUp") / pl.col("nCont"))
            .otherwise(0.0)
            .alias("win_rate"),
            pl.when(pl.col("nDays") > 0)
            .then(pl.col("nCont") / pl.col("nDays"))
            .otherwise(0.0)
            .alias("intensity"),
            pl.when(pl.col("nCont") > 0)
            .then(pl.col("nEntries") / pl.col("nCont"))
            .otherwise(0.0)
            .alias("entries_per_contest"),
            pl.when(pl.col("nCont") > 0)
            .then(pl.col("nLineups") / pl.col("nCont"))
            .otherwise(0.0)
            .alias("lineups_per_contest"),
            (pl.col("BirthYear") > 1900).cast(pl.Int8).alias("has_age"),
            pl.when(pl.col("BirthYear") > 1900)
            .then(2014 - pl.col("BirthYear"))
            .otherwise(None)
            .alias("age"),
            ((pl.col("DidNBA") == 1) | (pl.col("DidOth") == 1)).cast(pl.Int8).alias("is_multisport"),
            (
                (pl.col("DidNFL") == 1).cast(pl.Int8)
                + (pl.col("DidNBA") == 1).cast(pl.Int8)
                + (pl.col("DidOth") == 1).cast(pl.Int8)
            ).alias("n_sports"),
            pl.sum_horizontal("Cnt1", "Cnt2", "Cnt3", "Cnt4", "Cnt5", "Cnt6").alias("total_type_contests"),
            pl.max_horizontal("Cnt1", "Cnt2", "Cnt3", "Cnt4", "Cnt5", "Cnt6").alias("_max_type_count"),
            pl.concat_list(["Cnt1", "Cnt2", "Cnt3", "Cnt4", "Cnt5", "Cnt6"]).list.arg_max().alias("_dominant_idx"),
        ])
    )

    risk_q1, risk_q2, risk_q3 = df.select(
        pl.col("RiskScore").quantile(0.25).alias("q1"),
        pl.col("RiskScore").quantile(0.50).alias("q2"),
        pl.col("RiskScore").quantile(0.75).alias("q3"),
    ).row(0)
    buy_q1, buy_q2, buy_q3 = df.select(
        pl.col("AvgBuyIn").quantile(0.25).alias("q1"),
        pl.col("AvgBuyIn").quantile(0.50).alias("q2"),
        pl.col("AvgBuyIn").quantile(0.75).alias("q3"),
    ).row(0)

    df = (
        df.with_columns([
            pl.when(pl.col("age").is_null()).then(None)
            .when(pl.col("age") < 25).then(pl.lit("18-24"))
            .when(pl.col("age") < 30).then(pl.lit("25-29"))
            .when(pl.col("age") < 35).then(pl.lit("30-34"))
            .when(pl.col("age") < 45).then(pl.lit("35-44"))
            .otherwise(pl.lit("45+"))
            .alias("age_group"),
            _dominant_type_expr().alias("dominant_type"),
            _quartile_expr("RiskScore", risk_q1, risk_q2, risk_q3).alias("risk_quartile"),
            _quartile_expr("AvgBuyIn", buy_q1, buy_q2, buy_q3).alias("buyin_quartile"),
            pl.struct(["Cnt1", "Cnt2", "Cnt3", "Cnt4", "Cnt5", "Cnt6"])
            .map_elements(
                lambda row: _shannon_entropy(
                    np.array([row["Cnt1"], row["Cnt2"], row["Cnt3"], row["Cnt4"], row["Cnt5"], row["Cnt6"]], dtype=float)
                ),
                return_dtype=pl.Float64,
            )
            .alias("type_diversity"),
        ])
        .drop(["_max_type_count", "_dominant_idx"])
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(PARQUET_PATH)
    return df


@lru_cache(maxsize=1)
def get_players() -> pl.DataFrame:
    return build_features()


@lru_cache(maxsize=1)
def get_nfl_activity() -> pl.DataFrame:
    return load_nfl()


def validate_against_publication():
    df = build_features(force=True)

    print("=" * 60)
    print("VALIDATION vs. Nelson et al. (2019)")
    print("=" * 60)
    print(f"\nCohort size: {df.height} (expected: 10,385)")
    print(f"DidNFL: {df['DidNFL'].sum()} (expected: 10,385)")
    print(f"DidNBA: {df['DidNBA'].sum()} (expected: ~3,065)")
    print(f"DidOth: {df['DidOth'].sum()} (expected: ~4,429)")

    print("\n--- All-contest metrics (Table 2 in paper) ---")
    for col in ["nDays", "nCont", "TotFees", "TotWinnings"]:
        series = df[col]
        print(f"{col}: mean={series.mean():.1f}, sd={series.std():.1f}, median={series.median():.1f}")

    active = df.filter(pl.col("is_churned") == 0).height
    churned = df.filter(pl.col("is_churned") == 1).height
    print("\n--- Churn ---")
    print(f"Active at season end: {active} ({active / df.height * 100:.1f}%)")
    print(f"Churned before end:   {churned} ({churned / df.height * 100:.1f}%)")

    winners = df.filter(pl.col("net_pnl") > 0).height
    losers = df.filter(pl.col("net_pnl") < 0).height
    print("\n--- Financial ---")
    print(f"Winners: {winners} ({winners / df.height * 100:.1f}%)")
    print(f"Losers:  {losers} ({losers / df.height * 100:.1f}%)")
    print(f"Median net P&L: ${df['net_pnl'].median():.2f}")


if __name__ == "__main__":
    validate_against_publication()
