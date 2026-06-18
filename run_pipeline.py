"""Full LedgerLens detection pipeline entry point.

Usage:
    python run_pipeline.py --since 2024-01-01

Pipeline stages:
    1. Load historical trades and order-book events for all watched asset
       pairs (ingestion)
    2. Load account activity and build the wallet funding graph (optional,
       skipped with ``--no-graph``)
    3. Build the per-wallet feature matrix (Benford + ML + order-book features)
    4. Score each wallet with the trained ensemble (model_inference)
    5. Persist scored wallets via `RiskScoreStore`, optionally submit flagged
       wallets on-chain via the `ledgerlens-score` contract, and output those
       flagged above `config.RISK_SCORE_FLAG_THRESHOLD`

Stage 4 requires trained models in `config.MODEL_DIR` — run
`detection/model_training.py` against a labelled dataset first. Until
models are trained, this script falls back to reporting Benford-only flags
(and persistence is skipped, since the `RiskScore` shape isn't available).
"""

import argparse
from datetime import UTC, datetime

import pandas as pd

from config import config
from detection.feature_engineering import build_feature_matrix
from detection.risk_score_store import RiskScoreStore
from detection.wallet_graph import build_funding_graph
from ingestion.account_activity_loader import load_accounts_activity
from ingestion.historical_loader import load_watched_pairs_to_dataframe
from ingestion.orderbook_loader import load_accounts_orderbook_events
from utils.logging import get_logger

logger = get_logger(__name__)


def watched_pairs_label() -> str:
    """A single label identifying the configured set of watched pairs.

    Used as the `asset_pair` key for persisted `RiskScore` records until
    per-pair feature attribution is implemented (the feature matrix is
    currently built across all watched pairs combined).
    """
    if not config.WATCHED_ASSET_PAIRS:
        return "ALL"
    return "+".join(f"{code}:{issuer}" for code, issuer in config.WATCHED_ASSET_PAIRS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LedgerLens detection pipeline")
    parser.add_argument(
        "--since",
        type=lambda s: datetime.fromisoformat(s),
        default=None,
        help="ISO date to start loading historical trades from (default: all available)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing scored wallets to RISK_SCORE_DB_URL",
    )
    parser.add_argument(
        "--no-orderbook",
        action="store_true",
        help="Skip loading order-book events (faster, but order_cancellation_rate stays 0)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip loading account activity and building the funding graph (faster, but "
        "funding_source_similarity and network_centrality stay 0)",
    )
    parser.add_argument(
        "--submit-onchain",
        action="store_true",
        help="Submit flagged wallets' RiskScore to the ledgerlens-score contract",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all pipeline stages but skip all writes (DB persist and on-chain submission).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        logger.info("[DRY RUN] No data will be written.")

    logger.info("[1/5] Loading trades for watched pairs: %s", config.WATCHED_ASSET_PAIRS)
    trades_df = load_watched_pairs_to_dataframe(start_time=args.since)
    logger.info("      Loaded %d trades", len(trades_df))

    orderbook_events = None
    if not args.no_orderbook and not trades_df.empty:
        logger.info("[2/5] Loading order-book events")
        wallets = pd.unique(trades_df[["base_account", "counter_account"]].values.ravel())
        orderbook_events = load_accounts_orderbook_events(list(wallets))
        logger.info("      Loaded %d order-book events", len(orderbook_events))

    funding_graph = None
    if not args.no_graph and not trades_df.empty:
        logger.info("[3/5] Loading account activity and building funding graph")
        wallets = pd.unique(trades_df[["base_account", "counter_account"]].values.ravel())
        activities = load_accounts_activity(list(wallets))
        funding_graph = build_funding_graph(activities)
        logger.info(
            "      Built funding graph: %d nodes, %d edges",
            funding_graph.number_of_nodes(),
            funding_graph.number_of_edges(),
        )

    logger.info("[4/5] Building feature matrix")
    feature_matrix = build_feature_matrix(
        trades_df, orderbook_events=orderbook_events, funding_graph=funding_graph
    )
    logger.info("      Built features for %d wallets", len(feature_matrix))

    logger.info("[5/5] Scoring wallets")
    try:
        from detection.model_inference import RiskScorer

        scorer = RiskScorer()
        scored = scorer.score_matrix(feature_matrix)
    except (RuntimeError, ImportError) as exc:
        logger.warning("      Skipping ML scoring: %s", exc)
        logger.warning("      Falling back to Benford-only flags")
        mad_cols = [c for c in feature_matrix.columns if c.startswith("benford_mad_")]
        scored = feature_matrix[["wallet"] + mad_cols].copy()
        scored["benford_flag"] = (scored[mad_cols] > 0.015).any(axis=1)

    if "score" in scored:
        flagged = scored[scored["score"] >= config.RISK_SCORE_FLAG_THRESHOLD]

        if not args.no_persist and not args.dry_run:
            asset_pair = watched_pairs_label()
            store = RiskScoreStore()
            for _, row in scored.iterrows():
                store.upsert(
                    wallet=row["wallet"],
                    asset_pair=asset_pair,
                    risk_score={
                        "score": row["score"],
                        "benford_flag": row["benford_flag"],
                        "ml_flag": row["ml_flag"],
                        "confidence": row["confidence"],
                    },
                )
            logger.info(
                "      Persisted %d scored wallets to %s", len(scored), config.RISK_SCORE_DB_URL
            )
    else:
        flagged = scored[scored["benford_flag"]]

    logger.info("Flagged wallets (%d):\n%s", len(flagged), flagged)

    if args.submit_onchain:
        if args.dry_run:
            logger.warning("      [DRY RUN] Skipping on-chain submission")
        elif "score" not in scored:
            logger.warning("      Skipping on-chain submission: no ML scores available")
        else:
            submit_flagged_onchain(flagged)


def submit_flagged_onchain(flagged: pd.DataFrame) -> None:
    """Submit each flagged wallet's `RiskScore` to the `ledgerlens-score` contract."""
    from integrations.contract_client import LedgerLensContractClient

    client = LedgerLensContractClient()
    asset_pair = watched_pairs_label()
    timestamp = int(datetime.now(UTC).timestamp())

    for _, row in flagged.iterrows():
        risk_score = {
            "score": int(row["score"]),
            "benford_flag": bool(row["benford_flag"]),
            "ml_flag": bool(row["ml_flag"]),
            "timestamp": timestamp,
            "confidence": int(row["confidence"]),
        }
        client.submit_score(wallet=row["wallet"], asset_pair=asset_pair, risk_score=risk_score)

    logger.info("      Submitted %d RiskScores on-chain", len(flagged))


if __name__ == "__main__":
    main()
