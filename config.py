"""Central configuration loaded from environment variables / .env."""

import os

from dotenv import load_dotenv

load_dotenv()


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    pairs = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        code, _, issuer = entry.partition(":")
        pairs.append((code, issuer or "native"))
    return pairs


def _parse_int_list(raw: str) -> list[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


class Config:
    HORIZON_URL: str = os.getenv("HORIZON_URL", "https://horizon.stellar.org")
    STELLAR_NETWORK: str = os.getenv("STELLAR_NETWORK", "PUBLIC")

    WATCHED_ASSET_PAIRS: list[tuple[str, str]] = _parse_pairs(os.getenv("WATCHED_ASSET_PAIRS", ""))

    BENFORD_WINDOWS_HOURS: list[int] = _parse_int_list(
        os.getenv("BENFORD_WINDOWS_HOURS", "1,4,24,168,720")
    )

    CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS: int = int(
        os.getenv("CROSS_PAIR_SYNCHRONY_WINDOW_SECONDS", "30")
    )

    RISK_SCORE_FLAG_THRESHOLD: int = int(os.getenv("RISK_SCORE_FLAG_THRESHOLD", "70"))

    RISK_SCORE_DB_URL: str = os.getenv("RISK_SCORE_DB_URL", "sqlite:///ledgerlens.db")

    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")

    # ledgerlens-score Soroban contract
    SOROBAN_RPC_URL: str = os.getenv("SOROBAN_RPC_URL", "https://soroban-testnet.stellar.org")
    LEDGERLENS_CONTRACT_ID: str = os.getenv("LEDGERLENS_CONTRACT_ID", "")
    LEDGERLENS_SUBMITTER_SECRET: str = os.getenv("LEDGERLENS_SUBMITTER_SECRET", "")

    MIN_TRADES_FOR_SCORING: int = int(os.getenv("MIN_TRADES_FOR_SCORING", "20"))

    # Wallet funding graph — multi-hop traversal + wash-trading ring detection
    WALLET_GRAPH_MAX_DEPTH: int = int(os.getenv("WALLET_GRAPH_MAX_DEPTH", "4"))
    WASH_RING_MIN_SIZE: int = int(os.getenv("WASH_RING_MIN_SIZE", "3"))
    WASH_RING_RESOLUTION: float = float(os.getenv("WASH_RING_RESOLUTION", "1.0"))
    # Fixed seed keeps Louvain community detection deterministic in CI.
    WASH_RING_LOUVAIN_SEED: int = int(os.getenv("WASH_RING_LOUVAIN_SEED", "42"))

    # Real-time streaming / alerting
    ALERT_CHANNEL: str = os.getenv("ALERT_CHANNEL", "stdout")
    ALERT_WEBHOOK_URL: str | None = os.getenv("ALERT_WEBHOOK_URL")
    ALERT_COOLDOWN_SECONDS: int = int(os.getenv("ALERT_COOLDOWN_SECONDS", "3600"))
    WS_PORT: int = int(os.getenv("WS_PORT", "8765"))
    WS_BIND_HOST: str = os.getenv("WS_BIND_HOST", "127.0.0.1")
    WS_ALLOW_EXTERNAL: bool = os.getenv("WS_ALLOW_EXTERNAL", "") == "1"

    def validate(self, require_onchain: bool = True) -> None:
        """Raise ValueError if required config is missing."""
        if not self.WATCHED_ASSET_PAIRS:
            raise ValueError("WATCHED_ASSET_PAIRS is not configured")
        if require_onchain and not self.LEDGERLENS_CONTRACT_ID:
            raise ValueError("LEDGERLENS_CONTRACT_ID is not configured")

    # Adversarial training augmentation
    ADVERSARIAL_AUG_RATIO: float = float(os.getenv("ADVERSARIAL_AUG_RATIO", "0.0"))

    # Model integrity & BFT voting
    MODEL_SIGNING_PRIVATE_KEY_PATH: str = os.getenv("MODEL_SIGNING_PRIVATE_KEY_PATH", "")
    TRUSTED_SIGNING_KEY_FINGERPRINT: str = os.getenv("TRUSTED_SIGNING_KEY_FINGERPRINT", "")
    BFT_SCORE_DIVERGENCE_THRESHOLD: int = int(os.getenv("BFT_SCORE_DIVERGENCE_THRESHOLD", "30"))
    BFT_MIN_CONSENSUS: int = int(os.getenv("BFT_MIN_CONSENSUS", "2"))
    POISON_LABEL_RATIO_THRESHOLD: float = float(os.getenv("POISON_LABEL_RATIO_THRESHOLD", "0.15"))

    # Annotation integrity
    ANNOTATION_HMAC_SECRET: str = os.getenv("ANNOTATION_HMAC_SECRET", "")

    # Active learning
    AL_QUERY_STRATEGY: str = os.getenv("AL_QUERY_STRATEGY", "committee_disagreement")
    AL_BATCH_SIZE: int = int(os.getenv("AL_BATCH_SIZE", "20"))
    AL_RETRAIN_THRESHOLD: int = int(os.getenv("AL_RETRAIN_THRESHOLD", "50"))
    AL_ROLLBACK_AUC_DROP: float = float(os.getenv("AL_ROLLBACK_AUC_DROP", "0.01"))
    AL_QUEUE_PATH: str = os.getenv("AL_QUEUE_PATH", "data/annotation_queue.json")


config = Config()
