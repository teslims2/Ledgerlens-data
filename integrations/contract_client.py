"""Client for the `ledgerlens-score` Soroban contract.

Wraps `stellar_sdk.contract.ContractClient` to call the two functions
documented in the README's "Shared Contracts" section:

  - `submit_score(wallet, asset_pair, score, benford_flag, ml_flag,
    timestamp, confidence)` — writes a `RiskScore` record on-chain. Requires
    `LEDGERLENS_SUBMITTER_SECRET` (an authorized service-account secret key).
  - `get_score(wallet, asset_pair)` — permissionless read of the on-chain
    `RiskScore`.

`wallet` is a Stellar account ID (`G...`), `asset_pair` is the
`CODE:ISSUER/CODE:ISSUER` string from `ingestion.data_models.Asset.pair_id`.
"""

from stellar_sdk import Keypair, Network, scval
from stellar_sdk.contract import ContractClient

from config import config

_NETWORK_PASSPHRASES = {
    "PUBLIC": Network.PUBLIC_NETWORK_PASSPHRASE,
    "TESTNET": Network.TESTNET_NETWORK_PASSPHRASE,
}


class LedgerLensContractClient:
    """Thin wrapper around the `ledgerlens-score` contract's invocations."""

    def __init__(
        self,
        contract_id: str | None = None,
        rpc_url: str | None = None,
        network_passphrase: str | None = None,
        submitter_secret: str | None = None,
    ):
        self.contract_id = contract_id or config.LEDGERLENS_CONTRACT_ID
        if not self.contract_id:
            raise ValueError("LEDGERLENS_CONTRACT_ID is not configured")

        self.rpc_url = rpc_url or config.SOROBAN_RPC_URL
        self.network_passphrase = network_passphrase or _NETWORK_PASSPHRASES.get(
            config.STELLAR_NETWORK, Network.TESTNET_NETWORK_PASSPHRASE
        )
        self.submitter_secret = submitter_secret or config.LEDGERLENS_SUBMITTER_SECRET

        self._client = ContractClient(
            contract_id=self.contract_id,
            rpc_url=self.rpc_url,
            network_passphrase=self.network_passphrase,
        )

    def submit_score(self, wallet: str, asset_pair: str, risk_score: dict) -> object:
        """Submit a `RiskScore` record (the dict shape from `RiskScorer.score()`,
        plus an integer `timestamp`) for `(wallet, asset_pair)`.

        Returns the parsed contract result. Requires `LEDGERLENS_SUBMITTER_SECRET`.
        """
        if not self.submitter_secret:
            raise ValueError("LEDGERLENS_SUBMITTER_SECRET is not configured")

        signer = Keypair.from_secret(self.submitter_secret)

        params = [
            scval.to_address(wallet),
            scval.to_string(asset_pair),
            scval.to_uint32(int(risk_score["score"])),
            scval.to_bool(bool(risk_score["benford_flag"])),
            scval.to_bool(bool(risk_score["ml_flag"])),
            scval.to_uint64(int(risk_score["timestamp"])),
            scval.to_uint32(int(risk_score["confidence"])),
        ]

        tx = self._client.invoke(
            "submit_score",
            params,
            source=signer.public_key,
            signer=signer,
        )
        return tx.sign_and_submit()

    def get_score(self, wallet: str, asset_pair: str) -> dict:
        """Read the on-chain `RiskScore` for `(wallet, asset_pair)`."""
        params = [scval.to_address(wallet), scval.to_string(asset_pair)]
        tx = self._client.invoke("get_score", params, simulate=True)
        return scval.to_native(tx.result())
