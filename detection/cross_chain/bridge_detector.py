"""Bridge transaction detector.

Identifies Stellar <-> Ethereum/Solana bridge transfers from transaction lists
by detecting Ethereum/Solana addresses inside SEP-0006 memos.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# EVM address regex: 0x followed by 40 hex chars
_EVM_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_EVM_NO_PREFIX_RE = re.compile(r"^[a-fA-F0-9]{40}$")

# Solana address regex: base58 string of length 32 to 44
_SOL_ADDR_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


def bytes_to_base58(b: bytes) -> str:
    """Encode bytes to base58 string."""
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    n = int.from_bytes(b, "big")
    res = []
    while n > 0:
        n, r = divmod(n, 58)
        res.append(alphabet[r])
    for x in b:
        if x == 0:
            res.append(alphabet[0])
        else:
            break
    return "".join(reversed(res))


class BridgeDetector:
    """Detects cross-chain identities via Stellar bridge transaction memos."""

    def __init__(self, anchor_addresses: list[str] | None = None):
        self.anchor_addresses = set(anchor_addresses or [])

    def parse_memo_address(self, memo_type: str, memo_val: Any) -> tuple[str, str] | None:
        """Parse an Ethereum or Solana address from a memo.

        Returns (address, chain) if parsed, otherwise None.
        """
        if not memo_val:
            return None

        memo_type = memo_type.lower()

        if memo_type == "text":
            # Memo is a text string
            memo_str = str(memo_val).strip()
            # Check EVM address
            if _EVM_ADDR_RE.match(memo_str):
                return memo_str.lower(), "ethereum"
            if _EVM_NO_PREFIX_RE.match(memo_str):
                return f"0x{memo_str}".lower(), "ethereum"
            # Check Solana address
            if _SOL_ADDR_RE.match(memo_str):
                return memo_str, "solana"

        elif memo_type in ("hash", "return"):
            # Memo is 32-byte hash. Might be represented as a base64 string or hex string.
            raw_bytes = None
            if isinstance(memo_val, bytes):
                raw_bytes = memo_val
            elif isinstance(memo_val, str):
                memo_str = memo_val.strip()
                # Try base64
                try:
                    b = base64.b64decode(memo_str)
                    if len(b) == 32:
                        raw_bytes = b
                except Exception:
                    pass
                # Try hex if base64 failed or didn't produce 32 bytes
                if not raw_bytes:
                    try:
                        b = bytes.fromhex(memo_str)
                        if len(b) == 32:
                            raw_bytes = b
                    except Exception:
                        pass

            if not raw_bytes or len(raw_bytes) != 32:
                return None

            # 1. Check for padded EVM address (20 bytes)
            # Left padded (12 bytes of zeros followed by 20 bytes EVM)
            if raw_bytes[:12] == b"\x00" * 12:
                evm_addr = "0x" + raw_bytes[12:].hex()
                return evm_addr.lower(), "ethereum"
            # Right padded (20 bytes EVM followed by 12 bytes of zeros)
            if raw_bytes[20:] == b"\x00" * 12:
                evm_addr = "0x" + raw_bytes[:20].hex()
                return evm_addr.lower(), "ethereum"

            # 2. Treat the raw 32 bytes as a Solana public key
            sol_addr = bytes_to_base58(raw_bytes)
            # Verify it is a valid base58 Solana address format
            if _SOL_ADDR_RE.match(sol_addr):
                return sol_addr, "solana"

        return None

    def detect_bridge_links(self, transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify bridge links from a list of transaction records.

        Each transaction record should look like:
        {
            "id": "...",
            "source_account": "...",
            "memo_type": "...",
            "memo": "...",
            "from": "...",  # optional
            "to": "...",    # optional
        }
        """
        links = []
        for tx in transactions:
            memo_type = tx.get("memo_type")
            memo = tx.get("memo")
            if not memo_type or not memo:
                continue

            parsed = self.parse_memo_address(memo_type, memo)
            if not parsed:
                continue

            linked_addr, chain = parsed
            tx_id = tx.get("id") or tx.get("hash", "")

            # Determine the Stellar user address
            # Default to transaction fee payer/source_account
            stellar_addr = tx.get("source_account")

            # If there's an explicit payment source or destination, we can refine:
            tx_from = tx.get("from")
            tx_to = tx.get("to")

            if tx_from and tx_to:
                # If we know the anchor address, the user is the opposite party
                if tx_from in self.anchor_addresses:
                    stellar_addr = tx_to
                elif tx_to in self.anchor_addresses:
                    stellar_addr = tx_from
                else:
                    # Otherwise, use the source_account, or fall back to 'from'
                    stellar_addr = stellar_addr or tx_from

            if stellar_addr:
                links.append(
                    {
                        "stellar_address": stellar_addr,
                        "linked_address": linked_addr,
                        "chain": chain,
                        "tx_id": tx_id,
                        "memo": str(memo),
                        "confidence": 1.0,
                    }
                )

        return links
