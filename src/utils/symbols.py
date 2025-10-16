"""
Utilities for parsing trading pairs (symbol -> base/quote assets).
"""

from __future__ import annotations

from typing import Tuple


_KNOWN_QUOTES = [
    "USDT",
    "USDC",
    "BUSD",
    "USD",
    "EUR",
    "GBP",
    "AUD",
    "CAD",
    "JPY",
    "BTC",
    "ETH",
    "BNB",
    "TRY",
    "RUB",
    "BRL",
    "PAX",
    "DAI",
    "NGN",
    "IDRT",
    "BIDR",
    "UAH",
    "ZAR",
    "UST",
    "VAI",
]


def parse_symbol(symbol: str) -> Tuple[str, str]:
    """
    Split a trading symbol into base and quote assets.

    Examples
    --------
    >>> parse_symbol("BTCUSDT")
    ('BTC', 'USDT')
    >>> parse_symbol("ETHBTC")
    ('ETH', 'BTC')

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., "BTCUSDT").

    Returns
    -------
    (base, quote) : tuple[str, str]
    """
    sym = symbol.upper()
    for quote in sorted(_KNOWN_QUOTES, key=len, reverse=True):
        if sym.endswith(quote) and len(sym) > len(quote):
            return sym[:-len(quote)], quote
    # Fallback: assume last 3 chars is quote
    return sym[:-3], sym[-3:]

