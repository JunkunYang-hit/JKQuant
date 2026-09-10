from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def price_limit_rate(ts_code: str) -> float:
    """Return the normal daily price-limit rate for a non-ST A-share."""
    code, _, exchange = ts_code.partition(".")
    if exchange == "BJ":
        return 0.30
    if exchange == "SH" and code.startswith(("688", "689")):
        return 0.20
    if exchange == "SZ" and code.startswith(("300", "301")):
        return 0.20
    return 0.10


def limit_price(pre_close: float, rate: float, direction: int) -> float:
    raw = Decimal(str(pre_close)) * (Decimal("1") + Decimal(direction) * Decimal(str(rate)))
    return float(raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def is_open_limit_up(
    ts_code: str, open_price: float, pre_close: float, exact_limit: float | None = None,
) -> bool:
    threshold = exact_limit if exact_limit is not None and exact_limit > 0 else limit_price(pre_close, price_limit_rate(ts_code), 1)
    return open_price > 0 and pre_close > 0 and open_price >= threshold - 0.005


def is_open_limit_down(
    ts_code: str, open_price: float, pre_close: float, exact_limit: float | None = None,
) -> bool:
    threshold = exact_limit if exact_limit is not None and exact_limit > 0 else limit_price(pre_close, price_limit_rate(ts_code), -1)
    return open_price > 0 and pre_close > 0 and open_price <= threshold + 0.005
