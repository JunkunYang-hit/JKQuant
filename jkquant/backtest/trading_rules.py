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


def transaction_fee(
    notional: float, commission_rate: float, *, min_commission: float = 0.0,
    stamp_tax_rate: float = 0.0, slippage_rate: float = 0.0,
) -> float:
    """Calculate the monetary cost of one order, including its commission floor."""
    if notional <= 0:
        return 0.0
    commission = max(notional * commission_rate, min_commission)
    return commission + notional * (stamp_tax_rate + slippage_rate)


def affordable_buy_notional(
    cash_budget: float, commission_rate: float, *, min_commission: float = 0.0,
    slippage_rate: float = 0.0,
) -> float:
    """Return the largest buy notional whose notional and fees fit in the budget."""
    if cash_budget <= min_commission:
        return 0.0
    proportional = cash_budget / (1 + commission_rate + slippage_rate)
    if proportional * commission_rate >= min_commission:
        return proportional
    return max(0.0, (cash_budget - min_commission) / (1 + slippage_rate))
