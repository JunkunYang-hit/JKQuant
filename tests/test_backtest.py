from datetime import date

import pandas as pd
import yaml

from jkquant.backtest.engine import run_backtest
from jkquant.backtest.reporting import write_backtest_report
from jkquant.data.demo_provider import DemoProvider


def test_backtest_uses_lagged_signals_and_charges_costs(tmp_path) -> None:
    config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    config["data"]["provider"] = "demo"
    config["market"]["min_amount"] = 0
    config["backtest"]["top_k"] = 5
    provider = DemoProvider(stock_count=20, seed=11)
    daily = provider.daily(date(2024, 9, 1), date(2025, 6, 30))
    result = run_backtest(daily, provider.stock_basic(), config, date(2025, 1, 1), date(2025, 6, 30))

    assert not result.daily.empty
    assert (result.daily["signal_date"] < result.daily["trade_date"]).all()
    assert result.daily["transaction_cost"].sum() > 0
    assert (result.daily["equity_value"] > 0).all()
    assert set(result.metrics) >= {"annualized_return", "max_drawdown", "yearly_returns"}
    assert pd.to_datetime(result.trades["signal_date"]).lt(pd.to_datetime(result.trades["trade_date"])).all()
    paths = write_backtest_report(result, tmp_path / "backtest")
    assert all(path.exists() for path in paths.values())
