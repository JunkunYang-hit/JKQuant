from __future__ import annotations

import logging
from datetime import date, timedelta

from .provider import DataProvider
from .storage import ParquetStore

LOGGER = logging.getLogger(__name__)


def update_data(provider: DataProvider, store: ParquetStore, end_date: date, history_days: int) -> None:
    old = store.load_daily()
    if old.empty:
        start_date = end_date - timedelta(days=max(history_days * 2, 365))
    else:
        start_date = old["trade_date"].max().date() + timedelta(days=1)
    if start_date <= end_date:
        LOGGER.info("下载日线数据: %s 至 %s", start_date, end_date)
        store.save_daily(provider.daily(start_date, end_date))
    else:
        LOGGER.info("本地日线数据已是最新")
    store.save_basic(provider.stock_basic())

