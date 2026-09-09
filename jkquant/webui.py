from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
BACKTESTS_ROOT = PROJECT_ROOT / "backtests"


def _percent(value: float) -> str:
    return f"{value:.2%}"


def render_topk() -> None:
    st.title("每日 Top-K 候选")
    st.caption("120 积分模式：仅使用 Tushare 未复权日线量价数据；证券名称暂以代码显示。")
    files = sorted(REPORTS_ROOT.glob("*.csv"), reverse=True)
    if not files:
        st.info("尚无每日报告。请先运行：python scripts/run_daily.py")
        return
    selected = st.selectbox("报告日期", files, format_func=lambda path: path.stem)
    frame = pd.read_csv(selected)
    score_columns = [column for column in frame if column.endswith("_score")]
    first, second, third, fourth = st.columns(4)
    first.metric("候选数量", len(frame))
    second.metric("平均总分", f"{frame['total_score'].mean():.3f}")
    third.metric("平均成交额（千元）", f"{frame['amount'].mean():,.0f}")
    fourth.metric("数据日期", str(frame["trade_date"].iloc[0]))

    display = frame.copy()
    display["ts_code"] = display["ts_code"].astype(str)
    st.dataframe(display, width="stretch", hide_index=True, height=520)
    code = st.selectbox("查看单只股票的因子得分", display["ts_code"].tolist())
    row = display.loc[display["ts_code"].eq(code)].iloc[0]
    chart = pd.DataFrame(
        {"得分": [float(row[column]) for column in score_columns]},
        index=[column.removesuffix("_score") for column in score_columns],
    )
    st.bar_chart(chart)
    factor_columns = [
        column for column in display
        if column not in {"rank", "trade_date", "ts_code", "name", "close", "amount"}
        and not column.endswith("_score")
    ]
    st.dataframe(
        pd.DataFrame({"因子": factor_columns, "原始值": [row[column] for column in factor_columns]}),
        width="stretch",
        hide_index=True,
    )


def render_backtest() -> None:
    st.title("策略回测")
    folders = sorted(
        {path.parent for path in BACKTESTS_ROOT.glob("*/metrics.json")}, reverse=True
    )
    if not folders:
        st.info("尚无回测结果。请先运行：python scripts/run_backtest.py")
        return
    folder = st.selectbox("回测区间", folders, format_func=lambda path: path.name)
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(folder / "daily.csv", parse_dates=["trade_date", "signal_date"])
    trades_path = folder / "rebalances.csv"

    columns = st.columns(4)
    columns[0].metric("累计收益", _percent(metrics["cumulative_return"]))
    columns[1].metric("年化收益", _percent(metrics["annualized_return"]))
    columns[2].metric("基准收益", _percent(metrics["benchmark_return"]))
    columns[3].metric("超额收益", _percent(metrics["excess_return"]))
    columns = st.columns(4)
    columns[0].metric("Sharpe", f"{metrics['sharpe_ratio']:.3f}")
    columns[1].metric("最大回撤", _percent(metrics["max_drawdown"]))
    columns[2].metric("年化波动", _percent(metrics["annualized_volatility"]))
    columns[3].metric("胜率", _percent(metrics["win_rate"]))

    st.subheader("净值曲线")
    equity = daily.set_index("trade_date")[["equity", "benchmark_equity"]]
    equity.columns = ["策略", "全市场等权基准"]
    st.line_chart(equity)
    st.subheader("回撤")
    st.area_chart(daily.set_index("trade_date")[["drawdown"]])
    st.subheader("年度收益")
    yearly = pd.DataFrame.from_dict(metrics["yearly_returns"], orient="index", columns=["收益"])
    st.dataframe(yearly.style.format("{:.2%}"), width="stretch")
    st.subheader("逐日结果")
    st.dataframe(daily, width="stretch", hide_index=True, height=380)
    if trades_path.exists():
        st.subheader("调仓记录")
        st.dataframe(pd.read_csv(trades_path), width="stretch", hide_index=True)
    plot_path = folder / "equity_drawdown.png"
    if plot_path.exists():
        with st.expander("查看静态净值/回撤图"):
            st.image(str(plot_path), width="stretch")


def main() -> None:
    st.set_page_config(page_title="JKQuant", page_icon="📈", layout="wide")
    st.sidebar.title("JKQuant")
    page = st.sidebar.radio("页面", ["Top-K 候选", "回测指标"])
    st.sidebar.divider()
    st.sidebar.caption("本地只读展示界面，不执行自动交易。")
    st.sidebar.warning("120 积分数据不含可靠的证券名称、历史 ST 状态和复权因子。")
    if st.sidebar.button("刷新页面"):
        st.rerun()
    if page == "Top-K 候选":
        render_topk()
    else:
        render_backtest()


if __name__ == "__main__":
    main()
