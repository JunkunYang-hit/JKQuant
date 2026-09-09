from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
BACKTESTS_ROOT = PROJECT_ROOT / "backtests"

SCORE_NAMES = {
    "total_score": "综合得分",
    "momentum_score": "动量得分",
    "trend_score": "趋势得分",
    "risk_score": "低风险得分",
    "liquidity_score": "流动性得分",
}
FACTOR_NAMES = {
    "return_5d": "5日收益率",
    "return_20d": "20日收益率",
    "close_ma20": "价格相对20日均线",
    "ma5_ma20": "5日均线相对20日均线",
    "volatility_20d": "20日年化波动率",
    "max_drawdown_20d": "20日当前回撤",
    "amount_mean_20d": "20日平均成交额",
    "amount_ratio_5_20": "5日/20日成交额比",
}
TOPK_NAMES = {
    "rank": "排名", "trade_date": "数据日期", "ts_code": "股票代码",
    "name": "证券简称", "close": "收盘价", "amount": "成交额（千元）",
    **SCORE_NAMES, **FACTOR_NAMES,
}
DAILY_NAMES = {
    "trade_date": "交易日期", "signal_date": "信号日期",
    "gross_return": "策略毛收益", "net_return": "策略净收益",
    "benchmark_return": "基准收益", "turnover": "换手率",
    "transaction_cost": "交易成本", "holdings": "持仓数量",
    "rebalanced": "是否调仓", "equity": "策略净值",
    "equity_value": "账户权益", "benchmark_equity": "基准净值",
    "drawdown": "回撤",
}
TRADE_NAMES = {
    "trade_date": "交易日期", "signal_date": "信号日期", "holdings": "持仓数量",
    "buy_turnover": "买入换手", "sell_turnover": "卖出换手",
    "cost_rate": "成本率", "codes": "持仓代码",
}


def _percent(value: float) -> str:
    return f"{value:.2%}"


def render_topk() -> None:
    st.title("每日候选（Top-K）")
    st.caption("120 积分模式：使用 Tushare 日线量价选股，当前证券简称由 AKShare 补充。")
    with st.expander("这里的量价策略是什么意思？", expanded=False):
        st.markdown(
            "量价策略只研究价格与成交额，不使用财务报表或新闻。当前策略偏好近期收益和趋势较强、"
            "波动与回撤较低、成交较活跃的股票。各因子先在当日全市场转成 0～1 百分位得分，再按"
            "动量 40%、趋势 25%、低风险 20%、流动性 15% 合成综合得分。Top-K 是值得继续人工研究的候选，"
            "不是保证上涨或直接买入指令。"
        )
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
    st.dataframe(display.rename(columns=TOPK_NAMES), width="stretch", hide_index=True, height=520)
    choices = dict(zip(display["ts_code"], display["name"], strict=False))
    code = st.selectbox(
        "查看单只股票的因子得分", display["ts_code"].tolist(),
        format_func=lambda value: f"{value}｜{choices.get(value, value)}",
    )
    row = display.loc[display["ts_code"].eq(code)].iloc[0]
    chart = pd.DataFrame(
        {"得分": [float(row[column]) for column in score_columns]},
        index=[SCORE_NAMES.get(column, column) for column in score_columns],
    )
    st.bar_chart(chart)
    factor_columns = [
        column for column in display
        if column not in {"rank", "trade_date", "ts_code", "name", "close", "amount", "total_score"}
        and not column.endswith("_score")
    ]
    st.dataframe(
        pd.DataFrame({
            "因子": [FACTOR_NAMES.get(column, column) for column in factor_columns],
            "原始值": [row[column] for column in factor_columns],
        }),
        width="stretch",
        hide_index=True,
    )


def render_backtest() -> None:
    st.title("策略回测")
    with st.expander("指标与专业名词解释", expanded=True):
        st.markdown(
            "- **累计收益**：整个回测区间从起点到终点一共赚或亏多少。\n"
            "- **年化收益**：把累计收益折算成每年复利增长率，便于比较不同长度的回测。\n"
            "- **基准收益**：比较对象的同期收益。本系统当前基准是全市场等权组合。\n"
            "- **全市场等权（Universe EW）**：当天每只可交易股票权重相同，EW 是 Equal Weight。它不是沪深300。\n"
            "- **超额收益**：策略累计收益减去基准累计收益。\n"
            "- **夏普比率（Sharpe）**：单位波动承担获得的平均收益；当前按无风险利率为 0 计算。越高通常越好，负数表示风险没有换来正收益。\n"
            "- **最大回撤**：净值从历史高点到之后最低点的最大跌幅，用于衡量最难承受的亏损阶段。\n"
            "- **年化波动率**：日收益波动折算到一年，越高表示净值起伏越大。\n"
            "- **胜率**：净收益为正的交易日占比，不等于每次选股成功率。\n"
            "- **换手率**：调仓时买卖权重变化的总量；越高通常交易成本越大。"
        )
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
    columns[2].metric("全市场等权基准收益", _percent(metrics["benchmark_return"]))
    columns[3].metric("超额收益", _percent(metrics["excess_return"]))
    columns = st.columns(4)
    columns[0].metric("夏普比率", f"{metrics['sharpe_ratio']:.3f}")
    columns[1].metric("最大回撤", _percent(metrics["max_drawdown"]))
    columns[2].metric("年化波动", _percent(metrics["annualized_volatility"]))
    columns[3].metric("胜率", _percent(metrics["win_rate"]))

    st.subheader("净值曲线")
    equity = daily.set_index("trade_date")[["equity", "benchmark_equity"]]
    equity.columns = ["策略", "全市场等权基准"]
    st.line_chart(equity)
    st.subheader("回撤")
    drawdown = daily.set_index("trade_date")[["drawdown"]].rename(columns={"drawdown": "策略回撤"})
    st.area_chart(drawdown)
    st.subheader("年度收益")
    yearly = pd.DataFrame.from_dict(metrics["yearly_returns"], orient="index", columns=["收益"])
    st.dataframe(yearly.style.format("{:.2%}"), width="stretch")
    st.subheader("逐日结果")
    st.dataframe(daily.rename(columns=DAILY_NAMES), width="stretch", hide_index=True, height=380)
    if trades_path.exists():
        st.subheader("调仓记录")
        trades = pd.read_csv(trades_path).rename(columns=TRADE_NAMES)
        st.dataframe(trades, width="stretch", hide_index=True)


def render_system_help() -> None:
    st.title("系统说明")
    st.subheader("系统现在如何运行")
    st.markdown(
        "1. **更新数据**：按交易日从 Tushare 获取全市场日线，按月写入本地 Parquet。\n"
        "2. **补充名称**：每 30 天通过 AKShare 更新一次当前 A 股代码和证券简称。\n"
        "3. **计算因子**：每只股票只使用当日及之前的数据计算 8 个量价因子。\n"
        "4. **过滤股票**：排除当前 ST、成交额不足、零成交和历史记录太短的股票。\n"
        "5. **横截面打分**：把当日每个因子转成 0～1 的市场百分位得分。\n"
        "6. **生成候选**：按综合得分从高到低输出 Top-K，供人工继续研究。\n"
        "7. **历史回测**：使用前一交易日信号，在下一交易日执行，计入佣金、印花税和滑点。"
    )
    st.subheader("当前量价策略")
    st.markdown(
        "- **动量 40%**：5日、20日收益较强得分更高。\n"
        "- **趋势 25%**：价格高于20日均线、5日均线高于20日均线得分更高。\n"
        "- **低风险 20%**：20日波动和回撤越小得分越高。\n"
        "- **流动性 15%**：平均成交额和近期成交活跃度较高得分更高。\n\n"
        "这是一套固定规则，并不是 AI 预测模型。真实回测目前显著跑输基准，因此 Top-K 只能作为研究清单。"
    )
    st.subheader("仍需补齐的关键模块")
    st.markdown(
        "- 因子 IC、分层收益、相关性和稳定性诊断。\n"
        "- 涨跌停、停牌延续、100股整数手和最低佣金等真实成交约束。\n"
        "- 历史 ST、退市、名称和指数成分的时点数据。\n"
        "- 完整复权价格和沪深300/中证500等真实指数基准。\n"
        "- 行业、市值暴露约束，以及自定义/指数股票池。\n"
        "- 升级数据权限后的估值、质量和成长因子。"
    )
    st.warning("本系统输出的是量化候选，不构成投资建议，也不会自动下单。")


def main() -> None:
    st.set_page_config(page_title="JKQuant", page_icon="📈", layout="wide")
    st.sidebar.title("JKQuant")
    page = st.sidebar.radio("页面", ["每日候选", "回测指标", "系统说明"])
    st.sidebar.divider()
    st.sidebar.caption("本地只读展示界面，不执行自动交易。")
    st.sidebar.warning("AKShare 可补充当前证券简称，但 120 积分仍不含可靠的历史 ST 状态、行业和完整复权因子。")
    if st.sidebar.button("刷新页面"):
        st.rerun()
    if page == "每日候选":
        render_topk()
    elif page == "回测指标":
        render_backtest()
    else:
        render_system_help()


if __name__ == "__main__":
    main()
