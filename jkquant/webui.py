from __future__ import annotations

import json
import html
import math
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from jkquant.config import load_config
from jkquant.data.akshare_provider import AkshareMetadataProvider
from jkquant.pipeline import (
    RECOMMENDATION_HISTORY_START, available_selection_dates,
    recommendation_history_stats, selection_for_date,
)

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
    "return_5d": "5日收益率（%）",
    "return_20d": "20日收益率（%）",
    "close_ma20": "价格相对20日均线（%）",
    "ma5_ma20": "5日均线相对20日均线（%）",
    "volatility_20d": "20日年化波动率（%）",
    "max_drawdown_20d": "20日当前回撤（%）",
    "amount_mean_20d": "20日平均成交额（亿元）",
    "amount_ratio_5_20": "5日/20日成交额比（倍）",
}
TOPK_NAMES = {
    "rank": "排名", "trade_date": "数据日期", "ts_code": "股票代码",
    "name": "证券简称", "close": "收盘价（元）", "amount": "成交额（亿元）",
    "consecutive_top20": "连续Top-20（天）", "top50_count": "累计Top-50（次）",
    **SCORE_NAMES, **FACTOR_NAMES,
}
PERCENT_FACTORS = {
    "return_5d", "return_20d", "close_ma20", "ma5_ma20",
    "volatility_20d", "max_drawdown_20d",
}
FACTOR_EXPLANATIONS = {
    "return_5d": "最近5个交易日累计涨跌幅；越高代表短期动量越强，但过高也可能有回撤风险。",
    "return_20d": "最近20个交易日累计涨跌幅；越高代表中短期动量越强。",
    "close_ma20": "当前因子价格相对20日均线的偏离；大于0表示位于均线上方。",
    "ma5_ma20": "5日均线相对20日均线的偏离；大于0通常表示短期趋势偏强。",
    "volatility_20d": "20日日收益波动折算成年化；通常越低越稳定，20%以下较低、20%～40%中等、40%以上较高。",
    "max_drawdown_20d": "当前价格相对近20日最高点的跌幅；0%最好，数值越高表示回撤越深。",
    "amount_mean_20d": "最近20日平均成交额；越高通常越容易成交，但不代表未来收益更高。",
    "amount_ratio_5_20": "5日平均成交额÷20日平均成交额；大于1表示近期成交活跃度上升。",
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
    "cost_rate": "成本率", "codes": "持仓代码", "weights": "持仓权重",
}


def _percent(value: float) -> str:
    return f"{value:.2%}"


@st.cache_data(ttl=60)
def _selection_dates() -> list:
    return available_selection_dates(load_config(PROJECT_ROOT / "config.yaml"))


@st.cache_data(ttl=300, show_spinner=False)
def _intraday(ts_code: str, selected_date) -> pd.DataFrame:
    return AkshareMetadataProvider().intraday(ts_code, selected_date)


def _recommendation_table(
    page: pd.DataFrame, streaks: dict[str, dict[str, int]]
) -> str:
    columns = [
        "rank", "ts_code", "name", "consecutive_top20", "top50_count",
        "total_score", "momentum_score",
        "trend_score", "risk_score", "liquidity_score", "close", "amount",
    ]
    headers = "".join(f"<th>{html.escape(TOPK_NAMES[column])}</th>" for column in columns)
    rows = []
    for _, row in page.iterrows():
        code = str(row["ts_code"])
        stock_stats = streaks.get(code, {})
        streak = int(stock_stats.get("consecutive_top20", 0))
        name = html.escape(str(row.get("name", code)))
        if streak >= 2:
            name_html = (
                f'<span class="streak-name" title="连续 {streak} 个交易日进入 Top-20">'
                f"{name}</span>"
            )
        else:
            name_html = name
        values = {
            "rank": str(int(row["rank"])), "ts_code": html.escape(code), "name": name_html,
            "consecutive_top20": str(streak),
            "top50_count": str(int(stock_stats.get("top50_count", 0))),
            "total_score": f"{row['total_score']:.3f}",
            "momentum_score": f"{row['momentum_score']:.3f}",
            "trend_score": f"{row['trend_score']:.3f}",
            "risk_score": f"{row['risk_score']:.3f}",
            "liquidity_score": f"{row['liquidity_score']:.3f}",
            "close": f"{row['close']:.2f}", "amount": f"{row['amount']:,.2f}",
        }
        rows.append("<tr>" + "".join(f"<td>{values[column]}</td>" for column in columns) + "</tr>")
    return f"""
    <style>
      .topk-wrap {{overflow-x:auto; margin-bottom:0.75rem}}
      .topk-table {{border-collapse:collapse; width:100%; min-width:1250px; font-size:14px}}
      .topk-table th,.topk-table td {{border-bottom:1px solid #e5e7eb; padding:9px 10px; text-align:right; white-space:nowrap}}
      .topk-table th {{background:#f6f8fa; color:#374151}}
      .topk-table th:nth-child(2),.topk-table th:nth-child(3),
      .topk-table td:nth-child(2),.topk-table td:nth-child(3) {{text-align:left}}
      .streak-name {{color:#e02020; font-weight:700; cursor:help}}
    </style>
    <div class="topk-wrap"><table class="topk-table"><thead><tr>{headers}</tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
    """


def render_topk() -> None:
    st.title("每日候选（Top-K）")
    st.caption("120 积分模式：使用 Tushare 日线量价选股，当前证券简称由 AKShare 补充。")
    config = load_config(PROJECT_ROOT / "config.yaml")
    configured_top_k = int(config["strategy"]["top_k"])
    try:
        default_top_k = int(os.getenv("JKQUANT_TOP_K", configured_top_k))
    except ValueError:
        default_top_k = configured_top_k
    default_top_k = min(max(default_top_k, 1), 50)
    top_k = int(st.number_input(
        "推荐股票数量 K", min_value=1, max_value=50,
        value=default_top_k, step=1,
        help="修改后会按同一套策略重新取得 Top-K；不同 K 的结果分别缓存。",
    ))
    config["strategy"]["top_k"] = top_k
    dates = _selection_dates()
    if not dates:
        st.info("尚无本地历史行情。请先运行：python scripts/update_data.py")
        return
    selected_date = st.selectbox("选择推荐日期", list(reversed(dates)), format_func=str)
    calculation_status = st.status("正在准备选股计算…", expanded=True)
    progress = st.progress(10, text="读取日线数据和候选缓存")
    try:
        frame, calculation = selection_for_date(
            config, selected_date
        )
        progress.progress(60, text="读取历史 Top-50 入选记录")
        history_stats, history_coverage = recommendation_history_stats(config, selected_date)
        progress.progress(100, text="候选、连续上榜统计和页面数据已准备完成")
        calculation_status.update(label="选股计算完成", state="complete", expanded=False)
    except Exception:
        calculation_status.update(label="选股计算失败", state="error", expanded=True)
        raise
    source = "SQLite 缓存" if calculation["cached"] else "首次计算并写入 SQLite"
    st.caption(
        f"结果来源：{source}｜候选耗时：{calculation['elapsed_seconds']:.3f} 秒｜"
        f"历史 Top-50 缓存：{history_coverage['cached_days']}/{history_coverage['expected_days']} 个交易日"
    )
    score_columns = [column for column in frame if column.endswith("_score")]
    first, second, third, fourth = st.columns(4)
    first.metric("候选数量", len(frame))
    second.metric("平均总分", f"{frame['total_score'].mean():.3f}")
    third.metric("平均成交额", f"{frame['amount'].mean() / 100_000:,.2f} 亿元")
    fourth.metric("数据日期", str(frame["trade_date"].iloc[0]))

    display = frame.copy()
    display["ts_code"] = display["ts_code"].astype(str)
    display["amount"] = display["amount"] / 100_000
    display["amount_mean_20d"] = display["amount_mean_20d"] / 100_000
    for column in PERCENT_FACTORS:
        display[column] = display[column] * 100
    display["consecutive_top20"] = display["ts_code"].map(
        lambda code: history_stats.get(code, {}).get("consecutive_top20", 0)
    )
    display["top50_count"] = display["ts_code"].map(
        lambda code: history_stats.get(code, {}).get("top50_count", 0)
    )
    sort_options = [
        "rank", "ts_code", "name", "consecutive_top20", "top50_count",
        "total_score", "momentum_score", "trend_score", "risk_score",
        "liquidity_score", "close", "amount",
    ]
    sort_first, sort_second = st.columns([2, 1])
    sort_column = sort_first.selectbox(
        "排序字段", sort_options, format_func=lambda value: TOPK_NAMES.get(value, value),
    )
    descending = sort_second.toggle("倒序排列", value=sort_column != "rank")
    display = display.sort_values(
        sort_column, ascending=not descending, kind="stable", na_position="last"
    )
    page_size = 10
    page_count = max(1, math.ceil(len(display) / page_size))
    page_number = 1
    if page_count > 1:
        page_number = int(st.number_input(
            "页码", min_value=1, max_value=page_count, value=1, step=1,
            help=f"共 {page_count} 页，每页最多 {page_size} 只股票。",
        ))
    page_start = (page_number - 1) * page_size
    page = display.iloc[page_start:page_start + page_size]
    st.caption(f"第 {page_number}/{page_count} 页｜第 {page_start + 1}–{page_start + len(page)} 名")
    st.markdown(_recommendation_table(page, history_stats), unsafe_allow_html=True)
    st.caption(
        f"统计从 {RECOMMENDATION_HISTORY_START} 开始。红色简称表示连续至少 2 个交易日进入 Top-20；"
        "连续天数和累计进入 Top-50 次数已直接列为字段，也可悬停红色名称查看。"
    )
    choices = dict(zip(display["ts_code"], display["name"], strict=False))
    code = st.selectbox(
        "查看单只股票的因子得分", display["ts_code"].tolist(),
        format_func=lambda value: f"{value}｜{choices.get(value, value)}",
    )
    row = frame.loc[frame["ts_code"].astype(str).eq(code)].iloc[0]
    st.subheader("日内分时图")
    if selected_date not in set(dates[-5:]):
        st.info("免费 1 分钟数据仅覆盖最近 5 个交易日，请选择较近日期查看。")
    elif st.button("加载该股票的 1 分钟分时图", type="secondary"):
        try:
            with st.spinner("正在从 AKShare 获取分钟行情…"):
                minute = _intraday(code, selected_date)
            if minute.empty:
                st.warning("该日期没有取得分钟行情，可能是数据源尚未更新或接口临时不可用。")
            else:
                price_columns = ["close"] + (["average"] if "average" in minute and minute["average"].notna().any() else [])
                price = minute[["datetime", *price_columns]].rename(
                    columns={"close": "价格", "average": "均价"}
                ).melt("datetime", var_name="曲线", value_name="价格（元）")
                price_chart = alt.Chart(price).mark_line().encode(
                    x=alt.X("datetime:T", title="时间", axis=alt.Axis(format="%H:%M")),
                    y=alt.Y("价格（元）:Q", scale=alt.Scale(zero=False)),
                    color=alt.Color("曲线:N", scale=alt.Scale(
                        domain=["价格", "均价"], range=["#d62728", "#f2a900"]
                    )),
                    tooltip=[alt.Tooltip("datetime:T", title="时间", format="%H:%M"), "曲线:N", alt.Tooltip("价格（元）:Q", format=".2f")],
                ).properties(height=300)
                volume_chart = alt.Chart(minute).mark_bar(color="#6b93c6").encode(
                    x=alt.X("datetime:T", title="时间", axis=alt.Axis(format="%H:%M")),
                    y=alt.Y("volume:Q", title="成交量（手）"),
                    tooltip=[alt.Tooltip("datetime:T", title="时间", format="%H:%M"), alt.Tooltip("volume:Q", title="成交量（手）", format=",")],
                ).properties(height=120)
                st.altair_chart(alt.vconcat(price_chart, volume_chart).resolve_scale(x="shared"), width="stretch")
                st.caption("数据源：AKShare/东方财富。1 分钟数据不复权，仅提供近期交易日；点击按钮时按需获取。")
        except Exception as exc:
            st.warning(f"分钟行情暂时获取失败，不影响日线选股结果：{exc}")
    chart = pd.DataFrame(
        {
            "指标": [SCORE_NAMES.get(column, column) for column in score_columns],
            "得分": [float(row[column]) for column in score_columns],
        }
    )
    score_chart = alt.Chart(chart).mark_bar().encode(
        x=alt.X("得分:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("指标:N", sort="-x", axis=alt.Axis(labelAngle=0, title=None)),
        tooltip=["指标:N", alt.Tooltip("得分:Q", format=".3f")],
    ).properties(height=240)
    st.altair_chart(score_chart, width="stretch")
    factor_columns = [
        column for column in display
        if column not in {
            "rank", "trade_date", "ts_code", "name", "close", "amount",
            "total_score", "consecutive_top20", "top50_count",
        }
        and not column.endswith("_score")
    ]
    def format_factor(column: str) -> str:
        value = float(row[column])
        if column in PERCENT_FACTORS:
            return f"{value:.2%}"
        if column == "amount_mean_20d":
            return f"{value / 100_000:.2f} 亿元"
        if column == "amount_ratio_5_20":
            return f"{value:.2f} 倍"
        return f"{value:.4f}"

    st.dataframe(
        pd.DataFrame({
            "因子": [FACTOR_NAMES.get(column, column) for column in factor_columns],
            "当前数值": [format_factor(column) for column in factor_columns],
            "解释与判断": [FACTOR_EXPLANATIONS.get(column, "") for column in factor_columns],
        }),
        width="stretch",
        hide_index=True,
    )
    with st.expander("综合得分怎么看？"):
        st.markdown(
            "所有得分范围都是 **0～1，越高越好**。0.80 表示大致超过当日80%的有效股票；"
            "综合得分只表示符合当前规则的程度，不是上涨概率。"
        )


def render_backtest() -> None:
    st.title("策略回测")
    st.info(
        "当前回测测的是一套明确的固定规则：每日按 8 个量价因子打分，选择得分最高的 "
        "Top-10，只做多并按排名线性分配权重（第1名最高、第10名最低），每 3 个交易日调仓；"
        "信号在 T 日收盘后生成，T+1 执行，"
        "并扣除配置中的佣金、印花税和滑点。它不是 AI 预测，也不是某只股票的预测涨幅。"
    )
    with st.expander("指标与专业名词解释", expanded=True):
        st.markdown(
            "- **累计收益**：整个回测区间从起点到终点一共赚或亏多少。\n"
            "- **年化收益**：把累计收益折算成每年复利增长率；越高越好，但必须与基准和回撤一起看。\n"
            "- **基准收益**：比较对象的同期收益；策略年化收益高于基准才说明有相对价值。\n"
            "- **全市场等权（Universe EW）**：当天每只可交易股票权重相同，EW 是 Equal Weight。它不是沪深300。\n"
            "- **超额收益**：策略累计收益减去基准累计收益；大于0较好，小于0表示跑输基准。\n"
            "- **夏普比率**：越高越好。小于0较差，0～1偏弱，1～2较好，2以上通常优秀；当前无风险利率按0计算。\n"
            "- **索提诺比率**：只把下跌波动视为风险，越高越好；可粗略参考夏普的区间，但样本越短越不稳定。\n"
            "- **信息比率**：衡量超额收益相对基准的稳定性；小于0较差，0～0.5一般，0.5～1较好，1以上很强。\n"
            "- **卡玛比率**：年化收益÷最大回撤绝对值；越高越好，小于0较差，0～1一般，1以上较好。\n"
            "- **最大回撤**：越接近0越好。10%以内较低，10%～20%中等，30%以上通常属于高回撤。\n"
            "- **年化波动率**：越低越稳定；股票策略可粗略将15%以下视为较低、15%～30%中等、30%以上较高。\n"
            "- **胜率**：越高通常越好，50%以上代表正收益日更多，但还必须结合盈亏幅度。\n"
            "- **换手率**：越低越节省成本，但过低也可能反应迟钝；需要结合超额收益判断。以上区间都是研究经验值，不是保证。"
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
    columns = st.columns(4)
    columns[0].metric("索提诺比率", f"{metrics.get('sortino_ratio', 0):.3f}")
    columns[1].metric("信息比率", f"{metrics.get('information_ratio', 0):.3f}")
    columns[2].metric("卡玛比率", f"{metrics.get('calmar_ratio', 0):.3f}")
    columns[3].metric("调仓次数", int(metrics.get("rebalance_count", 0)))

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
    st.subheader("回测中的‘收益’从哪里来")
    st.markdown(
        "回测把历史每个调仓日当作当时正在运行系统：只用该日以前的数据选股，在下一交易日按开盘价模拟换仓，"
        "之后按持仓股票的实际历史涨跌计算组合每日盈亏。**净收益**是在股票涨跌形成的毛收益上再扣除模拟交易成本；"
        "逐日复合后得到累计收益和净值。因此它衡量的是‘过去机械执行当前规则会怎样’，不是未来收益承诺。"
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
