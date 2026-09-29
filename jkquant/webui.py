from __future__ import annotations

import html
import json
import math
import os
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from jkquant.ai import get_cached_analysis, run_ai_analysis, test_ai_connection
from jkquant.config import load_config
from jkquant.data.realtime_provider import RealtimeQuoteError, build_realtime_provider
from jkquant.holiday_risk import holiday_risk
from jkquant.limit_up import build_board_history, promotion_candidates, promotion_summary
from jkquant.low_position_pool import evaluate_realtime_alerts, screen_low_position_breakouts
from jkquant.pipeline import (
    RECOMMENDATION_HISTORY_START, available_selection_dates,
    build_store, run_daily, recommendation_history_stats, selection_for_date,
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
    "equity_value": "组合权益", "benchmark_equity": "基准净值",
    "drawdown": "回撤",
    "cash": "现金",
    "limit_up_buy_blocked": "涨停未买入（只）",
    "limit_down_sell_blocked": "跌停未卖出（只）",
}
TRADE_NAMES = {
    "trade_date": "交易日期", "signal_date": "信号日期", "holdings": "持仓数量",
    "buy_turnover": "买入换手", "sell_turnover": "卖出换手",
    "cost_rate": "成本率", "codes": "持仓代码", "weights": "持仓权重",
    "strategy_id": "策略编号", "ts_code": "股票代码", "name": "证券简称",
    "entry_date": "买入日期", "exit_date": "结束日期",
    "holding_period": "持股时间区间",
    "holding_trading_days": "持有交易日", "entry_price": "买入价",
    "exit_price": "结束价", "price_change": "价格变化", "net_return": "净收益",
    "exit_reason": "结束原因", "crossed20_date": "止盈触发日期", "status": "状态",
    "max_gain": "持有期最大盈利",
    "max_drawdown_during_holding": "持有期最大不利波动",
}
EVENT_NAMES = {
    "strategy_id": "策略编号", "trade_id": "交易编号", "event": "事件", "ts_code": "股票代码",
    "name": "证券简称", "entry_date": "买入日期", "event_date": "事件日期",
    "holding_period": "持股时间区间",
    "holding_trading_days": "已持有交易日", "entry_price": "买入价",
    "trigger_price": "触发价", "day_high": "当日最高价", "price_change": "最高涨幅",
    "final_exit_date": "最终结束日期", "final_exit_price": "最终结束价",
    "final_price_change": "最终价格变化", "final_net_return": "最终净收益",
    "final_exit_reason": "最终退出原因", "final_status": "最终状态",
}


def _percent(value: float) -> str:
    return f"{value:.2%}"


def _stock_search(
    label: str, options: list[str], names: dict[str, str], spellings: dict[str, str], key: str,
) -> str:
    """Search stocks by code, fuzzy Chinese name, or Tushare pinyin initials."""
    search_col, select_col, _ = st.columns([1.5, 2.4, 4.1])
    query = search_col.text_input(
        f"搜索{label}", key=f"{key}_query", placeholder="代码/简称/拼音首字母",
    ).strip().lower()
    filtered = options
    if query:
        filtered = [
            code for code in options
            if query in code.lower()
            or query in str(names.get(code, "")).lower()
            or query in str(spellings.get(code, "")).lower()
        ]
    if not filtered:
        search_col.warning("没有匹配股票")
        filtered = options[:1]
    return select_col.selectbox(
        label, filtered, key=f"{key}_select",
        format_func=lambda code: f"{code}｜{names.get(code, '')}",
    )


def _integerize_counts(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        label = str(column)
        if "次数" in label or label in {"交易次数", "记录数", "持仓数量"}:
            result[column] = pd.to_numeric(result[column], errors="coerce").round().astype("Int64")
    return result


def _basic_search_maps(config: dict) -> tuple[dict[str, str], dict[str, str]]:
    basic = build_store(config).load_basic().drop_duplicates("ts_code")
    names = basic.set_index("ts_code")["name"].fillna("").astype(str).to_dict()
    spellings = (
        basic.set_index("ts_code")["cnspell"].fillna("").astype(str).to_dict()
        if "cnspell" in basic else {}
    )
    return names, spellings


def _suite_folders() -> list[Path]:
    return sorted(
        {path.parent for path in (BACKTESTS_ROOT / "strategy_suite").glob("*/suite.json")},
        reverse=True,
    )


def _strategy_result_folders() -> list[Path]:
    batches = _suite_folders()
    if not batches:
        return []
    folders = sorted(
        {path.parent for path in batches[0].glob("*/metrics.json")},
        key=lambda folder: (folder / "metrics.json").stat().st_mtime, reverse=True,
    )
    return folders


def _result_label(folder: Path) -> str:
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    name = metrics.get("strategy_name", folder.name)
    return f"{name}｜{folder.parent.name}"


@st.cache_data(ttl=60)
def _selection_dates() -> list:
    return available_selection_dates(load_config(PROJECT_ROOT / "config.yaml"))


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
        if streak >= 3:
            name_html = (
                f'<span class="streak-name streak-purple" title="连续 {streak} 个交易日进入 Top-20">'
                f"{name}</span>"
            )
        elif streak == 2:
            name_html = (
                f'<span class="streak-name streak-red" title="连续 {streak} 个交易日进入 Top-20">'
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
      .streak-name {{font-weight:700; cursor:help}}
      .streak-red {{color:#e02020}}
      .streak-purple {{color:#7e22ce}}
    </style>
    <div class="topk-wrap"><table class="topk-table"><thead><tr>{headers}</tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>
    """


def render_topk() -> None:
    st.title("每日候选（Top-K）")
    recent_suites = _suite_folders()
    if recent_suites:
        recent_suite = json.loads((recent_suites[0] / "suite.json").read_text(encoding="utf-8"))
        evaluated = recent_suite.get("strategies", [])
        if evaluated and not any(float(item["metrics"].get("cumulative_return", 0)) > 0 for item in evaluated):
            st.warning(f"最新可交易股票池回测中，{len(evaluated)} 套主策略累计收益均未转正。"
                       "下方 Top-K 是量价观察名单，不是已经验证可盈利的买入指令。")
    show_holiday_hint = st.toggle("显示节假日前风险提示", value=True)
    if show_holiday_hint:
        holiday_status = holiday_risk(date.today())
        if holiday_status and holiday_status["level"] == "warning":
            st.warning(
                f"节前风险提示：距离{holiday_status['holiday_name']}假期还有 "
                f"{holiday_status['days_ahead']} 天。长假前资金可能趋于谨慎，可重点检查仓位和流动性；"
                "这只是风险提醒，系统不会自动清仓。"
            )
        elif holiday_status and holiday_status["level"] == "safe":
            st.success("安全提示：未来两天内没有中国法定节假日开始。")
    st.caption("2000 积分数据模式：证券简称来自 Tushare 股票列表，推荐仍以已验证的量价因子为主。")
    base_config = load_config(PROJECT_ROOT / "config.yaml")
    action_col, _ = st.columns([1.8, 6.2])
    if action_col.button("更新数据并计算今日推荐", type="primary", use_container_width=True):
        with st.spinner("正在更新最新数据并计算 Top-50……"):
            try:
                report_path, _ = run_daily(base_config, date.today())
                latest_date = build_store(base_config).load_daily()["trade_date"].max().date()
                st.success(f"已完成 {latest_date} 推荐：{report_path.name}")
            except Exception as exc:
                st.error(f"今日推荐计算失败：{exc}")
    dates = _selection_dates()
    if not dates:
        st.info("尚无本地历史行情。请先运行：python scripts/update_data.py")
        return
    configured_top_k = int(base_config["strategy"]["top_k"])
    try:
        default_top_k = int(os.getenv("JKQUANT_TOP_K", configured_top_k))
    except ValueError:
        default_top_k = configured_top_k
    default_top_k = min(max(default_top_k, 1), 50)
    filter_left, filter_right, _ = st.columns([1.1, 1.5, 5.4])
    top_k = int(filter_left.number_input(
        "推荐股票数量 K", min_value=1, max_value=50, value=default_top_k, step=1,
        help="数据库统一缓存 Top-50，修改 K 只截取前 K。",
    ))
    requested_date = filter_right.date_input(
        "选择推荐日期", value=date.today(), min_value=dates[0],
        max_value=max(date.today(), dates[-1]), format="YYYY-MM-DD",
    )
    config = base_config
    config["strategy"]["top_k"] = top_k
    available = [value for value in dates if value <= requested_date]
    selected_date = available[-1] if available else dates[0]
    if selected_date != requested_date:
        st.info(f"{requested_date} 不是本地交易日，已显示最近交易日 {selected_date} 的推荐。")
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
        "rank", "consecutive_top20", "top50_count", "total_score",
        "momentum_score", "trend_score", "risk_score", "liquidity_score",
        "close", "amount",
    ]
    sort_first, sort_second, _ = st.columns([1.6, 1.0, 5.4])
    sort_column = sort_first.selectbox(
        "排序字段", sort_options, format_func=lambda value: TOPK_NAMES.get(value, value),
    )
    descending = sort_second.toggle("倒序排列", value=sort_column != "rank")
    display = display.sort_values(
        sort_column, ascending=not descending, kind="stable", na_position="last"
    )
    page_size = 10
    page_count = max(1, math.ceil(len(display) / page_size))
    page_context = f"{selected_date}|{top_k}|{sort_column}|{descending}"
    if st.session_state.get("topk_page_context") != page_context:
        st.session_state["topk_page_context"] = page_context
        st.session_state["topk_page"] = 1
    page_number = min(max(int(st.session_state.get("topk_page", 1)), 1), page_count)
    st.session_state["topk_page"] = page_number
    page_start = (page_number - 1) * page_size
    page = display.iloc[page_start:page_start + page_size]
    st.markdown(_recommendation_table(page, history_stats), unsafe_allow_html=True)
    page_left, page_middle, page_right = st.columns([1, 2, 1])
    if page_left.button(
        "◀ 上一页", disabled=page_number <= 1, use_container_width=True,
    ):
        st.session_state["topk_page"] = page_number - 1
        st.rerun()
    page_middle.markdown(
        f"<div style='text-align:center;padding-top:0.45rem'>"
        f"第 {page_number} / {page_count} 页　·　第 {page_start + 1}–{page_start + len(page)} 条"
        "</div>",
        unsafe_allow_html=True,
    )
    if page_right.button(
        "下一页 ▶", disabled=page_number >= page_count, use_container_width=True,
    ):
        st.session_state["topk_page"] = page_number + 1
        st.rerun()
    st.caption(
        f"统计从 {RECOMMENDATION_HISTORY_START} 开始。连续 2 个交易日进入 Top-20 标红，连续 3 个及以上标紫；"
        "连续天数和累计进入 Top-50 次数已直接列为字段，也可悬停名称查看。"
    )
    names, spellings = _basic_search_maps(config)
    code = _stock_search(
        "查看单只股票的因子得分", display["ts_code"].tolist(), names, spellings,
        "topk_factor_stock",
    )
    row = frame.loc[frame["ts_code"].astype(str).eq(code)].iloc[0]
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


def _render_ai_result(result: dict) -> None:
    analysis = result["analysis"]
    risk = str(analysis.get("overall_risk_level", "未知"))
    if risk == "高":
        st.error(f"整体风险：{risk}")
    elif risk == "中":
        st.warning(f"整体风险：{risk}")
    else:
        st.success(f"整体风险：{risk}")
    st.subheader("市场与候选组合摘要")
    st.write(analysis.get("market_summary", "暂无摘要"))
    question_answer = analysis.get("user_question_answer", {})
    if question_answer.get("question"):
        st.subheader("AI对你的问题的回答")
        st.markdown(f"**你的问题：** {question_answer.get('question')}")
        st.write(question_answer.get("answer", "暂无回答"))
        support, limits = st.columns(2)
        with support:
            st.markdown("**回答依据**")
            for item in question_answer.get("supporting_data", []):
                st.markdown(f"- {item}")
        with limits:
            st.markdown("**回答边界**")
            for item in question_answer.get("limitations", []):
                st.markdown(f"- {item}")
    left, right = st.columns(2)
    with left:
        st.markdown("**组合观察**")
        for item in analysis.get("portfolio_observations", []):
            st.markdown(f"- {item}")
    with right:
        st.markdown("**集中度风险**")
        for item in analysis.get("concentration_risks", []):
            st.markdown(f"- {item}")

    candidates = analysis.get("candidates", [])
    if candidates:
        overview = pd.DataFrame([{
            "排名": item.get("rank"), "股票代码": item.get("ts_code"),
            "证券简称": item.get("name"), "AI关注级别": item.get("attention_level"),
            "摘要": item.get("summary"), "数据完整性": item.get("data_completeness"),
        } for item in candidates]).sort_values("排名")
        st.subheader("Top-20 AI复核结果")
        st.dataframe(overview, width="stretch", hide_index=True, height=520)
        st.caption("表格保持原始量价排名；AI关注级别不会改写选股分数或触发交易。")
        st.subheader("逐只分析")
        for item in sorted(candidates, key=lambda value: int(value.get("rank", 999))):
            label = (
                f"#{item.get('rank')} {item.get('ts_code')} {item.get('name')}｜"
                f"{item.get('attention_level', '未评级')}"
            )
            with st.expander(label):
                st.write(item.get("summary", ""))
                cols = st.columns(4)
                sections = (
                    ("积极因素", "positive_factors"), ("风险因素", "risk_factors"),
                    ("观察条件", "watch_conditions"), ("判断失效条件", "invalidation_conditions"),
                )
                for column, (heading, key) in zip(cols, sections):
                    column.markdown(f"**{heading}**")
                    for text_value in item.get(key, []):
                        column.markdown(f"- {text_value}")

    limitations = analysis.get("data_limitations", [])
    if limitations:
        with st.expander("数据边界与局限"):
            for item in limitations:
                st.markdown(f"- {item}")
    st.info(analysis.get("disclaimer", "仅供量化研究参考，不构成投资建议。"))
    usage = result.get("usage", {})
    st.caption(
        f"模型：{result.get('model', '未知')}｜提示词版本：{result.get('prompt_version', '未知')}｜"
        f"生成时间：{result.get('created_at', '未知')}｜来源：{'本地缓存' if result.get('cached') else 'DeepSeek API'}｜"
        f"Token：输入 {int(usage.get('prompt_tokens', 0) or 0)} / 输出 {int(usage.get('completion_tokens', 0) or 0)}"
    )
    with st.expander("查看本次发送给AI的结构化依据"):
        st.json(result.get("input", {}), expanded=False)


def render_ai_analysis() -> None:
    st.title("AI分析（DeepSeek）")
    st.caption(
        "使用本地Top-20、量价因子、估值、财务、宏观和沪深300ETF数据做二次复核。"
        "页面不会自动调用API，刷新不会重复产生费用。"
    )
    config = load_config(PROJECT_ROOT / "config.yaml")
    dates = _selection_dates()
    if not dates:
        st.info("尚无本地候选数据，请先在“每日候选”更新数据并计算推荐。")
        return
    model_labels = {
        "deepseek-flash": "DeepSeek V4.1 Flash（推荐，速度快）",
        "deepseek-v4-pro": "DeepSeek V4 Pro（兼容入口）",
    }
    configured_models = [
        str(value) for value in config.get("ai", {}).get(
            "models", ["deepseek-flash", "deepseek-v4-pro"],
        )
    ]
    default_model = str(config.get("ai", {}).get("model", configured_models[0]))
    date_col, model_col, force_col, check_col, action_col, _ = st.columns([1.4, 2.2, 1.35, 1.1, 1.5, 2.45])
    requested_date = date_col.date_input(
        "分析日期", value=date.today(), min_value=dates[0],
        max_value=max(date.today(), dates[-1]), format="YYYY-MM-DD", key="ai_analysis_date",
    )
    eligible = [value for value in dates if value <= requested_date]
    selected_date = eligible[-1] if eligible else dates[0]
    model_index = configured_models.index(default_model) if default_model in configured_models else 0
    selected_model = model_col.selectbox(
        "分析模型", configured_models, index=model_index,
        format_func=lambda value: model_labels.get(value, value),
    )
    force = force_col.toggle(
        "忽略缓存重新生成", value=False,
        help="开启后会再次调用DeepSeek并产生新的Token费用。",
    )
    if check_col.button("测试连接", use_container_width=True):
        try:
            status = test_ai_connection(config)
            st.success(f"连接正常。API 可用模型：{', '.join(status['models']) or '未返回列表'}")
        except Exception as exc:
            st.error(f"连接失败：{exc}")
    generate = action_col.button("生成AI分析", type="primary", use_container_width=True)
    user_question = st.text_area(
        "想让AI额外回答的问题（可选）",
        placeholder="例如：这20只股票中，哪些估值和波动相对均衡？请说明数据依据。",
        max_chars=2000,
        help="问题会和当日Top20的本地量价、估值、财务及宏观数据一起发送，并单独展示回答。",
    )
    if selected_model == "deepseek-v4-pro":
        st.caption(
            "官方当前仍接受 deepseek-v4-pro，但自2026-09-14起暂时路由到V4.1 Flash；"
            "待独立Pro重新上线后，界面无需改代码即可继续使用该模型名。"
        )
    if selected_date != requested_date:
        st.info(f"{requested_date} 不是本地交易日，已切换到 {selected_date}。")

    result = get_cached_analysis(config, selected_date)
    if generate:
        with st.spinner("正在整理Top-20和时点数据，并请求DeepSeek分析……"):
            try:
                result = run_ai_analysis(
                    config, selected_date, force=force,
                    model=selected_model, user_question=user_question,
                )
                if result.get("cached"):
                    st.success("输入数据没有变化，已直接读取本地AI分析缓存。")
                else:
                    st.success("DeepSeek分析已完成并持久化到本地SQLite。")
            except Exception as exc:
                st.error(f"生成失败：{exc}")
                return
    if result is None:
        st.info(
            "该日期还没有AI分析。请先在项目根目录 `.env` 中填写 "
            "`DEEPSEEK_API_KEY=你的密钥`，然后点击“生成AI分析”。"
        )
        return
    _render_ai_result(result)


def render_strategy_overview() -> None:
    st.title("策略总览")
    folders = _suite_folders()[:1]
    if not folders:
        st.info("尚无多策略结果。请运行：python scripts/run_strategy_suite.py")
        return
    control, _ = st.columns([2.2, 5.8])
    folder = control.selectbox("回测批次", folders, format_func=lambda value: value.name)
    suite = json.loads((folder / "suite.json").read_text(encoding="utf-8"))
    st.caption(
        f"统一比较区间：{suite['start_date']} 至 {suite['end_date']}｜"
        f"生成时间：{suite['generated_at']}"
    )
    st.caption(
        f"成交约束：{suite.get('trading_constraints', '未记录')}；"
        f"ST 过滤：按当前简称近似排除 {suite.get('st_filter', {}).get('excluded_count', 0)} 只。"
        "历史回测仍使用当前简称近似过滤 ST，后续应接入历史风险警示时点数据。"
    )
    visible_strategies = suite["strategies"]
    if not visible_strategies:
        st.info("该批次没有策略结果。")
        return
    st.caption(f"本批次共包含 {len(visible_strategies)} 套研究策略；结果按累计收益排序展示。")
    records = []
    for strategy in visible_strategies:
        metrics = strategy["metrics"]
        records.append({
            "策略": strategy["name"], "累计收益": metrics["cumulative_return"],
            "年化收益": metrics["annualized_return"], "超额收益": metrics["excess_return"],
            "最大回撤": metrics["max_drawdown"],
            "年化波动": metrics["annualized_volatility"], "日胜率": metrics["win_rate"],
            "总交易次数": metrics.get("total_trade_count"),
            "盈利次数": metrics.get("profitable_trade_count"),
            "亏损次数": metrics.get("losing_trade_count"),
            "交易胜率": metrics.get("profitable_trade_rate"),
            "止盈次数": metrics.get("take_profit_count", 0),
            "止损次数": metrics.get("stop_loss_count", 0),
            "连续未涨退出": metrics.get("non_up_exit_count", 0),
            "节前清仓次数": metrics.get("preholiday_exit_count", 0),
            "平均持有交易日": metrics.get("average_holding_days", 0),
            "涨停未买入": metrics.get("limit_up_buy_blocked_count", 0),
            "跌停未卖出": metrics.get("limit_down_sell_blocked_count", 0),
        })
    comparison = _integerize_counts(pd.DataFrame(records)).sort_values("累计收益", ascending=False)
    st.subheader("横向比较")
    st.dataframe(
        comparison.style.format({
            "累计收益": "{:.2%}", "年化收益": "{:.2%}", "超额收益": "{:.2%}",
            "最大回撤": "{:.2%}", "年化波动": "{:.2%}",
            "日胜率": "{:.2%}", "平均持有交易日": "{:.1f}",
            "交易胜率": "{:.2%}",
        }, na_rep="—"),
        width="stretch", hide_index=True,
    )
    st.subheader("累计收益对比")
    comparison_chart = alt.Chart(comparison).mark_bar().encode(
        x=alt.X("累计收益:Q", axis=alt.Axis(format=".0%", labelAngle=0)),
        y=alt.Y("策略:N", sort="-x", axis=alt.Axis(labelAngle=0, title=None)),
        tooltip=["策略:N", alt.Tooltip("累计收益:Q", format=".2%")],
        color=alt.condition(alt.datum["累计收益"] >= 0, alt.value("#d62728"), alt.value("#2ca02c")),
    ).properties(height=max(360, len(comparison) * 28))
    st.altair_chart(comparison_chart, width="stretch")
    control, _ = st.columns([3, 5])
    selected = control.selectbox("查看策略规则", visible_strategies, format_func=lambda value: value["name"])
    threshold = selected["metrics"].get("take_profit_threshold")
    threshold_note = (
        f"该策略在盈利达到{threshold:.0%}时记录完整事件并止盈。"
        if selected["metrics"].get("threshold_enabled", True) and threshold is not None
        else "该策略不应用固定止盈规则。"
    )
    stop_loss = selected["metrics"].get("stop_loss_threshold")
    if stop_loss is not None:
        threshold_note += f" 亏损达到{stop_loss:.0%}时止损。"
    st.info(selected["description"] + " " + threshold_note)
    st.caption("进入“策略分析”页面可查看所选策略的净值、回撤、逐日数据、交易区间和阈值事件。")


def _render_glossary() -> None:
    with st.expander("指标与专业名词解释", expanded=False):
        st.markdown(
            "- **累计收益**：整个回测区间从起点到终点一共赚或亏多少。\n"
            "- **年化收益**：把累计收益折算成每年复利增长率；越高越好，但必须与基准和回撤一起看。\n"
            "- **沪深300ETF基准收益**：同期买入并持有 510300.SH 的收益，是独立市场参照，不是你的基准策略。\n"
            "- **超额收益**：策略累计收益减去基准累计收益；大于0较好，小于0表示跑输基准。\n"
            "- **最大回撤**：越接近0越好。10%以内较低，10%～20%中等，30%以上通常属于高回撤。\n"
            "- **年化波动率**：越低越稳定；股票策略可粗略将15%以下视为较低、15%～30%中等、30%以上较高。\n"
            "- **日胜率**：正收益交易日占比；越高通常越好，但还必须结合每次盈亏幅度。\n"
            "- **固定止盈/止损**：持仓期间价格达到设定边界时模拟退出；小盘重点观察策略为25%止盈、10%止损。\n"
            "- **连续未上涨退出**：连续两个交易日收盘价均未高于各自前一日收盘价，第二日收盘确认后，于下一交易日开盘模拟退出。\n"
            "- **总交易次数**：已经完成买入和卖出的完整交易数；期末仍持有的仓位不计入。\n"
            "- **盈利/亏损次数**：按扣除买卖成本后的单笔净收益大于0或小于0统计，等于0单列为持平。\n"
            "- **交易胜率**：盈利交易次数÷总已平仓交易次数；持有中的仓位不计入。\n"
            "- **换手率**：越低越节省成本，但过低也可能反应迟钝；需要结合超额收益判断。以上区间都是研究经验值，不是保证。"
        )


def render_backtest() -> None:
    st.title("策略分析")
    folders = _strategy_result_folders()
    if not folders:
        st.info("当前没有可用的策略回测结果。请运行：python scripts/run_strategy_suite.py")
        return
    control, _ = st.columns([3, 5])
    folder = control.selectbox("选择策略回测结果", folders, format_func=_result_label)
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    st.info(metrics.get("strategy_description", "旧版定期调仓策略结果。"))
    trades_path = folder / "trades.csv"
    if not trades_path.exists():
        trades_path = folder / "rebalances.csv"
    events_path = folder / "threshold_events.csv"

    columns = st.columns(4)
    columns[0].metric("累计收益", _percent(metrics["cumulative_return"]))
    columns[1].metric("年化收益", _percent(metrics["annualized_return"]))
    columns[2].metric("沪深300ETF基准收益", _percent(metrics["benchmark_return"]))
    columns[3].metric("超额收益", _percent(metrics["excess_return"]))
    columns = st.columns(4)
    columns[0].metric("最大回撤", _percent(metrics["max_drawdown"]))
    columns[1].metric("年化波动", _percent(metrics["annualized_volatility"]))
    columns[2].metric("胜率", _percent(metrics["win_rate"]))
    columns[3].metric("交易/调仓次数", int(metrics.get("completed_trades", metrics.get("rebalance_count", 0))))
    if metrics.get("threshold_enabled", "crossed_20_count" in metrics):
        columns = st.columns(4)
        columns[0].metric("总交易次数", int(metrics.get("total_trade_count", 0)))
        columns[1].metric("盈利次数", int(metrics.get("profitable_trade_count", 0)))
        columns[2].metric("亏损次数", int(metrics.get("losing_trade_count", 0)))
        columns[3].metric("交易胜率", _percent(metrics.get("profitable_trade_rate", 0)))
        columns = st.columns(4)
        threshold = metrics.get("take_profit_threshold")
        threshold_label = f"{threshold:.0%}止盈次数" if threshold is not None else "止盈次数"
        columns[0].metric(threshold_label, int(metrics.get("take_profit_count", 0)))
        columns[1].metric("持平次数", int(metrics.get("flat_trade_count", 0)))
        columns[2].metric("平均持有交易日", f"{metrics.get('average_holding_days', 0):.1f}")
        columns[3].metric("期末持仓", int(metrics.get("open_positions", 0)))
        if metrics.get("stop_loss_threshold") is not None:
            columns = st.columns(4)
            columns[0].metric("止损次数", int(metrics.get("stop_loss_count", 0)))
            columns[1].metric("连续未涨退出", int(metrics.get("non_up_exit_count", 0)))
            columns[2].metric("节前清仓次数", int(metrics.get("preholiday_exit_count", 0)))
            columns[3].metric("模拟初始资金", f"{float(metrics.get('initial_cash', 0)):,.0f} 元")
        columns = st.columns(4)
        columns[0].metric("涨停未买入", int(metrics.get("limit_up_buy_blocked_count", 0)))
        columns[1].metric("跌停未卖出", int(metrics.get("limit_down_sell_blocked_count", 0)))
        columns[2].metric("平均盈利交易", _percent(metrics.get("average_winner_return", 0)))
        columns[3].metric("平均亏损交易", _percent(metrics.get("average_loser_return", 0)))

    subpage = st.radio(
        "分析子页面", ["绩效概览", "逐日结果", "交易区间", "盈利事件"],
        horizontal=True, key=f"backtest_subpage_{folder}",
    )
    if subpage == "绩效概览":
        daily = pd.read_csv(folder / "daily.csv", parse_dates=["trade_date", "signal_date"])
        st.subheader("净值曲线")
        equity = daily.set_index("trade_date")[["equity", "benchmark_equity"]]
        equity.columns = ["策略", "沪深300ETF（510300）"]
        st.line_chart(equity)
        st.subheader("回撤")
        drawdown = daily.set_index("trade_date")[["drawdown"]].rename(columns={"drawdown": "策略回撤"})
        st.area_chart(drawdown)
        st.subheader("年度收益")
        yearly = pd.DataFrame.from_dict(metrics["yearly_returns"], orient="index", columns=["收益"])
        st.dataframe(yearly.style.format("{:.2%}"), width="stretch")
    elif subpage == "逐日结果":
        daily = pd.read_csv(folder / "daily.csv", parse_dates=["trade_date", "signal_date"])
        st.subheader("逐日结果")
        st.dataframe(daily.rename(columns=DAILY_NAMES), width="stretch", hide_index=True, height=680)
    elif subpage == "交易区间" and trades_path.exists():
        st.subheader("交易与持股区间")
        trades = pd.read_csv(trades_path).rename(columns=TRADE_NAMES)
        for column in ["价格变化", "净收益", "持有期最大盈利", "持有期最大不利波动"]:
            if column in trades:
                trades[column] = trades[column].map(lambda value: f"{value:.2%}")
        st.dataframe(trades, width="stretch", hide_index=True, height=680)
    elif subpage == "交易区间":
        st.info("该回测没有交易区间明细。")
    elif subpage == "盈利事件" and events_path.exists() and metrics.get("threshold_enabled", True):
        st.subheader("盈利阈值事件")
        events = pd.read_csv(events_path).rename(columns=EVENT_NAMES)
        if events.empty:
            st.info("该策略没有出现触发固定止盈阈值的持仓事件。")
        else:
            if "最高涨幅" in events:
                events["最高涨幅"] = events["最高涨幅"].map(lambda value: f"{value:.2%}")
            for column in ["最终价格变化", "最终净收益"]:
                if column in events:
                    events[column] = events[column].map(lambda value: f"{value:.2%}")
            st.dataframe(events, width="stretch", hide_index=True, height=680)
    elif subpage == "盈利事件":
        st.info("该策略没有单独的盈利阈值事件页面。")
    st.caption(
        "成交规则说明：信号在收盘后产生，次一交易日开盘执行；开盘封涨停时买单视为无法成交，"
        "开盘封跌停时卖单顺延。持有期最大盈利越高表示曾出现更大浮盈；最大不利波动越负表示持仓期间承受的下跌越深。"
    )
    st.divider()
    _render_glossary()


def render_strategy_hub() -> None:
    st.info("这里统一展示原有五套Top-K规则，以及新增的小盘重点观察单股/双股重仓模拟。")
    section = st.sidebar.radio(
        "策略研究分类", ["策略总览", "策略分析"],
        help="两个分类共用同一个网址，切换时只加载当前分类。",
    )
    if section == "策略总览":
        render_strategy_overview()
    else:
        render_backtest()


def render_market_overview() -> None:
    st.title("市场概览")
    st.caption("把本地行情整理成日常阅读信息，用来判断当日市场强弱、成交活跃度和行业分布。")
    config = load_config(PROJECT_ROOT / "config.yaml")
    store = build_store(config)
    daily = store.load_daily()
    if daily.empty:
        st.info("尚无本地行情，请先在每日候选页更新数据。")
        return
    dates = [pd.Timestamp(value).date() for value in sorted(daily["trade_date"].unique())]
    control, _ = st.columns([1.6, 6.4])
    requested = control.date_input(
        "查看日期", value=dates[-1], min_value=dates[0], max_value=dates[-1],
        format="YYYY-MM-DD", key="market_overview_date",
    )
    selected = max(value for value in dates if value <= requested)
    day = daily[daily["trade_date"].eq(pd.Timestamp(selected))].copy()
    day["pct_chg"] = pd.to_numeric(day["pct_chg"], errors="coerce") / 100
    advances = int(day["pct_chg"].gt(0).sum())
    declines = int(day["pct_chg"].lt(0).sum())
    flats = int(day["pct_chg"].eq(0).sum())
    limits = store.load_market_dataset("limit")
    if not limits.empty:
        limit_day = limits[limits["trade_date"].eq(pd.Timestamp(selected))]
        day = day.merge(
            limit_day[["ts_code", "up_limit", "down_limit"]], on="ts_code", how="left",
        )
        limit_up = int(day["close"].ge(day["up_limit"] - .005).sum())
        limit_down = int(day["close"].le(day["down_limit"] + .005).sum())
    else:
        limit_up = limit_down = 0
    total_amount_trillion = float(day["amount"].sum()) / 1_000_000_000
    metrics = st.columns(6)
    metrics[0].metric("上涨家数", advances)
    metrics[1].metric("下跌家数", declines)
    metrics[2].metric("平盘家数", flats)
    metrics[3].metric("涨停 / 跌停", f"{limit_up} / {limit_down}")
    metrics[4].metric("涨跌幅中位数", f"{day['pct_chg'].median():.2%}")
    metrics[5].metric("全市场成交额", f"{total_amount_trillion:.2f} 万亿元")

    valuation = store.load_market_dataset("daily_basic")
    valuation_day = valuation[valuation["trade_date"].eq(pd.Timestamp(selected))].copy()
    benchmark = store.load_market_dataset("benchmark")
    benchmark_window = benchmark[benchmark["trade_date"].le(pd.Timestamp(selected))].sort_values("trade_date").tail(21)
    benchmark_return = (
        float(benchmark_window["close"].iloc[-1] / benchmark_window["close"].iloc[0] - 1)
        if len(benchmark_window) > 1 else None
    )
    secondary = st.columns(4)
    pe = pd.to_numeric(valuation_day.get("pe_ttm"), errors="coerce")
    pb = pd.to_numeric(valuation_day.get("pb"), errors="coerce")
    secondary[0].metric("沪深300ETF近20日", f"{benchmark_return:.2%}" if benchmark_return is not None else "—")
    secondary[1].metric("上涨股票占比", f"{advances / max(len(day), 1):.1%}")
    secondary[2].metric("盈利股票PE中位数", f"{pe[pe.gt(0)].median():.1f}" if pe.notna().any() else "—")
    secondary[3].metric("PB中位数", f"{pb[pb.gt(0)].median():.2f}" if pb.notna().any() else "—")

    left, right = st.columns([1.15, .85])
    distribution = day[day["pct_chg"].between(-.12, .12)].copy()
    chart = alt.Chart(distribution).mark_bar().encode(
        x=alt.X("pct_chg:Q", bin=alt.Bin(maxbins=40), axis=alt.Axis(format=".0%", labelAngle=0), title="当日涨跌幅"),
        y=alt.Y("count():Q", title="股票数量"),
        color=alt.condition(alt.datum.pct_chg >= 0, alt.value("#d62728"), alt.value("#2ca02c")),
    ).properties(height=350, title="市场涨跌分布")
    left.altair_chart(chart, width="stretch")
    basic = store.load_basic()
    industry = day.merge(
        basic[["ts_code", "industry"]].drop_duplicates("ts_code"), on="ts_code", how="left",
    )
    industry["industry"] = industry["industry"].fillna("未分类")
    industry_table = industry.groupby("industry").agg(
        股票数=("ts_code", "size"), 上涨占比=("pct_chg", lambda values: float(values.gt(0).mean())),
        涨跌幅中位数=("pct_chg", "median"), 成交额=("amount", "sum"),
    ).reset_index().rename(columns={"industry": "行业"})
    industry_table["成交额（亿元）"] = industry_table.pop("成交额") / 100_000
    industry_table = industry_table[industry_table["股票数"].ge(5)].sort_values("涨跌幅中位数", ascending=False)
    right.dataframe(
        industry_table.head(15).style.format({
            "上涨占比": "{:.1%}", "涨跌幅中位数": "{:.2%}", "成交额（亿元）": "{:.1f}",
        }), width="stretch", hide_index=True, height=385,
    )
    st.caption(
        f"行情日期：{selected}。行业采用当前基础信息，只用于阅读当日分布；PE只统计正值，"
        "沪深300ETF近20日为价格收益。"
    )


@st.cache_data(ttl=1800, show_spinner="正在整理连板历史和晋级观察指标…")
def _cached_board_history(cache_version: tuple[float, ...]) -> pd.DataFrame:
    config = load_config(PROJECT_ROOT / "config.yaml")
    store = build_store(config)
    return build_board_history(
        store.load_daily(), store.load_market_dataset("limit"), store.load_basic(),
        store.load_market_dataset("daily_basic"),
        exclude_star_market=config["market"].get("exclude_star_market", True),
        exclude_chinext_market=config["market"].get("exclude_chinext_market", True),
    )


@st.cache_data(ttl=1800, show_spinner="正在扫描小盘、低位、横盘和放量大阳线…")
def _cached_low_position_pool(
    cache_version: tuple[float, ...], selected_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(PROJECT_ROOT / "config.yaml")
    store = build_store(config)
    return screen_low_position_breakouts(
        store.load_daily(), store.load_market_dataset("daily_basic"), store.load_basic(),
        config["low_position_pool"], pd.Timestamp(selected_date), config["market"],
    )


def _low_position_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    fields = [
        "observation_level", "ts_code", "name", "industry", "source", "trade_time", "fetched_at",
        "pool_score", "alert_score",
        "circ_mv", "close", "prior_return_60", "prior_return_120",
        "daily_return", "body_pct", "volume_pace_ratio", "amount_pace_ratio", "turnover_pace",
        "volume_ratio", "amount_ratio", "previous_day_volume_ratio", "turnover_rate", "trading_progress",
        "low_position", "consolidation_range", "prior_volatility", "close_location",
        "amount",
    ]
    result = frame[[column for column in fields if column in frame]].copy()
    result["circ_mv"] = result["circ_mv"] / 10_000  # Tushare: 万元 -> 亿元
    result["amount"] = result["amount"] / 100_000  # Tushare: 千元 -> 亿元
    for column in ("prior_return_60", "prior_return_120", "daily_return", "body_pct",
                   "low_position", "consolidation_range", "prior_volatility", "close_location",
                   "trading_progress"):
        if column not in result:
            continue
        result[column] = result[column] * 100
    return result.rename(columns={
        "observation_level": "观察级别", "ts_code": "股票代码", "name": "证券简称",
        "industry": "行业(当前)", "source": "实时来源", "trade_time": "行情时间",
        "fetched_at": "获取时间", "pool_score": "备选评分", "alert_score": "异动强度",
        "circ_mv": "流通市值(亿元)", "prior_return_60": "此前60日涨幅(%)",
        "prior_return_120": "此前120日涨幅(%)",
        "close": "收盘价", "daily_return": "当日涨幅(%)", "body_pct": "实体涨幅(%)",
        "volume_pace_ratio": "盘中量能速度", "amount_pace_ratio": "盘中成交额速度",
        "turnover_pace": "预计全天换手率(%)", "trading_progress": "交易时段进度(%)",
        "previous_day_volume_ratio": "相对昨日成交量",
        "volume_ratio": "成交量倍数", "amount_ratio": "成交额倍数", "turnover_rate": "换手率(%)",
        "low_position": "250日价格位置(%)", "consolidation_range": "板前20日区间(%)",
        "prior_volatility": "板前年化波动(%)", "close_location": "收盘所处日内位置(%)",
        "amount": "成交额(亿元)", "matched_rules": "已满足", "failed_rules": "未满足",
    })


def render_low_position_pool() -> None:
    st.title("小盘蓄势备选池")
    st.caption(
        "先收集小盘、此前未明显上涨且近期横盘的股票；当其中某只突然放量收大阳线时，"
        "升级为红色“重点观察”。页面只使用截至所选日期已经产生的数据。"
    )
    config = load_config(PROJECT_ROOT / "config.yaml")
    store = build_store(config)
    paths = [store.daily_path, store.daily_basic_path, store.basic_path, PROJECT_ROOT / "config.yaml"]
    if not store.daily_path.exists() or not store.basic_path.exists():
        st.info("缺少日线或股票列表，请先在“每日候选”更新数据。")
        return
    if not store.daily_basic_path.exists():
        st.info("缺少每日估值数据，无法判断流通市值和换手率。请先运行每日数据更新。")
        return
    daily = store.load_daily()
    dates = sorted(pd.Timestamp(value).date() for value in daily["trade_date"].unique())
    control, _ = st.columns([1.4, 6.6])
    requested = control.date_input(
        "筛选日期", value=dates[-1], min_value=dates[0], max_value=dates[-1],
        format="YYYY-MM-DD", key="low_position_date",
    )
    selected = max((day for day in dates if day <= requested), default=dates[0])
    version = tuple(path.stat().st_mtime if path.exists() else 0.0 for path in paths)
    priority, pool = _cached_low_position_pool(version, selected)
    settings = config["low_position_pool"]

    counters = st.columns(4)
    counters[0].metric("长期备选", len(pool))
    counters[1].metric("今日重点异动", len(priority))
    counters[2].metric("小盘上限", f"{float(settings['max_circ_mv_yi']):.0f}亿元")
    counters[3].metric("最低放量", f"{float(settings['min_volume_ratio']):.1f}倍")

    with st.expander("当前筛选标准"):
        st.markdown(
            f"- **小盘**：流通市值 {settings['min_circ_mv_yi']}～{settings['max_circ_mv_yi']} 亿元。\n"
            f"- **价格范围**：最新收盘价必须在 {settings['min_price']}～{settings['max_price']} 元之间；"
            "边界价格包含在内。\n"
            f"- **没有明显上涨**：满足以下任一种即可：位于近 {settings['lookback_days']} 日区间的前 "
            f"{float(settings['max_low_position']):.0%}；或者此前60日涨幅不超过 "
            f"{float(settings['max_prior_return_60']):.0%} 且此前120日涨幅不超过 "
            f"{float(settings['max_prior_return_120']):.0%}。因此长期横盘但处于自身窄区间上沿的股票也能入池。\n"
            f"- **震荡收敛**：信号日前 {settings['consolidation_days']} 日区间不超过 "
            f"{float(settings['max_consolidation_range']):.0%}，年化波动不超过 "
            f"{float(settings['max_prior_volatility']):.0%}。\n"
            f"- **进入备选池**不要求当天上涨；只需同时满足小盘、未明显上涨、震荡收敛和基础流动性。\n"
            f"- **升级为重点观察**：当日成交量至少为此前20日均量的 {settings['min_volume_ratio']} 倍，"
            f"成交额至少为此前20日均额的 {settings['min_amount_ratio']} 倍，并且累计成交量至少达到"
            f"上一交易日全天成交量的 {settings['min_previous_day_volume_ratio']} 倍。\n"
            f"- **大阳线**：当日涨幅至少 {float(settings['min_daily_return']):.0%}、实体至少 "
            f"{float(settings['min_body_pct']):.0%}，收盘位于日内振幅上方 "
            f"{float(settings['min_close_location']):.0%}。\n"
            f"- **可交易性**：备选池换手率至少 {settings['pool_min_turnover_rate']}%；重点异动换手率至少 "
            f"{settings['min_turnover_rate']}%，最高 {settings['max_turnover_rate']}%，"
            "排除 ST、退市、科创板和创业板。"
        )

    if priority.empty:
        st.info("今天的备选池中没有出现同时满足放量、大阳线和强势收盘的重点异动。")
    else:
        labels = "、".join(f"{row['name']}（{row['ts_code']}）" for _, row in priority.head(12).iterrows())
        st.error(f"🔴 今日发现 {len(priority)} 只重点观察标的：{labels}")

    alert_tab, pool_tab = st.tabs(["🔴 重点异动提醒", f"长期备选池（共 {len(pool)} 只）"])
    formatters = {
        "备选评分": "{:.1f}", "异动强度": "{:.1f}", "流通市值(亿元)": "{:.1f}",
        "收盘价": "{:.2f}", "成交量倍数": "{:.2f}", "成交额倍数": "{:.2f}",
        "盘中量能速度": "{:.2f}", "盘中成交额速度": "{:.2f}",
        "相对昨日成交量": "{:.2f}",
        "成交额(亿元)": "{:.2f}",
    }
    with alert_tab:
        if priority.empty:
            st.info("所选日期没有重点异动，继续保留备选池观察即可。")
        else:
            display = _low_position_display(priority)
            st.dataframe(
                display.style.format(formatters).map(
                    lambda _: "color:#d62728;font-weight:700", subset=["观察级别", "证券简称"],
                ),
                hide_index=True, width="stretch",
            )
            st.download_button(
                "下载重点异动 CSV", priority.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"small_cap_priority_{selected}.csv", mime="text/csv",
            )
    with pool_tab:
        if pool.empty:
            st.info("所选日期没有符合基础条件的备选股票。")
        else:
            display = _low_position_display(pool)
            st.dataframe(
                display.style.format(formatters).map(
                    lambda value: "color:#d62728;font-weight:700" if value == "🔴 重点观察" else "",
                    subset=["观察级别"],
                ), hide_index=True, width="stretch",
            )
            st.download_button(
                "下载完整备选池 CSV", pool.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"small_cap_watch_pool_{selected}.csv", mime="text/csv",
            )

    st.subheader("盘中启动雷达")
    st.caption(
        "基于上一收盘日形成的小盘备选池，盘中按已经过去的交易分钟折算量能和成交额速度，"
        "尽早提示可能启动的股票；只做研究展示，不写入历史日线、不改变Top-K，也不产生交易指令。"
    )
    provider_names = {"tdxquant": "通达信 TdxQuant（推荐）", "tushare_rt": "Tushare rt_k（需单独付费权限）"}
    source_col, auto_col, test_col, _ = st.columns([2.2, 1.5, 1.5, 2.8])
    configured = str(config.get("intraday", {}).get("provider", "tdxquant"))
    source_ids = list(provider_names)
    source_id = source_col.selectbox(
        "实时行情源", source_ids, index=source_ids.index(configured) if configured in source_ids else 0,
        format_func=provider_names.get, key="live_pool_provider",
    )
    refresh_seconds = int(config.get("intraday", {}).get("refresh_seconds", 30))
    auto_refresh = auto_col.toggle(f"每{refresh_seconds}秒自动刷新", value=False, key="live_pool_auto")
    if test_col.button("测试行情连接", key="test_live_provider"):
        try:
            test_quote = build_realtime_provider(config, source_id).snapshot(["600000.SH"])
            if test_quote.empty or test_quote.get("close", pd.Series(dtype=float)).isna().all():
                st.warning("服务已响应，但没有返回有效行情，请检查登录状态和行情权限。")
            else:
                quote = test_quote.iloc[0]
                st.success(
                    f"连接成功：600000.SH 现价 {float(quote['close']):.2f}，"
                    f"行情时间 {quote.get('trade_time', '—')}。"
                )
        except RealtimeQuoteError as exc:
            st.error(str(exc))

    @st.fragment(run_every=f"{refresh_seconds}s" if auto_refresh else None)
    def live_pool_panel() -> None:
        if selected != dates[-1]:
            st.info(f"当前选择的是历史日期 {selected}；盘中监控只对本地最新交易日 {dates[-1]} 开放。")
            return
        refresh = st.button("立即刷新实时行情", type="primary", key="live_pool_refresh")
        state_key = f"live_pool_result_{source_id}"
        error_key = f"live_pool_error_{source_id}"
        if (refresh or auto_refresh) and not pool.empty:
            try:
                provider = build_realtime_provider(config, source_id)
                quotes = provider.snapshot(pool["ts_code"].astype(str).tolist())
                alerts, monitored = evaluate_realtime_alerts(
                    pool, daily, quotes, config["low_position_pool"],
                )
                st.session_state[state_key] = (alerts, monitored)
                st.session_state.pop(error_key, None)
            except RealtimeQuoteError as exc:
                st.session_state[error_key] = str(exc)
        if st.session_state.get(error_key):
            st.error(st.session_state[error_key])
            if source_id == "tdxquant":
                st.caption("请先安装并登录官方“金融终端（量化模拟）”或其他支持TQ的通达信客户端。")
            else:
                st.caption("你现有的2000积分权限不包含rt_k，需要在Tushare单独开通“A股日线RT”。")
        cached = st.session_state.get(state_key)
        if not cached:
            st.info("点击“立即刷新实时行情”。橙色表示可能启动，红色表示量价条件更强的重点观察。")
            return
        alerts, monitored = cached
        if monitored.empty:
            st.warning("行情源没有返回备选池股票的有效快照，请检查当前是否为交易时段及客户端状态。")
            return
        source = str(monitored.iloc[0].get("source", provider_names[source_id]))
        fetched_at = str(monitored.iloc[0].get("fetched_at", ""))
        st.caption(f"来源：{source}｜本次获取：{fetched_at}｜成功监控：{len(monitored)}/{len(pool)}只")
        if alerts.empty:
            st.success("当前备选池内尚未出现明显的盘中启动迹象。")
        else:
            priority_live = alerts[alerts["priority_alert"]]
            early_live = alerts[~alerts["priority_alert"] & alerts["early_warning"]]
            if not priority_live.empty:
                names = "、".join(
                    f"{row['name']}（{row['ts_code']}）" for _, row in priority_live.head(15).iterrows()
                )
                st.error(f"🔴 盘中重点观察 {len(priority_live)} 只：{names}")
            if not early_live.empty:
                names = "、".join(
                    f"{row['name']}（{row['ts_code']}）" for _, row in early_live.head(15).iterrows()
                )
                st.warning(f"🟠 可能启动 {len(early_live)} 只：{names}")
            live_display = _low_position_display(alerts)
            st.dataframe(
                live_display.style.format(formatters).map(
                    lambda value: (
                        "color:#d62728;font-weight:700" if str(value).startswith("🔴")
                        else "color:#e67e22;font-weight:700" if str(value).startswith("🟠") else ""
                    ), subset=["观察级别"],
                ), hide_index=True, width="stretch",
            )

    live_pool_panel()

    st.info(
        "橙色和红色均是盘中研究提醒：它们表示价格强度与成交速度出现异常，不等于确认“主力启动”，"
        "也不会触发下单、撤单或任何自动交易。盘中数据尚未收盘，信号可能随行情减弱或消失。"
    )


def render_limit_up() -> None:
    st.title("打板观察")
    st.caption("基于收盘封住涨停价的日线统计；已排除科创板和创业板。收盘后更新，不是盘中可成交的打板信号。")
    config = load_config(PROJECT_ROOT / "config.yaml")
    store = build_store(config)
    paths = [store.daily_path, store.limit_path, store.basic_path, store.daily_basic_path,
             PROJECT_ROOT / "config.yaml"]
    if any(not path.exists() for path in paths[:3]):
        st.info("缺少日线、涨跌停价或股票列表，请先运行每日数据更新。")
        return
    version = tuple(path.stat().st_mtime if path.exists() else 0.0 for path in paths)
    boards = _cached_board_history(version)
    if boards.empty:
        st.info("当前缓存内没有可识别的收盘涨停股票。")
        return
    dates = sorted(pd.Timestamp(value).date() for value in boards.trade_date.unique())
    control, _ = st.columns([1.4, 6.6])
    requested = control.date_input("观察日期", value=dates[-1], min_value=dates[0], max_value=dates[-1],
                                   format="YYYY-MM-DD", key="limit_up_date")
    selected = max((day for day in dates if day <= requested), default=dates[0])
    day = boards[boards.trade_date.eq(pd.Timestamp(selected))].copy()
    candidates, validation = promotion_candidates(boards, pd.Timestamp(selected))
    if not candidates.empty:
        day = day.merge(candidates[["ts_code", "estimated_promotion"]], on="ts_code", how="left")
    first, second, more = (day[day.streak.eq(1)], day[day.streak.eq(2)], day[day.streak.ge(3)])
    measures = st.columns(5)
    measures[0].metric("首板", len(first))
    measures[1].metric("二连板", len(second))
    measures[2].metric("三连板及以上", len(more))
    measures[3].metric("最高连板", int(day.streak.max()))
    measures[4].metric("一字板", int(day.one_word.sum()))
    history = promotion_summary(boards, pd.Timestamp(selected))
    if not history.empty:
        stat = {int(row["当前板数"]): row for _, row in history.iterrows()}
        for stage, label in ((1, "首板→二板"), (2, "二板→三板")):
            if stage in stat:
                item = stat[stage]
                st.caption(f"历史{label}：{item['历史晋级率']:.1%}（{int(item['晋级数'])}/{int(item['样本数'])}，"
                           "只统计所选日期之前已有次日结果的封板样本；不是今日个股预测概率）")
    tab1, tab2, tab3 = st.tabs(["连板名单", "行业与龙头候选", "晋级观察"])

    def display(frame: pd.DataFrame) -> pd.DataFrame:
        fields = ["ts_code", "name", "industry", "streak", "close", "amount", "pre20_range",
                  "low_position", "volume_surge", "turnover_rate", "industry_boards", "one_word",
                  "estimated_promotion"]
        result = frame[fields].copy()
        result["amount"] = result.amount / 100_000  # Tushare amount is in thousand yuan.
        for field in ("pre20_range", "low_position"):
            result[field] = (result[field] * 100).round(1)
        result["volume_surge"] = result.volume_surge.round(2)
        result["turnover_rate"] = result.turnover_rate.round(1)
        result["estimated_promotion"] = (result.estimated_promotion * 100).round(1)
        return result.rename(columns={"ts_code": "代码", "name": "简称", "industry": "行业(当前)",
                                      "streak": "连板数", "close": "收盘价", "amount": "成交额(亿元)",
                                      "pre20_range": "板前20日振幅(%)", "low_position": "120日价格位置(%)",
                                      "volume_surge": "当日放量倍数", "turnover_rate": "换手率(%)",
                                      "industry_boards": "同行业涨停数", "one_word": "一字板",
                                      "estimated_promotion": "统计估计晋级率(%)"})

    with tab1:
        choice = st.radio("连板筛选", ["全部", "首板", "二连板", "三连板及以上"], horizontal=True)
        view = {"全部": day, "首板": first, "二连板": second, "三连板及以上": more}[choice]
        st.dataframe(display(view.sort_values(["streak", "estimated_promotion", "amount"], ascending=False)),
                     hide_index=True, width="stretch")
    with tab2:
        industry = day.groupby("industry").agg(涨停家数=("ts_code", "size"),
                                                最高连板=("streak", "max"),
                                                二板及以上=("streak", lambda x: int(x.ge(2).sum())),
                                                成交额=("amount", "sum")).reset_index()
        industry["成交额(亿元)"] = (industry.pop("成交额") / 100_000).round(2)
        industry = industry.rename(columns={"industry": "行业(当前)"}).sort_values(
            ["涨停家数", "最高连板"], ascending=False)
        st.dataframe(industry, hide_index=True, width="stretch")
        st.subheader("各行业龙头观察候选")
        leaders = day.sort_values(["industry", "streak", "amount"],
                                  ascending=[True, False, False]).drop_duplicates("industry")
        st.dataframe(display(leaders).head(30), hide_index=True, width="stretch")
        st.caption("这里的‘龙头候选’仅指同一当前行业中连板数最高、再按成交额排序；不等于市场公认龙头。")
    with tab3:
        st.write("按历史相似条件估计次日继续收盘封板的概率，并分别排序首板与二板。")
        if not validation.empty:
            display_validation = validation.copy()
            for field in ("整体晋级率", "每日评分前20%晋级率"):
                display_validation[field] = (display_validation[field] * 100).round(1)
            st.dataframe(display_validation, hide_index=True, width="stretch")
            st.caption("上表两项晋级率单位为%；模型仅用验证期之前的数据拟合，在留出的较新日期检验。"
                       "晋级率提升不等于交易盈利，也不保证未来维持。")
        for stage, title in ((1, "首板→二板"), (2, "二板→三板")):
            st.subheader(title)
            view = day[day.streak.eq(stage)].sort_values("estimated_promotion", ascending=False)
            st.dataframe(display(view), hide_index=True, width="stretch")
        st.caption("模型使用板前横盘、复权口径价格位置、当日放量、换手、行业热度、一字板和成交额；"
                   "统计估计尚未经概率校准，不能当成可实现买入收益。放量不证明主力启动。"
                   "数据只有收盘日线，无法判断封单、开板次数或排队成交。"
                   "ST 仅用当前简称及 5% 涨停幅度近似排除；行业为当前分类，历史行业归属可能不同。")


def main() -> None:
    st.set_page_config(page_title="JKQuant", page_icon="📈", layout="wide")
    title, refresh = st.columns([9, 1])
    title.markdown("## JKQuant 选股系统")
    if refresh.button("刷新"):
        st.rerun()
    navigation = st.navigation([
        st.Page(render_topk, title="每日候选", icon="📋", url_path="candidates", default=True),
        st.Page(render_ai_analysis, title="AI分析", icon="🤖", url_path="ai-analysis"),
        st.Page(render_low_position_pool, title="小盘备选池", icon="🌱", url_path="low-position-pool"),
        st.Page(render_strategy_hub, title="策略研究", icon="📊", url_path="strategy-research"),
        st.Page(render_market_overview, title="市场概览", icon="🌐", url_path="market"),
        st.Page(render_limit_up, title="打板观察", icon="🚀", url_path="limit-up"),
    ], position="top")
    st.divider()
    navigation.run()


if __name__ == "__main__":
    main()
