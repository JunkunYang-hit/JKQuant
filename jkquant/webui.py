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
from jkquant.config import apply_strategy_profile, load_config
from jkquant.data.account_store import AccountStore
from jkquant.data.selection_cache import strategy_key
from jkquant.holiday_risk import holiday_risk
from jkquant.limit_up import build_board_history, promotion_candidates, promotion_summary
from jkquant.pipeline import (
    RECOMMENDATION_HISTORY_START, available_selection_dates,
    build_store, run_daily,
    combined_signal_recommendations, recommendation_history_stats, selection_for_date,
    stock_signal_reminders, top20_streak2_leader_probability,
)
from jkquant.strategy import is_chinext_market, is_star_market

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"
BACKTESTS_ROOT = PROJECT_ROOT / "backtests"
STRATEGY_LAB_ROOT = BACKTESTS_ROOT / "strategy_lab"


def _account_store() -> AccountStore:
    config = load_config(PROJECT_ROOT / "config.yaml")
    return AccountStore(build_store(config).root / "account.sqlite3")

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
    return [folder for folder in folders if _positive_result(folder / "metrics.json")]


def _positive_result(path: Path) -> bool:
    try:
        return float(json.loads(path.read_text(encoding="utf-8")).get("cumulative_return", 0)) > 0
    except (ValueError, OSError, TypeError):
        return False


def _result_label(folder: Path) -> str:
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    name = metrics.get("strategy_name", folder.name)
    if folder.parent.parent.name == "strategy_suite":
        return f"{name}｜{folder.parent.name}"
    if folder.parent.name == "streak2_leader":
        return f"{name}｜{folder.name}"
    if folder.parent.parent.name == "strategy_research":
        period = metrics.get("research_period", folder.name)
        return f"研究方案｜{period}｜{name}｜{folder.parent.name}"
    return f"旧版单策略｜{folder.name}"


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
        with st.spinner("正在更新最新数据、计算 Top-50 并同步信号缓存……"):
            try:
                report_path, _ = run_daily(base_config, date.today())
                latest_date = build_store(base_config).load_daily()["trade_date"].max().date()
                combined_signal_recommendations(base_config, latest_date)
                st.success(f"已完成 {latest_date} 推荐并同步到其他模块：{report_path.name}")
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
    filter_left, filter_right, profile_col, _ = st.columns([1.1, 1.5, 2.2, 3.2])
    top_k = int(filter_left.number_input(
        "推荐股票数量 K", min_value=1, max_value=50, value=default_top_k, step=1,
        help="数据库统一缓存 Top-50，修改 K 只截取前 K。",
    ))
    requested_date = filter_right.date_input(
        "选择推荐日期", value=date.today(), min_value=dates[0],
        max_value=max(date.today(), dates[-1]), format="YYYY-MM-DD",
    )
    profiles = base_config.get("strategy_profiles", {"baseline": {"name": "原固定权重"}})
    profile_id = profile_col.selectbox(
        "候选策略", list(profiles), format_func=lambda value: profiles[value].get("name", value),
        help="防守型策略是独立研究对照，不会覆盖原固定策略。",
    )
    config = apply_strategy_profile(base_config, profile_id)
    config["strategy"]["top_k"] = top_k
    st.caption(profiles[profile_id].get("description", ""))
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
            st.success(f"连接正常。账户可见模型：{', '.join(status['models']) or '未返回列表'}")
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
    visible_strategies = [item for item in suite["strategies"]
                          if float(item["metrics"].get("cumulative_return", 0)) > 0]
    st.caption(f"仅显示该批次累计收益大于 0 的 {len(visible_strategies)} 套策略；亏损结果保留在本地审计文件。")
    if not visible_strategies:
        st.info("该批次没有累计收益为正的策略。")
        return
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
    sweep_path = folder / suite.get("take_profit_sweep_file", "take_profit_sweep.csv")
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        positive_ids = {item["strategy_id"] for item in visible_strategies}
        sweep = sweep[sweep["strategy_id"].isin(positive_ids) & sweep["cumulative_return"].gt(0)]
        if not sweep.empty:
            st.subheader("盈利策略的止盈参数对照")
            best = sweep.sort_values("cumulative_return", ascending=False)[[
                "strategy_name", "threshold_label", "cumulative_return", "max_drawdown", "profitable_trade_rate",
            ]].rename(columns={"strategy_name": "策略", "threshold_label": "止盈参数",
                              "cumulative_return": "累计收益", "max_drawdown": "最大回撤",
                              "profitable_trade_rate": "交易胜率"})
            st.dataframe(best.style.format({"累计收益": "{:.2%}", "最大回撤": "{:.2%}",
                                            "交易胜率": "{:.2%}"}), width="stretch", hide_index=True)
    control, _ = st.columns([3, 5])
    selected = control.selectbox("查看策略规则", visible_strategies, format_func=lambda value: value["name"])
    threshold = selected["metrics"].get("take_profit_threshold")
    threshold_note = (
        f"该策略在盈利达到{threshold:.0%}时记录完整事件并止盈。"
        if selected["metrics"].get("threshold_enabled", True) and threshold is not None
        else "该策略不应用固定止盈规则。"
    )
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
            "- **固定止盈**：持仓期间最高价首次达到设定阈值时记录并模拟卖出；当前普通策略为20%，半仓策略为30%。\n"
            "- **总交易次数**：已经完成买入和卖出的完整交易数；期末仍持有的仓位不计入。\n"
            "- **盈利/亏损次数**：按扣除买卖成本后的单笔净收益大于0或小于0统计，等于0单列为持平。\n"
            "- **交易胜率**：盈利交易次数÷总已平仓交易次数；持有中的仓位不计入。\n"
            "- **换手率**：越低越节省成本，但过低也可能反应迟钝；需要结合超额收益判断。以上区间都是研究经验值，不是保证。"
        )


def render_backtest() -> None:
    st.title("策略分析")
    folders = _strategy_result_folders()
    if not folders:
        st.info("当前没有累计收益为正的回测结果。亏损交易仍可在‘亏损诊断’查看。")
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


def render_strategy_lab() -> None:
    st.title("策略试验场")
    st.caption(
        "围绕当前最佳规则做参数消融：连续3次进入 Top20/Top25，组合不同退出排名、"
        "卖出确认次数和固定止盈。全部结果均为同一区间的样本内比较。"
    )
    config = load_config(PROJECT_ROOT / "config.yaml")
    pool_config = {**config, "strategy": {**config["strategy"], "top_k": 50}}
    pool_key = strategy_key(pool_config)[:8]
    batches = _suite_folders()
    latest_batch = batches[0].name if batches else ""
    folders = sorted({path.parent for path in STRATEGY_LAB_ROOT.glob("*/progress.json")
                      if path.parent.name.startswith(f"{latest_batch}_")
                      and f"pool-{pool_key}" in path.parent.name}, reverse=True)
    if not folders:
        st.info("当前可交易股票池还没有匹配的试验场结果；旧股票池的试验已归档，不参与今日推荐。"
                "如需重测，运行 python scripts/run_strategy_lab.py。")
        return
    control, _ = st.columns([2.2, 5.8])
    folder = control.selectbox("试验批次", folders, format_func=lambda value: value.name)
    progress = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
    completed = int(progress.get("completed", 0))
    total = int(progress.get("total", 0))
    ratio = completed / total if total else 0.0
    status_names = {"running": "运行中", "completed": "已完成", "failed": "失败"}
    st.progress(ratio, text=f"{status_names.get(progress.get('status'), progress.get('status'))}：{completed}/{total}")
    columns = st.columns(4)
    columns[0].metric("已完成组合", completed)
    columns[1].metric("总组合数", total)
    columns[2].metric("完成比例", f"{ratio:.1%}")
    elapsed = float(progress.get("elapsed_seconds", 0))
    eta = elapsed / completed * (total - completed) if completed else 0
    columns[3].metric("预计剩余", f"{eta / 60:.1f} 分钟" if progress.get("status") == "running" else "0 分钟")
    st.caption(
        f"区间：{progress.get('start_date')} 至 {progress.get('end_date')}｜"
        f"最后更新：{progress.get('updated_at')}｜当前组合：{progress.get('current_experiment') or '无'}"
    )
    if progress.get("status") == "failed":
        st.error(progress.get("error", "试验进程失败"))
    if st.button("刷新试验进度"):
        st.rerun()
    results_path = folder / progress.get("results_file", "results.csv")
    if not results_path.exists():
        st.info("首个参数组合尚未完成。")
        return
    results = pd.read_csv(results_path)
    results = results[results["cumulative_return"].gt(0)].sort_values("cumulative_return", ascending=False).head(20).copy()
    if results.empty:
        st.info("已完成组合中暂无累计收益为正的策略。")
        return
    results.insert(0, "排名", range(1, len(results) + 1))
    display = _integerize_counts(results.rename(columns={
        "strategy_name": "试验规则", "entry_rank": "连续入选范围", "exit_rank": "跌出范围",
        "confirmation_days": "卖出确认次数", "take_profit": "止盈阈值",
        "cumulative_return": "累计收益", "annualized_return": "年化收益",
        "benchmark_return": "基准收益", "excess_return": "超额收益",
        "max_drawdown": "最大回撤",
        "annualized_volatility": "年化波动", "total_trade_count": "交易次数",
        "profitable_trade_rate": "交易胜率", "average_holding_days": "平均持有交易日",
        "take_profit_count": "止盈次数", "open_positions": "期末持仓",
        "runtime_seconds": "计算耗时（秒）",
    }))
    columns_to_show = [
        "排名", "试验规则", "连续入选范围", "跌出范围", "卖出确认次数", "止盈阈值",
        "累计收益", "年化收益", "超额收益", "最大回撤", "年化波动",
        "交易次数", "交易胜率", "平均持有交易日", "止盈次数",
    ]
    st.subheader("当前盈利组合前二十")
    st.dataframe(
        display[columns_to_show].style.format({
            "止盈阈值": "{:.0%}", "累计收益": "{:.2%}", "年化收益": "{:.2%}",
            "超额收益": "{:.2%}", "最大回撤": "{:.2%}",
            "年化波动": "{:.2%}", "交易胜率": "{:.2%}", "平均持有交易日": "{:.1f}",
        }), width="stretch", hide_index=True, height=760,
    )
    chart_data = results[["strategy_name", "cumulative_return"]].copy()
    lab_chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("cumulative_return:Q", title="累计收益", axis=alt.Axis(format=".0%", labelAngle=0)),
        y=alt.Y("strategy_name:N", title=None, sort="-x", axis=alt.Axis(labelAngle=0)),
        tooltip=["strategy_name:N", alt.Tooltip("cumulative_return:Q", format=".2%")],
    ).properties(height=max(360, len(chart_data) * 30))
    st.altair_chart(lab_chart, width="stretch")
    st.warning(
        "前二十名是在同一段历史数据上从大量组合中筛出的样本内结果，存在明显的数据挖掘和参数过拟合风险。"
        "最终参数必须用未参与本次排名的后续数据做样本外验证。"
    )


def render_signal_alerts() -> None:
    st.title("交易信号提醒")
    st.caption("仅汇总历史累计收益为正的策略；新买入候选已排除科创板和创业板。页面只给出研究提醒，不会自动下单。")
    st.info("旧八策略联合收益来自过期股票池，已从日常页面隐藏；试验场旧结果也不会参与当前联合信号。"
            "仅当新批次策略累计收益为正时才生成联合买入提醒。")
    config = load_config(PROJECT_ROOT / "config.yaml")
    dates = _selection_dates()
    if not dates:
        st.info("尚无本地历史行情。请先运行：python scripts/run_today.py")
        return
    date_col, _ = st.columns([1.7, 6.3])
    requested_date = date_col.date_input(
        "选择信号日期", value=date.today(), min_value=dates[0],
        max_value=max(date.today(), dates[-1]), format="YYYY-MM-DD",
    )
    available = [value for value in dates if value <= requested_date]
    selected_date = available[-1] if available else dates[0]
    if selected_date != requested_date:
        st.info(f"{requested_date} 不是本地交易日，已使用最近交易日 {selected_date}。")
    continuation, continuation_events = top20_streak2_leader_probability(
        config, selected_date,
    )
    st.subheader("连续两次Top20领跑者的次日延续概率")
    probability_cols = st.columns(4)
    win_rate = continuation.get("win_rate")
    probability_cols[0].metric("历史延续胜率", f"{win_rate:.1%}" if win_rate is not None else "样本不足")
    probability_cols[1].metric("成功次数", int(continuation.get("successes", 0)))
    probability_cols[2].metric("历史样本数", int(continuation.get("trials", 0)))
    current_candidate = continuation.get("current_candidate")
    probability_cols[3].metric(
        "当日符合标的",
        (
            f"{current_candidate.get('name') or current_candidate['ts_code']}（第{current_candidate['rank']}名）"
            if current_candidate else "无"
        ),
    )
    st.caption(
        "统计口径：每天在“截至当天恰好连续2次进入Top20”的股票中，选择当天排名最靠前的一只，"
        "检查它下一交易日是否仍在Top20；只有已经出现下一交易日的样本才计入胜率。"
        f"历史范围：{continuation.get('sample_start') or '—'} 至 {continuation.get('sample_end') or '—'}。"
        "该概率只表示标的次日继续位于Top20，不代表次日上涨或交易盈利。"
    )
    if continuation["cached_trading_dates"] < continuation["expected_trading_dates"]:
        st.warning(
            f"Top50历史缓存覆盖 {continuation['cached_trading_dates']}/"
            f"{continuation['expected_trading_dates']} 个交易日，当前概率只基于完整可识别样本。"
        )
    if not continuation_events.empty:
        with st.expander("查看最近20次历史样本"):
            history_display = continuation_events.tail(20).sort_values("signal_date", ascending=False).rename(columns={
                "signal_date": "连续2次确认日", "next_trade_date": "下一交易日",
                "ts_code": "股票代码", "name": "证券简称", "signal_rank": "确认日排名",
                "next_rank": "次日排名", "success": "是否延续至第3次",
            })
            history_display["是否延续至第3次"] = history_display["是否延续至第3次"].map(
                {True: "成功", False: "失败"},
            )
            st.dataframe(history_display, width="stretch", hide_index=True)
    with st.spinner("正在计算盈利策略联合信号……"):
        joint, metadata = combined_signal_recommendations(config, selected_date)
    definitions = pd.DataFrame(metadata.get("strategies", []))
    if not definitions.empty:
        rules = definitions.rename(columns={
            "source": "来源", "name": "策略", "entry_rank": "连续入选范围",
            "consecutive_days": "入选确认次数", "exit_rank": "跌出范围",
            "confirmation_days": "卖出确认次数", "take_profit": "止盈阈值",
            "historical_return": "历史累计收益",
        })
        with st.expander(f"查看{len(definitions)}套保留策略规则", expanded=False):
            st.dataframe(
                rules[["来源", "策略", "连续入选范围", "入选确认次数", "跌出范围", "卖出确认次数", "止盈阈值", "历史累计收益"]]
                .style.format({"止盈阈值": "{:.0%}", "历史累计收益": "{:.2%}"}),
                width="stretch", hide_index=True,
            )
    if joint.empty:
        st.warning(metadata.get("reason", "所选日期没有联合推荐。"))
        return
    history_stats, _ = recommendation_history_stats(config, selected_date)
    joint["consecutive_top20"] = joint["ts_code"].map(
        lambda code: history_stats.get(code, {}).get("consecutive_top20", 0)
    )
    st.subheader(f"{len(definitions)}策略联合推荐")
    joint_display = joint.rename(columns={
        "joint_rank": "联合排序", "ts_code": "股票代码", "name": "证券简称",
        "rank": "当日总排名", "total_score": "综合得分", "close": "收盘价",
        "consecutive_entry_days": "连续满足天数", "strategy_support_count": "策略支持数",
        "consecutive_top20": "连续Top-20（天）",
        "supporting_strategies": "支持策略", "best_supporting_return": "支持策略最佳历史收益",
    })
    joint_display = _integerize_counts(joint_display)
    joint_style = joint_display.style.format({
            "综合得分": "{:.3f}", "收盘价": "{:.2f}", "支持策略最佳历史收益": "{:.2%}",
        }).apply(
            lambda row: [
                (
                    "color:#7e22ce;font-weight:700" if row["连续Top-20（天）"] >= 3
                    else "color:#e02020;font-weight:700" if row["连续Top-20（天）"] == 2
                    else ""
                ) if column == "证券简称" else ""
                for column in row.index
            ], axis=1,
        )
    st.dataframe(joint_style, width="stretch", hide_index=True, height=430)
    st.caption(
        "联合排序依次考虑策略支持数、当日总排名和综合得分。多套策略共享相近的入场逻辑，"
        "支持数不是独立模型投票，也不是上涨概率。"
    )

    account = _account_store()
    holdings = account.list_holdings()
    choices = {
        row["ts_code"]: f"{row['ts_code']}｜{row['name']}｜{int(row['strategy_support_count'])}个策略支持"
        for _, row in joint.iterrows()
    }
    if not holdings.empty:
        for _, row in holdings.iterrows():
            choices.setdefault(row["ts_code"], f"{row['ts_code']}｜{row['name']}｜账户持仓监控")
    selectable_codes = list(joint["ts_code"])
    selectable_codes.extend(code for code in choices if code not in selectable_codes)
    basic_names, spellings = _basic_search_maps(config)
    search_names = {
        code: choices[code].split("｜", 1)[1] if "｜" in choices[code] else basic_names.get(code, "")
        for code in selectable_codes
    }
    selected_code = _stock_search(
        "选择联合推荐或账户持仓", selectable_codes, search_names, spellings,
        "signal_stock",
    )
    selected_rows = joint[joint["ts_code"].eq(selected_code)]
    store = build_store(config)
    if selected_rows.empty:
        market_row = store.load_daily()
        market_row = market_row[
            market_row["ts_code"].eq(selected_code) &
            market_row["trade_date"].dt.date.le(selected_date)
        ].sort_values("trade_date").iloc[-1]
        selected_stock = market_row
    else:
        selected_stock = selected_rows.iloc[0]
    holding = account.get(selected_code)
    held = holding is not None
    st.info("账户状态：已持有（持仓信息已持久化）" if held else "账户状态：未持有")
    input_left, input_right = st.columns(2)
    entry_price = float(input_left.number_input(
        "实际买入价（元）" if held else "假设买入价（元）",
        min_value=0.01,
        value=float(holding["cost_price"] if held else selected_stock["close"]), step=0.01,
        help="已持有时请填写真实含义上的持仓成本；未持有时默认用当日收盘价估算未来止盈/止损线。",
    ))
    price_stop_loss = float(input_right.number_input(
        "辅助价格止损幅度（%）", min_value=1.0, max_value=50.0, value=10.0, step=1.0,
        help="这是额外价格预警，没有纳入上述八套策略的历史回测。",
    )) / 100
    reminders, market = stock_signal_reminders(
        config, selected_date, selected_code, entry_price, price_stop_loss,
    )
    if not held:
        reminders["signal"] = reminders["entry_condition"].map(
            {True: "买入条件满足", False: "不满足买入条件"}
        )
    take_profit_count = int(reminders["take_profit_met"].sum())
    rank_exit_count = int(reminders["rank_exit_met"].sum())
    entry_count = int(reminders["entry_condition"].sum())
    cards = st.columns(5)
    cards[0].metric("当日收盘价", f"{market['close']:.2f} 元")
    cards[1].metric("相对成本收益", _percent(market["price_change_from_entry"]))
    cards[2].metric("满足买入策略", f"{entry_count}/8")
    cards[3].metric("触发止盈策略", f"{take_profit_count}/8")
    cards[4].metric("触发排名退出", f"{rank_exit_count}/8")
    if held:
        if take_profit_count:
            st.error(f"止盈提醒：已有 {take_profit_count} 套策略的止盈线被当日最高价触及。")
        if rank_exit_count:
            st.error(f"退出提醒：已有 {rank_exit_count} 套策略满足排名退出及确认次数，下一交易日尝试卖出。")
        if market["price_stop_met"]:
            st.error(f"辅助价格止损提醒：当日最低价已触及 {market['price_stop']:.2f} 元。")
        if not take_profit_count and not rank_exit_count and not market["price_stop_met"]:
            st.success("当前未触发止盈、排名退出或辅助价格止损提醒。")
    else:
        st.info(f"当前有 {entry_count}/8 套策略满足新开仓条件；止盈和止损价格仅为按假设买入价计算的参考线。")

    detail = reminders.rename(columns={
        "strategy_name": "策略", "source": "来源", "historical_return": "历史累计收益",
        "current_rank": "当前排名", "entry_condition": "满足买入", "entry_streak": "连续入选次数",
        "take_profit_rate": "止盈比例", "take_profit_price": "止盈价",
        "take_profit_met": "止盈已触发", "exit_rank": "跌出范围",
        "required_exit_confirmations": "所需退出确认", "current_exit_streak": "当前连续跌出次数",
        "rank_exit_met": "排名退出已触发", "signal": "当前信号",
    })
    st.subheader("逐策略信号明细")
    detail = _integerize_counts(detail)
    detail_style = detail.style.format({
        "历史累计收益": "{:.2%}", "止盈比例": "{:.0%}", "止盈价": "{:.2f}",
        "当前排名": "{:.0f}", "连续入选次数": "{:.0f}",
        "跌出范围": "{:.0f}", "所需退出确认": "{:.0f}", "当前连续跌出次数": "{:.0f}",
    })
    if held:
        liquidation_signals = {"止盈", "策略退出", "价格止损预警"}
        detail_style = detail_style.apply(
            lambda row: [
                "color:#16a34a;font-weight:700" if row["当前信号"] in liquidation_signals else ""
                for _ in row
            ], axis=1,
        )
    st.dataframe(
        detail_style, width="stretch", hide_index=True, height=520,
    )
    st.caption(
        "排名退出依据所选日期收盘后的 Top50 历史连续判断，实际卖出安排在下一交易日开盘；"
        "若开盘跌停则按回测规则顺延。日内止盈/价格止损用当日最高价和最低价判断，仅适用于你在当日之前已经持仓的情况。"
    )


def render_account() -> None:
    st.title("账户")
    st.caption("持仓保存在本机 SQLite；系统只记录和监控，不会连接券商或自动下单。")
    holiday_status = holiday_risk(date.today())
    if holiday_status and holiday_status["level"] == "warning":
        st.warning(
            f"节前风险提示：距离{holiday_status['holiday_name']}假期还有 "
            f"{holiday_status['days_ahead']} 天。请重点检查持仓流动性、隔夜风险和是否需要降低仓位。"
        )
    elif holiday_status and holiday_status["level"] == "safe":
        st.success("风险日历：未来两天内没有中国法定节假日开始。")
    config = load_config(PROJECT_ROOT / "config.yaml")
    store = build_store(config)
    account = _account_store()
    holdings = account.list_holdings()
    daily = store.load_daily()
    latest = (
        daily.sort_values("trade_date").drop_duplicates("ts_code", keep="last")
        [["ts_code", "close", "trade_date"]] if not daily.empty else pd.DataFrame()
    )
    if not holdings.empty:
        latest_date = daily["trade_date"].max().date()
        signal_rows = []
        for _, holding in holdings.iterrows():
            try:
                reminders, market = stock_signal_reminders(
                    config, latest_date, str(holding["ts_code"]),
                    float(holding["cost_price"]), 0.10,
                )
                take_profit_count = int(reminders["take_profit_met"].sum()) if not reminders.empty else 0
                rank_exit_count = int(reminders["rank_exit_met"].sum()) if not reminders.empty else 0
                stop_met = bool(market["price_stop_met"])
                if take_profit_count or rank_exit_count or stop_met:
                    signal = "检查清仓"
                elif reminders.empty:
                    signal = "暂无有效策略"
                else:
                    signal = "继续观察"
                signal_rows.append({
                    "股票代码": holding["ts_code"], "证券简称": holding["name"],
                    "当前信号": signal, "触发止盈策略": take_profit_count,
                    "触发排名退出策略": rank_exit_count,
                    "辅助10%止损": "已触发" if stop_met else "未触发",
                    "相对成本收益": market["price_change_from_entry"],
                    "信号日期": latest_date,
                })
            except Exception as exc:
                signal_rows.append({
                    "股票代码": holding["ts_code"], "证券简称": holding["name"],
                    "当前信号": "数据不足", "触发止盈策略": 0,
                    "触发排名退出策略": 0, "辅助10%止损": "未知",
                    "相对成本收益": pd.NA, "信号日期": latest_date,
                    "说明": str(exc),
                })
        st.subheader("持仓卖出信号")
        signals = _integerize_counts(pd.DataFrame(signal_rows))
        signal_style = signals.style.format({"相对成本收益": "{:.2%}"}, na_rep="—").apply(
            lambda row: [
                "color:#16a34a;font-weight:700" if row["当前信号"] == "检查清仓" else ""
                for _ in row
            ], axis=1,
        )
        st.dataframe(signal_style, width="stretch", hide_index=True)
        st.caption(
            "“检查清仓”表示至少一套联合策略触发止盈/排名退出，或价格触及辅助10%止损；"
            "排名退出按收盘信号在下一交易日开盘尝试执行，跌停时可能无法卖出。"
        )
        view = holdings.merge(latest, on="ts_code", how="left")
        view["market_value"] = view["quantity"] * view["close"]
        view["cost_value"] = view["quantity"] * view["cost_price"]
        view["profit"] = view["market_value"] - view["cost_value"]
        view["profit_rate"] = view["close"] / view["cost_price"] - 1
        display = view.rename(columns={
            "ts_code": "股票代码", "name": "证券简称", "quantity": "持仓数量",
            "cost_price": "成本价", "entry_date": "买入日期", "close": "最新收盘价",
            "trade_date": "行情日期", "market_value": "市值", "profit": "浮动盈亏",
            "profit_rate": "浮动收益率", "notes": "备注", "updated_at": "最后修改",
        })
        st.dataframe(
            display[["股票代码", "证券简称", "持仓数量", "成本价", "最新收盘价", "浮动收益率", "市值", "浮动盈亏", "买入日期", "行情日期", "备注"]]
            .style.format({"成本价": "{:.3f}", "最新收盘价": "{:.3f}", "浮动收益率": "{:.2%}", "市值": "{:,.2f}", "浮动盈亏": "{:,.2f}"}),
            width="stretch", hide_index=True,
        )
    else:
        st.info("当前没有持仓。请在下面录入第一笔持仓。")

    basic = store.load_basic()
    basic = basic[basic["list_status"].eq("L")].drop_duplicates("ts_code")
    basic = basic[~is_star_market(basic.ts_code, basic.get("market"))
                  & ~is_chinext_market(basic.ts_code, basic.get("market"))].copy()
    labels = dict(zip(basic["ts_code"], basic["name"], strict=False))
    spellings = (
        dict(zip(basic["ts_code"], basic["cnspell"].fillna(""), strict=False))
        if "cnspell" in basic else {}
    )
    st.subheader("新增或修改持仓")
    st.caption("录入选项已排除当前账户不能买入的科创板和创业板；原有持仓记录不会被删除。")
    code = _stock_search(
        "股票", basic["ts_code"].tolist(), labels, spellings, "account_stock",
    )
    with st.form("holding_form"):
        left, middle, right, _ = st.columns([1.2, 1.2, 1.5, 4.1])
        quantity = left.number_input("持仓数量（股）", min_value=1.0, value=100.0, step=100.0)
        cost_price = middle.number_input("持仓成本价（元）", min_value=0.001, value=10.0, step=0.01)
        entry_date = right.date_input("买入日期", value=date.today())
        notes = st.text_input("备注（可选）")
        if st.form_submit_button("保存持仓", type="primary"):
            account.upsert(code, labels.get(code, ""), quantity, cost_price, entry_date, notes)
            st.rerun()
    if not holdings.empty:
        holding_names = holdings.set_index("ts_code")["name"].to_dict()
        remove_code = _stock_search(
            "清仓并移除", holdings["ts_code"].tolist(), holding_names, spellings,
            "remove_holding",
        )
        if st.button("确认移除该持仓"):
            account.delete(remove_code)
            st.rerun()


def render_strategy_hub() -> None:
    st.info("当前新选股与新回测会排除科创板和创业板；旧批次不代表你的可交易股票池。"
            "亏损诊断仍保留原始记录用于追查价格口径。")
    section = st.sidebar.radio(
        "策略研究分类", ["策略总览", "策略分析", "策略试验场", "亏损诊断"],
        help="四个分类共用同一个网址，切换时只加载当前分类。",
    )
    if section == "策略总览":
        render_strategy_overview()
    elif section == "策略分析":
        render_backtest()
    elif section == "策略试验场":
        render_strategy_lab()
    else:
        render_loss_audit()


def render_loss_audit() -> None:
    st.title("各策略亏损诊断")
    root = BACKTESTS_ROOT / "loss_audit"
    detail_path, summary_path = root / "worst_10_by_strategy.csv", root / "strategy_summary.csv"
    if not detail_path.exists() or not summary_path.exists():
        st.info("尚未生成审计结果。运行 conda run -n jkquant python scripts/audit_worst_trades.py。")
        return
    detail = pd.read_csv(detail_path)
    summary = pd.read_csv(summary_path)
    st.warning("旧批次策略套件以未复权买卖价计算收益：除权、送转或分红日可能出现假亏损；"
               "新批次套件和研究引擎采用再投资代理。‘回测遗漏百分点’比较各引擎自身价格口径，"
               "‘除权影响百分点’仅表示原始价差变化，不等于遗漏；代理仍非真实分红入账。")
    st.dataframe(summary.drop(columns=["来源"], errors="ignore").style.format({
        "胜率": "{:.1%}", "最差单笔净收益": "{:.1%}",
    }), hide_index=True, width="stretch")
    selected = st.selectbox("查看策略最差十笔交易", summary["策略"].tolist(), key="loss_audit_strategy")
    trades = detail[detail["策略"].eq(selected)].drop(columns=["来源", "结果类型"], errors="ignore")
    st.dataframe(trades.style.format({"原始价差": "{:.2%}", "回测口径价差": "{:.2%}",
                                      "回测净收益": "{:.2%}", "除权参考价差": "{:.2%}",
                                      "除权影响百分点": "{:.2f}", "回测遗漏百分点": "{:.2f}",
                                      "持有期最高浮盈": "{:.2%}", "持有期最大浮亏": "{:.2%}"}),
                 hide_index=True, width="stretch")
    st.caption("仅列已平仓且亏损的交易，按原回测净收益从低到高排序。未生成逐股开平仓账本的"
               "三日调仓基线与八策略组合不在此表；新旧引擎成本和持仓口径也不可直接对比。")


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


def render_system_help() -> None:
    st.title("系统说明")
    st.subheader("系统现在如何运行")
    st.markdown(
        "1. **更新数据**：从 Tushare 增量获取股票列表、日/周/月线、每日估值、涨跌停、宏观数据和 510300 ETF，写入本地 Parquet。\n"
        "2. **更新财务**：利润表、资产负债表和现金流量表按股票断点续传，独立于每日选股任务。\n"
        "3. **计算因子**：每只股票只使用当日及之前的数据计算 8 个量价因子。\n"
        "4. **过滤股票**：排除科创板、创业板、当前 ST、成交额不足、零成交和历史记录太短的股票。\n"
        "5. **横截面打分**：把当日每个因子转成 0～1 的市场百分位得分。\n"
        "6. **生成候选**：按综合得分从高到低输出 Top-K，供人工继续研究。\n"
        "7. **历史回测**：使用前一交易日信号，在下一交易日执行；佣金万分之五且每笔最低5元，卖出另计印花税，并计入滑点。"
    )
    st.subheader("当前量价策略")
    st.markdown(
        "- **动量 40%**：5日、20日收益较强得分更高。\n"
        "- **趋势 25%**：价格高于20日均线、5日均线高于20日均线得分更高。\n"
        "- **低风险 20%**：20日波动和回撤越小得分越高。\n"
        "- **流动性 15%**：平均成交额和近期成交活跃度较高得分更高。\n\n"
        "这是原固定权重，并不是 AI 预测模型。每日候选还提供独立的‘近期低波动防守’对照："
        "动量10%、趋势10%、低风险70%、流动性10%，其中低风险类别内波动率占90%。"
        "该比例来自最近252交易日诊断，只用于对照，尚不能视为未来最优。"
    )
    st.subheader("回测中的‘收益’从哪里来")
    st.markdown(
        "回测把历史每个调仓日当作当时正在运行系统：只用该日以前的数据选股，在下一交易日按开盘价模拟换仓，"
        "之后按持仓股票的实际历史涨跌计算组合每日盈亏。**净收益**是在股票涨跌形成的毛收益上再扣除模拟交易成本；"
        "逐日复合后得到累计收益和净值。因此它衡量的是‘过去机械执行当前规则会怎样’，不是未来收益承诺。"
    )
    st.subheader("仍需补齐的关键模块")
    st.markdown(
        "- 基于诊断结果做滚动样本外因子权重校准，避免在同一区间反复挑参数。\n"
        "- 盘中开板排队、停牌延续、100股整数手和实际冲击成本等更精细的成交约束。\n"
        "- 更严格的历史 ST、退市、名称和指数成分时点数据。\n"
        "- 完整复权价格和更多可比较指数基准。\n"
        "- 行业、市值暴露约束，以及自定义/指数股票池。\n"
        "- 对新增估值、质量和成长因子做 IC、分层收益与样本外验证后再纳入评分。"
    )
    st.warning("本系统输出的是量化候选，不构成投资建议，也不会自动下单。")


def main() -> None:
    st.set_page_config(page_title="JKQuant", page_icon="📈", layout="wide")
    title, refresh = st.columns([9, 1])
    title.markdown("## JKQuant 选股系统")
    if refresh.button("刷新"):
        st.rerun()
    navigation = st.navigation([
        st.Page(render_topk, title="每日候选", icon="📋", url_path="candidates", default=True),
        st.Page(render_ai_analysis, title="AI分析", icon="🤖", url_path="ai-analysis"),
        st.Page(render_signal_alerts, title="交易信号提醒", icon="🔔", url_path="signals"),
        st.Page(render_account, title="账户", icon="💼", url_path="account"),
        st.Page(render_strategy_hub, title="策略研究", icon="📊", url_path="strategy-research"),
        st.Page(render_market_overview, title="市场概览", icon="🌐", url_path="market"),
        st.Page(render_limit_up, title="打板观察", icon="🚀", url_path="limit-up"),
        st.Page(render_system_help, title="系统说明", icon="ℹ️", url_path="help"),
    ], position="top")
    st.divider()
    navigation.run()


if __name__ == "__main__":
    main()
