# JKQuant

JKQuant 是一个面向实际选股研究的 A 股日频系统。它负责更新本地数据、计算多因子 Top-K、展示个股与市场信息、调用 DeepSeek 做辅助分析，并对规则策略进行统一历史模拟。

系统不会连接券商或自动下单，页面输出均是研究结果，不构成投资建议。

## 快速开始

```powershell
conda env create -f environment.yml
conda activate jkquant
Copy-Item .env.example .env
```

在 `.env` 中填写密钥，不要把真实密钥提交到 Git：

```dotenv
TUSHARE_TOKEN=你的_Tushare_Token
DEEPSEEK_API_KEY=你的_DeepSeek_API_Key
```

每天使用下面一个命令即可更新数据、生成最新推荐并自动打开浏览器：

```powershell
python scripts/run_today.py
```

只启动本地界面：

```powershell
python scripts/run_webui.py
```

## 页面功能

- 每日候选：按日期读取或计算 Top-K，支持 1～50 只、分页、排序、连续上榜标色和节前风险提示。
- AI 分析：选择 DeepSeek 模型，将当日 Top-20、量价因子和用户补充问题一并提交，结果缓存在本地。
- 小盘备选池：只从无需另开板块权限的沪深主板中，筛选收盘价 3～25 元、低位或长期未明显上涨、横盘整理的小市值股票；盘中按交易进度折算量价速度，分为橙色“可能启动”和红色“重点观察”。
- 策略研究：展示五套Top-K规则与两套小盘重点观察模拟，可查看净值、回撤、逐日结果和交易区间。
- 市场概览：展示本地市场数据的概览信息。
- 打板观察：按日线识别首板、二连板和更高连板，分析行业热度与历史晋级特征。

## 当前七套策略

“前五”以最新完整回测批次 `2025-09-01_2026-09-16` 的累计收益排序确定。即使它们仍为负收益，也只是相对排名最高，不代表已经达到实盘标准。

1. `s04_top50_streak2_confirm2`：连续两次进入 Top-50；连续两次跌出 Top-50 后卖出。
2. `s04_top50_streak2`：连续两次进入 Top-50；首次跌出 Top-50 后卖出。
3. `s14_top20_exact2_leader_full`：从恰好连续两次进入 Top-20 的股票中买入排名最高者；未完成第三次入选、之后跌出 Top-20，或盈利 32% 时卖出。
4. `s03_top50_streak3_confirm2`：连续三次进入 Top-50；连续两次跌出 Top-50 后卖出。
5. `s06_top20_streak2_confirm2`：连续两次进入 Top-20；连续两次跌出 Top-20 后卖出。
6. `smallcap_focus_top1`：重点观察信号确认后的下一交易日，选择异动最强的一只，1万元模拟本金、25%止盈、10%止损；连续两日未上涨则下一交易日开盘退出，法定长假前清仓。
7. `smallcap_focus_top2`：规则同上，但最多持有两只，每只目标约占权益的一半。

普通策略默认使用 20% 止盈；信号在收盘后形成，下一交易日开盘执行。回测计入买卖佣金万分之五、每笔最低 5 元、卖出印花税和滑点，并处理开盘涨停无法买入、开盘跌停无法卖出的情况。ST、科创板、创业板和北交所按当前配置排除。

小盘“红色重点观察”还要求当日累计成交量至少达到上一交易日全天成交量；橙色“可能启动”不受这条硬条件限制。小盘历史模拟使用日线，因此无法知道同一天内最高价和最低价的真实先后顺序；若止盈和止损在同一天同时触发，统一按更保守的止损优先处理。连续未上涨条件在第二日收盘确认，下一交易日开盘退出。科创板、创业板和北交所均排除。

运行五策略回测：

```powershell
python scripts/run_strategy_suite.py
```

结果写入 `backtests/strategy_suite/开始日期_结束日期/`。每套策略包含：

- `metrics.json`：累计收益、年化收益、沪深300ETF基准收益、最大回撤、交易次数和胜率等。
- `daily.csv`：逐日净值、基准净值、回撤、现金和持仓数。
- `trades.csv`：每笔买入、卖出、持有区间和最终收益。
- `threshold_events.csv`：止盈阈值触发与最终交易结果。

沪深300ETF `510300.SH` 是独立市场基准，用来回答策略是否跑赢同期市场，不是第六套策略。

## 选股逻辑

每日候选使用四类量价因子进行横截面百分位打分：

- 动量 40%：5 日和 20 日收益。
- 趋势 25%：价格相对 20 日均线、5 日均线相对 20 日均线。
- 低风险 20%：20 日波动率和当前回撤，数值越低得分越高。
- 流动性 15%：20 日平均成交额和近期成交额活跃度。

该分数是规则型排序，不是上涨概率。Top-K 会统一计算并缓存 Top-50，页面只截取需要的前 K，因此修改 K 不必重复计算因子。

## 数据与缓存

默认使用 Tushare 的股票列表、日/周/月线、每日估值、涨跌停、财务报表、宏观数据和沪深300ETF基准。AKShare 用于补充适合公开接口的数据。Tushare 频率和单次返回量仍以账号权限及官方接口说明为准。

主要本地数据：

- `data/cache/tushare/*.parquet`：行情、基础数据、估值、涨跌停和补充数据。
- `selection_results.sqlite3`：按策略配置哈希和交易日缓存 Top-50。
- `ai_analysis.sqlite3`：DeepSeek 分析缓存。

这些数据默认被 `.gitignore` 排除。修改因子权重或市场过滤条件后，缓存键会变化，不会误用旧配置结果。

独立更新命令：

```powershell
python scripts/update_data.py
python scripts/update_fundamentals.py
python scripts/warmup_recommendations.py
```

## 实时行情

`config.yaml` 的 `intraday` 节点控制盘中行情。当前支持：

- TdxQuant：连接本机已经登录的通达信 TQ 服务，适合个股快照。
- Tushare 实时接口：是否可用取决于账号权限和接口状态。

实时数据只用于观察和重点池监控，不会覆盖日频 Top-K，也不会生成、提交或自动触发任何交易指令。盘中提醒可能随未收盘行情减弱或消失。

通达信本地客户端启动并登录后，可先测试连接：

```powershell
python scripts/test_tdx_realtime.py
```

连接成功后，在“小盘备选池 → 盘中启动雷达”选择“通达信 TdxQuant”，即可手动刷新或每30秒自动刷新。系统用已过去的交易分钟估算全天成交速度，避免上午直接拿累计量与历史整日均量比较而造成提示过晚。

## AI 分析

先测试 DeepSeek 连通性：

```powershell
python scripts/run_ai_analysis.py --test
```

生成最新 Top-20 分析：

```powershell
python scripts/run_ai_analysis.py
```

模型名、超时、最大输出长度等位于 `config.yaml` 的 `ai` 节点。AI 只做结构化复核和风险提示，不应替代行情数据、回测证据或人工判断。

## 项目结构

```text
JKQuant/
├── config.yaml                     # 数据源、过滤条件、因子、回测成本、AI 与盘中行情配置
├── environment.yml                 # Conda 环境入口
├── pyproject.toml                  # Python 包元数据、依赖和命令行入口
├── .env.example                    # 密钥示例；真实 .env 不提交
├── jkquant/
│   ├── cli.py                      # update/daily 命令行入口
│   ├── config.py                   # 配置读取与校验
│   ├── pipeline.py                 # 数据更新、Top-K 缓存和五策略回测编排
│   ├── factors.py                  # 八个量价因子计算
│   ├── strategy.py                 # 股票过滤、评分和 Top-K 选择
│   ├── report.py                   # 每日候选 CSV 输出
│   ├── webui.py                    # Streamlit 多页面界面
│   ├── ai/                         # DeepSeek 请求、提示词、解析和缓存
│   ├── low_position_pool.py        # 小盘低位/横盘/放量大阳线筛选
│   ├── limit_up.py                 # 连板识别、板块热度与晋级统计
│   ├── holiday_risk.py             # 中国法定节假日前风险提示
│   ├── backtest/
│   │   ├── strategy_suite.py       # 五策略定义、事件驱动成交与结果写入
│   │   ├── metrics.py              # 收益、回撤、波动和胜率指标
│   │   ├── benchmark.py            # 510300 基准收益处理
│   │   └── trading_rules.py        # 手续费、滑点和涨跌停成交约束
│   └── data/
│       ├── tushare_provider.py     # Tushare 请求、限频、重试和超时
│       ├── updater.py              # 低频行情与市场数据增量更新
│       ├── fundamental_updater.py  # 财务报表断点续传
│       ├── akshare_provider.py     # AKShare 补充元数据
│       ├── realtime_provider.py    # 实时行情统一接口
│       ├── tdxquant_provider.py    # 本机 TdxQuant 客户端
│       ├── selection_cache.py      # Top-50 SQLite 缓存
│       └── storage.py              # Parquet 数据存储
├── scripts/
│   ├── run_today.py                # 每日更新、推荐并启动浏览器
│   ├── run_webui.py                # 只启动 WebUI
│   ├── run_daily.py                # 只生成每日推荐
│   ├── run_strategy_suite.py       # 运行五策略回测
│   ├── run_ai_analysis.py          # DeepSeek 连通性测试与分析
│   ├── test_tdx_realtime.py        # 测试通达信本地实时行情连接
│   ├── update_data.py              # 更新低频与市场数据
│   ├── update_fundamentals.py      # 更新财务数据
│   └── warmup_recommendations.py   # 预计算历史 Top-50
├── tests/                          # 核心因子、缓存、策略、行情和页面依赖测试
├── data/cache/                     # 本地数据与 SQLite，Git 忽略
├── reports/                        # 每日候选输出，Git 忽略
└── backtests/strategy_suite/       # 五策略回测结果，Git 忽略
```

日常调用链是：`scripts → cli/pipeline → data → factors/strategy → report/webui`。回测与每日候选复用同一份 Top-50 历史信号，策略层只负责定义入场、退出、持仓和止盈规则。

## 验证

```powershell
pytest -q
python -m compileall -q jkquant scripts
```

当前策略仍需要持续做滚动样本外验证。历史收益可能受到时点数据、ST 历史状态、复权代理、停牌、实际排队成交和冲击成本等因素影响，不能直接视为可实现收益。
