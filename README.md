# JKQuant

一个面向实际选股工作的 A 股日频多因子 MVP。它在收盘后更新日线数据，计算 8 个可解释的量价因子，过滤不可用股票，做横截面百分位排名并输出 Top-K CSV。当前版本不训练 AI 模型，也不自动交易。

## 已实现

- Tushare 日线、AKShare 当前证券简称，Parquet 本地增量缓存
- 可直接运行的确定性样例数据源（仅用于验证安装）
- 动量、趋势、风险、流动性四类共 8 个因子
- ST、退市/暂停上市、上市天数、成交额、零成交过滤
- 因子方向统一、类内加权、类别固定权重、Top-K CSV
- 避免未来函数的滚动计算测试；配置权重自动校验
- Top-K 等权、定期调仓回测，含买卖佣金、印花税和滑点
- 回测指标、逐日净值、调仓明细以及净值/回撤图

## 项目结构与职责

```text
JKQuant/
├── config.yaml                 # 唯一的运行参数入口：数据、过滤、因子、回测、成本
├── environment.yml            # Conda 环境定义
├── pyproject.toml              # Python 包、依赖和 jkquant 命令定义
├── .env                        # 本机 Tushare Token；被 Git 忽略，不会提交
├── jkquant/                    # 核心 Python 包
│   ├── cli.py                  # update / daily / backtest 命令解析及终端输出
│   ├── config.py               # YAML 加载、路径解析、权重和参数校验
│   ├── pipeline.py             # 编排数据更新、每日选股和历史回测
│   ├── factors.py              # 8 个量价因子的滚动计算，只使用当日及历史数据
│   ├── strategy.py             # 股票过滤、横截面排名、分类得分和 Top-K
│   ├── report.py               # 每日选股 CSV 输出
│   ├── webui.py                # Top-K 和回测结果的本地 Streamlit 页面
│   ├── data/
│   │   ├── provider.py         # 所有数据源必须实现的统一接口
│   │   ├── tushare_provider.py # Tushare 调用、限速、重试、字段标准化
│   │   ├── akshare_provider.py # 低权限时补充当前 A 股代码与证券简称
│   │   ├── demo_provider.py    # 可重复的模拟数据，只用于测试系统能否运行
│   │   ├── storage.py          # Parquet 读取、去重、合并和落盘
│   │   ├── selection_cache.py  # SQLite 历史 Top-K 缓存和策略版本隔离
│   │   └── updater.py          # 增量/回填下载、按月保存、基础信息刷新策略
│   └── backtest/
│       ├── engine.py           # 信号滞后、持仓、调仓、收益和交易成本计算
│       ├── metrics.py          # 收益、Sharpe、回撤、波动率、胜率等指标
│       └── reporting.py        # 回测 CSV、JSON 和净值/回撤图
├── scripts/
│   ├── update_data.py          # 只更新本地行情缓存
│   ├── run_daily.py            # 更新数据并生成当天 Top-K
│   ├── run_today.py            # 每日入口：更新、选股并自动打开 WebUI
│   ├── run_backtest.py         # 更新所需历史数据并执行回测
│   └── run_webui.py            # 启动本地 WebUI 并自动打开浏览器
├── tests/                      # 因子无未来数据、完整流程和回测测试
├── data/cache/<provider>/      # 本地数据缓存；demo/tushare 隔离且不提交
├── reports/                    # 每日选股结果；不提交
└── backtests/                  # 回测结果；不提交
```

日常运行的调用关系是 `scripts → cli → pipeline → data/factors/strategy → report`。回测复用完全相同的因子和选股逻辑，不另外维护一套“回测专用策略”，避免实盘选股与历史验证逻辑不一致。

## 安装（Conda）

```powershell
conda env create -f environment.yml
conda activate jkquant
pip install -e .
```

如需用 `demo` 数据验证安装，先临时把 `config.yaml` 的 `data.provider` 改为 `demo`，再运行：

```powershell
python scripts/run_daily.py
```

结果写入 `reports/YYYY-MM-DD.csv`。样例数据不代表真实证券，不得用于投资判断。

## 使用 Tushare 实盘数据

在 `config.yaml` 中把 `data.provider` 改为 `tushare`，然后在项目根目录的 `.env` 填写 Token：

```powershell
# .env 文件内容：TUSHARE_TOKEN=你的真实Token
python scripts/run_daily.py
```

程序会自动读取项目根目录的 `.env`。该文件已被 `.gitignore` 排除，不会被 Git 提交。也仍然支持只在当前 PowerShell 会话中设置 `$env:TUSHARE_TOKEN = "你的 token"`。

日常使用推荐直接运行下面这一条，它会依次拉取截至今天的最新数据、生成当日推荐，然后启动 WebUI 并自动打开浏览器：

```powershell
python scripts/run_today.py --top-k 20
```

`--top-k` 可设为 1 到 50；不填写时使用 `config.yaml` 中的 `strategy.top_k`。例如生成 Top-50：

```powershell
python scripts/run_today.py --top-k 50
```

WebUI 里也可以重新选择 K，推荐表每页最多显示 10 只。更改 K 后首次计算可能需要数秒，之后同一日期、同一 K 会直接命中 SQLite 缓存。

也可以仅更新数据：

```powershell
python scripts/update_data.py
```

每次运行会从缓存最后日期的下一天开始更新。首次下载约两倍 `history_days` 的自然日，以保证滚动窗口有足够交易日。Tushare 接口权限和积分不足时会原样报告接口错误。

### Tushare 频率与断点续传

Tushare 的限制取决于账户积分，并非所有用户都是同一频率。官方当前权限表列出：120 积分为每分钟 50 次且只能访问未复权日线，2000 积分为每分钟 200 次，5000 积分以上为每分钟 500 次。`daily` 单次最多返回 6000 行，官方说明可按交易日循环提取历史数据。参见 [积分与频次权限表](https://tushare.pro/document/2?doc_id=290) 和 [A 股日线接口](https://tushare.pro/document/1?doc_id=27)。

当前项目按你的 120 积分账户设置为纯 `daily` 模式，每分钟 45 次，为 50 次档预留余量：

```yaml
data:
  access_tier: 120
  fetch_stock_basic: false
  requests_per_minute: 45
  max_retries: 5
  retry_backoff_seconds: 5
  request_timeout_seconds: 15
  basic_refresh_days: 7
```

`daily(trade_date=...)` 一次返回该交易日全市场（本项目实测约 5,550 行）；`daily(ts_code=..., start_date=..., end_date=...)` 返回一个或多个标的一段历史。接口单次上限为 6,000 行。全市场历史应按日期循环，而不是循环五千多只股票。参见官方的 [高效获取数据说明](https://tushare.pro/document/1?doc_id=230)。

下载器会在每次 API 调用前限速，临时错误使用指数退避重试；历史日线每约一个月写入一次 Parquet，所以中断后重新运行会从缓存边界继续。代码不调用 120 积分不可用的 `trade_cal` 和 `stock_basic`：节假日请求会得到空结果，上市时间近似值从日线首次出现日期派生，次新股用有效历史交易记录数过滤。当前证券简称通过 AKShare `stock_info_a_code_name()` 补充并缓存 30 天，因此每日 Top-K 可以显示代码和简称。历史名称及历史 ST 状态仍不能 point-in-time 还原。

### 本地存储为什么同时用 Parquet 和 SQLite

- `daily.parquet` 保存数百万行连续行情，适合 pandas 批量读取和滚动因子计算。
- `selection_results.sqlite3` 保存每个日期只有 Top-K 行的选股结果，并建立“策略哈希、日期、排名”复合主键及日期索引。
- 首次选择一个尚未计算的日期时，会读取约 200 个自然日窗口并计算，通常需要数秒；再次选择同一日期直接读 SQLite，通常低于 0.1 秒。
- 策略权重或过滤配置变化后，策略哈希随之变化，会重新计算而不会误用旧结果。

Tushare 的 MySQL 示例强调历史数据要落地、建立主键和索引，而不是要求个人项目必须部署 MySQL。当前数据规模使用 Parquet + SQLite 更简单；只有在多用户并发、跨机器共享或数据量继续显著增长时才有必要迁移 PostgreSQL/MySQL。参见 [Tushare MySQL 落地示例](https://tushare.pro/document/1?doc_id=231)。

## 配置与因子

所有策略参数都集中在 `config.yaml`。类别权重必须合计为 1；因子 `direction: 1` 表示越大越好，`-1` 表示越小越好。当前因子为 5/20 日收益、收盘价相对 MA20、MA5 相对 MA20、20 日波动率、20 日当前回撤、20 日平均成交额、5/20 日成交额比。

筛选和打分使用选股日收盘后已知的数据。Tushare 停牌股票通常没有当日 daily 行，因此自然不进入当日横截面。当前 `all_a` 已实现；指数成分股和自定义股票池留待下一阶段。

## 验证

```powershell
pytest -q
```

## 历史回测

默认参数位于 `config.yaml` 的 `backtest` 节点。可以直接使用配置日期：

```powershell
python scripts/run_backtest.py
```

也可以覆盖日期：

```powershell
python scripts/run_backtest.py --start 2025-01-01 --end 2025-12-31
```

输出目录为 `backtests/开始日期_结束日期/`：

- `daily.csv`：逐日毛收益、净收益、成本、换手率、净值和回撤
- `rebalances.csv`：每次调仓的信号日、交易日、买卖换手和持仓代码
- `metrics.json`：年化/累计/基准/超额收益、Sharpe、最大回撤、年化波动、胜率及年度收益
- `equity_drawdown.png`：策略与全市场等权基准净值、策略回撤

信号在 T 日收盘后形成，最早于 T+1 开盘交易。调仓日将旧持仓隔夜收益和新持仓开盘至收盘收益分开计算，避免用 T+1 的价格选择 T 日股票。非调仓日使用收盘到收盘收益。当前基准是当日可交易股票的等权收益。

当前回测并非在测试一个未定义的 AI 模型，而是在测试本项目已经写入配置和代码的固定规则：用动量 40%、趋势 25%、低风险 20%、流动性 15% 合成得分，选择前 `backtest.top_k` 只股票，只做多、等权持有，并每 `rebalance_days` 个交易日调仓。回测中的“收益”是按这些历史持仓的真实历史价格变化模拟出的组合盈亏；净收益还会扣除佣金、印花税和滑点。它回答的是“过去机械执行这套规则会怎样”，不代表未来能获得相同收益。

## 本地 WebUI

先生成一次每日结果和回测结果，然后启动：

```powershell
python scripts/run_webui.py
```

程序会在 `127.0.0.1:8501` 启动仅本机可访问的服务，并自动打开浏览器。每日候选页可以选择任一本地可用交易日；首次计算后会自动缓存。页面包括：

- 带股票代码和当前中文简称的 Top-K 总表、得分概览、单只股票分类得分和底层因子
- 连续进入 Top-20 的证券简称标红，悬停显示连续交易日数；统计结果写入 SQLite 缓存
- 最近 5 个交易日可按需加载单只候选股的 1 分钟价格、均价和成交量图
- 累计/年化/基准/超额收益、Sharpe、最大回撤、波动率和胜率
- 策略与全市场等权基准净值、回撤、年度收益、逐日结果和调仓记录
- 系统运行流程、量价策略解释、专业术语和待建设模块

成交额按常见行情软件的阅读习惯显示为“亿元”，收益率、均线偏离、波动率和回撤统一显示为百分比。在启动 WebUI 的终端按 `Ctrl+C` 会关闭服务。界面只读取本地行情、SQLite 结果缓存和回测文件，不会自动下单。

1 分钟分时图来自 AKShare 的东方财富接口，单次只请求当前选择的一只股票。免费接口仅提供最近 5 个交易日且不复权，网络或上游接口失败只会影响分时图，不会阻断日线选股和历史回测。

## 当前边界

当前已完成量价选股 MVP、规则型历史回测和本地 WebUI。120 积分无法使用基本面/估值、财报公告日、复权因子和历史股票元数据。因子计算优先使用 `daily.pct_chg`（基于除权后的昨收）构造连续价格指数，降低分红送转对动量和趋势的影响，但仍不等于完整前/后复权价格。尚未实现严格历史指数成分、ST 状态、HTML/Excel 日报以及涨跌停成交撮合。长期回测结果仍应视为研究验证，不构成投资建议。

## 尚缺模块与建议顺序

要从“可运行候选系统”走向“可用于真实研究”，仍建议按以下顺序补齐：

1. **因子诊断**：IC/RankIC、分层收益、因子相关性、单因子换手和稳定性。当前只能看到组合结果，无法判断究竟哪个因子有效或拖累。
2. **更真实的成交模拟**：涨跌停无法买卖、停牌持仓延续、A 股 100 股整数手、最低佣金、实际调仓价和冲击成本。
3. **历史证券状态**：历史 ST、退市、上市日期和历史名称，解决当前名称映射只代表“现在”的问题。
4. **复权与基准**：完整复权价格，以及沪深300/中证500等真实指数基准；当前 Universe EW 只是全市场等权参考。
5. **股票池与风险约束**：沪深300/中证500/自定义池、单行业上限、个股权重上限、行业和市值中性化。
6. **基本面与估值**：升级到 2000 积分后再接入 `daily_basic`、PE/PB、市值，以及严格按公告日对齐的财务质量和成长因子。
7. **生产运行**：交易日调度、失败通知、数据完整性告警、日志归档、缓存备份和每日结果对比。

在当前真实回测显著跑输基准的情况下，优先级最高的是第 1、2 项，而不是直接扩大 Top-K 或用于真实买入。
