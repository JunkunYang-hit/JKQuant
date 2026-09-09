# JKQuant

一个面向实际选股工作的 A 股日频多因子 MVP。它在收盘后更新日线数据，计算 8 个可解释的量价因子，过滤不可用股票，做横截面百分位排名并输出 Top-K CSV。当前版本不训练 AI 模型，也不自动交易。

## 已实现

- Tushare 日线与股票基础信息，Parquet 本地增量缓存
- 可直接运行的确定性样例数据源（仅用于验证安装）
- 动量、趋势、风险、流动性四类共 8 个因子
- ST、退市/暂停上市、上市天数、成交额、零成交过滤
- 因子方向统一、类内加权、类别固定权重、Top-K CSV
- 避免未来函数的滚动计算测试；配置权重自动校验

## 安装（Conda）

```powershell
conda env create -f environment.yml
conda activate jkquant
pip install -e .
```

首次可直接用默认 `demo` 数据验证完整流程：

```powershell
python scripts/run_daily.py
```

结果写入 `reports/YYYY-MM-DD.csv`。样例数据不代表真实证券，不得用于投资判断。

## 使用 Tushare 实盘数据

在 `config.yaml` 中把 `data.provider` 改为 `tushare`，复制凭据模板并填写 Token：

```powershell
Copy-Item .env.example .env
# 用编辑器打开 .env，将占位内容替换为你的真实 Token
python scripts/run_daily.py
```

程序会自动读取项目根目录的 `.env`。该文件已被 `.gitignore` 排除，不会被 Git 提交。也仍然支持只在当前 PowerShell 会话中设置 `$env:TUSHARE_TOKEN = "你的 token"`。

也可以仅更新数据：

```powershell
python scripts/update_data.py
```

每次运行会从缓存最后日期的下一天开始更新。首次下载约两倍 `history_days` 的自然日，以保证滚动窗口有足够交易日。Tushare 接口权限和积分不足时会原样报告接口错误。

## 配置与因子

所有策略参数都集中在 `config.yaml`。类别权重必须合计为 1；因子 `direction: 1` 表示越大越好，`-1` 表示越小越好。当前因子为 5/20 日收益、收盘价相对 MA20、MA5 相对 MA20、20 日波动率、20 日当前回撤、20 日平均成交额、5/20 日成交额比。

筛选和打分使用选股日收盘后已知的数据。Tushare 停牌股票通常没有当日 daily 行，因此自然不进入当日横截面。当前 `all_a` 已实现；指数成分股和自定义股票池留待下一阶段。

## 验证

```powershell
pytest -q
```

## 当前边界

这是需求中的 Phase 1。尚未包含基本面/估值、财报公告日 point-in-time 对齐、历史成分股、回测、HTML/Excel 报告以及涨跌停成交撮合。它适合先稳定产出候选股，不构成投资建议。后续应优先增加带交易成本的历史回测，再扩展 daily_basic 和按公告日对齐的财务因子。
