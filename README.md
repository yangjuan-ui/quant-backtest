# 量化交易回测系统

每日自动运行的量化信号回测系统：动态构建股票/加密货币观察池，用可解释的分层策略生成信号并做动态退出的模拟回测，结果写入 SQLite，结果指标**永远是胜率 + 盈亏比 + 期望值 + 最大回撤一起看**——不单独吹捧胜率。

## 设计哲学

高胜率 ≠ 赚钱。一个胜率 70% 但平均亏损是盈利 3 倍的策略，长期是亏钱的。本系统每笔交易和每日汇总都记录胜率、平均盈亏比、期望值、最大回撤四个指标，缺一不可。

## 架构

```
GitHub Actions (每日 UTC 13:00)
  ├── fetch_top50_crypto.py     CoinGecko 市值前50 → ccxt 映射交易对 → OHLCV
  ├── fetch_top20_us_stocks.py  Wikipedia S&P500 候选池 → yfinance 市值排序取前20
  ├── fetch_ark_holdings.py     ARK 每日持仓 CSV（独立可插拔模块）
  ├── fetch_13f_filings.py      SEC EDGAR 13F 季度持仓（补充参考层，非每日信号）
  ├── strategy.py               状态过滤(Regime) → 趋势/均值回归双策略 → ATR 仓位管理
  ├── backtest.py               动态退出模拟 + 胜率/盈亏比/期望值/最大回撤
  ├── run_daily.py              每日编排脚本（入口）
  ├── database.py               SQLite：trades / run_summary / holdings_13f
  └── db/trading_records.db     永久数据库（每次运行后自动 commit 回仓库）
```

策略（全部可解释、非黑箱）：
- **状态过滤**：ADX(14) > 20 判为趋势市，否则震荡市（另有 ATR 比率备选方法）
- **趋势市**：双均线(20/50)金叉 + ADX > 25 确认 → ATR 止损(2×)/止盈(3×)动态退出，或趋势反转退出
- **震荡市**：RSI < 30 + 触及布林下轨 → 回到布林中轨（均值目标）或 RSI > 55 退出，ATR 止损(1.5×)
- **仓位**：按 ATR 动态 sizing（每笔风险固定比例权益），波动越大仓位越小

## 本地运行

```bash
pip install -r requirements.txt
python run_daily.py --quick   # 冒烟测试：少量标的，跳过 13F
python run_daily.py           # 完整每日运行
```

## 上线到 GitHub（4 步，需手动完成）

1. 在 GitHub 创建新仓库（公开仓库 Actions 免费分钟数无限），把本仓库所有文件放入。
2. 确认 `.github/workflows/daily_run.yml` 已存在（工作流文件已声明 `permissions: contents: write`，一般无需改仓库设置；若 commit 失败，再去 Settings → Actions → General 把 Workflow permissions 设为 Read and write）。
3. 去 Actions 页面手动点 **Run workflow** 跑一次，确认 `db/trading_records.db` 被自动 commit。
4. 之后每天 UTC 13:00（北京时间 21:00，美股开盘前）自动运行。

## 配置说明

- **ARK CSV 链接**：ARK 偶尔会改下载链接。若 `fetch_ark_holdings.py` 报错，按报错提示去 ark-funds.com 找到新的 CSV 链接，更新文件顶部的 `ARK_CSV_URLS` 即可。
- **13F 基金**：在 `fetch_13f_filings.py` 的 `FUND_CIKS` 里添加你想跟踪的基金 CIK（去 SEC EDGAR 查）。注意 13F 有最长 45 天滞后，它是季度参考层，不做每日信号。
- **SEC 联系方式**：EDGAR 要求请求带联系人 UA，把 `fetch_13f_filings.py` 里 `SEC_CONTACT` 换成你自己的邮箱（或设环境变量 `SEC_CONTACT`）。

## 诚实声明（重要）

- 回测是**模拟**：按日线收盘价成交、无滑点精确建模、做多-only、每标的独立等额资金。实盘会有差异。
- 免费数据源有速率限制和偶发失败：每个抓取模块都是"单点失败不拖垮全流程"设计，失败会记日志并继续。
- 本系统生成的是**代码**，定时运行靠你部署到 GitHub Actions 实现；AI 不替你注册账号、建仓库、推送代码。
