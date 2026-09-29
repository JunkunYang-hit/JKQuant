from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from jkquant.config import load_config
from jkquant.data.realtime_provider import RealtimeQuoteError, build_realtime_provider


def main() -> None:
    parser = argparse.ArgumentParser(description="测试通达信 TdxQuant 本地实时行情连接")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--codes", nargs="+", default=["600000.SH", "000001.SZ"],
        help="测试证券代码，必须带 .SH/.SZ 后缀",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config.get("intraday", {}).get("tdxquant", {})
    endpoint = str(settings.get("base_url", "http://127.0.0.1:17709/"))
    print(f"正在连接：{endpoint}")
    try:
        frame = build_realtime_provider(config, "tdxquant").snapshot(args.codes)
    except RealtimeQuoteError as exc:
        print(f"连接失败：{exc}")
        print("请启动并登录支持 TQ 的通达信金融终端/量化模拟版，然后保持客户端运行。")
        raise SystemExit(1) from exc
    if frame.empty or "close" not in frame or frame["close"].isna().all():
        print("服务已响应，但没有返回有效行情；请检查证券代码、登录状态和行情权限。")
        raise SystemExit(2)
    columns = [
        column for column in (
            "ts_code", "close", "pre_close", "open", "high", "low", "vol", "amount",
            "trade_time", "source", "fetched_at",
        ) if column in frame
    ]
    print("连接成功：")
    print(frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
