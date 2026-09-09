from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    app_path = project_root / "jkquant" / "webui.py"
    url = "http://127.0.0.1:8501"
    command = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address=127.0.0.1", "--server.port=8501",
        "--server.headless=true", "--browser.gatherUsageStats=false",
    ]
    process = subprocess.Popen(command, cwd=project_root)
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError("WebUI 启动失败，请查看上方 Streamlit 日志")
            try:
                urllib.request.urlopen(url, timeout=1).close()
                webbrowser.open_new_tab(url)
                print(f"WebUI 已打开: {url}")
                process.wait()
                return
            except OSError:
                time.sleep(0.5)
        raise TimeoutError("WebUI 在 30 秒内未能启动")
    except KeyboardInterrupt:
        print("正在关闭 WebUI...")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


if __name__ == "__main__":
    main()
