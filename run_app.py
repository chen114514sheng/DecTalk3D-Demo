from __future__ import annotations

import sys
import time
import webbrowser

import uvicorn


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"
    print(f"DecTalk3D demo 已启动：{url}")
    print("按 Ctrl+C 停止服务。")

    try:
        time.sleep(0.5)
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    sys.exit(main())
