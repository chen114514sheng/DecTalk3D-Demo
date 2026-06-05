from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
import time
import urllib.request
import webbrowser


REQUIRED_MODULES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "multipart": "python-multipart",
    "yaml": "pyyaml",
}


def ensure_dependencies() -> None:
    missing = [
        package
        for module, package in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return

    print("当前 Python 环境缺少启动 WebUI 所需依赖：", flush=True)
    for package in missing:
        print(f"  - {package}", flush=True)
    print("", flush=True)
    print("请先激活项目环境后再运行：", flush=True)
    print("  conda activate dectalk-demo", flush=True)
    print("  python run_app.py", flush=True)
    print("", flush=True)
    print("如果环境尚未安装依赖，请按 README 安装 requirements.txt。", flush=True)
    raise SystemExit(1)


def wait_until_ready(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=1) as response:
                return response.status == 200
        except Exception:
            time.sleep(0.3)
    return False


def open_browser_when_ready(url: str) -> None:
    if wait_until_ready(url):
        try:
            webbrowser.open(url, new=2)
        except Exception as exc:
            print(f"自动打开浏览器失败：{exc}", flush=True)
            print(f"请手动打开：{url}", flush=True)
    else:
        print(f"服务启动较慢，请稍后手动打开：{url}", flush=True)


def stop_on_enter(server) -> None:
    try:
        input()
    except EOFError:
        return
    server.should_exit = True


def main() -> None:
    ensure_dependencies()

    import uvicorn

    parser = argparse.ArgumentParser(description="启动 DecTalk3D WebUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()

    host = args.host
    port = args.port
    url = f"http://{host}:{port}"
    print(f"DecTalk3D demo 正在启动：{url}", flush=True)
    if not args.no_browser:
        print("服务启动后会自动打开网页。", flush=True)

    can_stop_with_enter = sys.stdin is not None and sys.stdin.isatty()
    if can_stop_with_enter:
        print("按 Enter 停止服务，也可以按 Ctrl+C 退出。", flush=True)
    else:
        print("服务会持续运行；按 Ctrl+C 停止。", flush=True)

    config = uvicorn.Config("app:app", host=host, port=port, reload=False)
    server = uvicorn.Server(config)
    if not args.no_browser:
        threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()
    if can_stop_with_enter:
        threading.Thread(target=stop_on_enter, args=(server,), daemon=True).start()
    server.run()


if __name__ == "__main__":
    sys.exit(main())
