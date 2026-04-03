"""
Aion Launcher — Start the backend server and Electron frontend together.

Usage:
    python launch.py              # Start both server and frontend
    python launch.py --server     # Start server only
    python launch.py --frontend   # Start frontend only
"""

import argparse
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(ROOT, "frontend", "orb-app")


def start_server():
    """Launch the FastAPI backend via uvicorn."""
    print("[LAUNCHER] Starting backend server...")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=ROOT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def start_frontend():
    """Launch the Electron frontend via npm."""
    print("[LAUNCHER] Starting frontend...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    return subprocess.Popen(
        [npm_cmd, "start"],
        cwd=FRONTEND_DIR,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )


def wait_for_server(timeout: int = 30) -> bool:
    """Wait until the backend server is responding."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description="Aion Launcher")
    parser.add_argument("--server", action="store_true", help="Start server only")
    parser.add_argument("--frontend", action="store_true", help="Start frontend only")
    args = parser.parse_args()

    both = not args.server and not args.frontend
    procs: list[subprocess.Popen] = []

    def shutdown(sig=None, frame=None):
        print("\n[LAUNCHER] Shutting down...")
        for p in procs:
            try:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, shutdown)

    try:
        if both or args.server:
            server = start_server()
            procs.append(server)

        if both:
            print("[LAUNCHER] Waiting for backend to be ready...")
            if wait_for_server():
                print("[LAUNCHER] Backend is ready!")
            else:
                print("[LAUNCHER] WARNING: Backend did not respond within timeout, starting frontend anyway")

        if both or args.frontend:
            frontend = start_frontend()
            procs.append(frontend)

        print(f"[LAUNCHER] Running ({len(procs)} process(es)). Press Ctrl+C to stop.")

        # Wait for any process to exit
        while True:
            for p in procs:
                ret = p.poll()
                if ret is not None:
                    name = "Server" if p == procs[0] and (both or args.server) else "Frontend"
                    print(f"[LAUNCHER] {name} exited with code {ret}")
                    shutdown()
            time.sleep(1)

    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
