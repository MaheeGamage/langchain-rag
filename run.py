"""
run.py — Start both the FastAPI server and the Streamlit UI with one command.

    python run.py

FastAPI   → http://localhost:8000
Streamlit → http://localhost:8501

Press Ctrl+C to stop both.
"""

import subprocess
import sys
import threading


FASTAPI_CMD = [
    sys.executable, "-m", "uvicorn",
    "app.api:app",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--reload",
]

STREAMLIT_CMD = [
    sys.executable, "-m", "streamlit",
    "run", "streamlit_app.py",
    "--server.port", "8501",
]


def _pipe_output(proc: subprocess.Popen, label: str) -> None:
    """Forward a subprocess's output to stdout with a short label prefix."""
    for line in proc.stdout:
        print(f"[{label}] {line}", end="", flush=True)


def main() -> None:
    procs: list[subprocess.Popen] = []

    try:
        api = subprocess.Popen(FASTAPI_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ui  = subprocess.Popen(STREAMLIT_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs = [api, ui]

        threading.Thread(target=_pipe_output, args=(api, "api"), daemon=True).start()
        threading.Thread(target=_pipe_output, args=(ui,  "ui "), daemon=True).start()

        print("━" * 40)
        print("FastAPI   → http://localhost:8000")
        print("Streamlit → http://localhost:8501")
        print("Press Ctrl+C to stop both servers.")
        print("━" * 40 + "\n", flush=True)

        # Block until either process exits unexpectedly (e.g. crash)
        while all(p.poll() is None for p in procs):
            pass

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("All servers stopped.")


if __name__ == "__main__":
    main()

