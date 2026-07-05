import os
import selectors
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ENGINE_SRC = REPO_ROOT / "engine" / "src"
GAMES_DIR = REPO_ROOT / "games"

REPLAY_CMD = [sys.executable, "-u", str(REPO_ROOT / "main_replay.py")]

TRIGGER_PHRASE = "Max ticks reached"   # <-- end-of-replay marker
TRIGGER_PHRASES = (
    TRIGGER_PHRASE,
    "Game ended. Disconnecting clients and saving replay.",
    "[ReplayRecorder] Saved locally",
)
DONE_PHRASE = "No pending replay JSON found."
URL = "http://localhost:8080"
_browser_opened = False

while True:
    print("\n=== Starting replay script ===\n")

    # Start process with stdout capture
    env = {
        **dict(os.environ),
        "PYTHONPATH": ":".join([
            str(ENGINE_SRC),
            str(GAMES_DIR),
            str(REPO_ROOT),
            os.environ.get("PYTHONPATH", ""),
        ]),
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        REPLAY_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=REPO_ROOT,
        env=env,
    )

    # open browser only once
    if not _browser_opened:
        webbrowser.open(URL)
        _browser_opened = True

    stdout = process.stdout
    if stdout is None:
        raise RuntimeError("Replay process stdout was not captured.")

    try:
        tick_log_path = None
        last_tick_log_size = -1
        last_tick_log_change = time.monotonic()
        replay_finished = False
        sel = selectors.DefaultSelector()
        sel.register(stdout, selectors.EVENT_READ)

        while not replay_finished:
            for _key, _mask in sel.select(timeout=0.25):
                line = stdout.readline()
                if line == "":
                    replay_finished = process.poll() is not None
                    continue

                print(line, end="")  # still print it normally
                if line.startswith("Writing tick log to:"):
                    tick_log_path = REPO_ROOT / line.split(":", 1)[1].strip()
                if DONE_PHRASE in line:
                    print("\n=== No pending replays. Done. ===\n")
                    raise SystemExit(0)
                if any(phrase in line for phrase in TRIGGER_PHRASES):
                    print("\n=== Replay finished ===\n")
                    replay_finished = True

            if tick_log_path and tick_log_path.exists():
                size = tick_log_path.stat().st_size
                if size != last_tick_log_size:
                    last_tick_log_size = size
                    last_tick_log_change = time.monotonic()
                elif size > 0 and time.monotonic() - last_tick_log_change >= 2.0:
                    print("\n=== Replay finished ===\n")
                    replay_finished = True

            if process.poll() is not None:
                print("\n=== Replay finished ===\n")
                replay_finished = True
    finally:
        # Always stop the replay server, including on Ctrl+C.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    print("Restarting in 1 second…")
    time.sleep(1)
