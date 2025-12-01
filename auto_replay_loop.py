import subprocess
import time
import webbrowser

REPLAY_CMD = [
    "/Users/younesstrittmatter/Documents/GitHub/younesStrittmatter/collectiveIntelligence/cooked/.venv/bin/python",
    "/Users/younesstrittmatter/Documents/GitHub/younesStrittmatter/collectiveIntelligence/cooked/main_replay.py"
]

TRIGGER_PHRASE = "Max ticks reached"   # <-- end-of-replay marker
URL = "http://localhost:8080"
_browser_opened = False

while True:
    print("\n=== Starting replay script ===\n")

    # Start process with stdout capture
    process = subprocess.Popen(
        REPLAY_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # open browser only once
    if not _browser_opened:
        webbrowser.open(URL)
        _browser_opened = True

    # Stream output line by line
    for line in process.stdout:
        print(line, end="")  # still print it normally
        if TRIGGER_PHRASE in line:
            print("\n=== Replay finished ===\n")
            break

    # Kill the process since we want to move to the next replay
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()

    print("Restarting in 1 second…")
    time.sleep(1)
