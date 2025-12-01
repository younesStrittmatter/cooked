# main_replay_with_record.py
"""
Replay -> Auto-record -> (optional) GIF

Quick start:
    pip install playwright
    playwright install chromium
    # optional for GIF:
    #   macOS: brew install ffmpeg
    #   Debian/Ubuntu: sudo apt-get install ffmpeg

Env toggles (all optional):
    HEADLESS=1               # default 1 (headless). Set 0 to watch the replay.
    FAST=1                   # default 0. Set 1 to force max-speed replay (overrides params).
    RECORD_SECONDS=0         # default 0 => auto: uses max_game_time + buffer. Or set a number.
    WIDTH=1024 HEIGHT=768    # viewport size (defaults shown).
    RECORD_OUT_DIR=analysis/videos
    GIF=1                    # default 1. Set 0 to skip making a GIF.
    GIF_FPS=12               # default 12
    GIF_WIDTH=720            # default 720 (height auto)
    SERVER_HOST=127.0.0.1    # default 127.0.0.1
    SERVER_PORT=8080         # default 8080
    REPLAY_DIR=analysis/replays
    TICKLOG_DIR=analysis/tick_logs
"""

# --- Make 'spoiled_broth' importable when packages live under ./games ---
import sys, os, json, time, logging, subprocess, socket
from pathlib import Path
from threading import Thread

REPO_ROOT = Path(__file__).resolve().parent
GAMES_DIR = REPO_ROOT / "games"
if str(GAMES_DIR) not in sys.path:
    sys.path.insert(0, str(GAMES_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------- Eventlet/engine imports ----------
import eventlet

eventlet.monkey_patch()

from engine.app.session_app import SessionApp
from engine.logging.replay_loader import load_replay_agents
from spoiled_broth.game import SpoiledBroth as Game
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from spoiled_broth_experiment_settings.params import (
    params_both as _params_both,
    params_replay as _params_replay,
)

# ---------- Logging ----------
log = logging.getLogger("replay_recorder")
log.setLevel(logging.INFO)
logging.getLogger("werkzeug").disabled = True  # quiet Flask logs

# ---------- Config via ENV with sane defaults ----------
HEADLESS = os.getenv("HEADLESS", "1") == "1"
FAST = os.getenv("FAST", "0") == "1"
RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "0"))
WIDTH = int(os.getenv("WIDTH", "1024"))
HEIGHT = int(os.getenv("HEIGHT", "768"))
RECORD_OUT_DIR = os.getenv("RECORD_OUT_DIR", "analysis/videos")
MAKE_GIF = os.getenv("GIF", "1") == "1"
GIF_FPS = int(os.getenv("GIF_FPS", "12"))
GIF_WIDTH = int(os.getenv("GIF_WIDTH", "720"))
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8080"))
REPLAY_DIR = os.getenv("REPLAY_DIR", "analysis/replays")
# REPLAY_DIR = os.getenv("REPLAY_DIR", "replays")
TICKLOG_DIR = os.getenv("TICKLOG_DIR", "analysis/tick_logs")

BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/"

# ---------- Paths ----------
path_root = REPO_ROOT / "games" / "spoiled_broth"


def get_next_replay_path() -> str | None:
    replay_folder = Path(REPLAY_DIR)
    replay_folder.mkdir(parents=True, exist_ok=True)
    video_dir = Path(RECORD_OUT_DIR)
    video_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(replay_folder.iterdir()):
        if filename.suffix == ".json":
            video_file = video_dir / f"{filename.stem}.gif"
            if not video_file.exists():
                return str(filename)
    return None


def build_engine_app(replay_path: str) -> tuple[SessionApp, dict]:
    with open(replay_path, "r") as f:
        log.info(f"Using replay: {replay_path}")
        replay_data = json.load(f)

    config = replay_data["config"]
    Path(TICKLOG_DIR).mkdir(parents=True, exist_ok=True)
    # config["tick_log_path"] = str(Path(TICKLOG_DIR) / f"{Path(replay_path).stem}.csv")

    replay_agents = load_replay_agents(replay_path)

    params_both = dict(_params_both)
    params_replay = dict(_params_replay)

    if FAST:
        params_replay["is_max_speed"] = True
        params_both["tick_rate"] = max(60, params_both.get("tick_rate", 5))
        ai_tick_rate = 60
    else:
        ai_tick_rate = 24

    engine_app = SessionApp(
        game_factory=lambda url_params=None: Game.from_state(config),
        ui_modules=[Renderer2DModule()],
        agent_map=replay_agents,
        path_root=path_root,
        tick_rate=params_both["tick_rate"],
        ai_tick_rate=ai_tick_rate,
        n_players=params_replay["n_players"],
        is_max_speed=params_replay["is_max_speed"],
        max_game_time=params_both["max_game_time"],
        is_served_locally=True,
    )

    map_name = replay_data['config']['init_args']['map_nr']
    # create a folder with the name if it does not exist

    # get player ids
    agent_ids = replay_data['agents'].keys()

    player_conditions = {}

    for player_id in agent_ids:
        player_conditions.update(get_player_stats(player_id, replay_data))

    first_player_condition = None
    last_player_condition = None
    for p_c in player_conditions:
        if int(player_conditions[p_c]['player_nr']) == 1:
            first_player_condition = player_conditions[p_c]
        else:
            last_player_condition = player_conditions[p_c]
    if first_player_condition is None or last_player_condition is None:
        print('Could not determine player conditions, skipping bundle')
        print(player_conditions)
        raise
    first_player_str = f"[p{first_player_condition['start_pos']}_cs({first_player_condition['cutting_speed']})_ws({first_player_condition['walking_speed']})]"
    last_player_str = f"[p{last_player_condition['start_pos']}_cs({last_player_condition['cutting_speed']})_ws({last_player_condition['walking_speed']})]"

    additional_info = first_player_condition['additional_condition_info']
    condition_str = f'{first_player_str}_{last_player_str}_{additional_info}'

    out_dir = Path(RECORD_OUT_DIR) / condition_str
    return engine_app, config, out_dir


def _run_server_thread(engine_app: SessionApp):
    # Runs in a daemon thread; no pickling required.
    engine_app.socketio.run(
        engine_app.app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        use_reloader=False,
    )


def _wait_for_port(host: str, port: int, timeout: float = 10.0):
    """Wait until a TCP port is accepting connections."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


async def _playwright_record_async(url: str, out_dir: str, width: int, height: int, seconds: int,
                                   headless: bool) -> str:
    from playwright.async_api import async_playwright
    from pathlib import Path as _P

    _P(out_dir).mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=out_dir,
            record_video_size={"width": width, "height": height},
        )
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        await page.wait_for_timeout(seconds * 1000)
        video_path = await page.video.path()
        await context.close()
        await browser.close()
        return video_path


def record_with_playwright(url: str, out_dir: str, width: int, height: int, seconds: int, headless: bool) -> str:
    import asyncio
    return asyncio.run(_playwright_record_async(url, out_dir, width, height, seconds, headless))


def webm_to_gif(webm_path: str, out_gif: str, fps: int, width: int):
    palette = str(Path(out_gif).with_suffix(".palette.png"))
    cmd1 = [
        "ffmpeg", "-y",
        "-i", webm_path,
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen",
        palette
    ]
    cmd2 = [
        "ffmpeg", "-y",
        "-i", webm_path,
        "-i", palette,
        "-lavfi", f"fps={fps},scale={width}:-1:flags=lanczos [x]; [x][1:v] paletteuse",
        out_gif
    ]
    try:
        subprocess.run(cmd1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log.info(f"GIF created: {out_gif}")
    except FileNotFoundError:
        log.warning("ffmpeg not found. Skipping GIF creation.")
    except subprocess.CalledProcessError as e:
        log.warning(f"ffmpeg failed: {e}. Skipping GIF creation. {e}")


def get_player_stats(player_id, replay_json):
    agent_info = replay_json['agents'][f'{player_id}']['initial_state']
    agent_url_params = agent_info["url_params"]
    if 'player_nr' in agent_info:
        player_nr = int(agent_info['player_nr'])

        player_starting_pos_x = int(agent_url_params[f'slot_x_p{player_nr}'][0])
        player_starting_pos_y = int(agent_url_params[f'slot_y_p{player_nr}'][0])
        player_cutting_speed = float(agent_url_params[f'cutting_speed_p{player_nr}'][0])
        player_walking_speed = float(agent_url_params[f'walking_speed_p{player_nr}'][0])
    else:

        player_nr = int(agent_url_params["player"][0][-1])
        player_starting_pos_x = int(agent_url_params[f'slot_x'][0])
        player_starting_pos_y = int(agent_url_params[f'slot_y'][0])
        player_cutting_speed = float(agent_url_params[f'cutting_speed'][0])
        player_walking_speed = float(agent_url_params[f'walking_speed'][0])
    additional_condition_info = 'default'
    if 'additional_condition_info' in agent_url_params:
        additional_condition_info = agent_url_params['additional_condition_info'][0]
    if 'start_time' in replay_json:
        if replay_json['start_time'].startswith('2025-09-04'):
            additional_condition_info = 'ability_hints'

    return {
        player_id: {
            'player_nr': player_nr,
            'start_pos': (player_starting_pos_x, player_starting_pos_y),
            'cutting_speed': player_cutting_speed,
            'walking_speed': player_walking_speed,
            'additional_condition_info': additional_condition_info,
        }
    }


def main():
    replay_path = get_next_replay_path()
    if not replay_path:
        print("No pending replay JSON found.")
        return

    engine_app, config, out_path = build_engine_app(replay_path)
    print(f"[{time.strftime('%H:%M:%S')}] Replay loaded. Starting server and recording...")

    # Decide recording length
    seconds = 20
    if seconds <= 0:
        max_t = int(config.get("max_game_time") or _params_both.get("max_game_time") or 20)
        seconds = max(5, max_t + (2 if FAST else 5))
    print(f"[{time.strftime('%H:%M:%S')}] Recording {seconds} seconds...")

    # Start server in a daemon thread
    server_thread = Thread(target=_run_server_thread, args=(engine_app,), daemon=True)
    server_thread.start()

    print(f"[{time.strftime('%H:%M:%S')}]Starting server...")

    # Wait for port to be ready (robust vs fixed sleep)
    if not _wait_for_port(SERVER_HOST, SERVER_PORT, timeout=15.0):
        raise RuntimeError(f"Server did not start on {SERVER_HOST}:{SERVER_PORT}")

    # Record
    video_path = record_with_playwright(
        url=BASE_URL,
        out_dir=out_path,
        width=WIDTH,
        height=HEIGHT,
        seconds=seconds,
        headless=HEADLESS,
    )
    video_path = str(Path(video_path).resolve())
    log.info(f"Recorded video: {video_path}")

    # Optional: GIF
    if MAKE_GIF:
        out_gif = str(Path(out_path) / f"{Path(replay_path).stem}.gif")
        webm_to_gif(video_path, out_gif, fps=GIF_FPS, width=GIF_WIDTH)

    # Exit: daemon thread will end when main process exits
    log.info("Done.")


if __name__ == "__main__":
    main()
