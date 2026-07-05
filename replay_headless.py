#!/usr/bin/env python3
"""Headless replay runner — regenerate tick logs without the SocketIO server or a browser.

This drives the *exact same* Engine / Game / ReplayController path that
``main_replay.py`` + ``auto_replay_loop.py`` use, but strips away everything that
only existed to serve a live browser client:

  - no Flask / SocketIO server, no eventlet, no browser, no per-tick state emit
  - no subprocess-per-replay restarts, no stdout-marker / file-size stall polling

Why this is safe (produces byte-identical tick logs):

  * The engine advances with a FIXED timestep ``dt = 1 / tick_rate`` (see
    ``engine/core.py``); it never uses wall-clock deltas. So removing real-time
    pacing cannot change any game step.
  * Replay intents are injected synchronously inside ``Engine.tick()`` keyed on the
    integer tick index (``ReplayController.sync_on_tick = True``). The recorded
    intents were captured at 24 Hz, so ``tick_rate`` MUST stay 24 — changing it
    (as the FAST mode in main_replay_with_record.py does) would re-time every step
    and corrupt the data. We pin it to the recorded rate.
  * The only RNG on the replay path feeds excluded fields (agent appearance lives
    under ``*_drawable_*``; tile sprites under ``grid_tiles``), both dropped by
    ``is_not_excludes``. ``random_game_state`` is RL-only and never runs here. So
    the logged columns are deterministic given map + intents + dt.
  * The tick-log CSV is written by the game itself (``_store_tick_data`` ->
    ``df.to_csv``) when ``game.tick_count == sum_ticks``; we run the full
    ``max_game_time * tick_rate`` ticks exactly like the watchdog does.

Usage:
    python replay_headless.py                  # process every pending replay (skips ones with a CSV)
    python replay_headless.py --jobs 8         # parallelize across processes (each replay is independent)
    python replay_headless.py --force          # regenerate even if a CSV already exists
    python replay_headless.py --only abc123.json --out-dir /tmp/verify   # one file to a scratch dir (verification)
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
for _p in (REPO_ROOT / "engine" / "src", REPO_ROOT / "games", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

REPLAY_DIR = REPO_ROOT / "analysis" / "replays"
TICKLOG_DIR = REPO_ROOT / "analysis" / "tick_logs"


def _run_one(replay_path: Path, out_dir: Path, quiet: bool = True) -> tuple[str, bool, str]:
    """Replay one game headless and write its tick-log CSV. Returns (stem, ok, message)."""
    # Imports are inside the function so this works cleanly as a multiprocessing entry point.
    from engine.core import Engine
    from engine.logging.replay_loader import load_replay_agents
    from spoiled_broth.game import SpoiledBroth as Game
    from spoiled_broth_experiment_settings.params import params_both

    stem = replay_path.stem
    try:
        with open(replay_path, "r") as f:
            replay_data = json.load(f)
    except Exception as e:  # corrupt/missing JSON — skip, don't kill the batch
        return (stem, False, f"could not read replay: {e}")

    config = dict(replay_data["config"])
    out_csv = out_dir / f"{stem}.csv"
    config["tick_log_path"] = str(out_csv)

    tick_rate = params_both["tick_rate"]          # 24 Hz — must match recorded intents
    max_ticks = params_both["max_game_time"] * tick_rate

    # Building the game / controllers prints a lot (every agent's full intent table);
    # swallow it unless we're debugging.
    sink = io.StringIO()
    redirect = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    try:
        with redirect:
            game = Game.from_state(config)
            engine = Engine(
                game,
                tick_rate=tick_rate,
                is_max_speed=True,          # fixed-dt fast loop, no real-time sleeps
                is_served_locally=True,
            )
            agents = load_replay_agents(str(replay_path))
            engine.agent_map = agents
            # Mirror EngineRunner.start() agent setup exactly so game state is identical.
            for agent_id, controller in agents.items():
                init_config = controller.agent_init_config() if hasattr(controller, "agent_init_config") else {}
                game.add_agent(agent_id, **init_config)
                controller.agent = game.gameObjects[agent_id]

            engine.start_loop(max_ticks=max_ticks)
    except Exception as e:
        return (stem, False, f"replay errored: {e}")

    if out_csv.exists():
        return (stem, True, "ok")
    return (stem, False, "finished but no CSV written (tick_count never reached sum_ticks?)")


def _pending(out_dir: Path, force: bool):
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j in sorted(REPLAY_DIR.glob("*.json")):
        if force or not (out_dir / f"{j.stem}.csv").exists():
            yield j


def _worker(args):
    replay_path_str, out_dir_str = args
    return _run_one(Path(replay_path_str), Path(out_dir_str), quiet=True)


def main():
    ap = argparse.ArgumentParser(description="Headless replay -> tick-log generator.")
    ap.add_argument("--out-dir", default=str(TICKLOG_DIR), help="where to write tick-log CSVs")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (default 1)")
    ap.add_argument("--force", action="store_true", help="regenerate even if a CSV already exists")
    ap.add_argument("--only", default=None, help="process just this replay (filename or stem)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N replays (0 = all)")
    ap.add_argument("--verbose", action="store_true", help="don't suppress per-replay engine prints")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        stem = Path(args.only).stem
        todo = [REPLAY_DIR / f"{stem}.json"]
    else:
        todo = list(_pending(out_dir, args.force))
    if args.limit:
        todo = todo[: args.limit]

    if not todo:
        print("No pending replays. Nothing to do.")
        return 0

    print(f">>> Headless replay: {len(todo)} replay(s) -> {out_dir}  (jobs={args.jobs})")
    t0 = time.time()
    ok = err = 0

    if args.jobs > 1 and not args.only:
        import multiprocessing as mp
        tasks = [(str(p), str(out_dir)) for p in todo]
        with mp.Pool(processes=args.jobs) as pool:
            for i, (stem, success, msg) in enumerate(pool.imap_unordered(_worker, tasks), 1):
                ok += success
                err += not success
                status = "ok " if success else "ERR"
                if not success or args.verbose:
                    print(f"[{i}/{len(todo)}] {status} {stem}: {msg}")
                elif i % 25 == 0 or i == len(todo):
                    print(f"[{i}/{len(todo)}] {ok} ok, {err} err  ({time.time()-t0:.1f}s)")
    else:
        for i, p in enumerate(todo, 1):
            stem, success, msg = _run_one(p, out_dir, quiet=not args.verbose)
            ok += success
            err += not success
            status = "ok " if success else "ERR"
            if not success or args.verbose:
                print(f"[{i}/{len(todo)}] {status} {stem}: {msg}")
            elif i % 25 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {ok} ok, {err} err  ({time.time()-t0:.1f}s)")

    dt = time.time() - t0
    print(f">>> Done: {ok} ok, {err} err in {dt:.1f}s "
          f"({dt/max(ok+err,1):.2f}s/replay).")
    return 1 if err and ok == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
