# stress_testing/bots.py
import asyncio
import os
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional, List

import socketio  # python-socketio (async)

# ---------- Args & defaults ----------
def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def parse_args():
    p = argparse.ArgumentParser()
    # Single server (kept for backward-compat; ignored if --servers provided)
    p.add_argument("--server", default=os.getenv("SERVER", "http://localhost:8080"))
    # Comma-separated list for sharding rooms across processes/ports
    p.add_argument("--servers", type=str, default=os.getenv("SERVERS", "http://localhost:8080"))
    p.add_argument("--games", type=int, default=env_int("GAMES", 10))
    p.add_argument("--players", type=int, default=env_int("PLAYERS", 2))
    p.add_argument("--duration", type=int, default=env_int("DURATION", 10))
    p.add_argument("--room-prefix", default=os.getenv("ROOM_PREFIX", "bench"))
    p.add_argument("--tick", type=int, default=env_int("TICK", 12))  # server tick for ideal calc
    p.add_argument("--ws-only", action="store_true", default=os.getenv("WS_ONLY", "1") == "1")
    p.add_argument("--ramp", type=float, default=float(os.getenv("RAMP", "0.0")),
                   help="Seconds to linearly ramp connections across all games (smooth start)")
    return p.parse_args()

# ---------- Shared stats ----------
@dataclass
class Stats:
    states_total: int = 0
    joins: int = 0
    conn: int = 0
    disc: int = 0
    errors: int = 0
    # bookkeeping
    _last_states: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def inc(self, field: str, delta: int = 1):
        async with self._lock:
            setattr(self, field, getattr(self, field) + delta)

    async def snap_and_delta_states(self) -> int:
        async with self._lock:
            curr = self.states_total
            delta = curr - self._last_states
            self._last_states = curr
            return delta

# ---------- Single client ----------
async def run_client(server: str, room: str, stats: Stats, ws_only: bool, start_delay: float = 0.0):
    sio = socketio.AsyncClient(
        reconnection=True,
        logger=False,
        engineio_logger=False,
    )

    session_id: Optional[str] = None
    agent_id: Optional[str] = None

    @sio.event
    async def connect():
        await stats.inc("joins", 1)
        await stats.inc("conn", 1)

    @sio.event
    async def disconnect():
        await stats.inc("conn", -1)
        await stats.inc("disc", 1)

    @sio.on("joined")
    async def on_joined(data):
        nonlocal session_id, agent_id
        session_id = data.get("session_id")
        agent_id = data.get("agent_id")

    @sio.on("state")
    async def on_state(_data):
        await stats.inc("states_total", 1)

    # Optional ramp to avoid a thundering herd
    if start_delay > 0:
        await asyncio.sleep(start_delay)

    url = f"{server}?room={room}"
    try:
        if ws_only:
            # IMPORTANT: WS-only avoids long-poll overhead with many clients
            await sio.connect(url, transports=["websocket"])
        else:
            await sio.connect(url)
    except Exception:
        await stats.inc("errors", 1)
        return

    try:
        while True:
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        try:
            await sio.disconnect()
        except Exception:
            pass
        raise

# ---------- Orchestrator ----------
async def ticker(args, stats: Stats, started_t: float):
    t = 0
    denom = max(1, args.games * args.players)
    ideal_sps = args.tick * denom if args.tick > 0 else 0

    print(f">>> Running bots: {args.games} games x {args.players} players for {args.duration}s")
    while True:
        await asyncio.sleep(1.0)
        t += 1
        sps = await stats.snap_and_delta_states()
        nsps = sps / denom  # normalized per-client Hz
        util = (100.0 * sps / ideal_sps) if ideal_sps else 0.0
        pend = max(0, denom - stats.conn - stats.disc)

        # snapshot (small race is fine for display)
        joins = stats.joins
        conn = stats.conn
        disc = stats.disc
        errors = stats.errors

        print(
            f"[tick] t={t:2d}s "
            f"sps={sps:4d} "
            f"nsps={nsps:6.2f} "
            f"util={util:6.1f}% "
            f"joins={joins:3d} conn={conn:3d} pend={pend:3d} disc={disc:3d} errors={errors:3d}"
        )

        if time.time() - started_t >= args.duration:
            break

async def main():
    args = parse_args()
    stats = Stats()
    started_t = time.time()

    # Build server list; fall back to --server if --servers is empty
    servers: List[str] = [s.strip() for s in args.servers.split(",") if s.strip()]
    if not servers:
        servers = [args.server]

    # Assign each ROOM to one server (round-robin), so both players in a room
    # hit the same backend and can actually play together.
    room_servers = {}
    for g in range(args.games):
        room = f"{args.room_prefix}{g}"
        room_servers[room] = servers[g % len(servers)]

    # Optional connection ramp: spread first connections over args.ramp seconds
    # Each room gets a base delay; both players use the same delay (+ tiny jitter).
    tasks = []
    for g in range(args.games):
        room = f"{args.room_prefix}{g}"
        server = room_servers[room]
        base_delay = (args.ramp * g / max(1, args.games - 1)) if args.ramp > 0 else 0.0

        for p in range(args.players):
            # small per-player jitter to avoid lockstep; keeps same server per room
            jitter = 0.03 * p
            tasks.append(asyncio.create_task(
                run_client(server, room, stats, args.ws_only, start_delay=base_delay + jitter)
            ))

    # run ticker for duration
    tick_task = asyncio.create_task(ticker(args, stats, started_t))
    await tick_task

    # cancel clients and wait
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    total_time = max(1.0, time.time() - started_t)
    total_states = stats._last_states or stats.states_total
    denom = max(1, args.games * args.players)
    nsps = total_states / total_time / denom
    print(
        f"[bots] ran {args.games} games x {args.players} players for {total_time:.1f}s | "
        f"total states={total_states} ({total_states/total_time:.1f}/s), "
        f"nsps={nsps:.2f}, joins={stats.joins}, disc={stats.disc}, errors={stats.errors}"
    )

if __name__ == "__main__":
    asyncio.run(main())
