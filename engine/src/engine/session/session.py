import secrets

from engine.core import Engine
from engine.engine_runner import EngineRunner
from eventlet.semaphore import Semaphore
from engine.logging.replay_recorder import ReplayRecorder
import time
import os
import json

NO_KICK = os.getenv("BENCH_NO_KICK", "0") == "1"

SHARD_ID = os.getenv("SHARD_ID", "1")
# REQUIRED: single place to point all shards
# Example: gs://cooked-455218-game-state/shards
SHARD_STATE_GS_URL = (os.getenv("SHARD_STATE_GS_URL", "") or "").strip()

try:
    from google.cloud import storage  # not needed locally
except Exception:
    print('[WARN] google.cloud.storage not available')
    storage = None

_gcs_client = None

# ---- GCS CAS helpers (SessionApp / game side) ----
from time import sleep
import random

_gcs_client = None


def _parse_gs_url(url: str):
    rest = url[len("gs://"):]
    if "/" in rest:
        b, p = rest.split("/", 1)
        return b, p.strip("/")
    return rest, ""


def _resolve_gcs_target():
    if not SHARD_STATE_GS_URL.startswith("gs://"):
        return None, None
    bucket, prefix = _parse_gs_url(SHARD_STATE_GS_URL)
    parts = [p for p in [prefix, SHARD_ID, "state", "state.json"] if p]
    return bucket, "/".join(parts)


def _read_gcs_state():
    bucket_name, blob_path = _resolve_gcs_target()
    if not bucket_name or not blob_path or storage is None:
        return None, None
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    b = _gcs_client.bucket(bucket_name)
    blob = b.blob(blob_path)
    if not blob.exists():
        return {}, None  # treat as empty doc (first writer will create)
    return json.loads(blob.download_as_text()), blob.generation


def _cas_merge(edit_fn, retries: int = 6):
    """
    Read-modify-write with if_generation_match.
    edit_fn receives a mutable dict 'st' and must return it (or None to abort).
    Retries on precondition failures with small jitter.
    """
    bucket_name, blob_path = _resolve_gcs_target()
    if not bucket_name or not blob_path or storage is None:
        return False
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    b = _gcs_client.bucket(bucket_name)
    blob = b.blob(blob_path)

    for attempt in range(retries):
        st, gen = _read_gcs_state()
        if st is None:
            return False
        # ensure we never drop lobby keys by accident
        st = dict(st)

        new_st = edit_fn(st)
        if new_st is None:
            return False

        body = json.dumps(new_st, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        try:
            if gen is None:
                # create only if not exists yet
                blob.upload_from_string(body, content_type="application/json", if_generation_match=0)
            else:
                blob.upload_from_string(body, content_type="application/json", if_generation_match=gen)
            blob.cache_control = "no-store"
            blob.patch()
            return True
        except Exception:
            # precondition failed / generation changed — retry with backoff
            sleep(0.02 + 0.02 * random.random())
    return False


def _parse_gs_url(url: str):
    # gs://bucket[/prefix...]
    rest = url[len("gs://"):]
    if "/" in rest:
        b, p = rest.split("/", 1)
        return b, p.strip("/")
    return rest, ""


def _resolve_gcs_target():
    """
    Returns (bucket_name, blob_path) or (None, None) if not configured.
    Writes to: <prefix>/<SHARD_ID>/state/state.json (prefix may be empty)
    """
    if not SHARD_STATE_GS_URL.startswith("gs://"):
        return None, None
    bucket, prefix = _parse_gs_url(SHARD_STATE_GS_URL)
    parts = [p for p in [prefix, SHARD_ID, "state", "state.json"] if p]
    return bucket, "/".join(parts)


def _write_state_out(payload: dict):
    """
    Merge 'payload' into existing shard doc under CAS.
    This preserves lobby fields like waiting_players/blocked_until.
    """
    if not isinstance(payload, dict):
        return

    def edit(st):
        # merge (dict.update) instead of replacing the document
        st.update(payload)
        return st

    _cas_merge(edit)


def _session_state_snapshot(session) -> dict:
    return {
        "nr_players": len(session.sid_to_agent),
        "started": getattr(session, "_started_global", True),
    }


def _write_state_to_store(session):
    _write_state_out(_session_state_snapshot(session))


# prod only # local fallback (e.g. ".local_state/shard-1")


class Session:
    def __init__(self,
                 game_factory,
                 ui_modules,
                 agent_map,
                 tick_rate,
                 ai_tick_rate,
                 is_max_speed,
                 socketio,
                 max_game_time=None,
                 redirect_link=None,
                 url_params=None,
                 is_served_locally=False):
        self.is_served_locally = is_served_locally
        self.id = secrets.token_hex(8)
        self.recorder = ReplayRecorder(self.id)

        self.game_start_time = None

        # --- NEW: emitter state ---
        self._dirty = False  # set True when a new tick happens
        self._last_emitted_tick = -1  # last tick we sent
        self._emitter_started = False
        self._started_global = False

        self.socketio = socketio
        self.sid_to_agent = {}
        self.agent_to_sid = {}
        self._maps_lock = Semaphore(1)
        self.engine = Engine(game_factory(url_params=url_params),
                             tick_rate=tick_rate,
                             is_max_speed=is_max_speed,
                             replay_recorder=self.recorder,
                             is_served_locally=self.is_served_locally)
        self.engine.add_on_tick_callback(self.broadcast_state)
        self.engine.add_on_game_end_callback(self.handle_game_end)
        self.recorder.set_config(self.engine.game.serialize_initial_state())
        self.ui_modules = ui_modules
        self._agents = []
        self.websockets = {}
        self.max_game_time = max_game_time
        self.redirect_link = redirect_link
        self.game_start_time = None

        self._runner = EngineRunner(
            game=self.engine.game,
            engine=self.engine,
            agent_map=agent_map,
            tick_rate=tick_rate,
            ai_tick_rate=ai_tick_rate,
            is_max_speed=is_max_speed,
            session=self
        )

    def register_websocket(self, agent_id, sid):
        with self._maps_lock:
            self.agent_to_sid[agent_id] = sid
            self.sid_to_agent[sid] = agent_id

    def get_agent_id_by_sid(self, sid):
        with self._maps_lock:
            return self.sid_to_agent.get(sid)

    def has_active_websockets(self):
        with self._maps_lock:
            return any(sid in self.sid_to_agent for sid in self.sid_to_agent)

    def add_agent(self, agent_id, url_params=None):
        if hasattr(self.engine.game, "add_agent"):
            self.engine.game.add_agent(agent_id, url_params=url_params)
        self._agents.append(agent_id)
        self.recorder.register_agent(agent_id, self.engine.game.agent_initial_state(agent_id))

    def register_agent(self, agent_id, initial_state):
        self.recorder.register_agent(agent_id, initial_state)

    def is_full(self, max_agents):
        return len(self._agents) >= max_agents

    def start(self):
        self.game_start_time = time.time()
        self._runner.start()

        if not self._emitter_started:
            self._emitter_started = True
            self.socketio.start_background_task(self._emit_loop)

    def broadcast_state(self):
        self._dirty = True

    def _emit_loop(self):
        while not self.engine.game.done:
            if self._dirty and self.engine.tick_count != self._last_emitted_tick:
                self._dirty = False
                self._last_emitted_tick = self.engine.tick_count
                self._emit_once()
            self.socketio.sleep(0.001)  # small yield; don’t busy spin
        self.handle_game_end(self.is_served_locally)

    def _emit_once(self):
        game = self.engine.game
        engine = self.engine
        # snapshot recipients safely
        with self._maps_lock:
            recipients = list(self.agent_to_sid.items())

        for agent_id, sid in recipients:
            payload = {"type": "state", "you": agent_id, "tick": engine.tick_count}
            for module in self.ui_modules:
                payload.update(module.serialize_for_agent(game, engine, agent_id))
            try:
                self.socketio.emit("state", payload, to=sid)
            except Exception as e:
                print(f"[WS] Error sending to {agent_id}: {e}")

    def handle_game_end(self, is_served_locally):
        if getattr(self, "_ended", False):
            return

        self._ended = True
        self._started_global = False

        print(f"[Session {self.id}] Game ended. Disconnecting clients and saving replay.")
        self.recorder.save()
        if not is_served_locally:
            _write_state_out({
                "nr_players": 0,
                "started": False,
            })
            print('[Session] Replay saved. (205)')

        for sid in list(self.sid_to_agent.keys()):
            print('[Session] Disconnecting (208)', sid)
            try:
                if not NO_KICK:
                    print('[Session] Kicking', sid)
                    if self.redirect_link is not None:
                        redirect_link = self.engine.game.redirect_link() if hasattr(self.engine.game,
                                                                                    "redirect_link") else self.redirect_link
                        self.socketio.emit("redirect", {"url": redirect_link}, to=sid)
                        self.socketio.sleep(0.5)

                    self.socketio.server.disconnect(sid)
            except Exception as e:
                print(f"[WS] Error disconnecting {sid}: {e}")
