from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room
from pathlib import Path
from engine.session.session_manager import SessionManager
from engine.interface.static_routes import register_static_routes
from engine.interface.engine_routes import register_engine_routes
from engine.interface.core_ui_module import CoreUIModule
from urllib.parse import parse_qs
import os, json, socket, time
import orjson

try:
    import redis  # pip install redis
except ImportError:
    redis = None

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


def _orjson_default(x):
    # Robust fallback for non-JSON types
    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
    except Exception:
        pass
    try:
        from pathlib import Path
        if isinstance(x, Path): return str(x)
    except Exception:
        pass
    if isinstance(x, set): return list(x)
    if hasattr(x, "model_dump"): return x.model_dump()
    if hasattr(x, "dict"): return x.dict()
    return str(x)


class OrjsonCompat:
    @staticmethod
    def dumps(obj, **kwargs) -> str:
        # Socket.IO may pass separators, sort_keys, default, etc.
        default = kwargs.get("default", _orjson_default)
        try:
            return orjson.dumps(obj, default=default).decode("utf-8")
        except Exception:
            # Any exotic kwargs → fall back to stdlib json
            return json.dumps(obj, **kwargs)

    @staticmethod
    def loads(s, **kwargs):
        try:
            return orjson.loads(s)
        except Exception:
            return json.loads(s, **kwargs)


# ---- Socket.IO config from env (single vs multi controlled by your script) ----
use_compress = os.getenv("SOCKIO_COMPRESS", "0") == "1"
force_base = os.getenv("SOCKIO_FORCE_BASE", "0") == "1"
mq_url = None if force_base else os.getenv("REDIS_URL")  # None => BaseManager (single)


def _make_redis():
    url = os.getenv("REDIS_URL")
    if not url or not redis:
        return None
    try:
        return redis.Redis.from_url(url)
    except Exception:
        return None


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


def _read_state_and_generation():
    bucket_name, blob_path = _resolve_gcs_target()
    if not bucket_name or not blob_path or storage is None:
        return None, None, None, None  # (bucket, blob, st/gen) unavailable
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    b = _gcs_client.bucket(bucket_name)
    blob = b.blob(blob_path)
    if blob.exists():
        st = json.loads(blob.download_as_text())
        return b, blob, st, blob.generation
    else:
        return b, blob, {}, None  # treat as empty, no generation yet


def _write_state_out(payload: dict, waiting_id=None, end_time=None, retries: int = 6):
    """
    CAS-safe: read-modify-write with if_generation_match, retrying on contention.
    - Merges 'payload' into existing doc (dict.update)
    - Removes 'waiting_id' from waiting_players (if present)
    - Sets 'end_time' if provided
    """
    if storage is None:
        return  # GCS not available (local)

    for attempt in range(retries):
        res = _read_state_and_generation()
        if not res:
            return
        b, blob, existing, gen = res

        # --- apply your exact logic deterministically ---
        new_doc = dict(existing)  # copy
        if payload:
            new_doc.update(payload)

        if waiting_id:
            wl = list(new_doc.get("waiting_players", []))
            # dedupe for safety; preserve order
            seen = set()
            wl = [x for x in wl if (x not in seen) and not seen.add(x)]
            try:
                wl.remove(waiting_id)
            except ValueError:
                pass
            new_doc["waiting_players"] = wl

        if end_time:
            new_doc["end_time"] = end_time

        body = json.dumps(new_doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        try:
            if gen is None:
                # only create if object does NOT exist yet
                blob.upload_from_string(body, content_type="application/json", if_generation_match=0, timeout=30)
            else:
                # update only if no one else changed it since we read it
                blob.upload_from_string(body, content_type="application/json", if_generation_match=gen, timeout=30)
            blob.cache_control = "no-store"
            blob.patch()
            return  # success
        except Exception:
            # Precondition failed or transient; retry with small jitter
            time.sleep(0.02 + 0.02 * random.random())


def _session_state_snapshot(session) -> dict:
    return {
        "nr_players": len(session.sid_to_agent),
        "started": getattr(session, "_started_global", True),
    }


def _write_state_to_store(session, waiting_id, end_time=0):
    _write_state_out(_session_state_snapshot(session), waiting_id, end_time)


class SessionApp:
    def __init__(self,
                 game_factory,
                 ui_modules=None,
                 agent_map=None,
                 path_root=None,
                 tick_rate=24,
                 ai_tick_rate=5,
                 n_players=2,
                 is_max_speed=False,
                 max_game_time=None,
                 redirect_link=None,
                 is_served_locally=False
                 ):
        self.is_served_locally = is_served_locally
        self.max_game_time = max_game_time

        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "replace-this-if-needed"

        self._redis = _make_redis()  # None in single-worker
        self._sid_to_roomkey = {}  # sid -> room key (for forwarding)

        self.socketio = SocketIO(self.app,
                                 cors_allowed_origins="*",
                                 async_mode="eventlet",
                                 ping_interval=30,
                                 ping_timeout=40,
                                 websocket_compression=use_compress,
                                 message_queue=mq_url,
                                 json=OrjsonCompat)

        print("SocketIO manager:", type(self.socketio.server.manager).__name__,
              "| compression:", use_compress, "| mq_url:", mq_url)

        self.path_root = Path(path_root) if path_root else Path(__file__).resolve().parents[2] / "game"
        self.ui_modules = (ui_modules or []) + [CoreUIModule()]
        for module in self.ui_modules:
            if hasattr(module, "set_path_root"):
                module.set_path_root(self.path_root)

        self.session_manager = SessionManager(
            game_factory=game_factory,
            ui_modules=self.ui_modules,
            agent_map=agent_map or {},
            tick_rate=tick_rate,
            ai_tick_rate=ai_tick_rate,
            n_players=n_players,
            is_max_speed=is_max_speed,
            socketio=self.socketio,
            max_game_time=max_game_time,
            redirect_link=redirect_link
        )

        register_engine_routes(self.app, self.session_manager)
        register_static_routes(self.app, self.path_root, ui_modules=self.ui_modules)

        self._register_websocket_routes()

    # in SessionApp.__init__ or anywhere in the class
    def _ensure_controller_poll(self, session):
        if getattr(session, "_poll_set", False):
            return

        engine = session.engine

        def _is_replay(ctrl):
            # robust: mark replay controllers in your loader with ctrl.sync_each_tick = True
            return (
                    getattr(ctrl, "sync_each_tick", False)
                    or hasattr(ctrl, "intent_by_tick")
                    or ctrl.__class__.__name__.lower().startswith("replay")
            )

        def controller_poll(tick: int):
            # Look up controllers dynamically each tick
            ctrls = getattr(session, "controllers", {}) or {}
            if not ctrls:
                return
            # Build observations once per tick (if available)
            obs = session.build_observations() if hasattr(session, "build_observations") else {}
            for aid, ctrl in sorted(ctrls.items()):
                if not _is_replay(ctrl):
                    continue  # only replay controllers are sync-per-tick
                try:
                    act = ctrl.choose_action(obs.get(aid, {}), tick=tick)
                except Exception:
                    act = {}
                # SYNC inject into THIS tick so replay is tick-perfect
                engine.inject_for_current_tick(aid, act)

        engine.set_controller_poll(controller_poll)
        setattr(session, "_poll_set", True)

    def _start_once(self, room_key: str, session):
        """Start the engine only once across all workers."""
        if self._redis and room_key:
            # cross-process idempotency
            key = f"sess:{room_key}:started"
            try:
                if self._redis.set(key, 1, nx=True, ex=24 * 3600):
                    # mark local too (optional)
                    setattr(session, "_started", True)
                    session.start()
                else:
                    print(f"[START] {session.id} already started elsewhere")
            except Exception:
                # fail-open locally
                if not getattr(session, "_started", False):
                    setattr(session, "_started", True)
                    session.start()
        else:
            # single worker
            if not getattr(session, "_started", False):
                setattr(session, "_started", True)
                session.start()

    # ---------- Redis helpers (used only in multi-worker) ----------
    def _owner_elect(self, room_key: str) -> bool:
        """True if this worker becomes owner for room_key."""
        if not self._redis or not room_key:
            return True
        key = f"sess:{room_key}:owner"
        try:
            owner_id = f"{socket.gethostname()}:{os.getpid()}"
            # first writer wins; expire in 24h
            return bool(self._redis.set(key, owner_id, nx=True, ex=24 * 3600))
        except Exception:
            return True

    def _set_canonical_session_id(self, room_key: str, session_id: str):
        if not self._redis:
            return
        try:
            self._redis.set(f"sess:{room_key}:sid", session_id, ex=24 * 3600)
        except Exception:
            pass

    def _get_canonical_session_id(self, room_key: str):
        if not self._redis:
            return None
        try:
            v = self._redis.get(f"sess:{room_key}:sid")
            return v.decode() if v else None
        except Exception:
            return None

    def _pub(self, room_key: str, event: str, payload: dict):
        if not self._redis:
            return
        chan = f"sess:{room_key}:ctrl"
        try:
            self._redis.publish(chan, json.dumps({"event": event, "data": payload}))
        except Exception as e:
            print("[REDIS PUBLISH] error:", e)

    def _start_ctrl_loop(self, room_key: str, session):
        """Owner-only: receive join/intent/disconnect from non-owners."""
        if not self._redis or not room_key:
            return
        chan = f"sess:{room_key}:ctrl"

        def _loop():
            try:
                pubsub = self._redis.pubsub()
                pubsub.subscribe(chan)
                print(f"[CTRL] Listening on {chan}")
                for item in pubsub.listen():
                    if item.get("type") != "message":
                        continue
                    try:
                        msg = json.loads(item["data"])
                        evt, data = msg.get("event"), msg.get("data", {})
                    except Exception:
                        continue

                    if evt == "join":
                        agent_id = data["agent_id"]
                        sid = data["sid"]
                        url_params = data.get("url_params", {})
                        if agent_id not in session._agents:
                            session.add_agent(agent_id, url_params=url_params)
                            session.register_websocket(agent_id, sid)
                            print(f"[CTRL] Remote join: {agent_id} → sid {sid}")
                            if session.is_full(self.session_manager.n_players):
                                print(f"[CTRL] Session {session.id} full (remote), starting")
                                session.start()

                    elif evt == "intent":
                        session.engine.submit_intent(data["agent_id"], data["action"])

                    elif evt == "disconnect":
                        sid = data["sid"]
                        agent_id = session.get_agent_id_by_sid(sid)
                        if agent_id:
                            if not session.is_full(self.session_manager.n_players):
                                if hasattr(session.engine.game, "remove_agent"):
                                    session.engine.game.remove_agent(agent_id)
                                if agent_id in session._agents:
                                    session._agents.remove(agent_id)
                                session.recorder.unregister_agent(agent_id)
                            session.websockets.pop(agent_id, None)
                            session.agent_to_sid.pop(agent_id, None)
                            print(f'[CTRL] Popping sid {sid} from sid_to_agent')
                            session.sid_to_agent.pop(sid, None)
            except Exception as e:
                print("[CTRL] loop error:", e)

        self.socketio.start_background_task(_loop)

    # ---------------- WebSocket routes ----------------
    def _register_websocket_routes(self):
        @self.socketio.on("connect")
        def on_connect(auth=None):
            sid = request.sid
            params = parse_qs(request.query_string.decode())

            if auth and isinstance(auth, dict):
                for k, v in auth.items():
                    params.setdefault(k, []).append(v)

            # IMPORTANT: multi-worker needs a stable key both players share.
            # Require ?room=XYZ when running with multiple workers.
            room_key = params.get("room", [None])[0] if self._redis else None
            waiting_id = params.get("pid", [None])[0]

            agent_id = self.session_manager.generate_agent_id()

            if self._redis and room_key:
                # MULTI-WORKER PATH (owner forwarding + per-SID emits)
                # Put this socket into a shared room for broadcast convenience (optional).
                room = f"sesskey:{room_key}"
                join_room(room)

                if self._owner_elect(room_key):
                    # I am the owner: create local session and register my sid.
                    session = self.session_manager.find_or_create_session(params, self.is_served_locally)
                    setattr(session, "room", room)
                    self._ensure_controller_poll(session)

                    if self.session_manager.n_players > 0:
                        session.add_agent(agent_id, url_params=params)
                    session.register_websocket(agent_id, sid)
                    if not self.is_served_locally:
                        _write_state_to_store(session, waiting_id,
                                              20 + time.time() + self.max_game_time if self.session_manager.max_game_time else 0)

                    # Publish the canonical session_id so non-owners can reply with the real id.
                    self._set_canonical_session_id(room_key, session.id)

                    # Listen for remote join/intent/disconnect
                    self._start_ctrl_loop(room_key, session)

                    if session.is_full(self.session_manager.n_players):
                        print(f"[SessionApp] (owner) Session {session.id} full, starting game")
                        session._started_global = True
                        self._start_once(room_key, session)

                    emit("joined", {"agent_id": agent_id, "session_id": session.id})

                else:
                    # Not the owner: do not create a local session; forward to owner.
                    self._sid_to_roomkey[sid] = room_key
                    self._pub(room_key, "join", {
                        "agent_id": agent_id,
                        "sid": sid,
                        "url_params": params,
                    })
                    # Wait briefly for the owner to publish the canonical session_id.
                    canon_id = None
                    deadline = time.time() + 2.0
                    while time.time() < deadline and not canon_id:
                        canon_id = self._get_canonical_session_id(room_key)
                        if not canon_id:
                            self.socketio.sleep(0.05)

                    emit("joined", {
                        "agent_id": agent_id,
                        "session_id": canon_id or room_key  # fallback rarely used
                    })

            else:
                # SINGLE WORKER (or no room provided): original behavior.
                session = self.session_manager.find_or_create_session(params, self.is_served_locally)
                room = f"session:{session.id}"
                join_room(room)
                setattr(session, "room", room)
                self._ensure_controller_poll(session)

                if self.session_manager.n_players > 0:
                    session.add_agent(agent_id, url_params=params)
                session.register_websocket(agent_id, sid)

                if session.is_full(self.session_manager.n_players):
                    print(f"[CTRL] Session {session.id} full (remote)")
                    self._start_once(room_key, session)
                    session._started_global = True
                if not self.is_served_locally:
                    _write_state_to_store(session, waiting_id,
                                      20 + time.time() + self.max_game_time if self.session_manager.max_game_time else 0)

                emit("joined", {"agent_id": agent_id, "session_id": session.id})

        @self.socketio.on("intent")
        def handle_intent(data):
            # Try local session (owner) first
            session = self.session_manager.get_session(data.get("session_id", ""))
            if session:
                action = data.get("action")
                # Normalize: always dict for downstream game code
                if isinstance(action, str):
                    action = {"type": action}
                elif action is None:
                    action = {}
                session.engine.submit_intent(data["agent_id"], action)
                return

            # Non-owner: forward to owner via room key
            sid = request.sid
            room_key = self._sid_to_roomkey.get(sid)
            if self._redis and room_key:
                action = data.get("action")
                # Normalize: always dict for downstream game code
                if isinstance(action, str):
                    action = {"type": action}
                elif action is None:
                    action = {}
                self._pub(room_key, "intent", {
                    "agent_id": data["agent_id"],
                    "action": action,
                })

        @self.socketio.on("disconnect")
        def on_disconnect(*_args, **_kwargs):
            sid = request.sid
            print(f"[WS] disconnect handler triggered for sid={sid}")
            for session in self.session_manager.sessions.values():
                print(f'[WS] checking session {session.id} with agents {session._agents}')
                with session._maps_lock:  # <— NEW
                    print("[WS] current sid_to_agent map:", session.sid_to_agent)
                    agent_id = session.sid_to_agent.get(sid)
                    print(agent_id)
                    if agent_id:
                        if not session.is_full(self.session_manager.n_players):
                            print(f"[WS] {agent_id} disconnected before game start, removing agent")
                            if hasattr(session.engine.game, "remove_agent"):
                                session.engine.game.remove_agent(agent_id)
                            if agent_id in session._agents:
                                session._agents.remove(agent_id)
                            session.recorder.unregister_agent(agent_id)
                        print(f"[WS] {agent_id} disconnected")
                        session.websockets.pop(agent_id, None)
                        session.agent_to_sid.pop(agent_id, None)
                        print(f"[WS] popping sid {sid} from sid_to_agent")
                        print(f"[WS] before pop: {session.sid_to_agent}")
                        session.sid_to_agent.pop(sid, None)
                        print(f"[WS] after pop: {session.sid_to_agent}")
                        if session._started_global and len(session.sid_to_agent) < 1:
                            session.handle_game_end(self.is_served_locally)
                        else:
                            if not self.is_served_locally:
                                _write_state_to_store(session, None)

                        break
            else:
                # Non-owner: notify owner
                room_key = self._sid_to_roomkey.pop(sid, None)
                if self._redis and room_key:
                    self._pub(room_key, "disconnect", {"sid": sid})
                    print(f"[WS] remote disconnect forwarded for sid={sid}")
