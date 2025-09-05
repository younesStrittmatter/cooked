# lobby/main.py
from flask import Flask, request, jsonify, redirect, make_response
from urllib.parse import urlencode
import hashlib, os, secrets, base64, threading, time, logging

app = Flask(__name__)

# -----------------------------
# Config / Env
# -----------------------------
_raw = [s.strip() for s in os.getenv("SHARD_URLS", "").split(",") if s.strip()]
SHARDS = [s[:-1] if s.endswith("/") else s for s in _raw] or ["http://localhost:8080"]
ROOM_SIZE = int(os.getenv("ROOM_SIZE", "2"))
ID_TTL = int(os.getenv("ID_TTL_SECONDS", "1800"))   # 30 minutes
DEBOUNCE_TTL = int(os.getenv("DEBOUNCE_TTL", "3"))  # 3 seconds for prefetch/debounce
PORT = int(os.getenv("PORT", "8080"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("lobby")

# -----------------------------
# Helpers (ids / shards / urls)
# -----------------------------
def pick_shard(game_id: str) -> str:
    """Stable shard selection by hashing the room id."""
    h = hashlib.sha256(game_id.encode()).digest()
    return SHARDS[h[0] % len(SHARDS)]

def new_game_id(nbytes: int = 8) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(nbytes)).decode().rstrip("=")

def build_join_url(shard_base: str, room: str, forward: dict) -> str:
    q = {"room": room, **forward}
    return f"{shard_base}?{urlencode(q, doseq=True)}"

def forward_params(args) -> dict:
    """
    Preserve all multi-value query params except 'room'.
    We DO forward lobby_player_id and PROLIFIC_PID to shards (harmless / useful for logs).
    """
    m = request.args.to_dict(flat=False)
    m.pop("room", None)
    return m

def _redirect(location: str, code: int = 302):
    resp = redirect(location, code=code)
    # Prevent intermediaries from retrying/caching joins
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp

# -----------------------------
# In-memory batching (singleton)
# -----------------------------
assign_lock = threading.Lock()
_room_lock = threading.Lock()
_current_room_id = None
_current_room_count = 0
_current_shard_idx = 0
_room_seq = 0

def _format_room_id(seq: int) -> str:
    return f"room_{seq:06d}"

def assign_auto_room_in_memory() -> tuple[str, str, int]:
    """
    Groups arrivals into rooms of size ROOM_SIZE.
    Round-robins rooms across SHARDS.
    Returns (room_id, shard_url, shard_index_for_room).
    """
    global _current_room_id, _current_room_count, _current_shard_idx, _room_seq
    with _room_lock:
        # start a new room if none is open or the current one is full
        if _current_room_id is None or _current_room_count >= ROOM_SIZE:
            room_id = _format_room_id(_room_seq)
            shard = SHARDS[_current_shard_idx]
            _current_room_id = room_id
            _current_room_count = 0
            _room_seq += 1
            _current_shard_idx = (_current_shard_idx + 1) % len(SHARDS)
        else:
            room_id = _current_room_id
            shard = SHARDS[(_current_shard_idx - 1) % len(SHARDS)]

        _current_room_count += 1
        if _current_room_count >= ROOM_SIZE:
            # close the room
            _current_room_id = None
            _current_room_count = 0

        return room_id, shard, (_current_shard_idx - 1) % len(SHARDS)

# -----------------------------
# Prefetch/bot guard + debounce
# -----------------------------
BOT_UAS = (
    "Slackbot", "Twitterbot", "facebookexternalhit", "Discordbot",
    "Google-InspectionTool", "curl", "Wget", "Pingdom", "Uptime"
)

def is_probable_prefetch_or_bot(req) -> bool:
    ua = (req.headers.get("User-Agent") or "").lower()
    if any(b.lower() in ua for b in BOT_UAS):
        return True
    # Link previews/prefetchers often signal intent
    purpose = (
        req.headers.get("Purpose")
        or req.headers.get("Sec-Purpose")
        or req.headers.get("X-Purpose")
        or ""
    ).lower()
    if purpose in ("prefetch", "preview", "prerender"):
        return True
    # Real user-initiated navigations have Sec-Fetch-User: ?1
    if req.headers.get("Sec-Fetch-User") != "?1":
        return True
    return False

_debounce = {}
_debounce_lock = threading.Lock()

def debounce_key(req) -> str:
    return f"{req.remote_addr}|{req.headers.get('User-Agent','')}|{req.path}"

def seen_very_recent(req) -> bool:
    now = time.time()
    key = debounce_key(req)
    with _debounce_lock:
        # purge
        for k, exp in list(_debounce.items()):
            if exp <= now:
                _debounce.pop(k, None)
        if key in _debounce:
            return True
        _debounce[key] = now + DEBOUNCE_TTL
        return False

# -----------------------------
# Idempotency maps (lobby_id + PROLIFIC_PID)
# -----------------------------
# Store by *either* key: "lid:<lobby_id>" or "prolific:<pid>"
# Value: {"room":..., "join_url":..., "shard":..., "exp": epoch}
_id_map: dict[str, dict] = {}
_id_lock = threading.Lock()

def key_lid(lid: str) -> str:
    return f"lid:{lid}"

def key_prolific(pid: str) -> str:
    return f"prolific:{pid}"

def _purge_ids(now: float):
    dead = [k for k, v in _id_map.items() if v.get("exp", 0) <= now]
    for k in dead:
        _id_map.pop(k, None)

def _remember(keys: list[str], room: str, join_url: str, shard: str):
    with _id_lock:
        _purge_ids(time.time())
        rec = {"room": room, "join_url": join_url, "shard": shard, "exp": time.time() + ID_TTL}
        for k in keys:
            _id_map[k] = rec

def _lookup_any(keys: list[str]):
    with _id_lock:
        _purge_ids(time.time())
        for k in keys:
            rec = _id_map.get(k)
            if rec:
                return rec
        return None

def request_keys() -> tuple[list[str], str | None, str | None]:
    """Return lookup keys (ordered), and raw ids (lid, prolific)."""
    lid = request.args.get("lobby_player_id") or None
    prolific = request.args.get("PROLIFIC_PID") or request.args.get("prolific_pid") or None
    keys = []
    if lid:
        keys.append(key_lid(lid))
    if prolific:
        keys.append(key_prolific(prolific))
    return keys, lid, prolific

# -----------------------------
# Routes
# -----------------------------
@app.get("/healthz")
def healthz():
    with _room_lock:
        open_room = _current_room_id
        open_count = _current_room_count
    with _id_lock:
        id_cache_size = len(_id_map)
    return {
        "ok": True,
        "shards": SHARDS,
        "room_size": ROOM_SIZE,
        "open_room": open_room,
        "open_room_count": open_count,
        "id_cache_size": id_cache_size
    }, 200

@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "shards": SHARDS,
        "hint": "Use /play_auto or /new_auto (idempotent; uses lobby_player_id and PROLIFIC_PID); /new and /where are stateless."
    }), 200

@app.get("/new")
def new_game():
    # Original behavior: random room (not batched)
    room = request.args.get("room") or new_game_id()
    shard = pick_shard(room)
    fwd = forward_params(request.args)
    join_url = build_join_url(shard, room, fwd)
    return jsonify({"room": room, "join_url": join_url, "shard": shard})

@app.get("/where")
def where():
    room = request.args.get("room")
    if not room:
        return jsonify(error="missing room"), 400
    shard = pick_shard(room)
    fwd = forward_params(request.args)
    join_url = build_join_url(shard, room, fwd)
    return jsonify({"room": room, "join_url": join_url, "shard": shard})

@app.get("/play")
def play():
    # Original stateless demo: create (or reuse provided) room and redirect
    room = request.args.get("room") or new_game_id()
    shard = pick_shard(room)
    fwd = forward_params(request.args)
    join_url = build_join_url(shard, room, fwd)
    return _redirect(join_url, 302)

@app.get("/play_auto")
def play_auto():
    """
    Idempotent auto-batching with guards:
    - If PROLIFIC_PID present (no lobby_player_id), we use that as the key (no mint/redirect).
    - Else if lobby_player_id missing, we only mint for real user-initiated navs
      (Sec-Fetch-User:?1) and not within debounce window; otherwise 204.
    - First-time key: assign once (under assign_lock), remember under all keys we have.
    - Seen key: reuse prior assignment (REPLAY).
    """
    keys, lid, prolific = request_keys()

    # 1) If neither id is present, decide whether to mint a lobby id or ignore.
    if not keys:
        if is_probable_prefetch_or_bot(request) or seen_very_recent(request):
            # Ignore non-user navs / rapid duplicate probes
            log.info("IGNORE prefetch/bot path=%s ua=%s", request.path, request.headers.get("User-Agent", ""))
            resp = make_response("", 204)
            resp.headers["Cache-Control"] = "no-store"
            return resp
        # Mint a lobby id and redirect back once (MINT)
        lid = new_game_id(9)
        q = request.args.to_dict(flat=False)
        q["lobby_player_id"] = [lid]
        location = request.path + "?" + urlencode(q, doseq=True)
        log.info("MINT lid=%s path=%s", lid, request.path)
        return _redirect(location, 302)

    # 2) If we have any key (lid or prolific), try to reuse
    prev = _lookup_any(keys)
    if prev:
        log.info("REPLAY keys=%s room=%s shard=%s", keys, prev["room"], prev["shard"])
        return _redirect(prev["join_url"], 302)

    # 3) First-time key(s): assign exactly once
    fwd = forward_params(request.args)
    explicit_room = request.args.get("room")

    with assign_lock:
        # Double-check under lock to avoid races
        prev2 = _lookup_any(keys)
        if prev2:
            log.info("REPLAY-LATE keys=%s room=%s shard=%s", keys, prev2["room"], prev2["shard"])
            return _redirect(prev2["join_url"], 302)

        if explicit_room:
            room = explicit_room
            shard = pick_shard(explicit_room)
            shard_idx = None
        else:
            room, shard, shard_idx = assign_auto_room_in_memory()

        join_url = build_join_url(shard, room, fwd)
        # Remember under ALL known keys (lid/prolific)
        store_keys = []
        if lid:
            store_keys.append(key_lid(lid))
        if prolific:
            store_keys.append(key_prolific(prolific))
        # If only one is present, we still store just that one.
        _remember(store_keys, room, join_url, shard)
        tag = "ASSIGN_P" if prolific and not lid else "ASSIGN"
        log.info("%s keys=%s room=%s shard=%s", tag, store_keys, room, shard)

    return _redirect(join_url, 302)

@app.get("/new_auto")
def new_auto():
    """
    JSON variant with the same idempotency and guards.
    """
    keys, lid, prolific = request_keys()

    if not keys:
        if is_probable_prefetch_or_bot(request) or seen_very_recent(request):
            resp = make_response(jsonify(ok=True, note="ignored non-user navigation"), 204)
            resp.headers["Cache-Control"] = "no-store"
            return resp
        lid = new_game_id(9)
        q = request.args.to_dict(flat=False)
        q["lobby_player_id"] = [lid]
        location = request.path + "?" + urlencode(q, doseq=True)
        log.info("MINT lid=%s path=%s", lid, request.path)
        return _redirect(location, 302)

    prev = _lookup_any(keys)
    if prev:
        log.info("REPLAY keys=%s room=%s shard=%s", keys, prev["room"], prev["shard"])
        return jsonify({
            "room": prev["room"],
            "join_url": prev["join_url"],
            "shard": prev["shard"],
            "room_size": ROOM_SIZE,
            "idempotent": True
        })

    fwd = forward_params(request.args)
    explicit_room = request.args.get("room")

    with assign_lock:
        prev2 = _lookup_any(keys)
        if prev2:
            log.info("REPLAY-LATE keys=%s room=%s shard=%s", keys, prev2["room"], prev2["shard"])
            return jsonify({
                "room": prev2["room"],
                "join_url": prev2["join_url"],
                "shard": prev2["shard"],
                "room_size": ROOM_SIZE,
                "idempotent": True
            })

        if explicit_room:
            room = explicit_room
            shard = pick_shard(explicit_room)
            shard_idx = None
        else:
            room, shard, shard_idx = assign_auto_room_in_memory()

        join_url = build_join_url(shard, room, fwd)
        store_keys = []
        if lid:
            store_keys.append(key_lid(lid))
        if prolific:
            store_keys.append(key_prolific(prolific))
        _remember(store_keys, room, join_url, shard)
        tag = "ASSIGN_P" if prolific and not lid else "ASSIGN"
        log.info("%s keys=%s room=%s shard=%s", tag, store_keys, room, shard)

    return jsonify({
        "room": room,
        "join_url": join_url,
        "shard": shard,
        "shard_index": shard_idx,
        "room_size": ROOM_SIZE,
        "idempotent": False
    })

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # For local dev; Cloud Run sets $PORT and handles binding.
    app.run(host="0.0.0.0", port=PORT, debug=False)
