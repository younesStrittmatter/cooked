import os, json
import uuid
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from flask import Flask, jsonify, render_template, request
import logging
import re


try:
    from google.cloud import storage
except Exception:
    storage = None

app = Flask(__name__, template_folder="templates", static_folder="static")

# Env from deploy
SHARD_URLS = [u.strip() for u in os.getenv("SHARD_URLS", "").split(",") if u.strip()]
GAME_CAPACITY = int(os.getenv("GAME_CAPACITY", "2"))
STATE_GS = (os.getenv("SHARD_STATE_GS_URL", "").strip())  # e.g. gs://my-bucket/shards

my_pid = None
my_shard_idx = None

def _build_shard_map(urls):
    """Map {shard_id:int -> url:str} by parsing '...-shard-<id>-...' (or end/boundary)."""
    m = {}
    for u in urls:
        # matches: ...-shard-12-..., ...-shard-12., ...-shard-12 (end)
        hit = re.search(r'shard-(\d+)(?:[.\-]|$)', u)
        if not hit:
            logging.warning("Could not parse shard id from URL: %s", u)
            continue
        sid = int(hit.group(1))
        if sid in m and m[sid] != u:
            logging.warning("Duplicate shard id %d for URLs %s and %s", sid, m[sid], u)
        m[sid] = u
    return m

SHARD_BY_ID = _build_shard_map(SHARD_URLS)

def shard_url_for_id(shard_id: int) -> str:
    """Return URL whose hostname contains '-shard-<shard_id>-', ignoring list order."""
    sid = int(shard_id)
    if sid in SHARD_BY_ID:
        return SHARD_BY_ID[sid]
    # Fallback: stable pick so we never crash if an id is missing
    logging.error("Shard id %s not found in SHARD_URLS; falling back deterministically.", sid)
    return SHARD_URLS[hash(str(sid)) % len(SHARD_URLS)]


def _parse_gs(url: str):
    if not url.startswith("gs://"): return None, None
    rest = url[5:]
    if "/" in rest:
        b, p = rest.split("/", 1)
        return b, p.strip("/")
    return rest, ""


BUCKET_NAME, PREFIX = _parse_gs(STATE_GS)
_gcs = storage.Client() if (storage and BUCKET_NAME) else None
_bucket = _gcs.bucket(BUCKET_NAME) if _gcs else None


def _state_blob_path(shard_id: int) -> str:
    # gs://<bucket>/<prefix>/<id>/state/state.json
    if PREFIX:
        return f"{PREFIX}/{shard_id}/state/state.json"
    return f"{shard_id}/state/state.json"

def _empty_state():
    return {
        "nr_players": 0,
        "waiting_players": [],
        "started": False,
        "end_time": None,
        "blocked_until": 0,
    }





def _read_state_with_meta(shard_id: int):
    """Return (state_dict, generation) or (None, None) if missing."""
    if not _bucket:
        return None, None
    path = _state_blob_path(shard_id)
    try:
        blob = _bucket.blob(path)
        if not blob.exists():
            return None, None
        data = blob.download_as_bytes()
        st = json.loads(data)
        return st, blob.generation
    except Exception as e:
        app.logger.warning(f"[state] read failed for {path}: {e}")
        return None, None


def _write_state_if_generation(shard_id: int, payload: dict, expected_generation: int | None):
    """
    Atomic write using if-generation-match.
    If object doesn't exist yet, expected_generation must be None.
    Returns new generation on success, None on precondition failure/error.
    """
    if not _bucket:
        return None
    path = _state_blob_path(shard_id)
    print('[state] writing to', path, 'expecting gen', expected_generation)
    try:
        blob = _bucket.blob(path)
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if expected_generation is None:
            # only succeed if object does NOT exist
            blob.upload_from_string(body, if_generation_match=0, content_type="application/json", timeout=30)
        else:
            blob.upload_from_string(body, if_generation_match=expected_generation, content_type="application/json",
                                    timeout=30)
        # prevent stale caching by downstream clients
        blob.cache_control = "no-store"
        blob.patch()
        return blob.generation
    except Exception as e:
        # precondition failed (race) or other error
        return None


import time, random

_CONNECT_BACKOFFS = [0.02, 0.05, 0.1, 0.2]


def _is_started(st: dict) -> bool:
    return bool(st.get("started", st.get("_started_global", False)))


def _find_shard_to_join(pid: str):
    for shard_idx, url in enumerate(SHARD_URLS, start=1):
        for backoff in _CONNECT_BACKOFFS + [0.25, 5]:
            st, gen = _read_state_with_meta(shard_idx)
            if not st:
                st = _empty_state()
                new_gen = _write_state_if_generation(shard_idx, st, gen)
                if new_gen is not None:
                    return None
                time.sleep(backoff * (1.0 + 0.2 * random.random()))

            if st.get("last_access", 0) + 3600 * 6 < time.time():
                st = _empty_state()
                st["last_access"] = time.time()
                new_gen = _write_state_if_generation(shard_idx, st, gen)
                if new_gen is not None:
                    return None
                time.sleep(backoff * (1.0 + 0.2 * random.random()))


            if not st.get('started', True):
                if not "last_checked" in st:
                    before = st.get("waiting_players", []).copy()
                    st["waiting_players"] = []
                    if before != st["waiting_players"]:
                        new_gen = _write_state_if_generation(shard_idx, st, gen)
                        if new_gen is not None:
                            return None
                        time.sleep(backoff * (1.0 + 0.2 * random.random()))
                else:
                    now = time.time()
                    waiting = st.get("waiting_players", [])
                    before = st.get("waiting_players", []).copy()
                    for entry in st.get("last_checked", []):
                        if entry.get('time', 0) + 30 > now:
                            continue
                        if entry.get('pid') in waiting:
                            waiting.remove(entry.get('pid'))
                    if before != waiting:
                        new_gen = _write_state_if_generation(shard_idx, st, gen)
                        if new_gen is not None:
                            return None
                        time.sleep(backoff * (1.0 + 0.2 * random.random()))

                if int(st.get('nr_players')) + len(st.get("waiting_players", [])) >= GAME_CAPACITY:
                    if st.get("blocked_until", 0) + 60 < time.time():
                        # game should have started but didn't; reset
                        if st.get('waiting_players'):
                            st['waiting_players'] = []
                            new_gen = _write_state_if_generation(shard_idx, st, gen)
                            if new_gen is not None:
                                return None
                            time.sleep(backoff * (1.0 + 0.2 * random.random()))
                        else:
                            st = _empty_state()
                            new_gen = _write_state_if_generation(shard_idx, st, gen)
                            if new_gen is not None:
                                return None
                            time.sleep(backoff * (1.0 + 0.2 * random.random()))

            if st.get('started', False):
                now = time.time()
                if st.get("end_time", None) is None:
                    break
                elif st.get("end_time", None) < now and now > st.get("blocked_until", 0):
                    st['started']  = False
                    st['end_time'] = None
                    st['nr_players'] = 0
                    st['waiting_players'] = []
                else:
                    break

                new_gen = _write_state_if_generation(shard_idx, st, gen)
                if new_gen is not None:
                    return None
                time.sleep(backoff * (1.0 + 0.2 * random.random()))



    def fill_of(i):
        st, _ = _read_state_with_meta(i)
        if not st:
            return -1
        return len(st.get("waiting_players", [])) if not _is_started(st) else -1

    candidates = list(range(1, len(SHARD_URLS) + 1))
    candidates.sort(key=fill_of, reverse=True)

    # 2) try to join; CAS with retries per shard
    for shard_id in candidates:
        for backoff in _CONNECT_BACKOFFS + [0.25, 0.5]:
            st, gen = _read_state_with_meta(shard_id)
            if not st:
                break
            if _is_started(st) or st.get("nr_players", 0) >= GAME_CAPACITY:
                break  # try next shard
            now = time.time()
            blocked_until = float(st.get("blocked_until", 0))
            if now < blocked_until:
                break
            waiting = st.get("waiting_players", [])
            if len(waiting) + st.get("nr_players", 0) >= GAME_CAPACITY:
                break  # full; next shard
            if pid not in waiting:
                waiting.append(pid)
            else:
                return shard_id
            st["waiting_players"] = waiting
            last_access = st.get("last_access", 0)
            if now > last_access:
                last_access = now
            st["last_access"] = last_access
            last_checked = st.get("last_checked", [])
            last_checked = [entry for entry in last_checked if entry.get('pid') != pid]
            last_checked.append({'time': now, 'pid': pid})
            st["last_checked"] = last_checked
            new_gen = _write_state_if_generation(shard_id, st, gen)
            if new_gen is not None:
                return shard_id
            time.sleep(backoff * (1.0 + 0.2 * random.random()))
    return None


def _check_own_shard(shard_id, pid):
    if not shard_id:
        return None
    try:
        shard_id = int(shard_id)
    except Exception:
        return None

    st, gen = _read_state_with_meta(shard_id)
    if not st:
        return None

    # use this as a heartbeat to keep the waiting list:
    for backoff in _CONNECT_BACKOFFS + [0.25, 0.5]:
        now = time.time()
        last_checked_state = st.get("last_checked", [])
        last_checked_state = [entry for entry in last_checked_state if entry.get('pid') != pid]
        last_checked_state.append({'time': now, 'pid': pid})
        st['last_checked'] = last_checked_state
        new_gen = _write_state_if_generation(shard_id, st, gen)
        if new_gen is not None:
            break
        time.sleep(backoff * (1.0 + 0.2 * random.random()))

    # If room has been marked started, redirect now
    if len(st.get("waiting_players", [])) + st.get('nr_players', 0) >= GAME_CAPACITY:
        for backoff in _CONNECT_BACKOFFS + [0.25, 0.5]:
            blocked_now = time.time()
            st, gen = _read_state_with_meta(shard_id)
            if not st:
                return None
            st["blocked_until"] = blocked_now + 15
            new_gen = _write_state_if_generation(shard_id, st, gen)
            if new_gen is not None:
                return shard_url_for_id(shard_id)
            time.sleep(backoff * (1.0 + 0.2 * random.random()))
        return None


    # Otherwise still waiting
    return None


def _remove_from_shard(pid, shard_id):
    if not shard_id:
        return
    try:
        shard_id = int(shard_id)
    except Exception:
        return
    if not pid:
        return

    for backoff in _CONNECT_BACKOFFS + [0.25]:
        st, gen = _read_state_with_meta(shard_id)
        if st is None:
            return
        waiting = st.setdefault("waiting_players", [])
        changed = False
        if pid in waiting:
            waiting.remove(pid)
            changed = True
        if not changed:
            return
        if _write_state_if_generation(shard_id, st, gen) is not None:
            return
        time.sleep(backoff)


def _merge_query(base_url: str, add_params: dict[str, list[str]]):
    """
    Merge add_params into base_url's query string.
    - Preserves existing params unless overridden by incoming keys
    - Supports multi-valued params (lists)
    """
    u = urlparse(base_url)
    existing = {}
    for k, v in parse_qsl(u.query, keep_blank_values=True):
        existing.setdefault(k, []).append(v)

    # override/extend with incoming
    for k, vals in add_params.items():
        # Replace existing key with provided list (common/expected behavior)
        existing[k] = vals if isinstance(vals, list) else [vals]

    # urlencode with doseq to support lists
    new_query = urlencode(existing, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))


@app.get("/")
def lobby_wait():
    return render_template("wait.html")


@app.get("/error")
def error():
    return render_template("error.html")


@app.post("/connect")
def connect():
    params = request.args.to_dict(flat=False)
    pid = params.get("pid", [None])[0]
    shard_id = _find_shard_to_join(pid)
    if shard_id is None:
        return jsonify({"connected": False, "reason": "no-shard"}), 200
    return jsonify({"connected": True, "shard_id": shard_id}), 200


@app.post("/check_back")
def check_back():
    print('possible urls:', SHARD_URLS)
    params = request.args.to_dict(flat=False)
    # accept shard_id via query (as your JS does)
    shard_id = params.get("shard_id", [None])[0]
    pid = params.get("pid", [None])[0]

    url = _check_own_shard(shard_id, pid)
    if url:
        # preserve original query string
        params = request.args.to_dict(flat=False)
        redirect_url = _merge_query(url, params) if params else url
        return jsonify({"redirect": redirect_url}), 200

    return jsonify({"wait": True}), 200


@app.post("/disconnect")
def disconnect():
    params = request.args.to_dict(flat=False)
    pid = params.get("pid", [None])[0]
    shard_id = params.get("shard_id", [None])[0]
    if pid is not None and shard_id is not None:
        _remove_from_shard(pid, shard_id)
    return jsonify({"ok": True}), 200
