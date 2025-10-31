export let agentId = null;
export let sessionId = null;
let _initPromise = null;

async function _init() {
  if (_initPromise) return _initPromise;

  _initPromise = fetch("/join", { method: "POST" })
    .then(res => res.json())
    .then(data => {
      agentId = data.agent_id;
      sessionId = data.session_id || "single";  // fallback for EngineApp
      console.log("[Engine] Joined as:", agentId, "in session:", sessionId);
    });

  return _initPromise;
}

export async function sendIntent(action) {
  await _init();

  await fetch("/intent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      agent_id: agentId,
      session_id: sessionId,
      action: action
    })
  });
}

export async function fetchState() {
  await _init();
  const res = await fetch(`/state?agent_id=${agentId}&session_id=${sessionId}`);
  return await res.json();
}

await _init();
