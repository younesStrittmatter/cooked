import { io } from "https://cdn.socket.io/4.7.2/socket.io.esm.min.js";

export let agentId = null;
export let sessionId = null;
let socket = null;
let onStateUpdate = null;

export function setStateUpdateHandler(handler) {
    onStateUpdate = handler;
}
const urlParams = new URLSearchParams(window.location.search);
const auth = Object.fromEntries(urlParams.entries()); // <- send these
// Connect with Socket.IO
socket = io("/", {
  transports: ["websocket"], // <-- no polling
  upgrade: false,            // <-- skip upgrade handshake
  forceNew: true,
    auth
}); // Automatically uses same host + port, and correct protocol

socket.on("connect", () => {
    console.log("[Engine] Socket.IO connected");
});

socket.on("joined", (msg) => {
    agentId = msg.agent_id;
    sessionId = msg.session_id || "single";
    console.log("[Engine] Joined as:", agentId, "in session:", sessionId);
});

socket.on("state", (msg) => {

    if (onStateUpdate) {
        onStateUpdate(msg);
    }
});

socket.on("disconnect", () => {
    console.warn("[Engine] Socket.IO disconnected");
});

socket.on("redirect", (data) => {
    window.location.href = data.url;
});

export function sendIntent(action) {
    if (socket && socket.connected && agentId && sessionId) {
        socket.emit("intent", {
            agent_id: agentId,
            session_id: sessionId,
            action: action
        });
    }
}

window.addEventListener("beforeunload", () => {
  socket.disconnect();
});