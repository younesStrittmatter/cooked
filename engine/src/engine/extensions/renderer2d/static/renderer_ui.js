// renderer_ui.js
import { sendIntent } from "/engine/static/engine_ui.js";

/* ---------------- asset base (robust, no template assumptions) ---------------- */


/* --------------------- image loader (no shard fallback) --------------------- */

const spriteCache = Object.create(null);
const MAX_PARALLEL_IMG = 8;
let inFlight = 0;
const queue = [];

function schedule(fn) {
  return new Promise((resolve, reject) => {
    queue.push({ fn, resolve, reject });
    pump();
  });
}
function pump() {
  while (inFlight < MAX_PARALLEL_IMG && queue.length) {
    const { fn, resolve, reject } = queue.shift();
    inFlight++;
    fn()
      .then((v) => {
        inFlight--;
        resolve(v);
        pump();
      })
      .catch((e) => {
        inFlight--;
        reject(e);
        pump();
      });
  }
}

function loadImageOnce(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`img load error: ${url}`));
    img.src = url;
  });
}

async function loadWithRetry(url, attempts = 3) {
  let backoff = 80;
  for (let i = 1; i <= attempts; i++) {
    try {
      return await loadImageOnce(url);
    } catch (e) {
      if (i === attempts) throw e;
      await new Promise((r) => setTimeout(r, backoff));
      backoff = Math.min(backoff * 2, 2000);
    }
  }
}

/**
 * Load a sprite by relative path under /sprites/
 * Example: loadSprite("world/basic-wall.png")
 */
async function loadSprite(src) {
  if (spriteCache[src]) return spriteCache[src];
  const base = (window.ASSET_BASE_URL
  ? String(window.ASSET_BASE_URL).replace(/\/+$/, "") + "/"
  : new URL("../", import.meta.url).toString());
  const url = base + "sprites/" + src;
  const img = await schedule(() => loadWithRetry(url));
  spriteCache[src] = img;
  return img;
}

/* -------------------------------- canvas -------------------------------- */

let canvas = document.getElementById("scene");
let text_overlay = document.getElementById("text_overlay");

if (!canvas) {
  canvas = document.createElement("canvas");
  canvas.id = "scene";
  canvas.width = 128;
  canvas.height = 128;
  document.body.appendChild(canvas);
}
if (!text_overlay) {
  text_overlay = document.createElement("canvas");
  text_overlay.id = "text_overlay";
  text_overlay.width = 1024;
  text_overlay.height = 1024;
  document.body.appendChild(text_overlay);
}

const ctx = canvas.getContext("2d");
ctx.imageSmoothingEnabled = false;
const ctx_text = text_overlay.getContext("2d");

let currentState = null;

/* ------------------------------- interaction ------------------------------- */

canvas.addEventListener("click", async (event) => {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const x = (event.clientX - rect.left) * sx;
  const y = (event.clientY - rect.top) * sy;

  const objects = currentState?.gameObjects || [];
  for (let i = objects.length - 1; i >= 0; i--) {
    const o = objects[i];
    if (!o.isClickable) continue;
    if (x >= o.left && x <= o.left + o.width && y >= o.top && y <= o.top + o.height) {
      await sendIntent({ type: "click", target: o.id });
      return;
    }
  }
});

/* --------------------------------- render --------------------------------- */

export async function renderScene(prevState = { gameObjects: [] }, currState = { gameObjects: [] }, t = 1) {
  currentState = currState;
  const objects = (currState?.gameObjects || []).slice();

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx_text.clearRect(0, 0, text_overlay.width, text_overlay.height);

  // back-to-front
  objects.sort((a, b) => (a.zIndex || 0) - (b.zIndex || 0));

  for (const obj of objects) {
    if (obj.class === "Basic2D") {
      const s_x = obj.normalize ? canvas.width : 1;
      const s_y = obj.normalize ? canvas.height : 1;

      const img = await loadSprite(obj.src);
      const id = obj.id;

      const currX = (obj.left ?? 0) * s_x;
      const currY = (obj.top ?? 0) * s_y;

      const prevObj = (prevState?.gameObjects || []).find((o) => o.id === id);
      const prevX = prevObj ? (prevObj.left ?? 0) * s_x : currX;
      const prevY = prevObj ? (prevObj.top ?? 0) * s_y : currY;

      const x = prevX + (currX - prevX) * t;
      const y = prevY + (currY - prevY) * t;

      const w = (obj.width ?? img.width) * s_x;
      const h = (obj.height ?? img.height) * s_y;

      ctx.drawImage(
        img,
        obj.srcX || 0,
        obj.srcY || 0,
        obj.srcW || img.width,
        obj.srcH || img.height,
        Math.round(x),
        Math.round(y),
        Math.round(w),
        Math.round(h)
      );
    }

    if (obj.class === "Text") {
      const s_x = obj.normalize ? text_overlay.width : 1;
      const s_y = obj.normalize ? text_overlay.height : 1;

      const computeXY = (o, sx, sy) => {
        const left = o?.left,
              top = o?.top,
              right = o?.right,
              bottom = o?.bottom;
        const x = left == null ? text_overlay.width - (right ?? 0) * sx : (left ?? 0) * sx;
        const y = top == null ? text_overlay.height - (bottom ?? 0) * sy : (top ?? 0) * sy;
        return { x, y };
      };

      const { x: cx, y: cy } = computeXY(obj, s_x, s_y);
      const prevObj = (prevState?.gameObjects || []).find((o) => o.id === obj.id) || {};
      const { x: px, y: py } = computeXY(prevObj, s_x, s_y);

      const x = px + (cx - px) * t;
      const y = py + (cy - py) * t;

      const lines = String(obj.content || "").split("\n");
      const fontSize = obj.fontSize || 48;
      ctx_text.font = `${fontSize}pt ${obj.fontFamily || "Arial"}`;
      ctx_text.fillStyle = obj.color || "black";
      ctx_text.textAlign = obj.align || "left";
      ctx_text.textBaseline = obj.baseline || "top";

      if (obj.top == null) {
        lines.forEach((line, i) => {
          ctx_text.fillText(
            line,
            Math.round(x),
            Math.round(y - lines.length * fontSize * 1.5 + i * fontSize * 1.5) + fontSize * 1.5
          );
        });
      } else {
        lines.forEach((line, i) => {
          ctx_text.fillText(line, Math.round(x), Math.round(y + i * fontSize * 1.5));
        });
      }
    }
  }
}

/* ------------------------------ mouse state ------------------------------ */

const mouse = { x: 0, y: 0, leftDown: false, rightDown: false };

function inCanvas(x, y) {
  return x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height;
}
function canvasPos(clientX, clientY) {
  const r = canvas.getBoundingClientRect();
  return { x: (clientX - r.left) / r.width, y: (clientY - r.top) / r.height };
}

canvas.addEventListener("mousemove", (e) => {
  const p = canvasPos(e.clientX, e.clientY);
  if (inCanvas(p.x * canvas.width, p.y * canvas.height)) {
    mouse.x = p.x;
    mouse.y = p.y;
  }
});
canvas.addEventListener("mousedown", (e) => {
  if (e.button === 0) mouse.leftDown = true;
  if (e.button === 2) mouse.rightDown = true;
});
canvas.addEventListener("mouseup", (e) => {
  if (e.button === 0) mouse.leftDown = false;
  if (e.button === 2) mouse.rightDown = false;
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

// Touch
canvas.addEventListener("touchstart", (e) => {
  const t = e.touches[0];
  const p = canvasPos(t.clientX, t.clientY);
  if (inCanvas(p.x * canvas.width, p.y * canvas.height)) {
    mouse.x = p.x;
    mouse.y = p.y;
  }
  mouse.leftDown = true;
});
canvas.addEventListener("touchmove", (e) => {
  const t = e.touches[0];
  const p = canvasPos(t.clientX, t.clientY);
  if (inCanvas(p.x * canvas.width, p.y * canvas.height)) {
    mouse.x = p.x;
    mouse.y = p.y;
  }
});
canvas.addEventListener("touchend", () => {
  mouse.leftDown = false;
});

export async function updateMouse() {
  await sendIntent({ type: "mouse", x: mouse.x, y: mouse.y, leftDown: mouse.leftDown, rightDown: mouse.rightDown });
}
