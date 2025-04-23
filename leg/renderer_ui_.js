// renderer_ui.js

import {sendIntent} from "/engine/static/engine_ui.js";

let canvas = document.getElementById("scene");
const prevPositions = {};

if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.id = "scene";
    canvas.width = 128;
    canvas.height = 128;
    document.body.appendChild(canvas);
}
const ctx = canvas.getContext("2d");
ctx.imageSmoothingEnabled = false;

const spriteCache = {};
let currentState = null;

function loadSprite(src) {
    return new Promise((resolve) => {
        if (spriteCache[src]) return resolve(spriteCache[src]);
        const img = new Image();
        img.src = `/sprites/${src}`;
        img.onload = () => {
            spriteCache[src] = img;
            resolve(img);
        };
    });
}

canvas.addEventListener("click", async (event) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const clickX = (event.clientX - rect.left) * scaleX;
    const clickY = (event.clientY - rect.top) * scaleY;

    const objects = currentState?.gameObjects || [];
    for (const obj of objects) {
        if (obj.isClickable) {
            const withinX = clickX >= obj.left && clickX <= obj.left + obj.width;
            const withinY = clickY >= obj.top && clickY <= obj.top + obj.height;
            if (withinX && withinY) {
                await sendIntent({type: "click", target: obj.id});
                return;
            }
        }
    }
});

export async function renderScene(state) {
    currentState = state;
    const objects = state['gameObjects'] || [];
    objects.sort((a, b) => (a.zIndex || 0) - (b.zIndex || 0));
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const rect = canvas.getBoundingClientRect();
    let scale_x = 1.
    let scale_y = 1.;
    for (const obj of objects) {
        if (obj.src) {

            if (obj.normalize) {
                scale_x = canvas.width;
                scale_y = canvas.height;
            }
            const img = await loadSprite(obj.src);
            const objId = obj.id;
            const targetX = obj.left * scale_x;
            const targetY = obj.top * scale_y;
            let prev = prevPositions[objId] || {x: targetX, y: targetY};

            const maxStep = 16; // Max pixels per frame — tweak this
            let dx = targetX - prev.x;
            let dy = targetY - prev.y;

            const dist = Math.hypot(dx, dy);

// Snap if teleporty
            if (dist > 50) {
                prev = {x: targetX, y: targetY};
                dx = 0;
                dy = 0;
            }

            if (dist > maxStep) {
                dx = (dx / dist) * maxStep;
                dy = (dy / dist) * maxStep;
            }

            const x = prev.x + dx;
            const y = prev.y + dy;

            prevPositions[objId] = {x, y};
            const w = obj.width * scale_x;
            const h = obj.height * scale_y;

            const sx = obj.srcX || 0;
            const sy = obj.srcY || 0;
            const sw = obj.srcW || img.width;
            const sh = obj.srcH || img.height;
            ctx.drawImage(img, sx, sy, sw, sh, Math.round(x), Math.round(y), Math.round(w), Math.round(h));
            scale_x = 1.;
            scale_y = 1.;
        }
    }
}

let mouse = {
    x: 0,
    y: 0,
    leftDown: false,
    rightDown: false
};

function isWithinCanvas(x, y) {
    return x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height;
}

function getCanvasPosition(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = 1 / rect.width;
    const scaleY = 1 / rect.height;
    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
}

// Mouse events
canvas.addEventListener("mousemove", (e) => {
    const pos = getCanvasPosition(e.clientX, e.clientY);
    if (isWithinCanvas(pos.x, pos.y)) {
        mouse.x = pos.x;
        mouse.y = pos.y;
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

// ✅ Touch events (mobile-friendly)
canvas.addEventListener("touchstart", (e) => {
    const touch = e.touches[0];
    const pos = getCanvasPosition(touch.clientX, touch.clientY);
    if (isWithinCanvas(pos.x, pos.y)) {
        mouse.x = pos.x;
        mouse.y = pos.y;
    }
    mouse.leftDown = true;
});

canvas.addEventListener("touchmove", (e) => {
    const touch = e.touches[0];
    const pos = getCanvasPosition(touch.clientX, touch.clientY);
    if (isWithinCanvas(pos.x, pos.y)) {
        mouse.x = pos.x;
        mouse.y = pos.y;
    }
});

canvas.addEventListener("touchend", (e) => {
    mouse.leftDown = false;
});

// Disable context menu on right-click
canvas.addEventListener("contextmenu", (e) => e.preventDefault());

export async function updateMouse() {
    await sendIntent({
        type: "mouse",
        x: mouse.x,
        y: mouse.y,
        leftDown: mouse.leftDown,
        rightDown: mouse.rightDown
    })
}