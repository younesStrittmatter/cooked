// renderer_ui.js

import {sendIntent} from "/engine/static/engine_ui.js";

let canvas = document.getElementById("scene");
let text_overlay = document.getElementById("text_overlay");
const prevPositions = {};

if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.id = "scene";
    canvas.width = 128;
    canvas.height = 128;
    document.body.appendChild(canvas);
}
if (!text_overlay) {
    text_overlay = document.createElement("text_overlay");
    text_overlay.id = "scene";
    text_overlay.width = 1024;
    text_overlay.height = 1024;
    document.body.appendChild(text_overlay);
}
const ctx = canvas.getContext("2d");
ctx.imageSmoothingEnabled = false;

const ctx_text = text_overlay.getContext("2d");

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

export async function renderScene(prevState, currState, t) {
    const objects = currState['gameObjects'] || [];
    currentState = currState;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx_text.clearRect(0, 0, text_overlay.width, text_overlay.height);

    objects.sort((a, b) => (a.zIndex || 0) - (b.zIndex || 0));

    for (const obj of objects) {
        if (obj.class === "Basic2D") {
            let s_x = canvas.width;
            let s_y = canvas.height;
            if (!obj.normalize) {
                s_x = 1.;
                s_y = 1.;
            }

            const img = await loadSprite(obj.src);
            const id = obj.id;

            const currX = obj.left * s_x;
            const currY = obj.top * s_y;

            let prevObj = (prevState.gameObjects || []).find(o => o.id === id);
            const prevX = prevObj ? prevObj.left * s_x : currX;
            const prevY = prevObj ? prevObj.top * s_y : currY;

            const x = prevX + (currX - prevX) * t;
            const y = prevY + (currY - prevY) * t;

            const w = obj.width * s_x;
            const h = obj.height * s_y;

            ctx.drawImage(img, obj.srcX || 0, obj.srcY || 0, obj.srcW || img.width, obj.srcH || img.height,
                Math.round(x), Math.round(y), Math.round(w), Math.round(h));
        }
        if (obj.class === "Text") {

            let s_x = text_overlay.width;
            let s_y = text_overlay.height;
            if (!obj.normalize) {
                s_x = 1.;
                s_y = 1.;
            }


            let currX
            if (obj.left === null) {
                currX = text_overlay.width - obj.right * s_x;
            } else {
                currX = obj.left * s_x;
            }

            let currY;
            if (obj.top === null) {
                currY = text_overlay.height - obj.bottom * s_y;
            } else {
                currY = obj.top * s_y;
            }


            let prevObj = (prevState.gameObjects || []).find(o => o.id === obj.id);

            let prevX;
            if (prevObj.left === null) {
                prevX = text_overlay.width - (prevObj ? prevObj.right * s_x : currX);
            } else {
                prevX = prevObj ? prevObj.left * s_x : currX;
            }

            let prevY;
            if (prevObj.top === null) {
                prevY = text_overlay.height - (prevObj ? prevObj.bottom * s_y : currY);
            } else {
                prevY = prevObj ? prevObj.top * s_y : currY;
            }

            const x = prevX + (currX - prevX) * t;
            const y = prevY + (currY - prevY) * t;

            const lines = (obj.content || '').split('\n');
            const fontSize = obj.fontSize || 48;
            ctx_text.font = `${fontSize}pt ${obj.fontFamily || 'Arial'}`;
            ctx_text.fillStyle = obj.color || 'black';
            ctx_text.textAlign = obj.align || 'left';
            ctx_text.textBaseline = obj.baseline || 'top';

            if (obj.top === null) {
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