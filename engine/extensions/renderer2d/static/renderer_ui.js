const TILE_SIZE = 32;

// Grab the canvas or create one if missing
let canvas = document.getElementById("scene");
if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.id = "scene";
    canvas.width = 512;
    canvas.height = 512;
    document.body.appendChild(canvas);
}

const ctx = canvas.getContext("2d");
const spriteCache = {};

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

export async function renderScene(state) {

    const objects = state['gameObjects'] || [];

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const obj of objects) {

        if (obj.src) {
            const img = await loadSprite(obj.src);
            const x = obj.x;
            const y = obj.y;
            const w = obj.w;
            const h = obj.h;
            const sx = obj.srcX || 0;
            const sy = obj.srcY || 0;
            const sw = obj.srcW || img.width;
            const sh = obj.srcH || img.height;
            ctx.drawImage(img, sx, sy, sw, sh, x, y, w, h);
        }
    }
}