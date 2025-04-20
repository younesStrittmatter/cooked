const TILE_SIZE = 32;

// Grab the canvas or create one if missing
let canvas = document.getElementById("scene");
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
    objects.sort((a, b) => (a.zIndex || 0) - (b.zIndex || 0));

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const obj of objects) {


        if (obj.src) {
            const img = await loadSprite(obj.src);
            const x = obj.left;
            const y = obj.top;
            const w = obj.width;
            const h = obj.height;
            const sx = obj.srcX || 0;
            const sy = obj.srcY || 0;
            const sw = obj.srcW || img.width;
            const sh = obj.srcH || img.height;
            ctx.drawImage(img, sx, sy, sw, sh, x, y, w, h);
        }
    }
}