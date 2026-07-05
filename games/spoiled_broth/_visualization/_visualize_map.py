"""
Visualize and store maps from sprites and a map specification image (maps)
"""

from pathlib import Path

from PIL import Image

TILE_PX = 16
OUTPUT_SCALE = 8  # 8× → 128px map becomes 1024×1024 (good for slides)

# Mirrors spoiled_broth.world.tiles.COLOR_MAP without engine imports.
COLOR_MAP = {
    '(0, 0, 0)': {'type': 'Floor'},
    '(50, 50, 50)': {'type': 'Wall'},
    '(255, 0, 0)': {'type': 'Dispenser', 'item': 'tomato'},
    '(255, 255, 0)': {'type': 'Dispenser', 'item': 'pumpkin'},
    '(0, 255, 0)': {'type': 'Dispenser', 'item': 'cabbage'},
    '(100, 100, 100)': {'type': 'Dispenser', 'item': 'plate'},
    '(0, 255, 255)': {'type': 'CuttingBoard'},
    '(255, 255, 255)': {'type': 'Counter'},
    '(255, 0, 255)': {'type': 'Delivery'},
}

def crop_sprite(sprite_image: Image.Image, tile_type: str, layer_idx: int) -> Image.Image:
    if tile_type == "Dispenser_tomato" and layer_idx == 1:
        return sprite_image.crop((0, 0, 16, 16))
    if tile_type == "Dispenser_pumpkin" and layer_idx == 1:
        return sprite_image.crop((0, 16, 16, 32))
    if tile_type == "Dispenser_plate" and layer_idx == 1:
        return sprite_image.crop((0, 48, 16, 64))
    if tile_type == "Dispenser_cabbage" and layer_idx == 1:
        return sprite_image.crop((0, 32, 16, 48))
    return sprite_image.crop((0, 0, 16, 16))


def show_map(path,
             width=8,
             height=8,
             name='default.png',
             scale: int = OUTPUT_SCALE):
    """
    Make map image from map data
    """
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize((width, height))
    if img.size != (width, height):
        raise ValueError("Image size does not match grid size")
    canvas = Image.new("RGBA", (width * TILE_PX, height * TILE_PX))
    for x in range(width):
        for y in range(height):
            color = img.getpixel((x, y))
            tile_info = COLOR_MAP[str(color)]
            tile_type = tile_info['type']
            if tile_type == "Dispenser":
                tile_type = f"Dispenser_{tile_info['item']}"
            sprite_paths = asset_map[tile_type]['paths']
            for i, p in enumerate(sprite_paths):
                sprite_image = Image.open(p).convert("RGBA")
                sprite_image = crop_sprite(sprite_image, tile_type, i)
                canvas.paste(sprite_image, (x * TILE_PX, y * TILE_PX), sprite_image)
    if scale != 1:
        canvas = canvas.resize(
            (canvas.width * scale, canvas.height * scale),
            resample=Image.NEAREST,
        )
    out = Path(__file__).parent
    canvas.save(out / name)


asset_map = {
    "Floor": {
        "paths": [Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-floor.png"],
    },
    "Wall": {
        "paths": [Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-wall.png"],
    },
    "Counter": {
        "paths": [Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png"],
    },
    "CuttingBoard": {
        "paths": [
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "cutting-board.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "knife.png",
        ],
    },
    "Dispenser_tomato": {
        "paths": [
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "item-dispenser.png",
        ],
    },
    "Dispenser_plate": {
        "paths": [
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "item-dispenser.png",
        ],
    },
    "Dispenser_cabbage": {
        "paths": [
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "item-dispenser.png",
        ],
    },
    "Dispenser_pumpkin": {
        "paths": [
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent.parent / "static" / "sprites" / "world" / "item-dispenser.png",
        ],
    },
    "Delivery": {
        "paths": [Path(__file__).parent.parent / "static" / "sprites" / "world" / "delivery.png"],
    },

}

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    maps = [
        # "baseline_division_of_labor_large.png",
        # "encouraged_division_of_labor_large.png",
        "forced_division_of_labor.png",
        "encouraged_division_of_labor_v2.png",
        "baseline_division_of_labor_v2.png",
    ]
    for m in maps:
        img_path = Path(__file__).parent / "maps" / f"{m}"
        show_map(img_path, width=8, height=8, name=m)
