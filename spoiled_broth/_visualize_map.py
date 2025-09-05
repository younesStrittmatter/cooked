"""
Visualize and store maps from sprites and a map specification image (maps)
"""

from pathlib import Path

from world.tiles import COLOR_MAP
from PIL import Image


def show_map(path,
             width=8,
             height=8,
             name='default.png'):
    """
    Make map image from map data
    """
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize((width, height))
    if img.size != (width, height):
        raise ValueError("Image size does not match grid size")
    # create a empty image (8 * 16 , 8 * 16)
    canvas = Image.new("RGBA", (width * 16, height * 16))
    for x in range(width):
        for y in range(height):
            color = img.getpixel((x, y))
            tile_type = COLOR_MAP[str(color)]['class'].__name__

            if tile_type == "Dispenser":
                dispenser_type = COLOR_MAP[str(color)]['kwargs']['item']
                tile_type = f"Dispenser_{dispenser_type}"
            sprite_paths = asset_map[tile_type]['paths']
            for i, p in enumerate(sprite_paths):
                # load with alpha
                sprite_image = Image.open(p).convert("RGBA")
                if tile_type == "Dispenser_tomato" and i == 1:
                    # if the sprite is a dispenser, we need to paste it on the counter
                    sprite_image = sprite_image.crop((0, 0, 16, 16))
                elif tile_type == "Dispenser_plate" and i == 1:
                    sprite_image = sprite_image.crop((0, 48, 16, 64))
                else:
                    sprite_image = sprite_image.crop((0, 0, 16, 16))
                # paste the sprite image into the empty at position x * 16, y * 16
                canvas.paste(sprite_image, (x * 16, y * 16), sprite_image)
    # save the canvas
    out = Path("_visualization")
    canvas.save(out / name)


asset_map = {
    "Floor": {
        "paths": [Path(__file__).parent / "static" / "sprites" / "world" / "basic-floor.png"],
    },
    "Wall": {
        "paths": [Path(__file__).parent / "static" / "sprites" / "world" / "basic-wall.png"],
    },
    "Counter": {
        "paths": [Path(__file__).parent / "static" / "sprites" / "world" / "basic-counter.png"],
    },
    "CuttingBoard": {
        "paths": [
            Path(__file__).parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent / "static" / "sprites" / "world" / "cutting-board.png",
            Path(__file__).parent / "static" / "sprites" / "world" / "knife.png",
        ],
    },
    "Dispenser_tomato": {
        "paths": [
            Path(__file__).parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent / "static" / "sprites" / "world" / "item-dispenser.png",
        ],
    },
    "Dispenser_plate": {
        "paths": [
            Path(__file__).parent / "static" / "sprites" / "world" / "basic-counter.png",
            Path(__file__).parent / "static" / "sprites" / "world" / "item-dispenser.png",
        ],
    },
    "Delivery": {
        "paths": [Path(__file__).parent / "static" / "sprites" / "world" / "delivery.png"],
    },

}

if __name__ == "__main__":
    root = Path(__file__).parent
    maps = [
        "baseline_division_of_labor.png",
        "encouraged_division_of_labor.png",
        "forced_division_of_labor.png",
    ]
    for m in maps:
        img_path = Path(__file__).parent / "maps" / f"{m}"
        show_map(img_path, width=8, height=8, name=m)
