"""
Create gifs showing abilities in action:

(1) fast/slow walking cook
(2) fast/slow chopping cook
"""

from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple

# ---------------------------
# CONFIG — EDIT THESE PATHS
# ---------------------------

# All sprites are horizontal strips of 16x16 tiles.
ASSETS = Path("../static/sprites")  # root folder for your images
OUT = Path("_show")

CONFIG = {
    # Cook walking strips (same number/order of frames, aligned by x index)
    # Each is a horizontal strip of 16x16 frames.
    "cook_walk": {
        "husk": ASSETS / "agent" / "cook-husk.png",
        "skin": ASSETS / "agent" / "skin" / "0.png",  # row 1: head, row 2: hands
        "hair": ASSETS / "agent" / "hair" / "1.png",
        "mustache": ASSETS / "agent" / "mustache" / "0.png",  # optional
    },
    # Chopping scene assets (each a horizontal strip of 16x16)
    "chop": {
        "counter": ASSETS / "world" / "basic-counter.png",
        "board": ASSETS / "world" / "cutting-board.png",
        "ingredient": ASSETS / "world" / "item-on-board.png",  # row 1: tomato
        "knife": ASSETS / "world" / "knife.png",  # 3 frames (cycle)
    },
    # Tile geometry
    "tile_w": 16,
    "tile_h": 16,
    # Output sizes (scale 16->64 for presentation)
    "scale": 4,
    # Speeds (ms per frame) for walking
    "walk_ms_fast": 48,
    "walk_ms_slow": 168,
    # Chopping timing:
    # - knife cycles per ingredient step: after this many knife frames, progress ingredient by 1
    # - ms per output frame (GIF frame duration)
    "chop_knife_cycles_per_step_fast": 1,
    "chop_knife_cycles_per_step_slow": 2,
    "chop_ms_fast": 60,
    "chop_ms_slow": 204,
    # How many full ingredient progressions to show in the chop GIF (1..N)
    "chop_progressions": 1,
}


# ---------------------------
# UTILITIES
# ---------------------------

def load_strip(img_path: Path, tile_w: int, tile_h: int, slot_y=0) -> List[Image.Image]:
    """Load a horizontal strip spritesheet (keyframes on X axis) into a list of frames."""
    im = Image.open(img_path).convert("RGBA")
    w, h = im.size
    assert w % tile_w == 0, f"Width {w} not divisible by tile {tile_w} for {img_path}"
    frames = []
    for i in range(w // tile_w):
        box = (i * tile_w, tile_h * slot_y, (i + 1) * tile_w, tile_h * (slot_y + 1))
        frames.append(im.crop(box))
    return frames


def scale_frame(frame: Image.Image, scale: int) -> Image.Image:
    return frame.resize((frame.width * scale, frame.height * scale), resample=Image.NEAREST)


def composite_layers(frame_layers: List[Image.Image]) -> Image.Image:
    """Alpha-composite a stack of same-sized RGBA frames in order."""
    base = frame_layers[0].copy()
    for layer in frame_layers[1:]:
        base.alpha_composite(layer)
    return base


def save_gif(frames: List[Image.Image], out_path: Path, ms_per_frame: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=ms_per_frame,
        loop=0,
        optimize=True,
        disposal=2,
        transparency=0
    )


def shortest_len(frame_lists: List[List[Image.Image]]) -> int:
    return min(len(fl) for fl in frame_lists)


# ---------------------------
# COOK — WALKING GIFS
# ---------------------------

def build_cook_walk_gif(
                        tile_w: int,
                        tile_h: int,
                        scale: int,
                        ms_per_frame: int,
                        out_path: Path) -> None:
    # Load each layer strip
    strips = {
        'head': load_strip(CONFIG["cook_walk"]["skin"], tile_w, tile_h, slot_y=0),
        'hair': load_strip(CONFIG["cook_walk"]["hair"], tile_w, tile_h),
        'mustache': load_strip(CONFIG["cook_walk"]["mustache"], tile_w, tile_h),
        'hands': load_strip(CONFIG["cook_walk"]["skin"], tile_w, tile_h, slot_y=1),
        'husk': load_strip(CONFIG["cook_walk"]["husk"], tile_w, tile_h),
    }
    n = shortest_len(list(strips.values()))
    frames = []
    for i in range(n):
        layers_i = [strips[name][i] for name in strips.keys()]  # keep declared order
        composite = composite_layers(layers_i)
        frames.append(scale_frame(composite, scale))
    save_gif(frames, out_path, ms_per_frame)


# ---------------------------
# CHOPPING GIFS
# ---------------------------

def build_chop_gif(chop_paths: Dict[str, Path],
                   tile_w: int,
                   tile_h: int,
                   scale: int,
                   knife_cycles_per_step: int,
                   ms_per_frame: int,
                   progressions: int,
                   out_path: Path) -> None:
    # Required strips
    counter_strip = load_strip(chop_paths["counter"], tile_w, tile_h)
    board_strip = load_strip(chop_paths["board"], tile_w, tile_h)
    ingredient_strip = load_strip(chop_paths["ingredient"], tile_w, tile_h)  # 4 frames
    knife_strip = load_strip(chop_paths["knife"], tile_w, tile_h)  # 3 frames


    # Use first board frame if static
    counter_frame = counter_strip[0]
    board_frame = board_strip[0]

    # We will iterate a sequence like:
    # (knife frame 0..K-1) * knife_cycles_per_step -> then ingredient_frame += 1
    # Repeat until ingredient reaches final frame; `progressions` times.
    frames_out: List[Image.Image] = []
    knife_len = len(knife_strip)
    ingr_len = len(ingredient_strip)

    for _ in range(progressions):
        ingr_idx = 0
        while ingr_idx < ingr_len:
            # Perform the knife cycles before advancing ingredient
            for _cyc in range(knife_cycles_per_step):
                for k in range(knife_len):
                    layers = [board_frame, ingredient_strip[ingr_idx], knife_strip[k]]
                    composite = composite_layers(layers)
                    frames_out.append(scale_frame(composite, scale))
            ingr_idx += 1  # progress the ingredient by one after full knife cycle block

    # Edge case: ensure at least one frame even if config is off
    if not frames_out:
        composite = composite_layers([board_frame, ingredient_strip[0], knife_strip[0]])
        frames_out = [scale_frame(composite, scale)]

    save_gif(frames_out, out_path, ms_per_frame)


# ---------------------------
# MAIN
# ---------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Walking — fast
    build_cook_walk_gif(
        tile_w=CONFIG["tile_w"], tile_h=CONFIG["tile_h"], scale=CONFIG["scale"],
        ms_per_frame=CONFIG["walk_ms_fast"],
        out_path=OUT / "cook_walk_fast.gif"
    )

    # Walking — slow
    build_cook_walk_gif(
        tile_w=CONFIG["tile_w"], tile_h=CONFIG["tile_h"], scale=CONFIG["scale"],
        ms_per_frame=CONFIG["walk_ms_slow"],
        out_path=OUT / "cook_walk_slow.gif"
    )
    #
    # Chopping — fast (ingredient progresses quicker)
    build_chop_gif(
        chop_paths=CONFIG["chop"],
        tile_w=CONFIG["tile_w"], tile_h=CONFIG["tile_h"], scale=CONFIG["scale"],
        knife_cycles_per_step=CONFIG["chop_knife_cycles_per_step_fast"],
        ms_per_frame=CONFIG["chop_ms_fast"],
        progressions=CONFIG["chop_progressions"],
        out_path=OUT / "cook_chop_fast.gif"
    )

    # Chopping — slow (ingredient progresses slower)
    build_chop_gif(
        chop_paths=CONFIG["chop"],
        tile_w=CONFIG["tile_w"], tile_h=CONFIG["tile_h"], scale=CONFIG["scale"],
        knife_cycles_per_step=CONFIG["chop_knife_cycles_per_step_slow"],
        ms_per_frame=CONFIG["chop_ms_slow"],
        progressions=CONFIG["chop_progressions"],
        out_path=OUT / "cook_chop_slow.gif"
    )


if __name__ == "__main__":
    main()
