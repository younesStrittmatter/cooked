"""
Create gifs showing abilities in action:

(1) fast/slow walking cook
(2) fast/slow chopping cook
"""

from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
import random

# ---------------------------
# CONFIG — EDIT THESE PATHS
# ---------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# All sprites are horizontal strips of 16x16 tiles.
ASSETS = SCRIPT_DIR / "../static/sprites"  # root folder for your images
OUT = SCRIPT_DIR
SHOW = SCRIPT_DIR / "_show"
SKIN_VARIANT_SEED = 42
# Each spec: has_hair, has_mustache (default True), held item (None = empty hands).
WALK_VARIANT_SPECS = [
    {"has_hair": True, "item": None},
    {"has_hair": True, "item": None},
    {"has_hair": True, "item": None},
    {"has_hair": True, "item": "tomato"},
    {"has_hair": True, "item": "tomato_salad"},
    {"has_hair": False, "item": None},
    {"has_hair": False, "item": None},
    {"has_hair": False, "item": None},
    {"has_hair": False, "item": "tomato"},
    {"has_hair": False, "item": "tomato_salad"},
    {"has_hair": True, "has_mustache": False, "item": None},
    {"has_hair": True, "has_mustache": False, "item": "tomato"},
    {"has_hair": False, "has_mustache": False, "item": None},
    {"has_hair": False, "has_mustache": False, "item": "tomato_salad"},
]

HELD_ITEM_ROWS = {
    "tomato": 0,
    "pumpkin": 1,
    "cabbage": 2,
    "plate": 3,
    "tomato_cut": 9,
    "pumpkin_cut": 10,
    "cabbage_cut": 11,
    "tomato_salad": 15,
    "pumpkin_salad": 16,
    "cabbage_salad": 17,
}

CONFIG = {
    # Cook walking strips (same number/order of frames, aligned by x index)
    # Each is a horizontal strip of 16x16 frames.
    "cook_walk": {
        "husk": ASSETS / "agent" / "cook-husk.png",
        "skin": ASSETS / "agent" / "skin" / "0.png",  # row 0: head, 1: hands, 2: hands+item pose
        "hair": ASSETS / "agent" / "hair" / "1.png",
        "mustache": ASSETS / "agent" / "mustache" / "0.png",  # optional
        "items_held": ASSETS / "agent" / "items-held.png",
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
    # Output size multiplier (16px tiles → 128px at scale 8). NEAREST keeps pixels crisp.
    "scale": 8,
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


def rotate_frames(frames: List[Image.Image], offset: int) -> List[Image.Image]:
    """Rotate GIF frame order so the animation starts at a different keyframe."""
    if not frames or offset == 0:
        return frames
    offset = offset % len(frames)
    return frames[offset:] + frames[:offset]


# ---------------------------
# COOK — WALKING GIFS
# ---------------------------

def build_cook_walk_gif(
                        tile_w: int,
                        tile_h: int,
                        scale: int,
                        ms_per_frame: int,
                        out_path: Path,
                        skin_path: Path,
                        hair_path: Optional[Path] = None,
                        mustache_path: Optional[Path] = None,
                        held_item: Optional[str] = None,
                        frame_offset: int = 0) -> None:
    hands_row = 2 if held_item else 1
    # Bottom-to-top order matches agent/base.py z-index: husk(1), head/hair/hands(2), item(3).
    layer_names = ["husk", "head"]
    strips = {
        "husk": load_strip(CONFIG["cook_walk"]["husk"], tile_w, tile_h),
        "head": load_strip(skin_path, tile_w, tile_h, slot_y=0),
        "hands": load_strip(skin_path, tile_w, tile_h, slot_y=hands_row),
    }
    if hair_path is not None:
        layer_names.append("hair")
        strips["hair"] = load_strip(hair_path, tile_w, tile_h)
    if mustache_path is not None:
        layer_names.append("mustache")
        strips["mustache"] = load_strip(mustache_path, tile_w, tile_h)
    layer_names.append("hands")
    if held_item:
        if held_item not in HELD_ITEM_ROWS:
            raise ValueError(f"Unknown held item: {held_item}")
        layer_names.append("item")
        strips["item"] = load_strip(
            CONFIG["cook_walk"]["items_held"],
            tile_w,
            tile_h,
            slot_y=HELD_ITEM_ROWS[held_item],
        )
    n = shortest_len([strips[name] for name in layer_names])
    frames = []
    for i in range(n):
        layers_i = [strips[name][i] for name in layer_names]
        composite = composite_layers(layers_i)
        frames.append(scale_frame(composite, scale))
    frames = rotate_frames(frames, frame_offset)
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
                   out_path: Path,
                   frame_offset: int = 0) -> None:
    # Required strips
    board_strip = load_strip(chop_paths["board"], tile_w, tile_h)
    ingredient_strip = load_strip(chop_paths["ingredient"], tile_w, tile_h)  # 4 frames
    knife_strip = load_strip(chop_paths["knife"], tile_w, tile_h)  # 3 frames

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

    frames_out = rotate_frames(frames_out, frame_offset)
    save_gif(frames_out, out_path, ms_per_frame)


def random_walk_variants(specs: List[Dict], seed: int) -> List[Dict]:
    rng = random.Random(seed)
    skin_dir = ASSETS / "agent" / "skin"
    hair_dir = ASSETS / "agent" / "hair"
    mustache_dir = ASSETS / "agent" / "mustache"
    skin_ids = sorted(int(p.stem) for p in skin_dir.glob("*.png"))
    hair_ids = sorted(int(p.stem) for p in hair_dir.glob("*.png"))
    mustache_ids = sorted(int(p.stem) for p in mustache_dir.glob("*.png"))

    variants = []
    seen = set()
    for spec in specs:
        has_mustache = spec.get("has_mustache", True)
        while True:
            skin_id = rng.choice(skin_ids)
            hair_id = rng.choice(hair_ids) if spec["has_hair"] else None
            mustache_id = rng.choice(mustache_ids) if has_mustache else None
            key = (skin_id, hair_id, mustache_id, spec.get("item"))
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "skin": skin_dir / f"{skin_id}.png",
                    "hair": hair_dir / f"{hair_id}.png" if hair_id is not None else None,
                    "mustache": mustache_dir / f"{mustache_id}.png" if mustache_id is not None else None,
                    "skin_id": skin_id,
                    "hair_id": hair_id,
                    "mustache_id": mustache_id,
                    "has_hair": spec["has_hair"],
                    "has_mustache": has_mustache,
                    "item": spec.get("item"),
                }
            )
            break
    return variants


def build_cook_walk_variants(out_dir: Path, ms_per_frame: int, speed_label: str) -> None:
    variants = random_walk_variants(WALK_VARIANT_SPECS, SKIN_VARIANT_SEED)
    walk_frames = len(load_strip(CONFIG["cook_walk"]["husk"], CONFIG["tile_w"], CONFIG["tile_h"]))
    for idx, variant in enumerate(variants):
        # Spread starting keyframes across the walk cycle (8 frames).
        frame_offset = (idx * 5) % walk_frames
        out_path = out_dir / f"cook_walk_{speed_label}_{idx}.gif"
        build_cook_walk_gif(
            tile_w=CONFIG["tile_w"],
            tile_h=CONFIG["tile_h"],
            scale=CONFIG["scale"],
            ms_per_frame=ms_per_frame,
            out_path=out_path,
            skin_path=variant["skin"],
            hair_path=variant["hair"],
            mustache_path=variant["mustache"],
            held_item=variant["item"],
            frame_offset=frame_offset,
        )
        hair_label = variant["hair_id"] if variant["has_hair"] else "none"
        mustache_label = variant["mustache_id"] if variant["has_mustache"] else "none"
        item_label = variant["item"] or "none"
        print(
            f"Wrote {out_path.name} "
            f"(skin={variant['skin_id']}, hair={hair_label}, "
            f"mustache={mustache_label}, item={item_label}, start={frame_offset})"
        )


# ---------------------------
# MAIN
# ---------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    SHOW.mkdir(parents=True, exist_ok=True)

    default_skin = CONFIG["cook_walk"]["skin"]
    default_hair = CONFIG["cook_walk"]["hair"]
    default_mustache = CONFIG["cook_walk"]["mustache"]

    # Walking — skin/hair/item variants (fast + slow)
    build_cook_walk_variants(OUT, CONFIG["walk_ms_fast"], "fast")
    build_cook_walk_variants(OUT, CONFIG["walk_ms_slow"], "slow")

    # Keep single default versions for quick reference
    build_cook_walk_gif(
        tile_w=CONFIG["tile_w"], tile_h=CONFIG["tile_h"], scale=CONFIG["scale"],
        ms_per_frame=CONFIG["walk_ms_fast"],
        out_path=OUT / "cook_walk_fast.gif",
        skin_path=default_skin,
        hair_path=default_hair,
        mustache_path=default_mustache,
    )
    build_cook_walk_gif(
        tile_w=CONFIG["tile_w"], tile_h=CONFIG["tile_h"], scale=CONFIG["scale"],
        ms_per_frame=CONFIG["walk_ms_slow"],
        out_path=OUT / "cook_walk_slow.gif",
        skin_path=default_skin,
        hair_path=default_hair,
        mustache_path=default_mustache,
    )
    #
    # Chopping — fast/slow at 8x scale, each with a different starting keyframe
    chop_knife_frames = len(load_strip(CONFIG["chop"]["knife"], CONFIG["tile_w"], CONFIG["tile_h"]))
    chop_fast_frames = (
        len(load_strip(CONFIG["chop"]["ingredient"], CONFIG["tile_w"], CONFIG["tile_h"]))
        * CONFIG["chop_knife_cycles_per_step_fast"]
        * chop_knife_frames
        * CONFIG["chop_progressions"]
    )
    chop_slow_frames = (
        len(load_strip(CONFIG["chop"]["ingredient"], CONFIG["tile_w"], CONFIG["tile_h"]))
        * CONFIG["chop_knife_cycles_per_step_slow"]
        * chop_knife_frames
        * CONFIG["chop_progressions"]
    )
    for speed_label, knife_cycles, ms, total_frames, start in [
        ("fast", CONFIG["chop_knife_cycles_per_step_fast"], CONFIG["chop_ms_fast"], chop_fast_frames, 0),
        ("slow", CONFIG["chop_knife_cycles_per_step_slow"], CONFIG["chop_ms_slow"], chop_slow_frames, chop_slow_frames // 3),
    ]:
        out_path = OUT / f"cook_chop_{speed_label}.gif"
        build_chop_gif(
            chop_paths=CONFIG["chop"],
            tile_w=CONFIG["tile_w"], tile_h=CONFIG["tile_h"], scale=CONFIG["scale"],
            knife_cycles_per_step=knife_cycles,
            ms_per_frame=ms,
            progressions=CONFIG["chop_progressions"],
            out_path=out_path,
            frame_offset=start,
        )
        print(f"Wrote {out_path.name} (start={start % total_frames if total_frames else 0})")


if __name__ == "__main__":
    main()
