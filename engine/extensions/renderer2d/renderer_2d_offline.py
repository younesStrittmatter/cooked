# renderer_2d_offline.py

import numpy as np
import cv2
from pathlib import Path
from PIL import Image


class Renderer2DOffline:
    """
    Offline 2D renderer for Spoiled Broth.
    - Tile size: 16x16 px
    - Grid size taken from game.grid.width / height
    - Supports agents with cook-husk + skin + hair + mustache
    - Interpolates between previous and current state
    """

    def __init__(self, path_root: Path, tile_size=16):
        self.tile_size = tile_size
        self.path_root = Path(path_root)
        self.sprite_folder = self.path_root / "static" / "sprites"
        self.sprite_cache = {}  # { "agent/skin/0.png": PIL.Image }
    
    def load_sprite(self, src: str):
        """Load sprite from disk, cache it"""
        if src in self.sprite_cache:
            return self.sprite_cache[src]
        path = self.sprite_folder / src
        if not path.exists():
            raise FileNotFoundError(f"Sprite not found: {path}")
        img = Image.open(path).convert("RGBA")
        self.sprite_cache[src] = img
        return img

    def compose_agent(self, agent_obj):
        """
        Compose agent layers: cook-husk + skin + hair + mustache
        Returns RGBA PIL.Image of size tile_size x tile_size
        """
        layers = []

        # helper to get attribute from object or dict
        def _get(key):
            if isinstance(agent_obj, dict):
                return agent_obj.get(key)
            return getattr(agent_obj, key, None)

        # Base cook-husk
        cook = _get("cook_husk_src")
        if cook:
            layers.append(self.load_sprite(cook))

        # Skin
        skin = _get("skin_src")
        if skin:
            layers.append(self.load_sprite(skin))

        # Hair
        hair = _get("hair_src")
        if hair:
            layers.append(self.load_sprite(hair))

        # Mustache
        mustache = _get("mustache_src")
        if mustache:
            layers.append(self.load_sprite(mustache))

        # Compose
        if not layers:
            # fallback: blank transparent
            return Image.new("RGBA", (self.tile_size, self.tile_size), (0, 0, 0, 0))

        base = layers[0].resize((self.tile_size, self.tile_size), resample=Image.NEAREST)
        for layer in layers[1:]:
            layer_resized = layer.resize((self.tile_size, self.tile_size), resample=Image.NEAREST)
            base.alpha_composite(layer_resized)
        return base

    def render_to_array(self, prev_state, curr_state, t: float):
        """
        Render interpolated frame between prev_state and curr_state
        t in [0,1] for interpolation
        Returns: np.array(H,W,3) uint8
        """
        # Determine grid size from current state
        grid_w = curr_state.get("grid_width", 8)
        grid_h = curr_state.get("grid_height", 8)

        canvas_w = grid_w * self.tile_size
        canvas_h = grid_h * self.tile_size

        # Start with transparent RGBA canvas
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 255))

        objects = curr_state.get("gameObjects", [])
        # Build map of previous objects, but ignore entries without an 'id'
        prev_objects = {}
        for o in (prev_state or {}).get("gameObjects", []):
            oid = o.get("id") if isinstance(o, dict) else None
            if oid is not None:
                prev_objects[oid] = o

        # Sort by zIndex
        objects = sorted(objects, key=lambda o: o.get("zIndex", 0))

        for obj in objects:
            # skip non-dict entries
            if not isinstance(obj, dict):
                continue

            obj_id = obj.get("id")
            # If object has no id, it's likely a UI-only payload (score, metadata) - skip
            if obj_id is None:
                continue

            prev_obj = prev_objects.get(obj_id, obj)

            # Interpolate position (use pixel coordinates when available)
            left_curr = obj.get("left", 0)
            top_curr = obj.get("top", 0)
            left_prev = prev_obj.get("left", left_curr)
            top_prev = prev_obj.get("top", top_curr)
            left = left_prev + (left_curr - left_prev) * t
            top = top_prev + (top_curr - top_prev) * t

            # Determine whether coordinates are normalized (0..1) or pixels
            normalize = obj.get("normalize", False)
            if normalize:
                # If values look like fractions (<=1), treat as fraction of canvas
                if 0 <= left <= 1 and 0 <= top <= 1:
                    left_px = int(left * canvas_w)
                    top_px = int(top * canvas_h)
                else:
                    left_px = int(left)
                    top_px = int(top)
            else:
                # left/top are pixel units
                left_px = int(left)
                top_px = int(top)

            # Determine sprite: either simple src sprite (with optional cropping) or composed agent
            src = obj.get("src")
            if src:
                sprite = self.load_sprite(src)

                # cropping
                srcX = obj.get("srcX", 0) or 0
                srcY = obj.get("srcY", 0) or 0
                srcW = obj.get("srcW") or sprite.width
                srcH = obj.get("srcH") or sprite.height

                try:
                    crop = sprite.crop((int(srcX), int(srcY), int(srcX + srcW), int(srcY + srcH)))
                except Exception:
                    crop = sprite

                # destination width/height
                dst_w = int(obj.get("width", srcW))
                dst_h = int(obj.get("height", srcH))

                # If width/height look like tile counts (small numbers), scale to tile_size
                if dst_w <= 0:
                    dst_w = self.tile_size
                if dst_h <= 0:
                    dst_h = self.tile_size

                sprite_resized = crop.resize((dst_w, dst_h), resample=Image.NEAREST)
                canvas.alpha_composite(sprite_resized, (left_px, top_px))
            else:
                # Possibly an agent composed of layers
                if obj.get("is_agent"):
                    sprite = self.compose_agent(obj)
                    canvas.alpha_composite(sprite, (left_px, top_px))
                else:
                    # No src and not agent -> skip
                    continue

        # Convert to BGR numpy array for OpenCV / VideoRecorder
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGBA2BGR)
