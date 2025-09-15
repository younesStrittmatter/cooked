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

        # Base cook-husk
        if hasattr(agent_obj, "cook_husk_src") and agent_obj.cook_husk_src:
            layers.append(self.load_sprite(agent_obj.cook_husk_src))

        # Skin
        if hasattr(agent_obj, "skin_src") and agent_obj.skin_src:
            layers.append(self.load_sprite(agent_obj.skin_src))

        # Hair
        if hasattr(agent_obj, "hair_src") and agent_obj.hair_src:
            layers.append(self.load_sprite(agent_obj.hair_src))

        # Mustache
        if hasattr(agent_obj, "mustache_src") and agent_obj.mustache_src:
            layers.append(self.load_sprite(agent_obj.mustache_src))

        # Compose
        if not layers:
            # fallback: blank transparent
            return Image.new("RGBA", (self.tile_size, self.tile_size))
        
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
        prev_objects = {o["id"]: o for o in (prev_state or {}).get("gameObjects", [])}

        # Sort by zIndex
        objects = sorted(objects, key=lambda o: o.get("zIndex", 0))

        for obj in objects:
            obj_id = obj["id"]
            prev_obj = prev_objects.get(obj_id, obj)

            # Interpolate position
            left = prev_obj.get("left", obj.get("left", 0)) + (obj.get("left", 0) - prev_obj.get("left", 0)) * t
            top = prev_obj.get("top", obj.get("top", 0)) + (obj.get("top", 0) - prev_obj.get("top", 0)) * t

            # Determine sprite
            src = obj.get("src")
            if not src:
                # Possibly an agent composed of layers
                if obj.get("is_agent"):
                    sprite = self.compose_agent(obj)
                else:
                    continue
            else:
                sprite = self.load_sprite(src)
                sprite = sprite.resize((self.tile_size, self.tile_size), resample=Image.NEAREST)

            canvas.alpha_composite(sprite, (int(left * self.tile_size), int(top * self.tile_size)))

        # Convert to BGR numpy array for OpenCV / VideoRecorder
        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGBA2BGR)
