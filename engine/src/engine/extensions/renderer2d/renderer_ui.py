# renderer_2d_module.py
from engine.interface.ui_module import UIModule
from flask import send_from_directory, Response, redirect
from urllib.parse import urljoin
from pathlib import Path
import os


def _asset_base_from_env() -> str:
    """
    Priority:
      1) ASSET_BASE_URL (e.g. https://storage.googleapis.com/<bucket>/)
      2) If ASSET_BUCKET is set, use GCS public URL
      3) Fallback to local / (so /sprites/... works in dev)
    """
    base = (os.getenv("ASSET_BASE_URL") or "").strip()
    if base:
        return base.rstrip("/") + "/"
    bucket = (os.getenv("ASSET_BUCKET") or "").strip()
    if bucket:
        return f"https://storage.googleapis.com/{bucket}/"
    return "/"  # dev: keep relative root so /sprites/... hits local static


class Renderer2DModule(UIModule):
    def __init__(self, sprite_folder_name="sprites"):
        self.sprite_folder_name = sprite_folder_name
        self._path_root = None
        self._static_path = Path(__file__).parent / "static"
        self._module_name = Path(__file__).parent.name
        self._asset_base = _asset_base_from_env()
        self._asset_version = os.getenv("ASSET_VERSION", "")

    def serialize_for_agent(self, game, engine, agent_id: str) -> dict:
        return {"objects": getattr(game, "objects", [])}

    def get_static_path(self):
        return self._static_path

    def get_client_scripts(self) -> list[str]:
        # /asset-config.js defines window.ASSET_BASE_URL and window.ASSET_VERSION
        return ["/asset-config.js", f"/extensions/{self._module_name}/static/renderer_ui.js"]

    def register_routes(self, app):
        @app.route("/asset-config.js")
        def asset_config():
            js = (
                f'window.ASSET_BASE_URL="{self._asset_base}";'
                f'window.ASSET_VERSION="{self._asset_version}";'
                'window.dispatchEvent(new Event("asset-config-ready"));'
            )
            return Response(
                js,
                200,
                {
                    "Content-Type": "application/javascript",
                    "Cache-Control": "no-store",
                },
            )

        @app.route("/sprites/<path:filename>")
        def serve_sprite(filename: str):
            # If ASSET_BASE_URL/BUCKET is configured, redirect to GCS (or any CDN) with cache-buster
            if self._asset_base.startswith("http"):
                url = urljoin(self._asset_base, f"sprites/{filename}")
                if self._asset_version:
                    sep = "&" if "?" in url else "?"
                    url = f"{url}{sep}v={self._asset_version}"
                return redirect(url, code=302)

            # Dev fallback: serve from local static folder
            # Prefer the engine's path_root if provided; otherwise use this module's static dir.
            sprite_path = (
                (self._path_root / "static" / self.sprite_folder_name)
                if self._path_root
                else (self._static_path / self.sprite_folder_name)
            )
            return send_from_directory(sprite_path, filename, max_age=0)

    def set_path_root(self, path_root):
        self._path_root = Path(path_root)
