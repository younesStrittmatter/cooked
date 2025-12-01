from flask import Flask
from pathlib import Path
import importlib.resources as r

# use the engine's own route wiring
from engine.interface.static_routes import register_static_routes
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule


app = Flask(__name__)

# ➜ point this at your game folder that contains a `static/` directory
#    (you can still test engine JS without a game, but index.html lives there)
path_root = Path(__file__).resolve().parent / "spoiled_broth"

# register ONLY the routes the engine provides
register_static_routes(app, path_root=path_root, ui_modules=[Renderer2DModule()])

# just to help you see where it's reading from (no new routes added)
print("engine static dir:", r.files("engine").joinpath("static"))
print("game static dir:", (path_root / "static").resolve())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
