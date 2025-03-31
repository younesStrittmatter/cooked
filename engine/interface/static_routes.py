# engine/interface/static_routes.py
from flask import send_from_directory
from pathlib import Path


def register_static_routes(app, path_root, ui_modules=None):
    ui_modules = ui_modules or []

    # 🔁 Let each module optionally register dynamic routes
    for module in ui_modules:
        if hasattr(module, "register_routes"):
            module.register_routes(app)

    # 🔁 Mount game and engine static files
    static_dir_game = Path(path_root) / "static"
    static_dir_engine = Path(__file__).resolve().parents[1] / "static"

    @app.route("/")
    def serve_index():
        return send_from_directory(static_dir_game, "index.html")

    @app.route("/game/static/<path:filename>")
    def serve_game_static(filename):
        return send_from_directory(static_dir_game, filename)

    @app.route("/engine/static/<path:filename>")
    def serve_engine_static(filename):
        return send_from_directory(static_dir_engine, filename)

    # 🔁 Mount each UI module's static files
    for idx, module in enumerate(ui_modules):
        static_path = module.get_static_path() if hasattr(module, "get_static_path") else None
        if static_path:
            route_prefix = f"/extension/static{idx}/<path:filename>"

            def make_handler(directory):
                return lambda filename: send_from_directory(directory, filename)

            app.add_url_rule(
                route_prefix,
                endpoint=f"extension_static_{idx}",
                view_func=make_handler(static_path)
            )
