# games/spoiled_broth/wsgi.py
import eventlet
eventlet.monkey_patch()

from pathlib import Path
from engine.app.session_app import SessionApp
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from spoiled_broth.game import SpoiledBroth as Game

# The game folder is the path root (contains static/index.html, sprites, etc.)
PATH_ROOT = Path(__file__).resolve().parent

engine_app = SessionApp(
    game_factory=Game,
    ui_modules=[Renderer2DModule()],
    path_root=PATH_ROOT,
    tick_rate=24,
    ai_tick_rate=0.5,
    n_players=2,
    is_max_speed=False,
    max_game_time=5 * 60 + 5,
    redirect_link="https://google.com",
)
app = engine_app.app  # <-- gunicorn expects `app`
