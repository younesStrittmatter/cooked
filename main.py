from engine.app.session_app import SessionApp
from spoiled_broth.game import SpoiledBroth as Game
from engine.extensions.topDownGridWorld.ai_controller.random_tile_clicker import RandomTileClicker
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from pathlib import Path

import logging

log = logging.getLogger('werkzeug')
log.disabled = True

path_root = Path(__file__).resolve().parent / "spoiled_broth"

engine_app = SessionApp(
    game_factory=Game,
    ui_modules=[Renderer2DModule()],
    agent_map={"ai_1": RandomTileClicker('ai_1')},
    path_root=path_root,
    tick_rate=12,
    ai_tick_rate=1,
    n_players=1,
    is_max_speed=False
)

app = engine_app.app

if __name__ == "__main__":
    app.run(port=5000)
