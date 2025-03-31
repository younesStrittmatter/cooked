from engine.app.session_app import SessionApp
# from simple_game.simple_game import SimpleGame as Game
from spoiled_broth.game import SpoiledBroth as Game
# from simple_game.simple_game_ui import SimpleGameUI as UI, SimpleGameUI
# from simple_game.simple_game_agents import RandomAgent
# from game2d.game2d import Game2d as Game
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from pathlib import Path

import logging
log = logging.getLogger('werkzeug')
log.disabled = True

path_root = Path(__file__).resolve().parent / "spoiled_broth"

engine_app = SessionApp(
    game_factory=Game,
    ui_modules=[Renderer2DModule()],
    # agent_map={"bot": RandomAgent()},
    path_root=path_root,
    tick_rate=10,
    max_agents=2
)

app = engine_app.app

if __name__ == "__main__":
    app.run(port=5000)
