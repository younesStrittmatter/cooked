from engine.app.session_app import SessionApp
from coboxing.game import CoBoxing as Game
from coboxing.ai.random_controller import RandomWalk

from engine.extensions.renderer2d.renderer_ui import Renderer2DModule
from pathlib import Path


import logging

log = logging.getLogger('werkzeug')
log.disabled = True

path_root = Path(__file__).resolve().parent / "coboxing"

engine_app = SessionApp(
    game_factory=Game,
    ui_modules=[Renderer2DModule()],
    # agent_map={
    #     'ai_1': RandomWalk('ai_1'),
    #     'ai_2': RandomWalk('ai_2'),
    #     'ai_3': RandomWalk('ai_3'),
    #     'ai_4': RandomWalk('ai_4'),
    #     'ai_5': RandomWalk('ai_5'),
    #     'ai_6': RandomWalk('ai_6'),
    # },
    path_root=path_root,
    tick_rate=24,
    ai_tick_rate=24,
    n_players=2,
    is_max_speed=False
)

app = engine_app.app

if __name__ == "__main__":
    app.run(port=5000)
