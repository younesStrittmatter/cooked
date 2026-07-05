import eventlet

eventlet.monkey_patch()
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT / "engine" / "src", REPO_ROOT / "games", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from engine.app.session_app import SessionApp
from spoiled_broth.game import SpoiledBroth as Game
from spoiled_broth.llm.llm_controler import LLMController
from engine.extensions.renderer2d.renderer_ui import Renderer2DModule

from spoiled_broth_experiment_settings.params import params_both, params_online

import logging

log = logging.getLogger('werkzeug')
log.disabled = True

path_root = Path(__file__).resolve().parent / "games/spoiled_broth"


engine_app = SessionApp(
    game_factory=Game,
    ui_modules=[Renderer2DModule()],
    agent_map={'llm_agent_1': LLMController('llm_agent_1'), 'llm_agent_2': LLMController('llm_agent_2')},
    path_root=path_root,
    tick_rate=params_both['tick_rate'],
    ai_tick_rate=.5,
    n_players=0,
    is_max_speed=params_online['is_max_speed'],
    max_game_time=params_both['max_game_time'],
    redirect_link='https://google.com',
    is_served_locally=True
)

app = engine_app.app

if __name__ == "__main__":
    import eventlet.wsgi

    engine_app.socketio.run(engine_app.app,
                            host="0.0.0.0",
                            port=8080,
                            debug=False)
