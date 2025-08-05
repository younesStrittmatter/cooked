


from engine.logging.replay_loader import load_replay_agents
from spoiled_broth.game import SpoiledBroth as Game

from pathlib import Path
import json


path_root = Path(__file__).resolve().parent / "spoiled_broth"

replay_path = "replays/replay.json"

with open(replay_path, "r") as f:
    replay_data = json.load(f)

config = replay_data["config"]
replay_agents = load_replay_agents(replay_path)

# Recreate the game from state
restored_game = Game.from_state(config)

print(restored_game)
