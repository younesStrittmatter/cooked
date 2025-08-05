import json
from engine.agents.replay_agent import ReplayController

def load_replay_agents(replay_path):
    with open(replay_path, "r") as f:
        data = json.load(f)
    return {
        agent_id: ReplayController(data, agent_id)
        for agent_id in data["agents"]
    }