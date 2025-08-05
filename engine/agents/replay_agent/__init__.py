from engine.ai.controller import Controller


class ReplayController(Controller):
    def __init__(self, replay_data, agent_id):
        super().__init__(agent_id)
        self.intent_by_tick = {
            int(entry["tick"]): entry["action"]
            for entry in replay_data["intents"]
            if entry["agent_id"] == agent_id
        }
        self.initial_config = {
            key: entry
            for key, entry in replay_data['agents'][agent_id]['initial_state'].items()
            if key != 'agent_id'
        }

    def choose_action(self, observation: dict, tick=None):
        intent = self.intent_by_tick.get(int(tick), None)
        if intent:
            return self.intent_by_tick.get(tick, {})

    def agent_init_config(self):
        return self.initial_config
