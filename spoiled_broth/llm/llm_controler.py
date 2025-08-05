import random

from engine.extensions.topDownGridWorld.ai_controller._base_controller import Controller
from spoiled_broth.llm.api import OpenAIWrapper, GeminiWrapper



from spoiled_broth.game import game_to_prompt


class LLMController(Controller):
    def __init__(self, agent_id, llm_wrapper=GeminiWrapper()):
        super().__init__(agent_id)

        self.wrapper = llm_wrapper
        self.agent_id = agent_id

    def choose_action(self, observation: dict, tick=None):
        if self.agent.is_moving:
            return {}

        # agent will be assigned by SessionApp, so we can access the game through it
        prompt = game_to_prompt(self.agent.game, self.agent_id)
        response = self.wrapper.prompt(prompt)
        # postprocess the response (expects a string (x, y) put should be flexible))
        grid = self.agent.grid
        try:
            response = response.strip().replace("(", "").replace(")", "")
            slot_x_str, slot_y_str = response.split(",")
            slot_x = int(slot_x_str.strip())
            slot_y = int(slot_y_str.strip())
        except Exception as e:
            print(f"[choose_action] Failed to parse response: '{response}' — {e}")
            return {}

        if slot_x < 0 or slot_y < 0:
            slot_x = random.randint(0, 7)
            slot_y = random.randint(0, 7)
        if slot_x > grid.width - 1 or slot_y > grid.height - 1:
            slot_x = random.randint(0, 7)
            slot_y = random.randint(0, 7)


        tile = grid.tiles[slot_x][slot_y]
        print(response)
        print(f'chosen tile: {tile.slot_x, tile.slot_y}')

        if tile and hasattr(tile, "click"):
            return {"type": "click", "target": tile.id}
        return {}
