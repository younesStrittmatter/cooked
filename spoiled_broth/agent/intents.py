import time
from engine.extensions.topDownGridWorld.intent import _base_intent

class PickUpIntent(_base_intent.Intent):
    def __init__(self, tile):
        super().__init__()
        self.tile = tile
        self.has_ended = False

    def update(self, agent):
        if not self.has_ended:
            self.has_ended = True
            agent.item = self.tile.item

    def finished(self, agent):
        return self.has_ended

class ItemExchangeIntent(_base_intent.Intent):
    def __init__(self, tile):
        super().__init__()
        self.tile = tile
        self.has_ended = False

    def update(self, agent):
        if not self.has_ended:
            self.has_ended = True
            # Case 1: agent has a cutted ingredient and tile has a plate
            if agent.item and self.tile.item and agent.item.endswith('_cut') and self.tile.item == 'plate':
                self.tile.item = agent.item.split('_')[0] + '_salad'
                self.tile.salad_item = self.tile.item
                agent.item = None

            # Case 2: agent has a plate and tile has a cutted ingredient
            elif agent.item == 'plate' and self.tile.item and self.tile.item.endswith('_cut'):
                self.tile.item = self.tile.item.split('_')[0] + '_salad'
                self.tile.salad_item = self.tile.item
                agent.item = None

            # Normal exchange
            else:
                _item = self.tile.item
                self.tile.item = agent.item
                agent.item = _item

    def finished(self, agent):
        return self.has_ended

class CuttingBoardIntent(_base_intent.Intent):
    def __init__(self, tile):
        super().__init__()
        self.tile = tile
        self.has_ended = False

    def update(self, agent):
        if not self.has_ended:
            # Single click behavior: instantly cut if agent has a valid item
            if agent.item in ['tomato', 'pumpkin', 'cabbage']:
                # Store original item and immediately create cut version
                original_item = agent.item
                cut_item = f"{original_item}_cut"
                
                # Give cut item directly to agent (instant cutting)
                if hasattr(agent, 'is_simulation') and agent.is_simulation:
                    agent.provisional_item = cut_item
                    agent.item = cut_item
                    # Set agent cutting state with start time for controller to manage waiting
                    agent.is_busy = True
                    agent.cutting_start_time = time.time()
                    # Get cutting_time from agent's game object
                    agent.cutting_duration = getattr(agent.game, 'cutting_time', 3.0) 
                else:
                    agent.item = cut_item
                
                # Set tile metadata for tracking
                self.tile.cut_item = cut_item
                self.tile.item = None  # Clear the cutting board
                
                # Intent completes immediately - cutting delay is handled by controller
                self.has_ended = True
            else:
                # No valid item to cut, end intent immediately
                self.tile.cut_item = None
                self.has_ended = True

    def finished(self, agent):
        return self.has_ended

class DeliveryIntent(_base_intent.Intent):
    def __init__(self, tile):
        super().__init__()
        self.tile = tile
        self.has_ended = False

    def update(self, agent):
        if not self.has_ended:
            self.has_ended = True
            valid_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']

            if agent.item in valid_items:
                self.tile.delivered_item = agent.item
                agent.item = None
                agent.score += 1
                if hasattr(self.tile, 'game') and 'score' in self.tile.game.gameObjects and self.tile.game.gameObjects['score'] is not None:
                    self.tile.game.gameObjects['score'].score += 1

    def finished(self, agent):
        return self.has_ended
