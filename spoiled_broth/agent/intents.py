from engine.extensions.topDownGridWorld.intent import _base_intent

class PickUpIntent(_base_intent.Intent):
    def __init__(self, tile, version):
        super().__init__()
        self.version = version
        self.tile = tile
        self.has_ended = False

    def update(self, agent, delta_time: float):
        if not self.has_ended:
            self.has_ended = True
            agent.item = self.tile.item

    def finished(self, agent):
        return self.has_ended


class ItemExchangeIntent(_base_intent.Intent):
    def __init__(self, tile, version):
        super().__init__()
        self.version = version
        self.tile = tile
        self.has_ended = False

    def update(self, agent, delta_time: float):
        if not self.has_ended:
            self.has_ended = True
            # Case 1: agent has a cutted ingredient and tile has a plate
            if agent.item and self.tile.item and agent.item.endswith('_cut') and self.tile.item == 'plate':
                self.tile.item = agent.item.split('_')[0] + '_salad'
                self.tile.salad_by = agent.id
                self.tile.salad_item = self.tile.item
                agent.item = None

            # Case 2: agent has a plate and tile has a cutted ingredient
            elif agent.item == 'plate' and self.tile.item and self.tile.item.endswith('_cut'):
                self.tile.item = self.tile.item.split('_')[0] + '_salad'
                self.tile.salad_by = agent.id
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
    def __init__(self, tile, version):
        super().__init__()
        self.tile = tile
        self.version = version
        self.has_ended = False
        self.has_started = False
        # elapsed cutting time while agent is at the board
        self._elapsed = 0.0
        # required cutting duration in seconds for the normal behavior
        self._required_cut_time = 3.0

    def update(self, agent, delta_time: float):
        if not self.has_started:
            if self.tile.item is not None:
                if self.tile.cut_stage >= 3:
                    _a_temp = agent.item
                    agent.item = f'{self.tile.item}_cut'
                    self.tile.cut_by = agent.id
                    self.tile.cut_item = agent.item
                    if _a_temp in ['tomato', 'pumpkin', 'cabbage']:
                        self.tile.cut_time_accumulated = 0
                        self.tile.item = _a_temp
                    else:
                        self.tile.item = None
                    self.has_ended = True
                elif agent.item is None:
                    self.has_started = True
                else:
                    self.has_ended = True
            else:
                if agent.item in ['tomato', 'pumpkin', 'cabbage']:
                    if self.version == "v2.1" or self.version == "v3.1":
                        # v2.1: cut instantly
                        agent.item = f'{agent.item}_cut'
                        self.tile.cut_by = agent.id
                        self.tile.cut_item = agent.item
                        self.tile.item = None
                        self.tile.cut_time_accumulated = 0
                        self.has_ended = True
                    elif self.version == "v2.2" or self.version == "v3.2":
                        # v2.2: put the item on the board and cut one time
                        self.tile.cut_time_accumulated = 1
                        self.tile.item = agent.item
                        agent.item = None
                        self.has_started = True
                    else:
                        # normal behavior: put the item on the board
                        self.tile.cut_time_accumulated = 0
                        self.tile.item = agent.item
                        agent.item = None
                        self.has_started = True
                else:
                    self.has_ended = True
        else:
            # accumulate local elapsed time while agent stays on the cutting board
            # agent.cut_speed may be used to scale speed, but we implement a fixed
            # wall-clock duration of 3.0 seconds (scaled by agent.cut_speed for
            # backwards compatibility if cut_speed != 1.0)
            try:
                speed = float(getattr(agent, 'cut_speed', 1.0))
            except Exception:
                speed = 1.0

            self._elapsed += delta_time * speed

            # also keep the tile-level accumulator in sync for compatibility
            try:
                self.tile.cut_time_accumulated += delta_time * speed
            except Exception:
                pass

            # When required duration reached, transfer cut item to agent and finish
            if self._elapsed >= self._required_cut_time:
                # Only perform the transfer if there is an ingredient on the board
                if self.tile.item is not None:
                    orig = self.tile.item
                    agent.item = f"{orig}_cut"
                    self.tile.cut_by = agent.id
                    self.tile.cut_item = agent.item
                    # Reset tile state: if agent took a basic ingredient back, place it
                    # otherwise clear the tile
                    # For this implementation, after cutting we clear the tile
                    self.tile.item = None
                    # Reset tile accumulator
                    try:
                        self.tile.cut_time_accumulated = 0
                    except Exception:
                        pass
                self.has_ended = True

    def finished(self, agent):
        return self.has_ended


class DeliveryIntent(_base_intent.Intent):
    def __init__(self, tile, version):
        super().__init__()
        self.tile = tile
        self.version = version
        self.has_ended = False

    def update(self, agent, delta_time: float):
        if not self.has_ended:
            self.has_ended = True

            if self.version == "v1":
                valid_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad', 'tomato']
            elif self.version == "v2.1" or self.version == "v2.2":
                valid_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad', 'tomato_cut']
            else:
                valid_items = ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']

            if agent.item in valid_items:
                self.tile.delivered_by = agent.id
                self.tile.delivered_item = agent.item
                agent.item = None
                agent.score += 1
                if 'score' in self.tile.game.gameObjects and self.tile.game.gameObjects['score'] is not None:
                    self.tile.game.gameObjects['score'].score += 1

    def finished(self, agent):
        return self.has_ended
