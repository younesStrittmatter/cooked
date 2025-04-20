from engine.extensions.topDownGridWorld.intent import _base_intent


class PickUpIntent(_base_intent.Intent):
    def __init__(self, tile):
        super().__init__()
        self.tile = tile
        self.has_ended = False

    def update(self, agent, delta_time: float):
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

    def update(self, agent, delta_time: float):
        if not self.has_ended:
            self.has_ended = True
            if agent.item and self.tile.item and agent.item.endswith('_cut') and self.tile.item == 'plate':
                self.tile.item = agent.item.split('_')[0] + '_salad'
                agent.item = None
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
        self.has_started = False

    def update(self, agent, delta_time: float):
        if not self.has_started:
            if self.tile.item is not None:
                if self.tile.cut_stage >= 3:
                    _a_temp = agent.item
                    agent.item = f'{self.tile.item}_cut'
                    if _a_temp in ['tomato', 'pumpkin', 'cabbage', None]:
                        self.tile.cut_time_accumulated = 0
                        self.tile.item = _a_temp
                    self.has_ended = True
                elif agent.item is None:
                    self.has_started = True
            else:
                if agent.item in ['tomato', 'pumpkin', 'cabbage']:
                    self.tile.cut_time_accumulated = 0
                    self.tile.item = agent.item
                    agent.item = None
                    self.has_started = True
                else:
                    self.has_ended = True
        else:
            self.tile.cut_time_accumulated += agent.cut_speed * delta_time

    def finished(self, agent):
        return self.has_ended


class DeliveryIntent(_base_intent.Intent):
    def __init__(self, tile):
        super().__init__()
        self.tile = tile
        self.has_ended = False

    def update(self, agent, delta_time: float):
        if not self.has_ended:
            self.has_ended = True
            if agent.item in ['tomato_salad', 'pumpkin_salad', 'cabbage_salad']:
                agent.item = None
                agent.score += 1
                if 'score' in self.tile.game.gameObjects is not None:
                    self.tile.game.gameObjects['score'].score += 1

    def finished(self, agent):
        return self.has_ended
