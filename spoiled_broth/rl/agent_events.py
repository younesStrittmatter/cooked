# ---- Agent events processing ---- #
def get_agent_events(self, agent_events):
    """
    Update agent events based on the current game state.
    """
    if self.game_mode == "classic":
        return get_agent_events_classic(self, agent_events)
    elif self.game_mode == "competition":
        return get_agent_events_competition(self, agent_events)
    else:
        raise ValueError(f"Unknown game mode: {self.game_mode}")
    
# ---- Classic mode without ownership awareness ---- #
def get_agent_events_classic(self, agent_events):
    for thing in self.game.gameObjects.values():
        if hasattr(thing, 'tiles'):
            for row in thing.tiles:
                for tile in row:
                    # Detect CUT
                    if hasattr(tile, "cut_by") and tile.cut_by:
                        if tile.cut_by in agent_events:
                            agent_events[tile.cut_by]["cut"] += 1
                            self.total_agent_events[tile.cut_by]["cut"] += 1
                        tile.cut_by = None

                    # Detect SALAD
                    if hasattr(tile, "salad_by") and tile.salad_by:
                        if tile.salad_by in agent_events:
                            agent_events[tile.salad_by]["salad"] += 1
                            self.total_agent_events[tile.salad_by]["salad"] += 1
                        tile.salad_by = None

    # Detect DELIVERY
    new_score = self.game.gameObjects["score"].score
    if (new_score - self._last_score) > 0:
        # Only reward the delivering agent(s)
        for thing in self.game.gameObjects.values():
            if hasattr(thing, 'tiles'):
                for row in thing.tiles:
                    for tile in row:
                        if hasattr(tile, "delivered_by") and tile.delivered_by:
                            if tile.delivered_by in agent_events:
                                agent_events[tile.delivered_by]["delivered"] += 1
                                self.total_agent_events[tile.delivered_by]["delivered"] += 1
                            tile.delivered_by = None
    self._last_score = new_score
    
    return self.total_agent_events, agent_events, self._last_score

# ---- Competition mode with ownership awareness ---- #
def get_agent_events_competition(self, agent_events):
    # Detect cut and salad events
    for thing in self.game.gameObjects.values():
        if hasattr(thing, 'tiles'):
            for row in thing.tiles:
                for tile in row:
                    # Detect CUT
                    if hasattr(tile, "cut_by") and tile.cut_by:
                        agent_id = tile.cut_by
                        if agent_id in agent_events:
                            food = getattr(tile, "cut_item", "")
                            if food.startswith(self.agent_food_type[agent_id]):
                                agent_events[agent_id]["cut_own"] += 1
                                self.total_result_events[agent_id]["cut_own"] += 1
                            else:
                                agent_events[agent_id]["cut_other"] += 1
                                self.total_result_events[agent_id]["cut_other"] += 1
                        tile.cut_by = None
                        tile.cut_item = None

                    # Detect SALAD
                    if hasattr(tile, "salad_by") and tile.salad_by:
                        agent_id = tile.salad_by
                        if agent_id in agent_events:
                            food = getattr(tile, "salad_item", "")
                            if food.startswith(self.agent_food_type[agent_id]):
                                agent_events[agent_id]["salad_own"] += 1
                                self.total_result_events[agent_id]["salad_own"] += 1
                            else:
                                agent_events[agent_id]["salad_other"] += 1
                                self.total_result_events[agent_id]["salad_other"] += 1
                        tile.salad_by = None
                tile.salad_item = None

    # Detect DELIVERY
    new_score = self.game.gameObjects["score"].score
    if (new_score - self._last_score) > 0:
        for thing in self.game.gameObjects.values():
            if hasattr(thing, 'tiles'):
                for row in thing.tiles:
                    for tile in row:
                        if hasattr(tile, "delivered_by") and tile.delivered_by:
                            agent_id = tile.delivered_by
                            if agent_id in agent_events:
                                food = getattr(tile, "delivered_item", "")
                                if food.startswith(self.agent_food_type[agent_id]):
                                    agent_events[agent_id]["delivered_own"] += 1
                                    self.total_result_events[agent_id]["delivered_own"] += 1
                                else:
                                    agent_events[agent_id]["delivered_other"] += 1
                                    self.total_result_events[agent_id]["delivered_other"] += 1
                            tile.delivered_by = None
                            tile.delivered_item = None
    self._last_score = new_score

    return self.total_agent_events, agent_events, self._last_score