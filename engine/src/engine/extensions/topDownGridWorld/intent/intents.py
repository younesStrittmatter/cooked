from engine.extensions.topDownGridWorld.intent._base_intent import Intent
from engine.extensions.topDownGridWorld.grid import Tile
from engine.extensions.topDownGridWorld.agent import Agent
from engine.extensions.topDownGridWorld.a_star import Node, get_neighbors, get_neighbors_ghost


class MoveToIntent(Intent):
    def __init__(self, target: Tile):
        super().__init__()
        self.target = target
        self.has_started = False
        self.has_ended = False
        self.is_ghost = False

    def update(self, agent: Agent, delta_time: float):
        if not self.has_started and not self.has_ended:

            if self.target.is_walkable:
                d = agent.get_distance(self.target)
                agent.set_move_target(self.target)
                self.has_started = True
            else:

                _n = agent.node
                target_node = Node(self.target.slot_x, self.target.slot_y)
                neighbors = get_neighbors_ghost(agent.grid, target_node, include_diagonal=False)
                if len(neighbors) == 0:
                    return
                else:
                    d = 9999
                    target = None
                    for neighbor in neighbors:
                        _d = agent.get_distance_ghost(neighbor)
                        if _d is not None and _d < d:
                            d = _d
                            target = neighbor
                        if target is not None:
                            self.target = agent.grid.tiles[target.x][target.y]
                            self.is_ghost = True
                if target is None:
                    return
                #self.target = agent.grid.tiles[target.x][target.y]
        agent.move(delta_time)

    def finished(self, agent: Agent):
        if self.has_started and agent.path_index >= len(agent.path) and not agent.is_partial:
            self.has_started = False
            self.has_ended = True
            return True
        return False
