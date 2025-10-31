import math


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Node) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Node({self.x}, {self.y})"


class _PriorityQueue:
    def __init__(self, debug=False):
        self.queue = []
        self.debug = debug

    def enqueue(self, element, priority):
        if not isinstance(priority, (int, float)) or math.isnan(priority):
            raise ValueError(f"Invalid priority value: {priority} for element: {element}")
        self.queue.append((element, priority))
        self.queue.sort(key=lambda x: x[1])
        if self.debug:
            print(f"[Enqueue] Added: {element}, Priority: {priority}")
            print(f"[Queue] State: {[p[0] for p in self.queue]}")

    def dequeue(self):
        if len(self.queue) <= 0:
            if self.debug:
                print("[Dequeue] Queue is empty!")
            return None
        element = self.queue.pop(0)[0]
        if self.debug:
            print(f"[Dequeue] Returning: {element}")
        return element


def euclidean_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path


def is_obstacle(grid, x, y):
    return not grid.tiles[x][y].is_walkable


def is_valid(node, grid):
    return 0 <= node.x < grid.width and 0 <= node.y < grid.height


def get_neighbors(grid, node, include_diagonal=True):
    candidates = [
        Node(node.x, node.y - 1),
        Node(node.x, node.y + 1),
        Node(node.x - 1, node.y),
        Node(node.x + 1, node.y)
    ]
    if include_diagonal:
        candidates += [
            Node(node.x - 1, node.y - 1),
            Node(node.x - 1, node.y + 1),
            Node(node.x + 1, node.y - 1),
            Node(node.x + 1, node.y + 1)
        ]
    return [n for n in candidates if is_valid(n, grid) and not is_obstacle(grid, n.x, n.y)]


def world_to_grid(x, y, grid):
    grid_x = math.floor(x / grid.tile_size)
    grid_y = math.floor(y / grid.tile_size)
    return Node(grid_x, grid_y)


def grid_to_world(node, grid):
    x = node.x * grid.tile_size + grid.tile_size / 2
    y = node.y * grid.tile_size + grid.tile_size / 2
    return {'x': x, 'y': y}


def a_star(grid, start, goal):
    open_set = _PriorityQueue()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: euclidean_distance(start, goal)}
    open_set.enqueue(start, f_score[start])

    while open_set.queue:
        current = open_set.dequeue()

        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in get_neighbors(grid, current):
            tentative_g = g_score[current] + euclidean_distance(current, neighbor)
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + euclidean_distance(neighbor, goal)

                if neighbor not in [n for n, _ in open_set.queue]:
                    open_set.enqueue(neighbor, f_score[neighbor])

    return None  # No path found

def find_path(grid, start, goal, is_walkable=True):
    if is_walkable:
        return a_star(grid, start, goal)
    else:
        target_node = goal
        neighbors = get_neighbors(grid, target_node, include_diagonal=False)
        if len(neighbors) == 0:
            return None
        else:
            d = 9999
            target = None
            for neighbor in neighbors:
                _d = distance(grid, start, neighbor)
                if _d is not None and _d < d:
                    d = _d
                    target = neighbor
        if target is None:
            return None
        return a_star(grid, start, target)

def path_length(path):
    if not path or len(path) < 2:
        return 0
    return sum(euclidean_distance(path[i], path[i + 1]) for i in range(len(path) - 1))


def distance(grid, start, goal):
    path = find_path(grid, start, goal)
    return path_length(path) if path else None
