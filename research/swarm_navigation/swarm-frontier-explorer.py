import pygame
import numpy as np
import random
import heapq

# ============================================================
# CONFIG
# ============================================================

GRID_W = 120
GRID_H = 80
CELL_SIZE = 10

WINDOW_W = GRID_W * CELL_SIZE
WINDOW_H = GRID_H * CELL_SIZE

NUM_ROBOTS = 6
SENSOR_RANGE = 6

FPS = 60

OBSTACLE_DENSITY = 0.06
OBSTACLE_SCALE = 6

# log odds
LO_OCC = 0.9
LO_FREE = -0.45
LO_MIN = -5.0
LO_MAX = 5.0
OCC_THRESH = 0.4


# ============================================================
# WORLD
# ============================================================

def generate_world():
    world = np.zeros((GRID_H, GRID_W), dtype=np.int8)

    for y in range(0, GRID_H, OBSTACLE_SCALE):
        for x in range(0, GRID_W, OBSTACLE_SCALE):

            if random.random() < OBSTACLE_DENSITY:

                for dy in range(OBSTACLE_SCALE):
                    for dx in range(OBSTACLE_SCALE):

                        yy = y + dy
                        xx = x + dx

                        if 0 <= yy < GRID_H and 0 <= xx < GRID_W:
                            world[yy, xx] = 1

    world[:10, :10] = 0
    return world


# ============================================================
# SWARM SLAM MAP
# ============================================================

def init_log_odds():
    return np.zeros((GRID_H, GRID_W), dtype=np.float32)


def update_log_odds(log_odds, x, y, is_occ, dx, dy):

    dist = (dx*dx + dy*dy) ** 0.5
    weight = 1.0 / (1.0 + dist)

    if is_occ:
        log_odds[y, x] += LO_OCC * weight
    else:
        log_odds[y, x] += LO_FREE * weight

    log_odds[y, x] = np.clip(log_odds[y, x], LO_MIN, LO_MAX)


def occupancy(log_odds):

    grid = np.zeros_like(log_odds, dtype=np.int8)

    # strongly occupied
    grid[log_odds > 1.5] = 1

    # strongly free
    grid[log_odds < -0.8] = 0

    # unknown stays "neutral"
    return grid


# ============================================================
# FRONTIER MANAGER (INCREMENTAL)
# ============================================================

class FrontierManager:

    def __init__(self):
        self.frontiers = set()

    def update(self, log_odds, changed):

        for x, y in changed:

            if log_odds[y, x] > OCC_THRESH:
                self.frontiers.discard((x, y))
                continue

            if abs(log_odds[y, x]) > 0.6:
                self.frontiers.discard((x, y))
                continue

            # check neighbors
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:

                nx, ny = x + dx, y + dy

                if 0 <= nx < GRID_W and 0 <= ny < GRID_H:

                    if abs(log_odds[ny, nx]) < 0.4:
                        self.frontiers.add((x, y))
                        break


# ============================================================
# A* (BOUNDED)
# ============================================================

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def astar(grid, start, goal):

    open_set = [(0, start)]
    came = {}
    g = {start: 0}

    expansions = 0
    MAX_EXP = 1200

    while open_set:

        _, cur = heapq.heappop(open_set)

        expansions += 1
        if expansions > MAX_EXP:
            return []

        if cur == goal:
            path = []
            while cur in came:
                path.append(cur)
                cur = came[cur]
            return path[::-1]

        x, y = cur

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:

            nx, ny = x + dx, y + dy

            if nx < 0 or ny < 0 or nx >= GRID_W or ny >= GRID_H:
                continue

            if grid[ny, nx] == 1:
                continue

            n = (nx, ny)
            ng = g[cur] + 1

            if ng < g.get(n, 1e9):

                g[n] = ng
                came[n] = cur
                f = ng + heuristic(n, goal)

                heapq.heappush(open_set, (f, n))

    return []


# ============================================================
# ROBOT
# ============================================================

class Robot:

    COLORS = [
        (255,0,0),(0,255,0),(0,150,255),
        (255,100,0),(255,0,255),(0,255,255)
    ]

    SPEED = 2.2

    def __init__(self, idx, x, y):

        self.x = float(x)
        self.y = float(y)

        self.idx = idx
        self.color = Robot.COLORS[idx % len(Robot.COLORS)]

        self.path = []
        self.cooldown = 0

    # --------------------------------------------------------
    # SENSOR + CHANGE TRACKING
    # --------------------------------------------------------
    def sensor_update(self, world, log_odds):

        changed = []

        cx, cy = int(self.x), int(self.y)

        for dy in range(-SENSOR_RANGE, SENSOR_RANGE + 1):
            for dx in range(-SENSOR_RANGE, SENSOR_RANGE + 1):

                x, y = cx + dx, cy + dy

                if x < 0 or y < 0 or x >= GRID_W or y >= GRID_H:
                    continue

                before = log_odds[y, x]

                is_occ = (world[y, x] == 1)

                dx = x - cx
                dy = y - cy

                update_log_odds(log_odds, x, y, is_occ, dx, dy)

                after = log_odds[y, x]

                if abs(after - before) > 0.2:
                    changed.append((x, y))

        return changed

    # --------------------------------------------------------
    # PLANNING
    # --------------------------------------------------------
    def choose_frontier(self, frontiers, log_odds, claimed):

        grid = occupancy(log_odds)

        start = (int(self.x), int(self.y))

        candidates = sorted(
            frontiers,
            key=lambda f: heuristic(start, f)
        )[:30]

        best = None
        best_path = None
        best_cost = 1e9

        for f in candidates:

            if f in claimed:
                continue

            path = astar(grid, start, f)

            if not path:
                continue

            if len(path) < best_cost:
                best_cost = len(path)
                best = f
                best_path = path

        if best:
            self.path = best_path
            claimed.add(best)
            self.cooldown = 15

    # --------------------------------------------------------
    # MOTION
    # --------------------------------------------------------
    def move(self):

        if self.cooldown > 0:
            self.cooldown -= 1

        if not self.path:
            return

        tx, ty = self.path[0]
        tx += 0.5
        ty += 0.5

        dx = tx - self.x
        dy = ty - self.y

        dist = (dx*dx + dy*dy) ** 0.5

        if dist < 0.15:
            self.path.pop(0)
            return

        self.x += (dx / dist) * self.SPEED
        self.y += (dy / dist) * self.SPEED


# ============================================================
# FRONTIER DETECTION (LIGHTWEIGHT LOCAL CHECK)
# ============================================================

def find_frontiers(log_odds):

    frontiers = []

    for y in range(1, GRID_H - 1):
        for x in range(1, GRID_W - 1):

            # must be UNKNOWN-ish (not strongly classified)
            if abs(log_odds[y, x]) > 0.5:
                continue

            # must touch known free space AND unknown space
            has_free = False
            has_unknown = False

            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:

                v = log_odds[y+dy, x+dx]

                if v < -0.8:
                    has_free = True
                if abs(v) < 0.5:
                    has_unknown = True

            if has_free and has_unknown:
                frontiers.append((x, y))

    return frontiers


# ============================================================
# INIT
# ============================================================

pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
clock = pygame.time.Clock()

world = generate_world()
log_odds = init_log_odds()

robots = [
    Robot(i, 2 + i, 2)
    for i in range(NUM_ROBOTS)
]

frontier_manager = FrontierManager()


# ============================================================
# LOOP
# ============================================================

running = True

while running:

    clock.tick(FPS)

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # -----------------------
    # SWARM SLAM UPDATE
    # -----------------------
    changed = []

    for r in robots:
        changed += r.sensor_update(world, log_odds)

    frontier_manager.update(log_odds, changed)

    frontiers = list(frontier_manager.frontiers)

    # -----------------------
    # PLANNING
    # -----------------------
    claimed = set()

    for r in robots:
        if not r.path and r.cooldown == 0:
            r.choose_frontier(frontiers, log_odds, claimed)

    # -----------------------
    # MOTION
    # -----------------------
    for r in robots:
        r.move()

    # -----------------------
    # RENDER
    # -----------------------
    screen.fill((30, 30, 30))

    occ = occupancy(log_odds)

    for y in range(GRID_H):
        for x in range(GRID_W):

            if occ[y, x] == 1:
                col = (0, 0, 0)
            else:
                col = (240, 240, 240)

            pygame.draw.rect(
                screen,
                col,
                (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )

    for x, y in frontiers:
        pygame.draw.rect(
            screen,
            (255, 220, 0),
            (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )

    for r in robots:
        pygame.draw.circle(
            screen,
            r.color,
            (int(r.x * CELL_SIZE), int(r.y * CELL_SIZE)),
            CELL_SIZE // 2
        )

    pygame.display.flip()

pygame.quit()