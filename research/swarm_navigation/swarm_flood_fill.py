import pygame
import random
from collections import deque

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 40
CELL = WIDTH // GRID_SIZE

NUM_ROBOTS = 6
OBSTACLE_DENSITY = 0.15

STEP_RATE = 10

# Colors
BG = (20, 20, 20)
WALL = (60, 60, 60)
FREE = (210, 210, 210)
UNEXPLORED = (25, 25, 25)
FRONTIER = (255, 140, 0)

ROBOT_COLORS = [
    (0, 200, 255),
    (0, 255, 150),
    (255, 80, 80),
    (255, 220, 0),
    (180, 0, 255),
    (0, 180, 255),
]

# ---------------- GRID ----------------
WALL_SPACE = 1
FREE_SPACE = 0

grid = [
    [FREE_SPACE if random.random() > OBSTACLE_DENSITY else WALL_SPACE
     for _ in range(GRID_SIZE)]
    for _ in range(GRID_SIZE)
]

explored = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# ---------------- ROBOTS ----------------
used_positions = set()
robots = []

for i in range(NUM_ROBOTS):
    while True:
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)

        if grid[y][x] == FREE_SPACE and (x, y) not in used_positions:
            used_positions.add((x, y))
            robots.append({
                "x": x,
                "y": y,
                "target": None
            })
            break

# ---------------- HELPERS ----------------
def in_bounds(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def neighbors(x, y):
    # 8-direction movement (omnidirectional grid motion)
    for dx, dy in [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny):
            yield nx, ny


# ---------------- FRONTIER DETECTION ----------------
def get_frontiers():
    frontiers = []

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):

            if not explored[y][x]:
                continue
            if grid[y][x] == WALL_SPACE:
                continue

            # if touches unexplored -> frontier
            for nx, ny in neighbors(x, y):
                if not explored[ny][nx] and grid[ny][nx] == FREE_SPACE:
                    frontiers.append((x, y))
                    break

    return frontiers


# ---------------- FLOOD FILL (BFS) ----------------
def bfs_distances(start):
    sx, sy = start
    q = deque([(sx, sy)])
    dist = {(sx, sy): 0}

    while q:
        x, y = q.popleft()

        for nx, ny in neighbors(x, y):
            if grid[ny][nx] == WALL_SPACE:
                continue

            if (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))

    return dist


# ---------------- FRONTIER ASSIGNMENT (IMPORTANT FIX) ----------------
def assign_frontiers(robots, frontiers):
    assigned = set()

    # precompute which areas are already "claimed"
    frontier_usage = {f: 0 for f in frontiers}

    for r in robots:
        dist_map = bfs_distances((r["x"], r["y"]))

        best_f = None
        best_score = float("inf")

        for f in frontiers:
            if f in assigned:
                continue
            if f not in dist_map:
                continue

            # -------------------------
            # CORE FIX: scoring function
            # -------------------------
            distance = dist_map[f]

            # congestion penalty (prevents collapse)
            congestion = frontier_usage[f] * 25

            # slight spatial diversity bias (breaks top-left pull)
            spatial_bias = (f[0] + f[1]) * 0.05

            score = distance + congestion + spatial_bias

            if score < best_score:
                best_score = score
                best_f = f

        if best_f is not None:
            r["target"] = best_f
            assigned.add(best_f)
            frontier_usage[best_f] += 1


# ---------------- MOVEMENT ----------------
def step_toward(a, b):
    ax, ay = a
    bx, by = b

    best = a
    best_d = float("inf")

    for nx, ny in neighbors(ax, ay):
        if grid[ny][nx] == WALL_SPACE:
            continue

        d = (nx - bx) ** 2 + (ny - by) ** 2
        if d < best_d:
            best_d = d
            best = (nx, ny)

    return best


# ---------------- INIT PYGAME ----------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

running = True

# ---------------- MAIN LOOP ----------------
while running:
    clock.tick(STEP_RATE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # mark explored
    for r in robots:
        explored[r["y"]][r["x"]] = True

    # compute frontiers
    frontiers = get_frontiers()

    # assign targets (FIXED SWARM BEHAVIOR)
    assign_frontiers(robots, frontiers)

    # move robots
    for r in robots:
        if r["target"] is not None:
            r["x"], r["y"] = step_toward((r["x"], r["y"]), r["target"])

    # ---------------- DRAW ----------------
    screen.fill(BG)

    # grid
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)

            if grid[y][x] == WALL_SPACE:
                color = WALL
            elif not explored[y][x]:
                color = UNEXPLORED
            else:
                color = FREE

            pygame.draw.rect(screen, color, rect)

    # frontiers
    for fx, fy in frontiers:
        pygame.draw.rect(
            screen,
            FRONTIER,
            (fx * CELL, fy * CELL, CELL, CELL)
        )

    # robots
    for i, r in enumerate(robots):
        pygame.draw.circle(
            screen,
            ROBOT_COLORS[i % len(ROBOT_COLORS)],
            (r["x"] * CELL + CELL // 2,
             r["y"] * CELL + CELL // 2),
            CELL // 3
        )

    pygame.display.flip()

pygame.quit()