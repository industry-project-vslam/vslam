import pygame
import numpy as np
import random
import heapq
from collections import deque

# ============================================================
# CONFIGURATION
# ============================================================

GRID_W = 120
GRID_H = 80
CELL_SIZE = 10

WINDOW_W = GRID_W * CELL_SIZE
WINDOW_H = GRID_H * CELL_SIZE

NUM_ROBOTS = 6
SENSOR_RANGE = 6

OBSTACLE_DENSITY = 0.20

FPS = 60

UNKNOWN = -1
FREE = 0
OBSTACLE = 1

# ============================================================
# MAP GENERATION
# ============================================================

def generate_world():
    world = np.zeros((GRID_H, GRID_W), dtype=np.int8)

    for y in range(GRID_H):
        for x in range(GRID_W):
            if random.random() < OBSTACLE_DENSITY:
                world[y, x] = OBSTACLE

    # Make sure origin region is free
    for y in range(8):
        for x in range(8):
            world[y, x] = FREE

    return world

# ============================================================
# A*
# ============================================================

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:

        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []

            while current in came_from:
                path.append(current)
                current = came_from[current]

            path.reverse()
            return path

        x, y = current

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:

            nx = x + dx
            ny = y + dy

            if nx < 0 or ny < 0 or nx >= GRID_W or ny >= GRID_H:
                continue

            if grid[ny, nx] == OBSTACLE:
                continue

            neighbor = (nx, ny)

            tentative = g_score[current] + 1

            if tentative < g_score.get(neighbor, float("inf")):

                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score[neighbor] = tentative + heuristic(neighbor, goal)

                heapq.heappush(
                    open_set,
                    (f_score[neighbor], neighbor)
                )

    return []

# ============================================================
# FRONTIER DETECTION
# ============================================================

def find_frontiers(explored):

    frontiers = []

    for y in range(1, GRID_H - 1):
        for x in range(1, GRID_W - 1):

            if explored[y, x] != FREE:
                continue

            has_unknown_neighbor = False

            for dx, dy in [
                (-1,0),
                (1,0),
                (0,-1),
                (0,1)
            ]:

                nx = x + dx
                ny = y + dy

                if explored[ny, nx] == UNKNOWN:
                    has_unknown_neighbor = True
                    break

            if has_unknown_neighbor:
                frontiers.append((x, y))

    return frontiers

# ============================================================
# ROBOT
# ============================================================

def build_planning_grid(explored):
    planning = explored.copy()
    planning[planning == UNKNOWN] = OBSTACLE
    return planning

class Robot:

    COLORS = [
        (255,0,0),
        (0,255,0),
        (0,150,255),
        (255,100,0),
        (255,0,255),
        (0,255,255),
        (255,255,0),
        (150,0,255),
    ]

    def __init__(self, idx, x, y):

        self.idx = idx
        self.x = x
        self.y = y

        self.path = []
        self.target = None

        self.color = Robot.COLORS[idx % len(Robot.COLORS)]

    def sensor_update(self, world, explored):

        for dy in range(-SENSOR_RANGE, SENSOR_RANGE + 1):
            for dx in range(-SENSOR_RANGE, SENSOR_RANGE + 1):

                nx = self.x + dx
                ny = self.y + dy

                if nx < 0 or ny < 0:
                    continue

                if nx >= GRID_W or ny >= GRID_H:
                    continue

                explored[ny, nx] = world[ny, nx]

    def choose_frontier(self, frontiers, explored, claimed):
        planning_grid = build_planning_grid(explored)

        best_frontier = None
        best_path = None
        best_cost = float("inf")

        for frontier in frontiers:

            if frontier in claimed:
                continue

            path = astar(
                planning_grid,
                (self.x, self.y),
                frontier
            )

            if not path:
                continue

            cost = len(path)

            if cost < best_cost:
                best_cost = cost
                best_frontier = frontier
                best_path = path

        if best_frontier is not None:

            self.target = best_frontier
            self.path = best_path

            claimed.add(best_frontier)

    def move(self):

        if not self.path:
            return

        nx, ny = self.path.pop(0)

        self.x = nx
        self.y = ny

# ============================================================
# MAIN
# ============================================================

pygame.init()

screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Multi-Robot Frontier Exploration")

clock = pygame.time.Clock()

world = generate_world()

explored = np.full((GRID_H, GRID_W), UNKNOWN, dtype=np.int8)

robots = []

for i in range(NUM_ROBOTS):
    robots.append(Robot(i, 0, 0))

running = True

# ============================================================
# LOOP
# ============================================================

while running:

    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --------------------------------
    # SENSE
    # --------------------------------

    for robot in robots:
        robot.sensor_update(world, explored)

    # --------------------------------
    # FRONTIERS
    # --------------------------------

    planning_grid = build_planning_grid(explored)

    frontiers = find_frontiers(explored)

    claimed = set()

    for robot in robots:

        if not robot.path:

            robot.choose_frontier(
                frontiers,
                explored,
                claimed
            )

    # --------------------------------
    # MOVE
    # --------------------------------

    for robot in robots:
        robot.move()

    # --------------------------------
    # DRAW MAP
    # --------------------------------

    screen.fill((30, 30, 30))

    for y in range(GRID_H):
        for x in range(GRID_W):

            cell = explored[y, x]

            if cell == UNKNOWN:
                color = (100, 100, 100)

            elif cell == FREE:
                color = (240, 240, 240)

            else:
                color = (0, 0, 0)

            pygame.draw.rect(
                screen,
                color,
                (
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
            )

    # Draw frontiers
    for x, y in frontiers:

        pygame.draw.rect(
            screen,
            (255, 255, 0),
            (
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
        )

    # Draw robot paths
    for robot in robots:

        if len(robot.path) > 1:

            pts = [
                (
                    x * CELL_SIZE + CELL_SIZE // 2,
                    y * CELL_SIZE + CELL_SIZE // 2
                )
                for x, y in robot.path
            ]

            pygame.draw.lines(
                screen,
                robot.color,
                False,
                pts,
                2
            )

    # Draw robots
    for robot in robots:

        pygame.draw.circle(
            screen,
            robot.color,
            (
                robot.x * CELL_SIZE + CELL_SIZE // 2,
                robot.y * CELL_SIZE + CELL_SIZE // 2
            ),
            CELL_SIZE // 2
        )

    pygame.display.flip()

pygame.quit()