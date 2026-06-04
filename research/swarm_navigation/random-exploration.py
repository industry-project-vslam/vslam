import pygame
import random

# ---------------- CONFIG ----------------
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 40
CELL = WIDTH // GRID_SIZE

NUM_ROBOTS = 10
OBSTACLE_DENSITY = 0.15

STEP_RATE = 12

# Colors
BG = (20, 20, 20)
WALL = (60, 60, 60)
FREE = (210, 210, 210)
UNEXPLORED = (25, 25, 25)

ROBOT_COLORS = [
    (0, 200, 255),
    (0, 255, 150),
    (255, 80, 80),
    (255, 220, 0),
    (180, 0, 255),
    (255, 120, 0),
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

# ---------------- HELPERS ----------------
def in_bounds(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def neighbors(x, y):
    # omnidirectional movement (8-direction)
    dirs = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    random.shuffle(dirs)  # makes movement more chaotic

    for dx, dy in dirs:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny):
            yield nx, ny


def random_move(x, y):
    for nx, ny in neighbors(x, y):
        if grid[ny][nx] == FREE_SPACE:
            return nx, ny
    return x, y


def avoid_collision(x, y, robots):
    for r in robots:
        if abs(r["x"] - x) + abs(r["y"] - y) <= 1:
            return False
    return True


# ---------------- ROBOTS ----------------
robots = []

used = set()
for _ in range(NUM_ROBOTS):
    while True:
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)

        if grid[y][x] == FREE_SPACE and (x, y) not in used:
            used.add((x, y))
            robots.append({"x": x, "y": y})
            break


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

    # move robots randomly
    for r in robots:
        nx, ny = random_move(r["x"], r["y"])

        if avoid_collision(nx, ny, robots):
            r["x"], r["y"] = nx, ny

    # ---------------- DRAW ----------------
    screen.fill(BG)

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