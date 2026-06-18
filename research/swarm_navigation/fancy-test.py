# explore_pygame_spread.py
# Continuous 2D multi-robot exploration simulator using pygame.
# All robots start at the center with repulsion to prevent clustering.
# Requirements: pygame (pip install pygame)

import pygame
import random
import math
import sys
from collections import deque

# ---------- Config ----------
SCREEN_W, SCREEN_H = 900, 700
WORLD_W, WORLD_H = 9.0, 7.0    # world in meters (1 meter = 100 pixels)
SCALE = SCREEN_W / WORLD_W
FPS = 30

NUM_ROBOTS = 6
ROBOT_RADIUS = 0.12   # meters
MAX_SPEED = 0.8       # m/s
TURN_RATE = math.pi   # rad/s
SENSOR_RADIUS = 1.0   # meters: how far a robot 'sees' (drops exploration pings)
PING_RATE = 0.5       # seconds between dropping exploration pings
GOAL_RECOMPUTE = 1.0  # seconds between goal selection
EXPLORATION_GRID_RES = 0.25  # internal occupancy sampling resolution (meters)
REPULSION_RADIUS = 2.0  # meters: within this distance, robots repel each other
REPULSION_STRENGTH = 1.5  # strength of repulsive force

# Colors
BG_COLOR = (20, 20, 30)
ROBOT_COLOR = (50, 200, 120)
ROBOT_GOAL_COLOR = (255, 100, 80)
PING_COLOR = (200, 200, 255)
OBST_COLOR = (130, 130, 140)

# ---------- Utilities ----------
def world_to_screen(pos):
    x, y = pos
    sx = int(x * SCALE)
    sy = int(SCREEN_H - y * SCALE)
    return sx, sy

def clamp(a, lo, hi): return max(lo, min(hi, a))

def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

# ---------- Environment ----------
class Obstacle:
    def __init__(self, rect):
        self.rect = rect  # (x, y, w, h) in meters (x,y lower-left)
    def collides_point(self, p):
        x,y = p
        rx, ry, rw, rh = self.rect
        return (rx <= x <= rx+rw) and (ry <= y <= ry+rh)
    def draw(self, surf):
        rx,ry,rw,rh = self.rect
        x,y = world_to_screen((rx, ry+rh))
        w = int(rw * SCALE)
        h = int(rh * SCALE)
        pygame.draw.rect(surf, OBST_COLOR, (x, y, w, h))

# ---------- Simple shared map of exploration pings ----------
class ExplorationMap:
    def __init__(self, world_w, world_h, res):
        self.w = world_w; self.h = world_h; self.res = res
        self.grid_w = int(math.ceil(world_w / res))
        self.grid_h = int(math.ceil(world_h / res))
        self.cells = [[0 for _ in range(self.grid_h)] for __ in range(self.grid_w)]
    def mark_circle(self, pos, radius):
        cx, cy = pos
        r = int(math.ceil(radius / self.res))
        ci = int(cx / self.res); cj = int(cy / self.res)
        for i in range(ci - r, ci + r + 1):
            if i < 0 or i >= self.grid_w: continue
            for j in range(cj - r, cj + r + 1):
                if j < 0 or j >= self.grid_h: continue
                x = (i + 0.5) * self.res
                y = (j + 0.5) * self.res
                if dist((cx,cy),(x,y)) <= radius:
                    self.cells[i][j] = 1
    def unexplored_points(self):
        pts = []
        for i in range(self.grid_w):
            for j in range(self.grid_h):
                if self.cells[i][j] == 0:
                    x = (i + 0.5) * self.res
                    y = (j + 0.5) * self.res
                    pts.append((x,y))
        return pts
    def fraction_explored(self):
        total = self.grid_w * self.grid_h
        seen = sum(self.cells[i][j] for i in range(self.grid_w) for j in range(self.grid_h))
        return seen / total

# ---------- Robot ----------
class Robot:
    def __init__(self, idx, pos, theta):
        self.id = idx
        self.pos = list(pos)
        self.theta = theta
        self.v = 0.0
        self.omega = 0.0
        self.goal = None
        self.last_ping = 0.0
        self.last_goal_compute = 0.0
        self.vx = 0.0  # velocity x component
        self.vy = 0.0  # velocity y component
    def step(self, dt, world, all_robots):
        # Compute repulsion force from other robots
        repulsion_x = 0.0
        repulsion_y = 0.0
        
        for other in all_robots:
            if other is self:
                continue
            dx = self.pos[0] - other.pos[0]
            dy = self.pos[1] - other.pos[1]
            d = math.hypot(dx, dy)
            if d > 0 and d < REPULSION_RADIUS:
                # Repulsion strength increases as distance decreases
                force = REPULSION_STRENGTH * (1.0 - d / REPULSION_RADIUS) / max(d, 0.01)
                repulsion_x += force * dx / d
                repulsion_y += force * dy / d
        
        # Apply repulsion to velocity
        self.vx += repulsion_x * dt
        self.vy += repulsion_y * dt
        
        # If have a goal, steer toward it
        if self.goal is not None:
            gx, gy = self.goal
            angle_to_goal = math.atan2(gy - self.pos[1], gx - self.pos[0])
            aerr = (angle_to_goal - self.theta + math.pi) % (2*math.pi) - math.pi
            # steer toward goal
            self.omega = clamp(aerr * 3.0, -TURN_RATE, TURN_RATE)
            # set forward speed reduced by heading error, but always move forward
            goal_speed = MAX_SPEED * max(0.5, clamp(1.0 - abs(aerr)/math.pi, 0.3, 1.0))
        else:
            self.omega = 0.0
            goal_speed = MAX_SPEED * 0.5  # move slowly if no goal
        
        # Combine goal direction with repulsion velocity
        goal_dx = math.cos(self.theta) * goal_speed
        goal_dy = math.sin(self.theta) * goal_speed
        
        # Apply repulsion to velocity (limit it)
        total_speed = math.hypot(goal_dx + self.vx, goal_dy + self.vy)
        if total_speed > MAX_SPEED:
            scale = MAX_SPEED / total_speed
            goal_dx *= scale
            goal_dy *= scale
        
        # Update velocity
        self.vx = clamp(self.vx, -MAX_SPEED, MAX_SPEED)
        self.vy = clamp(self.vy, -MAX_SPEED, MAX_SPEED)
        
        # Move in combined direction
        move_x = goal_dx + self.vx * 0.5
        move_y = goal_dy + self.vy * 0.5
        
        # Check simple collision lookahead
        lookahead = 0.2
        nx = self.pos[0] + move_x * dt + math.cos(self.theta) * lookahead
        ny = self.pos[1] + move_y * dt + math.sin(self.theta) * lookahead
        if not world.point_in_obstacle((nx,ny)):
            self.v = math.hypot(move_x, move_y)
        else:
            self.v = 0.0
            move_x = 0.0
            move_y = 0.0
        
        # Update heading based on movement direction
        if math.hypot(move_x, move_y) > 0.01:
            self.theta = math.atan2(move_y, move_x)
        
        # Integrate position
        self.pos[0] += move_x * dt
        self.pos[1] += move_y * dt
        
        # Decay repulsion velocity
        self.vx *= 0.95
        self.vy *= 0.95
        
        # clamp inside world bounds
        self.pos[0] = clamp(self.pos[0], 0.01, world.width-0.01)
        self.pos[1] = clamp(self.pos[1], 0.01, world.height-0.01)

# ---------- World ----------
class World:
    def __init__(self, width, height):
        self.width = width; self.height = height
        self.robots = []
        self.obstacles = []
        self.expl_map = ExplorationMap(width, height, EXPLORATION_GRID_RES)
        self.time = 0.0
    def add_robot(self, r):
        self.robots.append(r)
    def add_obstacle(self, obs):
        self.obstacles.append(obs)
    def point_in_obstacle(self, p):
        for o in self.obstacles:
            if o.collides_point(p): return True
        return False
    def step(self, dt):
        # update each robot; they drop pings and occasionally recompute goals
        for r in self.robots:
            r.step(dt, self, self.robots)
            # drop exploration ping
            if self.time - r.last_ping >= PING_RATE:
                r.last_ping = self.time
                self.expl_map.mark_circle(tuple(r.pos), SENSOR_RADIUS * 0.6)
        self.time += dt

    def compute_goal_for_robot(self, robot):
        # choose an unexplored sample that maximizes a simple score:
        # score = distance_to_robot * (1 - nearby_explored_density) - separation_penalty
        pts = self.expl_map.unexplored_points()
        if not pts:
            return None
        sample_count = min(200, len(pts))
        pts_sampled = random.sample(pts, sample_count)
        best = None; best_score = -1e9

        robot_positions = [r.pos for r in self.robots]

        for p in pts_sampled:
            # ignore if inside obstacle
            if self.point_in_obstacle(p): continue

            d = dist(p, robot.pos)
            if d < 0.2: continue

            # separation penalty: penalize goals that are close to ANY other robot
            separation_penalty = 0.0
            for other_pos in robot_positions:
                if other_pos is robot.pos: continue
                d_to_other = dist(p, other_pos)
                if d_to_other < REPULSION_RADIUS:
                    separation_penalty += (REPULSION_RADIUS - d_to_other) * 2.0

            # estimate local explored density around p
            neighbors = 0; seen = 0
            nbr_r = int(round(0.5 / self.expl_map.res))
            ci = int(p[0] / self.expl_map.res); cj = int(p[1] / self.expl_map.res)
            for i in range(ci-nbr_r, ci+nbr_r+1):
                if i<0 or i>=self.expl_map.grid_w: continue
                for j in range(cj-nbr_r, cj+nbr_r+1):
                    if j<0 or j>=self.expl_map.grid_h: continue
                    neighbors += 1
                    if self.expl_map.cells[i][j] == 1: seen += 1
            density = (seen / neighbors) if neighbors > 0 else 0.0

            # base score: prefer distant unexplored and low local density
            score = d * (1.0 - density) - separation_penalty - 0.2 * abs(robot.theta - math.atan2(p[1]-robot.pos[1], p[0]-robot.pos[0]))

            if score > best_score:
                best_score = score; best = p
        return best

# ---------- Visualization (Pygame) ----------
def draw_world(screen, world):
    screen.fill(BG_COLOR)
    # draw explored cells as faint dots
    for i in range(world.expl_map.grid_w):
        for j in range(world.expl_map.grid_h):
            if world.expl_map.cells[i][j] == 1:
                x = (i + 0.5) * world.expl_map.res
                y = (j + 0.5) * world.expl_map.res
                sx, sy = world_to_screen((x,y))
                pygame.draw.circle(screen, (40,40,60), (sx, sy), max(1,int(0.06*SCALE)))
    # draw obstacles
    for o in world.obstacles: o.draw(screen)
    # draw robots and goals
    for r in world.robots:
        sx, sy = world_to_screen(r.pos)
        # robot body
        pygame.draw.circle(screen, ROBOT_COLOR, (sx, sy), max(3, int(ROBOT_RADIUS*SCALE)))
        # heading line
        hx = sx + int(math.cos(r.theta) * ROBOT_RADIUS * SCALE * 2.0)
        hy = sy - int(math.sin(r.theta) * ROBOT_RADIUS * SCALE * 2.0)
        pygame.draw.line(screen, (20,20,20), (sx,sy), (hx,hy), 2)
        # goal marker
        if r.goal is not None:
            gx, gy = r.goal
            gsx, gsy = world_to_screen((gx, gy))
            pygame.draw.circle(screen, ROBOT_GOAL_COLOR, (gsx, gsy), 4)
    # HUD
    font = pygame.font.SysFont('Arial', 16)
    txt = font.render(f'Robots: {len(world.robots)}  Explored: {world.expl_map.fraction_explored()*100:.1f}%', True, (220,220,220))
    screen.blit(txt, (8,8))
    pygame.display.flip()

# ---------- Main ----------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    world = World(WORLD_W, WORLD_H)

    # add some obstacles
    world.add_obstacle(Obstacle((2.0, 1.0, 1.2, 2.0)))
    world.add_obstacle(Obstacle((5.2, 3.0, 0.8, 2.6)))
    world.add_obstacle(Obstacle((7.0, 0.6, 1.2, 1.6)))

    # All robots start at center, slightly offset so they are not on exactly the same point
    center_x = WORLD_W / 2.0
    center_y = WORLD_H / 2.0
    for i in range(NUM_ROBOTS):
        theta = random.uniform(-math.pi, math.pi)
        # small offset to avoid perfect overlap; spread along x a bit
        offset_x = (i - (NUM_ROBOTS-1)/2.0) * 0.08
        x = center_x + offset_x
        y = center_y
        r = Robot(i, (x, y), theta)
        world.add_robot(r)
        # initial exploration ping at center
        world.expl_map.mark_circle((x, y), SENSOR_RADIUS * 0.6)

    running = True
    sim_time = 0.0
    while running:
        dt = clock.tick(FPS) / 1000.0
        sim_time += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # compute goals occasionally
        for r in world.robots:
            if sim_time - r.last_goal_compute >= GOAL_RECOMPUTE:
                r.last_goal_compute = sim_time
                r.goal = world.compute_goal_for_robot(r)
        # world step
        world.step(dt)
        draw_world(screen, world)
        # check termination: nearly fully explored
        if world.expl_map.fraction_explored() > 0.98:
            print('Exploration complete')
            running = False

    pygame.quit()
    print('Done.')

if __name__ == '__main__':
    main()