import math

import pandas as pd
import matplotlib.pyplot as plt

MAXIMUM_RANGE = 5000

class NavigationService:
    def __init__(self):
        self.obstacle_coordinates = pd.DataFrame(columns=["x", "y"])

    def update_obstacles(self, meas):
        # angle = meas["yaw"]
        # obstacle_x = meas["x"] + meas["front"] * math.cos(angle)
        # obstacle_y = meas["y"] + meas["front"] * math.sin(angle)
        # print(f"x: {obstacle_x}, y: {obstacle_y}")
        # return
    
        yaw = meas["yaw"]
        x_position = meas["x"]
        y_position = meas["y"]

        angled_ranges = [
            (meas['front'], yaw),
            (meas['back'], yaw + math.pi),
            (meas['left'], yaw + math.pi / 2),
            (meas['right'], yaw - math.pi / 2)
        ]

        new_coordinates = []

        for (range, angle) in angled_ranges:
            if range >= MAXIMUM_RANGE or range < 0:
                continue

            obstacle_x = x_position + range * math.cos(angle)
            obstacle_y = y_position + range * math.sin(angle)
            new_coordinates.append({'x': obstacle_x, 'y': obstacle_y})
            
        if new_coordinates:
            new_df = pd.DataFrame(new_coordinates)
            self.obstacle_coordinates = pd.concat([self.obstacle_coordinates, new_df], ignore_index=True)
    
    def save_plot(self, filename="obstacles.png"):
        """Plot all collected obstacles and save to file"""
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot obstacles as scatter points
        if len(self.obstacle_coordinates) > 0:
            ax.scatter(
                self.obstacle_coordinates['x'],
                self.obstacle_coordinates['y'],
                c='red',
                s=50,
                alpha=0.6,
                label='Obstacles',
                marker='o'
            )
        
        # Set equal aspect ratio (so circles look circular)
        ax.set_aspect('equal', 'box')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set labels and title
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Obstacle Map', fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(loc='best')
        
        # Save plot to file
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def step(self):
        return {
            "radio://0/80/2M/E7E7E7E70A": {'x': 0.0, 'y': 0.0, 'z': 0.30, 'yaw': 0.0},
            "radio://0/80/2M/E7E7E7E701": {'x': 0.0, 'y': 0.0, 'z': 0.30, 'yaw': 0.0},
        }
