#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize Crazyflie pointcloud + safe zone.

What it shows:
- blue dots: filtered pointcloud
- black outline: detected room boundary
- green outline/fill: safe zone where other drones are allowed
- red circles: obstacle keep-out zones
- orange star: start position

Usage examples:
  python visualize_safezone.py --csv filtered_pointcloud_20260609_131814.csv --safezone safe_zone_20260609_131814.json
  python visualize_safezone.py --safezone safe_zone_20260609_131814.json
  python visualize_safezone.py --safezone safe_zone_20260609_131814.json --save safezone_view.png
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon, Circle


def close_polygon(points):
    """Return x/y arrays with the first point repeated at the end."""
    if not points:
        return [], []
    pts = list(points)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return xs, ys


def resolve_csv_path(csv_arg, safezone_path, safezone_data):
    """
    Resolve pointcloud CSV path.

    Priority:
    1. --csv argument if given
    2. path written inside safe_zone json
    3. same folder as json + basename of json csv path
    """
    if csv_arg:
        p = Path(csv_arg)
        if p.exists():
            return p

        # Try same folder as safezone file
        candidate = safezone_path.parent / p.name
        if candidate.exists():
            return candidate

        raise FileNotFoundError(f"CSV file not found: {csv_arg}")

    csv_from_json = safezone_data.get("filtered_pointcloud_csv")
    if not csv_from_json:
        raise FileNotFoundError("No --csv given and safezone JSON has no filtered_pointcloud_csv field.")

    p = Path(csv_from_json)
    if p.exists():
        return p

    # Windows paths in JSON may contain backslashes. Use only basename.
    basename = Path(str(csv_from_json).replace("\\", "/")).name
    candidate = safezone_path.parent / basename
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not find pointcloud CSV. Try passing it manually with --csv.\n"
        f"JSON points to: {csv_from_json}\n"
        f"Tried: {candidate}"
    )


def plot_safezone(csv_path, safezone_path, save_path=None, show_3d=False):
    safezone_path = Path(safezone_path)
    with open(safezone_path, "r", encoding="utf-8") as f:
        safezone = json.load(f)

    csv_path = resolve_csv_path(csv_path, safezone_path, safezone)
    df = pd.read_csv(csv_path)

    required_cols = {"x", "y", "z"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns x,y,z. Missing: {missing}")

    room_boundary = safezone.get("room_boundary_polygon", [])
    safe_poly = safezone.get("safe_zone_polygon", [])
    keepouts = safezone.get("obstacle_keepouts", [])
    start = safezone.get("start_position_xy", None)

    wall_margin = safezone.get("rules_for_other_drones", {}).get("wall_margin_m", None)
    obstacle_margin = safezone.get("rules_for_other_drones", {}).get("obstacle_margin_m", None)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Point cloud, top-down
    ax.scatter(df["x"], df["y"], s=4, alpha=0.45, label="filtered pointcloud")

    # Room boundary
    if room_boundary:
        rx, ry = close_polygon(room_boundary)
        ax.plot(rx, ry, linewidth=2, linestyle="--", label="room boundary")

    # Safe zone polygon
    if safe_poly:
        patch = Polygon(
            safe_poly,
            closed=True,
            alpha=0.18,
            linewidth=2,
            edgecolor="green",
            facecolor="green",
            label="SAFE ZONE"
        )
        ax.add_patch(patch)

        sx, sy = close_polygon(safe_poly)
        ax.plot(sx, sy, linewidth=2, color="green")

    # Obstacle keepout circles
    for ko in keepouts:
        center = ko.get("center", None)
        radius = ko.get("radius", None)
        ko_id = ko.get("id", "?")

        if center is None or radius is None:
            continue

        circle = Circle(
            (center[0], center[1]),
            radius,
            alpha=0.22,
            edgecolor="red",
            facecolor="red",
            linewidth=2,
            label="obstacle keepout" if ko_id == 1 else None
        )
        ax.add_patch(circle)

        ax.scatter([center[0]], [center[1]], s=80, marker="x", color="red")
        ax.text(center[0], center[1], f"  obstacle {ko_id}", color="red", fontsize=10)

    # Start position
    if start is not None:
        ax.scatter([start[0]], [start[1]], s=180, marker="*", color="orange", edgecolor="black", label="start")
        ax.text(start[0], start[1], "  start", color="orange", fontsize=10)

    title = "Crazyflie pointcloud + safe zone"
    subtitle_parts = []
    if wall_margin is not None:
        subtitle_parts.append(f"wall margin={wall_margin} m")
    if obstacle_margin is not None:
        subtitle_parts.append(f"obstacle margin={obstacle_margin} m")
    if keepouts:
        subtitle_parts.append(f"obstacles={len(keepouts)}")
    else:
        subtitle_parts.append("obstacles=0")

    ax.set_title(title + "\n" + ", ".join(subtitle_parts))
    ax.set_xlabel("x position [m]")
    ax.set_ylabel("y position [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Add a little padding around all geometry.
    all_x = list(df["x"])
    all_y = list(df["y"])
    for poly in [room_boundary, safe_poly]:
        for p in poly:
            all_x.append(p[0])
            all_y.append(p[1])
    for ko in keepouts:
        c = ko.get("center")
        r = ko.get("radius", 0)
        if c:
            all_x += [c[0] - r, c[0] + r]
            all_y += [c[1] - r, c[1] + r]

    if all_x and all_y:
        pad = 0.35
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved 2D safe-zone image to: {save_path}")

    if show_3d:
        fig3 = plt.figure(figsize=(10, 8))
        ax3 = fig3.add_subplot(111, projection="3d")
        ax3.scatter(df["x"], df["y"], df["z"], s=3, alpha=0.35)

        # Draw safe zone at floor level z=0
        if safe_poly:
            sx, sy = close_polygon(safe_poly)
            sz = [0.0 for _ in sx]
            ax3.plot(sx, sy, sz, linewidth=2, color="green")

        if room_boundary:
            rx, ry = close_polygon(room_boundary)
            rz = [0.0 for _ in rx]
            ax3.plot(rx, ry, rz, linewidth=2, linestyle="--")

        for ko in keepouts:
            c = ko.get("center")
            r = ko.get("radius")
            if c is None or r is None:
                continue
            angles = [2 * math.pi * i / 100 for i in range(101)]
            xs = [c[0] + r * math.cos(a) for a in angles]
            ys = [c[1] + r * math.sin(a) for a in angles]
            zs = [0.0 for _ in angles]
            ax3.plot(xs, ys, zs, color="red", linewidth=2)

        ax3.set_title("3D pointcloud with safe-zone overlay at z=0")
        ax3.set_xlabel("x [m]")
        ax3.set_ylabel("y [m]")
        ax3.set_zlabel("z [m]")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Filtered pointcloud CSV file with columns x,y,z")
    parser.add_argument("--safezone", required=True, help="Safe-zone JSON file")
    parser.add_argument("--save", default=None, help="Optional output image path, for example safezone_view.png")
    parser.add_argument("--show-3d", action="store_true", help="Also show a simple 3D view")
    args = parser.parse_args()

    plot_safezone(
        csv_path=args.csv,
        safezone_path=args.safezone,
        save_path=args.save,
        show_3d=args.show_3d
    )


if __name__ == "__main__":
    main()
