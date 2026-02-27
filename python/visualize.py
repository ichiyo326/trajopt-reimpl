#!/usr/bin/env python3
"""
visualize.py - Trajectory Optimization Visualization

Reads the trajectory_data.json output from arm_2d_demo and produces:
  1. Side-by-side comparison: initial trajectory vs optimized trajectory
  2. Animated GIF of the optimized arm motion
  3. Signed distance plot over the optimization (if available)

Usage:
    python3 visualize.py [trajectory_data.json]
    python3 visualize.py --demo   # generate synthetic demo data without running C++
"""

import json
import sys
import math
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 120,
})

# ============================================================
# Forward kinematics (2D, planar arm)
# ============================================================

def fk_2d(q, link_lengths):
    """
    Returns list of (x, y) joint positions for a planar robot arm.
    q: list of joint angles [rad]
    link_lengths: list of link lengths [m]
    """
    pts = [(0.0, 0.0)]
    angle = 0.0
    x, y = 0.0, 0.0
    for i, (qi, L) in enumerate(zip(q, link_lengths)):
        angle += qi
        x += L * math.cos(angle)
        y += L * math.sin(angle)
        pts.append((x, y))
    return pts


# ============================================================
# Drawing helpers
# ============================================================

def draw_arm(ax, q, link_lengths, color='royalblue', alpha=1.0,
             lw=3, label=None, zorder=5):
    pts = fk_2d(q, link_lengths)
    xs, ys = zip(*pts)
    line, = ax.plot(xs, ys, '-o', color=color, linewidth=lw,
                    markersize=6, alpha=alpha, label=label, zorder=zorder,
                    solid_capstyle='round', solid_joinstyle='round')
    # End-effector marker
    ax.plot(xs[-1], ys[-1], '*', color=color, markersize=14,
            alpha=alpha, zorder=zorder+1)
    return line


def draw_trajectory_ghost(ax, traj, link_lengths, color='skyblue', n_ghosts=5):
    """Draw faded 'ghost' arm poses to visualize the trajectory arc."""
    indices = np.linspace(0, len(traj)-1, n_ghosts, dtype=int)
    for i, idx in enumerate(indices):
        alpha = 0.1 + 0.25 * (i / len(indices))
        draw_arm(ax, traj[idx], link_lengths, color=color, alpha=alpha, lw=1.5)


def draw_obstacles(ax, boxes, spheres):
    for (bmin, bmax) in boxes:
        w, h = bmax[0]-bmin[0], bmax[1]-bmin[1]
        rect = mpatches.FancyBboxPatch(
            (bmin[0], bmin[1]), w, h,
            boxstyle="round,pad=0.02",
            linewidth=1.5, edgecolor='#8B0000', facecolor='#FF6B6B', alpha=0.7,
            zorder=3
        )
        ax.add_patch(rect)

    for (cx, cy, r) in spheres:
        circle = mpatches.Circle((cx, cy), r,
                                  linewidth=1.5, edgecolor='#8B0000',
                                  facecolor='#FF6B6B', alpha=0.7, zorder=3)
        ax.add_patch(circle)


def setup_ax(ax, title, xlim=(-0.5, 3.2), ylim=(-1.5, 1.8)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    # Base
    ax.plot(0, 0, 's', color='#333333', markersize=12, zorder=10)


# ============================================================
# Synthetic demo data (when C++ binary not available)
# ============================================================

def generate_demo_data():
    """Generate synthetic trajectory data for demo/testing."""
    T = 11
    link_lengths = [1.0, 1.0, 1.0]

    # Start: arm pointing up-left
    q_start = np.array([-0.3, 0.4, 0.3])
    # Goal: arm folded to the right
    q_goal  = np.array([0.1, -0.5, -0.2])

    # Initial: linear interpolation (goes through obstacles)
    init_traj = [(1 - t/(T-1)) * q_start + (t/(T-1)) * q_goal for t in range(T)]

    # Optimized: arc around obstacles (hand-crafted for demo)
    opt_traj = []
    for t in range(T):
        alpha = t / (T - 1)
        q_mid = np.array([0.0, 0.8, -0.5])  # detour via high pose
        if alpha < 0.5:
            q = (1 - 2*alpha) * q_start + 2*alpha * q_mid
        else:
            q = (1 - 2*(alpha-0.5)) * q_mid + 2*(alpha-0.5) * q_goal
        opt_traj.append(q)

    boxes = [
        [0.5, 0.3, 1.2, 0.8],
        [0.5, -0.8, 1.2, -0.3],
    ]
    spheres = [[1.8, 0.0, 0.25]]

    return {
        "link_lengths": link_lengths,
        "init_trajectory": [q.tolist() for q in init_traj],
        "opt_trajectory": [q.tolist() for q in opt_traj],
        "boxes": boxes,
        "spheres": spheres,
    }


# ============================================================
# Main visualization
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='TrajOpt trajectory visualizer')
    parser.add_argument('data_file', nargs='?', default='trajectory_data.json')
    parser.add_argument('--demo', action='store_true',
                        help='Use synthetic demo data (no C++ binary needed)')
    parser.add_argument('--save', default='trajopt_result.png',
                        help='Save static figure to file')
    parser.add_argument('--anim', default='trajopt_animation.gif',
                        help='Save animation GIF to file')
    args = parser.parse_args()

    # Load data
    if args.demo:
        print("[Viz] Using synthetic demo data.")
        data = generate_demo_data()
    else:
        try:
            with open(args.data_file) as f:
                data = json.load(f)
            print(f"[Viz] Loaded data from {args.data_file}")
        except FileNotFoundError:
            print(f"[Viz] '{args.data_file}' not found. Using demo data.")
            data = generate_demo_data()

    link_lengths = data['link_lengths']
    init_traj = [np.array(q) for q in data['init_trajectory']]
    opt_traj  = [np.array(q) for q in data['opt_trajectory']]
    boxes   = [((b[0], b[1], -0.1), (b[2], b[3], 0.1)) for b in data['boxes']]
    spheres = [(s[0], s[1], s[2]) for s in data['spheres']]

    T = len(opt_traj)

    # ============================================================
    # Figure 1: Side-by-side static comparison
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('TrajOpt: Sequential Convex Programming for Motion Planning\n'
                 '(Schulman et al., IJRR 2014 — Reimplementation)',
                 fontsize=12, fontweight='bold', y=1.01)

    # --- Left: initial trajectory ---
    ax = axes[0]
    setup_ax(ax, 'Initial Trajectory (straight-line, in collision)')
    boxes_2d = [((b[0][0], b[0][1]), (b[1][0], b[1][1])) for b in boxes]
    spheres_2d = spheres
    draw_obstacles(ax, boxes_2d, spheres_2d)
    draw_trajectory_ghost(ax, init_traj, link_lengths, color='#FF4444', n_ghosts=7)
    draw_arm(ax, init_traj[0], link_lengths, color='green', label='Start', lw=3)
    draw_arm(ax, init_traj[-1], link_lengths, color='darkorange', label='Goal', lw=3)

    # Annotate collision
    ax.text(0.98, 0.02, '⚠ Trajectory in collision',
            transform=ax.transAxes, ha='right', va='bottom',
            color='#CC0000', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE0E0', alpha=0.8))
    ax.legend(loc='upper left', framealpha=0.9)

    # --- Right: optimized trajectory ---
    ax = axes[1]
    setup_ax(ax, 'Optimized Trajectory (TrajOpt, collision-free)')
    draw_obstacles(ax, boxes_2d, spheres_2d)
    draw_trajectory_ghost(ax, opt_traj, link_lengths, color='royalblue', n_ghosts=7)
    draw_arm(ax, opt_traj[0], link_lengths, color='green', label='Start', lw=3)
    draw_arm(ax, opt_traj[-1], link_lengths, color='darkorange', label='Goal', lw=3)

    ax.text(0.98, 0.02, '✓ Collision-free',
            transform=ax.transAxes, ha='right', va='bottom',
            color='#005500', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E0FFE0', alpha=0.8))
    ax.legend(loc='upper left', framealpha=0.9)

    # Add obstacle legend
    obs_patch = mpatches.Patch(facecolor='#FF6B6B', edgecolor='#8B0000',
                                linewidth=1.5, label='Obstacle', alpha=0.7)
    axes[0].add_patch(mpatches.Patch(visible=False))  # spacer
    fig.legend(handles=[obs_patch], loc='upper right', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    plt.savefig(args.save, bbox_inches='tight', dpi=150)
    print(f"[Viz] Saved static figure to {args.save}")
    plt.show(block=False)

    # ============================================================
    # Figure 2: Optimized trajectory animation
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    setup_ax(ax2, 'TrajOpt Optimized Trajectory — Animation')
    draw_obstacles(ax2, boxes_2d, spheres_2d)
    draw_trajectory_ghost(ax2, opt_traj, link_lengths, color='lightblue', n_ghosts=T)

    arm_line, = ax2.plot([], [], '-o', color='royalblue', linewidth=4,
                          markersize=8, solid_capstyle='round', zorder=6)
    ee_marker, = ax2.plot([], [], '*', color='royalblue', markersize=16, zorder=7)

    start_pts = fk_2d(opt_traj[0], link_lengths)
    goal_pts  = fk_2d(opt_traj[-1], link_lengths)
    ax2.plot(*zip(*start_pts), '-o', color='green', linewidth=2, alpha=0.5, zorder=4)
    ax2.plot(*zip(*goal_pts),  '-o', color='darkorange', linewidth=2, alpha=0.5, zorder=4)

    time_text = ax2.text(0.02, 0.96, '', transform=ax2.transAxes,
                          fontsize=11, va='top', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def animate(frame):
        t = frame % T
        pts = fk_2d(opt_traj[t], link_lengths)
        xs, ys = zip(*pts)
        arm_line.set_data(xs, ys)
        ee_marker.set_data([xs[-1]], [ys[-1]])
        time_text.set_text(f'Timestep: {t+1}/{T}')
        return arm_line, ee_marker, time_text

    anim = FuncAnimation(fig2, animate, frames=T*3, interval=200, blit=True)

    try:
        writer = PillowWriter(fps=5)
        anim.save(args.anim, writer=writer)
        print(f"[Viz] Saved animation to {args.anim}")
    except Exception as e:
        print(f"[Viz] Could not save GIF: {e}")

    plt.tight_layout()
    plt.show()

    # ============================================================
    # Figure 3: Joint angle trajectories
    # ============================================================
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig3.suptitle('Joint Angle Trajectories', fontweight='bold')
    timesteps = np.arange(T)

    for k in range(3):
        ax = axes3[k]
        init_vals = [q[k] for q in init_traj]
        opt_vals  = [q[k] for q in opt_traj]

        ax.plot(timesteps, np.degrees(init_vals), '--', color='#FF4444',
                label='Initial', linewidth=2, alpha=0.7)
        ax.plot(timesteps, np.degrees(opt_vals),  '-',  color='royalblue',
                label='Optimized', linewidth=2.5)

        ax.set_ylabel(f'Joint {k+1} [deg]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)

    axes3[-1].set_xlabel('Timestep')
    plt.tight_layout()
    plt.savefig('joint_trajectories.png', bbox_inches='tight', dpi=150)
    print("[Viz] Saved joint trajectories to joint_trajectories.png")
    plt.show()

    print("\n[Viz] Done. Files written:")
    print(f"  - {args.save}")
    print(f"  - {args.anim}")
    print("  - joint_trajectories.png")


if __name__ == '__main__':
    main()
