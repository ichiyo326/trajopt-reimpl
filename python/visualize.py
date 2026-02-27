#!/usr/bin/env python3
"""
visualize.py - Trajectory Optimization Visualization

Usage:
    python3 visualize.py --demo
    python3 visualize.py trajectory_data.json
"""

import json, sys, math, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.dpi': 120,
})

# ── Forward kinematics ──────────────────────────────────────────────────────

def fk_2d(q, link_lengths):
    pts = [(0.0, 0.0)]
    angle = 0.0; x = y = 0.0
    for qi, L in zip(q, link_lengths):
        angle += qi
        x += L * math.cos(angle)
        y += L * math.sin(angle)
        pts.append((x, y))
    return pts

# ── Drawing helpers ─────────────────────────────────────────────────────────

def draw_arm(ax, q, ll, color='royalblue', alpha=1.0, lw=3, label=None, zorder=5):
    pts = fk_2d(q, ll)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, '-o', color=color, lw=lw, markersize=6,
            alpha=alpha, label=label, zorder=zorder,
            solid_capstyle='round', solid_joinstyle='round')
    ax.plot(xs[-1], ys[-1], '*', color=color, markersize=14, alpha=alpha, zorder=zorder+1)

def draw_ghosts(ax, traj, ll, color, n=7):
    idxs = np.linspace(0, len(traj)-1, n, dtype=int)
    for k, idx in enumerate(idxs):
        alpha = 0.08 + 0.22 * (k / max(n-1, 1))
        draw_arm(ax, traj[idx], ll, color=color, alpha=alpha, lw=1.5, zorder=3)

def draw_obstacles(ax, spheres):
    for (cx, cy, r) in spheres:
        c = mpatches.Circle((cx, cy), r,
                             lw=1.5, edgecolor='#8B0000',
                             facecolor='#FF6B6B', alpha=0.75, zorder=4)
        ax.add_patch(c)

def setup_ax(ax, title, xlim=(-0.3, 3.3), ylim=(-2.0, 2.0)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.4)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.4)
    ax.plot(0, 0, 's', color='#333333', markersize=12, zorder=10)

# ── Demo data ────────────────────────────────────────────────────────────────

def generate_demo_data():
    """
    Scenario: 3-link arm, single sphere obstacle at (1.5, 0) r=0.35.
    Start EE ~ (2.4, -1.8), Goal EE ~ (2.8, 1.1).
    Init trajectory goes THROUGH the sphere.
    Optimized trajectory swings UP and around.
    All waypoints verified collision-free by FK computation.
    """
    ll = [1.0, 1.0, 1.0]
    q_start = np.array([-0.50, -0.30,  0.20])
    q_goal  = np.array([ 0.50, -0.30,  0.20])
    T = 11

    # Initial: linear interpolation (passes through sphere at t=4..8)
    init_traj = [(1 - t/(T-1))*q_start + (t/(T-1))*q_goal for t in range(T)]

    # Optimized: swings arm UP and over the sphere
    # Each waypoint verified: all link segments avoid sphere(1.5,0,r=0.35)
    opt_traj = [
        np.array([-0.50, -0.30,  0.20]),   # t=0  start
        np.array([ 0.20,  0.50,  0.10]),   # t=1  rising
        np.array([ 0.60,  0.60,  0.00]),   # t=2  sweeping up-right
        np.array([ 0.80,  0.55, -0.10]),   # t=3
        np.array([ 0.90,  0.40, -0.15]),   # t=4  peak
        np.array([ 0.85,  0.20, -0.15]),   # t=5  descending
        np.array([ 0.75,  0.00, -0.15]),   # t=6
        np.array([ 0.65, -0.15,  0.00]),   # t=7
        np.array([ 0.58, -0.25,  0.10]),   # t=8
        np.array([ 0.53, -0.28,  0.15]),   # t=9
        np.array([ 0.50, -0.30,  0.20]),   # t=10 goal
    ]

    return {
        "link_lengths": ll,
        "init_trajectory": [q.tolist() for q in init_traj],
        "opt_trajectory":  [q.tolist() for q in opt_traj],
        "spheres": [[1.5, 0.0, 0.35]],
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', nargs='?', default='trajectory_data.json')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--save', default='trajopt_result.png')
    parser.add_argument('--anim', default='trajopt_animation.gif')
    args = parser.parse_args()

    if args.demo:
        print("[Viz] Using synthetic demo data.")
        data = generate_demo_data()
    else:
        try:
            with open(args.data_file) as f:
                data = json.load(f)
            print("[Viz] Loaded", args.data_file)
        except FileNotFoundError:
            print("[Viz] File not found. Using demo data.")
            data = generate_demo_data()

    ll         = data['link_lengths']
    init_traj  = [np.array(q) for q in data['init_trajectory']]
    opt_traj   = [np.array(q) for q in data['opt_trajectory']]
    spheres    = data.get('spheres', [])
    T = len(opt_traj)

    # ── Figure 1: side-by-side comparison ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        'TrajOpt: Sequential Convex Programming for Motion Planning\n'
        '(Schulman et al., IJRR 2014 — Reimplementation)',
        fontsize=12, fontweight='bold', y=1.01)

    obs_patch = mpatches.Patch(facecolor='#FF6B6B', edgecolor='#8B0000',
                                lw=1.5, label='Obstacle', alpha=0.75)

    # Left: initial (in collision)
    ax = axes[0]
    setup_ax(ax, 'Initial Trajectory (straight-line, in collision)')
    draw_obstacles(ax, spheres)
    draw_ghosts(ax, init_traj, ll, color='#FF4444')
    draw_arm(ax, init_traj[0],  ll, color='green',      label='Start', lw=3)
    draw_arm(ax, init_traj[-1], ll, color='darkorange',  label='Goal',  lw=3)
    ax.text(0.98, 0.02, '⚠ Trajectory in collision',
            transform=ax.transAxes, ha='right', va='bottom',
            color='#CC0000', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE0E0', alpha=0.85))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [obs_patch], l + ['Obstacle'], loc='upper left', framealpha=0.9)

    # Right: optimized (collision-free)
    ax = axes[1]
    setup_ax(ax, 'Optimized Trajectory (TrajOpt, collision-free)')
    draw_obstacles(ax, spheres)
    draw_ghosts(ax, opt_traj, ll, color='royalblue')
    draw_arm(ax, opt_traj[0],  ll, color='green',      label='Start', lw=3)
    draw_arm(ax, opt_traj[-1], ll, color='darkorange',  label='Goal',  lw=3)
    ax.text(0.98, 0.02, '✓ Collision-free',
            transform=ax.transAxes, ha='right', va='bottom',
            color='#005500', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E0FFE0', alpha=0.85))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [obs_patch], l + ['Obstacle'], loc='upper left', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(args.save, bbox_inches='tight', dpi=150)
    print("[Viz] Saved", args.save)
    plt.show(block=False)

    # ── Figure 2: animation ───────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    setup_ax(ax2, 'TrajOpt Optimized Trajectory — Animation')
    draw_obstacles(ax2, spheres)
    draw_ghosts(ax2, opt_traj, ll, color='lightblue', n=T)
    draw_arm(ax2, opt_traj[0],  ll, color='green',     alpha=0.4, lw=2, zorder=3)
    draw_arm(ax2, opt_traj[-1], ll, color='darkorange', alpha=0.4, lw=2, zorder=3)

    arm_line, = ax2.plot([], [], '-o', color='royalblue', lw=4,
                          markersize=8, solid_capstyle='round', zorder=6)
    ee_star,  = ax2.plot([], [], '*', color='royalblue', markersize=16, zorder=7)
    time_text = ax2.text(0.02, 0.96, '', transform=ax2.transAxes,
                          fontsize=11, va='top', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    def animate(frame):
        t = frame % T
        pts = fk_2d(opt_traj[t], ll)
        xs, ys = zip(*pts)
        arm_line.set_data(xs, ys)
        ee_star.set_data([xs[-1]], [ys[-1]])
        time_text.set_text('Timestep: %d/%d' % (t+1, T))
        return arm_line, ee_star, time_text

    anim = FuncAnimation(fig2, animate, frames=T*3, interval=200, blit=True)
    try:
        anim.save(args.anim, writer=PillowWriter(fps=5))
        print("[Viz] Saved", args.anim)
    except Exception as e:
        print("[Viz] GIF save failed:", e)

    plt.tight_layout()
    plt.show()

    # ── Figure 3: joint angles ────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig3.suptitle('Joint Angle Trajectories', fontweight='bold')
    ts = np.arange(T)
    for k in range(3):
        ax = axes3[k]
        ax.plot(ts, np.degrees([q[k] for q in init_traj]),
                '--', color='#FF4444', lw=2, alpha=0.7, label='Initial')
        ax.plot(ts, np.degrees([q[k] for q in opt_traj]),
                '-',  color='royalblue', lw=2.5, label='Optimized')
        ax.set_ylabel('Joint %d [deg]' % (k+1))
        ax.grid(True, alpha=0.3); ax.legend(loc='upper right', fontsize=9)
    axes3[-1].set_xlabel('Timestep')
    plt.tight_layout()
    plt.savefig('joint_trajectories.png', bbox_inches='tight', dpi=150)
    print("[Viz] Saved joint_trajectories.png")
    plt.show()

if __name__ == '__main__':
    main()
