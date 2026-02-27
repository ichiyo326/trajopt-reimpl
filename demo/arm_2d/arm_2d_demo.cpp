/**
 * arm_2d_demo.cpp - 2D Planar Robot Arm Trajectory Optimization Demo
 *
 * Demonstrates TrajOpt on a 3-link planar robot arm avoiding box and
 * sphere obstacles. Outputs trajectory data for Python visualization.
 *
 * Scene:
 *   - 3-link planar arm (all links in XY plane, Z=0)
 *   - 2 box obstacles forming a narrow corridor
 *   - 1 sphere obstacle in the middle of the workspace
 *
 * The straight-line trajectory initialization passes through the obstacles.
 * TrajOpt finds a collision-free path by sequential convex optimization.
 *
 * Output: trajectory_data.json (read by visualize.py)
 */

#include "../../include/scp/trajopt.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace trajopt;

// ---- Simple JSON output (no external JSON library needed) ----

void write_vector(std::ofstream& f, const Eigen::VectorXd& v) {
    f << "[";
    for (int i = 0; i < v.size(); ++i) {
        f << std::fixed << std::setprecision(6) << v(i);
        if (i < v.size()-1) f << ", ";
    }
    f << "]";
}

void write_trajectory_json(
    const std::string& path,
    const std::vector<Eigen::VectorXd>& init_traj,
    const std::vector<Eigen::VectorXd>& opt_traj,
    const RobotArm& arm,
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes,
    const std::vector<std::pair<Eigen::Vector3d, double>>& spheres)
{
    std::ofstream f(path);
    f << "{\n";

    // Arm link lengths
    f << "  \"link_lengths\": [";
    for (int k = 0; k < arm.n_joints(); ++k) {
        f << arm.links()[k].dh.a;
        if (k < arm.n_joints()-1) f << ", ";
    }
    f << "],\n";

    // Initial trajectory
    f << "  \"init_trajectory\": [\n";
    for (size_t t = 0; t < init_traj.size(); ++t) {
        f << "    "; write_vector(f, init_traj[t]);
        if (t < init_traj.size()-1) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // Optimized trajectory
    f << "  \"opt_trajectory\": [\n";
    for (size_t t = 0; t < opt_traj.size(); ++t) {
        f << "    "; write_vector(f, opt_traj[t]);
        if (t < opt_traj.size()-1) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // Box obstacles: [min_x, min_y, max_x, max_y]
    f << "  \"boxes\": [\n";
    for (size_t i = 0; i < boxes.size(); ++i) {
        auto& [bmin, bmax] = boxes[i];
        f << "    [" << bmin.x() << ", " << bmin.y() << ", "
                     << bmax.x() << ", " << bmax.y() << "]";
        if (i < boxes.size()-1) f << ",";
        f << "\n";
    }
    f << "  ],\n";

    // Sphere obstacles: [cx, cy, r]
    f << "  \"spheres\": [\n";
    for (size_t i = 0; i < spheres.size(); ++i) {
        auto& [c, r] = spheres[i];
        f << "    [" << c.x() << ", " << c.y() << ", " << r << "]";
        if (i < spheres.size()-1) f << ",";
        f << "\n";
    }
    f << "  ]\n";

    f << "}\n";
    std::cout << "[Demo] Wrote trajectory data to " << path << std::endl;
}

// ---- Forward kinematics for 2D arm (XY plane) ----
// Returns list of (x, y) joint positions for visualization.
std::vector<Eigen::Vector2d> fk_2d(const Eigen::VectorXd& q,
                                     const std::vector<double>& link_lengths) {
    std::vector<Eigen::Vector2d> pts;
    pts.push_back(Eigen::Vector2d::Zero());  // base
    double angle = 0.0;
    Eigen::Vector2d pos = Eigen::Vector2d::Zero();
    for (size_t i = 0; i < q.size(); ++i) {
        angle += q(i);
        pos += link_lengths[i] * Eigen::Vector2d(std::cos(angle), std::sin(angle));
        pts.push_back(pos);
    }
    return pts;
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  TrajOpt 2D Arm Demo\n";
    std::cout << "  3-link planar arm, obstacle avoidance\n";
    std::cout << "========================================\n\n";

    // ---- Create robot ----
    constexpr double L = 1.0;  // link length
    auto arm = RobotArm::make_planar_3dof(L);
    std::cout << "[Setup] 3-DOF planar arm, link length = " << L << " m\n";

    // ---- Create collision checker and add obstacles ----
    CollisionChecker checker(arm);

    // Two box obstacles forming a corridor
    // The straight-line trajectory passes through these
    Eigen::Vector3d b1_min(0.5, 0.3, -0.1), b1_max(1.2, 0.8, 0.1);
    Eigen::Vector3d b2_min(0.5, -0.8, -0.1), b2_max(1.2, -0.3, 0.1);
    checker.add_box_obstacle("box_upper", b1_min, b1_max);
    checker.add_box_obstacle("box_lower", b2_min, b2_max);

    // Sphere obstacle in front
    Eigen::Vector3d sphere_center(1.8, 0.0, 0.0);
    double sphere_r = 0.25;
    checker.add_sphere_obstacle("sphere", sphere_center, sphere_r);

    std::cout << "[Setup] 2 box obstacles, 1 sphere obstacle\n";

    // ---- Define start and goal ----
    // Start: arm pointing upward-left
    Eigen::VectorXd q_start(3);
    q_start << -0.3, 0.4, 0.3;  // joint angles [rad]

    // Goal: arm pointing toward the right (through the obstacle corridor)
    Eigen::VectorXd q_goal(3);
    q_goal << 0.1, -0.5, -0.2;

    std::cout << "\n[Setup] Start: [" << q_start.transpose() << "] rad\n";
    std::cout << "[Setup] Goal:  [" << q_goal.transpose() << "] rad\n";

    // ---- Check initial trajectory for collisions ----
    auto fk_start = arm.end_effector_pose(q_start);
    auto fk_goal  = arm.end_effector_pose(q_goal);
    std::cout << "\n[Setup] Start EE pos: ["
              << fk_start(0,3) << ", " << fk_start(1,3) << "] m\n";
    std::cout << "[Setup] Goal  EE pos: ["
              << fk_goal(0,3)  << ", " << fk_goal(1,3)  << "] m\n";

    // Save initial trajectory for visualization
    int T = 11;
    std::vector<Eigen::VectorXd> init_traj(T);
    for (int t = 0; t < T; ++t) {
        double alpha = static_cast<double>(t) / (T-1);
        init_traj[t] = (1.0 - alpha) * q_start + alpha * q_goal;
    }

    // ---- Set up TrajOpt problem ----
    TrajOptParams params;
    params.n_timesteps = T;
    params.d_safe = 0.05;
    params.d_check = 0.5;
    params.use_swept_volume = true;
    params.scp_params.verbose = true;
    params.scp_params.max_penalty_iter = 6;
    params.scp_params.max_convexify_iter = 15;
    params.scp_params.mu_0 = 20.0;
    params.scp_params.s_0  = 0.3;

    TrajOptProblem prob(arm, checker, params);
    prob.set_start(q_start);
    prob.set_goal(q_goal);

    std::cout << "\n[TrajOpt] Starting optimization...\n";
    std::cout << "  T = " << T << " timesteps\n";
    std::cout << "  d_safe = " << params.d_safe << " m\n";
    std::cout << "  Continuous-time collision: " << (params.use_swept_volume ? "yes" : "no") << "\n\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    auto result = prob.solve(init_traj);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // ---- Print results ----
    std::cout << "\n========================================\n";
    std::cout << "  Results\n";
    std::cout << "========================================\n";
    std::cout << "Status: " << result.message << "\n";
    std::cout << "Collision-free: " << (result.collision_free ? "YES" : "NO") << "\n";
    std::cout << "Path length: " << std::fixed << std::setprecision(4)
              << result.path_length << " rad\n";
    std::cout << "Max collision penetration: "
              << result.max_collision_penetration << " m\n";
    std::cout << "Computation time: " << elapsed << " s\n";

    std::cout << "\nOptimized trajectory:\n";
    for (int t = 0; t < T; ++t) {
        std::cout << "  t=" << t << ": [" << result.trajectory[t].transpose() << "]\n";
    }

    // ---- Write output for visualization ----
    std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d>> boxes = {
        {b1_min, b1_max}, {b2_min, b2_max}
    };
    std::vector<std::pair<Eigen::Vector3d,double>> spheres = {
        {sphere_center, sphere_r}
    };
    write_trajectory_json("trajectory_data.json",
                          init_traj, result.trajectory,
                          arm, boxes, spheres);

    std::cout << "\nRun 'python3 ../python/visualize.py' to see the animation.\n";
    return 0;
}
