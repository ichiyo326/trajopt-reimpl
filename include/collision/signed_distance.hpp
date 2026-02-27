#pragma once
/**
 * signed_distance.hpp - Collision Cost with Jacobian Linearization
 *
 * Implements the core collision penalty used in TrajOpt:
 *
 *   collision_cost = sum_{(link i, obstacle j)} |d_safe - sd(A_i, O_j)|_+
 *
 * where |x|_+ = max(x, 0) is the hinge loss.
 *
 * The signed distance is linearized around the current configuration x0:
 *
 *   sd_AB(x) ≈ sd_AB(x0) + n_hat^T * J_pA(x0) * (x - x0)    [Eq. 18-19]
 *
 * For continuous-time safety, the swept volume between timesteps is used:
 *
 *   sd(convhull(A(t), A(t+1)), O) > d_safe + d_arc             [Eq. 27]
 *
 * The linearized collision terms can be directly inserted into the QP
 * as linear inequality constraints.
 */

#include "gjk.hpp"
#include "../robot/arm.hpp"
#include <vector>

namespace trajopt {

/**
 * A single collision pair: (link_index, obstacle support function, closest points, normal).
 * Produced by the broad-phase collision check; used to form the QP constraint.
 */
struct CollisionData {
    int link_idx;               // which robot link
    Eigen::Vector3d p_link;     // closest point on link (local frame)
    Eigen::Vector3d p_obs;      // closest point on obstacle (world frame)
    Eigen::Vector3d normal;     // unit normal from obstacle toward link
    double signed_dist;         // sd > 0: separated, sd < 0: penetrating
};

/**
 * CollisionChecker: wraps the GJK algorithm for robot-obstacle queries.
 *
 * For each robot configuration q, checks all (link, obstacle) pairs
 * and returns the collision data needed to form QP constraints.
 */
class CollisionChecker {
public:
    struct Obstacle {
        std::string name;
        SupportFn support;
        Eigen::Vector3d center;  // for distance culling
    };

    explicit CollisionChecker(const RobotArm& arm) : arm_(arm) {}

    void add_obstacle(const std::string& name, SupportFn support,
                      const Eigen::Vector3d& center) {
        obstacles_.push_back({name, std::move(support), center});
    }

    void add_box_obstacle(const std::string& name,
                          const Eigen::Vector3d& box_min,
                          const Eigen::Vector3d& box_max) {
        Eigen::Vector3d center = 0.5 * (box_min + box_max);
        add_obstacle(name, box_support(box_min, box_max), center);
    }

    void add_sphere_obstacle(const std::string& name,
                             const Eigen::Vector3d& center, double radius) {
        add_obstacle(name, sphere_support(center, radius), center);
    }

    /**
     * Check all (link, obstacle) pairs and return collision data.
     *
     * d_check: only consider pairs where sd < d_check (broad-phase culling).
     */
    std::vector<CollisionData> check(
        const Eigen::VectorXd& q,
        double d_check = 0.3,
        const SE3& T_base = SE3::Identity()) const
    {
        std::vector<CollisionData> results;
        const auto T_world = arm_.forward_kinematics(q, T_base);

        for (int li = 0; li < arm_.n_joints(); ++li) {
            const SE3& T_link = T_world[li + 1];
            const Eigen::Vector3d link_pos = T_link.block<3,1>(0,3);

            // Link support function: capsule in world frame
            auto link_support = make_link_support(arm_.links()[li], T_link);

            for (const auto& obs : obstacles_) {
                // Broad-phase: distance between centers
                if ((link_pos - obs.center).norm() > d_check + 0.5)
                    continue;

                auto gjk_result = gjk_distance(link_support, obs.support);

                double sd = gjk_result.intersecting ? -0.01 : gjk_result.distance;

                if (sd < d_check) {
                    CollisionData cd;
                    cd.link_idx = li;
                    cd.normal = gjk_result.normal;
                    cd.signed_dist = sd;

                    // Convert world-frame closest point to local frame
                    cd.p_link = (T_link.inverse() *
                                 gjk_result.p_A.homogeneous()).head<3>();
                    cd.p_obs = gjk_result.p_B;

                    results.push_back(cd);
                }
            }
        }
        return results;
    }

    /**
     * Continuous-time collision check using swept volume.
     *
     * Between timesteps t and t+1, approximate the swept volume as the
     * convex hull of the link shapes at both poses. (Eq. 20-22)
     *
     * This is twice as expensive as discrete checking but handles
     * thin obstacles that the robot could "tunnel" through.
     */
    std::vector<CollisionData> check_swept(
        const Eigen::VectorXd& q_t,
        const Eigen::VectorXd& q_t1,
        double d_check = 0.3,
        const SE3& T_base = SE3::Identity()) const
    {
        std::vector<CollisionData> results;
        const auto T_t   = arm_.forward_kinematics(q_t,   T_base);
        const auto T_t1  = arm_.forward_kinematics(q_t1,  T_base);

        for (int li = 0; li < arm_.n_joints(); ++li) {
            auto supp_t  = make_link_support(arm_.links()[li], T_t[li+1]);
            auto supp_t1 = make_link_support(arm_.links()[li], T_t1[li+1]);
            auto supp_hull = convex_hull_support(supp_t, supp_t1);

            Eigen::Vector3d center_t   = T_t[li+1].block<3,1>(0,3);
            Eigen::Vector3d center_t1  = T_t1[li+1].block<3,1>(0,3);
            Eigen::Vector3d hull_center = 0.5 * (center_t + center_t1);

            for (const auto& obs : obstacles_) {
                if ((hull_center - obs.center).norm() > d_check + 0.5)
                    continue;

                auto gjk_result = gjk_distance(supp_hull, obs.support);
                double sd = gjk_result.intersecting ? -0.01 : gjk_result.distance;

                if (sd < d_check) {
                    // Determine which timestep the contact point belongs to (Eq. 23-24)
                    const Eigen::Vector3d p0 = supp_t(gjk_result.normal);
                    const Eigen::Vector3d p1 = supp_t1(gjk_result.normal);
                    const Eigen::Vector3d& ps = gjk_result.p_A;

                    double d0 = (ps - p0).norm();
                    double d1 = (ps - p1).norm();
                    double alpha = (d0 + d1 > 1e-10) ? d1 / (d0 + d1) : 0.5;

                    // Pack as two collision data entries (one per timestep)
                    // weighted by alpha and (1-alpha)
                    CollisionData cd;
                    cd.link_idx = li;
                    cd.normal = gjk_result.normal;
                    cd.signed_dist = sd;

                    // For now, assign to the closer timestep
                    // (a full implementation would split into two QP rows)
                    if (alpha > 0.5) {
                        cd.p_link = (T_t[li+1].inverse() * p0.homogeneous()).head<3>();
                    } else {
                        cd.p_link = (T_t1[li+1].inverse() * p1.homogeneous()).head<3>();
                    }
                    cd.p_obs = gjk_result.p_B;
                    results.push_back(cd);
                }
            }
        }
        return results;
    }

    const std::vector<Obstacle>& obstacles() const { return obstacles_; }

private:
    const RobotArm& arm_;
    std::vector<Obstacle> obstacles_;

    // Build world-frame support function for a robot link
    SupportFn make_link_support(const Link& link, const SE3& T_world) const {
        // Use the first collision geometry (capsule) in this link
        if (link.collision_geoms.empty()) {
            // Fallback: treat as a sphere at the link origin
            Eigen::Vector3d center = T_world.block<3,1>(0,3);
            return sphere_support(center, 0.05);
        }
        const auto& geom = link.collision_geoms[0];
        // Transform local capsule to world frame
        Eigen::Vector3d world_center = (T_world * geom.center.homogeneous()).head<3>();
        Eigen::Vector3d world_axis   = T_world.block<3,3>(0,0) * geom.axis;
        return capsule_support(world_center, world_axis, geom.radius, geom.half_length);
    }
};

/**
 * Linearized collision constraint for the QP solver.
 *
 * Given collision data (from GJK) and the robot Jacobian,
 * produces the linear inequality:
 *
 *   sd(x) ≈ sd(x0) + g^T * (x - x0) >= d_safe
 *   => -g^T * x <= sd(x0) - d_safe - g^T * x0
 *
 * where g = n_hat^T * J_pA (a row vector of size n_joints)
 */
struct LinearizedCollision {
    Eigen::VectorXd gradient;   // g = n_hat^T * J_pA, size = n_joints
    double sd_0;                // sd at current configuration
    double d_safe;              // safety margin

    /**
     * Hinge penalty value: max(d_safe - sd_0, 0)
     * This is one term of the ℓ1 collision penalty (Eq. 16).
     */
    double hinge_value() const {
        return std::max(d_safe - sd_0, 0.0);
    }

    /**
     * Is this collision active (sd < d_safe)?
     */
    bool is_active() const { return sd_0 < d_safe; }
};

/**
 * Linearize all collision constraints at configuration q.
 *
 * For each active collision pair, computes:
 *   gradient = n_hat^T * J_pA(x0)
 *
 * These gradients form the A matrix of the collision inequality constraints
 * in the QP subproblem.
 */
inline std::vector<LinearizedCollision> linearize_collisions(
    const RobotArm& arm,
    const Eigen::VectorXd& q,
    const std::vector<CollisionData>& collision_data,
    double d_safe,
    const SE3& T_base = SE3::Identity())
{
    std::vector<LinearizedCollision> lc_list;

    for (const auto& cd : collision_data) {
        if (cd.signed_dist > d_safe + 0.01)
            continue;  // Far away, no contribution

        // Position Jacobian of closest point on link w.r.t. joint angles
        Eigen::MatrixXd J = arm.position_jacobian(q, cd.link_idx + 1, cd.p_link, T_base);

        LinearizedCollision lc;
        // g = n_hat^T * J_pA  [1 x n_joints]  (Eq. 19)
        lc.gradient = cd.normal.transpose() * J;
        lc.sd_0 = cd.signed_dist;
        lc.d_safe = d_safe;

        lc_list.push_back(lc);
    }

    return lc_list;
}

/**
 * Total collision penalty (ℓ1 hinge sum) for a configuration.
 * Used to evaluate the merit function in the SQP trust region loop.
 */
inline double collision_penalty(
    const RobotArm& arm,
    CollisionChecker& checker,
    const Eigen::VectorXd& q,
    double d_safe,
    double penalty_coeff,
    const SE3& T_base = SE3::Identity())
{
    auto cd = checker.check(q, d_safe + 0.1, T_base);
    auto lc = linearize_collisions(arm, q, cd, d_safe, T_base);
    double total = 0.0;
    for (const auto& l : lc) total += l.hinge_value();
    return penalty_coeff * total;
}

} // namespace trajopt
