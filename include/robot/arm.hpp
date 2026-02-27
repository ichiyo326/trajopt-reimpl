#pragma once
/**
 * arm.hpp - N-Link Robot Arm: Forward Kinematics and Analytical Jacobian
 *
 * Implements a serial kinematic chain using Denavit-Hartenberg (DH) parameters.
 * Provides forward kinematics (joint angles -> end-effector pose) and the
 * geometric Jacobian needed for linearizing collision constraints.
 *
 * The Jacobian is central to TrajOpt's collision linearization (Eq. 19):
 *   sd_AB(x) ≈ sd_AB(x0) + n_hat^T * J_pA(x0) * (x - x0)
 */

#include "se3.hpp"
#include <vector>
#include <string>

namespace trajopt {

/**
 * Denavit-Hartenberg parameters for a single joint.
 * Standard DH convention:
 *   a     - link length (distance along x_{i-1})
 *   alpha - link twist  (rotation around x_{i-1})
 *   d     - link offset (distance along z_i)
 *   theta - joint angle (rotation around z_i) -- this is the variable
 */
struct DHParam {
    double a;      // link length [m]
    double alpha;  // link twist [rad]
    double d;      // link offset [m]
    double theta;  // joint angle offset [rad] (added to q_i)
    double q_min;  // joint limit lower [rad]
    double q_max;  // joint limit upper [rad]

    DHParam(double a, double alpha, double d, double theta,
            double q_min = -M_PI, double q_max = M_PI)
        : a(a), alpha(alpha), d(d), theta(theta), q_min(q_min), q_max(q_max) {}
};

/**
 * Rigid link with associated geometry (for collision checking).
 */
struct Link {
    std::string name;
    DHParam dh;
    // Collision geometry: a set of convex shapes attached to this link.
    // For simplicity, we represent each as a capsule: (local_center, radius, half_length).
    // In a full implementation, these would be convex meshes.
    struct CapsuleGeom {
        Eigen::Vector3d center;  // local frame center
        Eigen::Vector3d axis;    // local frame axis (unit vector)
        double radius;
        double half_length;
    };
    std::vector<CapsuleGeom> collision_geoms;

    Link(const std::string& name, const DHParam& dh) : name(name), dh(dh) {}
};

/**
 * Serial robot arm: N revolute joints.
 *
 * Configuration q in R^N (joint angles).
 * Forward kinematics gives the SE(3) pose of each link frame and the end-effector.
 */
class RobotArm {
public:
    explicit RobotArm(std::vector<Link> links) : links_(std::move(links)) {
        n_joints_ = links_.size();
    }

    int n_joints() const { return n_joints_; }
    const std::vector<Link>& links() const { return links_; }

    /**
     * Compute the DH transformation for a single joint.
     * T_i = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
     */
    SE3 dh_transform(const DHParam& dh, double q) const {
        const double theta = dh.theta + q;
        const double ct = std::cos(theta), st = std::sin(theta);
        const double ca = std::cos(dh.alpha), sa = std::sin(dh.alpha);

        SE3 T;
        T << ct, -st*ca,  st*sa, dh.a*ct,
             st,  ct*ca, -ct*sa, dh.a*st,
              0,     sa,     ca,     dh.d,
              0,      0,      0,        1;
        return T;
    }

    /**
     * Forward kinematics: joint angles -> link frame poses in world frame.
     *
     * Returns T_0^world, T_1^world, ..., T_N^world
     * where T_i^world is the pose of the i-th link frame.
     *
     * T_i^world = T_0 * T_1 * ... * T_i  (T_0 = base transform, identity by default)
     */
    std::vector<SE3> forward_kinematics(
        const Eigen::VectorXd& q,
        const SE3& T_base = SE3::Identity()) const
    {
        assert(q.size() == n_joints_);
        std::vector<SE3> T_world(n_joints_ + 1);
        T_world[0] = T_base;

        for (int i = 0; i < n_joints_; ++i) {
            T_world[i + 1] = T_world[i] * dh_transform(links_[i].dh, q(i));
        }
        return T_world;
    }

    /**
     * End-effector pose.
     */
    SE3 end_effector_pose(const Eigen::VectorXd& q,
                          const SE3& T_base = SE3::Identity()) const {
        return forward_kinematics(q, T_base).back();
    }

    /**
     * Geometric Jacobian of a point p_local attached to link k,
     * with respect to joint angles q.
     *
     * J in R^{6 x N}: top 3 rows = linear velocity Jacobian,
     *                 bottom 3 rows = angular velocity Jacobian
     *
     * For revolute joint i (i <= k):
     *   J_v_i = z_i x (p_world - o_i)   (linear component)
     *   J_w_i = z_i                       (angular component)
     * For joint i > k: J_i = 0
     *
     * This is the Jacobian J_pA needed in Eq. (19) of Schulman et al.
     */
    Eigen::MatrixXd jacobian(
        const Eigen::VectorXd& q,
        int link_idx,                    // which link the point is attached to
        const Eigen::Vector3d& p_local,  // point in link's local frame
        const SE3& T_base = SE3::Identity()) const
    {
        assert(q.size() == n_joints_);
        assert(link_idx >= 0 && link_idx <= n_joints_);

        const std::vector<SE3> T_world = forward_kinematics(q, T_base);

        // World-frame position of the point
        const Eigen::Vector3d p_world =
            (T_world[link_idx] * p_local.homogeneous()).head<3>();

        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, n_joints_);

        for (int i = 0; i < link_idx && i < n_joints_; ++i) {
            // z-axis of joint i's frame in world coordinates
            const Eigen::Vector3d z_i = T_world[i].block<3,1>(0,2);
            // Origin of joint i's frame in world coordinates
            const Eigen::Vector3d o_i = T_world[i].block<3,1>(0,3);

            J.block<3,1>(0, i) = z_i.cross(p_world - o_i);  // linear
            J.block<3,1>(3, i) = z_i;                         // angular
        }

        return J;
    }

    /**
     * 3xN position Jacobian (linear part only).
     * Convenience method for collision constraint linearization.
     */
    Eigen::MatrixXd position_jacobian(
        const Eigen::VectorXd& q,
        int link_idx,
        const Eigen::Vector3d& p_local,
        const SE3& T_base = SE3::Identity()) const
    {
        return jacobian(q, link_idx, p_local, T_base).topRows<3>();
    }

    /**
     * Check if configuration q satisfies joint limits.
     */
    bool within_joint_limits(const Eigen::VectorXd& q) const {
        for (int i = 0; i < n_joints_; ++i) {
            if (q(i) < links_[i].dh.q_min || q(i) > links_[i].dh.q_max)
                return false;
        }
        return true;
    }

    /**
     * Clamp q to joint limits (for initialization).
     */
    Eigen::VectorXd clamp_to_limits(const Eigen::VectorXd& q) const {
        Eigen::VectorXd q_clamped = q;
        for (int i = 0; i < n_joints_; ++i) {
            q_clamped(i) = std::max(links_[i].dh.q_min,
                          std::min(links_[i].dh.q_max, q(i)));
        }
        return q_clamped;
    }

    /**
     * Factory: create a standard 7-DOF robot arm (Panda-like).
     * DH parameters approximate the Franka Emika Panda.
     */
    static RobotArm make_panda() {
        using L = Link;
        using DH = DHParam;
        constexpr double pi = M_PI;
        std::vector<Link> links = {
            L("joint1", DH(0,      0,     0.333,  0,   -2.8973,  2.8973)),
            L("joint2", DH(0,    -pi/2,   0,      0,   -1.7628,  1.7628)),
            L("joint3", DH(0,     pi/2,   0.316,  0,   -2.8973,  2.8973)),
            L("joint4", DH(0.0825, pi/2,  0,      0,   -3.0718, -0.0698)),
            L("joint5", DH(-0.0825,-pi/2, 0.384,  0,   -2.8973,  2.8973)),
            L("joint6", DH(0,      pi/2,  0,      0,   -0.0175,  3.7525)),
            L("joint7", DH(0.088,  pi/2,  0.107,  0,   -2.8973,  2.8973)),
        };
        // Add simple capsule geometry for each link
        for (auto& link : links) {
            link.collision_geoms.push_back({
                Eigen::Vector3d(0, 0, 0.05),
                Eigen::Vector3d(0, 0, 1),
                0.06,  // radius
                0.05   // half_length
            });
        }
        return RobotArm(std::move(links));
    }

    /**
     * Factory: create a simple 3-DOF planar arm for 2D demos.
     * All joints rotate in the XY plane.
     */
    static RobotArm make_planar_3dof(double link_length = 1.0) {
        using L = Link;
        using DH = DHParam;
        std::vector<Link> links = {
            L("joint1", DH(link_length, 0, 0, 0, -M_PI, M_PI)),
            L("joint2", DH(link_length, 0, 0, 0, -M_PI, M_PI)),
            L("joint3", DH(link_length, 0, 0, 0, -M_PI, M_PI)),
        };
        return RobotArm(std::move(links));
    }

private:
    std::vector<Link> links_;
    int n_joints_;
};

} // namespace trajopt
