#pragma once
/**
 * se3.hpp - SE(3) Lie Group Operations for Trajectory Optimization
 *
 * Implements the Special Euclidean group SE(3) and its Lie algebra se(3),
 * following the formulation in:
 *   Schulman et al., "Motion Planning with Sequential Convex Optimization
 *   and Convex Collision Checking", IJRR 2014.
 *
 * Key references for the math:
 *   - Murray, Li, Sastry, "A Mathematical Introduction to Robotic Manipulation"
 *   - Blanco, "A Tutorial on SE(3) Transformation Parameterizations"
 *
 * Note for readers from signal processing backgrounds:
 *   SE(3) optimization is analogous to phase retrieval in that both involve
 *   non-convex domains that we linearize locally. The "incremental twist"
 *   parameterization here plays the same role as a local phase update in
 *   iterative phase retrieval algorithms (e.g., HIO, RAAR).
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace trajopt {

// A pose in SE(3): a 4x4 homogeneous transformation matrix
//   [ R  p ]
//   [ 0  1 ]
// where R in SO(3) and p in R^3
using SE3 = Eigen::Matrix4d;
using SO3 = Eigen::Matrix3d;

// A 6-vector [p; r] representing an element of se(3) (Lie algebra)
// p = translational component, r = rotational component (axis-angle)
using Twist = Eigen::Matrix<double, 6, 1>;

/**
 * Skew-symmetric matrix [r]x such that [r]x * v = r.cross(v)
 *
 * [r]x = [  0   -rz   ry ]
 *        [  rz   0   -rx ]
 *        [ -ry   rx    0  ]
 */
inline Eigen::Matrix3d skew(const Eigen::Vector3d& r) {
    Eigen::Matrix3d S;
    S <<    0, -r(2),  r(1),
         r(2),     0, -r(0),
        -r(1),  r(0),     0;
    return S;
}

/**
 * hat operator: R^6 -> se(3)
 * Maps a twist vector [p; r] to its 4x4 Lie algebra element.
 *
 * x_hat = [ [r]x  p ]
 *          [  0   0 ]
 */
inline Eigen::Matrix4d hat(const Twist& x) {
    Eigen::Matrix4d X = Eigen::Matrix4d::Zero();
    X.block<3,3>(0,0) = skew(x.tail<3>());  // [r]x in top-left
    X.block<3,1>(0,3) = x.head<3>();         // p in top-right
    return X;
}

/**
 * vee operator: se(3) -> R^6 (inverse of hat)
 * Extracts the twist vector from a 4x4 Lie algebra element.
 */
inline Twist vee(const Eigen::Matrix4d& X) {
    Twist x;
    x.head<3>() = X.block<3,1>(0,3);  // translation
    // Extract from skew-symmetric part: [r]x(2,1)=rx, [r]x(0,2)=ry, [r]x(1,0)=rz
    x(3) = X(2,1);
    x(4) = X(0,2);
    x(5) = X(1,0);
    return x;
}

/**
 * Exponential map: se(3) -> SE(3)
 * Converts a twist (Lie algebra element) to a rigid transformation.
 *
 * For r = 0:  exp(x^) = [ I  p ]
 *                        [ 0  1 ]
 *
 * For r != 0: exp(x^) = [ exp_r   A*p ]
 *                        [   0      1  ]
 *
 * where:
 *   exp_r = I + [r]/||r|| * sin(||r||) + [r]^2/||r||^2 * (1 - cos(||r||))
 *   A     = I + [r]/||r||^2 * (1 - cos(||r||)) + [r]^2/||r||^3 * (||r|| - sin(||r||))
 *
 * (Equations (8)-(10) in Schulman et al.)
 */
inline SE3 exp_se3(const Twist& x) {
    const Eigen::Vector3d p = x.head<3>();
    const Eigen::Vector3d r = x.tail<3>();
    const double theta = r.norm();

    SE3 T = SE3::Identity();

    if (theta < 1e-10) {
        // Pure translation: no rotation
        T.block<3,1>(0,3) = p;
        return T;
    }

    const Eigen::Matrix3d rx = skew(r);
    const double s = std::sin(theta);
    const double c = std::cos(theta);

    // Rodrigues' rotation formula
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity()
        + (s / theta) * rx
        + ((1.0 - c) / (theta * theta)) * rx * rx;

    // A matrix maps Lie algebra translation to actual translation
    const Eigen::Matrix3d A = Eigen::Matrix3d::Identity()
        + ((1.0 - c) / (theta * theta)) * rx
        + ((theta - s) / (theta * theta * theta)) * rx * rx;

    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = A * p;
    return T;
}

/**
 * Logarithm map: SE(3) -> se(3)
 * Inverse of the exponential map. Converts a rigid transformation to a twist.
 *
 * For R = I:   log(T) = [ p; 0 ]
 *
 * For R != I:  log(T) = [ A^{-1}*p; theta*omega_hat ]
 *
 * where theta = arccos((trace(R)-1)/2), omega_hat = R-R^T / (2*sin(theta))
 * (Equations (30)-(32) in Schulman et al.)
 */
inline Twist log_se3(const SE3& T) {
    const Eigen::Matrix3d R = T.block<3,3>(0,0);
    const Eigen::Vector3d p = T.block<3,1>(0,3);

    Twist x;
    const double trace_val = R.trace();
    const double cos_theta = (trace_val - 1.0) * 0.5;
    // Clamp for numerical safety
    const double theta = std::acos(std::max(-1.0, std::min(1.0, cos_theta)));

    if (theta < 1e-10) {
        // R ~ Identity, pure translation
        x.head<3>() = p;
        x.tail<3>().setZero();
        return x;
    }

    // axis-angle from rotation matrix
    const Eigen::Matrix3d omega_hat = (R - R.transpose()) / (2.0 * std::sin(theta));
    const Eigen::Vector3d r(omega_hat(2,1), omega_hat(0,2), omega_hat(1,0));
    const Eigen::Vector3d r_scaled = theta * r;

    // A^{-1} for recovering translational component
    const Eigen::Matrix3d rx = skew(r);
    const double s = std::sin(theta);
    const double c = std::cos(theta);
    const Eigen::Matrix3d A_inv = Eigen::Matrix3d::Identity()
        - 0.5 * rx
        + ((1.0 / (theta * theta)) - (1.0 + c) / (2.0 * theta * s)) * rx * rx;

    x.head<3>() = A_inv * p;
    x.tail<3>() = r_scaled;
    return x;
}

/**
 * Compose two SE(3) transformations: T1 * T2
 */
inline SE3 compose(const SE3& T1, const SE3& T2) {
    return T1 * T2;
}

/**
 * Inverse of an SE(3) transformation
 *   T^{-1} = [ R^T  -R^T*p ]
 *             [  0      1   ]
 */
inline SE3 inverse(const SE3& T) {
    SE3 T_inv = SE3::Identity();
    const Eigen::Matrix3d RT = T.block<3,3>(0,0).transpose();
    T_inv.block<3,3>(0,0) = RT;
    T_inv.block<3,1>(0,3) = -RT * T.block<3,1>(0,3);
    return T_inv;
}

/**
 * Retract: update a pose X by applying an incremental twist delta_x
 *   X_new = X * exp(delta_x^)
 *
 * This is the local update step used in each SQP iteration.
 * Analogous to the "step" in gradient descent, but respects the SE(3) geometry.
 */
inline SE3 retract(const SE3& X, const Twist& delta_x) {
    return compose(X, exp_se3(delta_x));
}

/**
 * Local difference between two poses: what twist takes X to Y?
 *   diff(X, Y) = log(X^{-1} * Y)
 */
inline Twist local_diff(const SE3& X, const SE3& Y) {
    return log_se3(compose(inverse(X), Y));
}

/**
 * Construct SE(3) from rotation matrix and translation vector
 */
inline SE3 make_pose(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
    SE3 T = SE3::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = p;
    return T;
}

/**
 * Construct SE(3) from axis-angle and translation
 */
inline SE3 make_pose(const Eigen::Vector3d& axis_angle, const Eigen::Vector3d& p) {
    Twist x;
    x.head<3>() = Eigen::Vector3d::Zero();
    x.tail<3>() = axis_angle;
    SE3 T = exp_se3(x);
    T.block<3,1>(0,3) = p;
    return T;
}

/**
 * Extract position from SE(3)
 */
inline Eigen::Vector3d position(const SE3& T) {
    return T.block<3,1>(0,3);
}

/**
 * Extract rotation matrix from SE(3)
 */
inline Eigen::Matrix3d rotation(const SE3& T) {
    return T.block<3,3>(0,0);
}

/**
 * Pose error as 6-vector: log(T_target^{-1} * T_current)
 * Used for end-effector equality constraints in motion planning.
 * (Equation (29) in Schulman et al.)
 */
inline Twist pose_error(const SE3& T_target, const SE3& T_current) {
    return log_se3(compose(inverse(T_target), T_current));
}

} // namespace trajopt
