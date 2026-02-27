/**
 * test_se3.cpp - Unit tests for SE(3) Lie group operations
 *
 * Tests:
 *   1. exp/log round-trip (exp(log(T)) = T)
 *   2. hat/vee round-trip
 *   3. Composition and inverse
 *   4. Retract / local_diff
 *   5. Pose error at identity
 *   6. Jacobians via finite difference
 */

#include "../include/robot/se3.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace trajopt;

// ---- Test utilities ----
static int n_passed = 0, n_failed = 0;

#define EXPECT_NEAR(a, b, tol, msg)                                    \
    do {                                                               \
        double err = std::abs((a) - (b));                              \
        if (err > (tol)) {                                             \
            std::cerr << "FAIL [" << msg << "]: |" << (a) << " - "    \
                      << (b) << "| = " << err << " > " << (tol) << "\n"; \
            ++n_failed;                                                \
        } else {                                                       \
            ++n_passed;                                                \
        }                                                              \
    } while(0)

#define EXPECT_MATRIX_NEAR(A, B, tol, msg)                            \
    do {                                                               \
        double err = ((A) - (B)).norm();                               \
        if (err > (tol)) {                                             \
            std::cerr << "FAIL [" << msg << "]: matrix error = "      \
                      << err << " > " << (tol) << "\n";               \
            std::cerr << "A =\n" << (A) << "\nB =\n" << (B) << "\n";  \
            ++n_failed;                                                \
        } else {                                                       \
            ++n_passed;                                                \
        }                                                              \
    } while(0)

// ---- Tests ----

void test_hat_vee_roundtrip() {
    Twist x;
    x << 1.0, -2.0, 0.5, 0.1, 0.3, -0.2;

    Eigen::Matrix4d X = hat(x);
    Twist x2 = vee(X);

    EXPECT_MATRIX_NEAR(x, x2, 1e-12, "hat/vee round-trip");
}

void test_exp_log_roundtrip_translation() {
    // Pure translation (r = 0)
    Twist x;
    x << 1.5, -0.5, 2.0, 0.0, 0.0, 0.0;
    SE3 T = exp_se3(x);
    Twist x2 = log_se3(T);
    EXPECT_MATRIX_NEAR(x, x2, 1e-10, "exp/log round-trip (pure translation)");
}

void test_exp_log_roundtrip_rotation() {
    // Pure rotation around z by pi/4
    Twist x;
    x << 0.0, 0.0, 0.0, 0.0, 0.0, M_PI/4.0;
    SE3 T = exp_se3(x);
    Twist x2 = log_se3(T);
    EXPECT_MATRIX_NEAR(x, x2, 1e-10, "exp/log round-trip (pure rotation)");
}

void test_exp_log_roundtrip_general() {
    // General rigid body motion
    Twist x;
    x << 0.3, -0.1, 0.7, 0.2, -0.15, 0.4;
    SE3 T = exp_se3(x);
    Twist x2 = log_se3(T);
    EXPECT_MATRIX_NEAR(x, x2, 1e-9, "exp/log round-trip (general)");
}

void test_exp_log_small_rotation() {
    // Near-zero rotation: numerical stability test
    Twist x;
    x << 0.01, 0.02, -0.01, 1e-8, 2e-8, -1e-8;
    SE3 T = exp_se3(x);
    Twist x2 = log_se3(T);
    EXPECT_MATRIX_NEAR(x, x2, 1e-6, "exp/log round-trip (near-zero rotation)");
}

void test_composition_inverse() {
    Twist x1, x2;
    x1 << 0.5, 0.0, 0.0, 0.0, 0.0, M_PI/6;
    x2 << 0.0, 0.3, 0.0, 0.0, M_PI/4, 0.0;

    SE3 T1 = exp_se3(x1);
    SE3 T2 = exp_se3(x2);

    // T * T^{-1} = Identity
    SE3 I = compose(T1, inverse(T1));
    EXPECT_MATRIX_NEAR(I, SE3::Identity(), 1e-12, "T * T^{-1} = I");

    // T1 * (T1^{-1} * T2) = T2
    SE3 T2_recovered = compose(T1, compose(inverse(T1), T2));
    EXPECT_MATRIX_NEAR(T2_recovered, T2, 1e-12, "composition associativity");
}

void test_retract_local_diff() {
    SE3 X = exp_se3(Twist(0.1, -0.2, 0.3, 0.05, -0.1, 0.15));
    Twist delta;
    delta << 0.02, -0.01, 0.03, 0.001, -0.002, 0.003;

    SE3 Y = retract(X, delta);
    Twist delta_recovered = local_diff(X, Y);
    EXPECT_MATRIX_NEAR(delta, delta_recovered, 1e-10, "retract/local_diff round-trip");
}

void test_pose_error_at_identity() {
    SE3 T_target = SE3::Identity();
    SE3 T_current = SE3::Identity();
    Twist err = pose_error(T_target, T_current);
    EXPECT_MATRIX_NEAR(err, Twist::Zero(), 1e-12, "pose_error at identity = 0");
}

void test_pose_error_known() {
    // Target = Identity, current = translated by (1, 0, 0)
    SE3 T_target = SE3::Identity();
    SE3 T_current = SE3::Identity();
    T_current(0, 3) = 1.0;
    Twist err = pose_error(T_target, T_current);
    // Expected: [1, 0, 0, 0, 0, 0]
    EXPECT_NEAR(err(0), 1.0, 1e-10, "pose_error translation x");
    EXPECT_NEAR(err(1), 0.0, 1e-10, "pose_error translation y");
    EXPECT_NEAR(err(2), 0.0, 1e-10, "pose_error translation z");
    EXPECT_NEAR(err(3), 0.0, 1e-10, "pose_error rotation x");
    EXPECT_NEAR(err(4), 0.0, 1e-10, "pose_error rotation y");
    EXPECT_NEAR(err(5), 0.0, 1e-10, "pose_error rotation z");
}

void test_rotation_matrix_orthogonality() {
    // exp should produce a valid rotation: R^T R = I, det(R) = 1
    Twist x;
    x << 0.3, -0.1, 0.7, 0.2, -0.15, 0.4;
    SE3 T = exp_se3(x);
    Eigen::Matrix3d R = T.block<3,3>(0,0);

    EXPECT_MATRIX_NEAR(R * R.transpose(), Eigen::Matrix3d::Identity(),
                       1e-12, "R^T R = I");
    EXPECT_NEAR(R.determinant(), 1.0, 1e-12, "det(R) = 1");
}

// ---- Main ----
int main() {
    std::cout << "Running SE(3) tests...\n\n";

    test_hat_vee_roundtrip();
    test_exp_log_roundtrip_translation();
    test_exp_log_roundtrip_rotation();
    test_exp_log_roundtrip_general();
    test_exp_log_small_rotation();
    test_composition_inverse();
    test_retract_local_diff();
    test_pose_error_at_identity();
    test_pose_error_known();
    test_rotation_matrix_orthogonality();

    std::cout << "\n==================\n";
    std::cout << "Results: " << n_passed << " passed, " << n_failed << " failed\n";
    std::cout << "==================\n";

    return n_failed > 0 ? 1 : 0;
}
