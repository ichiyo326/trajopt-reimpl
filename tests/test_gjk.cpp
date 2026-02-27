/**
 * test_gjk.cpp - Unit tests for GJK distance algorithm
 *
 * Tests:
 *   1. Two separated spheres
 *   2. Two overlapping spheres (penetration)
 *   3. Sphere vs box (known distance)
 *   4. Convex hull support function
 *   5. Capsule support
 *   6. Swept volume support (moving sphere)
 */

#include "../include/collision/gjk.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace trajopt;

static int n_passed = 0, n_failed = 0;

#define EXPECT_NEAR(a, b, tol, msg)                                           \
    do {                                                                      \
        double err = std::abs((a) - (b));                                     \
        if (err > (tol)) {                                                    \
            std::cerr << "FAIL [" << msg << "]: |" << (a) << " - " << (b)    \
                      << "| = " << err << " > " << (tol) << "\n";            \
            ++n_failed;                                                       \
        } else {                                                              \
            std::cout << "PASS [" << msg << "]\n";                           \
            ++n_passed;                                                       \
        }                                                                     \
    } while(0)

#define EXPECT_TRUE(cond, msg)                                                \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::cerr << "FAIL [" << msg << "]: expected true\n";            \
            ++n_failed;                                                       \
        } else {                                                              \
            std::cout << "PASS [" << msg << "]\n";                           \
            ++n_passed;                                                       \
        }                                                                     \
    } while(0)

#define EXPECT_FALSE(cond, msg)                                               \
    do {                                                                      \
        if (cond) {                                                           \
            std::cerr << "FAIL [" << msg << "]: expected false\n";           \
            ++n_failed;                                                       \
        } else {                                                              \
            std::cout << "PASS [" << msg << "]\n";                           \
            ++n_passed;                                                       \
        }                                                                     \
    } while(0)

void test_two_separated_spheres() {
    std::cout << "\n-- Two Separated Spheres --\n";
    // Sphere A centered at origin, radius 0.5
    // Sphere B centered at (2, 0, 0), radius 0.5
    // Expected distance: 2.0 - 0.5 - 0.5 = 1.0
    auto sA = sphere_support(Eigen::Vector3d(0, 0, 0), 0.5);
    auto sB = sphere_support(Eigen::Vector3d(2, 0, 0), 0.5);

    auto result = gjk_distance(sA, sB);

    EXPECT_FALSE(result.intersecting, "spheres not intersecting");
    EXPECT_NEAR(result.distance, 1.0, 0.01, "sphere-sphere distance");
}

void test_two_overlapping_spheres() {
    std::cout << "\n-- Two Overlapping Spheres --\n";
    // Both centered at origin -> definitely intersecting
    auto sA = sphere_support(Eigen::Vector3d(0, 0, 0), 0.5);
    auto sB = sphere_support(Eigen::Vector3d(0.3, 0, 0), 0.5);

    auto result = gjk_distance(sA, sB);
    EXPECT_TRUE(result.intersecting, "overlapping spheres detected");
}

void test_sphere_vs_box_separated() {
    std::cout << "\n-- Sphere vs Box (separated) --\n";
    // Box: [-0.5, -0.5, -0.5] to [0.5, 0.5, 0.5]
    // Sphere: center at (2, 0, 0), radius 0.3
    // Expected distance: 2.0 - 0.5 - 0.3 = 1.2
    auto sBox = box_support(Eigen::Vector3d(-0.5, -0.5, -0.5),
                             Eigen::Vector3d(0.5, 0.5, 0.5));
    auto sSphere = sphere_support(Eigen::Vector3d(2, 0, 0), 0.3);

    auto result = gjk_distance(sBox, sSphere);
    EXPECT_FALSE(result.intersecting, "sphere-box not intersecting");
    EXPECT_NEAR(result.distance, 1.2, 0.05, "sphere-box distance");
}

void test_sphere_vs_box_touching() {
    std::cout << "\n-- Sphere vs Box (touching) --\n";
    // Box: [0, -0.5, -0.5] to [1, 0.5, 0.5]
    // Sphere: center at (1.5, 0, 0), radius 0.5 -> touching at x=1
    auto sBox = box_support(Eigen::Vector3d(0, -0.5, -0.5),
                             Eigen::Vector3d(1, 0.5, 0.5));
    auto sSphere = sphere_support(Eigen::Vector3d(1.5, 0, 0), 0.5);

    auto result = gjk_distance(sBox, sSphere);
    // Distance should be ~0
    EXPECT_NEAR(result.distance, 0.0, 0.05, "touching sphere-box distance");
}

void test_capsule_support() {
    std::cout << "\n-- Capsule Support Function --\n";
    // Capsule along Z axis, center at origin, radius 0.1, half_length 0.5
    auto sCaps = capsule_support(Eigen::Vector3d(0, 0, 0),
                                  Eigen::Vector3d(0, 0, 1),
                                  0.1, 0.5);
    auto sSphere = sphere_support(Eigen::Vector3d(0, 0, 2), 0.2);

    // Distance: 2.0 - 0.5 - 0.1 - 0.2 = 1.2
    auto result = gjk_distance(sCaps, sSphere);
    EXPECT_FALSE(result.intersecting, "capsule-sphere not intersecting");
    EXPECT_NEAR(result.distance, 1.2, 0.1, "capsule-sphere distance");
}

void test_convex_hull_support() {
    std::cout << "\n-- Convex Hull Support (Swept Volume) --\n";
    // Shape A moves from center (0,0,0) to (1,0,0)
    // Convex hull of the two positions
    auto sA0 = sphere_support(Eigen::Vector3d(0, 0, 0), 0.2);
    auto sA1 = sphere_support(Eigen::Vector3d(1, 0, 0), 0.2);
    auto sHull = convex_hull_support(sA0, sA1);

    // Obstacle sphere at (2.5, 0, 0) radius 0.3
    // Expected gap: 2.5 - 1.0 - 0.2 - 0.3 = 1.0
    auto sObs = sphere_support(Eigen::Vector3d(2.5, 0, 0), 0.3);

    auto result = gjk_distance(sHull, sObs);
    EXPECT_FALSE(result.intersecting, "swept hull not intersecting");
    EXPECT_NEAR(result.distance, 1.0, 0.1, "swept hull distance");
}

void test_normal_direction() {
    std::cout << "\n-- Normal Direction --\n";
    // Sphere A at origin, sphere B at (1, 0, 0)
    // Normal from B toward A should be (-1, 0, 0)
    auto sA = sphere_support(Eigen::Vector3d(0, 0, 0), 0.2);
    auto sB = sphere_support(Eigen::Vector3d(1, 0, 0), 0.2);

    auto result = gjk_distance(sA, sB);
    EXPECT_FALSE(result.intersecting, "normal direction test: not intersecting");
    // Normal points from B toward A: (-1, 0, 0)
    EXPECT_NEAR(std::abs(result.normal(0)), 1.0, 0.05, "normal x ≈ ±1");
    EXPECT_NEAR(std::abs(result.normal(1)), 0.0, 0.05, "normal y ≈ 0");
    EXPECT_NEAR(std::abs(result.normal(2)), 0.0, 0.05, "normal z ≈ 0");
}

void test_box_box() {
    std::cout << "\n-- Box vs Box --\n";
    // Box A: [0,0,0] to [1,1,1]
    // Box B: [2,0,0] to [3,1,1]
    // Distance along x: 2 - 1 = 1.0
    auto sA = box_support(Eigen::Vector3d(0,0,0), Eigen::Vector3d(1,1,1));
    auto sB = box_support(Eigen::Vector3d(2,0,0), Eigen::Vector3d(3,1,1));

    auto result = gjk_distance(sA, sB);
    EXPECT_FALSE(result.intersecting, "box-box not intersecting");
    EXPECT_NEAR(result.distance, 1.0, 0.05, "box-box distance");
}

int main() {
    std::cout << "Running GJK tests...\n";

    test_two_separated_spheres();
    test_two_overlapping_spheres();
    test_sphere_vs_box_separated();
    test_sphere_vs_box_touching();
    test_capsule_support();
    test_convex_hull_support();
    test_normal_direction();
    test_box_box();

    std::cout << "\n==================\n";
    std::cout << "Results: " << n_passed << " passed, " << n_failed << " failed\n";
    std::cout << "==================\n";

    return n_failed > 0 ? 1 : 0;
}
