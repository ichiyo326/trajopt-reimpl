#pragma once
/**
 * gjk.hpp - Gilbert-Johnson-Keerthi (GJK) Distance Algorithm
 *
 * Computes the minimum distance between two convex shapes using their
 * support function representations. This is exactly the approach used
 * in TrajOpt and described in Section IV-A of Schulman et al.
 *
 * The key insight: any convex shape can be queried via its support mapping:
 *   s_A(v) = argmax_{p in A} v·p
 * This allows distance computation without explicit mesh representation.
 *
 * References:
 *   Gilbert, Johnson, Keerthi (1988) - original GJK paper
 *   van den Bergen (2001) - modern GJK with numerical robustness
 *   Ericson (2004) - "Real-Time Collision Detection" (practical details)
 */

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <functional>
#include <optional>

namespace trajopt {

// Support function type: maps direction v to furthest point in shape along v
using SupportFn = std::function<Eigen::Vector3d(const Eigen::Vector3d&)>;

/**
 * Result of GJK distance query.
 */
struct GJKResult {
    bool intersecting;        // true if shapes overlap
    double distance;          // min distance (0 if intersecting)
    Eigen::Vector3d p_A;      // closest point on A (world frame)
    Eigen::Vector3d p_B;      // closest point on B (world frame)
    Eigen::Vector3d normal;   // unit normal from B toward A (outward from B)
};

/**
 * Simplex: up to 4 points in the Minkowski difference space.
 * Used internally by GJK to track the current candidate simplex.
 */
class Simplex {
public:
    struct SupportPoint {
        Eigen::Vector3d v;    // point in Minkowski difference: p_A - p_B
        Eigen::Vector3d p_A;  // corresponding point on A
        Eigen::Vector3d p_B;  // corresponding point on B
    };

    void clear() { size_ = 0; }
    int size() const { return size_; }

    void push(const SupportPoint& sp) {
        // Shift and insert at front (most recent point first)
        for (int i = std::min(size_, 3); i > 0; --i)
            pts_[i] = pts_[i-1];
        pts_[0] = sp;
        size_ = std::min(size_ + 1, 4);
    }

    SupportPoint& operator[](int i) { return pts_[i]; }
    const SupportPoint& operator[](int i) const { return pts_[i]; }

    // Keep only specified indices
    void keep(std::initializer_list<int> indices) {
        std::array<SupportPoint, 4> tmp;
        int new_size = 0;
        for (int idx : indices) tmp[new_size++] = pts_[idx];
        for (int i = 0; i < new_size; ++i) pts_[i] = tmp[i];
        size_ = new_size;
    }

private:
    std::array<SupportPoint, 4> pts_;
    int size_ = 0;
};

/**
 * Support point on the Minkowski difference of A and B in direction d.
 * s_{A-B}(d) = s_A(d) - s_B(-d)
 */
inline Simplex::SupportPoint minkowski_support(
    const SupportFn& support_A,
    const SupportFn& support_B,
    const Eigen::Vector3d& d)
{
    Simplex::SupportPoint sp;
    sp.p_A = support_A(d);
    sp.p_B = support_B(-d);
    sp.v = sp.p_A - sp.p_B;
    return sp;
}

/**
 * Determine if the simplex contains the origin, and update search direction.
 * Returns true if origin is enclosed (intersection found).
 *
 * This implements the "do_simplex" subroutine from van den Bergen (2001).
 */
inline bool do_simplex(Simplex& simplex, Eigen::Vector3d& d) {
    const int n = simplex.size();

    if (n == 1) {
        // Point: direction toward origin
        d = -simplex[0].v;
        return d.squaredNorm() < 1e-14;
    }

    if (n == 2) {
        // Line segment AB
        const Eigen::Vector3d ab = simplex[1].v - simplex[0].v;
        const Eigen::Vector3d ao = -simplex[0].v;

        if (ab.dot(ao) > 0) {
            // Origin in Voronoi region of edge
            d = ab.cross(ao).cross(ab);
            if (d.squaredNorm() < 1e-14) {
                // Origin on line: need perpendicular from this plane
                d = Eigen::Vector3d(ab(1), -ab(0), 0);
                if (d.squaredNorm() < 1e-14)
                    d = Eigen::Vector3d(0, ab(2), -ab(1));
            }
        } else {
            // Origin closest to A
            simplex.keep({0});
            d = ao;
        }
        return false;
    }

    if (n == 3) {
        // Triangle ABC
        const Eigen::Vector3d a = simplex[0].v;
        const Eigen::Vector3d b = simplex[1].v;
        const Eigen::Vector3d c = simplex[2].v;

        const Eigen::Vector3d ab = b - a;
        const Eigen::Vector3d ac = c - a;
        const Eigen::Vector3d ao = -a;
        const Eigen::Vector3d abc = ab.cross(ac);  // triangle normal

        if (abc.cross(ac).dot(ao) > 0) {
            if (ac.dot(ao) > 0) {
                simplex.keep({0, 2});
                d = ac.cross(ao).cross(ac);
            } else {
                if (ab.dot(ao) > 0) {
                    simplex.keep({0, 1});
                    d = ab.cross(ao).cross(ab);
                } else {
                    simplex.keep({0});
                    d = ao;
                }
            }
        } else {
            if (ab.cross(abc).dot(ao) > 0) {
                if (ab.dot(ao) > 0) {
                    simplex.keep({0, 1});
                    d = ab.cross(ao).cross(ab);
                } else {
                    simplex.keep({0});
                    d = ao;
                }
            } else {
                // Origin in triangle region
                if (abc.dot(ao) > 0) {
                    d = abc;
                } else {
                    simplex.keep({0, 2, 1});
                    d = -abc;
                }
            }
        }
        return false;
    }

    // Tetrahedron: check if origin inside
    const Eigen::Vector3d a = simplex[0].v;
    const Eigen::Vector3d b = simplex[1].v;
    const Eigen::Vector3d c = simplex[2].v;
    const Eigen::Vector3d dd = simplex[3].v;

    const Eigen::Vector3d ab = b - a, ac = c - a, ad = dd - a;
    const Eigen::Vector3d ao = -a;

    const Eigen::Vector3d abc = ab.cross(ac);
    const Eigen::Vector3d acd = ac.cross(ad);
    const Eigen::Vector3d adb = ad.cross(ab);

    if (abc.dot(ao) > 0) {
        simplex.keep({0, 1, 2});
        d = abc;
        return do_simplex(simplex, d);
    }
    if (acd.dot(ao) > 0) {
        simplex.keep({0, 2, 3});
        d = acd;
        return do_simplex(simplex, d);
    }
    if (adb.dot(ao) > 0) {
        simplex.keep({0, 3, 1});
        d = adb;
        return do_simplex(simplex, d);
    }

    return true;  // origin inside tetrahedron -> intersection
}

/**
 * GJK algorithm: compute signed distance between two convex shapes.
 *
 * Returns GJKResult with:
 *   - intersecting = true if shapes overlap (distance = 0)
 *   - distance = minimum separation distance
 *   - p_A, p_B = closest points on respective shapes
 *   - normal = unit vector from B toward A at the closest points
 *
 * Time complexity: O(1) in practice (bounded iterations for convex shapes).
 */
inline GJKResult gjk_distance(
    const SupportFn& support_A,
    const SupportFn& support_B,
    int max_iterations = 64,
    double tolerance = 1e-8)
{
    Simplex simplex;
    Eigen::Vector3d d(1, 0, 0);  // initial search direction

    // Initial support point
    auto sp = minkowski_support(support_A, support_B, d);
    simplex.push(sp);
    d = -sp.v;

    GJKResult result;
    result.intersecting = false;

    for (int iter = 0; iter < max_iterations; ++iter) {
        if (d.squaredNorm() < 1e-14) {
            // Origin coincides with simplex point -> intersecting
            result.intersecting = true;
            result.distance = 0.0;
            result.p_A = simplex[0].p_A;
            result.p_B = simplex[0].p_B;
            result.normal = Eigen::Vector3d(0, 0, 1);
            return result;
        }

        d.normalize();
        sp = minkowski_support(support_A, support_B, d);

        // Check if new point is beyond the current simplex in direction d
        double proj = sp.v.dot(d);
        if (proj < simplex[0].v.dot(d) + tolerance) {
            // No progress: we've found the closest point
            break;
        }

        simplex.push(sp);

        if (do_simplex(simplex, d)) {
            result.intersecting = true;
            result.distance = 0.0;
            result.p_A = sp.p_A;
            result.p_B = sp.p_B;
            result.normal = Eigen::Vector3d(0, 0, 1);
            return result;
        }
    }

    // Extract closest points from simplex barycentric coordinates
    // For the final simplex (point, edge, or triangle), compute closest point to origin
    if (simplex.size() == 1) {
        result.p_A = simplex[0].p_A;
        result.p_B = simplex[0].p_B;
    } else if (simplex.size() == 2) {
        // Closest point on line segment to origin
        const Eigen::Vector3d ab = simplex[1].v - simplex[0].v;
        double t = -simplex[0].v.dot(ab) / ab.squaredNorm();
        t = std::max(0.0, std::min(1.0, t));
        result.p_A = simplex[0].p_A + t * (simplex[1].p_A - simplex[0].p_A);
        result.p_B = simplex[0].p_B + t * (simplex[1].p_B - simplex[0].p_B);
    } else {
        // Closest point on triangle to origin (barycentric)
        const Eigen::Vector3d a = simplex[0].v, b = simplex[1].v, c = simplex[2].v;
        const Eigen::Vector3d ab = b-a, ac = c-a, ao = -a;
        double u = ab.dot(ab), v = ab.dot(ac), w = ac.dot(ac);
        double p = ab.dot(ao), q = ac.dot(ao);
        double denom = u*w - v*v;

        double s = (w*p - v*q) / denom;
        double t2 = (u*q - v*p) / denom;
        s = std::max(0.0, std::min(1.0, s));
        t2 = std::max(0.0, std::min(1.0 - s, t2));

        result.p_A = simplex[0].p_A + s*(simplex[1].p_A-simplex[0].p_A)
                                     + t2*(simplex[2].p_A-simplex[0].p_A);
        result.p_B = simplex[0].p_B + s*(simplex[1].p_B-simplex[0].p_B)
                                     + t2*(simplex[2].p_B-simplex[0].p_B);
    }

    Eigen::Vector3d diff = result.p_A - result.p_B;
    result.distance = diff.norm();
    result.normal = (result.distance > 1e-10) ? diff / result.distance
                                               : Eigen::Vector3d(0,0,1);
    result.intersecting = false;
    return result;
}

// ============================================================
// Common support function factories
// ============================================================

/**
 * Support function for a sphere.
 */
inline SupportFn sphere_support(const Eigen::Vector3d& center, double radius) {
    return [center, radius](const Eigen::Vector3d& d) -> Eigen::Vector3d {
        if (d.squaredNorm() < 1e-14) return center;
        return center + radius * d.normalized();
    };
}

/**
 * Support function for an axis-aligned box [min, max].
 */
inline SupportFn box_support(const Eigen::Vector3d& box_min,
                              const Eigen::Vector3d& box_max) {
    return [box_min, box_max](const Eigen::Vector3d& d) -> Eigen::Vector3d {
        return {d(0) >= 0 ? box_max(0) : box_min(0),
                d(1) >= 0 ? box_max(1) : box_min(1),
                d(2) >= 0 ? box_max(2) : box_min(2)};
    };
}

/**
 * Support function for a capsule (cylinder with hemispherical end caps).
 * center: midpoint, axis: unit direction, radius: radius, half_len: half length of cylinder.
 */
inline SupportFn capsule_support(const Eigen::Vector3d& center,
                                  const Eigen::Vector3d& axis,
                                  double radius,
                                  double half_len) {
    return [center, axis, radius, half_len](const Eigen::Vector3d& d) -> Eigen::Vector3d {
        // Project d onto axis, pick endpoint
        double proj = d.dot(axis);
        Eigen::Vector3d tip = center + (proj >= 0 ? half_len : -half_len) * axis;
        // Add sphere support at tip
        if (d.squaredNorm() < 1e-14) return tip;
        return tip + radius * d.normalized();
    };
}

/**
 * Support function for the convex hull of two convex shapes A and B.
 * Used for swept-volume collision checking (Eq. 22 in Schulman et al.):
 *   s_{conv(A,B)}(v) = s_A(v)  if s_A(v)·v > s_B(v)·v
 *                    = s_B(v)  otherwise
 */
inline SupportFn convex_hull_support(const SupportFn& support_A,
                                      const SupportFn& support_B) {
    return [support_A, support_B](const Eigen::Vector3d& d) -> Eigen::Vector3d {
        const Eigen::Vector3d pA = support_A(d);
        const Eigen::Vector3d pB = support_B(d);
        return (pA.dot(d) > pB.dot(d)) ? pA : pB;
    };
}

} // namespace trajopt
