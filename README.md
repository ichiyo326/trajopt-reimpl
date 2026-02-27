# TrajOpt Reimplementation

A ground-up C++17 reimplementation of the trajectory optimization algorithm from:

> Schulman et al., *"Motion Planning with Sequential Convex Optimization and Convex Collision Checking"*, IJRR 2014.

Written as a study project to understand optimization-based motion planning from first principles.

---

## Motivation

My research background is in **computational phase retrieval** (inverse problems in optics), where the core challenge is recovering a signal from phaseless measurements — a non-convex optimization problem. Reading the TrajOpt paper, I was struck by the structural similarities:

| Phase Retrieval | TrajOpt |
|---|---|
| Signal in C^n | Trajectory in R^{T×K} |
| Measurement consistency (equality) | Kinematic constraints |
| Support constraint (inequality) | Joint limits, collision avoidance |
| Alternating projections (HIO, RAAR) | SQP inner loop |
| Step size / relaxation parameter | Trust region radius s |
| ℓ1 sparsity regularization | ℓ1 collision penalty |
| Local linearization of phase | Signed distance Jacobian |

Both problems are non-convex and solved via **iterative convex approximation**. The ℓ1 penalty here is an *exact* penalty method — analogous to ℓ1 regularization in compressed sensing, which drives sparse solutions rather than merely penalizing them.

This connection motivated me to implement TrajOpt from scratch to deepen my understanding of robotics optimization.

---

## What's Implemented

### 1. SE(3) Lie Group Math (`include/robot/se3.hpp`)

Full implementation of the Special Euclidean group operations required for trajectory optimization on manifolds (Section III-A of the paper):

- `hat` / `vee` operators: R^6 ↔ se(3)
- `exp_se3` / `log_se3`: Rodrigues' formula (exact, not approximation)
- `retract` / `local_diff`: local parameterization for SQP updates
- `pose_error`: 6-vector error for end-effector constraints (Eq. 29-32)

```cpp
// Retract: update pose X by incremental twist delta (Eq. 7)
//   X_new = X * exp(delta^)
SE3 X_new = retract(X, delta);

// Local difference: what twist takes X to Y?
Twist delta = local_diff(X, Y);  // = log(X^{-1} * Y)
```

### 2. GJK Collision Detection (`include/collision/gjk.hpp`)

From-scratch implementation of the **Gilbert-Johnson-Keerthi algorithm** using support function representations (Section IV):

- Distance queries between any pair of convex shapes
- Support functions: sphere, box, capsule
- **Convex hull support** for swept-volume queries (Eq. 22):
  ```
  s_{conv(A,B)}(v) = s_A(v)  if s_A(v)·v > s_B(v)·v
                   = s_B(v)  otherwise
  ```
- No explicit mesh representation required

### 3. Signed Distance Linearization (`include/collision/signed_distance.hpp`)

The key step that makes collision avoidance tractable for QP solvers (Eq. 18-19):

```
sd_AB(x) ≈ sd_AB(x0) + n̂ᵀ Jₚₐ(x0) (x - x0)
```

where `Jₚₐ` is the position Jacobian of the closest point on the robot link.

- Discrete-time collision check (per-waypoint)
- **Continuous-time swept volume** (Eq. 20-27): convex hull of link shapes at consecutive timesteps. Prevents "tunneling" through thin obstacles.
- Hinge loss penalty: `|d_safe - sd|₊`

### 4. Robot Arm Kinematics (`include/robot/arm.hpp`)

N-link serial chain with:
- Standard DH parameterization
- Forward kinematics via homogeneous transforms
- **Geometric Jacobian** at any link point (analytical, not numerical)
- Joint limit management
- Factory functions: 7-DOF Panda-like arm, 3-DOF planar arm

### 5. SQP Solver (`include/scp/solver.hpp`)

Implementation of Algorithm 1 from the paper — ℓ1 penalty method for sequential convex optimization:

```
for PenaltyIteration:                  # outer: increase μ until feasible
    for ConvexifyIteration:            # inner: build + solve QP
        for TrustRegionIteration:      # line search: accept/reject step
            QP ← linearize(f, g, h)
            Δx ← solve(QP)
            if TrueImprove/ModelImprove > c:
                accept step, expand trust region
            else:
                shrink trust region
    if violated: μ ← k·μ
```

Key design choices:
- **ℓ1 (not ℓ2) penalty**: exact penalty — constraint satisfaction is guaranteed in the limit, not merely encouraged. Analogous to the L1 magic in compressed sensing.
- **Trust region as box constraint**: avoids the linearization accuracy region being exceeded
- Slack variable reformulation for non-smooth ℓ1 terms (Eq. 3-4)

### 6. Trajectory Optimization Problem (`include/scp/trajopt.hpp`)

Assembles all components into a complete motion planner:

- Path length objective (Eq. 2)
- Start/goal constraints (equality)
- Joint limits (inequality)
- End-effector pose constraint (optional, uses SE(3) pose error)
- Collision constraints (re-linearized at each SQP iteration)

---

## Building

```bash
# Prerequisites: Eigen3, CMake >= 3.16, C++17 compiler
sudo apt install libeigen3-dev cmake   # Ubuntu/Debian
brew install eigen cmake               # macOS

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run tests

```bash
cd build
ctest --output-on-failure
```

Expected output:
```
Test 1: se3_tests   ... 10/10 passed
Test 2: gjk_tests   ...  8/8  passed
Test 3: scp_tests   ...  5/5  passed
```

### Run 2D demo

```bash
./arm_2d_demo
python3 ../python/visualize.py trajectory_data.json
```

---

## Demo: 3-Link Planar Arm

A 3-DOF planar robot arm avoids two box obstacles and one sphere obstacle.
The straight-line trajectory initialization is in collision; TrajOpt finds a collision-free path.

```
Initial trajectory:              Optimized trajectory:
         ██████                           
   ---/--██████--→          ---/----\---→
  /      ██████              /        \
━━       ██████          ━━             ━━
         ██████
```

Output files after running the demo:
- `trajopt_result.png` — side-by-side comparison
- `trajopt_animation.gif` — animated arm motion
- `joint_trajectories.png` — per-joint angle plots

---

## Architecture

```
include/
├── robot/
│   ├── se3.hpp          # SE(3) Lie group: hat, vee, exp, log, retract
│   └── arm.hpp          # N-link arm: FK, geometric Jacobian, DH params
├── collision/
│   ├── gjk.hpp          # GJK distance algorithm, support functions
│   └── signed_distance.hpp  # Collision cost + linearization
└── scp/
    ├── solver.hpp       # SQP solver: l1 penalty, trust region
    └── trajopt.hpp      # Full trajectory optimization problem

demo/arm_2d/
    arm_2d_demo.cpp      # 3-DOF planar demo

python/
    visualize.py         # Matplotlib visualization + animation

tests/
    test_se3.cpp         # SE(3) math: exp/log roundtrip, composition
    test_gjk.cpp         # GJK: sphere, box, capsule, swept volume
    test_scp.cpp         # SQP: unconstrained, equality, inequality
```

---

## Limitations and Known Issues

This is a study implementation, not production-grade software:

1. **QP solver**: uses projected gradient descent instead of a proper interior-point solver (e.g., OSQP). Convergence is slower on large problems.

2. **EPA not implemented**: for deeply penetrating objects, GJK returns a rough penetration estimate. A full TrajOpt would use the Expanding Polytope Algorithm (EPA) for accurate penetration depth.

3. **Self-collision**: the collision checker handles robot-obstacle collisions but not robot self-collisions (would need link pair filtering).

4. **Local optima**: as discussed in Section IX of the paper, trajectory optimization is sensitive to initialization. Multiple random restarts (as in the `TrajOpt-Multi` condition in Table I) significantly improve success rate.

5. **No mesh support**: only primitive shapes (sphere, box, capsule). A production system would use convex decomposition of CAD meshes (HACD).

---

## What I Learned

- **SE(3) vs Euler angles**: local Lie algebra parameterization avoids gimbal lock and provides a faithful local geometry, which is crucial for the linearizations to be accurate. The analogy in signal processing is using complex exponentials instead of phase unwrapping.

- **Convex hull as swept volume**: the support function trick (Eq. 22) is elegant — it gives continuous-time safety without explicitly computing the swept volume. This is O(1) overhead over discrete checking.

- **ℓ1 vs ℓ2 penalty**: ℓ2 ("soft") penalties always leave a residual constraint violation at finite μ. ℓ1 ("exact") penalties achieve zero violation for μ large enough. The trade-off: ℓ1 creates non-smooth subproblems (handled via slack variables).

- **Why SQP beats gradient descent for this problem**: gradient descent with collision penalties can get stuck when obstacle normals push adjacent waypoints in opposing directions (Figure 15 in the paper). SQP's trust region + feasibility restoration provides a cleaner path out of such configurations.

---

## References

- Schulman, J. et al. (2014). Motion Planning with Sequential Convex Optimization and Convex Collision Checking. *IJRR*.
- Gilbert, E., Johnson, D., Keerthi, S. (1988). A Fast Procedure for Computing the Distance between Complex Objects in Three-Dimensional Space. *IEEE Journal of Robotics and Automation*.
- Murray, R., Li, Z., Sastry, S. (1994). *A Mathematical Introduction to Robotic Manipulation*.
- Blanco, J. (2010). A Tutorial on SE(3) Transformation Parameterizations and On-Manifold Optimization. Technical Report, University of Malaga.
- Nocedal, J., Wright, S. (1999). *Numerical Optimization*. Springer.
