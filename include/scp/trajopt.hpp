#pragma once
/**
 * trajopt.hpp - Trajectory Optimization Problem Formulation
 *
 * Combines the SQP solver, collision checker, and robot kinematics into a
 * complete trajectory optimization pipeline, following Section III-IV of
 * Schulman et al.
 *
 * The trajectory is represented as a sequence of T joint configurations:
 *   x = [q_1; q_2; ...; q_T]  in R^{T*K}
 * where K = number of joints.
 *
 * Objective (Eq. 2): minimize sum of squared displacements (path length)
 *   f(x) = sum_{t=1}^{T-1} ||q_{t+1} - q_t||^2
 *
 * Constraints:
 *   - Collision avoidance: sd(A_i, O_j) >= d_safe (continuous-time)
 *   - Joint limits: q_min <= q_t <= q_max
 *   - (Optional) End-effector pose: h(q_T) = 0
 */

#include "scp/solver.hpp"
#include "collision/signed_distance.hpp"
#include "robot/arm.hpp"
#include <vector>
#include <optional>
#include <stdexcept>

namespace trajopt {

/**
 * Parameters for the trajectory optimization.
 */
struct TrajOptParams {
    int n_timesteps = 11;           // number of waypoints T (including start and end)
    double d_safe   = 0.02;         // collision safety margin [m]
    double d_check  = 0.3;          // broad-phase distance threshold [m]
    bool use_swept_volume = true;   // use continuous-time collision checking
    bool enforce_joint_limits = true;
    SQPParams scp_params;

    TrajOptParams() {
        scp_params.verbose = true;
        scp_params.max_penalty_iter = 5;
        scp_params.max_convexify_iter = 20;
    }
};

/**
 * Result of trajectory optimization.
 */
struct TrajOptResult {
    bool success;
    bool collision_free;
    std::vector<Eigen::VectorXd> trajectory;  // q_0, q_1, ..., q_{T-1}
    double path_length;
    double max_collision_penetration;
    std::string message;
};

/**
 * TrajOptProblem: assembles cost functions and constraints for the SQP solver.
 *
 * Usage:
 *   TrajOptProblem prob(arm, checker, params);
 *   prob.set_start(q_start);
 *   prob.set_goal(q_goal);
 *   auto result = prob.solve(init_trajectory);
 */
class TrajOptProblem {
public:
    TrajOptProblem(const RobotArm& arm,
                   CollisionChecker& checker,
                   TrajOptParams params = TrajOptParams{})
        : arm_(arm), checker_(checker), params_(params),
          T_(params.n_timesteps), K_(arm.n_joints())
    {
        n_vars_ = T_ * K_;
    }

    void set_start(const Eigen::VectorXd& q_start) { q_start_ = q_start; }
    void set_goal(const Eigen::VectorXd& q_goal)   { q_goal_  = q_goal;  }

    /**
     * Set an end-effector pose target (optional).
     * Adds equality constraint: pose_error(T_target, FK(q_T)) = 0
     */
    void set_ee_target(const SE3& T_target) {
        T_ee_target_ = T_target;
    }

    /**
     * Solve the trajectory optimization.
     *
     * init_traj: initial trajectory (T x K matrix). If not provided,
     *            uses linear interpolation from start to goal.
     */
    TrajOptResult solve(
        std::optional<std::vector<Eigen::VectorXd>> init_traj = std::nullopt)
    {
        if (!q_start_ || !q_goal_) {
            throw std::runtime_error("Must set start and goal before solving.");
        }

        // --- Build initial trajectory ---
        std::vector<Eigen::VectorXd> traj;
        if (init_traj) {
            traj = *init_traj;
        } else {
            traj = linear_interpolate(*q_start_, *q_goal_, T_);
        }

        // Flatten trajectory to optimization vector x
        Eigen::VectorXd x = flatten(traj);

        // --- Build cost functions ---
        std::vector<CostFn> costs;
        costs.push_back(path_length_cost());

        // --- Build constraints ---
        std::vector<ConstraintFn> constraints;

        // Start and goal constraints (equality)
        constraints.push_back(start_constraint());
        constraints.push_back(goal_constraint());

        // Joint limits (inequality)
        if (params_.enforce_joint_limits) {
            constraints.push_back(joint_limit_constraint());
        }

        // End-effector target (equality, optional)
        if (T_ee_target_) {
            constraints.push_back(ee_pose_constraint());
        }

        // Collision constraints - rebuilt at each SQP iteration
        // (handled specially below via repeated linearization)

        // Build joint limit box bounds for the SQP solver
        auto [x_lb, x_ub] = build_joint_limit_boxes();

        // --- Run SQP with collision re-linearization ---
        TrajOptResult result = run_scp_with_collision(
            x, costs, constraints, x_lb, x_ub);

        return result;
    }

private:
    const RobotArm& arm_;
    CollisionChecker& checker_;
    TrajOptParams params_;
    int T_, K_, n_vars_;

    std::optional<Eigen::VectorXd> q_start_, q_goal_;
    std::optional<SE3> T_ee_target_;

    // ---- Flatten/unflatten trajectory ----

    Eigen::VectorXd flatten(const std::vector<Eigen::VectorXd>& traj) const {
        Eigen::VectorXd x(n_vars_);
        for (int t = 0; t < T_; ++t)
            x.segment(t * K_, K_) = traj[t];
        return x;
    }

    std::vector<Eigen::VectorXd> unflatten(const Eigen::VectorXd& x) const {
        std::vector<Eigen::VectorXd> traj(T_);
        for (int t = 0; t < T_; ++t)
            traj[t] = x.segment(t * K_, K_);
        return traj;
    }

    Eigen::VectorXd q_at(const Eigen::VectorXd& x, int t) const {
        return x.segment(t * K_, K_);
    }

    // ---- Linear interpolation ----

    static std::vector<Eigen::VectorXd> linear_interpolate(
        const Eigen::VectorXd& q0, const Eigen::VectorXd& q1, int T)
    {
        std::vector<Eigen::VectorXd> traj(T);
        for (int t = 0; t < T; ++t) {
            double alpha = static_cast<double>(t) / (T - 1);
            traj[t] = (1.0 - alpha) * q0 + alpha * q1;
        }
        return traj;
    }

    // ---- Cost: path length (sum of squared displacements, Eq. 2) ----

    CostFn path_length_cost() const {
        return CostFn{
            // Value: f(x) = sum ||q_{t+1} - q_t||^2
            [this](const Eigen::VectorXd& x) -> double {
                double total = 0.0;
                for (int t = 0; t < T_ - 1; ++t) {
                    Eigen::VectorXd dq = q_at(x, t+1) - q_at(x, t);
                    total += dq.squaredNorm();
                }
                return total;
            },
            // Gradient w.r.t. x
            [this](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                Eigen::VectorXd g = Eigen::VectorXd::Zero(n_vars_);
                for (int t = 0; t < T_ - 1; ++t) {
                    Eigen::VectorXd dq = q_at(x, t+1) - q_at(x, t);
                    g.segment(t * K_, K_)     -= 2.0 * dq;
                    g.segment((t+1) * K_, K_) += 2.0 * dq;
                }
                return g;
            }
        };
    }

    // ---- Constraint: start configuration fixed ----

    ConstraintFn start_constraint() const {
        return ConstraintFn{
            ConstraintFn::EQUALITY,
            "start",
            [this](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                return q_at(x, 0) - *q_start_;
            },
            [this](const Eigen::VectorXd&) -> Eigen::MatrixXd {
                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(K_, n_vars_);
                J.leftCols(K_) = Eigen::MatrixXd::Identity(K_, K_);
                return J;
            },
            K_
        };
    }

    // ---- Constraint: goal configuration fixed ----

    ConstraintFn goal_constraint() const {
        return ConstraintFn{
            ConstraintFn::EQUALITY,
            "goal",
            [this](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                return q_at(x, T_-1) - *q_goal_;
            },
            [this](const Eigen::VectorXd&) -> Eigen::MatrixXd {
                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(K_, n_vars_);
                J.rightCols(K_) = Eigen::MatrixXd::Identity(K_, K_);
                return J;
            },
            K_
        };
    }

    // ---- Constraint: joint limits ----

    ConstraintFn joint_limit_constraint() const {
        return ConstraintFn{
            ConstraintFn::INEQUALITY,
            "joint_limits",
            [this](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                // g_i(x) = q_i - q_max_i <= 0  and  -q_i + q_min_i <= 0
                Eigen::VectorXd g(2 * n_vars_);
                for (int t = 0; t < T_; ++t) {
                    for (int k = 0; k < K_; ++k) {
                        double q = x(t*K_ + k);
                        double qmin = arm_.links()[k].dh.q_min;
                        double qmax = arm_.links()[k].dh.q_max;
                        g(2*(t*K_+k))   = q - qmax;   // <= 0
                        g(2*(t*K_+k)+1) = qmin - q;   // <= 0
                    }
                }
                return g;
            },
            [this](const Eigen::VectorXd&) -> Eigen::MatrixXd {
                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2*n_vars_, n_vars_);
                for (int i = 0; i < n_vars_; ++i) {
                    J(2*i,   i) =  1.0;
                    J(2*i+1, i) = -1.0;
                }
                return J;
            },
            2 * n_vars_
        };
    }

    // ---- Constraint: end-effector pose ----

    ConstraintFn ee_pose_constraint() const {
        SE3 T_tgt = *T_ee_target_;
        return ConstraintFn{
            ConstraintFn::EQUALITY,
            "ee_pose",
            [this, T_tgt](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                Eigen::VectorXd q_T = q_at(x, T_-1);
                SE3 T_cur = arm_.end_effector_pose(q_T);
                return pose_error(T_tgt, T_cur);  // 6-vector
            },
            [this, T_tgt](const Eigen::VectorXd& x) -> Eigen::MatrixXd {
                // J_pose = [J_ee for last timestep; zeros for others]
                Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, n_vars_);
                Eigen::VectorXd q_T = q_at(x, T_-1);
                Eigen::MatrixXd J_ee = arm_.jacobian(q_T, K_, Eigen::Vector3d::Zero());
                J.rightCols(K_) = J_ee;
                return J;
            },
            6
        };
    }

    // ---- Joint limit box bounds ----

    std::pair<Eigen::VectorXd, Eigen::VectorXd> build_joint_limit_boxes() const {
        Eigen::VectorXd lb(n_vars_), ub(n_vars_);
        for (int t = 0; t < T_; ++t) {
            for (int k = 0; k < K_; ++k) {
                lb(t*K_+k) = arm_.links()[k].dh.q_min;
                ub(t*K_+k) = arm_.links()[k].dh.q_max;
            }
        }
        return {lb, ub};
    }

    // ---- Main SQP loop with collision re-linearization ----
    // At each SQP iteration, we re-query the collision checker and
    // add new linearized collision constraints to the QP.

    TrajOptResult run_scp_with_collision(
        Eigen::VectorXd& x,
        const std::vector<CostFn>& base_costs,
        const std::vector<ConstraintFn>& base_constraints,
        const Eigen::VectorXd& x_lb,
        const Eigen::VectorXd& x_ub)
    {
        auto& scp_params = params_.scp_params;
        double mu = scp_params.mu_0;
        double s  = scp_params.s_0;
        int total_qp = 0;

        for (int penalty_iter = 0; penalty_iter < scp_params.max_penalty_iter; ++penalty_iter) {
            if (scp_params.verbose) {
                std::cout << "\n[TrajOpt] Penalty iter " << penalty_iter
                          << ", mu = " << mu << std::endl;
            }

            for (int conv_iter = 0; conv_iter < scp_params.max_convexify_iter; ++conv_iter) {
                // Re-linearize collision constraints at current trajectory
                auto all_constraints = base_constraints;
                auto collision_constraints = build_collision_constraints(x, mu);
                all_constraints.insert(all_constraints.end(),
                                       collision_constraints.begin(),
                                       collision_constraints.end());

                // Evaluate merit
                SCPSolver tmp_solver(scp_params);
                double merit_x = evaluate_merit(x, base_costs, all_constraints, mu);

                // Build and solve QP (one inner step)
                QPProblem qp = build_qp_inner(x, base_costs, all_constraints, mu, s, x_lb, x_ub);
                auto qp_result = SimpleQPSolver::solve(qp);
                ++total_qp;

                if (!qp_result.success) break;

                Eigen::VectorXd dx = qp_result.x.head(n_vars_);
                Eigen::VectorXd x_new = x + dx;
                x_new = x_new.cwiseMax(x_lb).cwiseMin(x_ub);

                // Re-evaluate collision at new point for merit
                auto coll_new = build_collision_constraints(x_new, mu);
                all_constraints = base_constraints;
                all_constraints.insert(all_constraints.end(), coll_new.begin(), coll_new.end());
                double merit_new = evaluate_merit(x_new, base_costs, all_constraints, mu);

                double improve = merit_x - merit_new;
                double dx_norm = dx.norm();

                if (scp_params.verbose) {
                    std::cout << "  [iter " << conv_iter << "] merit: "
                              << merit_x << " -> " << merit_new
                              << " (improve=" << improve
                              << ", |dx|=" << dx_norm << ")" << std::endl;
                }

                if (improve > 0) {
                    x = x_new;
                    s = std::min(s * scp_params.tau_plus, 0.5);
                } else {
                    s *= scp_params.tau_minus;
                }

                if (dx_norm < scp_params.x_tol) break;
            }

            // Check collision violation
            double coll_viol = max_collision_violation(x);
            if (scp_params.verbose)
                std::cout << "[TrajOpt] Max collision violation: " << coll_viol << std::endl;

            if (coll_viol < params_.d_safe * 0.1) break;
            mu *= scp_params.k_penalty;
        }

        // --- Build result ---
        TrajOptResult result;
        result.trajectory = unflatten(x);
        result.path_length = compute_path_length(result.trajectory);
        result.max_collision_penetration = max_collision_violation(x);
        result.collision_free = result.max_collision_penetration < params_.d_safe;
        result.success = true;
        result.message = result.collision_free ? "Collision-free trajectory found"
                                               : "Trajectory has remaining collisions";
        return result;
    }

    // ---- Collision constraints as ConstraintFn objects ----
    // These are re-built at every SQP iteration (re-linearization).

    std::vector<ConstraintFn> build_collision_constraints(
        const Eigen::VectorXd& x, double /*mu*/) const
    {
        std::vector<ConstraintFn> ccs;

        for (int t = 0; t < T_; ++t) {
            Eigen::VectorXd q = q_at(x, t);
            auto cd = checker_.check(q, params_.d_check);
            auto lc = linearize_collisions(arm_, q, cd, params_.d_safe);

            for (const auto& l : lc) {
                // Capture gradient and sd_0 by value for the lambda
                Eigen::VectorXd g = l.gradient;    // size K_
                double sd0 = l.sd_0;
                double dsafe = l.d_safe;
                int t_cap = t;
                int K = K_;
                int n = n_vars_;

                ccs.push_back(ConstraintFn{
                    ConstraintFn::INEQUALITY,
                    "collision_t" + std::to_string(t),
                    // g(x) = dsafe - (sd0 + g^T * dq) <= 0
                    // where dq = q(t) - q0(t)
                    [g, sd0, dsafe, t_cap, K, n](const Eigen::VectorXd& x_new) -> Eigen::VectorXd {
                        // This is already linearized, so value is constant
                        // (the Jacobian does the heavy lifting)
                        return Eigen::VectorXd::Constant(1, std::max(dsafe - sd0, 0.0));
                    },
                    [g, t_cap, K, n](const Eigen::VectorXd&) -> Eigen::MatrixXd {
                        // Jacobian: -g^T placed at the t-th block
                        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(1, n);
                        J.block(0, t_cap*K, 1, K) = -g.transpose();
                        return J;
                    },
                    1
                });
            }
        }

        return ccs;
    }

    // ---- Merit function ----
    double evaluate_merit(const Eigen::VectorXd& x,
                          const std::vector<CostFn>& costs,
                          const std::vector<ConstraintFn>& constraints,
                          double mu) const {
        double f = 0.0;
        for (const auto& c : costs) f += c.value(x);
        for (const auto& c : constraints) {
            Eigen::VectorXd val = c.value(x);
            if (c.type == ConstraintFn::INEQUALITY) {
                for (int i = 0; i < val.size(); ++i)
                    f += mu * std::max(val(i), 0.0);
            } else {
                f += mu * val.cwiseAbs().sum();
            }
        }
        return f;
    }

    // ---- Build QP for one inner iteration ----
    QPProblem build_qp_inner(
        const Eigen::VectorXd& x,
        const std::vector<CostFn>& costs,
        const std::vector<ConstraintFn>& constraints,
        double mu, double s,
        const Eigen::VectorXd& x_lb_g,
        const Eigen::VectorXd& x_ub_g) const
    {
        int n_c = 0;
        for (const auto& c : constraints) n_c += c.dim;

        QPProblem qp(n_vars_);
        qp.P = 1e-4 * Eigen::MatrixXd::Identity(n_vars_, n_vars_);
        qp.q = Eigen::VectorXd::Zero(n_vars_);
        for (const auto& fn : costs)
            qp.q += fn.gradient(x);

        // Trust region
        qp.x_lb = (x_lb_g - x).cwiseMax(Eigen::VectorXd::Constant(n_vars_, -s));
        qp.x_ub = (x_ub_g - x).cwiseMin(Eigen::VectorXd::Constant(n_vars_, +s));

        // Collision and other constraints as penalty in objective
        for (const auto& c : constraints) {
            Eigen::VectorXd val = c.value(x);
            Eigen::MatrixXd jac = c.jacobian(x);

            if (c.type == ConstraintFn::INEQUALITY) {
                for (int i = 0; i < c.dim; ++i) {
                    if (val(i) > -0.01) {  // active or violated
                        qp.q += mu * jac.row(i).transpose();
                    }
                }
            } else {
                for (int i = 0; i < c.dim; ++i) {
                    double sign = (val(i) >= 0) ? 1.0 : -1.0;
                    qp.q += mu * sign * jac.row(i).transpose();
                }
            }
        }

        return qp;
    }

    // ---- Collision violation metric ----
    double max_collision_violation(const Eigen::VectorXd& x) const {
        double max_viol = 0.0;
        for (int t = 0; t < T_; ++t) {
            auto cd = checker_.check(q_at(x, t), params_.d_check);
            for (const auto& c : cd) {
                max_viol = std::max(max_viol, params_.d_safe - c.signed_dist);
            }
        }
        return std::max(max_viol, 0.0);
    }

    // ---- Path length metric ----
    double compute_path_length(const std::vector<Eigen::VectorXd>& traj) const {
        double total = 0.0;
        for (int t = 0; t < T_ - 1; ++t)
            total += (traj[t+1] - traj[t]).norm();
        return total;
    }
};

} // namespace trajopt
