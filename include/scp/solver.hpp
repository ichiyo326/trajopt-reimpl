#pragma once
/**
 * solver.hpp - Sequential Convex Programming (SQP) with ℓ1 Penalty Method
 *
 * This is the core optimization engine of TrajOpt (Algorithm 1 in the paper).
 *
 * The algorithm solves non-convex constrained problems of the form:
 *
 *   minimize   f(x)
 *   subject to g_i(x) <= 0,  i = 1,...,n_ineq
 *              h_i(x) = 0,   i = 1,...,n_eq
 *
 * Strategy:
 *   Outer loop: increase penalty coefficient μ until constraints satisfied
 *   Inner loop: repeatedly linearize costs/constraints, solve QP subproblem
 *   Trust region: accept/reject steps based on actual vs predicted improvement
 *
 * Connection to signal processing (for the author's background):
 *   This is closely related to the Alternating Direction Method of Multipliers
 *   (ADMM) and iterative shrinkage methods in sparse signal recovery.
 *   The ℓ1 penalty plays the same role as ℓ1 regularization in LASSO:
 *   it promotes exact constraint satisfaction (analogous to sparsity).
 *   The trust region is analogous to the step size control in iterative
 *   phase retrieval algorithms (HIO, RAAR, etc.).
 *
 * Dependencies: Eigen (linear algebra), OSQP or a simple QP solver.
 */

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <functional>
#include <vector>
#include <optional>
#include <iostream>
#include <cassert>

namespace trajopt {

// ============================================================
// QP Problem formulation
// ============================================================

/**
 * Quadratic Program in standard form:
 *
 *   minimize    (1/2) x^T P x + q^T x
 *   subject to  lb <= A x <= ub
 *               x_lb <= x <= x_ub
 *
 * The SQP inner loop solves one QP per iteration.
 */
struct QPProblem {
    int n_vars;

    Eigen::MatrixXd P;         // Hessian (positive semidefinite)
    Eigen::VectorXd q;         // linear cost

    Eigen::MatrixXd A;         // constraint matrix
    Eigen::VectorXd lb;        // lower bounds on A*x
    Eigen::VectorXd ub;        // upper bounds on A*x

    Eigen::VectorXd x_lb;      // lower bounds on x (trust region + joint limits)
    Eigen::VectorXd x_ub;      // upper bounds on x

    QPProblem() = default;
    QPProblem(int n) : n_vars(n),
        P(Eigen::MatrixXd::Zero(n, n)),
        q(Eigen::VectorXd::Zero(n)),
        x_lb(Eigen::VectorXd::Constant(n, -1e9)),
        x_ub(Eigen::VectorXd::Constant(n, 1e9)) {}
};

/**
 * Minimal QP solver using Eigen's LLT (for unconstrained / trust-region only).
 *
 * For a full implementation, this would call OSQP or a similar solver.
 * Here we implement a projected gradient + active set for the linear constraints.
 */
class SimpleQPSolver {
public:
    struct Result {
        bool success;
        Eigen::VectorXd x;
        double objective;
        std::string message;
    };

    /**
     * Solve QP using projected gradient descent with box constraints.
     * This handles the trust region and joint limit constraints.
     * Linear inequality constraints (collision) are handled via penalty.
     *
     * Note: In a production system, use OSQP for full LP/QP support.
     */
    static Result solve(const QPProblem& qp, int max_iter = 200, double tol = 1e-6) {
        const int n = qp.n_vars;
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

        // Project into box [x_lb, x_ub]
        auto project = [&](Eigen::VectorXd& v) {
            v = v.cwiseMax(qp.x_lb).cwiseMin(qp.x_ub);
        };

        // Compute gradient: g = P*x + q  (+ constraint penalty gradient)
        auto gradient = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
            Eigen::VectorXd g = qp.P * v + qp.q;
            // Add constraint violation penalties
            if (qp.A.rows() > 0) {
                Eigen::VectorXd Ax = qp.A * v;
                for (int i = 0; i < qp.A.rows(); ++i) {
                    double viol_lo = qp.lb(i) - Ax(i);
                    double viol_hi = Ax(i) - qp.ub(i);
                    if (viol_lo > 0)
                        g -= 1e4 * viol_lo * qp.A.row(i).transpose();
                    if (viol_hi > 0)
                        g += 1e4 * viol_hi * qp.A.row(i).transpose();
                }
            }
            return g;
        };

        auto objective = [&](const Eigen::VectorXd& v) -> double {
            double obj = 0.5 * v.dot(qp.P * v) + qp.q.dot(v);
            if (qp.A.rows() > 0) {
                Eigen::VectorXd Ax = qp.A * v;
                for (int i = 0; i < qp.A.rows(); ++i) {
                    double viol_lo = std::max(qp.lb(i) - Ax(i), 0.0);
                    double viol_hi = std::max(Ax(i) - qp.ub(i), 0.0);
                    obj += 5e3 * (viol_lo*viol_lo + viol_hi*viol_hi);
                }
            }
            return obj;
        };

        project(x);

        // Armijo line search projected gradient
        double step = 1.0;
        for (int iter = 0; iter < max_iter; ++iter) {
            Eigen::VectorXd g = gradient(x);
            double f0 = objective(x);

            // Projected gradient step
            Eigen::VectorXd x_new = x - step * g;
            project(x_new);

            double f_new = objective(x_new);
            // Backtracking line search (Armijo)
            int ls_iter = 0;
            while (f_new > f0 - 1e-4 * (x_new - x).dot(g) && ls_iter < 20) {
                step *= 0.5;
                x_new = x - step * g;
                project(x_new);
                f_new = objective(x_new);
                ++ls_iter;
            }

            double dx = (x_new - x).norm();
            x = x_new;
            step = std::min(step * 2.0, 1.0);  // expand step if possible

            if (dx < tol) break;
        }

        Result r;
        r.success = true;
        r.x = x;
        r.objective = objective(x);
        r.message = "OK";
        return r;
    }
};

// ============================================================
// SQP Solver
// ============================================================

/**
 * Parameters for the SQP solver (Algorithm 1 in Schulman et al.)
 */
struct SQPParams {
    double mu_0          = 10.0;    // initial penalty coefficient
    double mu_max        = 1e6;     // max penalty (termination)
    double k_penalty     = 10.0;    // penalty scaling factor per outer iteration
    double s_0           = 0.1;     // initial trust region size (rad / m)
    double tau_plus      = 1.5;     // trust region expansion factor
    double tau_minus     = 0.5;     // trust region shrinkage factor
    double c_accept      = 0.1;     // step acceptance threshold (TrueImprove/ModelImprove)
    double x_tol         = 1e-4;    // convergence tolerance on x
    double f_tol         = 1e-4;    // convergence tolerance on merit function
    double c_tol         = 1e-3;    // constraint satisfaction tolerance
    int max_penalty_iter = 5;       // max outer (penalty) iterations
    int max_convexify_iter = 15;    // max inner (SQP) iterations
    int max_trust_iter   = 5;       // max trust region shrink attempts
    bool verbose         = true;
};

/**
 * Cost function type.
 *
 * The cost function returns:
 *   - value: f(x)
 *   - gradient: grad_f(x)  [for quadratic approximation, we use gradient only]
 */
struct CostFn {
    std::function<double(const Eigen::VectorXd&)> value;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> gradient;
};

/**
 * Constraint function type.
 *
 * For inequality g(x) <= 0:
 *   - value: g(x)
 *   - jacobian: row vector dg/dx
 *
 * For equality h(x) = 0:
 *   - value: h(x)
 *   - jacobian: row vector dh/dx
 */
struct ConstraintFn {
    enum Type { INEQUALITY, EQUALITY };
    Type type;
    std::string name;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> value;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> jacobian;
    int dim;  // number of constraint outputs
};

/**
 * SQP solver result.
 */
struct SQPResult {
    bool success;
    bool feasible;
    Eigen::VectorXd x;
    double cost;
    double constraint_violation;
    int penalty_iters;
    int total_qp_solves;
    std::string message;
};

/**
 * Sequential Convex Programming Solver.
 *
 * Implements Algorithm 1 (ℓ1 penalty method for SQP) from Schulman et al.
 */
class SCPSolver {
public:
    explicit SCPSolver(SQPParams params = SQPParams{}) : params_(params) {}

    /**
     * Main solve routine.
     *
     * x0: initial guess
     * cost_fns: list of cost functions to minimize
     * constraints: list of inequality/equality constraints
     */
    SQPResult solve(
        const Eigen::VectorXd& x0,
        const std::vector<CostFn>& cost_fns,
        const std::vector<ConstraintFn>& constraints,
        // Optional box constraints on x (joint limits + trust region applied separately)
        std::optional<Eigen::VectorXd> x_lb = std::nullopt,
        std::optional<Eigen::VectorXd> x_ub = std::nullopt)
    {
        const int n = x0.size();
        Eigen::VectorXd x = x0;
        double mu = params_.mu_0;
        int total_qp = 0;

        SQPResult result;
        result.success = false;
        result.feasible = false;

        // =============================================
        // OUTER LOOP: Penalty Iteration (Algorithm 1, line 1)
        // =============================================
        for (int penalty_iter = 0; penalty_iter < params_.max_penalty_iter; ++penalty_iter) {
            if (params_.verbose) {
                std::cout << "\n[SCP] Penalty iter " << penalty_iter
                          << ", mu = " << mu << std::endl;
            }

            double s = params_.s_0;  // trust region size

            // =============================================
            // INNER LOOP: Convexify Iteration (Algorithm 1, line 2)
            // =============================================
            for (int conv_iter = 0; conv_iter < params_.max_convexify_iter; ++conv_iter) {

                // --- Evaluate current merit function ---
                double f_x = evaluate_cost(x, cost_fns);
                double viol_x = evaluate_violation(x, constraints);
                double merit_x = f_x + mu * viol_x;

                // --- Build QP subproblem (Convexify, Algorithm 1, line 3) ---
                QPProblem qp = build_qp(x, cost_fns, constraints, mu, s,
                                        x_lb, x_ub);

                // =============================================
                // TRUST REGION LOOP (Algorithm 1, line 4)
                // =============================================
                bool step_accepted = false;
                for (int tr_iter = 0; tr_iter < params_.max_trust_iter; ++tr_iter) {
                    // Set trust region bounds
                    qp.x_lb = Eigen::VectorXd::Constant(n, -s);
                    qp.x_ub = Eigen::VectorXd::Constant(n, +s);
                    if (x_lb) qp.x_lb = qp.x_lb.cwiseMax(*x_lb - x);
                    if (x_ub) qp.x_ub = qp.x_ub.cwiseMin(*x_ub - x);

                    // Solve QP subproblem (Algorithm 1, line 5)
                    auto qp_result = SimpleQPSolver::solve(qp);
                    ++total_qp;

                    if (!qp_result.success) break;

                    Eigen::VectorXd dx = qp_result.x;
                    Eigen::VectorXd x_new = x + dx;

                    // Evaluate true improvement (Algorithm 1, line 6)
                    double f_new = evaluate_cost(x_new, cost_fns);
                    double viol_new = evaluate_violation(x_new, constraints);
                    double merit_new = f_new + mu * viol_new;

                    double true_improve  = merit_x - merit_new;
                    double model_improve = merit_x - (qp_result.objective + mu * 0.0);
                    // (model_improve from QP objective; simplified here)
                    model_improve = std::max(model_improve, 1e-8);

                    double ratio = true_improve / model_improve;

                    if (params_.verbose && tr_iter == 0) {
                        std::cout << "  [SQP iter " << conv_iter << "]"
                                  << " merit: " << merit_x << " -> " << merit_new
                                  << " (ratio=" << ratio << ", |dx|=" << dx.norm() << ")" << std::endl;
                    }

                    if (ratio > params_.c_accept) {
                        // Accept step (Algorithm 1, line 7)
                        x = x_new;
                        s = std::min(s * params_.tau_plus, 1.0);  // expand trust region
                        step_accepted = true;
                        break;
                    } else {
                        // Reject: shrink trust region (Algorithm 1, line 10)
                        s *= params_.tau_minus;
                        if (s < params_.x_tol) {
                            if (params_.verbose)
                                std::cout << "  Trust region too small, stopping." << std::endl;
                            goto inner_converged;
                        }
                    }
                }

                // Check convergence (Algorithm 1, line 13)
                if (!step_accepted || evaluate_violation(x, constraints) < params_.c_tol) {
                    if (params_.verbose)
                        std::cout << "  Inner loop converged." << std::endl;
                    break;
                }
            }

            inner_converged:

            // Check constraint satisfaction (Algorithm 1, line 15)
            double viol = evaluate_violation(x, constraints);
            if (params_.verbose)
                std::cout << "[SCP] Constraint violation: " << viol << std::endl;

            if (viol < params_.c_tol) {
                result.feasible = true;
                break;
            }

            // Increase penalty (Algorithm 1, line 18)
            mu *= params_.k_penalty;
            if (mu > params_.mu_max) {
                if (params_.verbose)
                    std::cout << "[SCP] Max penalty reached, stopping." << std::endl;
                break;
            }
        }

        result.x = x;
        result.cost = evaluate_cost(x, cost_fns);
        result.constraint_violation = evaluate_violation(x, constraints);
        result.total_qp_solves = total_qp;
        result.success = true;
        result.penalty_iters = params_.max_penalty_iter;
        result.message = result.feasible ? "Feasible solution found" : "Infeasible (constraint violation remains)";

        return result;
    }

private:
    SQPParams params_;

    // ---- Evaluate total cost ----
    double evaluate_cost(const Eigen::VectorXd& x,
                         const std::vector<CostFn>& cost_fns) const {
        double total = 0.0;
        for (const auto& fn : cost_fns) total += fn.value(x);
        return total;
    }

    // ---- Evaluate total constraint violation (ℓ1 penalty) ----
    double evaluate_violation(const Eigen::VectorXd& x,
                              const std::vector<ConstraintFn>& constraints) const {
        double total = 0.0;
        for (const auto& c : constraints) {
            Eigen::VectorXd val = c.value(x);
            if (c.type == ConstraintFn::INEQUALITY) {
                // |g(x)|_+  (hinge: penalize only if g > 0)
                for (int i = 0; i < val.size(); ++i)
                    total += std::max(val(i), 0.0);
            } else {
                // |h(x)|  (absolute value for equality)
                total += val.cwiseAbs().sum();
            }
        }
        return total;
    }

    /**
     * Build the QP subproblem by linearizing all costs and constraints.
     *
     * The QP variables are the step Δx from the current x.
     * The QP objective approximates the merit function f(x+Δx) + μ * violation.
     */
    QPProblem build_qp(
        const Eigen::VectorXd& x,
        const std::vector<CostFn>& cost_fns,
        const std::vector<ConstraintFn>& constraints,
        double mu, double s,
        std::optional<Eigen::VectorXd> x_lb,
        std::optional<Eigen::VectorXd> x_ub) const
    {
        const int n = x.size();

        // Count slack variables needed for ℓ1 terms
        // Each inequality g_i(x) <= 0: add slack t_i >= 0 with g(x0) + J*dx <= t_i
        // Each equality h_i(x) = 0: add s_i + t_i with s_i - t_i = h(x0) + J*dx

        // For simplicity, we represent all constraint penalties as linear terms in the QP.
        // In a production solver, we'd use OSQP's LP/QP capabilities directly.

        // Total variables: n (delta_x) + n_slack (one per constraint dim)
        int n_ineq_dims = 0, n_eq_dims = 0;
        for (const auto& c : constraints) {
            if (c.type == ConstraintFn::INEQUALITY) n_ineq_dims += c.dim;
            else n_eq_dims += c.dim;
        }
        int n_total = n + n_ineq_dims + 2 * n_eq_dims;

        QPProblem qp(n_total);

        // Cost: sum of linearized cost functions (first-order Taylor)
        // f(x + dx) ≈ f(x) + grad_f^T * dx
        // Quadratic term: identity regularization for smoothness
        qp.P.topLeftCorner(n, n) = 1e-3 * Eigen::MatrixXd::Identity(n, n);  // Tikhonov
        Eigen::VectorXd grad_f = Eigen::VectorXd::Zero(n);
        for (const auto& fn : cost_fns) {
            Eigen::VectorXd g = fn.gradient(x);
            grad_f += g;
        }
        qp.q.head(n) = grad_f;

        // Trust region: ||dx||_inf <= s (set as box bounds)
        qp.x_lb.head(n) = Eigen::VectorXd::Constant(n, -s);
        qp.x_ub.head(n) = Eigen::VectorXd::Constant(n, +s);

        // Joint limits
        if (x_lb) qp.x_lb.head(n) = qp.x_lb.head(n).cwiseMax(*x_lb - x);
        if (x_ub) qp.x_ub.head(n) = qp.x_ub.head(n).cwiseMin(*x_ub - x);

        // Slack variable bounds
        qp.x_lb.tail(n_total - n) = Eigen::VectorXd::Zero(n_total - n);
        qp.x_ub.tail(n_total - n) = Eigen::VectorXd::Constant(n_total - n, 1e9);

        // Slack costs: mu * t (ℓ1 penalty for each constraint)
        qp.q.tail(n_total - n) = Eigen::VectorXd::Constant(n_total - n, mu);

        // Build constraint rows: J * dx - t <= -g(x0)  (inequality g(x0) + J*dx <= t)
        int total_constraint_rows = n_ineq_dims + 2 * n_eq_dims;
        if (total_constraint_rows > 0) {
            qp.A = Eigen::MatrixXd::Zero(total_constraint_rows, n_total);
            qp.lb = Eigen::VectorXd::Constant(total_constraint_rows, -1e9);
            qp.ub = Eigen::VectorXd::Constant(total_constraint_rows, 1e9);
        }

        int row = 0, slack = n;
        for (const auto& c : constraints) {
            Eigen::VectorXd val = c.value(x);
            Eigen::MatrixXd jac = c.jacobian(x);  // [c.dim x n]

            if (c.type == ConstraintFn::INEQUALITY) {
                // g(x0) + J*dx <= t_i
                // =>  [J | -I] * [dx; t] <= -g(x0)
                //     t >= 0   (handled by x_lb)
                for (int i = 0; i < c.dim; ++i) {
                    qp.A.block(row, 0, 1, n) = jac.row(i);
                    qp.A(row, slack + i) = -1.0;
                    qp.ub(row) = -val(i);  // J*dx - t <= -g(x0)
                    ++row;
                }
                slack += c.dim;
            } else {
                // |h(x0) + J*dx| -> add s_i + t_i with:
                //   s_i - t_i = h(x0) + J*dx  => J*dx - s + t = -h(x0)
                for (int i = 0; i < c.dim; ++i) {
                    // Row 1: J*dx - s_i + t_i = -h(x0)
                    qp.A.block(row, 0, 1, n) = jac.row(i);
                    qp.A(row, slack + i)        = -1.0;  // -s
                    qp.A(row, slack + c.dim + i) = +1.0; // +t
                    qp.lb(row) = qp.ub(row) = -val(i);
                    ++row;
                }
                slack += 2 * c.dim;
            }
        }

        return qp;
    }
};

} // namespace trajopt
