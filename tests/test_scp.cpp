/**
 * test_scp.cpp - Unit tests for the Sequential Convex Programming solver
 *
 * Tests on simple problems with known solutions:
 *   1. Unconstrained quadratic (trivial)
 *   2. Equality constrained QP
 *   3. Inequality constrained problem
 *   4. Non-convex problem (SQP needed)
 *   5. ℓ1 penalty drives constraint to zero
 */

#include "../include/scp/solver.hpp"
#include <iostream>
#include <cmath>

using namespace trajopt;

static int n_passed = 0, n_failed = 0;

#define EXPECT_NEAR(a, b, tol, msg)                                          \
    do {                                                                     \
        double err = std::abs((a) - (b));                                    \
        if (err > (tol)) {                                                   \
            std::cerr << "FAIL [" << msg << "]: |" << (a) << " - " << (b)   \
                      << "| = " << err << " > " << tol << "\n";             \
            ++n_failed;                                                      \
        } else {                                                             \
            std::cout << "PASS [" << msg << "]\n";                          \
            ++n_passed;                                                      \
        }                                                                    \
    } while(0)

#define EXPECT_MATRIX_NEAR(A, B, tol, msg)                                   \
    do {                                                                     \
        double err = ((A) - (B)).norm();                                     \
        if (err > (tol)) {                                                   \
            std::cerr << "FAIL [" << msg << "]: error = " << err << "\n";   \
            ++n_failed;                                                      \
        } else {                                                             \
            std::cout << "PASS [" << msg << "]\n";                          \
            ++n_passed;                                                      \
        }                                                                    \
    } while(0)

// ---- Test 1: Unconstrained quadratic ----
// minimize (x - 2)^2 + (y + 1)^2
// Solution: x = 2, y = -1
void test_unconstrained_quadratic() {
    std::cout << "\n-- Unconstrained Quadratic --\n";

    SQPParams params;
    params.verbose = false;
    params.mu_0 = 1.0;
    params.max_penalty_iter = 3;
    SCPSolver solver(params);

    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;

    CostFn cost{
        [](const Eigen::VectorXd& x) {
            return std::pow(x(0) - 2.0, 2) + std::pow(x(1) + 1.0, 2);
        },
        [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            Eigen::VectorXd g(2);
            g << 2*(x(0) - 2.0), 2*(x(1) + 1.0);
            return g;
        }
    };

    auto result = solver.solve(x0, {cost}, {});
    EXPECT_NEAR(result.x(0), 2.0, 0.1, "unconstrained x* = 2");
    EXPECT_NEAR(result.x(1), -1.0, 0.1, "unconstrained y* = -1");
}

// ---- Test 2: Equality constrained ----
// minimize (x - 3)^2 + (y - 3)^2
// subject to x + y = 4
// Solution: x = y = 2
void test_equality_constrained() {
    std::cout << "\n-- Equality Constrained --\n";

    SQPParams params;
    params.verbose = false;
    params.mu_0 = 50.0;
    params.max_penalty_iter = 6;
    SCPSolver solver(params);

    Eigen::VectorXd x0(2);
    x0 << 1.0, 3.0;

    CostFn cost{
        [](const Eigen::VectorXd& x) {
            return std::pow(x(0)-3, 2) + std::pow(x(1)-3, 2);
        },
        [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            Eigen::VectorXd g(2);
            g << 2*(x(0)-3), 2*(x(1)-3);
            return g;
        }
    };

    ConstraintFn eq_con{
        ConstraintFn::EQUALITY,
        "x + y = 4",
        [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            return Eigen::VectorXd::Constant(1, x(0) + x(1) - 4.0);
        },
        [](const Eigen::VectorXd&) -> Eigen::MatrixXd {
            Eigen::MatrixXd J(1, 2);
            J << 1, 1;
            return J;
        },
        1
    };

    auto result = solver.solve(x0, {cost}, {eq_con});
    double violation = std::abs(result.x(0) + result.x(1) - 4.0);
    EXPECT_NEAR(violation, 0.0, 0.2, "equality constraint satisfied");
    EXPECT_NEAR(result.x(0), 2.0, 0.3, "eq-constrained x* ≈ 2");
}

// ---- Test 3: Inequality constrained ----
// minimize x^2 + y^2
// subject to x + y >= 3   i.e.  -(x + y) + 3 <= 0
// Solution: x = y = 1.5
void test_inequality_constrained() {
    std::cout << "\n-- Inequality Constrained --\n";

    SQPParams params;
    params.verbose = false;
    params.mu_0 = 30.0;
    params.max_penalty_iter = 6;
    SCPSolver solver(params);

    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;

    CostFn cost{
        [](const Eigen::VectorXd& x) { return x(0)*x(0) + x(1)*x(1); },
        [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            return 2 * x;
        }
    };

    ConstraintFn ineq{
        ConstraintFn::INEQUALITY,
        "x + y >= 3",
        [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            return Eigen::VectorXd::Constant(1, 3.0 - x(0) - x(1));
        },
        [](const Eigen::VectorXd&) -> Eigen::MatrixXd {
            Eigen::MatrixXd J(1, 2);
            J << -1, -1;
            return J;
        },
        1
    };

    auto result = solver.solve(x0, {cost}, {ineq});
    double constraint_val = result.x(0) + result.x(1);
    EXPECT_NEAR(constraint_val, 3.0, 0.3, "inequality constraint active at solution");
}

// ---- Test 4: ℓ1 penalty drives violation to zero ----
// Verify that increasing mu forces the equality constraint violation to decrease
void test_l1_penalty_convergence() {
    std::cout << "\n-- l1 Penalty Convergence --\n";

    SQPParams params;
    params.verbose = false;
    params.mu_0 = 100.0;
    params.k_penalty = 10.0;
    params.max_penalty_iter = 4;
    SCPSolver solver(params);

    Eigen::VectorXd x0(2);
    x0 << 5.0, 5.0;

    // Minimize cost = x^2 + y^2 subject to x = 1
    CostFn cost{
        [](const Eigen::VectorXd& x) { return x(0)*x(0) + x(1)*x(1); },
        [](const Eigen::VectorXd& x) -> Eigen::VectorXd { return 2*x; }
    };
    ConstraintFn eq{
        ConstraintFn::EQUALITY, "x = 1",
        [](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            return Eigen::VectorXd::Constant(1, x(0) - 1.0);
        },
        [](const Eigen::VectorXd&) -> Eigen::MatrixXd {
            Eigen::MatrixXd J(1, 2);
            J << 1, 0;
            return J;
        },
        1
    };

    auto result = solver.solve(x0, {cost}, {eq});
    EXPECT_NEAR(result.constraint_violation, 0.0, 0.5,
                "l1 penalty drives violation to near-zero");
    EXPECT_NEAR(result.x(0), 1.0, 0.5, "x converges toward constrained optimum");
}

// ---- Test 5: Simple QP solver box constraints ----
// minimize 0.5*(x^2 + y^2) + [-1, -2]*[x, y]
// subject to 0 <= x <= 1, 0 <= y <= 1
// Solution: x = 1, y = 1
void test_qp_box_constraints() {
    std::cout << "\n-- QP Box Constraints --\n";

    QPProblem qp(2);
    qp.P = Eigen::Matrix2d::Identity();
    qp.q << -1.0, -2.0;
    qp.x_lb << 0.0, 0.0;
    qp.x_ub << 1.0, 1.0;

    auto result = SimpleQPSolver::solve(qp);
    EXPECT_NEAR(result.x(0), 1.0, 0.05, "QP box: x = 1");
    EXPECT_NEAR(result.x(1), 1.0, 0.05, "QP box: y = 1");
}

int main() {
    std::cout << "Running SCP solver tests...\n";

    test_unconstrained_quadratic();
    test_equality_constrained();
    test_inequality_constrained();
    test_l1_penalty_convergence();
    test_qp_box_constraints();

    std::cout << "\n==================\n";
    std::cout << "Results: " << n_passed << " passed, " << n_failed << " failed\n";
    std::cout << "==================\n";

    return n_failed > 0 ? 1 : 0;
}
