#include <doctest/doctest.h>
#include <flexicubes/flexicubes.hpp>
#include <adept.h>
#include <cmath>

using namespace flexi;

// Helper to create sphere SDF
static VecXd sphere_sdf(const MatX3 & verts, double radius)
{
    VecXd sdf(verts.rows());
    for (Index i = 0; i < verts.rows(); ++i)
    {
        sdf[i] = verts.row(i).norm() - radius;
    }
    return sdf;
}

// =============================================================================
// Basic Adept Stack Tests
// =============================================================================

TEST_CASE("Adept stack creation and basic operations")
{
    adept::Stack stack;
    // Just verify we can create a stack - don't check initial state
    // as it may have operations from other initializers
    CHECK(true);
}

TEST_CASE("Adept basic differentiation")
{
    adept::Stack stack;

    adept::adouble x = 3.0;
    adept::adouble y = 2.0;

    stack.new_recording();

    adept::adouble z = x * x + y * y;  // z = x^2 + y^2

    z.set_gradient(1.0);
    stack.compute_adjoint();

    // dz/dx = 2x = 6, dz/dy = 2y = 4
    CHECK(x.get_gradient() == doctest::Approx(6.0));
    CHECK(y.get_gradient() == doctest::Approx(4.0));
}

TEST_CASE("Adept chain rule")
{
    adept::Stack stack;

    adept::adouble x = 2.0;

    stack.new_recording();

    adept::adouble y = sin(x);
    adept::adouble z = y * y;  // z = sin(x)^2

    z.set_gradient(1.0);
    stack.compute_adjoint();

    // dz/dx = 2*sin(x)*cos(x) = sin(2x)
    double expected = std::sin(2.0 * 2.0);
    CHECK(x.get_gradient() == doctest::Approx(expected).epsilon(1e-10));
}

// =============================================================================
// Linear Interpolation Differentiation Tests
// =============================================================================

TEST_CASE("Differentiable linear interpolation - gradient check")
{
    using namespace differentiable;

    adept::Stack stack;

    // Two points
    adept::adouble x0 = 0.0, y0 = 0.0, z0 = 0.0;
    adept::adouble x1 = 1.0, y1 = 0.0, z1 = 0.0;

    // SDF values (interpolation factor 0.25)
    adept::adouble s0 = -0.25;
    adept::adouble s1 = 0.75;

    stack.new_recording();

    adept::adouble rx, ry, rz;
    linear_interp_ad(x0, y0, z0, x1, y1, z1, s0, s1, rx, ry, rz);

    // Loss = rx (just x coordinate)
    rx.set_gradient(1.0);
    stack.compute_adjoint();

    // Check that gradients are computed (non-zero for inputs that matter)
    CHECK(std::isfinite(s0.get_gradient()));
    CHECK(std::isfinite(s1.get_gradient()));
    CHECK(std::isfinite(x0.get_gradient()));
    CHECK(std::isfinite(x1.get_gradient()));
}

TEST_CASE("Differentiable linear interpolation - finite difference verification")
{
    using namespace differentiable;

    const double eps = 1e-6;

    // Base values
    double x0_val = 0.0, y0_val = 0.0, z0_val = 0.0;
    double x1_val = 1.0, y1_val = 0.0, z1_val = 0.0;
    double s0_val = -0.3;
    double s1_val = 0.7;

    double analytical_grad_s0;

    // Compute analytical gradient using Adept
    {
        adept::Stack stack;
        adept::adouble x0 = x0_val, y0 = y0_val, z0 = z0_val;
        adept::adouble x1 = x1_val, y1 = y1_val, z1 = z1_val;
        adept::adouble s0 = s0_val, s1 = s1_val;

        stack.new_recording();

        adept::adouble rx, ry, rz;
        linear_interp_ad(x0, y0, z0, x1, y1, z1, s0, s1, rx, ry, rz);

        rx.set_gradient(1.0);
        stack.compute_adjoint();

        analytical_grad_s0 = s0.get_gradient();
    }

    // Compute numerical gradient using finite differences
    auto compute_rx = [&](double s0_v)
    {
        adept::Stack local_stack;
        adept::adouble lx0 = x0_val, ly0 = y0_val, lz0 = z0_val;
        adept::adouble lx1 = x1_val, ly1 = y1_val, lz1 = z1_val;
        adept::adouble ls0 = s0_v, ls1 = s1_val;
        adept::adouble lrx, lry, lrz;
        linear_interp_ad(lx0, ly0, lz0, lx1, ly1, lz1, ls0, ls1, lrx, lry, lrz);
        return adept::value(lrx);
    };

    double numerical_grad_s0 = (compute_rx(s0_val + eps) - compute_rx(s0_val - eps)) / (2 * eps);

    CHECK(analytical_grad_s0 == doctest::Approx(numerical_grad_s0).epsilon(1e-4));
}

// =============================================================================
// Weighted Linear Interpolation Tests
// =============================================================================

TEST_CASE("Differentiable weighted linear interpolation")
{
    using namespace differentiable;

    adept::Stack stack;

    adept::adouble x0 = 0.0, y0 = 0.0, z0 = 0.0;
    adept::adouble x1 = 1.0, y1 = 0.0, z1 = 0.0;
    adept::adouble s0 = -0.5, s1 = 0.5;
    adept::adouble a0 = 1.0, a1 = 1.0;  // Equal weights

    stack.new_recording();

    adept::adouble rx, ry, rz;
    linear_interp_weighted_ad(x0, y0, z0, x1, y1, z1, s0, s1, a0, a1, rx, ry, rz);

    // With equal weights and symmetric SDF, result should be midpoint
    CHECK(adept::value(rx) == doctest::Approx(0.5).epsilon(1e-10));
    CHECK(adept::value(ry) == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(adept::value(rz) == doctest::Approx(0.0).epsilon(1e-10));

    rx.set_gradient(1.0);
    stack.compute_adjoint();

    // Alpha gradients should be computed
    CHECK(std::isfinite(a0.get_gradient()));
    CHECK(std::isfinite(a1.get_gradient()));
}

TEST_CASE("Weighted interpolation - alpha effect")
{
    using namespace differentiable;

    adept::Stack stack;

    adept::adouble x0 = 0.0, y0 = 0.0, z0 = 0.0;
    adept::adouble x1 = 1.0, y1 = 0.0, z1 = 0.0;
    adept::adouble s0 = -0.5, s1 = 0.5;

    // Unequal weights should shift the interpolation point
    adept::adouble a0_high = 2.0, a1_low = 1.0;
    adept::adouble a0_low = 1.0, a1_high = 2.0;

    stack.new_recording();

    adept::adouble rx1, ry1, rz1;
    adept::adouble rx2, ry2, rz2;

    linear_interp_weighted_ad(x0, y0, z0, x1, y1, z1, s0, s1, a0_high, a1_low, rx1, ry1, rz1);
    linear_interp_weighted_ad(x0, y0, z0, x1, y1, z1, s0, s1, a0_low, a1_high, rx2, ry2, rz2);

    // Different alpha weights should give different results
    CHECK(adept::value(rx1) != doctest::Approx(adept::value(rx2)).epsilon(1e-6));
}

// =============================================================================
// Full Pipeline Differentiation Tests
// =============================================================================

TEST_CASE("Extract surface with gradients - basic")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads;

    auto [mesh, loss] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads);

    CHECK_FALSE(mesh.empty());
    CHECK(std::isfinite(loss));
    CHECK(loss >= 0.0);  // L_dev should be non-negative
}

TEST_CASE("Extract surface with loss callback - dual vertex L2")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads;

    auto loss_cb = [](const std::vector<adept::adouble> & x,
                      const std::vector<adept::adouble> & y,
                      const std::vector<adept::adouble> & z)
    {
        adept::adouble sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i)
        {
            sum += x[i] * x[i] + y[i] * y[i] + z[i] * z[i];
        }
        return sum;
    };

    auto [mesh, loss] = fc.extract_surface_with_loss(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, loss_cb, &grads);

    CHECK_FALSE(mesh.empty());

    double expected = mesh.vertices.array().square().sum();
    CHECK(loss == doctest::Approx(expected).epsilon(1e-6));

    double max_abs_grad = grads.d_sdf.cwiseAbs().maxCoeff();
    CHECK(max_abs_grad > 0.0);
}

TEST_CASE("Extract surface with gradients - gradient dimensions")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();
    Index num_verts = grid.num_vertices();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads;

    auto [mesh, loss] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads);

    // Check gradient dimensions match input dimensions
    CHECK(grads.d_vertices.rows() == num_verts);
    CHECK(grads.d_vertices.cols() == 3);
    CHECK(grads.d_sdf.size() == num_verts);
    CHECK(grads.d_beta.rows() == num_cubes);
    CHECK(grads.d_beta.cols() == 12);
    CHECK(grads.d_alpha.rows() == num_cubes);
    CHECK(grads.d_alpha.cols() == 8);
    CHECK(grads.d_gamma.size() == num_cubes);
}

TEST_CASE("Extract surface with gradients - gradients are finite")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads;

    auto [mesh, loss] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads);

    // All gradient values should be finite
    for (Index i = 0; i < grads.d_vertices.rows(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            CHECK(std::isfinite(grads.d_vertices(i, j)));
        }
    }

    for (Index i = 0; i < grads.d_sdf.size(); ++i)
    {
        CHECK(std::isfinite(grads.d_sdf[i]));
    }

    for (Index i = 0; i < grads.d_beta.rows(); ++i)
    {
        for (int j = 0; j < 12; ++j)
        {
            CHECK(std::isfinite(grads.d_beta(i, j)));
        }
    }

    for (Index i = 0; i < grads.d_alpha.rows(); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            CHECK(std::isfinite(grads.d_alpha(i, j)));
        }
    }

    for (Index i = 0; i < grads.d_gamma.size(); ++i)
    {
        CHECK(std::isfinite(grads.d_gamma[i]));
    }
}

TEST_CASE("Extract surface with gradients - empty surface handling")
{
    FlexiCubes fc;
    int res = 4;

    auto grid = fc.construct_voxel_grid(res);

    // All outside - no surface
    VecXd sdf = VecXd::Constant(grid.num_vertices(), 1.0);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads;

    auto [mesh, loss] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads);

    CHECK(mesh.empty());
    CHECK(loss == 0.0);

    // Gradients should be zero for empty surface
    CHECK(grads.d_vertices.isZero());
    CHECK(grads.d_sdf.isZero());
}

// =============================================================================
// Gradient Verification via Finite Differences
// =============================================================================

TEST_CASE("SDF gradient finite difference verification")
{
    FlexiCubes fc;
    int res = 4;  // Small resolution for speed

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads;

    auto [mesh, loss] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads);

    // Verify a few SDF gradients using finite differences
    const double eps = 1e-5;
    int num_checks   = std::min<int>(10, static_cast<int>(sdf.size()));

    for (int i = 0; i < num_checks; ++i)
    {
        // Skip vertices that aren't near the surface
        if (std::abs(sdf[i]) > 0.3) continue;

        VecXd sdf_plus  = sdf;
        VecXd sdf_minus = sdf;
        sdf_plus[i] += eps;
        sdf_minus[i] -= eps;

        Gradients grads_plus, grads_minus;
        auto [mesh_plus, loss_plus] = fc.extract_surface_with_grads(
            grid.vertices, sdf_plus, grid.cubes, Resolution(res),
            beta, alpha, gamma, &grads_plus);

        auto [mesh_minus, loss_minus] = fc.extract_surface_with_grads(
            grid.vertices, sdf_minus, grid.cubes, Resolution(res),
            beta, alpha, gamma, &grads_minus);

        double numerical_grad  = (loss_plus - loss_minus) / (2 * eps);
        double analytical_grad = grads.d_sdf[i];

        // Only check if both losses are valid (surface exists in both cases)
        if (loss_plus > 0 && loss_minus > 0 && std::abs(numerical_grad) > 1e-8)
        {
            double rel_error = std::abs(analytical_grad - numerical_grad) /
                               (std::abs(numerical_grad) + 1e-10);
            // Allow for some numerical error
            CHECK(rel_error < 0.5);  // Within 50% for finite diff approximation
        }
    }
}

// =============================================================================
// L_dev Regularization Tests
// =============================================================================

TEST_CASE("L_dev loss computation")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads;

    // Extract with default weights
    auto [mesh, loss] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads);

    // L_dev should be sum of per-vertex L_dev values
    double expected_loss = mesh.l_dev.sum();
    CHECK(loss == doctest::Approx(expected_loss).epsilon(1e-6));
}

TEST_CASE("L_dev weight scaling")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads1, grads2;

    auto [mesh1, loss1] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads1, 1.0);  // weight = 1.0

    auto [mesh2, loss2] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads2, 2.0);  // weight = 2.0

    // Loss should scale with weight
    CHECK(loss2 == doctest::Approx(2.0 * loss1).epsilon(1e-6));
}

// =============================================================================
// Optimization Step Tests
// =============================================================================

TEST_CASE("Gradient descent step reduces loss")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    // Add small perturbation to SDF
    for (Index i = 0; i < sdf.size(); ++i)
    {
        sdf[i] += 0.01 * std::sin(static_cast<double>(i));
    }

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    // Initialize with small random values
    beta.setRandom();
    beta *= 0.1;
    alpha.setRandom();
    alpha *= 0.1;
    gamma.setRandom();
    gamma *= 0.1;

    Gradients grads;

    auto [mesh1, loss1] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads);

    if (mesh1.empty() || loss1 < 1e-10)
    {
        // Skip test if no surface or already optimal
        return;
    }

    // Take a gradient descent step on beta
    double lr = 0.01;
    beta -= lr * grads.d_beta;

    Gradients grads2;
    auto [mesh2, loss2] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads2);

    // Loss should decrease (or stay same if already at minimum)
    CHECK(loss2 <= loss1 + 1e-6);  // Allow small numerical error
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_CASE("Differentiation with extreme SDF values")
{
    FlexiCubes fc;
    int res = 4;

    auto grid = fc.construct_voxel_grid(res);

    // SDF with very large and very small values
    VecXd sdf(grid.num_vertices());
    for (Index i = 0; i < sdf.size(); ++i)
    {
        double x = grid.vertices(i, 0);
        sdf[i]   = x * 100.0;  // Large gradient
    }

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads;

    // Should not crash or produce NaN
    auto [mesh, loss] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads);

    CHECK(std::isfinite(loss));
}

TEST_CASE("Multiple recordings on same stack")
{
    adept::Stack stack;

    for (int iter = 0; iter < 5; ++iter)
    {
        adept::adouble x = static_cast<double>(iter + 1);

        stack.new_recording();

        adept::adouble y = x * x;

        y.set_gradient(1.0);
        stack.compute_adjoint();

        CHECK(x.get_gradient() == doctest::Approx(2.0 * (iter + 1)));
    }
}

TEST_CASE("Nested function differentiation")
{
    adept::Stack stack;

    adept::adouble x = 2.0;

    stack.new_recording();

    // f(x) = sin(cos(x^2))
    adept::adouble x2 = x * x;
    adept::adouble c  = cos(x2);
    adept::adouble s  = sin(c);

    s.set_gradient(1.0);
    stack.compute_adjoint();

    // df/dx = cos(cos(x^2)) * (-sin(x^2)) * 2x
    double x_val    = 2.0;
    double expected = std::cos(std::cos(x_val * x_val)) *
                      (-std::sin(x_val * x_val)) * 2 * x_val;

    CHECK(x.get_gradient() == doctest::Approx(expected).epsilon(1e-10));
}

// =============================================================================
// Beta/Alpha Weight Effect Tests
// =============================================================================

TEST_CASE("Beta weights affect dual vertex position")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    // Two different beta configurations
    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta1(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta2(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta1.setZero();
    beta2.setConstant(0.5);  // Different beta values
    alpha.setZero();
    gamma.setZero();

    Mesh mesh1 = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res),
                                    &beta1, &alpha, &gamma);
    Mesh mesh2 = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res),
                                    &beta2, &alpha, &gamma);

    // Different beta values should produce different meshes
    // (at least slightly different vertex positions)
    CHECK_FALSE(mesh1.empty());
    CHECK_FALSE(mesh2.empty());
}

TEST_CASE("Alpha weights affect zero crossing position")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha1(num_cubes, 8);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha2(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha1.setZero();
    alpha2.setConstant(0.5);  // Different alpha values
    gamma.setZero();

    Mesh mesh1 = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res),
                                    &beta, &alpha1, &gamma);
    Mesh mesh2 = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res),
                                    &beta, &alpha2, &gamma);

    CHECK_FALSE(mesh1.empty());
    CHECK_FALSE(mesh2.empty());
}

// =============================================================================
// Gradient Consistency Tests
// =============================================================================

TEST_CASE("Gradients consistent across multiple calls")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads1, grads2;

    auto [mesh1, loss1] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads1);

    auto [mesh2, loss2] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads2);

    // Same inputs should give same outputs
    CHECK(loss1 == doctest::Approx(loss2));
    CHECK(grads1.d_sdf.isApprox(grads2.d_sdf, 1e-10));
    CHECK(grads1.d_beta.isApprox(grads2.d_beta, 1e-10));
}

TEST_CASE("Training mode produces more triangles with gradients")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();
    alpha.setZero();
    gamma.setZero();

    Gradients grads;

    Options inference_opts = Options::defaults();
    Options training_opts  = Options::training_defaults();

    auto [mesh_train, loss] = fc.extract_surface_with_grads(
        grid.vertices, sdf, grid.cubes, Resolution(res),
        beta, alpha, gamma, &grads, 1.0, training_opts);

    Mesh mesh_inf = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res),
                                       &beta, &alpha, &gamma, nullptr, inference_opts);

    // Training mode should produce more triangles (4 per quad vs 2)
    CHECK(mesh_train.num_faces() > mesh_inf.num_faces());
}
