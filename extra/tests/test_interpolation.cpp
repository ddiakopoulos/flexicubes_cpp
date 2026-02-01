#include <doctest/doctest.h>
#include <flexicubes/dual_vertex.hpp>

using namespace flexi;

TEST_CASE("Linear interpolation - midpoint")
{
    Vec3 p0(0, 0, 0);
    Vec3 p1(1, 0, 0);
    double s0 = -1.0;
    double s1 = 1.0;

    Vec3 result = linear_interp(p0, p1, s0, s1);

    CHECK(result.x() == doctest::Approx(0.5));
    CHECK(result.y() == doctest::Approx(0.0));
    CHECK(result.z() == doctest::Approx(0.0));
}

TEST_CASE("Linear interpolation - quarter point")
{
    Vec3 p0(0, 0, 0);
    Vec3 p1(1, 0, 0);
    double s0 = -1.0;
    double s1 = 3.0;  // Zero at 0.25

    Vec3 result = linear_interp(p0, p1, s0, s1);

    CHECK(result.x() == doctest::Approx(0.25));
    CHECK(result.y() == doctest::Approx(0.0));
    CHECK(result.z() == doctest::Approx(0.0));
}

TEST_CASE("Linear interpolation - 3D")
{
    Vec3 p0(1, 2, 3);
    Vec3 p1(5, 6, 7);
    double s0 = -1.0;
    double s1 = 1.0;

    Vec3 result = linear_interp(p0, p1, s0, s1);

    CHECK(result.x() == doctest::Approx(3.0));
    CHECK(result.y() == doctest::Approx(4.0));
    CHECK(result.z() == doctest::Approx(5.0));
}

TEST_CASE("Linear interpolation - near-zero denominator")
{
    Vec3 p0(0, 0, 0);
    Vec3 p1(1, 1, 1);
    double s0 = 0.0;
    double s1 = 1e-15;  // Very small

    Vec3 result = linear_interp(p0, p1, s0, s1);

    // Should fallback to midpoint
    CHECK(result.x() == doctest::Approx(0.5).epsilon(0.01));
}

TEST_CASE("Weighted linear interpolation")
{
    Vec3 p0(0, 0, 0);
    Vec3 p1(1, 0, 0);
    double s0 = -1.0;
    double s1 = 1.0;

    // Uniform weights should give same as unweighted
    Vec3 result1 = linear_interp_weighted(p0, p1, s0, s1, 1.0, 1.0);
    CHECK(result1.x() == doctest::Approx(0.5));

    // Non-uniform weights shift the result
    Vec3 result2 = linear_interp_weighted(p0, p1, s0, s1, 2.0, 1.0);
    // s0_weighted = -2, s1_weighted = 1, zero at 2/3
    CHECK(result2.x() == doctest::Approx(2.0 / 3.0));

    Vec3 result3 = linear_interp_weighted(p0, p1, s0, s1, 1.0, 2.0);
    // s0_weighted = -1, s1_weighted = 2, zero at 1/3
    CHECK(result3.x() == doctest::Approx(1.0 / 3.0));
}

TEST_CASE("Zero crossings computation")
{
    // Create simple edge data
    MatX3 vertices(4, 3);
    vertices << 0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        1, 1, 0;

    VecXd sdf(4);
    sdf << -1, 1, -1, 1;  // Alternating inside/outside

    Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> edges(2, 2);
    edges << 0, 1,
        2, 3;

    MatX3 crossings = compute_zero_crossings(vertices, sdf, edges);

    CHECK(crossings.rows() == 2);
    CHECK(crossings.cols() == 3);

    // First edge: (0,0,0) to (1,0,0), sdf -1 to 1, crossing at (0.5, 0, 0)
    CHECK(crossings(0, 0) == doctest::Approx(0.5));
    CHECK(crossings(0, 1) == doctest::Approx(0.0));
    CHECK(crossings(0, 2) == doctest::Approx(0.0));

    // Second edge: (0,1,0) to (1,1,0), sdf -1 to 1, crossing at (0.5, 1, 0)
    CHECK(crossings(1, 0) == doctest::Approx(0.5));
    CHECK(crossings(1, 1) == doctest::Approx(1.0));
    CHECK(crossings(1, 2) == doctest::Approx(0.0));
}

TEST_CASE("Weight normalization - defaults")
{
    Index num_cubes = 10;

    auto [beta, alpha, gamma] = normalize_weights(nullptr, nullptr, nullptr, num_cubes);

    CHECK(beta.rows() == num_cubes);
    CHECK(beta.cols() == 12);
    CHECK(alpha.rows() == num_cubes);
    CHECK(alpha.cols() == 8);
    CHECK(gamma.size() == num_cubes);

    // Default weights should be 1.0
    CHECK(beta(0, 0) == doctest::Approx(1.0));
    CHECK(alpha(0, 0) == doctest::Approx(1.0));
    CHECK(gamma(0) == doctest::Approx(1.0));
}

TEST_CASE("Weight normalization - with input")
{
    Index num_cubes = 5;

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta_in(num_cubes, 12);
    beta_in.setZero();  // tanh(0) * 0.99 + 1 = 1.0

    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha_in(num_cubes, 8);
    alpha_in.setConstant(1.0);  // tanh(1) * 0.99 + 1 ≈ 1.75

    VecXd gamma_in(num_cubes);
    gamma_in.setZero();  // sigmoid(0) * 0.99 + 0.005 = 0.5

    auto [beta, alpha, gamma] = normalize_weights(&beta_in, &alpha_in, &gamma_in, num_cubes);

    CHECK(beta(0, 0) == doctest::Approx(1.0));
    CHECK(alpha(0, 0) == doctest::Approx(std::tanh(1.0) * 0.99 + 1.0));
    CHECK(gamma(0) == doctest::Approx(0.5));
}
