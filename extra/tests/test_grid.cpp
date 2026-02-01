#include <doctest/doctest.h>
#include <flexicubes/grid.hpp>
#include <set>

using namespace flexi;

TEST_CASE("Uniform resolution grid")
{
    auto grid = construct_voxel_grid(4);

    // Check dimensions
    CHECK(grid.num_cubes() == 4 * 4 * 4);  // 64 cubes
    CHECK(grid.cubes.cols() == 8);         // 8 corners per cube

    // Vertices should be unique
    std::set<std::tuple<int, int, int>> unique_verts;
    for (Index i = 0; i < grid.vertices.rows(); ++i)
    {
        // Quantize to check uniqueness
        int x = static_cast<int>(std::round(grid.vertices(i, 0) * 1000));
        int y = static_cast<int>(std::round(grid.vertices(i, 1) * 1000));
        int z = static_cast<int>(std::round(grid.vertices(i, 2) * 1000));
        unique_verts.insert({x, y, z});
    }
    CHECK(unique_verts.size() == static_cast<size_t>(grid.vertices.rows()));

    // Expected unique vertices for 4x4x4 grid: (4+1)^3 = 125
    CHECK(grid.vertices.rows() == 5 * 5 * 5);
}

TEST_CASE("Non-uniform resolution grid")
{
    Resolution res(2, 3, 4);
    auto grid = construct_voxel_grid(res);

    CHECK(grid.num_cubes() == 2 * 3 * 4);      // 24 cubes
    CHECK(grid.vertices.rows() == 3 * 4 * 5);  // (2+1) * (3+1) * (4+1) = 60
}

TEST_CASE("Grid vertex bounds")
{
    auto grid = construct_voxel_grid(8);

    // Vertices should be centered at origin, in [-0.5, 0.5]^3
    Vec3 min_corner = grid.vertices.colwise().minCoeff();
    Vec3 max_corner = grid.vertices.colwise().maxCoeff();

    CHECK(min_corner.x() == doctest::Approx(-0.5));
    CHECK(min_corner.y() == doctest::Approx(-0.5));
    CHECK(min_corner.z() == doctest::Approx(-0.5));

    CHECK(max_corner.x() == doctest::Approx(0.5));
    CHECK(max_corner.y() == doctest::Approx(0.5));
    CHECK(max_corner.z() == doctest::Approx(0.5));
}

TEST_CASE("Cube indices valid")
{
    auto grid = construct_voxel_grid(4);

    // All cube indices should reference valid vertices
    for (Index i = 0; i < grid.cubes.rows(); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int idx = grid.cubes(i, j);
            CHECK(idx >= 0);
            CHECK(idx < grid.vertices.rows());
        }
    }

    // Each cube should have 8 distinct corners
    for (Index i = 0; i < grid.cubes.rows(); ++i)
    {
        std::set<int> corners;
        for (int j = 0; j < 8; ++j)
        {
            corners.insert(grid.cubes(i, j));
        }
        CHECK(corners.size() == 8);
    }
}

TEST_CASE("Cube corner ordering")
{
    auto grid = construct_voxel_grid(2);

    // Check that cube corners follow expected ordering
    // Corner 0: (0,0,0), Corner 1: (1,0,0), etc.
    for (Index c = 0; c < grid.cubes.rows(); ++c)
    {
        auto verts = get_cube_vertices(grid, c);

        // Corner differences should match expected offsets
        Vec3 v0 = verts.row(0);

        // Along X: corner 1 should be (+dx, 0, 0) from corner 0
        Vec3 diff = verts.row(1).transpose() - v0;
        CHECK(diff.x() > 0);
        CHECK(std::abs(diff.y()) < 1e-10);
        CHECK(std::abs(diff.z()) < 1e-10);
    }
}

TEST_CASE("Cube edges")
{
    auto grid = construct_voxel_grid(2);

    auto edges = get_cube_edges(grid, 0);
    CHECK(edges.size() == 12);

    // Each edge should connect two different vertices
    for (const auto & [v0, v1] : edges)
    {
        CHECK(v0 != v1);
        CHECK(v0 >= 0);
        CHECK(v1 >= 0);
        CHECK(v0 < grid.vertices.rows());
        CHECK(v1 < grid.vertices.rows());
    }
}

TEST_CASE("Resolution equality")
{
    Resolution r1(4, 4, 4);
    Resolution r2(4);

    CHECK(r1 == r2);
    CHECK(r1.total_cubes() == 64);
    CHECK(r1.total_verts() == 125);

    Resolution r3(2, 3, 4);
    CHECK_FALSE(r1 == r3);
}
