#include <doctest/doctest.h>
#include <flexicubes/surface.hpp>
#include <set>

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

TEST_CASE("Surface cube identification - sphere")
{
    auto grid = construct_voxel_grid(8);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    auto surf_cubes = identify_surface_cubes(sdf, grid.cubes);

    // Should have some surface cubes (sphere intersects grid)
    CHECK(surf_cubes.num_surface_cubes() > 0);
    CHECK(surf_cubes.num_surface_cubes() < grid.num_cubes());

    // Check mask dimensions
    CHECK(surf_cubes.mask.size() == grid.num_cubes());
    CHECK(surf_cubes.occupancy.rows() == grid.num_cubes());
    CHECK(surf_cubes.occupancy.cols() == 8);
}

TEST_CASE("Surface cube identification - all inside")
{
    auto grid = construct_voxel_grid(4);

    // All vertices inside (negative SDF)
    VecXd sdf = VecXd::Constant(grid.vertices.rows(), -1.0);

    auto surf_cubes = identify_surface_cubes(sdf, grid.cubes);

    // No surface cubes (all inside)
    CHECK(surf_cubes.num_surface_cubes() == 0);

    // All corners should be marked inside
    for (Index i = 0; i < surf_cubes.occupancy.rows(); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            CHECK(surf_cubes.occupancy(i, j) == 1);
        }
    }
}

TEST_CASE("Surface cube identification - all outside")
{
    auto grid = construct_voxel_grid(4);

    // All vertices outside (positive SDF)
    VecXd sdf = VecXd::Constant(grid.vertices.rows(), 1.0);

    auto surf_cubes = identify_surface_cubes(sdf, grid.cubes);

    // No surface cubes (all outside)
    CHECK(surf_cubes.num_surface_cubes() == 0);

    // All corners should be marked outside
    for (Index i = 0; i < surf_cubes.occupancy.rows(); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            CHECK(surf_cubes.occupancy(i, j) == 0);
        }
    }
}

TEST_CASE("Surface edge identification")
{
    auto grid = construct_voxel_grid(8);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    auto surf_cubes = identify_surface_cubes(sdf, grid.cubes);
    auto surf_edges = identify_surface_edges(sdf, grid.cubes, surf_cubes);

    // Should have surface edges
    CHECK(surf_edges.num_surface_edges() > 0);

    // Check edge array dimensions
    CHECK(surf_edges.edges.cols() == 2);

    // All edges should have valid vertex indices
    for (Index i = 0; i < surf_edges.edges.rows(); ++i)
    {
        int v0 = surf_edges.edges(i, 0);
        int v1 = surf_edges.edges(i, 1);
        CHECK(v0 >= 0);
        CHECK(v1 >= 0);
        CHECK(v0 < grid.vertices.rows());
        CHECK(v1 < grid.vertices.rows());
        CHECK(v0 != v1);

        // Edge should cross the surface (opposite signs)
        CHECK(((sdf[v0] < 0) != (sdf[v1] < 0)));
    }
}

TEST_CASE("Surface edge index mapping")
{
    auto grid = construct_voxel_grid(4);
    VecXd sdf = sphere_sdf(grid.vertices, 0.3);

    auto surf_cubes = identify_surface_cubes(sdf, grid.cubes);
    auto surf_edges = identify_surface_edges(sdf, grid.cubes, surf_cubes);

    // Check idx_map dimensions
    CHECK(surf_edges.idx_map.rows() == surf_cubes.num_surface_cubes());
    CHECK(surf_edges.idx_map.cols() == 12);

    // Valid indices should be in range or -1
    for (Index i = 0; i < surf_edges.idx_map.rows(); ++i)
    {
        for (int j = 0; j < 12; ++j)
        {
            int idx = surf_edges.idx_map(i, j);
            CHECK((idx == -1 || (idx >= 0 && idx < surf_edges.num_surface_edges())));
        }
    }
}

TEST_CASE("Get surface cube indices")
{
    auto grid = construct_voxel_grid(8);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    auto surf_cubes = identify_surface_cubes(sdf, grid.cubes);
    auto indices    = get_surface_cube_indices(surf_cubes);

    CHECK(indices.size() == surf_cubes.num_surface_cubes());

    // Indices should be unique and in range
    std::set<int> unique_indices(indices.data(), indices.data() + indices.size());
    CHECK(unique_indices.size() == static_cast<size_t>(indices.size()));

    for (Index i = 0; i < indices.size(); ++i)
    {
        CHECK(indices[i] >= 0);
        CHECK(indices[i] < grid.num_cubes());
        CHECK(surf_cubes.mask[indices[i]]);
    }
}

TEST_CASE("Get surface cubes fx8")
{
    auto grid = construct_voxel_grid(8);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    auto surf_cubes = identify_surface_cubes(sdf, grid.cubes);
    auto surf_fx8   = get_surface_cubes_fx8(grid.cubes, surf_cubes);

    CHECK(surf_fx8.rows() == surf_cubes.num_surface_cubes());
    CHECK(surf_fx8.cols() == 8);

    // All indices should be valid
    for (Index i = 0; i < surf_fx8.rows(); ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            int idx = surf_fx8(i, j);
            CHECK(idx >= 0);
            CHECK(idx < grid.vertices.rows());
        }
    }
}
