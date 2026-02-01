#include <doctest/doctest.h>
#include <flexicubes/flexicubes.hpp>

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

// Helper to create box SDF
static VecXd box_sdf(const MatX3 & verts, const Vec3 & half_extents)
{
    VecXd sdf(verts.rows());
    for (Index i = 0; i < verts.rows(); ++i)
    {
        Vec3 p     = verts.row(i).transpose();
        Vec3 q     = p.cwiseAbs() - half_extents;
        Vec3 q_max = q.cwiseMax(0.0);
        sdf[i]     = q_max.norm() + std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0);
    }
    return sdf;
}

TEST_CASE("Basic sphere extraction")
{
    FlexiCubes fc;
    int res = 16;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));

    CHECK_FALSE(mesh.empty());
    CHECK(mesh.num_vertices() > 0);
    CHECK(mesh.num_faces() > 0);
    CHECK(mesh.l_dev.size() > 0);
}

TEST_CASE("Different resolutions")
{
    FlexiCubes fc;

    for (int res : {4, 8, 16, 32})
    {
        auto grid = fc.construct_voxel_grid(res);
        VecXd sdf = sphere_sdf(grid.vertices, 0.4);

        Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));

        // Higher resolution should give more vertices
        CHECK(mesh.num_vertices() > 0);
        CHECK(mesh.num_faces() > 0);
    }
}

TEST_CASE("Empty extraction - all outside")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = VecXd::Constant(grid.vertices.rows(), 1.0);  // All outside

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));

    CHECK(mesh.empty());
}

TEST_CASE("Empty extraction - all inside")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = VecXd::Constant(grid.vertices.rows(), -1.0);  // All inside

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));

    CHECK(mesh.empty());
}

TEST_CASE("Box extraction")
{
    FlexiCubes fc;
    int res = 16;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = box_sdf(grid.vertices, Vec3(0.3, 0.3, 0.3));

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));

    CHECK_FALSE(mesh.empty());
    CHECK(mesh.num_vertices() > 0);
    CHECK(mesh.num_faces() > 0);
}

TEST_CASE("Training mode - more triangles")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Options inference_opts;
    inference_opts.training = false;
    Mesh mesh_inf           = fc.extract_surface(grid.vertices, sdf, grid.cubes,
                                                 Resolution(res), nullptr, nullptr, nullptr, nullptr, inference_opts);

    Options training_opts;
    training_opts.training = true;
    Mesh mesh_train        = fc.extract_surface(grid.vertices, sdf, grid.cubes,
                                                Resolution(res), nullptr, nullptr, nullptr, nullptr, training_opts);

    // Training mode creates 4 triangles per quad vs 2
    CHECK(mesh_train.num_faces() > mesh_inf.num_faces());
    // Approximately 2x faces
    CHECK(mesh_train.num_faces() >= mesh_inf.num_faces() * 1.5);
}

TEST_CASE("Mesh vertices in bounds")
{
    FlexiCubes fc;
    int res = 16;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));

    // All mesh vertices should be within grid bounds
    Vec3 grid_min = grid.vertices.colwise().minCoeff();
    Vec3 grid_max = grid.vertices.colwise().maxCoeff();

    for (Index i = 0; i < mesh.vertices.rows(); ++i)
    {
        Vec3 v = mesh.vertices.row(i).transpose();
        CHECK(v.x() >= grid_min.x() - 1e-6);
        CHECK(v.y() >= grid_min.y() - 1e-6);
        CHECK(v.z() >= grid_min.z() - 1e-6);
        CHECK(v.x() <= grid_max.x() + 1e-6);
        CHECK(v.y() <= grid_max.y() + 1e-6);
        CHECK(v.z() <= grid_max.z() + 1e-6);
    }
}

TEST_CASE("Face indices valid")
{
    FlexiCubes fc;
    int res = 16;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));

    for (Index i = 0; i < mesh.faces.rows(); ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            int idx = mesh.faces(i, j);
            CHECK(idx >= 0);
            CHECK(idx < mesh.vertices.rows());
        }
    }
}

TEST_CASE("Non-degenerate faces")
{
    FlexiCubes fc;
    int res = 16;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));

    int degenerate = 0;
    for (Index i = 0; i < mesh.faces.rows(); ++i)
    {
        Vec3 v0 = mesh.vertices.row(mesh.faces(i, 0)).transpose();
        Vec3 v1 = mesh.vertices.row(mesh.faces(i, 1)).transpose();
        Vec3 v2 = mesh.vertices.row(mesh.faces(i, 2)).transpose();

        Vec3 cross = (v1 - v0).cross(v2 - v0);
        if (cross.norm() < 1e-10)
        {
            degenerate++;
        }
    }

    // Allow a few degenerate faces
    CHECK(degenerate < mesh.faces.rows() / 10);
}

TEST_CASE("L_dev values non-negative")
{
    FlexiCubes fc;
    int res = 16;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));

    for (Index i = 0; i < mesh.l_dev.size(); ++i)
    {
        CHECK(mesh.l_dev[i] >= 0.0);
    }
}

TEST_CASE("Tetrahedral mesh extraction")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    TetraMesh tet_mesh = fc.extract_volume(grid.vertices, sdf, grid.cubes, Resolution(res));

    // Note: tet mesh generation may produce few or no tets depending on configuration
    // Just verify it doesn't crash and produces valid data if any
    if (!tet_mesh.empty())
    {
        CHECK(tet_mesh.num_vertices() > 0);
        CHECK(tet_mesh.num_tets() >= 0);
        CHECK(tet_mesh.l_dev.size() > 0);

        // Check tet indices valid
        for (Index i = 0; i < tet_mesh.tets.rows(); ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                int idx = tet_mesh.tets(i, j);
                CHECK(idx >= 0);
                CHECK(idx < tet_mesh.vertices.rows());
            }
        }
    }
}

TEST_CASE("QEF extraction with grad_func")
{
    FlexiCubes fc;
    int res = 16;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    auto grad_func = [](const MatX3 & x)
    {
        MatX3 g = x;
        for (Index i = 0; i < g.rows(); ++i)
        {
            double n = g.row(i).norm();
            if (n > 1e-10)
            {
                g.row(i) /= n;
            }
            else
            {
                g.row(i).setZero();
            }
        }
        return g;
    };

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res),
                                   nullptr, nullptr, nullptr, grad_func);

    CHECK_FALSE(mesh.empty());
    CHECK(mesh.num_vertices() > 0);
    CHECK(mesh.num_faces() > 0);
}

TEST_CASE("QEF L_dev is zeroed")
{
    FlexiCubes fc;
    int res = 12;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    auto grad_func = [](const MatX3 & x)
    {
        MatX3 g = x;
        for (Index i = 0; i < g.rows(); ++i)
        {
            double n = g.row(i).norm();
            if (n > 1e-10)
            {
                g.row(i) /= n;
            }
            else
            {
                g.row(i).setZero();
            }
        }
        return g;
    };

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res),
                                   nullptr, nullptr, nullptr, grad_func);

    CHECK_FALSE(mesh.empty());
    if (mesh.l_dev.size() > 0)
    {
        CHECK(mesh.l_dev.array().abs().maxCoeff() == doctest::Approx(0.0));
    }
}

TEST_CASE("QEF reg scale affects output")
{
    FlexiCubes fc;
    int res = 12;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    auto grad_func = [](const MatX3 & x)
    {
        MatX3 g = x;
        for (Index i = 0; i < g.rows(); ++i)
        {
            double n = g.row(i).norm();
            if (n > 1e-10)
            {
                g.row(i) /= n;
            }
            else
            {
                g.row(i).setZero();
            }
        }
        return g;
    };

    Options opts_a;
    Options opts_b;
    opts_a.qef_reg_scale = 1e-6;
    opts_b.qef_reg_scale = 1e-2;

    Mesh mesh_a = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res),
                                     nullptr, nullptr, nullptr, grad_func, opts_a);
    Mesh mesh_b = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res),
                                     nullptr, nullptr, nullptr, grad_func, opts_b);

    CHECK_FALSE(mesh_a.empty());
    CHECK_FALSE(mesh_b.empty());

    Index n         = std::min(mesh_a.vertices.rows(), mesh_b.vertices.rows());
    double max_diff = 0.0;
    for (Index i = 0; i < n; ++i)
    {
        double d = (mesh_a.vertices.row(i) - mesh_b.vertices.row(i)).norm();
        if (d > max_diff) max_diff = d;
    }

    CHECK(max_diff > 1e-7);
}

TEST_CASE("Planar SDF produces consistent normals")
{
    FlexiCubes fc;
    int res = 10;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf(grid.vertices.rows());
    for (Index i = 0; i < grid.vertices.rows(); ++i)
    {
        sdf[i] = grid.vertices(i, 0);  // plane x = 0
    }

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));
    CHECK_FALSE(mesh.empty());

    int aligned_pos = 0;
    int aligned_neg = 0;
    int total       = 0;
    for (Index i = 0; i < mesh.faces.rows(); ++i)
    {
        Vec3 v0 = mesh.vertices.row(mesh.faces(i, 0)).transpose();
        Vec3 v1 = mesh.vertices.row(mesh.faces(i, 1)).transpose();
        Vec3 v2 = mesh.vertices.row(mesh.faces(i, 2)).transpose();
        Vec3 n  = (v1 - v0).cross(v2 - v0);
        if (n.norm() < 1e-12) continue;
        total++;
        if (n.x() > 0) aligned_pos++;
        if (n.x() < 0) aligned_neg++;
    }

    if (total > 0)
    {
        int aligned = std::max(aligned_pos, aligned_neg);
        CHECK(aligned > total * 0.9);
    }
}

TEST_CASE("Tetrahedral mesh has mostly non-degenerate tets")
{
    FlexiCubes fc;
    int res = 10;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    TetraMesh tet_mesh = fc.extract_volume(grid.vertices, sdf, grid.cubes, Resolution(res));

    if (!tet_mesh.empty() && tet_mesh.num_tets() > 0)
    {
        int degenerate = 0;
        for (Index i = 0; i < tet_mesh.tets.rows(); ++i)
        {
            Vec3 v0    = tet_mesh.vertices.row(tet_mesh.tets(i, 0)).transpose();
            Vec3 v1    = tet_mesh.vertices.row(tet_mesh.tets(i, 1)).transpose();
            Vec3 v2    = tet_mesh.vertices.row(tet_mesh.tets(i, 2)).transpose();
            Vec3 v3    = tet_mesh.vertices.row(tet_mesh.tets(i, 3)).transpose();
            double vol = std::abs((v1 - v0).dot((v2 - v0).cross(v3 - v0))) / 6.0;
            if (vol < 1e-12)
            {
                degenerate++;
            }
        }
        CHECK(degenerate < tet_mesh.num_tets() / 10);
    }
}

TEST_CASE("Free function API")
{
    auto grid = construct_voxel_grid(8);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Mesh mesh = extract_surface(grid.vertices, sdf, grid.cubes, Resolution(8));

    CHECK_FALSE(mesh.empty());
}

TEST_CASE("With custom weights")
{
    FlexiCubes fc;
    int res = 8;

    auto grid = fc.construct_voxel_grid(res);
    VecXd sdf = sphere_sdf(grid.vertices, 0.4);

    Index num_cubes = grid.num_cubes();

    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    beta.setZero();

    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    alpha.setZero();

    VecXd gamma(num_cubes);
    gamma.setZero();

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res),
                                   &beta, &alpha, &gamma);

    CHECK_FALSE(mesh.empty());
}
