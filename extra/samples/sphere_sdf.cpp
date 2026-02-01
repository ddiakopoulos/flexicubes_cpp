#include <flexicubes/flexicubes.hpp>
#include <common/obj_loader.hpp>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace flexi;

/**
 * Compute sphere SDF at given positions.
 *
 * @param positions Nx3 matrix of query positions
 * @param center Sphere center
 * @param radius Sphere radius
 * @return SDF values (negative inside, positive outside)
 */
VecXd sphere_sdf(const MatX3 & positions, const Vec3 & center, double radius)
{
    VecXd sdf(positions.rows());
    for (Index i = 0; i < positions.rows(); ++i)
    {
        Vec3 p = positions.row(i).transpose();
        sdf[i] = (p - center).norm() - radius;
    }
    return sdf;
}

/**
 * Compute sphere gradient (normalized).
 */
MatX3 sphere_gradient(const MatX3 & positions, const Vec3 & center)
{
    MatX3 grad(positions.rows(), 3);
    for (Index i = 0; i < positions.rows(); ++i)
    {
        Vec3 p     = positions.row(i).transpose();
        Vec3 d     = p - center;
        double len = d.norm();
        if (len > 1e-10)
        {
            grad.row(i) = (d / len).transpose();
        }
        else
        {
            grad.row(i) = Vec3(1, 0, 0).transpose();
        }
    }
    return grad;
}

/**
 * Compute box SDF at given positions.
 */
VecXd box_sdf(const MatX3 & positions, const Vec3 & center, const Vec3 & half_extents)
{
    VecXd sdf(positions.rows());
    for (Index i = 0; i < positions.rows(); ++i)
    {
        Vec3 p     = positions.row(i).transpose();
        Vec3 q     = (p - center).cwiseAbs() - half_extents;
        Vec3 q_max = q.cwiseMax(0.0);
        sdf[i]     = q_max.norm() + std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0);
    }
    return sdf;
}

/**
 * Compute torus SDF at given positions.
 */
VecXd torus_sdf(const MatX3 & positions, const Vec3 & center, double major_radius, double minor_radius)
{
    VecXd sdf(positions.rows());
    for (Index i = 0; i < positions.rows(); ++i)
    {
        Vec3 p     = positions.row(i).transpose() - center;
        double q_x = std::sqrt(p.x() * p.x() + p.z() * p.z()) - major_radius;
        sdf[i]     = std::sqrt(q_x * q_x + p.y() * p.y()) - minor_radius;
    }
    return sdf;
}

int main()
{
    std::cout << "FlexiCubes Sphere SDF Example\n";
    std::cout << "==============================\n\n";

    // Create FlexiCubes extractor
    FlexiCubes fc;

    // Test different resolutions
    for (int res : {8, 16, 32, 64})
    {
        std::cout << "Resolution: " << res << "x" << res << "x" << res << "\n";

        // Generate voxel grid
        auto grid = fc.construct_voxel_grid(res);
        std::cout << "  Grid vertices: " << grid.num_vertices() << "\n";
        std::cout << "  Grid cubes: " << grid.num_cubes() << "\n";

        // Compute sphere SDF (radius 0.4, centered at origin)
        Vec3 center(0, 0, 0);
        double radius = 0.4;
        VecXd sdf     = sphere_sdf(grid.vertices, center, radius);

        // Extract mesh
        Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));

        std::cout << "  Mesh vertices: " << mesh.num_vertices() << "\n";
        std::cout << "  Mesh faces: " << mesh.num_faces() << "\n";

        if (!mesh.empty())
        {
            // Compute mesh quality
            auto quality = compute_mesh_quality(mesh.vertices, mesh.faces,
                                                DualVertexResult {mesh.vertices, mesh.l_dev, {}, {}, mesh.vertices.rows()});
            std::cout << "  Mean regularity: " << quality.mean_regularity << "\n";
            std::cout << "  Mean L_dev: " << quality.mean_l_dev << "\n";
        }
        std::cout << "\n";
    }

    // Test with box SDF
    std::cout << "Box SDF Example\n";
    std::cout << "---------------\n";
    {
        int res   = 32;
        auto grid = fc.construct_voxel_grid(res);
        Vec3 center(0, 0, 0);
        Vec3 half_extents(0.3, 0.3, 0.3);
        VecXd sdf = box_sdf(grid.vertices, center, half_extents);

        Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));
        std::cout << "  Mesh vertices: " << mesh.num_vertices() << "\n";
        std::cout << "  Mesh faces: " << mesh.num_faces() << "\n\n";
    }

    // Test with torus SDF
    std::cout << "Torus SDF Example\n";
    std::cout << "-----------------\n";
    {
        int res   = 32;
        auto grid = fc.construct_voxel_grid(res);
        Vec3 center(0, 0, 0);
        double major_r = 0.3;
        double minor_r = 0.1;
        VecXd sdf      = torus_sdf(grid.vertices, center, major_r, minor_r);

        Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(res));
        std::cout << "  Mesh vertices: " << mesh.num_vertices() << "\n";
        std::cout << "  Mesh faces: " << mesh.num_faces() << "\n\n";
    }

    // Save a sphere mesh to OBJ for visual inspection
    std::cout << "Saving sphere to OBJ\n";
    std::cout << "--------------------\n";
    {
        int res   = 32;
        auto grid = fc.construct_voxel_grid(res);
        Vec3 center(0, 0, 0);
        double radius  = 0.4;
        VecXd sdf_vals = sphere_sdf(grid.vertices, center, radius);

        Mesh mesh = fc.extract_surface(grid.vertices, sdf_vals, grid.cubes, Resolution(res));
        std::cout << "  Mesh vertices: " << mesh.num_vertices() << "\n";
        std::cout << "  Mesh faces: " << mesh.num_faces() << "\n";

        save_obj("output/sphere_r32.obj", mesh);
        std::cout << "  Saved: output/sphere_r32.obj\n\n";
    }

    std::cout << "Done!\n";
    return 0;
}
