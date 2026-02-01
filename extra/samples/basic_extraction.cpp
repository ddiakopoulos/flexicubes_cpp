#include <flexicubes/flexicubes.hpp>
#include "../common/obj_loader.hpp"

#include <iostream>
#include <fstream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace flexi;

int main(int argc, char * argv[])
{
    std::cout << "FlexiCubes Basic Extraction Example\n";
    std::cout << "====================================\n\n";

    // Parse arguments
    int resolution = 32;
    if (argc > 1)
    {
        resolution = std::atoi(argv[1]);
        if (resolution < 4 || resolution > 256)
        {
            std::cerr << "Resolution must be between 4 and 256\n";
            return 1;
        }
    }

    std::cout << "Using resolution: " << resolution << "\n\n";

    // Create FlexiCubes extractor
    FlexiCubes fc;

    // Step 1: Generate voxel grid
    std::cout << "1. Generating voxel grid...\n";
    auto grid = fc.construct_voxel_grid(resolution);
    std::cout << "   Vertices: " << grid.num_vertices() << "\n";
    std::cout << "   Cubes: " << grid.num_cubes() << "\n\n";

    // Step 2: Compute SDF (sphere of radius 0.4)
    std::cout << "2. Computing SDF (sphere r=0.4)...\n";
    VecXd sdf(grid.vertices.rows());
    Vec3 center(0, 0, 0);
    double radius = 0.4;

    for (Index i = 0; i < grid.vertices.rows(); ++i)
    {
        Vec3 p = grid.vertices.row(i).transpose();
        sdf[i] = (p - center).norm() - radius;
    }

    // Count inside/outside vertices
    int inside = 0, outside = 0;
    for (Index i = 0; i < sdf.size(); ++i)
    {
        if (sdf[i] < 0) inside++;
        else
            outside++;
    }
    std::cout << "   Inside vertices: " << inside << "\n";
    std::cout << "   Outside vertices: " << outside << "\n\n";

    // Step 3: Extract surface mesh
    std::cout << "3. Extracting surface mesh...\n";
    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(resolution));

    if (mesh.empty())
    {
        std::cerr << "   No surface found!\n";
        return 1;
    }

    std::cout << "   Output vertices: " << mesh.num_vertices() << "\n";
    std::cout << "   Output faces: " << mesh.num_faces() << "\n";
    std::cout << "   L_dev values: " << mesh.l_dev.size() << "\n\n";

    // Step 4: Compute some statistics
    std::cout << "4. Mesh statistics:\n";

    // Bounding box
    Vec3 min_corner = mesh.vertices.colwise().minCoeff();
    Vec3 max_corner = mesh.vertices.colwise().maxCoeff();
    std::cout << "   Bounding box: [" << min_corner.transpose() << "] to [" << max_corner.transpose() << "]\n";

    // L_dev statistics
    if (mesh.l_dev.size() > 0)
    {
        std::cout << "   L_dev min: " << mesh.l_dev.minCoeff() << "\n";
        std::cout << "   L_dev max: " << mesh.l_dev.maxCoeff() << "\n";
        std::cout << "   L_dev mean: " << mesh.l_dev.mean() << "\n";
    }

    // Approximate surface area
    double area = 0.0;
    for (Index i = 0; i < mesh.faces.rows(); ++i)
    {
        Vec3 v0 = mesh.vertices.row(mesh.faces(i, 0)).transpose();
        Vec3 v1 = mesh.vertices.row(mesh.faces(i, 1)).transpose();
        Vec3 v2 = mesh.vertices.row(mesh.faces(i, 2)).transpose();
        area += 0.5 * (v1 - v0).cross(v2 - v0).norm();
    }
    std::cout << "   Surface area: " << area << "\n";
    std::cout << "   Expected (4*pi*r^2): " << 4.0 * M_PI * radius * radius << "\n\n";

    // Step 5: Write output
    std::cout << "5. Writing output...\n";
    save_obj("sphere.obj", mesh);

    // Also test tetrahedral mesh extraction
    std::cout << "\n6. Extracting tetrahedral mesh...\n";
    TetraMesh tet_mesh = fc.extract_volume(grid.vertices, sdf, grid.cubes, Resolution(resolution));

    if (!tet_mesh.empty())
    {
        std::cout << "   Tet vertices: " << tet_mesh.num_vertices() << "\n";
        std::cout << "   Tetrahedra: " << tet_mesh.num_tets() << "\n";
    }
    else
    {
        std::cout << "   No tetrahedra generated\n";
    }

    std::cout << "\nDone!\n";
    return 0;
}
