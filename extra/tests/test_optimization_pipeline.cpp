#include <doctest/doctest.h>
#include <flexicubes/flexicubes.hpp>
#include <adept.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <filesystem>

using namespace flexi;
namespace fs = std::filesystem;

extern bool g_run_slow_tests;

static bool run_slow_tests()
{
    return g_run_slow_tests;
}

// =============================================================================
// Simple OBJ Loader
// =============================================================================

struct ObjMesh
{
    std::vector<Vec3> vertices;
    std::vector<Vec3i> faces;
    std::string name;

    Index num_vertices() const { return static_cast<Index>(vertices.size()); }
    Index num_faces() const { return static_cast<Index>(faces.size()); }
    bool empty() const { return vertices.empty(); }

    // Compute axis-aligned bounding box
    void compute_bounds(Vec3 & min_corner, Vec3 & max_corner) const
    {
        if (vertices.empty())
        {
            min_corner = max_corner = Vec3::Zero();
            return;
        }
        min_corner = vertices[0];
        max_corner = vertices[0];
        for (const auto & v : vertices)
        {
            min_corner = min_corner.cwiseMin(v);
            max_corner = max_corner.cwiseMax(v);
        }
    }

    // Compute surface area
    double compute_surface_area() const
    {
        double area = 0.0;
        for (const auto & f : faces)
        {
            Vec3 v0 = vertices[f[0]];
            Vec3 v1 = vertices[f[1]];
            Vec3 v2 = vertices[f[2]];
            area += 0.5 * (v1 - v0).cross(v2 - v0).norm();
        }
        return area;
    }

    // Compute volume (assumes watertight mesh)
    double compute_volume() const
    {
        double vol = 0.0;
        for (const auto & f : faces)
        {
            Vec3 v0 = vertices[f[0]];
            Vec3 v1 = vertices[f[1]];
            Vec3 v2 = vertices[f[2]];
            vol += v0.dot(v1.cross(v2)) / 6.0;
        }
        return std::abs(vol);
    }

    // Center mesh at origin
    void center()
    {
        Vec3 min_c, max_c;
        compute_bounds(min_c, max_c);
        Vec3 center = (min_c + max_c) / 2.0;
        for (auto & v : vertices)
        {
            v -= center;
        }
    }

    // Scale to fit in unit cube [-0.45, 0.45]^3
    void normalize_scale()
    {
        Vec3 min_c, max_c;
        compute_bounds(min_c, max_c);
        Vec3 extent       = max_c - min_c;
        double max_extent = extent.maxCoeff();
        if (max_extent > 1e-10)
        {
            double scale = 0.9 / max_extent;
            for (auto & v : vertices)
            {
                v *= scale;
            }
        }
    }

    // Build face normals
    std::vector<Vec3> compute_face_normals() const
    {
        std::vector<Vec3> normals;
        normals.reserve(faces.size());
        for (const auto & f : faces)
        {
            Vec3 v0    = vertices[f[0]];
            Vec3 v1    = vertices[f[1]];
            Vec3 v2    = vertices[f[2]];
            Vec3 n     = (v1 - v0).cross(v2 - v0);
            double len = n.norm();
            if (len > 1e-10)
            {
                normals.push_back(n / len);
            }
            else
            {
                normals.push_back(Vec3::UnitZ());
            }
        }
        return normals;
    }
};

// Load OBJ file
ObjMesh load_obj(const std::string & filepath)
{
    ObjMesh mesh;
    mesh.name = fs::path(filepath).stem().string();

    std::ifstream file(filepath);
    if (!file.is_open())
    {
        return mesh;  // Return empty mesh
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v")
        {
            Vec3 v;
            iss >> v[0] >> v[1] >> v[2];
            mesh.vertices.push_back(v);
        }
        else if (prefix == "f")
        {
            std::vector<int> indices;
            std::string token;
            while (iss >> token)
            {
                // Parse vertex index (handle v, v/vt, v/vt/vn, v//vn formats)
                size_t slash_pos = token.find('/');
                int idx;
                if (slash_pos != std::string::npos)
                {
                    idx = std::stoi(token.substr(0, slash_pos));
                }
                else
                {
                    idx = std::stoi(token);
                }
                // OBJ indices are 1-based
                indices.push_back(idx - 1);
            }
            // Triangulate if necessary (fan triangulation)
            for (size_t i = 1; i + 1 < indices.size(); ++i)
            {
                mesh.faces.push_back(Vec3i(indices[0], indices[i], indices[i + 1]));
            }
        }
    }

    return mesh;
}

// =============================================================================
// SDF Computation from Mesh
// =============================================================================

// Compute signed distance from a point to a triangle
double point_triangle_distance(const Vec3 & p, const Vec3 & a, const Vec3 & b, const Vec3 & c)
{
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    Vec3 ap = p - a;

    double d1 = ab.dot(ap);
    double d2 = ac.dot(ap);
    if (d1 <= 0.0 && d2 <= 0.0) return (p - a).norm();

    Vec3 bp   = p - b;
    double d3 = ab.dot(bp);
    double d4 = ac.dot(bp);
    if (d3 >= 0.0 && d4 <= d3) return (p - b).norm();

    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        double v = d1 / (d1 - d3);
        return (p - (a + v * ab)).norm();
    }

    Vec3 cp   = p - c;
    double d5 = ab.dot(cp);
    double d6 = ac.dot(cp);
    if (d6 >= 0.0 && d5 <= d6) return (p - c).norm();

    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        double w = d2 / (d2 - d6);
        return (p - (a + w * ac)).norm();
    }

    double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + w * (c - b))).norm();
    }

    double denom = 1.0 / (va + vb + vc);
    double v     = vb * denom;
    double w     = vc * denom;
    return (p - (a + ab * v + ac * w)).norm();
}

// Compute unsigned distance field from mesh
VecXd compute_udf(const MatX3 & query_points, const ObjMesh & mesh)
{
    VecXd udf(query_points.rows());

    for (Index i = 0; i < query_points.rows(); ++i)
    {
        Vec3 p          = query_points.row(i).transpose();
        double min_dist = std::numeric_limits<double>::max();

        for (const auto & f : mesh.faces)
        {
            double dist = point_triangle_distance(p, mesh.vertices[f[0]],
                                                  mesh.vertices[f[1]],
                                                  mesh.vertices[f[2]]);
            min_dist    = std::min(min_dist, dist);
        }
        udf[i] = min_dist;
    }

    return udf;
}

// Compute approximate signed distance (using ray casting for sign)
VecXd compute_sdf(const MatX3 & query_points, const ObjMesh & mesh)
{
    VecXd udf = compute_udf(query_points, mesh);

    // Use pseudo-normal method for sign
    std::vector<Vec3> face_normals = mesh.compute_face_normals();

    for (Index i = 0; i < query_points.rows(); ++i)
    {
        Vec3 p = query_points.row(i).transpose();

        // Find closest face
        double min_dist  = std::numeric_limits<double>::max();
        int closest_face = -1;
        Vec3 closest_point;

        for (size_t fi = 0; fi < mesh.faces.size(); ++fi)
        {
            const auto & f = mesh.faces[fi];
            Vec3 a         = mesh.vertices[f[0]];
            Vec3 b         = mesh.vertices[f[1]];
            Vec3 c         = mesh.vertices[f[2]];

            // Project onto triangle plane
            Vec3 n      = face_normals[fi];
            double dist = point_triangle_distance(p, a, b, c);

            if (dist < min_dist)
            {
                min_dist     = dist;
                closest_face = static_cast<int>(fi);
            }
        }

        if (closest_face >= 0)
        {
            // Use face normal to determine sign
            const auto & f = mesh.faces[closest_face];
            Vec3 centroid  = (mesh.vertices[f[0]] + mesh.vertices[f[1]] + mesh.vertices[f[2]]) / 3.0;
            Vec3 to_point  = p - centroid;

            if (to_point.dot(face_normals[closest_face]) < 0)
            {
                udf[i] = -udf[i];  // Inside
            }
        }
    }

    return udf;
}

// =============================================================================
// Statistics Computation
// =============================================================================

struct MeshStatistics
{
    // Geometry stats
    Index num_vertices  = 0;
    Index num_faces     = 0;
    double surface_area = 0.0;
    Vec3 bbox_min       = Vec3::Zero();
    Vec3 bbox_max       = Vec3::Zero();
    Vec3 bbox_extent    = Vec3::Zero();
    Vec3 centroid       = Vec3::Zero();

    // Quality stats
    double min_edge_length  = 0.0;
    double max_edge_length  = 0.0;
    double mean_edge_length = 0.0;
    double min_face_area    = 0.0;
    double max_face_area    = 0.0;
    double mean_face_area   = 0.0;
    int degenerate_faces    = 0;

    // L_dev stats (for FlexiCubes output)
    double l_dev_min  = 0.0;
    double l_dev_max  = 0.0;
    double l_dev_mean = 0.0;
    double l_dev_std  = 0.0;
    double l_dev_sum  = 0.0;

    void print(const std::string & label = "") const
    {
        if (!label.empty())
        {
            std::cout << "=== " << label << " ===\n";
        }
        std::cout << "  Vertices: " << num_vertices << ", Faces: " << num_faces << "\n";
        std::cout << "  Surface area: " << surface_area << "\n";
        std::cout << "  BBox: [" << bbox_min.transpose() << "] to [" << bbox_max.transpose() << "]\n";
        std::cout << "  BBox extent: [" << bbox_extent.transpose() << "]\n";
        std::cout << "  Centroid: [" << centroid.transpose() << "]\n";
        std::cout << "  Edge lengths: min=" << min_edge_length << ", max=" << max_edge_length
                  << ", mean=" << mean_edge_length << "\n";
        std::cout << "  Face areas: min=" << min_face_area << ", max=" << max_face_area
                  << ", mean=" << mean_face_area << "\n";
        std::cout << "  Degenerate faces: " << degenerate_faces << "\n";
        if (l_dev_sum > 0)
        {
            std::cout << "  L_dev: min=" << l_dev_min << ", max=" << l_dev_max
                      << ", mean=" << l_dev_mean << ", std=" << l_dev_std
                      << ", sum=" << l_dev_sum << "\n";
        }
    }
};

MeshStatistics compute_mesh_statistics(const Mesh & mesh)
{
    MeshStatistics stats;
    stats.num_vertices = mesh.num_vertices();
    stats.num_faces    = mesh.num_faces();

    if (mesh.empty()) return stats;

    // Bounding box and centroid
    stats.bbox_min    = mesh.vertices.colwise().minCoeff();
    stats.bbox_max    = mesh.vertices.colwise().maxCoeff();
    stats.bbox_extent = stats.bbox_max - stats.bbox_min;
    stats.centroid    = mesh.vertices.colwise().mean();

    // Edge and face statistics
    std::vector<double> edge_lengths;
    std::vector<double> face_areas;
    stats.surface_area = 0.0;

    for (Index i = 0; i < mesh.faces.rows(); ++i)
    {
        Vec3 v0 = mesh.vertices.row(mesh.faces(i, 0)).transpose();
        Vec3 v1 = mesh.vertices.row(mesh.faces(i, 1)).transpose();
        Vec3 v2 = mesh.vertices.row(mesh.faces(i, 2)).transpose();

        double e0 = (v1 - v0).norm();
        double e1 = (v2 - v1).norm();
        double e2 = (v0 - v2).norm();

        edge_lengths.push_back(e0);
        edge_lengths.push_back(e1);
        edge_lengths.push_back(e2);

        Vec3 cross  = (v1 - v0).cross(v2 - v0);
        double area = 0.5 * cross.norm();
        face_areas.push_back(area);
        stats.surface_area += area;

        if (area < 1e-12)
        {
            stats.degenerate_faces++;
        }
    }

    if (!edge_lengths.empty())
    {
        stats.min_edge_length  = *std::min_element(edge_lengths.begin(), edge_lengths.end());
        stats.max_edge_length  = *std::max_element(edge_lengths.begin(), edge_lengths.end());
        stats.mean_edge_length = std::accumulate(edge_lengths.begin(), edge_lengths.end(), 0.0) /
                                 edge_lengths.size();
    }

    if (!face_areas.empty())
    {
        stats.min_face_area  = *std::min_element(face_areas.begin(), face_areas.end());
        stats.max_face_area  = *std::max_element(face_areas.begin(), face_areas.end());
        stats.mean_face_area = std::accumulate(face_areas.begin(), face_areas.end(), 0.0) /
                               face_areas.size();
    }

    // L_dev statistics
    if (mesh.l_dev.size() > 0)
    {
        stats.l_dev_min  = mesh.l_dev.minCoeff();
        stats.l_dev_max  = mesh.l_dev.maxCoeff();
        stats.l_dev_mean = mesh.l_dev.mean();
        stats.l_dev_sum  = mesh.l_dev.sum();

        double sq_sum = 0.0;
        for (Index i = 0; i < mesh.l_dev.size(); ++i)
        {
            double diff = mesh.l_dev[i] - stats.l_dev_mean;
            sq_sum += diff * diff;
        }
        stats.l_dev_std = std::sqrt(sq_sum / mesh.l_dev.size());
    }

    return stats;
}

// =============================================================================
// Optimization Pipeline Statistics
// =============================================================================

struct OptimizationResult
{
    std::string mesh_name;
    int resolution;
    int num_iterations;
    double initial_loss;
    double final_loss;
    double loss_reduction_pct;
    double time_ms;
    MeshStatistics initial_stats;
    MeshStatistics final_stats;
    std::vector<double> loss_history;
    std::vector<double> l_dev_history;
    bool converged;
    std::string error_message;

    void print() const
    {
        std::cout << "\n========================================\n";
        std::cout << "Optimization Result: " << mesh_name << "\n";
        std::cout << "========================================\n";
        std::cout << "Resolution: " << resolution << "\n";
        std::cout << "Iterations: " << num_iterations << "\n";
        std::cout << "Time: " << time_ms << " ms\n";
        std::cout << "Initial loss: " << initial_loss << "\n";
        std::cout << "Final loss: " << final_loss << "\n";
        std::cout << "Loss reduction: " << loss_reduction_pct << "%\n";
        std::cout << "Converged: " << (converged ? "yes" : "no") << "\n";
        if (!error_message.empty())
        {
            std::cout << "Error: " << error_message << "\n";
        }
        std::cout << "\nInitial mesh:\n";
        initial_stats.print();
        std::cout << "\nFinal mesh:\n";
        final_stats.print();
    }
};

// =============================================================================
// Full Optimization Pipeline
// =============================================================================

OptimizationResult run_optimization_pipeline(
    const ObjMesh & target_mesh,
    int resolution               = 64,
    int max_iterations           = 256,
    double learning_rate         = 0.01,
    double convergence_threshold = 1e-2)
{
    OptimizationResult result;
    result.mesh_name  = target_mesh.name;
    result.resolution = resolution;
    result.converged  = false;

    if (target_mesh.empty())
    {
        result.error_message = "Empty target mesh";
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    try
    {
        FlexiCubes fc;

        // Generate voxel grid
        auto grid = fc.construct_voxel_grid(resolution);

        // Compute SDF from target mesh
        VecXd sdf = compute_sdf(grid.vertices, target_mesh);

        // Initialize learnable weights
        Index num_cubes = grid.num_cubes();
        Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
        Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
        VecXd gamma(num_cubes);

        beta.setZero();
        alpha.setZero();
        gamma.setZero();

        // Initial extraction
        Gradients grads;
        auto [initial_mesh, initial_loss] = fc.extract_surface_with_grads(
            grid.vertices, sdf, grid.cubes, Resolution(resolution),
            beta, alpha, gamma, &grads);

        result.initial_loss  = initial_loss;
        result.initial_stats = compute_mesh_statistics(initial_mesh);
        result.loss_history.push_back(initial_loss);
        result.l_dev_history.push_back(result.initial_stats.l_dev_sum);

        if (initial_mesh.empty())
        {
            result.error_message  = "Initial extraction produced empty mesh";
            result.final_loss     = initial_loss;
            result.final_stats    = result.initial_stats;
            result.num_iterations = 0;
            return result;
        }

        // Optimization loop
        double prev_loss = initial_loss;
        int iter         = 0;

        for (iter = 0; iter < max_iterations; ++iter)
        {
            // Gradient descent step
            beta -= learning_rate * grads.d_beta;
            alpha -= learning_rate * grads.d_alpha;
            gamma -= learning_rate * grads.d_gamma;

            // Clamp weights to reasonable range
            beta  = beta.cwiseMax(-2.0).cwiseMin(2.0);
            alpha = alpha.cwiseMax(-2.0).cwiseMin(2.0);
            gamma = gamma.cwiseMax(-2.0).cwiseMin(2.0);

            // Re-extract with new weights
            auto [mesh, loss] = fc.extract_surface_with_grads(
                grid.vertices, sdf, grid.cubes, Resolution(resolution),
                beta, alpha, gamma, &grads);

            result.loss_history.push_back(loss);

            if (!mesh.empty())
            {
                result.l_dev_history.push_back(mesh.l_dev.sum());
            }

            // Check convergence
            double loss_change = std::abs(prev_loss - loss);
            if (loss_change < convergence_threshold && iter > 5)
            {
                result.converged = true;
                break;
            }

            prev_loss = loss;
        }

        result.num_iterations = iter + 1;

        // Final extraction
        auto [final_mesh, final_loss] = fc.extract_surface_with_grads(
            grid.vertices, sdf, grid.cubes, Resolution(resolution),
            beta, alpha, gamma, &grads);

        result.final_loss  = final_loss;
        result.final_stats = compute_mesh_statistics(final_mesh);

        if (result.initial_loss > 1e-10)
        {
            result.loss_reduction_pct = 100.0 * (result.initial_loss - result.final_loss) / result.initial_loss;
        }
    }
    catch (const std::exception & e)
    {
        result.error_message = e.what();
    }

    auto end_time  = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return result;
}

// =============================================================================
// Tests
// =============================================================================

// Get path to assets directory
static std::string get_assets_path()
{
    // Try relative paths from likely build locations
    std::vector<std::string> candidates = {
        "../../assets",
        "../assets",
        "assets",
        "../../../assets",
        "../../../../flexicubes-cpp/assets"};

    for (const auto & path : candidates)
    {
        if (fs::exists(path) && fs::is_directory(path))
        {
            return path;
        }
    }

    // Try absolute path
    std::string abs_path = "C:/Users/Dimitri Diakopoulos/Desktop/flexi/flexicubes-cpp/assets";
    if (fs::exists(abs_path))
    {
        return abs_path;
    }

    return "";
}

TEST_CASE("OBJ loader - basic functionality")
{
    if (!run_slow_tests())
    {
        WARN("Re-run with --run-slow-tests (or --slow) to enable slow tests.");
        return;
    }
    std::string assets_path = get_assets_path();
    REQUIRE_FALSE(assets_path.empty());

    std::string obj_path = assets_path + "/CapsuleUniform.obj";
    REQUIRE(fs::exists(obj_path));

    ObjMesh mesh = load_obj(obj_path);

    CHECK_FALSE(mesh.empty());
    CHECK(mesh.num_vertices() > 0);
    CHECK(mesh.num_faces() > 0);
    CHECK(mesh.name == "CapsuleUniform");

    // Verify valid face indices
    for (const auto & f : mesh.faces)
    {
        CHECK(f[0] >= 0);
        CHECK(f[0] < mesh.num_vertices());
        CHECK(f[1] >= 0);
        CHECK(f[1] < mesh.num_vertices());
        CHECK(f[2] >= 0);
        CHECK(f[2] < mesh.num_vertices());
    }
}

TEST_CASE("OBJ mesh normalization")
{
    if (!run_slow_tests())
    {
        WARN("Re-run with --run-slow-tests (or --slow) to enable slow tests.");
        return;
    }
    std::string assets_path = get_assets_path();
    REQUIRE_FALSE(assets_path.empty());

    ObjMesh mesh = load_obj(assets_path + "/CapsuleUniform.obj");
    REQUIRE_FALSE(mesh.empty());

    mesh.center();
    mesh.normalize_scale();

    Vec3 min_c, max_c;
    mesh.compute_bounds(min_c, max_c);

    // Should be roughly centered
    Vec3 center = (min_c + max_c) / 2.0;
    CHECK(center.norm() < 0.1);

    // Should fit in [-0.5, 0.5]^3
    CHECK(min_c.minCoeff() >= -0.5);
    CHECK(max_c.maxCoeff() <= 0.5);
}

TEST_CASE("SDF computation from mesh")
{
    if (!run_slow_tests())
    {
        WARN("Re-run with --run-slow-tests (or --slow) to enable slow tests.");
        return;
    }
    std::string assets_path = get_assets_path();
    REQUIRE_FALSE(assets_path.empty());

    ObjMesh mesh = load_obj(assets_path + "/CapsuleUniform.obj");
    REQUIRE_FALSE(mesh.empty());

    mesh.center();
    mesh.normalize_scale();

    // Create a small test grid
    FlexiCubes fc;
    auto grid = fc.construct_voxel_grid(8);

    VecXd sdf = compute_sdf(grid.vertices, mesh);

    CHECK(sdf.size() == grid.num_vertices());

    // Should have both positive and negative values (inside and outside)
    bool has_positive = false;
    bool has_negative = false;
    for (Index i = 0; i < sdf.size(); ++i)
    {
        if (sdf[i] > 0) has_positive = true;
        if (sdf[i] < 0) has_negative = true;
    }
    CHECK(has_positive);
    CHECK(has_negative);

    // All values should be finite
    for (Index i = 0; i < sdf.size(); ++i)
    {
        CHECK(std::isfinite(sdf[i]));
    }
}

TEST_CASE("Full pipeline - CapsuleUniform")
{
    if (!run_slow_tests())
    {
        WARN("Re-run with --run-slow-tests (or --slow) to enable slow tests.");
        return;
    }
    std::string assets_path = get_assets_path();
    REQUIRE_FALSE(assets_path.empty());

    ObjMesh target = load_obj(assets_path + "/CapsuleUniform.obj");
    REQUIRE_FALSE(target.empty());

    target.center();
    target.normalize_scale();

    OptimizationResult result = run_optimization_pipeline(target, 16, 20, 0.01);

    // Basic validation
    CHECK(result.error_message.empty());
    CHECK(result.initial_stats.num_vertices > 0);
    CHECK(result.initial_stats.num_faces > 0);
    CHECK(result.final_stats.num_vertices > 0);
    CHECK(result.final_stats.num_faces > 0);

    // Loss should be finite
    CHECK(std::isfinite(result.initial_loss));
    CHECK(std::isfinite(result.final_loss));

    // L_dev should be non-negative
    CHECK(result.initial_stats.l_dev_min >= 0);
    CHECK(result.final_stats.l_dev_min >= 0);

    // Print results
    result.print();
}

TEST_CASE("Full pipeline - HexagonUniform")
{
    if (!run_slow_tests())
    {
        WARN("Re-run with --run-slow-tests (or --slow) to enable slow tests.");
        return;
    }
    std::string assets_path = get_assets_path();
    REQUIRE_FALSE(assets_path.empty());

    ObjMesh target = load_obj(assets_path + "/HexagonUniform.obj");
    REQUIRE_FALSE(target.empty());

    target.center();
    target.normalize_scale();

    OptimizationResult result = run_optimization_pipeline(target, 16, 20, 0.01);

    CHECK(result.error_message.empty());
    CHECK(result.initial_stats.num_vertices > 0);
    CHECK(result.final_stats.num_vertices > 0);
    CHECK(std::isfinite(result.final_loss));

    result.print();
}

TEST_CASE("Full pipeline - TorusKnotUniform")
{
    if (!run_slow_tests())
    {
        WARN("Re-run with --run-slow-tests (or --slow) to enable slow tests.");
        return;
    }
    std::string assets_path = get_assets_path();
    REQUIRE_FALSE(assets_path.empty());

    ObjMesh target = load_obj(assets_path + "/TorusKnotUniform.obj");
    REQUIRE_FALSE(target.empty());

    target.center();
    target.normalize_scale();

    OptimizationResult result = run_optimization_pipeline(target, 24, 30, 0.01);

    CHECK(result.error_message.empty());
    CHECK(result.initial_stats.num_vertices > 0);
    CHECK(result.final_stats.num_vertices > 0);

    // Torus knot is complex - verify we get a reasonable mesh
    CHECK(result.final_stats.num_faces > 100);
    CHECK(result.final_stats.degenerate_faces < result.final_stats.num_faces / 10);

    result.print();
}

TEST_CASE("Full pipeline - all assets")
{
    // This test is skipped by default due to long runtime
    // Run with --test-case="Full pipeline - all assets" to enable

    std::string assets_path = get_assets_path();
    REQUIRE_FALSE(assets_path.empty());

    std::vector<std::string> obj_files;
    for (const auto & entry : fs::directory_iterator(assets_path))
    {
        if (entry.path().extension() == ".obj")
        {
            obj_files.push_back(entry.path().string());
        }
    }

    std::cout << "\n\nRunning optimization on " << obj_files.size() << " OBJ files...\n\n";

    std::vector<OptimizationResult> all_results;

    for (const auto & filepath : obj_files)
    {
        std::cout << "Processing: " << fs::path(filepath).filename() << "\n";

        ObjMesh target = load_obj(filepath);
        if (target.empty())
        {
            std::cout << "  SKIP: Failed to load\n";
            continue;
        }

        target.center();
        target.normalize_scale();

        OptimizationResult result = run_optimization_pipeline(target, 24, 30, 0.01);
        all_results.push_back(result);

        std::cout << "  Vertices: " << result.final_stats.num_vertices
                  << ", Faces: " << result.final_stats.num_faces
                  << ", Loss: " << result.final_loss
                  << ", Time: " << result.time_ms << "ms\n";
    }

    // Summary statistics
    std::cout << "\n\n=== SUMMARY ===\n";
    std::cout << "Total meshes processed: " << all_results.size() << "\n";

    int converged_count = 0;
    double total_time   = 0;
    for (const auto & r : all_results)
    {
        if (r.converged) converged_count++;
        total_time += r.time_ms;
    }

    std::cout << "Converged: " << converged_count << "/" << all_results.size() << "\n";
    std::cout << "Total time: " << total_time << " ms\n";
    std::cout << "Average time per mesh: " << total_time / all_results.size() << " ms\n";
}

TEST_CASE("Statistics computation")
{
    if (!run_slow_tests())
    {
        WARN("Re-run with --run-slow-tests (or --slow) to enable slow tests.");
        return;
    }
    FlexiCubes fc;
    auto grid = fc.construct_voxel_grid(16);

    VecXd sdf(grid.num_vertices());
    for (Index i = 0; i < grid.num_vertices(); ++i)
    {
        sdf[i] = grid.vertices.row(i).norm() - 0.4;
    }

    Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(16));

    MeshStatistics stats = compute_mesh_statistics(mesh);

    CHECK(stats.num_vertices == mesh.num_vertices());
    CHECK(stats.num_faces == mesh.num_faces());
    CHECK(stats.surface_area > 0);
    CHECK(stats.min_edge_length > 0);
    CHECK(stats.max_edge_length >= stats.min_edge_length);
    CHECK(stats.mean_edge_length > 0);
    CHECK(stats.l_dev_min >= 0);
    CHECK(stats.l_dev_max >= stats.l_dev_min);
    CHECK(stats.l_dev_std >= 0);

    stats.print("Sphere mesh");
}

TEST_CASE("Optimization convergence behavior")
{
    if (!run_slow_tests())
    {
        WARN("Re-run with --run-slow-tests (or --slow) to enable slow tests.");
        return;
    }
    // Create a simple sphere target
    FlexiCubes fc;
    int res   = 16;
    auto grid = fc.construct_voxel_grid(res);

    VecXd sdf(grid.num_vertices());
    for (Index i = 0; i < grid.num_vertices(); ++i)
    {
        sdf[i] = grid.vertices.row(i).norm() - 0.4;
    }

    Index num_cubes = grid.num_cubes();
    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    // Start with perturbed weights
    beta.setRandom();
    beta *= 0.5;
    alpha.setRandom();
    alpha *= 0.5;
    gamma.setRandom();
    gamma *= 0.5;

    std::vector<double> losses;
    double lr = 0.05;

    for (int iter = 0; iter < 50; ++iter)
    {
        Gradients grads;
        auto [mesh, loss] = fc.extract_surface_with_grads(
            grid.vertices, sdf, grid.cubes, Resolution(res),
            beta, alpha, gamma, &grads);

        losses.push_back(loss);

        beta -= lr * grads.d_beta;
        alpha -= lr * grads.d_alpha;
        gamma -= lr * grads.d_gamma;
    }

    // Verify loss generally decreases
    CHECK(losses.back() <= losses.front() + 1.0);  // Allow some noise

    // Print convergence curve
    std::cout << "\nConvergence curve:\n";
    for (size_t i = 0; i < losses.size(); i += 5)
    {
        std::cout << "  Iter " << i << ": " << losses[i] << "\n";
    }
    std::cout << "  Final: " << losses.back() << "\n";
}

TEST_CASE("Gradient magnitude analysis")
{
    if (!run_slow_tests())
    {
        WARN("Re-run with --run-slow-tests (or --slow) to enable slow tests.");
        return;
    }
    FlexiCubes fc;
    int res   = 8;
    auto grid = fc.construct_voxel_grid(res);

    VecXd sdf(grid.num_vertices());
    for (Index i = 0; i < grid.num_vertices(); ++i)
    {
        sdf[i] = grid.vertices.row(i).norm() - 0.4;
    }

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

    // Compute gradient statistics
    double beta_grad_norm  = grads.d_beta.norm();
    double alpha_grad_norm = grads.d_alpha.norm();
    double gamma_grad_norm = grads.d_gamma.norm();
    double sdf_grad_norm   = grads.d_sdf.norm();
    double vert_grad_norm  = grads.d_vertices.norm();

    std::cout << "\nGradient magnitudes:\n";
    std::cout << "  d_beta norm: " << beta_grad_norm << "\n";
    std::cout << "  d_alpha norm: " << alpha_grad_norm << "\n";
    std::cout << "  d_gamma norm: " << gamma_grad_norm << "\n";
    std::cout << "  d_sdf norm: " << sdf_grad_norm << "\n";
    std::cout << "  d_vertices norm: " << vert_grad_norm << "\n";

    // All gradient norms should be finite
    CHECK(std::isfinite(beta_grad_norm));
    CHECK(std::isfinite(alpha_grad_norm));
    CHECK(std::isfinite(gamma_grad_norm));
    CHECK(std::isfinite(sdf_grad_norm));
    CHECK(std::isfinite(vert_grad_norm));

    // Count non-zero gradients
    int nonzero_beta = 0, nonzero_alpha = 0, nonzero_gamma = 0;
    for (Index i = 0; i < num_cubes; ++i)
    {
        for (int j = 0; j < 12; ++j)
        {
            if (std::abs(grads.d_beta(i, j)) > 1e-10) nonzero_beta++;
        }
        for (int j = 0; j < 8; ++j)
        {
            if (std::abs(grads.d_alpha(i, j)) > 1e-10) nonzero_alpha++;
        }
        if (std::abs(grads.d_gamma[i]) > 1e-10) nonzero_gamma++;
    }

    std::cout << "  Non-zero beta gradients: " << nonzero_beta << "/" << num_cubes * 12 << "\n";
    std::cout << "  Non-zero alpha gradients: " << nonzero_alpha << "/" << num_cubes * 8 << "\n";
    std::cout << "  Non-zero gamma gradients: " << nonzero_gamma << "/" << num_cubes << "\n";
}

TEST_CASE("Resolution scaling behavior")
{
    if (!run_slow_tests())
    {
        WARN("Re-run with --run-slow-tests (or --slow) to enable slow tests.");
        return;
    }
    std::string assets_path = get_assets_path();
    if (assets_path.empty())
    {
        WARN("Assets path not found, skipping resolution scaling test");
        return;
    }

    ObjMesh target = load_obj(assets_path + "/CapsuleUniform.obj");
    if (target.empty())
    {
        WARN("Could not load CapsuleUniform.obj, skipping test");
        return;
    }

    target.center();
    target.normalize_scale();

    std::vector<int> resolutions = {8, 16, 24};
    std::vector<OptimizationResult> results;

    std::cout << "\nResolution scaling analysis:\n";

    for (int res : resolutions)
    {
        OptimizationResult result = run_optimization_pipeline(target, res, 20, 0.01);
        results.push_back(result);

        std::cout << "  Res " << res << ": "
                  << result.final_stats.num_vertices << " verts, "
                  << result.final_stats.num_faces << " faces, "
                  << "L_dev=" << result.final_stats.l_dev_sum << ", "
                  << "Time=" << result.time_ms << "ms\n";
    }

    // Higher resolution should give more vertices/faces
    CHECK(results[1].final_stats.num_vertices >= results[0].final_stats.num_vertices);
    CHECK(results[2].final_stats.num_vertices >= results[1].final_stats.num_vertices);
}
