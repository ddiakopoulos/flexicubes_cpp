#include <flexicubes/flexicubes.hpp>
#include <common/obj_loader.hpp>
#include <common/perf_timer.hpp>
#include <tmd/TriangleMeshDistance.h>

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <unordered_map>

namespace fs = std::filesystem;

using namespace flexi;

// Command-line options
struct CmdOptions
{
    int resolution = 64;
    std::string input_file;
    std::string output_dir = "output";
    std::string assets_dir = "../../assets";
    bool process_all       = true;
    bool verbose           = true;
    bool export_tet_mesh   = true;
    bool tet_boundary_only = false;
    bool optimize          = true;
    int opt_iterations     = 32;
    double opt_lr          = 0.005;
    double opt_convergence = 1e-4;
    double opt_clamp       = 2.0;
    double opt_l_dev_weight = 1.0;
    int opt_report_every   = 1;
};

void print_usage(const char * program)
{
    std::cout << "Usage: " << program << " [options]\n"
              << "\n"
              << "Mesh-to-SDF Test Runner for FlexiCubes\n"
              << "Loads OBJ meshes, computes SDF, and reconstructs using FlexiCubes.\n"
              << "\n"
              << "Options:\n"
              << "  --resolution N    Grid resolution (default: 64)\n"
              << "  --input FILE      Process a single OBJ file\n"
              << "  --all             Process all OBJ files in assets directory\n"
              << "  --output DIR      Output directory (default: output/)\n"
              << "  --assets DIR      Assets directory (default: assets/)\n"
              << "  --tet-mesh        Export each tetrahedron as a triangle mesh\n"
              << "  --tet-boundary    Export only boundary faces of tetrahedra\n"
              << "  --optimize        Run gradient-based optimization on weights\n"
              << "  --opt-iters N     Optimization iterations (default: 256)\n"
              << "  --opt-lr X        Optimization learning rate (default: 0.01)\n"
              << "  --opt-conv X      Convergence threshold (default: 1e-2)\n"
              << "  --opt-clamp X     Clamp weights to [-X, X] (default: 2.0)\n"
              << "  --opt-l-dev X     L_dev weight (default: 1.0)\n"
              << "  --opt-report N    Print loss every N iterations (default: 10)\n"
              << "  --verbose         Enable verbose output\n"
              << "  --help            Show this help message\n"
              << "\n"
              << "Examples:\n"
              << "  " << program << " --input assets/spot_triangulated.obj --resolution 32\n"
              << "  " << program << " --all --resolution 64\n";
}

CmdOptions parse_args(int argc, char * argv[])
{
    CmdOptions opts;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "--resolution" && i + 1 < argc)
        {
            opts.resolution = std::stoi(argv[++i]);
        }
        else if (arg == "--input" && i + 1 < argc)
        {
            opts.input_file = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            opts.output_dir = argv[++i];
        }
        else if (arg == "--assets" && i + 1 < argc)
        {
            opts.assets_dir = argv[++i];
        }
        else if (arg == "--all")
        {
            opts.process_all = true;
        }
        else if (arg == "--tet-mesh")
        {
            opts.export_tet_mesh = true;
        }
        else if (arg == "--tet-boundary")
        {
            opts.export_tet_mesh = true;
            opts.tet_boundary_only = true;
        }
        else if (arg == "--optimize")
        {
            opts.optimize = true;
        }
        else if (arg == "--opt-iters" && i + 1 < argc)
        {
            opts.opt_iterations = std::stoi(argv[++i]);
        }
        else if (arg == "--opt-lr" && i + 1 < argc)
        {
            opts.opt_lr = std::stod(argv[++i]);
        }
        else if (arg == "--opt-conv" && i + 1 < argc)
        {
            opts.opt_convergence = std::stod(argv[++i]);
        }
        else if (arg == "--opt-clamp" && i + 1 < argc)
        {
            opts.opt_clamp = std::stod(argv[++i]);
        }
        else if (arg == "--opt-l-dev" && i + 1 < argc)
        {
            opts.opt_l_dev_weight = std::stod(argv[++i]);
        }
        else if (arg == "--opt-report" && i + 1 < argc)
        {
            opts.opt_report_every = std::stoi(argv[++i]);
        }
        else if (arg == "--verbose" || arg == "-v")
        {
            opts.verbose = true;
        }
        else
        {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    // Validate
    if (!opts.process_all && opts.input_file.empty())
    {
        std::cerr << "Error: Must specify --input FILE or --all\n\n";
        print_usage(argv[0]);
        std::exit(1);
    }

    if (opts.resolution < 4 || opts.resolution > 512)
    {
        std::cerr << "Error: Resolution must be between 4 and 512\n";
        std::exit(1);
    }

    return opts;
}

// Get all OBJ files in a directory
std::vector<std::string> find_obj_files(const std::string & dir)
{
    std::vector<std::string> files;

    if (!fs::exists(dir))
    {
        std::cerr << "Warning: Directory does not exist: " << dir << "\n";
        return files;
    }

    for (const auto & entry : fs::directory_iterator(dir))
    {
        if (entry.is_regular_file())
        {
            std::string path = entry.path().string();
            std::string ext  = entry.path().extension().string();
            // Case-insensitive comparison
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".obj")
            {
                files.push_back(path);
            }
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

// Process a single OBJ file
bool process_mesh(const std::string & input_path,
                  const std::string & output_path,
                  int resolution,
                  bool verbose,
                  bool export_tet_mesh,
                  bool tet_boundary_only,
                  bool optimize,
                  int opt_iterations,
                  double opt_lr,
                  double opt_convergence,
                  double opt_clamp,
                  double opt_l_dev_weight,
                  int opt_report_every)
{
    const bool log_perf           = true;
    const std::string perf_prefix = "  [perf] ";
    flexi::scoped_timer total_timer("total", std::cout, log_perf, perf_prefix);

    std::string filename = fs::path(input_path).filename().string();
    std::cout << "Processing: " << filename << "\n";

    // Step 1: Load the mesh
    ObjMesh input;
    try
    {
        flexi::scoped_timer timer("load mesh", std::cout, log_perf, perf_prefix);
        input = load_obj(input_path);
    }
    catch (const std::exception & e)
    {
        std::cerr << "  Error loading: " << e.what() << "\n";
        return false;
    }

    if (verbose)
    {
        std::cout << "  Loaded: " << input.num_vertices() << " vertices, "
                  << input.num_faces() << " faces\n";
        std::cout << "  Bounds: [" << input.min_bound.transpose() << "] to ["
                  << input.max_bound.transpose() << "]\n";
    }

    // Step 2: Normalize to unit cube
    double scale = 1.0;
    {
        flexi::scoped_timer timer("normalize mesh", std::cout, log_perf, perf_prefix);
        scale = input.normalize_to_unit_cube();
    }
    if (verbose)
    {
        std::cout << "  Normalized with scale: " << scale << "\n";
    }

    // Step 3: Build TriangleMeshDistance query structure
    // Convert Eigen data to flat arrays for TMD
    std::vector<double> verts_flat(input.num_vertices() * 3);
    std::vector<int> faces_flat(input.num_faces() * 3);

    {
        flexi::scoped_timer timer("flatten mesh data", std::cout, log_perf, perf_prefix);
        for (Index i = 0; i < input.num_vertices(); ++i)
        {
            verts_flat[i * 3 + 0] = input.vertices(i, 0);
            verts_flat[i * 3 + 1] = input.vertices(i, 1);
            verts_flat[i * 3 + 2] = input.vertices(i, 2);
        }

        for (Index i = 0; i < input.num_faces(); ++i)
        {
            faces_flat[i * 3 + 0] = input.faces(i, 0);
            faces_flat[i * 3 + 1] = input.faces(i, 1);
            faces_flat[i * 3 + 2] = input.faces(i, 2);
        }
    }

    std::unique_ptr<tmd::TriangleMeshDistance> mesh_dist;
    {
        flexi::scoped_timer timer("build TMD", std::cout, log_perf, perf_prefix);
        mesh_dist = std::make_unique<tmd::TriangleMeshDistance>(
            verts_flat.data(),
            static_cast<size_t>(input.num_vertices()),
            faces_flat.data(),
            static_cast<size_t>(input.num_faces()));
    }

    if (verbose)
    {
        std::cout << "  Mesh is " << (mesh_dist->is_mesh_manifold() ? "manifold" : "non-manifold") << "\n";
    }

    // Step 4: Create FlexiCubes grid
    FlexiCubes fc;
    VoxelGrid grid;
    {
        flexi::scoped_timer timer("construct grid", std::cout, log_perf, perf_prefix);
        grid = fc.construct_voxel_grid(resolution);
    }

    if (verbose)
    {
        std::cout << "  Grid: " << grid.num_vertices() << " vertices, "
                  << grid.num_cubes() << " cubes\n";
    }

    // Step 5: Compute SDF at each grid vertex
    VecXd sdf(grid.vertices.rows());
    {
        flexi::scoped_timer timer("compute SDF", std::cout, log_perf, perf_prefix);
        for (Index i = 0; i < grid.vertices.rows(); ++i)
        {
            double x = grid.vertices(i, 0);
            double y = grid.vertices(i, 1);
            double z = grid.vertices(i, 2);

            auto result = mesh_dist->signed_distance({x, y, z});
            sdf[i]      = result.distance;
        }
    }

    if (verbose)
    {
        double sdf_min = sdf.minCoeff();
        double sdf_max = sdf.maxCoeff();
        std::cout << "  SDF range: [" << sdf_min << ", " << sdf_max << "]\n";
    }

    // Step 6: Extract mesh using FlexiCubes
    // Use training mode for better mesh quality (4 triangles per quad instead of 2)
    flexi::Options fc_opts;
    fc_opts.training = true;  // More robust triangulation
    Mesh output;

    if (optimize)
    {
        Index num_cubes = grid.num_cubes();
        Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
        Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
        VecXd gamma(num_cubes);

        beta.setZero();
        alpha.setZero();
        gamma.setZero();

        Gradients grads;
        double loss = 0.0;

        {
            flexi::scoped_timer timer("extract surface (init)", std::cout, log_perf, perf_prefix);
            auto [mesh, loss_value] = fc.extract_surface_with_grads(
                grid.vertices, sdf, grid.cubes, Resolution(resolution),
                beta, alpha, gamma, &grads, opt_l_dev_weight, fc_opts);
            output = mesh;
            loss = loss_value;
        }

        if (!output.empty())
        {
            std::string pre_opt_name = fs::path(output_path).stem().string() + "_preopt.obj";
            std::string pre_opt_path = (fs::path(output_path).parent_path() / pre_opt_name).string();
            try
            {
                flexi::scoped_timer timer("save pre-opt mesh", std::cout, log_perf, perf_prefix);
                save_obj(pre_opt_path, output);
                std::cout << "  Saved: " << pre_opt_path << "\n";
            }
            catch (const std::exception & e)
            {
                std::cerr << "  Error saving pre-opt mesh: " << e.what() << "\n";
                return false;
            }
        }

        if (output.empty())
        {
            std::cerr << "  Warning: Initial extraction produced empty mesh\n";
            return false;
        }

        double prev_loss = loss;

        if (opt_report_every > 0)
        {
            std::cout << "  Optimization: iter 0 loss=" << loss << "\n";
        }

        {
            flexi::scoped_timer timer("optimize weights", std::cout, log_perf, perf_prefix);
            for (int iter = 0; iter < opt_iterations; ++iter)
            {
                beta -= opt_lr * grads.d_beta;
                alpha -= opt_lr * grads.d_alpha;
                gamma -= opt_lr * grads.d_gamma;

                if (opt_clamp > 0.0)
                {
                    beta  = beta.cwiseMax(-opt_clamp).cwiseMin(opt_clamp);
                    alpha = alpha.cwiseMax(-opt_clamp).cwiseMin(opt_clamp);
                    gamma = gamma.cwiseMax(-opt_clamp).cwiseMin(opt_clamp);
                }

                auto [mesh, loss_value] = fc.extract_surface_with_grads(
                    grid.vertices, sdf, grid.cubes, Resolution(resolution),
                    beta, alpha, gamma, &grads, opt_l_dev_weight, fc_opts);

                output = mesh;
                loss = loss_value;

                if (opt_report_every > 0 && ((iter + 1) % opt_report_every == 0 || iter + 1 == opt_iterations))
                {
                    std::cout << "  Optimization: iter " << (iter + 1) << " loss=" << loss << "\n";
                }

                double loss_change = std::abs(prev_loss - loss);
                if (loss_change < opt_convergence && iter > 5)
                {
                    if (opt_report_every > 0)
                    {
                        std::cout << "  Optimization converged (loss change " << loss_change << ")\n";
                    }
                    break;
                }

                prev_loss = loss;
            }
        }

        if (!output.empty())
        {
            std::string post_opt_name = fs::path(output_path).stem().string() + "_postopt.obj";
            std::string post_opt_path = (fs::path(output_path).parent_path() / post_opt_name).string();
            try
            {
                flexi::scoped_timer timer("save post-opt mesh", std::cout, log_perf, perf_prefix);
                save_obj(post_opt_path, output);
                std::cout << "  Saved: " << post_opt_path << "\n";
            }
            catch (const std::exception & e)
            {
                std::cerr << "  Error saving post-opt mesh: " << e.what() << "\n";
                return false;
            }
        }
    }
    else
    {
        {
            flexi::scoped_timer timer("extract surface", std::cout, log_perf, perf_prefix);
            output = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(resolution),
                                        nullptr, nullptr, nullptr, nullptr, fc_opts);
        }
    }

    if (output.empty())
    {
        std::cerr << "  Warning: No surface extracted (mesh may be too small for resolution)\n";
        return false;
    }

    if (verbose)
    {
        std::cout << "  Output: " << output.num_vertices() << " vertices, " << output.num_faces() << " faces\n";
    }

    // Step 7: Save output
    try
    {
        flexi::scoped_timer timer("save output", std::cout, log_perf, perf_prefix);
        save_obj(output_path, output);
    }
    catch (const std::exception & e)
    {
        std::cerr << "  Error saving: " << e.what() << "\n";
        return false;
    }

    std::cout << "  Saved: " << output_path << "\n";

    // Optional: Export tetrahedra as a triangle mesh for visualization
    if (export_tet_mesh)
    {
        TetraMesh tet_mesh;
        {
            flexi::scoped_timer timer("extract volume", std::cout, log_perf, perf_prefix);
            tet_mesh = fc.extract_volume(grid.vertices, sdf, grid.cubes, Resolution(resolution), nullptr, nullptr, nullptr, nullptr, fc_opts);
        }

        if (!tet_mesh.empty() && tet_mesh.num_tets() > 0)
        {
            Mesh tet_surface;
            tet_surface.vertices = tet_mesh.vertices;

            if (tet_boundary_only)
            {
                struct FaceKey
                {
                    std::array<int, 3> v;

                    bool operator==(const FaceKey & other) const { return v == other.v; }
                };

                struct FaceKeyHash
                {
                    size_t operator()(const FaceKey & key) const
                    {
                        size_t h0 = std::hash<int>{}(key.v[0]);
                        size_t h1 = std::hash<int>{}(key.v[1]);
                        size_t h2 = std::hash<int>{}(key.v[2]);
                        return h0 ^ (h1 << 1) ^ (h2 << 2);
                    }
                };

                std::unordered_map<FaceKey, Vec3i, FaceKeyHash> face_map;
                std::unordered_map<FaceKey, int, FaceKeyHash> face_counts;

                auto add_face = [&](int a, int b, int c)
                {
                    std::array<int, 3> key = {a, b, c};
                    std::sort(key.begin(), key.end());
                    FaceKey fkey{key};
                    face_counts[fkey] += 1;
                    if (face_map.find(fkey) == face_map.end())
                    {
                        face_map.emplace(fkey, Vec3i(a, b, c));
                    }
                };

                for (Index i = 0; i < tet_mesh.tets.rows(); ++i)
                {
                    int a = tet_mesh.tets(i, 0);
                    int b = tet_mesh.tets(i, 1);
                    int c = tet_mesh.tets(i, 2);
                    int d = tet_mesh.tets(i, 3);

                    add_face(a, b, c);
                    add_face(a, b, d);
                    add_face(a, c, d);
                    add_face(b, c, d);
                }

                std::vector<Vec3i> boundary_faces;
                boundary_faces.reserve(face_map.size());
                for (const auto & entry : face_counts)
                {
                    if (entry.second == 1)
                    {
                        boundary_faces.push_back(face_map[entry.first]);
                    }
                }

                tet_surface.faces.resize(static_cast<Index>(boundary_faces.size()), 3);
                for (Index i = 0; i < static_cast<Index>(boundary_faces.size()); ++i)
                {
                    tet_surface.faces.row(i) = boundary_faces[i];
                }
            }
            else
            {
                tet_surface.faces.resize(tet_mesh.tets.rows() * 4, 3);

                Index f = 0;
                for (Index i = 0; i < tet_mesh.tets.rows(); ++i)
                {
                    int a = tet_mesh.tets(i, 0);
                    int b = tet_mesh.tets(i, 1);
                    int c = tet_mesh.tets(i, 2);
                    int d = tet_mesh.tets(i, 3);

                    tet_surface.faces.row(f++) = Vec3i(a, b, c);
                    tet_surface.faces.row(f++) = Vec3i(a, b, d);
                    tet_surface.faces.row(f++) = Vec3i(a, c, d);
                    tet_surface.faces.row(f++) = Vec3i(b, c, d);
                }
            }

            std::string suffix = tet_boundary_only ? "_tets_boundary.obj" : "_tets.obj";
            std::string tet_filename = fs::path(output_path).stem().string() + suffix;
            std::string tet_path     = (fs::path(output_path).parent_path() / tet_filename).string();

            try
            {
                flexi::scoped_timer timer("save tet mesh", std::cout, log_perf, perf_prefix);
                save_obj(tet_path, tet_surface);
            }
            catch (const std::exception & e)
            {
                std::cerr << "  Error saving tet mesh: " << e.what() << "\n";
                return false;
            }

            std::cout << "  Saved: " << tet_path << "\n";
        }
        else
        {
            std::cerr << "  Warning: No tetrahedra extracted\n";
        }
    }

    return true;
}

int main(int argc, char * argv[])
{
    std::cout << "===========================================\n";
    std::cout << " FlexiCubes Mesh-to-SDF Test Runner\n";
    std::cout << "===========================================\n\n";

    CmdOptions opts = parse_args(argc, argv);

    // Create output directory
    if (!fs::exists(opts.output_dir))
    {
        fs::create_directories(opts.output_dir);
        std::cout << "Created output directory: " << opts.output_dir << "\n";
    }

    std::cout << "Resolution: " << opts.resolution << "^3\n\n";

    // Collect files to process
    std::vector<std::string> files;
    if (opts.process_all)
    {
        files = find_obj_files(opts.assets_dir);
        if (files.empty())
        {
            std::cerr << "No OBJ files found in: " << opts.assets_dir << "\n";
            return 1;
        }
        std::cout << "Found " << files.size() << " OBJ files in " << opts.assets_dir << "\n\n";
    }
    else
    {
        files.push_back(opts.input_file);
    }

    // Process each file
    int success_count = 0;
    int fail_count    = 0;

    for (const auto & input_path : files)
    {
        // Generate output filename
        std::string stem            = fs::path(input_path).stem().string();
        std::string output_filename = stem + "_fc_r" + std::to_string(opts.resolution) + ".obj";
        std::string output_path     = (fs::path(opts.output_dir) / output_filename).string();

        bool ok = process_mesh(input_path, output_path, opts.resolution, opts.verbose,
                               opts.export_tet_mesh, opts.tet_boundary_only, opts.optimize,
                               opts.opt_iterations, opts.opt_lr, opts.opt_convergence,
                               opts.opt_clamp, opts.opt_l_dev_weight, opts.opt_report_every);
        if (ok)
        {
            ++success_count;
        }
        else
        {
            ++fail_count;
        }
        std::cout << "\n";
    }

    // Summary
    std::cout << "===========================================\n";
    std::cout << " Summary\n";
    std::cout << "===========================================\n";
    std::cout << "Processed: " << success_count << " succeeded, " << fail_count << " failed\n";
    std::cout << "Output directory: " << opts.output_dir << "\n";

    return fail_count > 0 ? 1 : 0;
}
