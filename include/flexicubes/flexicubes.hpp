#pragma once

/**
 * FlexiCubes is a differentiable variant of the Dual Marching Cubes (DMC) scheme
 * that enhances the geometric fidelity and mesh quality of reconstructed meshes
 * by dynamically adjusting the surface representation through gradient-based
 * optimization.
 *
 * Basic usage:
 * @code
 *   #include <flexicubes/flexicubes.hpp>
 *
 *   using namespace flexi;
 *
 *   // Create extractor
 *   FlexiCubes fc;
 *
 *   // Generate voxel grid
 *   auto grid = fc.construct_voxel_grid(32);
 *
 *   // Compute SDF at grid vertices (example: sphere)
 *   VecXd sdf(grid.vertices.rows());
 *   for (Index i = 0; i < grid.vertices.rows(); ++i) {
 *       sdf[i] = grid.vertices.row(i).norm() - 0.4;  // Sphere of radius 0.4
 *   }
 *
 *   // Extract mesh
 *   Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(32));
 *
 *   // Use mesh.vertices and mesh.faces
 * @endcode
 *
 * For optimization (training mode):
 * @code
 *   Options opts = Options::training_defaults();
 *   Mesh mesh = fc.extract_surface(verts, sdf, cubes, res,
 *                                  &beta, &alpha, &gamma, opts);
 *   // mesh.l_dev contains regularization values for loss computation
 * @endcode
 */

#include <functional>

#include "types.hpp"
#include "grid.hpp"
#include "surface.hpp"
#include "case_id.hpp"
#include "dual_vertex.hpp"
#include "triangulate.hpp"
#include "tetrahedralize.hpp"
#include "regularizer.hpp"
#include "tables.hpp"
#include "qef_solver.hpp"
#include "differentiable.hpp"

namespace flexi
{
    struct Mesh
    {
        MatX3 vertices;  // Nx3 vertex positions
        MatX3i faces;    // Mx3 triangle indices
        VecXd l_dev;     // Per-dual-vertex L_dev regularization values

        Index num_vertices() const { return vertices.rows(); }
        Index num_faces() const { return faces.rows(); }
        bool empty() const { return vertices.rows() == 0; }
    };

    struct TetraMesh
    {
        MatX3 vertices;  // Nx3 vertex positions
        MatX4i tets;     // Tx4 tetrahedron indices
        VecXd l_dev;     // Per-dual-vertex L_dev regularization values

        Index num_vertices() const { return vertices.rows(); }
        Index num_tets() const { return tets.rows(); }
        bool empty() const { return vertices.rows() == 0; }
    };

    struct Gradients
    {
        MatX3 d_vertices; // Gradient w.r.t. input vertex positions
        VecXd d_sdf; // Gradient w.r.t. input SDF values
        Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> d_beta; // Gradient w.r.t. beta
        Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> d_alpha; // Gradient w.r.t. alpha
        VecXd d_gamma; // Gradient w.r.t. gamma
    };

    struct Options
    {
        double weight_scale  = 0.99; // Scale for weight normalization
        double qef_reg_scale = 1e-3; // QEF regularization scale
        bool training = false; // Training mode (4 triangles per quad)
        bool output_tetmesh  = false; // Output tetrahedral mesh instead of triangles

        static Options defaults()
        {
            return Options {};
        }

        static Options training_defaults()
        {
            Options opts;
            opts.training = true;
            return opts;
        }
    };

    /**
    * Basic usage:
    * @code
    *   FlexiCubes fc;
    *   auto [verts, cubes] = fc.construct_voxel_grid(32);
    *   VecXd sdf = compute_sdf(verts);  // Your SDF function
    *   Mesh mesh = fc.extract_surface(verts, sdf, cubes, Resolution(32));
    * @endcode
    */
    class FlexiCubes
    {
    public:

        /**
        * @param weight_scale Scale for weight normalization (0 to 1)
        * @param qef_reg_scale QEF regularization scale
        */
        explicit FlexiCubes(double weight_scale = 0.99, double qef_reg_scale = 1e-3) : weight_scale_(weight_scale), qef_reg_scale_(qef_reg_scale) {}

        /**
        * @param res Grid resolution (uniform or per-axis)
        * @return VoxelGrid with vertices and cube indices
        */
        VoxelGrid construct_voxel_grid(const Resolution & res) const
        {
            return flexi::construct_voxel_grid(res);
        }

        VoxelGrid construct_voxel_grid(int res) const
        {
            return flexi::construct_voxel_grid(Resolution(res));
        }

        /**
        * Extract a triangle mesh from a scalar field.
        *
        * @param vertices Grid vertex positions (Nx3)
        * @param sdf Scalar field values at vertices (N), negative = inside
        * @param cubes Cube corner indices (Mx8)
        * @param res Grid resolution
        * @param beta Edge weights (Mx12), or nullptr for default
        * @param alpha Corner weights (Mx8), or nullptr for default
        * @param gamma Quad split weights (M), or nullptr for default
        * @param opts Extraction options
        * @return Extracted triangle mesh
        */
        Mesh extract_surface(
            const MatX3 & vertices,
            const VecXd & sdf,
            const MatX8i & cubes,
            const Resolution & res,
            const Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> * beta = nullptr,
            const Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> * alpha = nullptr,
            const VecXd * gamma                                                     = nullptr,
            std::function<MatX3(const MatX3 &)> grad_func                           = nullptr,
            const Options & opts                                                    = Options::defaults()) const
        {
            // Identify surface cubes
            auto surf_cubes = identify_surface_cubes(sdf, cubes);
            if (surf_cubes.num_surface_cubes() == 0)
            {
                return Mesh {};
            }

            // Normalize weights
            auto [norm_beta, norm_alpha, norm_gamma] = normalize_weights(
                beta, alpha, gamma, cubes.rows(), weight_scale_);

            // Filter weights to surface cubes only
            Index num_surf = surf_cubes.num_surface_cubes();
            Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> surf_beta(num_surf, 12);
            Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> surf_alpha(num_surf, 8);
            VecXd surf_gamma(num_surf);

            Index j = 0;
            for (Index i = 0; i < surf_cubes.mask.size(); ++i)
            {
                if (surf_cubes.mask[i])
                {
                    surf_beta.row(j)  = norm_beta.row(i);
                    surf_alpha.row(j) = norm_alpha.row(i);
                    surf_gamma[j]     = norm_gamma[i];
                    ++j;
                }
            }

            // Compute case IDs
            auto case_ids = compute_case_ids_with_resolution(surf_cubes, res);

            // Identify surface edges
            auto surf_edges = identify_surface_edges(sdf, cubes, surf_cubes);

            // Get surface cube vertex indices
            auto surf_cubes_fx8 = get_surface_cubes_fx8(cubes, surf_cubes);

            // Compute dual vertices
            auto vd = compute_dual_vertices(
                vertices, surf_cubes_fx8, surf_edges, sdf, case_ids,
                surf_beta, surf_alpha, surf_gamma, grad_func, opts.qef_reg_scale);

            // Triangulate
            auto tri_result = triangulate(sdf, surf_edges, vd, opts.training, grad_func);

            // Build output mesh
            Mesh mesh;
            mesh.vertices = tri_result.vertices;
            mesh.faces    = tri_result.faces;
            mesh.l_dev    = vd.L_dev;

            return mesh;
        }

        /**
        * Extract a tetrahedral mesh from a scalar field.
        *
        * @param vertices Grid vertex positions (Nx3)
        * @param sdf Scalar field values at vertices (N), negative = inside
        * @param cubes Cube corner indices (Mx8)
        * @param res Grid resolution
        * @param beta Edge weights (Mx12), or nullptr for default
        * @param alpha Corner weights (Mx8), or nullptr for default
        * @param gamma Quad split weights (M), or nullptr for default
        * @param opts Extraction options
        * @return Extracted tetrahedral mesh
        */
        TetraMesh extract_volume(
            const MatX3 & vertices,
            const VecXd & sdf,
            const MatX8i & cubes,
            const Resolution & res,
            const Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> * beta = nullptr,
            const Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> * alpha = nullptr,
            const VecXd * gamma                                                     = nullptr,
            std::function<MatX3(const MatX3 &)> grad_func                           = nullptr,
            const Options & opts                                                    = Options::defaults()) const
        {
            // Identify surface cubes
            auto surf_cubes = identify_surface_cubes(sdf, cubes);
            if (surf_cubes.num_surface_cubes() == 0)
            {
                return TetraMesh {};
            }

            // Normalize weights
            auto [norm_beta, norm_alpha, norm_gamma] = normalize_weights(
                beta, alpha, gamma, cubes.rows(), weight_scale_);

            // Filter weights to surface cubes
            Index num_surf = surf_cubes.num_surface_cubes();
            Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> surf_beta(num_surf, 12);
            Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> surf_alpha(num_surf, 8);
            VecXd surf_gamma(num_surf);

            Index j = 0;
            for (Index i = 0; i < surf_cubes.mask.size(); ++i)
            {
                if (surf_cubes.mask[i])
                {
                    surf_beta.row(j)  = norm_beta.row(i);
                    surf_alpha.row(j) = norm_alpha.row(i);
                    surf_gamma[j]     = norm_gamma[i];
                    ++j;
                }
            }

            auto case_ids = compute_case_ids_with_resolution(surf_cubes, res);
            auto surf_edges = identify_surface_edges(sdf, cubes, surf_cubes);
            auto surf_cubes_fx8 = get_surface_cubes_fx8(cubes, surf_cubes);
            auto vd = compute_dual_vertices(vertices, surf_cubes_fx8, surf_edges, sdf, case_ids, surf_beta, surf_alpha, surf_gamma, grad_func, opts.qef_reg_scale);
            auto tri_result = triangulate(sdf, surf_edges, vd, opts.training, grad_func);
            auto tet_result = tetrahedralize(vertices, sdf, cubes, tri_result, surf_edges, vd.vd_idx_map, case_ids, surf_cubes, opts.training);

            TetraMesh mesh;
            mesh.vertices = tet_result.vertices;
            mesh.tets = tet_result.tets;
            mesh.l_dev = vd.L_dev;

            return mesh;
        }

        /**
        * Extract a triangle mesh with gradient computation using Adept-2.
        *
        * This function performs differentiable mesh extraction, computing gradients
        * of the output mesh vertices and L_dev loss with respect to all inputs.
        *
        * @param vertices Grid vertex positions (Nx3)
        * @param sdf Scalar field values at vertices (N), negative = inside
        * @param cubes Cube corner indices (Mx8)
        * @param res Grid resolution
        * @param beta Edge weights (Mx12)
        * @param alpha Corner weights (Mx8)
        * @param gamma Quad split weights (M)
        * @param grads_out Output gradients (must not be null)
        * @param loss_weights Weights for loss components (l_dev_weight, vertex_weight)
        * @param opts Extraction options
        * @return Extracted triangle mesh and total loss value
        */
        std::pair<Mesh, double> extract_surface_with_grads(
            const MatX3 & vertices,
            const VecXd & sdf,
            const MatX8i & cubes,
            const Resolution & res,
            const Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> & beta,
            const Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> & alpha,
            const VecXd & gamma,
            Gradients * grads_out,
            double l_dev_weight  = 1.0,
            const Options & opts = Options::defaults()) const
        {
            using namespace differentiable;

            // Create Adept stack for this computation
            adept::Stack stack;

            // Identify surface cubes (non-differentiable)
            auto surf_cubes = identify_surface_cubes(sdf, cubes);
            if (surf_cubes.num_surface_cubes() == 0)
            {
                if (grads_out)
                {
                    grads_out->d_vertices.setZero(vertices.rows(), 3);
                    grads_out->d_sdf.setZero(sdf.size());
                    grads_out->d_beta.setZero(beta.rows(), 12);
                    grads_out->d_alpha.setZero(alpha.rows(), 8);
                    grads_out->d_gamma.setZero(gamma.size());
                }
                return {Mesh {}, 0.0};
            }

            // Convert inputs to adept::adouble
            const Index num_verts = vertices.rows();
            const Index num_cubes = cubes.rows();
            const Index num_surf  = surf_cubes.num_surface_cubes();

            // Vertex positions
            std::vector<adept::adouble> ad_verts_x(num_verts);
            std::vector<adept::adouble> ad_verts_y(num_verts);
            std::vector<adept::adouble> ad_verts_z(num_verts);
            for (Index i = 0; i < num_verts; ++i)
            {
                ad_verts_x[i] = vertices(i, 0);
                ad_verts_y[i] = vertices(i, 1);
                ad_verts_z[i] = vertices(i, 2);
            }

            // SDF values
            std::vector<adept::adouble> ad_sdf(num_verts);
            for (Index i = 0; i < num_verts; ++i)
            {
                ad_sdf[i] = sdf[i];
            }

            // Normalize and convert weights
            auto [norm_beta, norm_alpha, norm_gamma] = normalize_weights(
                &beta, &alpha, &gamma, num_cubes, weight_scale_);

            // Filter to surface cubes and convert to adept
            std::vector<std::vector<adept::adouble>> ad_beta(num_surf);
            std::vector<std::vector<adept::adouble>> ad_alpha(num_surf);
            std::vector<adept::adouble> ad_gamma(num_surf);

            Index j = 0;
            for (Index i = 0; i < num_cubes; ++i)
            {
                if (surf_cubes.mask[i])
                {
                    ad_beta[j].resize(12);
                    ad_alpha[j].resize(8);
                    for (int e = 0; e < 12; ++e)
                    {
                        ad_beta[j][e] = norm_beta(i, e);
                    }
                    for (int c = 0; c < 8; ++c)
                    {
                        ad_alpha[j][c] = norm_alpha(i, c);
                    }
                    ad_gamma[j] = norm_gamma[i];
                    ++j;
                }
            }

            // Start recording
            stack.new_recording();

            // Compute case IDs (non-differentiable)
            auto case_ids = compute_case_ids_with_resolution(surf_cubes, res);

            // Identify surface edges (non-differentiable)
            auto surf_edges = identify_surface_edges(sdf, cubes, surf_cubes);

            // Get surface cube indices
            auto surf_cubes_fx8 = get_surface_cubes_fx8(cubes, surf_cubes);

            // Compute dual vertices (differentiable)
            auto vd_diff = compute_dual_vertices_diff(
                ad_verts_x, ad_verts_y, ad_verts_z,
                surf_cubes_fx8, surf_edges, ad_sdf, case_ids,
                ad_beta, ad_alpha, ad_gamma);

            // Compute loss
            adept::adouble total_loss = l_dev_weight * compute_total_l_dev_diff(vd_diff);

            // Set adjoint and compute gradients
            total_loss.set_gradient(1.0);
            stack.compute_adjoint();

            // Extract gradients
            if (grads_out)
            {
                grads_out->d_vertices.resize(num_verts, 3);
                grads_out->d_sdf.resize(num_verts);
                grads_out->d_beta.resize(num_cubes, 12);
                grads_out->d_alpha.resize(num_cubes, 8);
                grads_out->d_gamma.resize(num_cubes);

                grads_out->d_vertices.setZero();
                grads_out->d_sdf.setZero();
                grads_out->d_beta.setZero();
                grads_out->d_alpha.setZero();
                grads_out->d_gamma.setZero();

                for (Index i = 0; i < num_verts; ++i)
                {
                    grads_out->d_vertices(i, 0) = ad_verts_x[i].get_gradient();
                    grads_out->d_vertices(i, 1) = ad_verts_y[i].get_gradient();
                    grads_out->d_vertices(i, 2) = ad_verts_z[i].get_gradient();
                    grads_out->d_sdf[i]         = ad_sdf[i].get_gradient();
                }

                j = 0;
                for (Index i = 0; i < num_cubes; ++i)
                {
                    if (surf_cubes.mask[i])
                    {
                        for (int e = 0; e < 12; ++e)
                        {
                            grads_out->d_beta(i, e) = ad_beta[j][e].get_gradient();
                        }
                        for (int c = 0; c < 8; ++c)
                        {
                            grads_out->d_alpha(i, c) = ad_alpha[j][c].get_gradient();
                        }
                        grads_out->d_gamma[i] = ad_gamma[j].get_gradient();
                        ++j;
                    }
                }
            }

            // Build output mesh from non-differentiable values
            DualVertexResult vd;
            vd.vertices.resize(vd_diff.num_dual_vertices, 3);
            vd.L_dev.resize(vd_diff.num_dual_vertices);
            vd.gamma.resize(vd_diff.num_dual_vertices);
            vd.vd_idx_map        = vd_diff.vd_idx_map;
            vd.num_dual_vertices = vd_diff.num_dual_vertices;

            for (Index i = 0; i < vd_diff.num_dual_vertices; ++i)
            {
                vd.vertices(i, 0) = adept::value(vd_diff.vd_x[i]);
                vd.vertices(i, 1) = adept::value(vd_diff.vd_y[i]);
                vd.vertices(i, 2) = adept::value(vd_diff.vd_z[i]);
                vd.L_dev[i]       = adept::value(vd_diff.l_dev[i]);
            }

            // Copy gamma (from surface cubes)
            j               = 0;
            Index vd_offset = 0;
            for (Index i = 0; i < num_cubes; ++i)
            {
                if (surf_cubes.mask[i])
                {
                    int num_vd_cube = tables::get_num_dual_vertices(case_ids[j]);
                    for (int v = 0; v < num_vd_cube; ++v)
                    {
                        vd.gamma[vd_offset + v] = adept::value(ad_gamma[j]);
                    }
                    vd_offset += num_vd_cube;
                    ++j;
                }
            }

            // Triangulate
            auto tri_result = triangulate(sdf, surf_edges, vd, opts.training, nullptr);

            Mesh mesh;
            mesh.vertices = tri_result.vertices;
            mesh.faces    = tri_result.faces;
            mesh.l_dev    = vd.L_dev;

            return {mesh, adept::value(total_loss)};
        }

        /**
        * Extract a triangle mesh and compute gradients of a user-defined loss
        * on the dual vertices using Adept-2.
        *
        * The loss callback receives the dual vertex positions (vd_x, vd_y, vd_z)
        * as adept::adouble arrays and must return a scalar loss.
        *
        * Note: topology decisions (surface cubes/edges, case IDs, quad grouping)
        * remain non-differentiable, consistent with the Python reference.
        *
        * @param vertices Grid vertex positions (Nx3)
        * @param sdf Scalar field values at vertices (N), negative = inside
        * @param cubes Cube corner indices (Mx8)
        * @param res Grid resolution
        * @param beta Edge weights (Mx12)
        * @param alpha Corner weights (Mx8)
        * @param gamma Quad split weights (M)
        * @param loss_cb User-defined loss on dual vertices (required)
        * @param grads_out Output gradients (optional)
        * @param opts Extraction options
        * @return Extracted triangle mesh and loss value
        */
        std::pair<Mesh, double> extract_surface_with_loss(
            const MatX3 & vertices,
            const VecXd & sdf,
            const MatX8i & cubes,
            const Resolution & res,
            const Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> & beta,
            const Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> & alpha,
            const VecXd & gamma,
            const std::function<adept::adouble(
                const std::vector<adept::adouble> &,
                const std::vector<adept::adouble> &,
                const std::vector<adept::adouble> &)> & loss_cb,
            Gradients * grads_out,
            const Options & opts = Options::defaults()) const
        {
            using namespace differentiable;

            // Create Adept stack for this computation
            adept::Stack stack;

            // Identify surface cubes (non-differentiable)
            auto surf_cubes = identify_surface_cubes(sdf, cubes);
            if (surf_cubes.num_surface_cubes() == 0)
            {
                if (grads_out)
                {
                    grads_out->d_vertices.setZero(vertices.rows(), 3);
                    grads_out->d_sdf.setZero(sdf.size());
                    grads_out->d_beta.setZero(beta.rows(), 12);
                    grads_out->d_alpha.setZero(alpha.rows(), 8);
                    grads_out->d_gamma.setZero(gamma.size());
                }
                return {Mesh {}, 0.0};
            }

            const Index num_verts = vertices.rows();
            const Index num_cubes = cubes.rows();
            const Index num_surf  = surf_cubes.num_surface_cubes();

            // Vertex positions
            std::vector<adept::adouble> ad_verts_x(num_verts);
            std::vector<adept::adouble> ad_verts_y(num_verts);
            std::vector<adept::adouble> ad_verts_z(num_verts);
            for (Index i = 0; i < num_verts; ++i)
            {
                ad_verts_x[i] = vertices(i, 0);
                ad_verts_y[i] = vertices(i, 1);
                ad_verts_z[i] = vertices(i, 2);
            }

            // SDF values
            std::vector<adept::adouble> ad_sdf(num_verts);
            for (Index i = 0; i < num_verts; ++i)
            {
                ad_sdf[i] = sdf[i];
            }

            // Normalize and convert weights
            auto [norm_beta, norm_alpha, norm_gamma] = normalize_weights(
                &beta, &alpha, &gamma, num_cubes, weight_scale_);

            // Filter to surface cubes and convert to adept
            std::vector<std::vector<adept::adouble>> ad_beta(num_surf);
            std::vector<std::vector<adept::adouble>> ad_alpha(num_surf);
            std::vector<adept::adouble> ad_gamma(num_surf);

            Index j = 0;
            for (Index i = 0; i < num_cubes; ++i)
            {
                if (surf_cubes.mask[i])
                {
                    ad_beta[j].resize(12);
                    ad_alpha[j].resize(8);
                    for (int e = 0; e < 12; ++e)
                    {
                        ad_beta[j][e] = norm_beta(i, e);
                    }
                    for (int c = 0; c < 8; ++c)
                    {
                        ad_alpha[j][c] = norm_alpha(i, c);
                    }
                    ad_gamma[j] = norm_gamma[i];
                    ++j;
                }
            }

            // Start recording
            stack.new_recording();

            // Compute case IDs (non-differentiable)
            auto case_ids = compute_case_ids_with_resolution(surf_cubes, res);

            // Identify surface edges (non-differentiable)
            auto surf_edges = identify_surface_edges(sdf, cubes, surf_cubes);

            // Get surface cube indices
            auto surf_cubes_fx8 = get_surface_cubes_fx8(cubes, surf_cubes);

            // Compute dual vertices (differentiable)
            auto vd_diff = compute_dual_vertices_diff(
                ad_verts_x, ad_verts_y, ad_verts_z,
                surf_cubes_fx8, surf_edges, ad_sdf, case_ids,
                ad_beta, ad_alpha, ad_gamma);

            // User-defined loss on dual vertices
            adept::adouble total_loss = loss_cb(vd_diff.vd_x, vd_diff.vd_y, vd_diff.vd_z);

            // Set adjoint and compute gradients
            total_loss.set_gradient(1.0);
            stack.compute_adjoint();

            if (grads_out)
            {
                grads_out->d_vertices.resize(num_verts, 3);
                grads_out->d_sdf.resize(num_verts);
                grads_out->d_beta.resize(num_cubes, 12);
                grads_out->d_alpha.resize(num_cubes, 8);
                grads_out->d_gamma.resize(num_cubes);

                grads_out->d_vertices.setZero();
                grads_out->d_sdf.setZero();
                grads_out->d_beta.setZero();
                grads_out->d_alpha.setZero();
                grads_out->d_gamma.setZero();

                for (Index i = 0; i < num_verts; ++i)
                {
                    grads_out->d_vertices(i, 0) = ad_verts_x[i].get_gradient();
                    grads_out->d_vertices(i, 1) = ad_verts_y[i].get_gradient();
                    grads_out->d_vertices(i, 2) = ad_verts_z[i].get_gradient();
                    grads_out->d_sdf[i]         = ad_sdf[i].get_gradient();
                }

                j = 0;
                for (Index i = 0; i < num_cubes; ++i)
                {
                    if (surf_cubes.mask[i])
                    {
                        for (int e = 0; e < 12; ++e)
                        {
                            grads_out->d_beta(i, e) = ad_beta[j][e].get_gradient();
                        }
                        for (int c = 0; c < 8; ++c)
                        {
                            grads_out->d_alpha(i, c) = ad_alpha[j][c].get_gradient();
                        }
                        grads_out->d_gamma[i] = ad_gamma[j].get_gradient();
                        ++j;
                    }
                }
            }

            // Build output mesh from non-differentiable values
            DualVertexResult vd;
            vd.vertices.resize(vd_diff.num_dual_vertices, 3);
            vd.L_dev.resize(vd_diff.num_dual_vertices);
            vd.gamma.resize(vd_diff.num_dual_vertices);
            vd.vd_idx_map        = vd_diff.vd_idx_map;
            vd.num_dual_vertices = vd_diff.num_dual_vertices;

            for (Index i = 0; i < vd_diff.num_dual_vertices; ++i)
            {
                vd.vertices(i, 0) = adept::value(vd_diff.vd_x[i]);
                vd.vertices(i, 1) = adept::value(vd_diff.vd_y[i]);
                vd.vertices(i, 2) = adept::value(vd_diff.vd_z[i]);
                vd.L_dev[i]       = adept::value(vd_diff.l_dev[i]);
            }

            // Copy gamma (from surface cubes)
            j               = 0;
            Index vd_offset = 0;
            for (Index i = 0; i < num_cubes; ++i)
            {
                if (surf_cubes.mask[i])
                {
                    int num_vd_cube = tables::get_num_dual_vertices(case_ids[j]);
                    for (int v = 0; v < num_vd_cube; ++v)
                    {
                        vd.gamma[vd_offset + v] = adept::value(ad_gamma[j]);
                    }
                    vd_offset += num_vd_cube;
                    ++j;
                }
            }

            // Triangulate (non-differentiable)
            auto tri_result = triangulate(sdf, surf_edges, vd, opts.training, nullptr);

            Mesh mesh;
            mesh.vertices = tri_result.vertices;
            mesh.faces    = tri_result.faces;
            mesh.l_dev    = vd.L_dev;

            return {mesh, adept::value(total_loss)};
        }

        // Accessors
        double weight_scale() const { return weight_scale_; }
        double qef_reg_scale() const { return qef_reg_scale_; }

    private:
        double weight_scale_;
        double qef_reg_scale_;
    };

    inline Mesh extract_surface(
        const MatX3 & vertices,
        const VecXd & sdf,
        const MatX8i & cubes,
        const Resolution & res,
        std::function<MatX3(const MatX3 &)> grad_func = nullptr,
        const Options & opts                          = Options::defaults())
    {
        FlexiCubes fc;
        return fc.extract_surface(vertices, sdf, cubes, res, nullptr, nullptr, nullptr, grad_func, opts);
    }

    inline TetraMesh extract_volume(
        const MatX3 & vertices,
        const VecXd & sdf,
        const MatX8i & cubes,
        const Resolution & res,
        std::function<MatX3(const MatX3 &)> grad_func = nullptr,
        const Options & opts                          = Options::defaults())
    {
        FlexiCubes fc;
        return fc.extract_volume(vertices, sdf, cubes, res, nullptr, nullptr, nullptr, grad_func, opts);
    }

}  // namespace flexi
