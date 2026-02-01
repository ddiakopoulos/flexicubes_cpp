#pragma once

#include "types.hpp"
#include "tables.hpp"
#include "surface.hpp"
#include "case_id.hpp"
#include "qef_solver.hpp"
#include <functional>

namespace flexi
{

    struct DualVertexResult
    {
        MatX3 vertices;      // Dual vertex positions (num_vd x 3)
        VecXd L_dev;         // Regularization loss per dual vertex
        VecXd gamma;         // Gamma weight per dual vertex
        MatX12i vd_idx_map;  // Maps cube-edge to dual vertex index
        Index num_dual_vertices;
    };

    /**
    * Compute zero-crossing position on an edge using linear interpolation.
    *
    * Given two endpoints with scalar values, finds the point where the
    * linear interpolation crosses zero (the isosurface).
    *
    * @param p0 First endpoint position
    * @param p1 Second endpoint position
    * @param s0 Scalar value at first endpoint
    * @param s1 Scalar value at second endpoint
    * @return Zero-crossing position
    */
    inline Vec3 linear_interp(const Vec3 & p0, const Vec3 & p1, double s0, double s1)
    {
        // Interpolation: find t where s0 + t*(s1-s0) = 0
        // t = -s0 / (s1 - s0) = s0 / (s0 - s1)
        // But we use the formulation: result = (s1*p0 - s0*p1) / (s1 - s0)
        double denom = s1 - s0;
        if (std::abs(denom) < 1e-10)
        {
            return 0.5 * (p0 + p1);  // Fallback to midpoint
        }
        return (s1 * p0 - s0 * p1) / denom;
    }

    /**
    * Compute weighted zero-crossing with alpha parameters.
    *
    * The alpha weights modify the interpolation to allow the surface
    * to be adjusted during optimization.
    *
    * @param p0 First endpoint position
    * @param p1 Second endpoint position
    * @param s0 Scalar value at first endpoint
    * @param s1 Scalar value at second endpoint
    * @param alpha0 Alpha weight for first endpoint
    * @param alpha1 Alpha weight for second endpoint
    * @return Weighted zero-crossing position
    */
    inline Vec3 linear_interp_weighted(
        const Vec3 & p0, const Vec3 & p1,
        double s0, double s1,
        double alpha0, double alpha1)
    {
        // Weight the scalar values by alpha
        double ws0 = s0 * alpha0;
        double ws1 = s1 * alpha1;
        return linear_interp(p0, p1, ws0, ws1);
    }

    /**
    * Compute all zero-crossing positions for surface edges.
    *
    * @param vertices Grid vertex positions
    * @param sdf Scalar field values
    * @param surf_edges Surface edge vertex pairs
    * @return Zero-crossing positions for each surface edge
    */
    inline MatX3 compute_zero_crossings(
        const MatX3 & vertices,
        const VecXd & sdf,
        const Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> & surf_edges)
    {
        const Index num_edges = surf_edges.rows();
        MatX3 crossings(num_edges, 3);

        for (Index i = 0; i < num_edges; ++i)
        {
            int v0           = surf_edges(i, 0);
            int v1           = surf_edges(i, 1);
            crossings.row(i) = linear_interp(
                                   vertices.row(v0).transpose(),
                                   vertices.row(v1).transpose(),
                                   sdf[v0], sdf[v1])
                                   .transpose();
        }

        return crossings;
    }

    /**
    * Compute dual vertices using the differentiable FlexiCubes method.
    *
    * This is the core computation that produces dual vertex positions from
    * edge zero-crossings, weighted by beta and alpha parameters.
    *
    * For each dual vertex:
    * 1. Get contributing edges from DMC table
    * 2. Compute weighted zero-crossings (using alpha)
    * 3. Average weighted by beta
    *
    * @param vertices Grid vertex positions
    * @param surf_cubes_fx8 Surface cube vertex indices
    * @param surf_edges Surface edge information
    * @param sdf Scalar field values
    * @param case_ids Case IDs for surface cubes
    * @param beta Edge weights (num_surf_cubes x 12)
    * @param alpha Corner weights (num_surf_cubes x 8)
    * @param gamma Quad splitting weights (num_surf_cubes)
    * @param idx_map Edge index mapping
    * @param grad_func Optional gradient function for QEF mode
    * @return DualVertexResult with positions, L_dev, gamma, and index map
    */
    inline DualVertexResult compute_dual_vertices(
        const MatX3 & vertices,
        const MatX8i & surf_cubes_fx8,
        const SurfaceEdges & surf_edges,
        const VecXd & sdf,
        const VecXi & case_ids,
        const Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> & beta,
        const Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> & alpha,
        const VecXd & gamma,
        std::function<MatX3(const MatX3 &)> grad_func = nullptr,
        double qef_reg_scale                          = 1e-3)
    {
        const Index num_surf_cubes = surf_cubes_fx8.rows();

        // Pre-compute all zero-crossings
        MatX3 zero_crossings = compute_zero_crossings(vertices, sdf, surf_edges.edges);

        // Compute alpha for each edge (average of corner alphas)
        // alpha_nx12x2[cube][edge][0/1] = alpha at edge endpoints
        std::vector<Eigen::Matrix<double, 12, 2>> alpha_edge(num_surf_cubes);
        for (Index c = 0; c < num_surf_cubes; ++c)
        {
            for (int e = 0; e < 12; ++e)
            {
                int c0              = detail::CUBE_EDGES_FLAT[e * 2];
                int c1              = detail::CUBE_EDGES_FLAT[e * 2 + 1];
                alpha_edge[c](e, 0) = alpha(c, c0);
                alpha_edge[c](e, 1) = alpha(c, c1);
            }
        }

        // Count total dual vertices
        Index total_num_vd = 0;
        for (Index c = 0; c < num_surf_cubes; ++c)
        {
            total_num_vd += tables::get_num_dual_vertices(case_ids[c]);
        }

        // Prepare result
        DualVertexResult result;
        result.vertices.resize(total_num_vd, 3);
        result.vertices.setZero();
        result.L_dev.resize(total_num_vd);
        result.L_dev.setZero();
        result.gamma.resize(total_num_vd);
        result.vd_idx_map.resize(num_surf_cubes, 12);
        result.vd_idx_map.setConstant(-1);
        result.num_dual_vertices = total_num_vd;

        if (total_num_vd == 0)
        {
            return result;
        }

        if (grad_func)
        {
            // QEF mode: use surface gradients to solve for dual vertices (non-differentiable)
            MatX3 normals = grad_func(zero_crossings);
            for (Index i = 0; i < normals.rows(); ++i)
            {
                double n = normals.row(i).norm();
                if (n > 1e-10)
                {
                    normals.row(i) /= n;
                }
                else
                {
                    normals.row(i).setZero();
                }
            }

            Index vd_offset = 0;
            for (Index c = 0; c < num_surf_cubes; ++c)
            {
                int case_id = case_ids[c];
                int num_vd  = tables::get_num_dual_vertices(case_id);

                Vec3 v0 = vertices.row(surf_cubes_fx8(c, 0)).transpose();

                for (int vd = 0; vd < num_vd; ++vd)
                {
                    Index vd_idx         = vd_offset + vd;
                    result.gamma[vd_idx] = gamma[c];

                    const auto & edge_group = tables::get_dual_vertex_edges(case_id, vd);
                    std::vector<Vec3> pos_list;
                    std::vector<Vec3> norm_list;
                    pos_list.reserve(7);
                    norm_list.reserve(7);

                    for (int ei = 0; ei < 7; ++ei)
                    {
                        int local_edge = edge_group[ei];
                        if (local_edge < 0) continue;

                        int surf_edge_idx = surf_edges.idx_map(c, local_edge);
                        if (surf_edge_idx < 0) continue;

                        pos_list.push_back(zero_crossings.row(surf_edge_idx).transpose());
                        norm_list.push_back(normals.row(surf_edge_idx).transpose());

                        result.vd_idx_map(c, local_edge) = static_cast<int>(vd_idx);
                    }

                    if (!pos_list.empty())
                    {
                        MatX3 positions(pos_list.size(), 3);
                        MatX3 norms(pos_list.size(), 3);
                        Vec3 centroid = Vec3::Zero();
                        for (size_t i = 0; i < pos_list.size(); ++i)
                        {
                            positions.row(i) = (pos_list[i] - v0).transpose();
                            norms.row(i)     = norm_list[i].transpose();
                            centroid += pos_list[i];
                        }
                        centroid /= static_cast<double>(pos_list.size());

                        Vec3 dv                     = solve_qef(positions, norms, centroid - v0, qef_reg_scale) + v0;
                        result.vertices.row(vd_idx) = dv.transpose();
                    }
                }

                vd_offset += num_vd;
            }

            // L_dev is not defined for QEF mode in the reference
            result.L_dev.setZero();
        }
        else
        {
            // Accumulators for beta-weighted averaging
            VecXd beta_sum(total_num_vd);
            beta_sum.setZero();

            // Process each cube
            Index vd_offset = 0;
            for (Index c = 0; c < num_surf_cubes; ++c)
            {
                int case_id = case_ids[c];
                int num_vd  = tables::get_num_dual_vertices(case_id);

                for (int vd = 0; vd < num_vd; ++vd)
                {
                    Index vd_idx         = vd_offset + vd;
                    result.gamma[vd_idx] = gamma[c];

                    // Get edges contributing to this dual vertex
                    const auto & edge_group = tables::get_dual_vertex_edges(case_id, vd);

                    // Process each contributing edge
                    for (int ei = 0; ei < 7; ++ei)
                    {
                        int local_edge = edge_group[ei];
                        if (local_edge < 0) continue;

                        // Get unique surface edge index
                        int surf_edge_idx = surf_edges.idx_map(c, local_edge);
                        if (surf_edge_idx < 0) continue;

                        // Get edge vertex indices
                        int c0 = detail::CUBE_EDGES_FLAT[local_edge * 2];
                        int c1 = detail::CUBE_EDGES_FLAT[local_edge * 2 + 1];
                        int v0 = surf_cubes_fx8(c, c0);
                        int v1 = surf_cubes_fx8(c, c1);

                        // Compute alpha-weighted zero crossing
                        Vec3 ue = linear_interp_weighted(
                            vertices.row(v0).transpose(),
                            vertices.row(v1).transpose(),
                            sdf[v0], sdf[v1],
                            alpha_edge[c](local_edge, 0),
                            alpha_edge[c](local_edge, 1));

                        // Add beta-weighted contribution
                        double b = beta(c, local_edge);
                        result.vertices.row(vd_idx) += b * ue.transpose();
                        beta_sum[vd_idx] += b;

                        // Map this edge to this dual vertex
                        result.vd_idx_map(c, local_edge) = static_cast<int>(vd_idx);
                    }
                }

                vd_offset += num_vd;
            }

            // Normalize by beta sum
            for (Index i = 0; i < total_num_vd; ++i)
            {
                if (beta_sum[i] > 1e-10)
                {
                    result.vertices.row(i) /= beta_sum[i];
                }
            }

            // Compute L_dev regularization loss
            // L_dev measures deviation of zero-crossings from dual vertex
            vd_offset = 0;
            for (Index c = 0; c < num_surf_cubes; ++c)
            {
                int case_id = case_ids[c];
                int num_vd  = tables::get_num_dual_vertices(case_id);

                for (int vd = 0; vd < num_vd; ++vd)
                {
                    Index vd_idx            = vd_offset + vd;
                    const auto & edge_group = tables::get_dual_vertex_edges(case_id, vd);

                    double sum_dist = 0.0;
                    int count       = 0;

                    for (int ei = 0; ei < 7; ++ei)
                    {
                        int local_edge = edge_group[ei];
                        if (local_edge < 0) continue;

                        int surf_edge_idx = surf_edges.idx_map(c, local_edge);
                        if (surf_edge_idx < 0) continue;

                        Vec3 zc = zero_crossings.row(surf_edge_idx).transpose();
                        Vec3 dv = result.vertices.row(vd_idx).transpose();
                        sum_dist += (zc - dv).norm();
                        count++;
                    }

                    if (count > 0)
                    {
                        double mean_dist = sum_dist / count;

                        // Compute MAD (mean absolute deviation)
                        double mad = 0.0;
                        for (int ei = 0; ei < 7; ++ei)
                        {
                            int local_edge = edge_group[ei];
                            if (local_edge < 0) continue;

                            int surf_edge_idx = surf_edges.idx_map(c, local_edge);
                            if (surf_edge_idx < 0) continue;

                            Vec3 zc     = zero_crossings.row(surf_edge_idx).transpose();
                            Vec3 dv     = result.vertices.row(vd_idx).transpose();
                            double dist = (zc - dv).norm();
                            mad += std::abs(dist - mean_dist);
                        }
                        result.L_dev[vd_idx] = mad / count;
                    }
                }

                vd_offset += num_vd;
            }
        }

        return result;
    }

    /**
    * Normalize weight parameters as done in the original FlexiCubes.
    *
    * @param beta Raw beta values (or nullptr for default)
    * @param alpha Raw alpha values (or nullptr for default)
    * @param gamma Raw gamma values (or nullptr for default)
    * @param num_cubes Number of cubes
    * @param weight_scale Scale factor for weights (default 0.99)
    * @return Tuple of normalized (beta, alpha, gamma)
    */
    inline std::tuple<
        Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor>,
        Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor>,
        VecXd>
    normalize_weights(
        const Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> * beta,
        const Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> * alpha,
        const VecXd * gamma,
        Index num_cubes,
        double weight_scale = 0.99)
    {
        Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> norm_beta;
        Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> norm_alpha;
        VecXd norm_gamma;

        // Beta: tanh(x) * weight_scale + 1
        if (beta)
        {
            norm_beta = beta->array().tanh() * weight_scale + 1.0;
        }
        else
        {
            norm_beta.resize(num_cubes, 12);
            norm_beta.setOnes();
        }

        // Alpha: tanh(x) * weight_scale + 1
        if (alpha)
        {
            norm_alpha = alpha->array().tanh() * weight_scale + 1.0;
        }
        else
        {
            norm_alpha.resize(num_cubes, 8);
            norm_alpha.setOnes();
        }

        // Gamma: sigmoid(x) * weight_scale + (1 - weight_scale) / 2
        if (gamma)
        {
            norm_gamma.resize(gamma->size());
            for (Index i = 0; i < gamma->size(); ++i)
            {
                double sig    = 1.0 / (1.0 + std::exp(-(*gamma)[i]));
                norm_gamma[i] = sig * weight_scale + (1.0 - weight_scale) / 2.0;
            }
        }
        else
        {
            norm_gamma.resize(num_cubes);
            norm_gamma.setOnes();
        }

        return {std::move(norm_beta), std::move(norm_alpha), std::move(norm_gamma)};
    }

}  // namespace flexi
