#pragma once

#include "types.hpp"
#include "tables.hpp"
#include "surface.hpp"
#include "case_id.hpp"

#include <adept.h>
#include <vector>
#include <functional>

namespace flexi
{
    namespace differentiable
    {

        /**
        * Differentiable linear interpolation.
        * Computes zero-crossing position with gradient tracking.
        */
        inline void linear_interp_ad(
            adept::adouble x0, adept::adouble y0, adept::adouble z0,
            adept::adouble x1, adept::adouble y1, adept::adouble z1,
            adept::adouble s0, adept::adouble s1,
            adept::adouble & rx, adept::adouble & ry, adept::adouble & rz)
        {
            adept::adouble denom = s1 - s0;
            // Avoid division by zero
            adept::adouble safe_denom = denom;
            if (adept::value(denom) > -1e-10 && adept::value(denom) < 1e-10)
            {
                // Fallback to midpoint
                rx = 0.5 * (x0 + x1);
                ry = 0.5 * (y0 + y1);
                rz = 0.5 * (z0 + z1);
            }
            else
            {
                rx = (s1 * x0 - s0 * x1) / denom;
                ry = (s1 * y0 - s0 * y1) / denom;
                rz = (s1 * z0 - s0 * z1) / denom;
            }
        }

        /**
        * Differentiable weighted linear interpolation.
        */
        inline void linear_interp_weighted_ad(
            adept::adouble x0, adept::adouble y0, adept::adouble z0,
            adept::adouble x1, adept::adouble y1, adept::adouble z1,
            adept::adouble s0, adept::adouble s1,
            adept::adouble alpha0, adept::adouble alpha1,
            adept::adouble & rx, adept::adouble & ry, adept::adouble & rz)
        {
            adept::adouble ws0 = s0 * alpha0;
            adept::adouble ws1 = s1 * alpha1;
            linear_interp_ad(x0, y0, z0, x1, y1, z1, ws0, ws1, rx, ry, rz);
        }

        /**
        * Result of differentiable dual vertex computation.
        */
        struct DiffDualVertexResult
        {
            std::vector<adept::adouble> vd_x, vd_y, vd_z;  // Dual vertex positions
            std::vector<adept::adouble> l_dev;             // Per-vertex L_dev
            MatX12i vd_idx_map;                            // Mapping
            Index num_dual_vertices;
        };

        /**
        * Compute dual vertices with Adept autodiff.
        *
        * All inputs that need gradients must be adept::adouble.
        */
        inline DiffDualVertexResult compute_dual_vertices_diff(
            const std::vector<adept::adouble> & verts_x,
            const std::vector<adept::adouble> & verts_y,
            const std::vector<adept::adouble> & verts_z,
            const MatX8i & surf_cubes_fx8,
            const SurfaceEdges & surf_edges,
            const std::vector<adept::adouble> & sdf,
            const VecXi & case_ids,
            const std::vector<std::vector<adept::adouble>> & beta,   // [cube][edge]
            const std::vector<std::vector<adept::adouble>> & alpha,  // [cube][corner]
            const std::vector<adept::adouble> & gamma)
        {
            const Index num_surf_cubes = surf_cubes_fx8.rows();

            // Count total dual vertices
            Index total_num_vd = 0;
            for (Index c = 0; c < num_surf_cubes; ++c)
            {
                total_num_vd += tables::get_num_dual_vertices(case_ids[c]);
            }

            DiffDualVertexResult result;
            result.vd_x.resize(total_num_vd);
            result.vd_y.resize(total_num_vd);
            result.vd_z.resize(total_num_vd);
            result.l_dev.resize(total_num_vd);
            result.vd_idx_map.resize(num_surf_cubes, 12);
            result.vd_idx_map.setConstant(-1);
            result.num_dual_vertices = total_num_vd;

            if (total_num_vd == 0)
            {
                return result;
            }

            // Initialize accumulators
            std::vector<adept::adouble> beta_sum(total_num_vd);
            for (Index i = 0; i < total_num_vd; ++i)
            {
                result.vd_x[i] = 0.0;
                result.vd_y[i] = 0.0;
                result.vd_z[i] = 0.0;
                beta_sum[i]    = 0.0;
            }

            // Precompute zero crossings for L_dev computation
            std::vector<adept::adouble> zc_x(surf_edges.num_surface_edges());
            std::vector<adept::adouble> zc_y(surf_edges.num_surface_edges());
            std::vector<adept::adouble> zc_z(surf_edges.num_surface_edges());

            for (Index i = 0; i < surf_edges.num_surface_edges(); ++i)
            {
                int v0 = surf_edges.edges(i, 0);
                int v1 = surf_edges.edges(i, 1);
                linear_interp_ad(
                    verts_x[v0], verts_y[v0], verts_z[v0],
                    verts_x[v1], verts_y[v1], verts_z[v1],
                    sdf[v0], sdf[v1],
                    zc_x[i], zc_y[i], zc_z[i]);
            }

            // Process each cube
            Index vd_offset = 0;
            for (Index c = 0; c < num_surf_cubes; ++c)
            {
                int case_id = case_ids[c];
                int num_vd  = tables::get_num_dual_vertices(case_id);

                for (int vd = 0; vd < num_vd; ++vd)
                {
                    Index vd_idx = vd_offset + vd;

                    // Get edges contributing to this dual vertex
                    const auto & edge_group = tables::get_dual_vertex_edges(case_id, vd);

                    for (int ei = 0; ei < 7; ++ei)
                    {
                        int local_edge = edge_group[ei];
                        if (local_edge < 0) continue;

                        int surf_edge_idx = surf_edges.idx_map(c, local_edge);
                        if (surf_edge_idx < 0) continue;

                        // Get edge vertex indices
                        int c0 = detail::CUBE_EDGES_FLAT[local_edge * 2];
                        int c1 = detail::CUBE_EDGES_FLAT[local_edge * 2 + 1];
                        int v0 = surf_cubes_fx8(c, c0);
                        int v1 = surf_cubes_fx8(c, c1);

                        // Alpha for this edge
                        adept::adouble a0 = alpha[c][c0];
                        adept::adouble a1 = alpha[c][c1];

                        // Compute alpha-weighted zero crossing
                        adept::adouble ue_x, ue_y, ue_z;
                        linear_interp_weighted_ad(
                            verts_x[v0], verts_y[v0], verts_z[v0],
                            verts_x[v1], verts_y[v1], verts_z[v1],
                            sdf[v0], sdf[v1], a0, a1,
                            ue_x, ue_y, ue_z);

                        // Add beta-weighted contribution
                        adept::adouble b = beta[c][local_edge];
                        result.vd_x[vd_idx] += b * ue_x;
                        result.vd_y[vd_idx] += b * ue_y;
                        result.vd_z[vd_idx] += b * ue_z;
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
                adept::adouble safe_beta = beta_sum[i];
                if (adept::value(safe_beta) < 1e-10)
                {
                    safe_beta = 1.0;
                }
                result.vd_x[i] = result.vd_x[i] / safe_beta;
                result.vd_y[i] = result.vd_y[i] / safe_beta;
                result.vd_z[i] = result.vd_z[i] / safe_beta;
            }

            // Compute L_dev
            vd_offset = 0;
            for (Index c = 0; c < num_surf_cubes; ++c)
            {
                int case_id = case_ids[c];
                int num_vd  = tables::get_num_dual_vertices(case_id);

                for (int vd = 0; vd < num_vd; ++vd)
                {
                    Index vd_idx            = vd_offset + vd;
                    const auto & edge_group = tables::get_dual_vertex_edges(case_id, vd);

                    adept::adouble sum_dist = 0.0;
                    int count               = 0;

                    std::vector<adept::adouble> distances;
                    for (int ei = 0; ei < 7; ++ei)
                    {
                        int local_edge = edge_group[ei];
                        if (local_edge < 0) continue;

                        int surf_edge_idx = surf_edges.idx_map(c, local_edge);
                        if (surf_edge_idx < 0) continue;

                        adept::adouble dx   = zc_x[surf_edge_idx] - result.vd_x[vd_idx];
                        adept::adouble dy   = zc_y[surf_edge_idx] - result.vd_y[vd_idx];
                        adept::adouble dz   = zc_z[surf_edge_idx] - result.vd_z[vd_idx];
                        adept::adouble dist = sqrt(dx * dx + dy * dy + dz * dz);
                        distances.push_back(dist);
                        sum_dist += dist;
                        count++;
                    }

                    if (count > 0)
                    {
                        adept::adouble mean_dist = sum_dist / static_cast<double>(count);
                        adept::adouble mad       = 0.0;
                        for (const auto & d : distances)
                        {
                            mad += abs(d - mean_dist);
                        }
                        result.l_dev[vd_idx] = mad / static_cast<double>(count);
                    }
                    else
                    {
                        result.l_dev[vd_idx] = 0.0;
                    }
                }
                vd_offset += num_vd;
            }

            return result;
        }

        /**
        * Compute total L_dev loss (differentiable).
        */
        inline adept::adouble compute_total_l_dev_diff(const DiffDualVertexResult & vd)
        {
            adept::adouble total = 0.0;
            for (const auto & l : vd.l_dev)
            {
                total += l;
            }
            return total;
        }

    }  // namespace differentiable
}  // namespace flexi
