#pragma once

#include "types.hpp"
#include "surface.hpp"
#include "dual_vertex.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>

namespace flexi
{

    struct TriangulationResult
    {
        MatX3 vertices;                                                     // Output vertices (may include quad centers in training mode)
        MatX3i faces;                                                       // Triangle faces (Fx3)
        Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> s_edges;  // SDF at edge endpoints (tetrahedralization)
        VecXi edge_indices;                                                 // Edge indices for tetrahedralization
    };

    /**
    * Triangulate quads formed by dual vertices.
    *
    * Each surface edge shared by 4 cubes forms a quad from the 4 dual vertices.
    * The quad is split into 2 triangles based on the gamma parameter.
    *
    * In training mode, quads are split into 4 triangles by adding the center point,
    * weighted by gamma values from each diagonal.
    *
    * @param sdf Scalar field values
    * @param surf_edges Surface edge information
    * @param vd Dual vertex computation result
    * @param edge_counts Edge sharing counts
    * @param idx_map Edge index mapping
    * @param training Use training mode (4 triangles per quad)
    * @return TriangulationResult with vertices and faces
    */
    inline TriangulationResult triangulate(
        const VecXd & sdf,
        const SurfaceEdges & surf_edges,
        const DualVertexResult & vd,
        bool training                                 = false,
        std::function<MatX3(const MatX3 &)> grad_func = nullptr)
    {
        TriangulationResult result;

        const Index num_surf_cubes = surf_edges.idx_map.rows();
        const Index num_slots      = num_surf_cubes * 12;

        struct GroupEntry
        {
            int edge_idx;
            int vd_idx;
        };

        std::vector<GroupEntry> group_entries;
        group_entries.reserve(num_slots);

        for (Index slot = 0; slot < num_slots; ++slot)
        {
            if (surf_edges.counts[slot] != 4 || !surf_edges.surf_edges_mask[slot])
            {
                continue;
            }
            Index c      = slot / 12;
            int e        = static_cast<int>(slot % 12);
            int edge_idx = surf_edges.idx_map(c, e);
            int vd_idx   = vd.vd_idx_map(c, e);
            if (edge_idx < 0 || vd_idx < 0)
            {
                continue;
            }
            group_entries.push_back({edge_idx, vd_idx});
        }

        if (group_entries.empty())
        {
            result.vertices = vd.vertices;
            result.faces.resize(0, 3);
            result.s_edges.resize(0, 2);
            result.edge_indices.resize(0);
            return result;
        }

        // Stable sort by edge index (matches Python torch.sort with stable=True)
        std::vector<size_t> order(group_entries.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b)
                         { return group_entries[a].edge_idx < group_entries[b].edge_idx; });

        // Build quads from consecutive groups of 4
        std::vector<std::array<int, 4>> quads;
        std::vector<int> quad_edge_indices;
        for (size_t i = 0; i + 3 < order.size(); i += 4)
        {
            int edge_idx = group_entries[order[i]].edge_idx;
            // Ensure all 4 entries belong to the same edge index
            if (group_entries[order[i + 1]].edge_idx != edge_idx ||
                group_entries[order[i + 2]].edge_idx != edge_idx ||
                group_entries[order[i + 3]].edge_idx != edge_idx)
            {
                continue;
            }
            std::array<int, 4> quad = {
                group_entries[order[i]].vd_idx,
                group_entries[order[i + 1]].vd_idx,
                group_entries[order[i + 2]].vd_idx,
                group_entries[order[i + 3]].vd_idx};
            quads.push_back(quad);
            quad_edge_indices.push_back(edge_idx);
        }

        if (quads.empty())
        {
            result.vertices = vd.vertices;
            result.faces.resize(0, 3);
            result.s_edges.resize(0, 2);
            result.edge_indices.resize(0);
            return result;
        }

        // Determine winding order based on SDF signs at edge endpoints
        result.s_edges.resize(quads.size(), 2);
        result.edge_indices.resize(quads.size());

        for (size_t i = 0; i < quads.size(); ++i)
        {
            int edge_idx           = quad_edge_indices[i];
            int v0                 = surf_edges.edges(edge_idx, 0);
            int v1                 = surf_edges.edges(edge_idx, 1);
            result.s_edges(i, 0)   = sdf[v0];
            result.s_edges(i, 1)   = sdf[v1];
            result.edge_indices[i] = edge_idx;

            auto & q = quads[i];

            if (result.s_edges(i, 0) > 0)
            {
                // Flip to match Python ordering when first endpoint is outside
                q = {q[0], q[1], q[3], q[2]};
            }
            else
            {
                q = {q[2], q[3], q[1], q[0]};
            }
        }

        MatX3 grad_normals;
        if (grad_func)
        {
            grad_normals = grad_func(vd.vertices);
            for (Index i = 0; i < grad_normals.rows(); ++i)
            {
                double n = grad_normals.row(i).norm();
                if (n > 1e-10)
                {
                    grad_normals.row(i) /= n;
                }
                else
                {
                    grad_normals.row(i).setZero();
                }
            }
        }

        if (!training)
        {
            // Inference mode: split each quad into 2 triangles
            // Choose diagonal based on gamma values
            result.vertices = vd.vertices;
            result.faces.resize(quads.size() * 2, 3);

            for (size_t i = 0; i < quads.size(); ++i)
            {
                const auto & q = quads[i];

                // Compute gamma product for each diagonal
                double gamma_02 = 0.0;
                double gamma_13 = 0.0;
                if (grad_func)
                {
                    gamma_02 = grad_normals.row(q[0]).dot(grad_normals.row(q[2]));
                    gamma_13 = grad_normals.row(q[1]).dot(grad_normals.row(q[3]));
                }
                else
                {
                    gamma_02 = vd.gamma[q[0]] * vd.gamma[q[2]];
                    gamma_13 = vd.gamma[q[1]] * vd.gamma[q[3]];
                }

                if (gamma_02 > gamma_13)
                {
                    // Split along 0-2 diagonal: triangles (0,1,2) and (0,2,3)
                    result.faces(i * 2, 0)     = q[0];
                    result.faces(i * 2, 1)     = q[1];
                    result.faces(i * 2, 2)     = q[2];
                    result.faces(i * 2 + 1, 0) = q[0];
                    result.faces(i * 2 + 1, 1) = q[2];
                    result.faces(i * 2 + 1, 2) = q[3];
                }
                else
                {
                    // Split along 1-3 diagonal: triangles (0,1,3) and (3,1,2)
                    result.faces(i * 2, 0)     = q[0];
                    result.faces(i * 2, 1)     = q[1];
                    result.faces(i * 2, 2)     = q[3];
                    result.faces(i * 2 + 1, 0) = q[3];
                    result.faces(i * 2 + 1, 1) = q[1];
                    result.faces(i * 2 + 1, 2) = q[2];
                }
            }
        }
        else
        {
            // Training mode: split each quad into 4 triangles with center vertex
            Index num_orig_verts = vd.vertices.rows();
            result.vertices.resize(num_orig_verts + quads.size(), 3);
            result.vertices.topRows(num_orig_verts) = vd.vertices;
            result.faces.resize(quads.size() * 4, 3);

            for (size_t i = 0; i < quads.size(); ++i)
            {
                const auto & q = quads[i];

                // Compute weighted center based on gamma
                double gamma_02 = 0.0;
                double gamma_13 = 0.0;
                if (grad_func)
                {
                    gamma_02 = grad_normals.row(q[0]).dot(grad_normals.row(q[2]));
                    gamma_13 = grad_normals.row(q[1]).dot(grad_normals.row(q[3]));
                }
                else
                {
                    gamma_02 = vd.gamma[q[0]] * vd.gamma[q[2]];
                    gamma_13 = vd.gamma[q[1]] * vd.gamma[q[3]];
                }
                double weight_sum = gamma_02 + gamma_13 + 1e-8;

                Vec3 center_02 = (vd.vertices.row(q[0]) + vd.vertices.row(q[2])).transpose() / 2.0;
                Vec3 center_13 = (vd.vertices.row(q[1]) + vd.vertices.row(q[3])).transpose() / 2.0;
                Vec3 center    = (gamma_02 * center_02 + gamma_13 * center_13) / weight_sum;

                int center_idx                  = static_cast<int>(num_orig_verts + i);
                result.vertices.row(center_idx) = center.transpose();

                // Create 4 triangles: center to each edge
                // (0,1,center), (1,2,center), (2,3,center), (3,0,center)
                result.faces(i * 4, 0) = q[0];
                result.faces(i * 4, 1) = q[1];
                result.faces(i * 4, 2) = center_idx;

                result.faces(i * 4 + 1, 0) = q[1];
                result.faces(i * 4 + 1, 1) = q[2];
                result.faces(i * 4 + 1, 2) = center_idx;

                result.faces(i * 4 + 2, 0) = q[2];
                result.faces(i * 4 + 2, 1) = q[3];
                result.faces(i * 4 + 2, 2) = center_idx;

                result.faces(i * 4 + 3, 0) = q[3];
                result.faces(i * 4 + 3, 1) = q[0];
                result.faces(i * 4 + 3, 2) = center_idx;
            }
        }

        return result;
    }

    /**
    * Get number of triangles that will be produced.
    *
    * @param num_quads Number of quads
    * @param training Whether training mode is enabled
    * @return Number of output triangles
    */
    inline Index get_triangle_count(Index num_quads, bool training)
    {
        return training ? num_quads * 4 : num_quads * 2;
    }

}  // namespace flexi
