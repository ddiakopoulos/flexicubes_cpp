#pragma once

#include "types.hpp"
#include "tables.hpp"
#include "surface.hpp"
#include "triangulate.hpp"

#include <vector>
#include <unordered_map>
#include <algorithm>

namespace flexi
{

    struct TetrahedralMesh
    {
        MatX3 vertices;  // Vertex positions
        MatX4i tets;     // Tetrahedra (Tx4 vertex indices)
    };

    /**
    * Tetrahedralize the interior volume to produce a tetrahedral mesh.
    *
    * The tetrahedralization process:
    * 1. For each surface triangle, form a tetrahedron by connecting to the
    *    inside grid vertex of the corresponding edge.
    * 2. For fully inside cubes, add center vertex and connect to neighbors.
    * 3. For edges connecting same-sign vertices, form tetrahedra with adjacent
    *    mesh vertices.
    *
    * @param vertices Grid vertex positions
    * @param sdf Scalar field values
    * @param cubes All cube indices
    * @param tri_result Triangulation result
    * @param surf_edges Surface edge information
    * @param vd_idx_map Dual vertex index map
    * @param case_ids Case IDs for surface cubes
    * @param surf_cubes Surface cube identification
    * @param training Training mode flag
    * @return TetrahedralMesh with vertices and tetrahedra
    */
    inline TetrahedralMesh tetrahedralize(
        const MatX3 & vertices,
        const VecXd & sdf,
        const MatX8i & cubes,
        const TriangulationResult & tri_result,
        const SurfaceEdges & surf_edges,
        const MatX12i & vd_idx_map,
        const VecXi & case_ids,
        const SurfaceCubes & surf_cubes,
        bool training = false)
    {
        const Index num_total_cubes = cubes.rows();

        // Occupancy at each vertex
        ArrayXb occ_n = sdf.array() < 0;

        // Find inside vertices and create mapping (with output index offset)
        Index num_mesh_verts = tri_result.vertices.rows();
        std::vector<Index> inside_verts;
        VecXi mapping_inside_verts(vertices.rows());
        mapping_inside_verts.setConstant(-1);
        for (Index i = 0; i < vertices.rows(); ++i)
        {
            if (occ_n[i])
            {
                mapping_inside_verts[i] = static_cast<int>(num_mesh_verts + inside_verts.size());
                inside_verts.push_back(i);
            }
        }

        // Find fully inside cubes and compute their centers
        ArrayXb inside_cubes(num_total_cubes);
        inside_cubes.setConstant(false);
        for (Index i = 0; i < num_total_cubes; ++i)
        {
            int occ_sum = 0;
            for (int c = 0; c < 8; ++c)
            {
                if (occ_n[cubes(i, c)]) occ_sum++;
            }
            inside_cubes[i] = (occ_sum == 8);
        }

        std::vector<Vec3> inside_cube_centers;
        std::vector<int> inside_cube_center_map(num_total_cubes, -1);
        for (Index i = 0; i < num_total_cubes; ++i)
        {
            if (inside_cubes[i])
            {
                Vec3 center = Vec3::Zero();
                for (int c = 0; c < 8; ++c)
                {
                    center += vertices.row(cubes(i, c)).transpose();
                }
                center /= 8.0;
                inside_cube_center_map[i] = static_cast<int>(inside_cube_centers.size());
                inside_cube_centers.push_back(center);
            }
        }

        // Build output vertices: [mesh verts, inside grid verts, inside cube centers]
        Index num_inside_verts   = inside_verts.size();
        Index num_inside_centers = inside_cube_centers.size();
        Index total_verts        = num_mesh_verts + num_inside_verts + num_inside_centers;

        TetrahedralMesh result;
        result.vertices.resize(total_verts, 3);
        result.vertices.topRows(num_mesh_verts) = tri_result.vertices;
        for (size_t i = 0; i < inside_verts.size(); ++i)
        {
            result.vertices.row(num_mesh_verts + i) = vertices.row(inside_verts[i]);
        }
        for (size_t i = 0; i < inside_cube_centers.size(); ++i)
        {
            result.vertices.row(num_mesh_verts + num_inside_verts + i) =
                inside_cube_centers[i].transpose();
        }

        // Collect tetrahedra
        std::vector<Eigen::Vector4i> tets;

        // 1. Surface tetrahedra: connect each face to inside vertex
        const Index faces_per_quad = training ? 4 : 2;
        for (Index f = 0; f < tri_result.faces.rows(); ++f)
        {
            Index quad_idx = f / faces_per_quad;
            if (quad_idx >= tri_result.edge_indices.size()) continue;

            int edge_idx = tri_result.edge_indices[quad_idx];
            int v0       = surf_edges.edges(edge_idx, 0);
            int v1       = surf_edges.edges(edge_idx, 1);

            int inside_vert = -1;
            if (tri_result.s_edges(quad_idx, 0) < 0)
            {
                inside_vert = mapping_inside_verts[v0];
            }
            else if (tri_result.s_edges(quad_idx, 1) < 0)
            {
                inside_vert = mapping_inside_verts[v1];
            }
            if (inside_vert < 0) continue;

            Eigen::Vector4i tet;
            tet[0] = tri_result.faces(f, 0);
            tet[1] = tri_result.faces(f, 1);
            tet[2] = tri_result.faces(f, 2);
            tet[3] = inside_vert;
            tets.push_back(tet);
        }

        // 2. Inside tetrahedra: for edges connecting same-sign (inside) vertices
        ArrayXb surface_n_inside(num_total_cubes);
        for (Index i = 0; i < num_total_cubes; ++i)
        {
            surface_n_inside[i] = surf_cubes.mask[i] || inside_cubes[i];
        }

        std::vector<Index> combined_to_original;
        combined_to_original.reserve(surface_n_inside.count());
        std::vector<char> combined_is_surface;
        std::vector<char> combined_is_inside;
        combined_is_surface.reserve(surface_n_inside.count());
        combined_is_inside.reserve(surface_n_inside.count());

        for (Index i = 0; i < num_total_cubes; ++i)
        {
            if (surface_n_inside[i])
            {
                combined_to_original.push_back(i);
                combined_is_surface.push_back(static_cast<char>(surf_cubes.mask[i]));
                combined_is_inside.push_back(static_cast<char>(inside_cubes[i]));
            }
        }

        Index num_combined_cubes = static_cast<Index>(combined_to_original.size());
        Eigen::Matrix<int, Eigen::Dynamic, 13, Eigen::RowMajor> edge_center_vertex_idx(num_combined_cubes, 13);
        edge_center_vertex_idx.setConstant(-1);

        // Map original cube index -> surface cube index
        std::vector<int> surf_cube_map(num_total_cubes, -1);
        Index surf_idx = 0;
        for (Index i = 0; i < num_total_cubes; ++i)
        {
            if (surf_cubes.mask[i])
            {
                surf_cube_map[i] = static_cast<int>(surf_idx++);
            }
        }

        // Fill edge-center indices and case IDs for combined cubes
        std::vector<int> case_ids_expand(num_combined_cubes, 255);
        for (Index ci = 0; ci < num_combined_cubes; ++ci)
        {
            Index orig = combined_to_original[ci];
            if (combined_is_surface[ci])
            {
                int sc_idx = surf_cube_map[orig];
                if (sc_idx >= 0 && sc_idx < case_ids.size())
                {
                    case_ids_expand[ci] = case_ids[sc_idx];
                    for (int e = 0; e < 12; ++e)
                    {
                        edge_center_vertex_idx(ci, e) = vd_idx_map(sc_idx, e);
                    }
                }
            }
            if (combined_is_inside[ci])
            {
                int center_idx = inside_cube_center_map[orig];
                if (center_idx >= 0)
                {
                    edge_center_vertex_idx(ci, 12) = static_cast<int>(num_mesh_verts + num_inside_verts + center_idx);
                }
            }
        }

        // Build all edges for combined cubes
        struct EdgeKey
        {
            int v0, v1;
            bool operator==(const EdgeKey & other) const { return v0 == other.v0 && v1 == other.v1; }
        };
        struct EdgeKeyHash
        {
            std::size_t operator()(const EdgeKey & e) const
            {
                return std::hash<int>()(e.v0) ^ (std::hash<int>()(e.v1) * 73856093);
            }
        };

        const Index num_edge_slots = num_combined_cubes * 12;
        std::vector<EdgeKey> all_edges;
        all_edges.reserve(num_edge_slots);
        std::vector<int> slot_cube_idx;
        std::vector<int> slot_edge_idx;
        slot_cube_idx.reserve(num_edge_slots);
        slot_edge_idx.reserve(num_edge_slots);

        for (Index ci = 0; ci < num_combined_cubes; ++ci)
        {
            Index orig = combined_to_original[ci];
            for (int e = 0; e < 12; ++e)
            {
                int c0 = detail::CUBE_EDGES_FLAT[e * 2];
                int c1 = detail::CUBE_EDGES_FLAT[e * 2 + 1];
                int v0 = cubes(orig, c0);
                int v1 = cubes(orig, c1);
                all_edges.push_back({v0, v1});
                slot_cube_idx.push_back(static_cast<int>(ci));
                slot_edge_idx.push_back(e);
            }
        }

        std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_to_unique;
        std::vector<EdgeKey> unique_edges;
        std::vector<int> edge_counts;
        std::vector<int> edge_mapping(all_edges.size());

        for (size_t i = 0; i < all_edges.size(); ++i)
        {
            auto it = edge_to_unique.find(all_edges[i]);
            if (it == edge_to_unique.end())
            {
                int idx                      = static_cast<int>(unique_edges.size());
                edge_to_unique[all_edges[i]] = idx;
                unique_edges.push_back(all_edges[i]);
                edge_counts.push_back(1);
                edge_mapping[i] = idx;
            }
            else
            {
                edge_counts[it->second]++;
                edge_mapping[i] = it->second;
            }
        }

        std::vector<bool> mask_edges(unique_edges.size(), false);
        for (size_t i = 0; i < unique_edges.size(); ++i)
        {
            const auto & e = unique_edges[i];
            mask_edges[i]  = occ_n[e.v0] && occ_n[e.v1];
        }

        std::vector<int> counts_per_slot(all_edges.size());
        std::vector<bool> mask_per_slot(all_edges.size());
        for (size_t i = 0; i < all_edges.size(); ++i)
        {
            int u              = edge_mapping[i];
            counts_per_slot[i] = edge_counts[u];
            mask_per_slot[i]   = mask_edges[u];
        }

        std::vector<int> inside_edge_remap(unique_edges.size(), -1);
        std::vector<EdgeKey> inside_unique_edges;
        for (size_t i = 0; i < unique_edges.size(); ++i)
        {
            if (mask_edges[i])
            {
                inside_edge_remap[i] = static_cast<int>(inside_unique_edges.size());
                inside_unique_edges.push_back(unique_edges[i]);
            }
        }

        struct GroupEntry
        {
            int edge_id;
            int cube_idx;
            int edge_idx;
        };
        std::vector<GroupEntry> group_entries;
        group_entries.reserve(all_edges.size());
        for (size_t i = 0; i < all_edges.size(); ++i)
        {
            if (counts_per_slot[i] == 4 && mask_per_slot[i])
            {
                int edge_id = inside_edge_remap[edge_mapping[i]];
                if (edge_id >= 0)
                {
                    group_entries.push_back({edge_id, slot_cube_idx[i], slot_edge_idx[i]});
                }
            }
        }

        // Stable sort by inside-edge id
        std::vector<size_t> order(group_entries.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b)
                         { return group_entries[a].edge_id < group_entries[b].edge_id; });

        for (size_t i = 0; i + 3 < order.size(); i += 4)
        {
            int edge_id = group_entries[order[i]].edge_id;
            if (group_entries[order[i + 1]].edge_id != edge_id ||
                group_entries[order[i + 2]].edge_id != edge_id ||
                group_entries[order[i + 3]].edge_id != edge_id)
            {
                continue;
            }

            std::array<int, 4> cube_idx_4 = {
                group_entries[order[i]].cube_idx,
                group_entries[order[i + 1]].cube_idx,
                group_entries[order[i + 2]].cube_idx,
                group_entries[order[i + 3]].cube_idx};
            int edge_dir = detail::EDGE_DIR_TABLE[group_entries[order[i]].edge_idx];

            const auto & faces_4x2   = detail::DIR_FACES_TABLE[edge_dir];
            const auto & inside_edge = inside_unique_edges[edge_id];
            int inside_v0            = mapping_inside_verts[inside_edge.v0];
            int inside_v1            = mapping_inside_verts[inside_edge.v1];
            if (inside_v0 < 0 || inside_v1 < 0) continue;

            for (int k = 0; k < 4; ++k)
            {
                int c_a    = cube_idx_4[detail::ADJ_PAIRS[2 * k]];
                int c_b    = cube_idx_4[detail::ADJ_PAIRS[2 * k + 1]];
                int face_a = faces_4x2[k][0];
                int face_b = faces_4x2[k][1];

                int case_a = case_ids_expand[c_a];
                int case_b = case_ids_expand[c_b];
                int edge_a = tables::get_tet_edge(case_a, face_a);
                int edge_b = tables::get_tet_edge(case_b, face_b);
                if (edge_a < 0 || edge_a >= 12 || edge_b < 0 || edge_b >= 12) continue;

                int quad_edge_a = edge_center_vertex_idx(c_a, edge_a);
                int quad_edge_b = edge_center_vertex_idx(c_b, edge_b);
                if (quad_edge_a < 0 || quad_edge_b < 0) continue;

                Eigen::Vector4i tet;
                tet[0] = quad_edge_a;
                tet[1] = quad_edge_b;
                tet[2] = inside_v0;
                tet[3] = inside_v1;
                tets.push_back(tet);
            }
        }

        // Convert to output matrix
        result.tets.resize(tets.size(), 4);
        for (size_t i = 0; i < tets.size(); ++i)
        {
            result.tets.row(i) = tets[i].transpose();
        }

        return result;
    }

}  // namespace flexi
