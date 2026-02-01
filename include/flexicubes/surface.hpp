#pragma once

#include "types.hpp"
#include "grid.hpp"
#include <unordered_map>
#include <utility>

namespace flexi
{

    struct SurfaceCubes
    {
        ArrayXb mask;      // Boolean mask: true for surface cubes
        MatX8i occupancy;  // Per-cube occupancy (8 corners, true if inside)
        Index num_surface_cubes() const { return mask.count(); }
    };

    struct EdgeKey
    {
        int v0, v1;  // Vertex indices, always stored with v0 < v1
        EdgeKey(int a, int b) : v0(std::min(a, b)), v1(std::max(a, b)) {}
        bool operator==(const EdgeKey & other) const { return v0 == other.v0 && v1 == other.v1; }
    };

    struct EdgeKeyHash
    {
        std::size_t operator()(const EdgeKey & e) const
        {
            return std::hash<int>()(e.v0) ^ (std::hash<int>()(e.v1) * 73856093);
        }
    };

    struct SurfaceEdges
    {
        Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> edges;  // Unique surface edges (v0, v1)
        MatX12i idx_map;                                               // Per-cube-edge mapping to unique edge index (-1 if not surface)
        VecXi counts;                                                  // Number of cubes sharing each cube-edge slot
        ArrayXb surf_edges_mask;                                       // Per-cube-edge: true if surface edge

        Index num_surface_edges() const { return edges.rows(); }
    };

    /**
    * Identifies grid cubes that intersect with the isosurface.
    *
    * A cube intersects the surface if its 8 corner vertices have mixed signs
    * (some positive, some negative in the scalar field).
    *
    * @param sdf Scalar field values at grid vertices (negative = inside)
    * @param cubes Cube corner indices (Mx8)
    * @return SurfaceCubes with mask and per-corner occupancy
    */
    inline SurfaceCubes identify_surface_cubes(const VecXd & sdf, const MatX8i & cubes)
    {
        const Index num_cubes = cubes.rows();

        SurfaceCubes result;
        result.mask.resize(num_cubes);
        result.occupancy.resize(num_cubes, 8);

        for (Index i = 0; i < num_cubes; ++i)
        {
            int occ_sum = 0;
            for (int c = 0; c < 8; ++c)
            {
                int vert_idx           = cubes(i, c);
                bool is_inside         = sdf[vert_idx] < 0.0;
                result.occupancy(i, c) = is_inside ? 1 : 0;
                occ_sum += is_inside ? 1 : 0;
            }
            // Surface cube: has at least one corner inside and one outside
            result.mask[i] = (occ_sum > 0) && (occ_sum < 8);
        }

        return result;
    }

    /**
    * Identifies grid edges that intersect with the isosurface.
    *
    * An edge intersects the surface if its two endpoint vertices have opposite
    * signs in the scalar field. This function also deduplicates edges shared
    * by multiple cubes and creates a mapping from cube-local edges to unique
    * surface edge indices.
    *
    * @param sdf Scalar field values at grid vertices
    * @param cubes Cube corner indices (Mx8)
    * @param surf_cubes Surface cube identification result
    * @return SurfaceEdges with unique edges and mapping information
    *
    */
    inline SurfaceEdges identify_surface_edges(
        const VecXd & sdf,
        const MatX8i & cubes,
        const SurfaceCubes & surf_cubes)
    {
        const Index num_cubes = cubes.rows();

        // Count surface cubes
        Index num_surf_cubes = surf_cubes.num_surface_cubes();

        // Build all edges from surface cubes
        // Each surface cube contributes 12 edges
        std::vector<EdgeKey> all_edges;
        std::vector<Index> cube_indices;  // Which surface cube each edge came from
        std::vector<int> edge_indices;    // Which local edge (0-11) in the cube

        all_edges.reserve(num_surf_cubes * 12);
        cube_indices.reserve(num_surf_cubes * 12);
        edge_indices.reserve(num_surf_cubes * 12);

        for (Index i = 0; i < num_cubes; ++i)
        {
            if (!surf_cubes.mask[i]) continue;

            for (int e = 0; e < 12; ++e)
            {
                int c0 = detail::CUBE_EDGES_FLAT[e * 2];
                int c1 = detail::CUBE_EDGES_FLAT[e * 2 + 1];
                int v0 = cubes(i, c0);
                int v1 = cubes(i, c1);
                all_edges.emplace_back(v0, v1);
                cube_indices.push_back(i);
                edge_indices.push_back(e);
            }
        }

        // Find unique edges and count occurrences
        std::unordered_map<EdgeKey, int, EdgeKeyHash> edge_to_unique;
        std::vector<EdgeKey> unique_edges;
        std::vector<int> edge_counts;
        std::vector<int> edge_mapping(all_edges.size());  // Maps to unique edge index

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

        // Determine which unique edges are surface edges (sign change)
        std::vector<bool> is_surface_edge(unique_edges.size());
        for (size_t i = 0; i < unique_edges.size(); ++i)
        {
            bool occ0          = sdf[unique_edges[i].v0] < 0.0;
            bool occ1          = sdf[unique_edges[i].v1] < 0.0;
            is_surface_edge[i] = (occ0 != occ1);
        }

        // Create mapping from surface edges to contiguous indices
        std::vector<int> surf_edge_remap(unique_edges.size(), -1);
        int num_surf_edges = 0;
        for (size_t i = 0; i < unique_edges.size(); ++i)
        {
            if (is_surface_edge[i])
            {
                surf_edge_remap[i] = num_surf_edges++;
            }
        }

        // Build result
        SurfaceEdges result;

        // Surface edges array
        result.edges.resize(num_surf_edges, 2);
        for (size_t i = 0; i < unique_edges.size(); ++i)
        {
            if (is_surface_edge[i])
            {
                int idx              = surf_edge_remap[i];
                result.edges(idx, 0) = unique_edges[i].v0;
                result.edges(idx, 1) = unique_edges[i].v1;
            }
        }

        // Per-cube-edge index map and masks
        result.idx_map.resize(num_surf_cubes, 12);
        result.idx_map.setConstant(-1);
        result.counts.resize(num_surf_cubes * 12);
        result.surf_edges_mask.resize(num_surf_cubes * 12);

        Index surf_cube_idx = 0;
        for (Index i = 0; i < num_cubes; ++i)
        {
            if (!surf_cubes.mask[i]) continue;

            for (int e = 0; e < 12; ++e)
            {
                size_t flat_idx = surf_cube_idx * 12 + e;
                // Find the index in all_edges for this cube/edge
                // Note: all_edges is built in order, so we can compute directly
                size_t all_edges_idx  = 0;
                Index seen_surf_cubes = 0;
                for (Index j = 0; j < i; ++j)
                {
                    if (surf_cubes.mask[j]) seen_surf_cubes++;
                }
                all_edges_idx = seen_surf_cubes * 12 + e;

                int unique_idx                   = edge_mapping[all_edges_idx];
                result.counts[flat_idx]          = edge_counts[unique_idx];
                result.surf_edges_mask[flat_idx] = is_surface_edge[unique_idx];

                if (is_surface_edge[unique_idx])
                {
                    result.idx_map(surf_cube_idx, e) = surf_edge_remap[unique_idx];
                }
            }
            surf_cube_idx++;
        }

        return result;
    }

    inline VecXi get_surface_cube_indices(const SurfaceCubes & surf_cubes)
    {
        Index count = surf_cubes.num_surface_cubes();
        VecXi indices(count);

        Index j = 0;
        for (Index i = 0; i < surf_cubes.mask.size(); ++i)
        {
            if (surf_cubes.mask[i])
            {
                indices[j++] = static_cast<int>(i);
            }
        }

        return indices;
    }

    inline MatX8i get_surface_cubes_fx8(const MatX8i & cubes, const SurfaceCubes & surf_cubes)
    {
        Index count = surf_cubes.num_surface_cubes();
        MatX8i result(count, 8);

        Index j = 0;
        for (Index i = 0; i < surf_cubes.mask.size(); ++i)
        {
            if (surf_cubes.mask[i])
            {
                result.row(j++) = cubes.row(i);
            }
        }

        return result;
    }

    inline MatX8i get_surface_occupancy(const SurfaceCubes & surf_cubes)
    {
        Index count = surf_cubes.num_surface_cubes();
        MatX8i result(count, 8);

        Index j = 0;
        for (Index i = 0; i < surf_cubes.mask.size(); ++i)
        {
            if (surf_cubes.mask[i])
            {
                result.row(j++) = surf_cubes.occupancy.row(i);
            }
        }

        return result;
    }

}  // namespace flexi
