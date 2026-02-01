#pragma once

#include "types.hpp"
#include "tables.hpp"
#include "surface.hpp"
#include <vector>

namespace flexi
{

    /**
    * Compute Marching Cubes case IDs from corner occupancy.
    *
    * The case ID is a number from 0-255 where each bit represents
    * whether that corner is inside (1) or outside (0) the surface.
    *
    * @param occupancy Per-cube corner occupancy (Nx8, 1=inside, 0=outside)
    * @return Vector of case IDs (0-255)
    */
    inline VecXi compute_case_ids(const MatX8i & occupancy)
    {
        const Index n = occupancy.rows();
        VecXi case_ids(n);

        for (Index i = 0; i < n; ++i)
        {
            int case_id = 0;
            for (int c = 0; c < 8; ++c)
            {
                if (occupancy(i, c))
                {
                    case_id += detail::CUBE_CORNERS_IDX[c];
                }
            }
            case_ids[i] = case_id;
        }

        return case_ids;
    }

    /**
    * Resolve ambiguous cases C16 and C19 in the Dual Marching Cubes algorithm.
    *
    * Cases C16 and C19 have topological ambiguity that can lead to cracks if
    * adjacent cubes resolve the ambiguity differently. This function checks
    * neighboring cubes and inverts cases when necessary to maintain consistency.
    *
    * The algorithm:
    * 1. Identify cubes with problematic configurations (check_table[case_id][0] == 1)
    * 2. For each problematic cube, look at the adjacent cube in the direction
    *    specified by check_table[case_id][1:4]
    * 3. If the adjacent cube also has a problematic configuration (is_problematic == 1),
    *    invert this cube's case to the value in check_table[case_id][4]
    *
    * @param case_ids Initial case IDs (modified in place)
    * @param surf_cubes Surface cube identification result
    * @param res Grid resolution
    */
    inline void resolve_c16_c19(
        VecXi & case_ids,
        const SurfaceCubes & surf_cubes,
        const Resolution & res)
    {
        const int rx = res.x;
        const int ry = res.y;
        const int rz = res.z;

        // Build mapping from surface cube index to grid position
        // and 3D array to look up surface cube status
        const Index num_total_cubes = surf_cubes.mask.size();
        (void) case_ids.size();  // num_surf_cubes unused but kept for documentation

        // Create 3D grid to store problem configuration info
        // For each grid cell: [is_problematic, offset_x, offset_y, offset_z, inverted_case]
        struct ProblemConfig
        {
            bool is_problematic = false;
            int offset_x = 0, offset_y = 0, offset_z = 0;
            int inverted_case   = 0;
            Index surf_cube_idx = -1;  // Index into surface cubes array
        };

        std::vector<ProblemConfig> problem_grid(rx * ry * rz);

        // Helper to get linear index from 3D position
        auto grid_idx = [rx, ry](int x, int y, int z) -> Index
        {
            return z * ry * rx + y * rx + x;
        };

        // Helper to get 3D position from linear index
        auto grid_pos = [rx, ry](Index idx) -> std::tuple<int, int, int>
        {
            int z   = static_cast<int>(idx / (rx * ry));
            int rem = static_cast<int>(idx % (rx * ry));
            int y   = rem / rx;
            int x   = rem % rx;
            return {x, y, z};
        };

        // First pass: identify problematic configurations
        Index surf_idx = 0;
        for (Index i = 0; i < num_total_cubes; ++i)
        {
            if (!surf_cubes.mask[i]) continue;

            int case_id        = case_ids[surf_idx];
            const auto & check = tables::CHECK_TABLE[case_id];

            if (check[0] == 1)
            {  // Is problematic
                auto [x, y, z]        = grid_pos(i);
                auto & config         = problem_grid[grid_idx(x, y, z)];
                config.is_problematic = true;
                config.offset_x       = check[1];
                config.offset_y       = check[2];
                config.offset_z       = check[3];
                config.inverted_case  = check[4];
                config.surf_cube_idx  = surf_idx;
            }
            surf_idx++;
        }

        // Second pass: check adjacent cubes and resolve ambiguity
        surf_idx = 0;
        for (Index i = 0; i < num_total_cubes; ++i)
        {
            if (!surf_cubes.mask[i]) continue;

            int case_id        = case_ids[surf_idx];
            const auto & check = tables::CHECK_TABLE[case_id];

            if (check[0] == 1)
            {  // Is problematic
                auto [x, y, z] = grid_pos(i);

                // Compute adjacent position
                int adj_x = x + check[1];
                int adj_y = y + check[2];
                int adj_z = z + check[3];

                // Check if adjacent is within bounds
                if (adj_x >= 0 && adj_x < rx &&
                    adj_y >= 0 && adj_y < ry &&
                    adj_z >= 0 && adj_z < rz)
                {
                    const auto & adj_config = problem_grid[grid_idx(adj_x, adj_y, adj_z)];

                    // If adjacent cube also has a problematic configuration, invert this one
                    if (adj_config.is_problematic)
                    {
                        case_ids[surf_idx] = check[4];  // Use inverted case
                    }
                }
            }
            surf_idx++;
        }
    }

    /**
    * Compute case IDs for surface cubes with C16/C19 ambiguity resolution.
    *
    * This is the main entry point that combines case ID computation with
    * ambiguity resolution.
    *
    * @param surf_cubes Surface cube identification result
    * @param res Grid resolution
    * @return Case IDs for each surface cube (with ambiguity resolved)
    *
    */
    inline VecXi compute_case_ids_with_resolution(
        const SurfaceCubes & surf_cubes,
        const Resolution & res)
    {
        // Get occupancy for surface cubes only
        MatX8i surf_occ = get_surface_occupancy(surf_cubes);

        // Compute initial case IDs
        VecXi case_ids = compute_case_ids(surf_occ);

        // Resolve C16/C19 ambiguity
        resolve_c16_c19(case_ids, surf_cubes, res);

        return case_ids;
    }

    inline int get_num_dual_vertices(int case_id)
    {
        return tables::get_num_dual_vertices(case_id);
    }

    /**
    * Get edge indices that contribute to a dual vertex.
    *
    * @param case_id Marching Cubes case ID
    * @param vd_idx Dual vertex index (0 to num_dual_vertices-1)
    * @return Vector of edge indices (0-11), excluding -1 sentinel values
    */
    inline std::vector<int> get_dual_vertex_edges(int case_id, int vd_idx)
    {
        const auto & edges = tables::get_dual_vertex_edges(case_id, vd_idx);
        std::vector<int> result;
        result.reserve(7);
        for (int e : edges)
        {
            if (e >= 0)
            {
                result.push_back(e);
            }
        }
        return result;
    }

}  // namespace flexi
