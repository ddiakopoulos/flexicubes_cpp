#pragma once

#include "tables_data.inl"

namespace flexi
{
    namespace tables
    {

        /**
        * Get the number of dual vertices for a given MC case.
        * @param case_id Marching Cubes case ID (0-255)
        * @return Number of dual vertices (0-4)
        */
        inline constexpr int get_num_dual_vertices(int case_id)
        {
            return NUM_VD_TABLE[case_id];
        }

        /**
        * Get edge indices for a dual vertex in a given MC case.
        * @param case_id Marching Cubes case ID (0-255)
        * @param vd_idx Dual vertex index (0-3)
        * @return Array of 7 edge indices (-1 indicates unused)
        */
        inline constexpr const std::array<int, 7> & get_dual_vertex_edges(int case_id, int vd_idx)
        {
            return DMC_TABLE[case_id][vd_idx];
        }

        /**
        * Check if a MC case needs ambiguity resolution (C16 or C19).
        * @param case_id Marching Cubes case ID (0-255)
        * @return true if case requires ambiguity check
        */
        inline constexpr bool is_ambiguous_case(int case_id)
        {
            return CHECK_TABLE[case_id][0] == 1;
        }

        /**
        * Get the neighbor offset for ambiguity resolution.
        * @param case_id Marching Cubes case ID (0-255)
        * @return Vec3i-like offset to adjacent cube
        */
        inline constexpr std::array<int, 3> get_ambiguity_offset(int case_id)
        {
            return {CHECK_TABLE[case_id][1], CHECK_TABLE[case_id][2], CHECK_TABLE[case_id][3]};
        }

        /**
        * Get the inverted case ID for ambiguity resolution.
        * @param case_id Marching Cubes case ID (0-255)
        * @return Inverted case ID
        */
        inline constexpr int get_inverted_case(int case_id)
        {
            return CHECK_TABLE[case_id][4];
        }

        /**
        * Get tetrahedralization lookup for a case and face.
        * @param case_id Marching Cubes case ID (0-255)
        * @param face_idx Face index (0-5)
        * @return Edge index for tetrahedralization
        */
        inline constexpr int get_tet_edge(int case_id, int face_idx)
        {
            return TET_TABLE[case_id][face_idx];
        }


        /**
        * Validate table dimensions and basic invariants.
        * @return true if tables pass validation
        */
        inline constexpr bool validate_tables()
        {
            // Check DMC table dimensions
            if (DMC_TABLE.size() != 256) return false;
            for (const auto & entry : DMC_TABLE)
            {
                if (entry.size() != 4) return false;
                for (const auto & row : entry)
                {
                    if (row.size() != 7) return false;
                }
            }

            // Check NUM_VD table
            if (NUM_VD_TABLE.size() != 256) return false;
            for (int v : NUM_VD_TABLE)
            {
                if (v < 0 || v > 4) return false;
            }

            // Check CHECK table
            if (CHECK_TABLE.size() != 256) return false;
            for (const auto & entry : CHECK_TABLE)
            {
                if (entry.size() != 5) return false;
            }

            // Check TET table
            if (TET_TABLE.size() != 256) return false;
            for (const auto & entry : TET_TABLE)
            {
                if (entry.size() != 6) return false;
            }

            // Case 0 should have 0 dual vertices (all corners same sign)
            if (NUM_VD_TABLE[0] != 0) return false;

            // Case 255 should have 0 dual vertices (all corners same sign)
            if (NUM_VD_TABLE[255] != 0) return false;

            return true;
        }

        static_assert(validate_tables(), "FlexiCubes lookup tables failed validation");

    }  // namespace tables

}  // namespace flexi
