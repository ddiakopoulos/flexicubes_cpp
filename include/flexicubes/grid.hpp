#pragma once

#include "types.hpp"
#include <unordered_map>
#include <cmath>

namespace flexi
{
    struct VoxelGrid
    {
        MatX3 vertices;  // Nx3 vertex positions, centered at origin
        MatX8i cubes;    // Mx8 cube corner indices into vertices

        Index num_vertices() const { return vertices.rows(); }
        Index num_cubes() const { return cubes.rows(); }
    };

    namespace detail
    {
        struct Vec3iHash
        {
            std::size_t operator()(const Vec3i & v) const
            {
                return std::hash<int>()(v[0]) ^ (std::hash<int>()(v[1]) * 73856093) ^ (std::hash<int>()(v[2]) * 19349663);
            }
        };
    }  // namespace detail

    /**
    * Creates a grid of cubes where each cube has 8 corner vertices. Vertices
    * are deduplicated so shared corners are stored only once. The grid is
    * centered at the origin with unit length along each dimension.
    *
    * @param res The resolution of the voxel grid. Can be uniform or per-axis.
    * @return VoxelGrid containing vertices and cube corner indices.
    */
    inline VoxelGrid construct_voxel_grid(const Resolution & res)
    {
        const int rx = res.x;
        const int ry = res.y;
        const int rz = res.z;

        // Total number of cubes
        const int num_cubes = rx * ry * rz;

        // Map from quantized vertex position to unique index
        // Using integer coordinates multiplied by precision factor
        constexpr int PRECISION = 100000;

        std::unordered_map<Vec3i, int, detail::Vec3iHash> vertex_map;
        std::vector<Vec3> vertex_list;
        vertex_list.reserve(num_cubes * 8 / 4);  // Approximate unique vertices

        // Allocate cube indices
        MatX8i cubes(num_cubes, 8);

        int cube_idx = 0;
        for (int iz = 0; iz < rz; ++iz)
        {
            for (int iy = 0; iy < ry; ++iy)
            {
                for (int ix = 0; ix < rx; ++ix)
                {
                    double base_x = static_cast<double>(ix) / rx;
                    double base_y = static_cast<double>(iy) / ry;
                    double base_z = static_cast<double>(iz) / rz;

                    // Process each of the 8 corners
                    for (int corner = 0; corner < 8; ++corner)
                    {
                        int cx = detail::CUBE_CORNERS[corner][0];
                        int cy = detail::CUBE_CORNERS[corner][1];
                        int cz = detail::CUBE_CORNERS[corner][2];

                        double vx = base_x + static_cast<double>(cx) / rx;
                        double vy = base_y + static_cast<double>(cy) / ry;
                        double vz = base_z + static_cast<double>(cz) / rz;

                        Vec3i key(static_cast<int>(std::round(vx * PRECISION)), static_cast<int>(std::round(vy * PRECISION)), static_cast<int>(std::round(vz * PRECISION)));

                        auto it = vertex_map.find(key);
                        int vert_idx;
                        if (it != vertex_map.end())
                        {
                            vert_idx = it->second;
                        }
                        else
                        {
                            vert_idx        = static_cast<int>(vertex_list.size());
                            vertex_map[key] = vert_idx;
                            vertex_list.emplace_back(vx - 0.5, vy - 0.5, vz - 0.5);
                        }

                        cubes(cube_idx, corner) = vert_idx;
                    }
                    ++cube_idx;
                }
            }
        }

        VoxelGrid result;
        result.vertices.resize(vertex_list.size(), 3);
        for (size_t i = 0; i < vertex_list.size(); ++i)
        {
            result.vertices.row(i) = vertex_list[i].transpose();
        }
        result.cubes = std::move(cubes);

        return result;
    }

    inline VoxelGrid construct_voxel_grid(int res)
    {
        return construct_voxel_grid(Resolution(res));
    }

    /**
    * Get vertex positions for a specific cube.
    *
    * @param grid The voxel grid
    * @param cube_idx Index of the cube
    * @return 8x3 matrix of corner positions
    */
    inline Eigen::Matrix<double, 8, 3> get_cube_vertices(const VoxelGrid & grid, Index cube_idx)
    {
        Eigen::Matrix<double, 8, 3> result;
        for (int i = 0; i < 8; ++i)
        {
            int vert_idx  = grid.cubes(cube_idx, i);
            result.row(i) = grid.vertices.row(vert_idx);
        }
        return result;
    }

    /**
    * Get all 12 edge vertex pairs for a cube.
    *
    * @param grid The voxel grid
    * @param cube_idx Index of the cube
    * @return Vector of 12 pairs (v0_idx, v1_idx) for each edge
    */
    inline std::array<std::pair<int, int>, 12> get_cube_edges(const VoxelGrid & grid, Index cube_idx)
    {
        std::array<std::pair<int, int>, 12> edges;
        for (int e = 0; e < 12; ++e)
        {
            int c0   = detail::CUBE_EDGES_FLAT[e * 2];
            int c1   = detail::CUBE_EDGES_FLAT[e * 2 + 1];
            edges[e] = {grid.cubes(cube_idx, c0), grid.cubes(cube_idx, c1)};
        }
        return edges;
    }

}  // namespace flexi
