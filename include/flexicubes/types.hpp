#pragma once

#include <iostream>
#include <string>
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>

#include <adept.h>
#include <adept_arrays.h>

namespace flexi
{
    // Scalar types
    using Scalar = double;
    using Index  = Eigen::Index;

    // Fixed-size vectors
    using Vec2 = Eigen::Vector2d;
    using Vec3 = Eigen::Vector3d;
    using Vec4 = Eigen::Vector4d;

    using Vec2i = Eigen::Vector2i;
    using Vec3i = Eigen::Vector3i;
    using Vec4i = Eigen::Vector4i;

    // Fixed-size matrices
    using Mat3 = Eigen::Matrix3d;
    using Mat4 = Eigen::Matrix4d;

    // Dynamic matrices (row-major for cache efficiency with vertex data)
    using MatX3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using MatX4 = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;
    using MatXd = Eigen::MatrixXd;

    // Dynamic integer matrices for indices
    using MatX3i  = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;
    using MatX4i  = Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>;
    using MatX8i  = Eigen::Matrix<int, Eigen::Dynamic, 8, Eigen::RowMajor>;
    using MatX12i = Eigen::Matrix<int, Eigen::Dynamic, 12, Eigen::RowMajor>;
    using MatXi   = Eigen::MatrixXi;

    // Dynamic vectors
    using VecXd = Eigen::VectorXd;
    using VecXi = Eigen::VectorXi;

    // Array types (for element-wise operations)
    using ArrayXd = Eigen::ArrayXd;
    using ArrayXi = Eigen::ArrayXi;
    using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;

    struct Resolution
    {
        int x, y, z;

        Resolution(int uniform) : x(uniform), y(uniform), z(uniform) {}
        Resolution(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}

        int total_cubes() const { return x * y * z; }
        int total_verts() const { return (x + 1) * (y + 1) * (z + 1); }

        bool operator==(const Resolution & other) const
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    namespace detail
    {
        // Standard cube corners ordered as in FlexiCubes:
        // Corner 0: (0,0,0), Corner 1: (1,0,0), Corner 2: (0,1,0), Corner 3: (1,1,0)
        // Corner 4: (0,0,1), Corner 5: (1,0,1), Corner 6: (0,1,1), Corner 7: (1,1,1)
        constexpr std::array<std::array<int, 3>, 8> CUBE_CORNERS = {
            {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}}
        };

        // Powers of 2 for corner indices (used in case ID computation)
        constexpr std::array<int, 8> CUBE_CORNERS_IDX = {1, 2, 4, 8, 16, 32, 64, 128};

        // Cube edges as pairs of corner indices (12 edges total)
        // Edges 0-3: along X axis, Edges 4-7: along Z axis, Edges 8-11: along Y axis
        constexpr std::array<std::array<int, 2>, 12> CUBE_EDGES = {
            {{0, 1}, {1, 5}, {4, 5}, {0, 4}, {2, 3}, {3, 7}, {6, 7}, {2, 6}, {2, 0}, {3, 1}, {7, 5}, {6, 4}}
        };

        // Flat edge array for indexing (matches Python cube_edges)
        constexpr std::array<int, 24> CUBE_EDGES_FLAT = {0, 1, 1, 5, 4, 5, 0, 4, 2, 3, 3, 7, 6, 7, 2, 6, 2, 0, 3, 1, 7, 5, 6, 4};

        // Edge direction table: maps edge index to axis (0=X, 1=Y, 2=Z)
        constexpr std::array<int, 12> EDGE_DIR_TABLE = {0, 2, 0, 2, 0, 2, 0, 2, 1, 1, 1, 1};

        // Direction-to-faces table for tetrahedralization
        // dir_faces_table[axis][pair] = [face1, face2]
        constexpr std::array<std::array<std::array<int, 2>, 4>, 3> DIR_FACES_TABLE = {{{{{5, 4}, {3, 2}, {4, 5}, {2, 3}}},
                                                                                       {{{5, 4}, {1, 0}, {4, 5}, {0, 1}}},
                                                                                       {{{3, 2}, {1, 0}, {2, 3}, {0, 1}}}}};

        // Adjacent pairs for tetrahedralization
        constexpr std::array<int, 8> ADJ_PAIRS = {0, 1, 1, 3, 3, 2, 2, 0};

        // Quad split patterns
        constexpr std::array<int, 6> QUAD_SPLIT_1     = {0, 1, 2, 0, 2, 3};
        constexpr std::array<int, 6> QUAD_SPLIT_2     = {0, 1, 3, 3, 1, 2};
        constexpr std::array<int, 8> QUAD_SPLIT_TRAIN = {0, 1, 1, 2, 2, 3, 3, 0};

    }  // namespace detail

}  // namespace flexi

namespace flexi
{
    namespace differentiable
    {

        using AScalar = adept::adouble;
        using AVector = adept::aVector;
        using AMatrix = adept::aMatrix;

        // Helper to convert Eigen to Adept
        inline AVector eigen_to_adept(const VecXd & v)
        {
            AVector result(v.size());
            for (Index i = 0; i < v.size(); ++i)
            {
                result[i] = v[i];
            }
            return result;
        }

        // Helper to convert Adept to Eigen
        inline VecXd adept_to_eigen(const AVector & v)
        {
            VecXd result(v.size());
            for (int i = 0; i < v.size(); ++i)
            {
                result[i] = adept::value(v[i]);
            }
            return result;
        }

    }  // namespace differentiable
}  // namespace flexi
