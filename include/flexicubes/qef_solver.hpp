#pragma once

#include "types.hpp"

namespace flexi
{

    /**
    * Solve Quadratic Error Function (QEF) for dual vertex positioning.
    *
    * The QEF minimizes the sum of squared distances from the dual vertex
    * to the tangent planes at each edge zero-crossing point. This produces
    * feature-preserving vertices that lie on sharp edges and corners.
    *
    * The system to solve is:
    *   A * v = b
    * where A contains normals, b contains dot(point, normal) values.
    *
    * We add Tikhonov regularization to prevent singularity:
    *   [A; reg_scale * I] * v = [b; reg_scale * centroid]
    *
    * @param positions Edge zero-crossing positions (Nx3)
    * @param normals Surface normals at zero-crossings (Nx3)
    * @param centroid Centroid of positions (for regularization)
    * @param reg_scale Regularization scale (default 1e-3)
    * @return Optimal dual vertex position
    *
    * Matches Python: FlexiCubes._solve_vd_QEF()
    */
    inline Vec3 solve_qef(const MatX3 & positions, const MatX3 & normals, const Vec3 & centroid, double reg_scale = 1e-3)
    {
        const Index n = positions.rows();

        if (n == 0)
        {
            return centroid;
        }

        // Build the linear system with regularization
        // A = [normals; reg_scale * I]
        // b = [dot(positions, normals); reg_scale * centroid]
        Eigen::Matrix<double, Eigen::Dynamic, 3> A(n + 3, 3);
        Eigen::VectorXd b(n + 3);

        // Fill normal equations
        A.topRows(n) = normals;
        for (Index i = 0; i < n; ++i)
        {
            b[i] = positions.row(i).dot(normals.row(i));
        }

        // Add regularization (pull towards centroid)
        A.bottomRows(3) = reg_scale * Eigen::Matrix3d::Identity();
        b.tail(3)       = reg_scale * centroid;

        // Solve using least squares
        Vec3 result = A.colPivHouseholderQr().solve(b);

        return result;
    }

    /**
    * Batch QEF solver for multiple dual vertices.
    *
    * Solves QEF for multiple dual vertices where each has up to max_edges
    * contributing edge crossings.
    *
    * @param positions Batched positions [batch, max_edges, 3]
    * @param normals Batched normals [batch, max_edges, 3]
    * @param valid_mask Mask indicating valid edges [batch, max_edges]
    * @param centroids Centroid for each batch item [batch, 3]
    * @param reg_scale Regularization scale
    * @return Dual vertex positions [batch, 3]
    */
    inline MatX3 solve_qef_batch(
        const std::vector<MatX3> & positions,
        const std::vector<MatX3> & normals,
        const std::vector<ArrayXb> & valid_mask,
        const MatX3 & centroids,
        double reg_scale = 1e-3)
    {
        const Index batch_size = static_cast<Index>(positions.size());
        MatX3 results(batch_size, 3);

        for (Index i = 0; i < batch_size; ++i)
        {
            // Count valid edges
            Index valid_count = valid_mask[i].count();

            if (valid_count == 0)
            {
                results.row(i) = centroids.row(i);
                continue;
            }

            // Extract valid positions and normals
            MatX3 valid_pos(valid_count, 3);
            MatX3 valid_norm(valid_count, 3);
            Index j = 0;
            for (Index k = 0; k < valid_mask[i].size(); ++k)
            {
                if (valid_mask[i][k])
                {
                    valid_pos.row(j)  = positions[i].row(k);
                    valid_norm.row(j) = normals[i].row(k);
                    ++j;
                }
            }

            results.row(i) = solve_qef(valid_pos, valid_norm, centroids.row(i).transpose(), reg_scale);
        }

        return results;
    }

    /**
    * Simplified QEF solver without explicit normals.
    *
    * When normals are not available, we can use a simpler averaging
    * approach weighted by some criteria.
    *
    * @param positions Edge zero-crossing positions (Nx3)
    * @param weights Per-position weights (N)
    * @return Weighted average position
    */
    inline Vec3 solve_qef_weighted_average(const MatX3 & positions, const VecXd & weights)
    {
        const Index n = positions.rows();

        if (n == 0)
        {
            return Vec3::Zero();
        }

        double total_weight = weights.sum();
        if (total_weight < 1e-10)
        {
            // Fall back to simple average
            return positions.colwise().mean();
        }

        Vec3 result = Vec3::Zero();
        for (Index i = 0; i < n; ++i)
        {
            result += weights[i] * positions.row(i).transpose();
        }

        return result / total_weight;
    }

}  // namespace flexi
