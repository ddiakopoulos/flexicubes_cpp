#pragma once

#include "types.hpp"
#include "dual_vertex.hpp"

namespace flexi
{
    /**
    * Compute the L_dev regularization loss.
    *
    * L_dev measures the deviation of edge zero-crossings from the dual vertex,
    * as defined in Equation 8 of the FlexiCubes paper. This regularization
    * encourages dual vertices to stay close to the edge zero-crossings.
    *
    * For each dual vertex:
    * 1. Compute mean L2 distance from zero-crossings to dual vertex
    * 2. Compute Mean Absolute Deviation (MAD) of distances from the mean
    *
    * The MAD formulation penalizes irregular vertex distributions while
    * allowing controlled deviation from the exact zero-crossing average.
    *
    * @param vd Dual vertex computation result (contains pre-computed L_dev)
    * @return Total L_dev loss (sum over all dual vertices)
    */
    inline double compute_total_l_dev(const DualVertexResult & vd)
    {
        return vd.L_dev.sum();
    }

    /**
    * Compute mean L_dev per dual vertex.
    *
    * @param vd Dual vertex computation result
    * @return Mean L_dev value
    */
    inline double compute_mean_l_dev(const DualVertexResult & vd)
    {
        if (vd.num_dual_vertices == 0) return 0.0;
        return vd.L_dev.sum() / static_cast<double>(vd.num_dual_vertices);
    }

    /**
    * Compute L_dev for a single dual vertex.
    *
    * @param dual_vertex Position of the dual vertex
    * @param zero_crossings Zero-crossing positions that contribute to this vertex
    * @return L_dev value (Mean Absolute Deviation of distances)
    */
    inline double compute_l_dev_single(
        const Vec3 & dual_vertex,
        const std::vector<Vec3> & zero_crossings)
    {
        if (zero_crossings.empty()) return 0.0;

        // Compute distances from each zero-crossing to dual vertex
        std::vector<double> distances;
        distances.reserve(zero_crossings.size());
        double sum_dist = 0.0;

        for (const auto & zc : zero_crossings)
        {
            double d = (zc - dual_vertex).norm();
            distances.push_back(d);
            sum_dist += d;
        }

        // Mean distance
        double mean_dist = sum_dist / distances.size();

        // Mean Absolute Deviation
        double mad = 0.0;
        for (double d : distances)
        {
            mad += std::abs(d - mean_dist);
        }

        return mad / distances.size();
    }

    /**
    * Structure to hold regularization weights for optimization.
    */
    struct RegularizationWeights
    {
        double l_dev_weight      = 0.5;  // Weight for L_dev loss
        double smoothness_weight = 0.0;  // Weight for mesh smoothness
        double volume_weight     = 0.0;  // Weight for volume preservation

        static RegularizationWeights defaults()
        {
            return RegularizationWeights {};
        }
    };

    /**
    * Compute combined regularization loss.
    *
    * @param vd Dual vertex computation result
    * @param weights Regularization weights
    * @return Weighted sum of regularization terms
    */
    inline double compute_regularization_loss(const DualVertexResult & vd, const RegularizationWeights & weights = RegularizationWeights::defaults())
    {
        double loss = 0.0;

        if (weights.l_dev_weight > 0.0)
        {
            loss += weights.l_dev_weight * compute_mean_l_dev(vd);
        }

        // Additional regularization terms can be added here
        // e.g., mesh smoothness, volume preservation, etc.

        return loss;
    }

    /**
    * Compute per-face regularity metric.
    *
    * Measures how regular (equilateral) each triangle face is.
    * Returns values in [0, 1] where 1 is perfectly equilateral.
    *
    * @param vertices Mesh vertices
    * @param faces Triangle faces
    * @return Per-face regularity values
    */
    inline VecXd compute_face_regularity(const MatX3 & vertices, const MatX3i & faces)
    {
        const Index num_faces = faces.rows();
        VecXd regularity(num_faces);

        for (Index f = 0; f < num_faces; ++f)
        {
            Vec3 v0 = vertices.row(faces(f, 0)).transpose();
            Vec3 v1 = vertices.row(faces(f, 1)).transpose();
            Vec3 v2 = vertices.row(faces(f, 2)).transpose();

            // Edge lengths
            double e0 = (v1 - v0).norm();
            double e1 = (v2 - v1).norm();
            double e2 = (v0 - v2).norm();

            // For equilateral triangle, all edges are equal
            double mean_edge = (e0 + e1 + e2) / 3.0;
            if (mean_edge < 1e-10)
            {
                regularity[f] = 0.0;
                continue;
            }

            // Compute coefficient of variation (lower is more regular)
            double var = ((e0 - mean_edge) * (e0 - mean_edge) +
                          (e1 - mean_edge) * (e1 - mean_edge) +
                          (e2 - mean_edge) * (e2 - mean_edge)) /
                         3.0;
            double cv = std::sqrt(var) / mean_edge;

            // Convert to regularity measure (1 = perfect, 0 = degenerate)
            regularity[f] = std::exp(-cv * cv);
        }

        return regularity;
    }

    /**
    * Compute mesh quality statistics.
    */
    struct MeshQualityStats
    {
        double min_regularity;
        double max_regularity;
        double mean_regularity;
        double mean_l_dev;
        Index num_degenerate_faces;  // Faces with near-zero area
    };

    inline MeshQualityStats compute_mesh_quality(const MatX3 & vertices, const MatX3i & faces, const DualVertexResult & vd)
    {
        MeshQualityStats stats;

        VecXd reg = compute_face_regularity(vertices, faces);

        stats.min_regularity  = reg.minCoeff();
        stats.max_regularity  = reg.maxCoeff();
        stats.mean_regularity = reg.mean();
        stats.mean_l_dev      = compute_mean_l_dev(vd);

        // Count degenerate faces
        stats.num_degenerate_faces       = 0;
        constexpr double DEGEN_THRESHOLD = 1e-8;
        for (Index f = 0; f < faces.rows(); ++f)
        {
            Vec3 v0 = vertices.row(faces(f, 0)).transpose();
            Vec3 v1 = vertices.row(faces(f, 1)).transpose();
            Vec3 v2 = vertices.row(faces(f, 2)).transpose();

            Vec3 cross = (v1 - v0).cross(v2 - v0);
            if (cross.norm() < DEGEN_THRESHOLD)
            {
                stats.num_degenerate_faces++;
            }
        }

        return stats;
    }

}  // namespace flexi
