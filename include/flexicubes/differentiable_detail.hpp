#pragma once

#include "types.hpp"
#include "differentiable.hpp"

namespace flexi
{
    namespace detail
    {
        /**
        * Differentiable linear interpolation
        * Computes the zero-crossing position with automatic differentiation support.
        */
        inline void linear_interp_diff(const adept::aVector & p0, const adept::aVector & p1, adept::adouble s0, adept::adouble s1, adept::aVector & result)
        {
            adept::adouble denom = s1 - s0;

            // Handle near-zero denominator
            adept::adouble safe_denom = adept::fmax(adept::fabs(denom), 1e-10);
            adept::adouble sign       = (denom >= 0) ? 1.0 : -1.0;
            safe_denom                = safe_denom * sign;

            result = (s1 * p0 - s0 * p1) / safe_denom;
        }

        /**
        * Differentiable weighted linear interpolation.
        */
        inline void linear_interp_weighted_diff(
            const adept::aVector & p0,
            const adept::aVector & p1,
            adept::adouble s0,
            adept::adouble s1,
            adept::adouble alpha0,
            adept::adouble alpha1,
            adept::aVector & result)
        {
            adept::adouble ws0 = s0 * alpha0;
            adept::adouble ws1 = s1 * alpha1;
            linear_interp_diff(p0, p1, ws0, ws1, result);
        }

        /**
        * Differentiable dual vertex computation for a single vertex.
        *
        * @param edge_positions Edge endpoint positions [num_edges, 2, 3]
        * @param edge_sdf SDF values at edge endpoints [num_edges, 2]
        * @param edge_alpha Alpha weights at edge endpoints [num_edges, 2]
        * @param edge_beta Beta weights for each edge [num_edges]
        * @param result Output dual vertex position [3]
        */
        inline void compute_dual_vertex_diff(
            const std::vector<std::array<adept::aVector, 2>> & edge_positions,
            const std::vector<std::array<adept::adouble, 2>> & edge_sdf,
            const std::vector<std::array<adept::adouble, 2>> & edge_alpha,
            const std::vector<adept::adouble> & edge_beta,
            adept::aVector & result)
        {
            const size_t num_edges = edge_positions.size();

            result.resize(3);
            result                  = 0.0;
            adept::adouble beta_sum = 0.0;

            for (size_t i = 0; i < num_edges; ++i)
            {
                // Compute alpha-weighted zero crossing
                adept::aVector ue(3);
                linear_interp_weighted_diff(
                    edge_positions[i][0], edge_positions[i][1],
                    edge_sdf[i][0], edge_sdf[i][1],
                    edge_alpha[i][0], edge_alpha[i][1],
                    ue);

                // Accumulate beta-weighted contribution
                result += edge_beta[i] * ue;
                beta_sum += edge_beta[i];
            }

            // Normalize
            result = result / adept::fmax(beta_sum, 1e-10);
        }

        /**
        * Compute L_dev loss with Adept differentiation.
        */
        inline adept::adouble compute_l_dev_diff(
            const adept::aVector & dual_vertex,
            const std::vector<adept::aVector> & zero_crossings)
        {
            const size_t n = zero_crossings.size();
            if (n == 0) return 0.0;

            // Compute distances
            std::vector<adept::adouble> distances(n);
            adept::adouble sum_dist = 0.0;

            for (size_t i = 0; i < n; ++i)
            {
                adept::aVector diff = zero_crossings[i] - dual_vertex;
                distances[i] = adept::norm2(diff);
                sum_dist += distances[i];
            }

            adept::adouble mean_dist = sum_dist / static_cast<double>(n);

            // MAD
            adept::adouble mad = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                mad += adept::fabs(distances[i] - mean_dist);
            }

            return mad / static_cast<double>(n);
        }

    }  // namespace detail

}  // namespace flexi
