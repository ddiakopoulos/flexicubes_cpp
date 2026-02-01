// Example showing optimization loop with learnable weights.
// This demonstrates the training mode of FlexiCubes.

#include <flexicubes/flexicubes.hpp>
#include <iostream>
#include <random>

using namespace flexi;

/**
 * Simple gradient descent optimizer.
 */
class SimpleOptimizer
{
public:
    SimpleOptimizer(double learning_rate) : lr_(learning_rate) {}

    void step(VecXd & params, const VecXd & grads)
    {
        params -= lr_ * grads;
    }

    template <typename MatType>
    void step(MatType & params, const MatType & grads)
    {
        params -= lr_ * grads;
    }

private:
    double lr_;
};

/**
 * Compute target sphere SDF.
 */
VecXd sphere_sdf(const MatX3 & positions, double radius)
{
    VecXd sdf(positions.rows());
    for (Index i = 0; i < positions.rows(); ++i)
    {
        sdf[i] = positions.row(i).norm() - radius;
    }
    return sdf;
}

/**
 * Compute simple loss between two meshes (vertex distance).
 */
double compute_loss(const Mesh & mesh, const Mesh & target)
{
    if (mesh.empty() || target.empty()) return 1e10;

    // Simple loss: compare number of vertices
    // In a real application, you would use Chamfer distance or similar
    double loss = std::abs(static_cast<double>(mesh.num_vertices() - target.num_vertices()));

    // Add L_dev regularization
    loss += 0.1 * mesh.l_dev.sum();

    return loss;
}

int main()
{
    std::cout << "FlexiCubes Optimization Loop Example\n";
    std::cout << "=====================================\n\n";

    // Adept-2 is always available

    // Setup
    const int resolution       = 16;
    const int num_iterations   = 10;
    const double learning_rate = 0.01;

    FlexiCubes fc;

    // Generate grid
    auto grid = fc.construct_voxel_grid(resolution);
    std::cout << "Grid: " << grid.num_vertices() << " vertices, "
              << grid.num_cubes() << " cubes\n\n";

    // Target: sphere with radius 0.4
    VecXd target_sdf = sphere_sdf(grid.vertices, 0.4);

    // Extract target mesh for comparison
    Options opts     = Options::training_defaults();
    Mesh target_mesh = fc.extract_surface(grid.vertices, target_sdf, grid.cubes,  Resolution(resolution), nullptr, nullptr, nullptr, nullptr, opts);
    std::cout << "Target mesh: " << target_mesh.num_vertices() << " vertices, " << target_mesh.num_faces() << " faces\n\n";

    // Initialize learnable parameters (small random perturbations)
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.1);

    // Perturbed SDF (what we're trying to optimize)
    VecXd sdf = target_sdf;
    for (Index i = 0; i < sdf.size(); ++i)
    {
        sdf[i] += noise(rng) * 0.05;
    }

    // Learnable weights
    const Index num_cubes = grid.num_cubes();
    Eigen::Matrix<double, Eigen::Dynamic, 12, Eigen::RowMajor> beta(num_cubes, 12);
    Eigen::Matrix<double, Eigen::Dynamic, 8, Eigen::RowMajor> alpha(num_cubes, 8);
    VecXd gamma(num_cubes);

    beta.setZero();  // Will be normalized via tanh
    alpha.setZero();
    gamma.setZero();

    // Add noise to weights
    for (Index i = 0; i < num_cubes; ++i)
    {
        for (int j = 0; j < 12; ++j) beta(i, j) = noise(rng) * 0.1;
        for (int j = 0; j < 8; ++j) alpha(i, j) = noise(rng) * 0.1;
        gamma[i] = noise(rng) * 0.1;
    }

    SimpleOptimizer optimizer(learning_rate);

    std::cout << "Starting optimization...\n";
    std::cout << "-----------------------\n";

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        // Extract mesh with current parameters
        Mesh mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes,
                                       Resolution(resolution),
                                       &beta, &alpha, &gamma, nullptr, opts);

        // Compute loss
        double loss = compute_loss(mesh, target_mesh);

        std::cout << "Iteration " << iter << ": loss = " << loss
                  << ", vertices = " << mesh.num_vertices()
                  << ", faces = " << mesh.num_faces()
                  << ", L_dev = " << (mesh.l_dev.size() > 0 ? mesh.l_dev.mean() : 0.0)
                  << "\n";

        // Compute gradients using finite differences
        // (In practice with Adept, you would get analytical gradients)
        const double eps = 1e-4;

        // Update SDF using finite difference gradients
        VecXd sdf_grad(sdf.size());
        sdf_grad.setZero();

        // Only compute gradients for a subset (for speed in this example)
        for (Index i = 0; i < std::min<Index>(100, sdf.size()); ++i)
        {
            sdf[i] += eps;
            Mesh mesh_plus   = fc.extract_surface(grid.vertices, sdf, grid.cubes,
                                                  Resolution(resolution),
                                                  &beta, &alpha, &gamma, nullptr, opts);
            double loss_plus = compute_loss(mesh_plus, target_mesh);

            sdf[i] -= 2 * eps;
            Mesh mesh_minus   = fc.extract_surface(grid.vertices, sdf, grid.cubes,
                                                   Resolution(resolution),
                                                   &beta, &alpha, &gamma, nullptr, opts);
            double loss_minus = compute_loss(mesh_minus, target_mesh);

            sdf[i] += eps;  // Reset
            sdf_grad[i] = (loss_plus - loss_minus) / (2 * eps);
        }

        // Apply gradients
        optimizer.step(sdf, sdf_grad);

        // Clamp SDF to reasonable range
        for (Index i = 0; i < sdf.size(); ++i)
        {
            sdf[i] = std::clamp(sdf[i], -1.0, 1.0);
        }
    }

    Mesh final_mesh = fc.extract_surface(grid.vertices, sdf, grid.cubes, Resolution(resolution), &beta, &alpha, &gamma, nullptr, opts);
    std::cout << "Final: " << final_mesh.num_vertices() << " vertices, " << final_mesh.num_faces() << " faces\n";

    double final_loss = compute_loss(final_mesh, target_mesh);
    std::cout << "Final loss: " << final_loss << "\n";

    return 0;
}
