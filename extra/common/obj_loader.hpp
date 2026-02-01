#pragma once

#include "flexicubes/types.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <limits>

namespace flexi
{

    struct ObjMesh
    {
        MatX3 vertices;  // Nx3 vertex positions
        MatX3i faces;    // Mx3 triangle indices (0-indexed)
        Vec3 min_bound;  // Axis-aligned bounding box minimum
        Vec3 max_bound;  // Axis-aligned bounding box maximum

        void compute_bounds()
        {
            if (vertices.rows() == 0)
            {
                min_bound = Vec3::Zero();
                max_bound = Vec3::Zero();
                return;
            }
            min_bound = vertices.colwise().minCoeff();
            max_bound = vertices.colwise().maxCoeff();
        }

        Vec3 center() const
        {
            return (min_bound + max_bound) * 0.5;
        }

        Vec3 size() const
        {
            return max_bound - min_bound;
        }

        double normalize_to_unit_cube()
        {
            compute_bounds();

            // Compute center and scale
            Vec3 c            = center();
            Vec3 s            = size();
            double max_extent = s.maxCoeff();

            if (max_extent < 1e-10)
            {
                return 1.0;  // Degenerate mesh
            }

            // Scale to fit in [-0.45, 0.45] to leave some margin
            double scale = 0.9 / max_extent;

            // Transform vertices
            for (Index i = 0; i < vertices.rows(); ++i)
            {
                vertices.row(i) = (vertices.row(i).transpose() - c) * scale;
            }

            // Update bounds
            min_bound = (min_bound - c) * scale;
            max_bound = (max_bound - c) * scale;

            return scale;
        }

        bool is_valid() const
        {
            return vertices.rows() > 0 && faces.rows() > 0;
        }

        Index num_vertices() const { return vertices.rows(); }
        Index num_faces() const { return faces.rows(); }
    };

    /**
    * Supports basic OBJ format:
    * - v x y z (vertex positions)
    * - f v1 v2 v3 (triangle faces, 1-indexed)
    * - f v1/vt1 v2/vt2 v3/vt3 (faces with texture coords, ignored)
    * - f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 (faces with normals, ignored)
    * - f v1//vn1 v2//vn2 v3//vn3 (faces with normals only, ignored)
    */
    inline ObjMesh load_obj(const std::string & path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open OBJ file: " + path);
        }

        std::vector<Vec3> temp_vertices;
        std::vector<Vec3i> temp_faces;

        std::string line;
        int line_number = 0;

        while (std::getline(file, line))
        {
            ++line_number;

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#')
            {
                continue;
            }

            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "v")
            {
                // Vertex position
                double x, y, z;
                if (!(iss >> x >> y >> z))
                {
                    throw std::runtime_error("Invalid vertex at line " + std::to_string(line_number));
                }
                temp_vertices.emplace_back(x, y, z);
            }
            else if (prefix == "f")
            {
                // Face - parse vertex indices (may include texture/normal indices)
                std::vector<int> indices;
                std::string token;
                while (iss >> token)
                {
                    // Parse vertex index (before any '/')
                    int v_idx;
                    size_t slash_pos = token.find('/');
                    if (slash_pos != std::string::npos)
                    {
                        v_idx = std::stoi(token.substr(0, slash_pos));
                    }
                    else
                    {
                        v_idx = std::stoi(token);
                    }

                    // OBJ indices are 1-based; negative indices are relative
                    if (v_idx < 0)
                    {
                        v_idx = static_cast<int>(temp_vertices.size()) + v_idx + 1;
                    }

                    // Convert to 0-based
                    indices.push_back(v_idx - 1);
                }

                // Triangulate if needed (simple fan triangulation)
                if (indices.size() >= 3)
                {
                    for (size_t i = 1; i + 1 < indices.size(); ++i)
                    {
                        temp_faces.emplace_back(indices[0], indices[i], indices[i + 1]);
                    }
                }
            }
            // Ignore other lines (vt, vn, mtllib, usemtl, o, g, s, etc.)
        }

        if (temp_vertices.empty())
        {
            throw std::runtime_error("No vertices found in OBJ file: " + path);
        }

        if (temp_faces.empty())
        {
            throw std::runtime_error("No faces found in OBJ file: " + path);
        }

        // Copy to Eigen matrices
        ObjMesh mesh;
        mesh.vertices.resize(temp_vertices.size(), 3);
        mesh.faces.resize(temp_faces.size(), 3);

        for (size_t i = 0; i < temp_vertices.size(); ++i)
        {
            mesh.vertices.row(i) = temp_vertices[i].transpose();
        }

        for (size_t i = 0; i < temp_faces.size(); ++i)
        {
            mesh.faces.row(i) = temp_faces[i].transpose();
        }

        mesh.compute_bounds();

        return mesh;
    }

    template <typename MeshType>
    void save_obj(const std::string & path, const MeshType & mesh)
    {
        std::ofstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to create OBJ file: " + path);
        }

        file << "# FlexiCubes C++ output\n";
        file << "# Vertices: " << mesh.vertices.rows() << "\n";
        file << "# Faces: " << mesh.faces.rows() << "\n\n";

        // Write vertices
        file << std::fixed;
        file.precision(6);
        for (Index i = 0; i < mesh.vertices.rows(); ++i)
        {
            file << "v " << mesh.vertices(i, 0) << " "
                 << mesh.vertices(i, 1) << " "
                 << mesh.vertices(i, 2) << "\n";
        }

        file << "\n";

        // Write faces (1-indexed)
        for (Index i = 0; i < mesh.faces.rows(); ++i)
        {
            file << "f " << (mesh.faces(i, 0) + 1) << " "
                 << (mesh.faces(i, 1) + 1) << " "
                 << (mesh.faces(i, 2) + 1) << "\n";
        }

        if (!file.good())
        {
            throw std::runtime_error("Error writing to OBJ file: " + path);
        }
    }

    inline void save_obj(const std::string & path, const ObjMesh & mesh)
    {
        std::ofstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to create OBJ file: " + path);
        }

        file << "# FlexiCubes C++ output\n";
        file << "# Vertices: " << mesh.vertices.rows() << "\n";
        file << "# Faces: " << mesh.faces.rows() << "\n\n";

        // Write vertices
        file << std::fixed;
        file.precision(6);
        for (Index i = 0; i < mesh.vertices.rows(); ++i)
        {
            file << "v " << mesh.vertices(i, 0) << " "
                 << mesh.vertices(i, 1) << " "
                 << mesh.vertices(i, 2) << "\n";
        }

        file << "\n";

        // Write faces (1-indexed)
        for (Index i = 0; i < mesh.faces.rows(); ++i)
        {
            file << "f " << (mesh.faces(i, 0) + 1) << " "
                 << (mesh.faces(i, 1) + 1) << " "
                 << (mesh.faces(i, 2) + 1) << "\n";
        }

        if (!file.good())
        {
            throw std::runtime_error("Error writing to OBJ file: " + path);
        }
    }

}  // namespace flexi
