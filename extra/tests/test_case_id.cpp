#include <doctest/doctest.h>
#include <flexicubes/case_id.hpp>
#include <flexicubes/tables.hpp>

using namespace flexi;

TEST_CASE("Ambiguity resolution in C16/C19 cases")
{
    Resolution res(2, 2, 2);
    const int rx        = res.x;
    const int ry        = res.y;
    const int rz        = res.z;
    const int num_cubes = rx * ry * rz;

    auto linear_idx = [rx, ry](int x, int y, int z)
    {
        return z * ry * rx + y * rx + x;
    };

    int case_id = -1;
    int off_x = 0, off_y = 0, off_z = 0;
    int base_x = -1, base_y = -1, base_z = -1;

    for (int c = 0; c < 256 && case_id < 0; ++c)
    {
        const auto & entry = tables::CHECK_TABLE[c];
        if (entry[0] != 1)
        {
            continue;
        }
        off_x = entry[1];
        off_y = entry[2];
        off_z = entry[3];

        for (int z = 0; z < rz; ++z)
        {
            for (int y = 0; y < ry; ++y)
            {
                for (int x = 0; x < rx; ++x)
                {
                    int nx = x + off_x;
                    int ny = y + off_y;
                    int nz = z + off_z;
                    if (nx >= 0 && nx < rx &&
                        ny >= 0 && ny < ry &&
                        nz >= 0 && nz < rz)
                    {
                        case_id = c;
                        base_x  = x;
                        base_y  = y;
                        base_z  = z;
                        break;
                    }
                }
                if (case_id >= 0) break;
            }
            if (case_id >= 0) break;
        }
    }

    REQUIRE(case_id >= 0);

    SurfaceCubes surf;
    surf.mask.resize(num_cubes);
    surf.mask.setConstant(false);
    surf.occupancy.resize(num_cubes, 8);
    surf.occupancy.setZero();

    int cube_a = linear_idx(base_x, base_y, base_z);
    int cube_b = linear_idx(base_x + off_x, base_y + off_y, base_z + off_z);

    surf.mask[cube_a] = true;
    surf.mask[cube_b] = true;

    // Populate occupancy for both cubes to match case_id
    for (int c = 0; c < 8; ++c)
    {
        int bit                   = (case_id >> c) & 1;
        surf.occupancy(cube_a, c) = bit;
        surf.occupancy(cube_b, c) = bit;
    }

    VecXi case_ids = compute_case_ids_with_resolution(surf, res);

    // Surface cube order follows mask scan order
    int idx_a = (cube_a < cube_b) ? 0 : 1;
    int idx_b = (cube_a < cube_b) ? 1 : 0;

    int inverted = tables::CHECK_TABLE[case_id][4];
    CHECK(case_ids[idx_a] == inverted);
    CHECK(case_ids[idx_b] == case_id);
}
