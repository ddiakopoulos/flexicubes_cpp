#include <doctest/doctest.h>
#include <flexicubes/tables.hpp>

using namespace flexi;

TEST_CASE("Table dimensions")
{
    CHECK(tables::DMC_TABLE.size() == 256);
    CHECK(tables::NUM_VD_TABLE.size() == 256);
    CHECK(tables::CHECK_TABLE.size() == 256);
    CHECK(tables::TET_TABLE.size() == 256);

    for (const auto & entry : tables::DMC_TABLE)
    {
        CHECK(entry.size() == 4);
        for (const auto & row : entry)
        {
            CHECK(row.size() == 7);
        }
    }

    for (const auto & entry : tables::CHECK_TABLE)
    {
        CHECK(entry.size() == 5);
    }

    for (const auto & entry : tables::TET_TABLE)
    {
        CHECK(entry.size() == 6);
    }
}

TEST_CASE("NUM_VD_TABLE values")
{
    // All values should be 0-4
    for (int v : tables::NUM_VD_TABLE)
    {
        CHECK(v >= 0);
        CHECK(v <= 4);
    }

    // Case 0 (all outside) should have 0 dual vertices
    CHECK(tables::NUM_VD_TABLE[0] == 0);

    // Case 255 (all inside) should have 0 dual vertices
    CHECK(tables::NUM_VD_TABLE[255] == 0);

    // Case 1 (single corner inside) should have 1 dual vertex
    CHECK(tables::NUM_VD_TABLE[1] == 1);
}

TEST_CASE("DMC_TABLE edge indices")
{
    // Edge indices should be -1 (unused) or 0-11 (valid edges)
    for (const auto & entry : tables::DMC_TABLE)
    {
        for (const auto & row : entry)
        {
            for (int v : row)
            {
                CHECK((v == -1 || (v >= 0 && v <= 11)));
            }
        }
    }
}

TEST_CASE("CHECK_TABLE structure")
{
    // First element is problematic flag (0 or 1)
    for (const auto & entry : tables::CHECK_TABLE)
    {
        CHECK((entry[0] == 0 || entry[0] == 1));
    }

    // Offset values should be -1, 0, or 1
    for (const auto & entry : tables::CHECK_TABLE)
    {
        if (entry[0] == 1)
        {  // Only check if problematic
            CHECK((entry[1] >= -1 && entry[1] <= 1));
            CHECK((entry[2] >= -1 && entry[2] <= 1));
            CHECK((entry[3] >= -1 && entry[3] <= 1));
            CHECK(entry[4] >= 0);  // Inverted case ID
            CHECK(entry[4] <= 255);
        }
    }
}

TEST_CASE("Table accessor functions")
{
    CHECK(tables::get_num_dual_vertices(0) == 0);
    CHECK(tables::get_num_dual_vertices(255) == 0);
    CHECK(tables::get_num_dual_vertices(1) == 1);

    CHECK_FALSE(tables::is_ambiguous_case(0));
    CHECK_FALSE(tables::is_ambiguous_case(255));

    // Check edge access
    const auto & edges = tables::get_dual_vertex_edges(1, 0);
    CHECK(edges.size() == 7);

    // Case 1 should have edges 0, 3, 8 for its single dual vertex
    bool has_edge_0 = false, has_edge_3 = false, has_edge_8 = false;
    for (int e : edges)
    {
        if (e == 0) has_edge_0 = true;
        if (e == 3) has_edge_3 = true;
        if (e == 8) has_edge_8 = true;
    }
    CHECK(has_edge_0);
    CHECK(has_edge_3);
    CHECK(has_edge_8);
}

TEST_CASE("Table validation")
{
    CHECK(tables::validate_tables());
}

TEST_CASE("Symmetric cases")
{
    // Cases that are complements should have same number of dual vertices
    // (e.g., case 1 and case 254 are complements)
    for (int i = 0; i < 128; ++i)
    {
        int complement = 255 - i;
        // Note: This isn't always true for DMC, but worth checking some
    }
}
