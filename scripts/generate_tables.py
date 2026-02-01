#!/usr/bin/env python3
"""
Generate C++ constexpr lookup tables from FlexiCubes Python tables.py

This script reads the tables.py file and generates tables_data.inl with
constexpr arrays suitable for C++ compilation.

Usage:
    python generate_tables.py <path_to_tables.py> <output_directory>
"""

import sys
import os
import re

def parse_tables_py(filepath):
    """Parse the Python tables.py file and extract table data."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract dmc_table - it's a list of 256 entries, each with 4 sub-arrays of 7 values
    dmc_match = re.search(r'dmc_table\s*=\s*\[(.*?)\]\s*\nnum_vd_table', content, re.DOTALL)
    if not dmc_match:
        raise ValueError("Could not find dmc_table in tables.py")

    dmc_str = dmc_match.group(1)
    # Parse nested arrays
    dmc_table = []
    # Find all 4x7 blocks
    entry_pattern = re.compile(r'\[\s*\[([-\d,\s]+)\],\s*\[([-\d,\s]+)\],\s*\[([-\d,\s]+)\],\s*\[([-\d,\s]+)\]\s*\]')
    for m in entry_pattern.finditer(dmc_str):
        entry = []
        for i in range(1, 5):
            vals = [int(x.strip()) for x in m.group(i).split(',')]
            entry.append(vals)
        dmc_table.append(entry)

    # Extract num_vd_table - flat list of 256 integers
    num_vd_match = re.search(r'num_vd_table\s*=\s*\[([\d,\s]+)\]', content, re.DOTALL)
    if not num_vd_match:
        raise ValueError("Could not find num_vd_table in tables.py")
    num_vd_table = [int(x.strip()) for x in num_vd_match.group(1).split(',')]

    # Extract check_table - 256 entries of 5 values each
    check_match = re.search(r'check_table\s*=\s*\[(.*?)\]\s*\ntet_table', content, re.DOTALL)
    if not check_match:
        raise ValueError("Could not find check_table in tables.py")

    check_str = check_match.group(1)
    check_table = []
    entry_pattern = re.compile(r'\[([-\d,\s]+)\]')
    for m in entry_pattern.finditer(check_str):
        vals = [int(x.strip()) for x in m.group(1).split(',')]
        check_table.append(vals)

    # Extract tet_table - 256 entries of 6 values each
    tet_match = re.search(r'tet_table\s*=\s*\[(.*?)\]\s*$', content, re.DOTALL)
    if not tet_match:
        raise ValueError("Could not find tet_table in tables.py")

    tet_str = tet_match.group(1)
    tet_table = []
    for m in entry_pattern.finditer(tet_str):
        vals = [int(x.strip()) for x in m.group(1).split(',')]
        tet_table.append(vals)

    return {
        'dmc_table': dmc_table,
        'num_vd_table': num_vd_table,
        'check_table': check_table,
        'tet_table': tet_table
    }


def generate_cpp_header(tables, output_path):
    """Generate C++ header file with constexpr tables."""

    with open(output_path, 'w') as f:
        f.write("""// Auto-generated from tables.py - DO NOT EDIT
// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2025 - FlexiCubes C++ Implementation
// Licensed under the Apache License, Version 2.0

#pragma once

#include <array>
#include <cstdint>

namespace flexicubes {
namespace tables {

""")

        # DMC table: 256 x 4 x 7
        f.write("// Dual Marching Cubes table: edges associated with each dual vertex\n")
        f.write("// Shape: [256 cases][4 max dual vertices][7 max edges per vertex]\n")
        f.write("// -1 indicates unused slot\n")
        f.write("constexpr std::array<std::array<std::array<int, 7>, 4>, 256> DMC_TABLE = {{\n")
        for i, entry in enumerate(tables['dmc_table']):
            f.write("    {{ // Case %d\n" % i)
            for j, row in enumerate(entry):
                row_str = ", ".join("%2d" % v for v in row)
                f.write("        {%s}%s\n" % (row_str, "," if j < 3 else ""))
            f.write("    }}%s\n" % ("," if i < 255 else ""))
        f.write("}};\n\n")

        # Num VD table: 256 values
        f.write("// Number of dual vertices per MC case\n")
        f.write("constexpr std::array<int, 256> NUM_VD_TABLE = {\n    ")
        for i, v in enumerate(tables['num_vd_table']):
            f.write("%d" % v)
            if i < 255:
                f.write(",")
            if (i + 1) % 16 == 0 and i < 255:
                f.write("\n    ")
            elif i < 255:
                f.write(" ")
        f.write("\n};\n\n")

        # Check table: 256 x 5
        f.write("// Ambiguity resolution table for C16/C19 cases\n")
        f.write("// [0]: is_problematic, [1-3]: neighbor offset, [4]: inverted case id\n")
        f.write("constexpr std::array<std::array<int, 5>, 256> CHECK_TABLE = {{\n")
        for i, entry in enumerate(tables['check_table']):
            row_str = ", ".join("%3d" % v for v in entry)
            f.write("    {%s}%s" % (row_str, "," if i < 255 else ""))
            if (i + 1) % 4 == 0:
                f.write(" // %d-%d\n" % (i-3, i))
            else:
                f.write(" ")
        f.write("}};\n\n")

        # Tet table: 256 x 6
        f.write("// Tetrahedralization lookup table\n")
        f.write("constexpr std::array<std::array<int, 6>, 256> TET_TABLE = {{\n")
        for i, entry in enumerate(tables['tet_table']):
            row_str = ", ".join("%2d" % v for v in entry)
            f.write("    {%s}%s" % (row_str, "," if i < 255 else ""))
            if (i + 1) % 4 == 0:
                f.write(" // %d-%d\n" % (i-3, i))
            else:
                f.write(" ")
        f.write("}};\n\n")

        f.write("} // namespace tables\n")
        f.write("} // namespace flexicubes\n")

    print(f"Generated: {output_path}")
    print(f"  DMC table: {len(tables['dmc_table'])} entries (256 x 4 x 7)")
    print(f"  NUM_VD table: {len(tables['num_vd_table'])} entries")
    print(f"  CHECK table: {len(tables['check_table'])} entries (256 x 5)")
    print(f"  TET table: {len(tables['tet_table'])} entries (256 x 6)")


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_tables.py <tables.py> <output_dir>")
        print("Example: python generate_tables.py ../../tables.py ../include/flexicubes/")
        sys.exit(1)

    tables_path = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(tables_path):
        print(f"Error: tables.py not found at {tables_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing {tables_path}...")
    tables = parse_tables_py(tables_path)

    output_path = os.path.join(output_dir, "tables_data.inl")
    generate_cpp_header(tables, output_path)


if __name__ == '__main__':
    main()
