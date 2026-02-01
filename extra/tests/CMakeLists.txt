# FlexiCubes Tests

# Fetch doctest
include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

set(EXTRA_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# Test executable
add_executable(flexicubes_tests
    test_case_id.cpp
    test_main.cpp
    test_tables.cpp
    test_grid.cpp
    test_surface.cpp
    test_interpolation.cpp
    test_extraction.cpp
    test_differentiation.cpp
    test_optimization_pipeline.cpp
)

target_link_libraries(flexicubes_tests PRIVATE
    flexicubes::flexicubes
    doctest::doctest
)
target_include_directories(flexicubes_tests PRIVATE "${EXTRA_ROOT_DIR}")

# Register tests with CTest
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(flexicubes_tests)
