cmake_minimum_required(VERSION 3.16)
project(flexicubes-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(FLEXICUBES_BUILD_EXAMPLES "Build example applications" ON)
option(FLEXICUBES_BUILD_TESTS "Build unit tests" ON)

# Third-party directory
set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# =============================================================================
# Eigen3 (from third_party/eigen)
# =============================================================================
set(EIGEN_DIR "${THIRD_PARTY_DIR}/eigen")
if(NOT EXISTS "${EIGEN_DIR}/Eigen/Core")
    message(FATAL_ERROR "Eigen not found in ${EIGEN_DIR}. Please clone Eigen to third_party/eigen")
endif()

add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE "${EIGEN_DIR}")
add_library(Eigen3::Eigen ALIAS eigen)
message(STATUS "Using Eigen from: ${EIGEN_DIR}")

# =============================================================================
# Adept-2 (from third_party/adept)
# =============================================================================
set(ADEPT_DIR "${THIRD_PARTY_DIR}/adept")
if(NOT EXISTS "${ADEPT_DIR}/include/adept.h")
    message(FATAL_ERROR "Adept-2 not found in ${ADEPT_DIR}. Please clone Adept-2 to third_party/adept")
endif()

file(GLOB ADEPT_SOURCES "${ADEPT_DIR}/adept/*.cpp")
if(NOT ADEPT_SOURCES)
    message(FATAL_ERROR "Adept-2 source files not found in ${ADEPT_DIR}/adept/")
endif()

# Build Adept as a static library
add_library(adept STATIC ${ADEPT_SOURCES})
target_include_directories(adept PUBLIC "${ADEPT_DIR}/include")

# Adept compile definitions
target_compile_definitions(adept PRIVATE ADEPT_NO_AUTOMATIC_DIFFERENTIATION=0)

# Suppress warnings in third-party code
if(MSVC)
    target_compile_options(adept PRIVATE /W0)
else()
    target_compile_options(adept PRIVATE -w)
endif()

message(STATUS "Using Adept-2 from: ${ADEPT_DIR}")

# =============================================================================
# TriangleMeshDistance (header-only)
# =============================================================================
set(TMD_DIR "${THIRD_PARTY_DIR}/TriangleMeshDistance")
if(EXISTS "${TMD_DIR}/TriangleMeshDistance/include/tmd/TriangleMeshDistance.h")
    add_library(tmd INTERFACE)
    target_include_directories(tmd INTERFACE "${TMD_DIR}/TriangleMeshDistance/include")
    message(STATUS "Using TriangleMeshDistance from: ${TMD_DIR}")
else()
    message(STATUS "TriangleMeshDistance not found - mesh_to_sdf_test will not be built")
endif()

# =============================================================================
# FlexiCubes Library 
# =============================================================================
file(GLOB_RECURSE FLEXICUBES_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/flexicubes/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/flexicubes/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/flexicubes/*.inl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/flexicubes/*.ipp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/flexicubes/*.tpp"
)

add_library(flexicubes_headers INTERFACE)
add_library(flexicubes::headers ALIAS flexicubes_headers)
target_sources(flexicubes_headers INTERFACE ${FLEXICUBES_HEADERS})
target_include_directories(flexicubes_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Visual Studio doesn't  show INTERFACE targets; add a header-only utility target.
add_custom_target(flexicubes_header_only SOURCES ${FLEXICUBES_HEADERS})

add_library(flexicubes INTERFACE)
add_library(flexicubes::flexicubes ALIAS flexicubes)

target_include_directories(flexicubes INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(flexicubes INTERFACE ${FLEXICUBES_HEADERS})

# Link dependencies
target_link_libraries(flexicubes INTERFACE eigen adept)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(flexicubes INTERFACE
        $<BUILD_INTERFACE:-Wall -Wextra -Wpedantic>
    )
elseif(MSVC)
    target_compile_options(flexicubes INTERFACE
        $<BUILD_INTERFACE:/W4>
    )
endif()

if(FLEXICUBES_BUILD_EXAMPLES)
    add_subdirectory(extra/samples)
endif()

if(FLEXICUBES_BUILD_TESTS)
    enable_testing()
    add_subdirectory(extra/tests)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "FlexiCubes C++ Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen: ${EIGEN_DIR}")
message(STATUS "  Adept-2: ${ADEPT_DIR}")
if(TARGET tmd)
    message(STATUS "  TriangleMeshDistance: ${TMD_DIR}")
endif()
message(STATUS "  Build examples: ${FLEXICUBES_BUILD_EXAMPLES}")
message(STATUS "  Build tests: ${FLEXICUBES_BUILD_TESTS}")
message(STATUS "")
