# FlexiCubes Examples

set(EXTRA_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(basic_extraction basic_extraction.cpp)
target_link_libraries(basic_extraction PRIVATE flexicubes::flexicubes)

add_executable(sphere_sdf sphere_sdf.cpp)
target_link_libraries(sphere_sdf PRIVATE flexicubes::flexicubes)
target_include_directories(sphere_sdf PRIVATE "${EXTRA_ROOT_DIR}")

add_executable(optimization_loop optimization_loop.cpp)
target_link_libraries(optimization_loop PRIVATE flexicubes::flexicubes)

# Mesh-to-SDF test runner (requires TriangleMeshDistance)
if(TARGET tmd)
    add_executable(mesh_to_sdf_test mesh_to_sdf_test.cpp)
    target_link_libraries(mesh_to_sdf_test PRIVATE flexicubes::flexicubes tmd)
    target_include_directories(mesh_to_sdf_test PRIVATE "${EXTRA_ROOT_DIR}")
    # Suppress warnings in TriangleMeshDistance header
    if(MSVC)
        target_compile_options(mesh_to_sdf_test PRIVATE /wd4244 /wd4267)
    endif()
endif()
