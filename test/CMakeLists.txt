include(GoogleTest)

add_executable(test_plabundle
    test_adaptive_camera_model.cpp
    test_backend_selection.cpp
    test_constraint_jacobians.cpp
    test_camera.cpp
    test_problem.cpp
    test_solver.cpp
    test_solver_internals.cpp
    public_header_adaptive_camera_model.cpp
    public_header_backend.cpp
    public_header_camera.cpp
    public_header_constraints.cpp
    public_header_options.cpp
    public_header_problem.cpp
    public_header_result.cpp
    public_header_solver.cpp
    public_header_version.cpp
)
target_link_libraries(test_plabundle PRIVATE
    plabundle::plabundle
    GTest::gtest_main
)
target_include_directories(test_plabundle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
gtest_discover_tests(test_plabundle TEST_PREFIX "PlaBundle.")

add_test(
    NAME PlaBundle.PublicDependencyBoundary
    COMMAND ${CMAKE_COMMAND}
        -DPLABUNDLE_PUBLIC_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../include/plabundle
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_public_dependencies.cmake
)
