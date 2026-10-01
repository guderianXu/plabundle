include(GoogleTest)

add_executable(test_plabundle
    test_covariance_whitening.cpp
    test_native_problem.cpp
    test_linescan.cpp
    test_options.cpp
    public_header_adaptive_camera_model.cpp
    public_header_backend.cpp
    public_header_linescan.cpp
    public_header_constraints.cpp
    public_header_options.cpp
    public_header_photo_alignment.cpp
    public_header_problem.cpp
    public_header_result.cpp
    public_header_solver.cpp
    public_header_version.cpp
)
target_link_libraries(test_plabundle PRIVATE
    plabundle::plabundle
    placamera::placamera
    plamatrix::plamatrix
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

add_test(
    NAME PlaBundle.CameraBoundary
    COMMAND ${CMAKE_COMMAND}
        -DPLABUNDLE_PUBLIC_INCLUDE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../include/plabundle
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_camera_boundary.cmake
)

add_test(
    NAME PlaBundle.PlaMatrixBoundary
    COMMAND ${CMAKE_COMMAND}
        -DPLABUNDLE_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/..
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_plamatrix_boundary.cmake
)
