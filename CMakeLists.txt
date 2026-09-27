cmake_minimum_required(VERSION 3.25)

project(plabundle
    VERSION 0.1.0
    DESCRIPTION "C++ bundle and control-network adjustment library powered by PlaMatrix"
    LANGUAGES CXX
)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(PLABUNDLE_BUILD_TESTS "Build PlaBundle unit tests" ${PROJECT_IS_TOP_LEVEL})
option(PLABUNDLE_BUILD_BENCHMARKS "Build PlaBundle benchmarks" OFF)
option(PLABUNDLE_BUILD_EXAMPLES "Build PlaBundle examples" OFF)
option(PLABUNDLE_ENABLE_OPENMP "Enable OpenMP in PlaBundle internal loops" ON)
set(PLABUNDLE_PLAMATRIX_SOURCE_DIR "" CACHE PATH
    "Optional PlaMatrix source tree used by standalone developer builds")

if(NOT TARGET plamatrix::plamatrix)
    if(PLABUNDLE_PLAMATRIX_SOURCE_DIR)
        if(NOT EXISTS "${PLABUNDLE_PLAMATRIX_SOURCE_DIR}/CMakeLists.txt")
            message(FATAL_ERROR
                "PLABUNDLE_PLAMATRIX_SOURCE_DIR does not contain a CMake project: "
                "${PLABUNDLE_PLAMATRIX_SOURCE_DIR}")
        endif()
        add_subdirectory(
            "${PLABUNDLE_PLAMATRIX_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}/plamatrix"
            EXCLUDE_FROM_ALL
        )
    else()
        find_package(plamatrix CONFIG REQUIRED)
    endif()
endif()

if(PLABUNDLE_ENABLE_OPENMP AND NOT TARGET OpenMP::OpenMP_CXX)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

add_library(plabundle STATIC
    src/adaptive_camera_model.cpp
    src/backend.cpp
    src/camera.cpp
    src/control_point.cpp
    src/linescan.cpp
    src/linescan_assembly.cpp
    src/linescan_model.cpp
    src/linescan_plamatrix.cpp
    src/options.cpp
    src/photo_alignment.cpp
    src/pose_prior.cpp
    src/problem.cpp
    src/rig.cpp
    src/solver.cpp
    src/internal/BundleAdjustAdaptiveCameraModel.cpp
    src/internal/BundleAdjustPlaMatrix.cpp
    src/internal/BundleAdjustPlaMatrixAssembly.cpp
    src/internal/BundleAdjustPlaMatrixAssemblySurvey.cpp
    src/internal/BundleAdjustPlaMatrixConstraints.cpp
    src/internal/BundleAdjustPlaMatrixModel.cpp
    src/internal/BundleAdjustPlaMatrixProblem.cpp
    src/internal/BundleAdjustPlaMatrixProjection.cpp
    src/internal/BundleAdjustPlaMatrixReferenceSchur.cpp
    src/internal/BundleAdjustPlaMatrixRuntime.cpp
    src/internal/BundleAdjustProjection.cpp
    src/internal/BundleAdjustQuality.cpp
    src/internal/BundleAdjustValidation.cpp
    src/internal/CameraState.cpp
    src/internal/Conversion.cpp
)
add_library(plabundle::plabundle ALIAS plabundle)

target_compile_features(plabundle PUBLIC cxx_std_20)
target_include_directories(plabundle
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(plabundle PRIVATE plamatrix::plamatrix)
if(PLABUNDLE_ENABLE_OPENMP)
    target_link_libraries(plabundle PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(plabundle PRIVATE PLABUNDLE_ENABLE_OPENMP=1)
endif()

set_target_properties(plabundle PROPERTIES
    CXX_EXTENSIONS OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

if(PLABUNDLE_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(test)
endif()

if(PLABUNDLE_BUILD_BENCHMARKS)
    add_executable(plabundle_benchmark benchmark/plabundle_benchmark.cpp)
    target_compile_features(plabundle_benchmark PRIVATE cxx_std_20)
    target_link_libraries(plabundle_benchmark PRIVATE plabundle::plabundle)
endif()

if(PLABUNDLE_BUILD_EXAMPLES)
    add_subdirectory(examples/consumer)
endif()

install(TARGETS plabundle
    EXPORT plabundleTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/plabundle DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT plabundleTargets
    NAMESPACE plabundle::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plabundle
)

configure_package_config_file(
    cmake/plabundleConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/plabundleConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plabundle
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/plabundleConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/plabundleConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/plabundleConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/plabundle
)
install(FILES LICENSE README.md CHANGELOG.md
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/plabundle
)
