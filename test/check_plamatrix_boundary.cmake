if(NOT IS_DIRECTORY "${PLABUNDLE_SOURCE_ROOT}")
    message(FATAL_ERROR "PLABUNDLE_SOURCE_ROOT must name the PlaBundle source tree")
endif()

# These are generic numeric primitives, not camera or bundle-adjustment models.
# Adding a new PlaMatrix include requires an explicit boundary review.
set(_allowed_plamatrix_headers
    dense/matrix.h
    dense/small_inverse.h
    internal/core/execution_policy.h
    internal/opencl/runtime.h
    internal/ops/statistics.h
    internal/optimization/block_schur.h
    internal/optimization/levenberg_marquardt.h
    internal/optimization/robust_loss.h
    internal/vulkan/capabilities.h
)

set(_legacy_type_pattern
    "plamatrix::(DenseMatrix|CSRMatrix|Vec[234]|Mat[234]|DeviceMatrix|DeviceVector)([^A-Za-z0-9_]|$)")
if(NOT "plamatrix::DenseMatrix<double>" MATCHES "${_legacy_type_pattern}"
        OR NOT "plamatrix::CSRMatrix<double>" MATCHES "${_legacy_type_pattern}"
        OR NOT "plamatrix::Vec2<double>" MATCHES "${_legacy_type_pattern}"
        OR NOT "plamatrix::Vec3<double>" MATCHES "${_legacy_type_pattern}"
        OR NOT "plamatrix::Vec4<double>" MATCHES "${_legacy_type_pattern}"
        OR NOT "plamatrix::Mat2<double>" MATCHES "${_legacy_type_pattern}"
        OR NOT "plamatrix::Mat3<double>" MATCHES "${_legacy_type_pattern}"
        OR NOT "plamatrix::Mat4<double>" MATCHES "${_legacy_type_pattern}"
        OR NOT "plamatrix::DeviceMatrix<double>" MATCHES "${_legacy_type_pattern}"
        OR NOT "plamatrix::DeviceVector<double>" MATCHES "${_legacy_type_pattern}"
        OR "plamatrix::SparseMatrix<double>" MATCHES "${_legacy_type_pattern}"
        OR "plamatrix::Matrix<double>" MATCHES "${_legacy_type_pattern}"
        OR "plamatrix::Vector3d" MATCHES "${_legacy_type_pattern}"
        OR "plamatrix::ResidentMatrix<double>" MATCHES "${_legacy_type_pattern}")
    message(FATAL_ERROR "PlaMatrix legacy-type boundary pattern is invalid")
endif()

set(_using_namespace_pattern
    "(^|\n)[ \t]*using[ \t]+namespace[ \t]+plamatrix([ \t]*;|::)")
if(NOT "\nusing namespace plamatrix;" MATCHES "${_using_namespace_pattern}"
        OR NOT "\nusing namespace plamatrix::v1;" MATCHES "${_using_namespace_pattern}"
        OR "\nusing namespace plabundle;" MATCHES "${_using_namespace_pattern}")
    message(FATAL_ERROR "PlaMatrix using-namespace boundary pattern is invalid")
endif()

set(_sources)
foreach(_directory include src test benchmark examples)
    if(IS_DIRECTORY "${PLABUNDLE_SOURCE_ROOT}/${_directory}")
        file(GLOB_RECURSE _directory_sources LIST_DIRECTORIES FALSE
            "${PLABUNDLE_SOURCE_ROOT}/${_directory}/*.h"
            "${PLABUNDLE_SOURCE_ROOT}/${_directory}/*.hpp"
            "${PLABUNDLE_SOURCE_ROOT}/${_directory}/*.cpp"
            "${PLABUNDLE_SOURCE_ROOT}/${_directory}/*.cu"
        )
        list(APPEND _sources ${_directory_sources})
    endif()
endforeach()

if(NOT _sources)
    message(FATAL_ERROR "No PlaBundle C++ sources found in ${PLABUNDLE_SOURCE_ROOT}")
endif()

foreach(_source IN LISTS _sources)
    file(RELATIVE_PATH _relative_source "${PLABUNDLE_SOURCE_ROOT}" "${_source}")
    file(STRINGS "${_source}" _includes
        REGEX "^[ \t]*#[ \t]*include[ \t]*[<\"]plamatrix/"
    )
    if(_relative_source MATCHES "^include/" AND _includes)
        message(FATAL_ERROR "PlaMatrix include leaked into public header: ${_source}")
    endif()
    foreach(_include IN LISTS _includes)
        string(REGEX REPLACE "^.*plamatrix/([^>\"]+)[>\"].*$" "\\1" _header "${_include}")
        list(FIND _allowed_plamatrix_headers "${_header}" _allowed_index)
        if(_allowed_index EQUAL -1)
            message(FATAL_ERROR "Unreviewed PlaMatrix include '${_header}' in ${_source}")
        endif()
    endforeach()

    file(READ "${_source}" _contents)
    if(_contents MATCHES "${_legacy_type_pattern}")
        message(FATAL_ERROR "Legacy PlaMatrix type used in ${_source}")
    endif()
    if(_contents MATCHES "${_using_namespace_pattern}")
        message(FATAL_ERROR "PlaMatrix using-namespace directive used in ${_source}")
    endif()
endforeach()
