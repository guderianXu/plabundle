file(GLOB _plabundle_public_headers
    "${PLABUNDLE_PUBLIC_INCLUDE_DIR}/*.h"
    "${PLABUNDLE_PUBLIC_INCLUDE_DIR}/*.hpp"
)
if(NOT _plabundle_public_headers)
    message(FATAL_ERROR "No PlaBundle public headers found in ${PLABUNDLE_PUBLIC_INCLUDE_DIR}")
endif()

foreach(_header IN LISTS _plabundle_public_headers)
    file(READ "${_header}" _contents)
    if(_contents MATCHES "#[ \t]*include[ \t]*[<\"]Q")
        message(FATAL_ERROR "Qt include leaked into public header: ${_header}")
    endif()
    foreach(_forbidden
            "xjw::" "PlaScan" "plascan" "PlaPoint" "plapoint" "Qt"
            "/home/" "/Users/" "/workspace/")
        string(FIND "${_contents}" "${_forbidden}" _position)
        if(NOT _position EQUAL -1)
            message(FATAL_ERROR "Forbidden token '${_forbidden}' in public header: ${_header}")
        endif()
    endforeach()
    if(_contents MATCHES "[A-Za-z]:\\\\")
        message(FATAL_ERROR "Absolute Windows path leaked into public header: ${_header}")
    endif()
endforeach()
