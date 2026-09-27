#include <plabundle/linescan.h>

#include <type_traits>
#include <utility>

static_assert(sizeof(plabundle::linescan::ImageObservation) > 0);
static_assert(std::is_same_v<decltype(plabundle::linescan::solve(std::declval<const plabundle::linescan::Problem&>(),
                                                                 std::declval<const plabundle::linescan::Options&>())),
                             plabundle::linescan::Result>);
