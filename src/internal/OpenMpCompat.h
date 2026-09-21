#pragma once

#ifdef PLABUNDLE_ENABLE_OPENMP
#include <omp.h>
#endif

namespace plabundle::internal
{

    inline int openMpMaxThreads() noexcept
    {
#ifdef PLABUNDLE_ENABLE_OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }

} // namespace plabundle::internal
