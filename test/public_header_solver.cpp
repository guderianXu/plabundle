#include <plabundle/solver.h>

int plabundlePublicHeaderSolver()
{
    return plabundle::Solver::isBackendAvailable(plabundle::Backend::PlaMatrixCpu) ? 1 : 0;
}
