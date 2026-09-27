#include <plabundle/options.h>

int plabundlePublicHeaderOptions()
{
    return plabundle::SolveOptions{}.solver.maxIterations + plabundle::Options{}.maxIterations +
           static_cast<int>(plabundle::ImageRobustLoss::Cauchy);
}
