#pragma once

#include <plabundle/linescan.h>

namespace plabundle::linescan::detail
{

    bool evaluateImageObservation(const Problem& problem,
                                  const ImageObservation& observation,
                                  const std::array<double, 6>& camera,
                                  const std::array<double, 3>& point,
                                  double imageSigmaPixels,
                                  double* residuals);

    double imageRms(const Problem& problem);
    double laserRangeRms(const Problem& problem);

    bool solvePlaMatrix(Problem* problem, const Options& options, Result* result, std::string* errorMessage);

} // namespace plabundle::linescan::detail
