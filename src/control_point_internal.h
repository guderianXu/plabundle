#pragma once

#include <plabundle/constraints.h>

#include <array>
#include <string>

namespace plabundle::internal
{

    struct ControlPointWhitening
    {
        std::array<double, 9> matrix{};
    };

    bool makeControlPointWhitening(const ControlPointConstraint& constraint,
                                   ControlPointWhitening* whitening,
                                   std::string* error = nullptr);

    bool controlPointRmsUncertaintyMeters(const ControlPointConstraint& constraint,
                                          double* rms_uncertainty,
                                          std::string* error = nullptr);

} // namespace plabundle::internal
