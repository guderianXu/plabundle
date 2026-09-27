#include <plabundle/constraints.h>

int plabundlePublicHeaderConstraints()
{
    return plabundle::Observation{}.cameraIndex + static_cast<int>(plabundle::ControlPointUncertainty::SqrtInformation);
}
