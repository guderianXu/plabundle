#include <plabundle/rig.h>

int plabundlePublicHeaderRig()
{
    return plabundle::RigTopology{}.empty() ? 0 : 1;
}
