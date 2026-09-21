#include <plabundle/result.h>

int plabundlePublicHeaderResult()
{
    return plabundle::Result{}.usable() ? 1 : 0;
}
