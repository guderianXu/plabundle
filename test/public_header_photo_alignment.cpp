#include <plabundle/photo_alignment.h>

int plabundlePublicHeaderPhotoAlignment()
{
    return plabundle::PhotoAlignmentComparison{}.comparable ? 1 : 0;
}
