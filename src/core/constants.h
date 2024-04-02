#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string> // For std::string

/// General constants
#define PI              3.141592653589793

/// LBM specific constants
#define C_S             (0.5773502691896258)
#define C_S_POW2        (0.3333333333333333)
#define C_S_POW4        (0.11111111111111117)
#define C_S_POW6        (0.03703703703703707)
#define C_S_POW8        (0.012345679012345692)
#define C_S_POW2_INV    (3.0)
#define C_S_POW4_INV    (9.0)
#define C_S_POW6_INV    (27.0)
#define C_S_POW8_INV    (81.0)

/// Allowed boundary locations
enum class BoundaryLocation {
    WEST,
    EAST,
    SOUTH,
    NORTH,
    BOTTOM,
    TOP
};

/// Transform BoundaryLocation members to strings for output
std::string boundaryLocationToString(BoundaryLocation location);

//#include "constants.hh"

#endif // CONSTANTS_H
