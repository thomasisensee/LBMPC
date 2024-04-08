#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string> // For std::string

/// Allowed boundary locations
enum class BoundaryLocation {
    WEST,
    EAST,
    SOUTH,
    NORTH
};

/// Transform BoundaryLocation members to strings for output
std::string boundaryLocationToString(BoundaryLocation location);

//#include "constants.hh"

#endif // CONSTANTS_H
