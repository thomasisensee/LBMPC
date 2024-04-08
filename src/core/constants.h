#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string> // For std::string

/// Allowed boundary locations
enum class BoundaryLocation {
    WEST = 0,
    EAST,
    SOUTH,
    NORTH,
    COUNT // Sentinel, not a real location, just marks the end
};

//#include "constants.hh"

#endif // CONSTANTS_H
