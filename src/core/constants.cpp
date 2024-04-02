#ifndef CONSTANTS_HH
#define CONSTANTS_HH

#include "constants.h"

/// Transform BoundaryLocation members to strings for output
std::string boundaryLocationToString(BoundaryLocation location) {
    switch (location) {
        case BoundaryLocation::WEST:    return "WEST";
        case BoundaryLocation::EAST:    return "EAST";
        case BoundaryLocation::SOUTH:   return "SOUTH";
        case BoundaryLocation::NORTH:   return "NORTH";
        default: return "UNKNOWN";
    }
}

#endif // CONSTANTS_HH
