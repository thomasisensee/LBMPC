#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>

#include "constants.h"

/// Transform BoundaryLocation members to strings for output
std::string boundaryLocationToString(BoundaryLocation location);

/// Transform time in milliseconds into human readable format
std::string formatElapsedTime(float milliseconds);

#endif // UTILITIES_H