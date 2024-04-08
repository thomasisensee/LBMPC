#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include "utilities.h"

std::string boundaryLocationToString(BoundaryLocation location) {
    switch (location) {
        case BoundaryLocation::WEST:    return "WEST";
        case BoundaryLocation::EAST:    return "EAST";
        case BoundaryLocation::SOUTH:   return "SOUTH";
        case BoundaryLocation::NORTH:   return "NORTH";
        default: return "UNKNOWN";
    }
}

std::string formatElapsedTime(float milliseconds) {
    // Convert milliseconds to total seconds
    int totalSeconds = static_cast<int>(milliseconds / 1000);
    
    // Calculate days, hours, minutes, and seconds
    int days = totalSeconds / (24 * 3600);
    totalSeconds %= (24 * 3600);
    int hours = totalSeconds / 3600;
    totalSeconds %= 3600;
    int minutes = totalSeconds / 60;
    int seconds = totalSeconds % 60;

    // Format the string based on the time components
    std::stringstream ss;
    if (days > 0) {
        ss << days << " d:";
    }
    if (hours > 0 || days > 0) { // Show hours if days are shown
        ss << std::setw(2) << std::setfill('0') << hours << " h:";
    }
    if (minutes > 0 || hours > 0 || days > 0) { // Show minutes if hours are shown
        ss << std::setw(2) << std::setfill('0') << minutes << " m:";
    }
    // Always show seconds
    ss << std::setw(2) << std::setfill('0') << seconds << " s";

    return ss.str();
}