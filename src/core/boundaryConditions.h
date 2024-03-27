#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <stdio.h>
#include <iostream>     // For std::cout
#include <map>          // For std::map
#include <memory>       // For std::unique_ptr
#include <string>       // Also include this if you're using std::string


/// Allowed boundary locations
enum class BoundaryLocation {
    EAST, WEST, SOUTH, NORTH
};

/// Transform BoundaryLocation members to strings for output
std::string boundaryLocationToString(BoundaryLocation location) {
    switch (location) {
        case BoundaryLocation::EAST: return "EAST";
        case BoundaryLocation::WEST: return "WEST";
        case BoundaryLocation::SOUTH: return "SOUTH";
        case BoundaryLocation::NORTH: return "NORTH";
        default: return "UNKNOWN";
    }
}

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
class BoundaryCondition {
private:
    /// Parameters to pass to cuda kernels
    BoundaryParams<T> hostParams;
    BoundaryParams<T>* deviceParams = nullptr;
public:
    /// Destructor
    ~BoundaryCondition();
    void prepareKernelParams(LBMParams<T>* lbmParams);
    void copyKernelParamsToDevice();
    virtual void apply(T* lbmField) = 0;

};


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
class BounceBack : public BoundaryCondition<T> {
public:
    void apply(T* lbmField) override;
};

template<typename T>
class FixedVelocityBoundary : public BoundaryCondition<T> {
    std::vector<T> wallVelocity;
public:
    FixedVelocityBoundary(const std::vector<T>& velocity);
    void prepareKernelParams(LBMParams<T>* lbmParams);
    void apply(T* lbmField) override;
};

template<typename T>
class PeriodicBoundary : public BoundaryCondition<T> {
public:
    void apply(T* lbmField) override;
};


/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T>
class BoundaryConditionManager {
    std::map<BoundaryLocation, std::map<std::string, std::unique_ptr<BoundaryCondition<T>>>> boundaryConditions;
public:
    /// Constructor
    BoundaryConditionManager();
    void addBoundaryCondition(BoundaryLocation boundary, const std::string& name, std::unique_ptr<BoundaryCondition<T>> condition);
    void prepareAndCopyKernelParamsToDevice();
    void apply(T* lbmField);
    void print() const;
};

#include "boundaryConditions.hh"

#endif // BOUNDARY_CONDITIONS_H
