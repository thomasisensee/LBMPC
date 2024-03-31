#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <stdio.h>
#include <iostream>     // For std::cout
#include <map>          // For std::map
#include <memory>       // For std::unique_ptr
#include <string>       // Also include this if you're using std::string

#include "core/kernelParams.h"

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
protected:
    BoundaryLocation location;
    /// Parameters to pass to cuda kernels
    BoundaryParams<T> hostParams;
    BoundaryParams<T>* deviceParams = nullptr;
public:
    /// Constructor
    BoundaryCondition(BoundaryLocation loc);
    /// Destructor
    ~BoundaryCondition();
    BoundaryLocation getLocation() const;
    void prepareKernelParams(LBParams<T>* lbParams);
    void copyKernelParamsToDevice();
    virtual void apply(T* lbField) = 0;
};


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
class BounceBack : public BoundaryCondition<T> {
public:
    /// Constructor
    BounceBack(BoundaryLocation loc);
    void apply(T* lbField) override;
};

template<typename T>
class FixedVelocityBoundary : public BoundaryCondition<T> {
    std::vector<T> wallVelocity;
public:
    /// Constructor
    FixedVelocityBoundary(BoundaryLocation loc, const std::vector<T>& velocity);
    void prepareKernelParams(LBParams<T>* lbParams);
    void apply(T* lbField) override;
};

template<typename T>
class PeriodicBoundary : public BoundaryCondition<T> {
public:
    /// Constructor
    PeriodicBoundary(BoundaryLocation loc);
    void apply(T* lbField) override;
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
    void addBoundaryCondition(const std::string& name, std::unique_ptr<BoundaryCondition<T>> condition);
    void prepareAndCopyKernelParamsToDevice();
    void apply(T* lbField);
    void print() const;
};

#include "boundaryConditions.hh"

#endif // BOUNDARY_CONDITIONS_H
