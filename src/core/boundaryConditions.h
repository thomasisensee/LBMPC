#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <stdio.h>
#include <iostream>     // For std::cout
#include <map>          // For std::map
#include <memory>       // For std::unique_ptr
#include <string>       // Also include this if you're using std::string

#include "core/kernelParameters.h"

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

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
class BoundaryCondition {
protected:
    /// location from enum class (only WEST, EAST, SOUTH, NORTH allowed)
    BoundaryLocation _location;

    /// Parameters to pass to cuda kernels
    BoundaryParamsWrapper<T> _params;

    /// Cuda grid and block size
    unsigned int _numBlocks;
    unsigned int _threadsPerBlock;
public:
    /// Constructor
    BoundaryCondition(BoundaryLocation loc);

    /// Destructor
    virtual ~BoundaryCondition() = default;

    BoundaryLocation getLocation() const;
    void prepareKernelParams(const LBParams<T>& lbParams, const LBModel<T>* lbModel);
    virtual void apply(T* lbField) = 0;
};


/**************************************/
/***** Derived class 01: Periodic *****/
/**************************************/
template<typename T>
class PeriodicBoundary final : public BoundaryCondition<T> {
public:
    /// Constructor
    PeriodicBoundary(BoundaryLocation loc);

    /// Destructor
    virtual ~PeriodicBoundary() = default;

    void apply(T* lbField) override;
};

/*****************************************/
/***** Derived class 02: Bounce Back *****/
/*****************************************/
template<typename T>
class BounceBack : public BoundaryCondition<T> {
public:
    /// Constructor
    BounceBack(BoundaryLocation loc);

    /// Destructor
    virtual ~BounceBack() = default;

    void apply(T* lbField) override;
};

/**********************************************************/
/***** Derived class 03: Fixed Velocity (Bounce Back) *****/
/**********************************************************/
template<typename T>
class FixedVelocityBoundary final : public BounceBack<T> {
    /// Wall velocity
    std::vector<T> _wallVelocity;

public:
    /// Constructor
    FixedVelocityBoundary(BoundaryLocation loc, const std::vector<T>& velocity);

    /// Destructor
    virtual ~FixedVelocityBoundary() = default;

    void prepareKernelParams(const LBParams<T>& lbParams, const LBModel<T>* lbModel);
};


/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T>
class BoundaryConditionManager {
    /// Map of location, name, boundary condition object
    std::map<BoundaryLocation, std::map<std::string, std::unique_ptr<BoundaryCondition<T>>>> boundaryConditions;

public:
    /// Constructor
    BoundaryConditionManager();

    /// Destructor
    ~BoundaryConditionManager() = default;

    void addBoundaryCondition(const std::string& name, std::unique_ptr<BoundaryCondition<T>> condition);
    void prepareKernelParams(const LBParams<T>& lbParams, const LBModel<T>* lbModel);
    void apply(T* lbField);
    void print() const;
};

#include "boundaryConditions.hh"

#endif // BOUNDARY_CONDITIONS_H
