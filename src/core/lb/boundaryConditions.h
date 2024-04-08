#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <stdio.h>
#include <iostream>     // For std::cout
#include <map>          // For std::map
#include <memory>       // For std::unique_ptr
#include <string>       // For std::string

#include "core/constants.h"
#include "core/kernelParameters.h"

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
    virtual void prepareKernelParams(const LBParams<T>& lbParams, const LBModel<T>* lbModel);
    virtual void apply(T* lbField) = 0;

    virtual void print() const;
    virtual void printBoundaryLocation() const;

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

    void print() const override;
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

    virtual void apply(T* lbField) override;

    void print() const override;

};

/**********************************************************/
/***** Derived class 03: Fixed Velocity (Bounce Back) *****/
/**********************************************************/
template<typename T>
class MovingWall final : public BounceBack<T> {
    /// Wall velocity
    std::vector<T> _wallVelocity;

public:
    /// Constructor
    MovingWall(BoundaryLocation loc, const std::vector<T>& velocity);

    /// Destructor
    virtual ~MovingWall() = default;

    void prepareKernelParams(const LBParams<T>& lbParams, const LBModel<T>* lbModel) override;

    const std::vector<T>& getWallVelocity() const;
    void print() const override;

    //void apply(T* lbField) override;
};


/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T>
class BoundaryConditionManager {
    /// Vector of boundary condition objects
    std::vector<std::unique_ptr<BoundaryCondition<T>>> boundaryConditions;

public:
    /// Constructor
    BoundaryConditionManager();

    /// Destructor
    ~BoundaryConditionManager() = default;

    void addBoundaryCondition(std::unique_ptr<BoundaryCondition<T>> condition);
    void prepareKernelParams(const LBParams<T>& lbParams, const LBModel<T>* lbModel);
    void apply(T* lbField);
    void print() const;
};

#include "boundaryConditions.hh"

#endif // BOUNDARY_CONDITIONS_H
