#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <stdio.h>
#include <memory>

#include "core/constants.h"
#include "core/lb/lbConstants.h"

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

    virtual void printParameters() const;
    virtual void printBoundaryLocation() const;

};


/**************************************/
/***** Derived class 01: Periodic *****/
/**************************************/
template<typename T>
class Periodic final : public BoundaryCondition<T> {
public:
    /// Constructor
    Periodic(BoundaryLocation loc);

    /// Destructor
    virtual ~Periodic() = default;

    void apply(T* lbField) override;

    void printParameters() const override;
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

    void printParameters() const override;

};

/**********************************************************/
/***** Derived class 03: Fixed Velocity (Bounce Back) *****/
/**********************************************************/
template<typename T>
class MovingWall final : public BounceBack<T> {
    /// Wall velocity
    std::vector<T> _wallVelocity;

    /// Helper variables (currently only for printing the correct velocity)
    T _dxdt; // dx/dt

public:
    /// Constructor
    MovingWall(BoundaryLocation loc, const std::vector<T>& velocity);

    /// Destructor
    virtual ~MovingWall() = default;

    void prepareKernelParams(const LBParams<T>& lbParams, const LBModel<T>* lbModel) override;

    const std::vector<T>& getWallVelocity() const;
    void printParameters() const override;

    /// Setter for _dxdt
    void setDxdt(T dxdt);
};


/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T>
class BoundaryConditionManager {
    /// Vector of boundary condition objects
    std::vector<std::unique_ptr<BoundaryCondition<T>>> boundaryConditions;

    /// Helper variables (currently only for printing the correct velocity in case of moving wall)
    T _dxdt; // dx/dt

public:
    /// Constructor
    BoundaryConditionManager();

    /// Destructor
    ~BoundaryConditionManager() = default;

    /// Setter for _dxdt
    void setDxdt(T dxdt);

    void addBoundaryCondition(std::unique_ptr<BoundaryCondition<T>> condition);
    void prepareKernelParams(const LBParams<T>& lbParams, const LBModel<T>* lbModel);
    void apply(T* lbField);
    void printParameters() const;
};

#include "boundaryConditions.hh"

#endif // BOUNDARY_CONDITIONS_H
