#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <stdio.h>
#include <memory>

#include "core/constants.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T,typename DESCRIPTOR>
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
    virtual void prepareKernelParams(const BaseParams& baseParams);
    virtual void apply(T* lbField) = 0;

    virtual void printParameters() const;
    virtual void printBoundaryLocation() const;

};


/**************************************/
/***** Derived class 01: Periodic *****/
/**************************************/
template<typename T,typename DESCRIPTOR>
class Periodic final : public BoundaryCondition<T,DESCRIPTOR> {
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
template<typename T,typename DESCRIPTOR>
class BounceBack : public BoundaryCondition<T,DESCRIPTOR> {
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
template<typename T,typename DESCRIPTOR>
class MovingWall final : public BounceBack<T,DESCRIPTOR> {
private:
    /// Wall velocity
    std::vector<T> _wallVelocity;

    /// Helper variables (currently only for printing the correct velocity)
    T _dxdt; // dx/dt

public:
    /// Constructor
    MovingWall(BoundaryLocation loc, const std::vector<T>& velocity);

    /// Destructor
    virtual ~MovingWall() = default;

    void prepareKernelParams(const BaseParams& baseParams) override;

    const std::vector<T>& getWallVelocity() const;
    void printParameters() const override;

    /// Setter for _dxdt
    void setDxdt(T dxdt);
};

/**********************************************/
/***** Derived class 04: Anti Bounce Back *****/
/**********************************************/
template<typename T,typename DESCRIPTOR>
class AntiBounceBack : public BoundaryCondition<T,DESCRIPTOR> {
private:
    /// Value of the scalar field (temperature) at the wall
    T _wallValue;

public:
    /// Constructor
    AntiBounceBack(BoundaryLocation loc, T wallValue);

    /// Destructor
    virtual ~AntiBounceBack() = default;

    virtual void apply(T* lbField) override;

    void prepareKernelParams(const BaseParams& baseParams) override;

    const T getWallValue() const;
    void printParameters() const override;
};

/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T,typename DESCRIPTOR>
class BoundaryConditionManager {
    /// Vector of boundary condition objects
    std::vector<std::unique_ptr<BoundaryCondition<T,DESCRIPTOR>>> boundaryConditions;

    /// Helper variables (currently only for printing the correct velocity in case of moving wall)
    T _dxdt; // dx/dt

public:
    /// Constructor
    BoundaryConditionManager();

    /// Destructor
    ~BoundaryConditionManager() = default;

    /// Setter for _dxdt
    void setDxdt(T dxdt);

    void addBoundaryCondition(std::unique_ptr<BoundaryCondition<T,DESCRIPTOR>> condition);
    void prepareKernelParams(const BaseParams& baseParams);
    void apply(T* lbField);
    void printParameters() const;
};

#include "boundaryConditions.hh"

#endif // BOUNDARY_CONDITIONS_H
