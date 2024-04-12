#ifndef KERNEL_PARAMETERS_H
#define KERNEL_PARAMETERS_H

#include <vector>
#include "core/constants.h"

/***********************************************/
/***** Structs for passing to cuda kernels *****/
/***********************************************/

/***********************/
/***** Base struct *****/
/***********************/
struct BaseParams {
    /// Grid
    unsigned int Nx;
    unsigned int Ny;
};

/************************************************/
/***** Derived struct 01: LBParams (BGK) ****/
/************************************************/
template<typename T>
struct CollisionParamsBGK : public BaseParams {
    /// Relaxation parameter related to shear viscosity
    T omegaShear;
};

/************************************************/
/***** Derived struct 02: CollisionParamsCHM ****/
/************************************************/
template<typename T>
struct CollisionParamsCHM : public CollisionParamsBGK<T> {
    /// Relaxation parameter related to bulk viscosity
    T omegaBulk;
};

/********************************************/
/***** Derived struct 03: BoundaryParams ****/
/********************************************/
template<typename T>
struct BoundaryParams : public BaseParams {
    /// Properties needed for boundary conditions
    T* WALL_VELOCITY = nullptr;
    BoundaryLocation location;
};


/*********************************************************************************/
/***** Wrapper classes to hold host and device versions of the above structs *****/
/*********************************************************************************/

/**********************************/
/***** Base class (templated) *****/
/**********************************/
template<typename T, typename ParamsType>
class ParamsWrapper {
protected:
    ParamsType  _hostParams;
    ParamsType* _deviceParams = nullptr;

public:
    /// Default constructor
    ParamsWrapper();

    /// Parameterized constructor
    ParamsWrapper(
        unsigned int nx,
        unsigned int ny
    );

    /// Destructor
    virtual ~ParamsWrapper();

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny
    );

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() = 0;

    /// Cleans up host memory
    virtual void cleanupHost() = 0;

    /// Cleans up device memory
    virtual void cleanupDevice() = 0;

    /// Accessors for host and device params
    const ParamsType& getHostParams() const;
    ParamsType* getDeviceParams();
};

/************************************************/
/***** Derived class 02: CollisionParamsBGK *****/
/************************************************/
template<typename T>
class CollisionParamsBGKWrapper : public ParamsWrapper<T, CollisionParamsBGK<T>> {
public:
    /// Default constructor
    CollisionParamsBGKWrapper();

    /// Parameterized constructor
    CollisionParamsBGKWrapper(
        unsigned int nx,
        unsigned int ny,
        T omegaShear
    );

    /// Destructor
    virtual ~CollisionParamsBGKWrapper();

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny,
        T omegaShear
    );

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;

    /// Cleans up host memory
    void cleanupHost() override;

    /// Cleans up device memory
    void cleanupDevice() override;
};

/************************************************/
/***** Derived class 03: CollisionParamsCHM *****/
/************************************************/
template<typename T>
class CollisionParamsCHMWrapper : public CollisionParamsBGKWrapper<T> {
public:
    /// Default constructor
    CollisionParamsCHMWrapper();

    /// Parameterized constructor
    CollisionParamsCHMWrapper(
        unsigned int nx,
        unsigned int ny,
        T omegaShear,
        T omegaBulk
    );

    /// Destructor
    virtual ~CollisionParamsCHMWrapper();

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny,
        T omegaShear,
        T omegaBulk
    );

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;

    /// Cleans up host memory
    void cleanupHost() override;

    /// Cleans up device memory
    void cleanupDevice() override;
};

/********************************************/
/***** Derived class 04: BoundaryParams *****/
/********************************************/
template<typename T>
class BoundaryParamsWrapper : public ParamsWrapper<T, BoundaryParams<T>> {
private:
    // For keeping track of the wallVelocity dimension/size
    unsigned int _D;

public:
    /// Default constructor
    BoundaryParamsWrapper();

    /// Parameterized constructor
    BoundaryParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        const T* WALL_VELOCITY,
        BoundaryLocation location
    );

    /// Destructor
    virtual ~BoundaryParamsWrapper();

    /// Set values and trigger trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny,
        const T* WALL_VELOCITY,
        BoundaryLocation location
    );

    /// Set wall velocity specifically and trigger allocateAndCopyToDevice
    void setWallVelocity(const std::vector<T>& wallVelocity);
    
    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;

    /// Cleans up host memory
    void cleanupHost() override;

    /// Cleans up device memory
    void cleanupDevice() override;
};

#include "kernelParameters.hh"

#endif // KERNEL_PARAMETERS_H
