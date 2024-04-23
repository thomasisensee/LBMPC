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

/****************************************************/
/***** Derived struct 01: CollisionParams (BGK)  ****/
/****************************************************/
template<typename T>
struct CollisionParamsBGK : public BaseParams {
    /// Relaxation parameter related to shear viscosity
    T omegaShear;
};

/****************************************************/
/***** Derived struct 02: CollisionParams (CHM)  ****/
/****************************************************/
template<typename T>
struct CollisionParamsCHM : public CollisionParamsBGK<T> {
    /// Relaxation parameter related to bulk viscosity
    T omegaBulk;
};

/***********************************************/
/***** Derived structs (03): BoundaryParams ****/
/***********************************************/
struct BoundaryParams : public BaseParams {
    /// Boundary location
    BoundaryLocation location;
};

struct PeriodicParams : public BoundaryParams {};

struct BounceBackParams : public BoundaryParams {};

template<typename T>
struct AntiBounceBackParams : public BoundaryParams {
    /// Scalar value for dirichlet condition (via anti-bounce-back)
    T wallValue = 0.0;
};

template<typename T>
struct MovingWallParams : public BoundaryParams {
    /// Wall velocity
    T* wallVelocity = nullptr;
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
    ParamsWrapper() = default;

    /// Constructor
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
    virtual void allocateAndCopyToDevice();

    /// Cleans up host memory
    virtual void cleanupHost();

    /// Cleans up device memory
    virtual void cleanupDevice();

    /// Accessors for host and device params
    const ParamsType& getHostParams() const;
    ParamsType* getDeviceParams();
};

/****************************************/
/***** Derived class 01: PaseParams *****/
/****************************************/
template<typename T>
class BaseParamsWrapper : public ParamsWrapper<T, BaseParams> {
public:
    /// Default constructor
    BaseParamsWrapper() = default;

    /// Parameterized constructor
    BaseParamsWrapper(
        unsigned int nx,
        unsigned int ny
    );

    /// Destructor
    virtual ~BaseParamsWrapper();

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny
    );
};

/************************************************/
/***** Derived class 02: CollisionParamsBGK *****/
/************************************************/
template<typename T>
class CollisionParamsBGKWrapper : public ParamsWrapper<T, CollisionParamsBGK<T>> {
public:
    /// Default constructor
    CollisionParamsBGKWrapper() = default;

    /// Parameterized constructor
    CollisionParamsBGKWrapper(
        unsigned int nx,
        unsigned int ny,
        T omegaShear
    );

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny,
        T omegaShear
    );
};

/************************************************/
/***** Derived class 03: CollisionParamsCHM *****/
/************************************************/
template<typename T>
class CollisionParamsCHMWrapper final : public ParamsWrapper<T, CollisionParamsCHM<T>> {
public:
    /// Default constructor
    CollisionParamsCHMWrapper() = default;

    /// Parameterized constructor
    CollisionParamsCHMWrapper(
        unsigned int nx,
        unsigned int ny,
        T omegaShear,
        T omegaBulk
    );

    /// Set values and trigger allocateAndCopyToDevice
    void setValues(
        unsigned int nx,
        unsigned int ny,
        T omegaShear,
        T omegaBulk
    );
};

/********************************************/
/***** Derived class 04: BoundaryParams *****/
/********************************************/
template<typename T>
class BoundaryParamsWrapper : public ParamsWrapper<T, BoundaryParams> {
public:
    /// Default constructor
    BoundaryParamsWrapper() = default;

    /// Parameterized constructor
    BoundaryParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    );

    /// Set values and trigger trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    );
};

/********************************************/
/***** Derived class 05: PeriodicParams *****/
/********************************************/
template<typename T>
class PeriodicParamsWrapper : public ParamsWrapper<T, PeriodicParams> {
public:
    /// Default constructor
    PeriodicParamsWrapper() = default;

    /// Parameterized constructor
    PeriodicParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    );

    /// Set values and trigger trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    );
};

/**********************************************/
/***** Derived class 06: BounceBackParams *****/
/**********************************************/
template<typename T>
class BounceBackParamsWrapper : public ParamsWrapper<T, BounceBackParams> {
public:
    /// Default constructor
    BounceBackParamsWrapper() = default;

    /// Parameterized constructor
    BounceBackParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    );

    /// Set values and trigger trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    );
};

template<typename T>
class AntiBounceBackParamsWrapper : public ParamsWrapper<T, AntiBounceBackParams<T>> {
public:
    /// Default constructor
    AntiBounceBackParamsWrapper() = default;

    /// Parameterized constructor
    AntiBounceBackParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location,
        T wallValue
    );

    /// Destructor
    virtual ~AntiBounceBackParamsWrapper();

    /// Set values and trigger trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location,
        T wallValue
    );

    /// Set wall velocity specifically and trigger allocateAndCopyToDevice
    void setWallValue(T wallValue);
    
    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;
};

template<typename T>
class MovingWallParamsWrapper : public ParamsWrapper<T, MovingWallParams<T>> {
private:
    // For keeping track of the wallVelocity dimension/size
    unsigned int _D;

public:
    /// Default constructor
    MovingWallParamsWrapper() = default;

    /// Parameterized constructor
    MovingWallParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location,
        const T* wallVelocity
    );

    /// Destructor
    virtual ~MovingWallParamsWrapper();

    /// Set values and trigger trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location,
        const T* wallVelocity
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