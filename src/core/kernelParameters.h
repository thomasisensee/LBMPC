#ifndef KERNEL_PARAMETERS_H
#define KERNEL_PARAMETERS_H

#include <cuda_runtime.h>
#include "core/constants.h"
#include "cuda/cudaConstants.cuh"
#include "cuda/cudaErrorHandler.cuh"

/***********************************************/
/***** Structs for passing to cuda kernels *****/
/***********************************************/

/***********************/
/***** Base struct *****/
/***********************/
template<typename T>
struct BaseParams {
    /// Grid
    unsigned int D;
    unsigned int Nx;
    unsigned int Ny;
};

/**************************************/
/***** Derived struct 01: LBParams ****/
/**************************************/
template<typename T>
struct LBParams : public BaseParams<T> {
    unsigned int Q;
    int* LATTICE_VELOCITIES = nullptr;
    T* LATTICE_WEIGHTS      = nullptr;
};

/************************************************/
/***** Derived struct 02: CollisionParamsBGK ****/
/************************************************/
template<typename T>
struct CollisionParamsBGK : public LBParams<T> {
    T omegaShear;
};

/************************************************/
/***** Derived struct 03: CollisionParamsCHM ****/
/************************************************/
template<typename T>
struct CollisionParamsCHM : public CollisionParamsBGK<T> {
    T omegaBulk;
};

/********************************************/
/***** Derived struct 04: BoundaryParams ****/
/********************************************/
template<typename T>
struct BoundaryParams : public LBParams<T> {
    unsigned int* OPPOSITE_POPULATION   = nullptr; 
    T* WALL_VELOCITY                    = nullptr;
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
    ParamsType  hostParams;
    ParamsType* deviceParams = nullptr;

public:
    /// Default constructor
    ParamsWrapper();

    /// Parameterized constructor
    ParamsWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny
    );

    /// Destructor
    virtual ~ParamsWrapper();

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny
    );

    /// Set wall velocity specifically and trigger allocateAndCopyToDevice
    void setWallVelocity(T* WALL_VELOCITY);

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice();

    /// Cleans up host memory
    void cleanupHost();

    /// Cleans up device memory
    void cleanupDevice();

    /// Accessors for host and device params
    const ParamsType& getHostParams() const;
    ParamsType* getDeviceParams();
};

/**************************************/
/***** Derived class 01: LBParams *****/
/**************************************/
template<typename T>
class LBParamsWrapper : public ParamsWrapper<T, LBParams<T>> {
public:
    /// Default constructor
    LBParamsWrapper();
    
    /// Parameterized constructor
    LBParamsWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS
    );

    /// Destructor
    virtual ~LBParamsWrapper();

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS
    );

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;
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
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        T omegaShear
    );

    /// Destructor
    virtual ~CollisionParamsBGKWrapper();

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        T omegaShear
    );

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;
};

/************************************************/
/***** Derived class 03: CollisionParamsCHM *****/
/************************************************/
template<typename T>
class CollisionParamsCHMWrapper : public ParamsWrapper<T, CollisionParamsCHM<T>> {
public:
    /// Default constructor
    CollisionParamsCHMWrapper();

    /// Parameterized constructor
    CollisionParamsCHMWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        T omegaShear,
        T omegaBulk
    );

    /// Destructor
    virtual ~CollisionParamsCHMWrapper();

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        T omegaShear,
        T omegaBulk
    );

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;
};

/********************************************/
/***** Derived class 04: BoundaryParams *****/
/********************************************/
template<typename T>
class BoundaryParamsWrapper : public ParamsWrapper<T, BoundaryParams<T>> {
public:
    /// Default constructor
    BoundaryParamsWrapper();

    /// Parameterized constructor
    BoundaryParamsWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        unsigned int* OPPOSITE_POPULATION,
        T* WALL_VELOCITY,
        BoundaryLocation location
    );

    /// Destructor
    virtual ~BoundaryParamsWrapper();

    /// Set values and trigger trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        unsigned int* OPPOSITE_POPULATION,
        T* WALL_VELOCITY,
        BoundaryLocation location
    );
    
    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;

    /// Cleans up host memory
    void cleanupHost();

    /// Cleans up device memory
    void cleanupDevice();
};

#include "kernelParameters.hh"

#endif // KERNEL_PARAMETERS_H
