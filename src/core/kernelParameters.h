#ifndef KERNEL_PARAMETERS_H
#define KERNEL_PARAMETERS_H

#include <cuda_runtime.h>
#include "cuda/cudaConstants.cuh"
#include "cuda/cudaErrorHandler.cuh"

/// Allowed boundary locations
enum class BoundaryLocation {
    WEST, EAST, SOUTH, NORTH
};

/***********************************************/
/***** Structs for passing to cuda kernels *****/
/***********************************************/

template<typename T>
struct BaseParams {
    /// Grid
    unsigned int D;
    unsigned int Nx;
    unsigned int Ny;
};

template<typename T>
struct LBParams : public BaseParams<T> {
    unsigned int Q;
    int* LATTICE_VELOCITIES;
    T* LATTICE_WEIGHTS;
};

template<typename T>
struct CollisionParamsBGK : public LBParams<T> {
    T omegaShear;
};

template<typename T>
struct CollisionParamsCHM : public CollisionParamsBGK<T> {
    T omegaBulk;
};

template<typename T>
struct BoundaryParams : public LBParams<T> {
    unsigned int* OPPOSITE_POPULATION; 
    T* wallVelocity;
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
    ParamsType hostParams;
    ParamsType* deviceParams = nullptr;

public:
    /// Default constructor
    ParamsWrapper();

    /// Parameterized constructor
    ParamsWrapper(unsigned int dim, unsigned int nx, unsigned int ny);

    /// Destructor
    virtual ~ParamsWrapper() = default;

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(unsigned int dim, unsigned int nx, unsigned int ny);

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice();

    /// Cleans up host memory
    virtual void cleanupHost() = 0;

    /// Cleans up device memory
    virtual void cleanupDevice() = 0;

    /// Accessors for host and device params
    ParamsType& getHostParams();
    ParamsType* getDeviceParams();
};

/***************************************/
/***** Derived classe 01: LBParams *****/
/***************************************/
template<typename T>
class LBParamsWrapper : public ParamsWrapper<T, LBParams<T>> {
public:
    /// Default constructor
    LBParamsWrapper();
    
    /// Parameterized constructor
    LBParamsWrapper(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights);

    /// Destructor
    ~LBParamsWrapper() override;

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights);

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;

    /// Cleans up host memory
    virtual void cleanupHost() override;

    /// Cleans up device memory
    virtual void cleanupDevice() override;
};

/*************************************************/
/***** Derived classe 02: CollisionParamsBGK *****/
/*************************************************/
template<typename T>
class CollisionParamsBGKWrapper : public ParamsWrapper<T, CollisionParamsBGK<T>> {
public:
    /// Default constructor
    CollisionParamsBGKWrapper();

    /// Parameterized constructor
    CollisionParamsBGKWrapper(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights, T omegaShear);

    /// Destructor
    ~CollisionParamsBGKWrapper() override;

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights, T omegaShear);

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;

    /// Cleans up host memory
    virtual void cleanupHost() override;

    /// Cleans up device memory
    virtual void cleanupDevice() override;
};

/*************************************************/
/***** Derived classe 03: CollisionParamsCHM *****/
/*************************************************/
template<typename T>
class CollisionParamsCHMWrapper : public ParamsWrapper<T, CollisionParamsCHM<T>> {
public:
    /// Default constructor
    CollisionParamsCHMWrapper();

    /// Parameterized constructor
    CollisionParamsCHMWrapper(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights, T omegaShear, T omegaBulk);

    /// Destructor
    ~CollisionParamsCHMWrapper() override;

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights, T omegaShear, T omegaBulk);

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;

    /// Cleans up host memory
    virtual void cleanupHost() override;

    /// Cleans up device memory
    virtual void cleanupDevice() override;
};

/*********************************************/
/***** Derived classe 04: BoundaryParams *****/
/*********************************************/
template<typename T>
class BoundaryParamsWrapper : public ParamsWrapper<T, BoundaryParams<T>> {
public:
    /// Default constructor
    BoundaryParamsWrapper();

    /// Parameterized constructor
    BoundaryParamsWrapper(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights, unsigned int* OPPOSITE_POPULATION, T* wallVelocity, BoundaryLocation location);

    /// Destructor
    ~BoundaryParamsWrapper() override;

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights, unsigned int* OPPOSITE_POPULATION, T* wallVelocity, BoundaryLocation location);

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;

    /// Cleans up host memory
    virtual void cleanupHost() override;

    /// Cleans up device memory
    virtual void cleanupDevice() override;
};

#include "kernelParameters.hh"

#endif // KERNEL_PARAMETERS_H
