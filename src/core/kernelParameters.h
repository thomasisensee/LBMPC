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
    unsigned int D;
    unsigned int Nx;
    unsigned int Ny;
};

/**************************************/
/***** Derived struct 01: LBParams ****/
/**************************************/
template<typename T>
struct LBParams : public BaseParams {
    /// Lattice properties
    unsigned int Q;
    int* LATTICE_VELOCITIES = nullptr;
    T* LATTICE_WEIGHTS      = nullptr;
};

/************************************************/
/***** Derived struct 02: CollisionParamsBGK ****/
/************************************************/
template<typename T>
struct CollisionParamsBGK : public LBParams<T> {
    /// Relaxation parameter related to shear viscosity
    T omegaShear;
};

/************************************************/
/***** Derived struct 03: CollisionParamsCHM ****/
/************************************************/
template<typename T>
struct CollisionParamsCHM : public CollisionParamsBGK<T> {
    /// Relaxation parameter related to bulk viscosity
    T omegaBulk;
};

/********************************************/
/***** Derived struct 04: BoundaryParams ****/
/********************************************/
template<typename T>
struct BoundaryParams : public LBParams<T> {
    /// Properties needed for boundary conditions
    unsigned int* OPPOSITE_POPULATION   = nullptr;
    unsigned int* POPULATION            = nullptr;
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
    ParamsType  _hostParams;
    ParamsType* _deviceParams = nullptr;

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
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS
    );

    /// Destructor
    virtual ~LBParamsWrapper();

    /// Set values and trigger allocateAndCopyToDevice
    virtual void setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS
    );

    /// Allocates device memory and copies data from the host instance
    virtual void allocateAndCopyToDevice() override;

    /// Cleans up host memory
    void cleanupHost() override;

    /// Cleans up device memory
    void cleanupDevice() override;
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
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
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
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
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
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
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
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
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
public:
    /// Default constructor
    BoundaryParamsWrapper();

    /// Parameterized constructor
    BoundaryParamsWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
        const unsigned int* POPULATION,
        const unsigned int* OPPOSITE_POPULATION,
        const T* WALL_VELOCITY,
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
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
        const unsigned int* POPULATION,           // Populations that have to be set when boundary condition are applied
        const unsigned int* OPPOSITE_POPULATION,  // Opposite populations for each Q populations
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
