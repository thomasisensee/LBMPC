#ifndef COLLISION_MODEL_H
#define COLLISION_MODEL_H

#include <stdio.h>
#include <vector>

#include "core/kernelParameters.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
class CollisionModel {
protected:
    /// Relaxation parameter associated with shear viscosity
    const T _omegaShear;

public:
    /// Constructor
    CollisionModel(T omega);

    /// Destructor
    virtual ~CollisionModel() = default;

    T getOmegaShear() const;
    virtual void prepareKernelParams(const LBParams<T>& lbParams) = 0;
    virtual void doCollision(
        T* distribution,
        std::pair<unsigned int, unsigned int> numBlocks,
        std::pair<unsigned int, unsigned int> threadsPerBlock
    ) = 0;
    virtual void printParameters() = 0;
};



/******************************************************/
/***** Derived class 01: BGK Collision parameters *****/
/******************************************************/
template<typename T>
class CollisionBGK final : public CollisionModel<T> {
private:
    /// Parameters to pass to cuda kernels
    CollisionParamsBGKWrapper<T> _params;

public:
    /// Constructor
    CollisionBGK(T omegaS);

    /// Destructor
    virtual ~CollisionBGK();

    virtual void prepareKernelParams(const LBParams<T>& lbParams) override;
    virtual void doCollision(
        T* distribution,
        std::pair<unsigned int, unsigned int> numBlocks,
        std::pair<unsigned int, unsigned int> threadsPerBlock
    ) override;
    virtual void printParameters();
};

/******************************************************/
/***** Derived class 02: CHM Collision parameters *****/
/******************************************************/
template<typename T>
class CollisionCHM final : public CollisionModel<T> { // only implemented for D2Q9 lattices
private:
    /// Relaxation parameter associated with bulk viscosity
    const T _omegaBulk;

    /// Parameters to pass to cuda kernels
    CollisionParamsCHMWrapper<T> _params;

public:
    /// Constructor
    CollisionCHM(T omegaS, T omegaB);

    /// Destructor
    virtual ~CollisionCHM();

    T getOmegaBulk() const;
    virtual void prepareKernelParams(const LBParams<T>& lbParams) override;
    virtual void doCollision(
        T* distribution,
        std::pair<unsigned int, unsigned int> numBlocks,
        std::pair<unsigned int, unsigned int> threadsPerBlock
    ) override;
    virtual void printParameters() override;
};

#include "collisionModel.hh"

#endif // COLLISION_MODEL_H
