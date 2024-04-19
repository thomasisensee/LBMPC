#ifndef COLLISION_MODEL_H
#define COLLISION_MODEL_H

#include <stdio.h>
#include <utility>

#include "core/kernelParameters.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T,typename DESCRIPTOR>
class CollisionModel {
protected:
    /// Relaxation parameter associated with shear viscosity
    const T _omegaShear;

public:
    /// Constructor
    explicit CollisionModel(T omega);

    /// Destructor
    virtual ~CollisionModel() = default;

    T getOmegaShear() const;
    virtual void prepareKernelParams(const CollisionParamsBGK<T>& CollisionParamsBGK) = 0;
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
template<typename T,typename DESCRIPTOR>
class CollisionBGK final : public CollisionModel<T,DESCRIPTOR> {
private:
    /// Parameters to pass to cuda kernels
    CollisionParamsBGKWrapper<T> _params;

public:
    /// Constructor
    explicit CollisionBGK(T omegaS);

    /// Destructor
    ~CollisionBGK();

    void prepareKernelParams(const CollisionParamsBGK<T>& CollisionParamsBGK) override;
    void doCollision(
        T* distribution,
        std::pair<unsigned int, unsigned int> numBlocks,
        std::pair<unsigned int, unsigned int> threadsPerBlock
    ) override;
    void printParameters();
};

/******************************************************/
/***** Derived class 02: CHM Collision parameters *****/
/******************************************************/
template<typename T,typename DESCRIPTOR>
class CollisionCHM final : public CollisionModel<T,DESCRIPTOR> { // only implemented for D2Q9 lattices
private:
    /// Relaxation parameter associated with bulk viscosity
    const T _omegaBulk;

    /// Parameters to pass to cuda kernels
    CollisionParamsCHMWrapper<T> _params;

public:
    /// Constructor
    CollisionCHM(T omegaS, T omegaB);

    /// Destructor
    ~CollisionCHM();

    T getOmegaBulk() const;
    void prepareKernelParams(const CollisionParamsBGK<T>& CollisionParamsBGK) override;
    void doCollision(
        T* distribution,
        std::pair<unsigned int, unsigned int> numBlocks,
        std::pair<unsigned int, unsigned int> threadsPerBlock
    ) override;
    void printParameters() override;
};

#include "collisionModel.hh"

#endif // COLLISION_MODEL_H
