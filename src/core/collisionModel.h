#ifndef COLLISION_MODEL_H
#define COLLISION_MODEL_H

#include <stdio.h>
#include <vector>

#include "kernelParameters.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
class CollisionModel {
protected:
    /// Relaxation parameter associated with shear viscosity
    const T omegaShear;
public:
    /// Constructor
    CollisionModel(T omega);

    /// Destructor
    virtual ~CollisionModel() = default;

    T getOmegaShear() const;
    virtual void prepareKernelParams(LBParams<T>* lbParams) = 0;
    virtual void copyKernelParamsToDevice() = 0;
    virtual void doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) = 0;
    virtual void print() = 0;
};



/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
class CollisionBGK : public CollisionModel<T> {
private:
    /// Parameters to pass to cuda kernels
    CollisionParamsBGK<T> hostParams;
    CollisionParamsBGK<T>* deviceParams = nullptr;
public:
    /// Destructor
    virtual ~CollisionBGK();

    virtual void prepareKernelParams(LBParams<T>* lbParams);
    virtual void copyKernelParamsToDevice();
    virtual void doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) override;
    virtual void print();
};

template<typename T>
class CollisionCHM : public CollisionModel<T> { // only implemented for D2Q9 lattices
private:
    /// Relaxation parameter associated with bulk viscosity
    const T omegaBulk;
    /// Parameters to pass to cuda kernels
    CollisionParamsCHM<T> hostParams;
    CollisionParamsCHM<T>* deviceParams = nullptr;
public:
    /// Constructor
    CollisionCHM(T omegaS, T omegaB);

    /// Destructor
    virtual ~CollisionCHM();

    T getOmegaBulk() const;
    virtual void prepareKernelParams(LBParams<T>* lbParams);
    virtual void copyKernelParamsToDevice();
    virtual void doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) override;
    virtual void print() override;
};

#include "collisionModel.hh"

#endif // COLLISION_MODEL_H
