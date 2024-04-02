#ifndef LB_COLLISION_MODEL_HH
#define LB_COLLISION_MODEL_HH

#include <cuda_runtime.h>

#include "lbModel.h"
#include "collisionModel.h"
#include "cuda/cudaConstants.cuh"
#include "cuda/cudaKernels.cuh"
#include "cuda/cudaErrorHandler.cuh"

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
CollisionModel<T>::CollisionModel(T omega) : omegaShear(omega) {}

template<typename T>
T CollisionModel<T>::getOmegaShear() const {
    return this->omegaShear;
}


/*************************************************/
/***** Derived class 01: BGK Collision model *****/
/*************************************************/
template<typename T>
CollisionBGK<T>::CollisionBGK(T omegaS) : CollisionModel<T>(omegaS) {}

template<typename T>
CollisionBGK<T>::~CollisionBGK() {}

template<typename T>
void CollisionBGK<T>::prepareKernelParams(const LBParams<T>& lbParams) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        lbParams.D,
        lbParams.Nx,
        lbParams.Ny,
        lbParams.Q,
        lbParams.LATTICE_VELOCITIES,
        lbParams.LATTICE_WEIGHTS,
        this->omegaShear
    );
}

template<typename T>
void CollisionBGK<T>::doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) {
    dim3 blockSize(threadsPerBlock.first, threadsPerBlock.second);
    dim3 gridSize(numBlocks.first, numBlocks.first);
    
    //doCollisionBGKCaller(distribution, deviceParams, gridSize, blockSize);  
}

template<typename T>
void CollisionBGK<T>::print() {
    std::cout << "============= Collision Model: BGK ===============" << std::endl;
    std::cout << "== Omega shear:" << this->getOmegaShear() << "\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}


/*************************************************/
/***** Derived class 02: CHM Collision model *****/
/*************************************************/
template<typename T>
CollisionCHM<T>::CollisionCHM(T omegaS, T omegaB) : CollisionModel<T>(omegaS), omegaBulk(omegaB) {}

template<typename T>
CollisionCHM<T>::~CollisionCHM() {}

template<typename T>
T CollisionCHM<T>::getOmegaBulk() const {
    return this->omegaBulk;
}

template<typename T>
void CollisionCHM<T>::prepareKernelParams(const LBParams<T>& lbParams) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        lbParams.D,
        lbParams.Nx,
        lbParams.Ny,
        lbParams.Q,
        lbParams.LATTICE_VELOCITIES,
        lbParams.LATTICE_WEIGHTS,
        this->omegaShear,
        this->omegaBulk
    );
}

template<typename T>
void CollisionCHM<T>::doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) {
    dim3 blockSize(threadsPerBlock.first, threadsPerBlock.second);
    dim3 gridSize(numBlocks.first, numBlocks.first);

    doCollisionCHMCaller(distribution, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T>
void CollisionCHM<T>::print() {
    std::cout << "============= Collision Model: CHM ===============" << std::endl;
    std::cout << "== Omega shear:" << this->getOmegaShear() << "\t\t\t\t==" << std::endl;
    std::cout << "== Omega bulk:" << this->getOmegaBulk() << "\t\t\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}


#endif // LB_COLLISION_MODEL_HH
