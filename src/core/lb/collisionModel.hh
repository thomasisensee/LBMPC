#ifndef LB_COLLISION_MODEL_HH
#define LB_COLLISION_MODEL_HH

#include <cuda_runtime.h>

#include "collisionModel.h"
#include "cuda/cudaConstants.cuh"
#include "cuda/cudaKernels.cuh"

/**********************/
/***** Base class *****/
/**********************/
template<typename T, unsigned int D, unsigned int Q>
CollisionModel<T,D,Q>::CollisionModel(T omega) : _omegaShear(omega) {}

template<typename T, unsigned int D, unsigned int Q>
T CollisionModel<T,D,Q>::getOmegaShear() const {
    return this->_omegaShear;
}

/*************************************************/
/***** Derived class 01: BGK Collision model *****/
/*************************************************/
template<typename T, unsigned int D, unsigned int Q>
CollisionBGK<T,D,Q>::CollisionBGK(T omegaS) : CollisionModel<T,D,Q>(omegaS) {}

template<typename T, unsigned int D, unsigned int Q>
CollisionBGK<T,D,Q>::~CollisionBGK() {}

template<typename T, unsigned int D, unsigned int Q>
void CollisionBGK<T,D,Q>::prepareKernelParams(const CollisionParamsBGK<T>& CollisionParamsBGK) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        CollisionParamsBGK.Nx,
        CollisionParamsBGK.Ny,
        this->_omegaShear
    );
}

template<typename T, unsigned int D, unsigned int Q>
void CollisionBGK<T,D,Q>::doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) {
    dim3 blockSize(threadsPerBlock.first, threadsPerBlock.second);
    dim3 gridSize(numBlocks.first, numBlocks.first);

    doCollisionBGKCaller<T,D,Q>(distribution, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T, unsigned int D, unsigned int Q>
void CollisionBGK<T,D,Q>::printParameters() {
    std::cout << "============= Collision Model: BGK ===============" << std::endl;
    std::cout << "== Omega shear:\t" << this->getOmegaShear() << "\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}


/*************************************************/
/***** Derived class 02: CHM Collision model *****/
/*************************************************/
template<typename T, unsigned int D, unsigned int Q>
CollisionCHM<T,D,Q>::CollisionCHM(T omegaS, T omegaB) : CollisionModel<T,D,Q>(omegaS), _omegaBulk(omegaB) {}

template<typename T, unsigned int D, unsigned int Q>
CollisionCHM<T,D,Q>::~CollisionCHM() {}

template<typename T, unsigned int D, unsigned int Q>
T CollisionCHM<T,D,Q>::getOmegaBulk() const {
    return this->_omegaBulk;
}

template<typename T, unsigned int D, unsigned int Q>
void CollisionCHM<T,D,Q>::prepareKernelParams(const CollisionParamsBGK<T>& CollisionParamsBGK) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        CollisionParamsBGK.Nx,
        CollisionParamsBGK.Ny,
        this->_omegaShear,
        this->_omegaBulk
    );
}

template<typename T, unsigned int D, unsigned int Q>
void CollisionCHM<T,D,Q>::doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) {
    dim3 blockSize(threadsPerBlock.first, threadsPerBlock.second);
    dim3 gridSize(numBlocks.first, numBlocks.first);

    doCollisionCHMCaller<T,D,Q>(distribution, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T, unsigned int D, unsigned int Q>
void CollisionCHM<T,D,Q>::printParameters() {
    std::cout << "============= Collision Model: CHM ===============" << std::endl;
    std::cout << "== Omega shear:\t" << this->getOmegaShear() << "\t\t\t\t==" << std::endl;
    std::cout << "== Omega bulk:\t" << this->getOmegaBulk() << "\t\t\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}

#endif // LB_COLLISION_MODEL_HH