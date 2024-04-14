#ifndef LB_COLLISION_MODEL_HH
#define LB_COLLISION_MODEL_HH

#include <cuda_runtime.h>

#include "collisionModel.h"
#include "cuda/cudaConstants.cuh"
#include "cuda/cudaKernels.cuh"

/**********************/
/***** Base class *****/
/**********************/
template<typename T,typename DESCRIPTOR>
CollisionModel<T,DESCRIPTOR>::CollisionModel(T omega) : _omegaShear(omega) {}

template<typename T,typename DESCRIPTOR>
T CollisionModel<T,DESCRIPTOR>::getOmegaShear() const {
    return this->_omegaShear;
}

/*************************************************/
/***** Derived class 01: BGK Collision model *****/
/*************************************************/
template<typename T,typename DESCRIPTOR>
CollisionBGK<T,DESCRIPTOR>::CollisionBGK(T omegaS) : CollisionModel<T,DESCRIPTOR>(omegaS) {}

template<typename T,typename DESCRIPTOR>
CollisionBGK<T,DESCRIPTOR>::~CollisionBGK() {}

template<typename T,typename DESCRIPTOR>
void CollisionBGK<T,DESCRIPTOR>::prepareKernelParams(const CollisionParamsBGK<T>& CollisionParamsBGK) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        CollisionParamsBGK.Nx,
        CollisionParamsBGK.Ny,
        this->_omegaShear
    );
}

template<typename T,typename DESCRIPTOR>
void CollisionBGK<T,DESCRIPTOR>::doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) {
    dim3 blockSize(threadsPerBlock.first, threadsPerBlock.second);
    dim3 gridSize(numBlocks.first, numBlocks.first);

    doCollisionBGKCaller<T,DESCRIPTOR>(distribution, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T,typename DESCRIPTOR>
void CollisionBGK<T,DESCRIPTOR>::printParameters() {
    std::cout << "============= Collision Model: BGK ===============" << std::endl;
    std::cout << "== Omega shear:\t" << this->getOmegaShear() << "\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}


/*************************************************/
/***** Derived class 02: CHM Collision model *****/
/*************************************************/
template<typename T,typename DESCRIPTOR>
CollisionCHM<T,DESCRIPTOR>::CollisionCHM(T omegaS, T omegaB) : CollisionModel<T,DESCRIPTOR>(omegaS), _omegaBulk(omegaB) {}

template<typename T,typename DESCRIPTOR>
CollisionCHM<T,DESCRIPTOR>::~CollisionCHM() {}

template<typename T,typename DESCRIPTOR>
T CollisionCHM<T,DESCRIPTOR>::getOmegaBulk() const {
    return this->_omegaBulk;
}

template<typename T,typename DESCRIPTOR>
void CollisionCHM<T,DESCRIPTOR>::prepareKernelParams(const CollisionParamsBGK<T>& CollisionParamsBGK) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        CollisionParamsBGK.Nx,
        CollisionParamsBGK.Ny,
        this->_omegaShear,
        this->_omegaBulk
    );
}

template<typename T,typename DESCRIPTOR>
void CollisionCHM<T,DESCRIPTOR>::doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) {
    dim3 blockSize(threadsPerBlock.first, threadsPerBlock.second);
    dim3 gridSize(numBlocks.first, numBlocks.first);

    doCollisionCHMCaller<T,DESCRIPTOR>(distribution, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T,typename DESCRIPTOR>
void CollisionCHM<T,DESCRIPTOR>::printParameters() {
    std::cout << "============= Collision Model: CHM ===============" << std::endl;
    std::cout << "== Omega shear:\t" << this->getOmegaShear() << "\t\t\t\t==" << std::endl;
    std::cout << "== Omega bulk:\t" << this->getOmegaBulk() << "\t\t\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}

#endif // LB_COLLISION_MODEL_HH