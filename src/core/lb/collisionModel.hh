#ifndef LB_COLLISION_MODEL_HH
#define LB_COLLISION_MODEL_HH

#include <cuda_runtime.h>

#include "collisionModel.h"
#include "cuda/cudaConstants.cuh"
#include "cuda/cudaKernels.cuh"

/**********************/
/***** Base class *****/
/**********************/
template<typename T, typename LatticeDescriptor>
CollisionModel<T, LatticeDescriptor>::CollisionModel(T omega) : _omegaShear(omega) {}

template<typename T, typename LatticeDescriptor>
T CollisionModel<T, LatticeDescriptor>::getOmegaShear() const {
    return this->_omegaShear;
}

/*************************************************/
/***** Derived class 01: BGK Collision model *****/
/*************************************************/
template<typename T, typename LatticeDescriptor>
CollisionBGK<T, LatticeDescriptor>::CollisionBGK(T omegaS) : CollisionModel<T, LatticeDescriptor>(omegaS) {}

template<typename T, typename LatticeDescriptor>
CollisionBGK<T, LatticeDescriptor>::~CollisionBGK() {}

template<typename T, typename LatticeDescriptor>
void CollisionBGK<T, LatticeDescriptor>::prepareKernelParams(const LBParams<T>& lbParams) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        lbParams.Nx,
        lbParams.Ny,
        this->_omegaShear
    );
}

template<typename T, typename LatticeDescriptor>
void CollisionBGK<T, LatticeDescriptor>::doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) {
    dim3 blockSize(threadsPerBlock.first, threadsPerBlock.second);
    dim3 gridSize(numBlocks.first, numBlocks.first);

    doCollisionBGKCaller<T, LatticeDescriptor>(distribution, _params.getDeviceParams(), gridSize, blockSize);
    //testKernelCaller(distribution, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T, typename LatticeDescriptor>
void CollisionBGK<T, LatticeDescriptor>::printParameters() {
    std::cout << "============= Collision Model: BGK ===============" << std::endl;
    std::cout << "== Omega shear:\t" << this->getOmegaShear() << "\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}


/*************************************************/
/***** Derived class 02: CHM Collision model *****/
/*************************************************/
template<typename T, typename LatticeDescriptor>
CollisionCHM<T, LatticeDescriptor>::CollisionCHM(T omegaS, T omegaB) : CollisionModel<T, LatticeDescriptor>(omegaS), _omegaBulk(omegaB) {}

template<typename T, typename LatticeDescriptor>
CollisionCHM<T, LatticeDescriptor>::~CollisionCHM() {}

template<typename T, typename LatticeDescriptor>
T CollisionCHM<T, LatticeDescriptor>::getOmegaBulk() const {
    return this->_omegaBulk;
}

template<typename T, typename LatticeDescriptor>
void CollisionCHM<T, LatticeDescriptor>::prepareKernelParams(const LBParams<T>& lbParams) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        lbParams.Nx,
        lbParams.Ny,
        this->_omegaShear,
        this->_omegaBulk
    );
}

template<typename T, typename LatticeDescriptor>
void CollisionCHM<T, LatticeDescriptor>::doCollision(T* distribution, std::pair<unsigned int, unsigned int> numBlocks, std::pair<unsigned int, unsigned int> threadsPerBlock) {
    dim3 blockSize(threadsPerBlock.first, threadsPerBlock.second);
    dim3 gridSize(numBlocks.first, numBlocks.first);

    doCollisionCHMCaller<T, LatticeDescriptor>(distribution, _params.getDeviceParams(), gridSize, blockSize);
    //testKernelCaller(distribution, _params.getDeviceParams(), gridSize, blockSize);
}

template<typename T, typename LatticeDescriptor>
void CollisionCHM<T, LatticeDescriptor>::printParameters() {
    std::cout << "============= Collision Model: CHM ===============" << std::endl;
    std::cout << "== Omega shear:\t" << this->getOmegaShear() << "\t\t\t\t==" << std::endl;
    std::cout << "== Omega bulk:\t" << this->getOmegaBulk() << "\t\t\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}

#endif // LB_COLLISION_MODEL_HH
