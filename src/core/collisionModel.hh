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
void CollisionModel<T>::setOmegaShear(T omegaShear) {
    this->omegaShear = omegaShear;
}

template<typename T>
T CollisionModel<T>::getOmegaShear() const {
    return this->omegaShear;
}


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
void CollisionBGK<T>::prepareKernelParams(LBParams<T>* lbParams) {
    this->hostParams.D                  = lbParams->D;
    this->hostParams.Nx                 = lbParams->Nx;
    this->hostParams.Ny                 = lbParams->Ny;
    this->hostParams.Q                  = lbParams->Q;
    this->hostParams.LATTICE_VELOCITIES = lbParams->LATTICE_VELOCITIES;
    this->hostParams.LATTICE_WEIGHTS    = lbParams->LATTICE_WEIGHTS;    
    this->hostParams.omegaShear         = this->omegaShear;
}


template<typename T>
void CollisionBGK<T>::copyKernelParamsToDevice() {
    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->hostParams.Q * this->hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of LBParams with device pointers
    CollisionParamsBGK<T> paramsTemp = hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&deviceParams, sizeof(LBParams<T>)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(deviceParams, &paramsTemp, sizeof(LBParams<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void CollisionBGK<T>::doCollision(T* distribution) {

}

template<typename T>
void CollisionBGK<T>::print() {
    std::cout << "============= Collision Model: BGK ===============" << std::endl;
    std::cout << "== Omega shear:" << this->getOmegaShear() << "\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}

template<typename T>
CollisionCHM<T>::CollisionCHM(T omegaS, T omegaB) : CollisionModel<T>(omegaS), omegaBulk(omegaB) {}


template<typename T>
void CollisionCHM<T>::setOmegaBulk(T omegaBulk) {
    this->omegaBulk = omegaBulk;
}

template<typename T>
T CollisionCHM<T>::getOmegaBulk() const {
    return this->omegaBulk;
}

template<typename T>
void CollisionCHM<T>::prepareKernelParams(LBParams<T>* lbParams) {
    this->hostParams.D                  = lbParams->D;
    this->hostParams.Nx                 = lbParams->Nx;
    this->hostParams.Ny                 = lbParams->Ny;
    this->hostParams.Q                  = lbParams->Q;
    this->hostParams.LATTICE_VELOCITIES = lbParams->LATTICE_VELOCITIES;
    this->hostParams.LATTICE_WEIGHTS    = lbParams->LATTICE_WEIGHTS;    
    this->hostParams.omegaShear         = this->omegaShear;
    this->hostParams.omegaBulk          = this->omegaBulk;
}

template<typename T>
void CollisionCHM<T>::copyKernelParamsToDevice() {
    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->hostParams.Q * this->hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of LBParams with device pointers
    CollisionParamsCHM<T> paramsTemp = hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&deviceParams, sizeof(LBParams<T>)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(deviceParams, &paramsTemp, sizeof(LBParams<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void CollisionCHM<T>::doCollision(T* distribution) {
    dim3 threadsPerBlock(THREADS_PER_BLOCK_DIMENSION, THREADS_PER_BLOCK_DIMENSION);
    dim3 numBlocks((this->hostParams.Nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (this->hostParams.Ny + threadsPerBlock.y - 1) / threadsPerBlock.y);
    doCollisionCHMCaller(distribution, deviceParams, numBlocks, threadsPerBlock);
}

template<typename T>
void CollisionCHM<T>::print() {
    std::cout << "============= Collision Model: CHM ===============" << std::endl;
    std::cout << "== Omega shear:" << this->getOmegaShear() << "\t\t\t\t==" << std::endl;
    std::cout << "== Omega bulk:" << this->getOmegaBulk() << "\t\t\t\t\t==" << std::endl;
    std::cout << "==================================================\n" << std::endl;
}


#endif // LB_COLLISION_MODEL_HH
