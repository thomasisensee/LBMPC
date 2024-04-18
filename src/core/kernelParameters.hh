#ifndef KERNEL_PARAMETERS_HH
#define KERNEL_PARAMETERS_HH

#include <iostream>

#include "cuda/cudaErrorHandler.cuh"

/**********************************/
/***** Base class (templated) *****/
/**********************************/
template<typename T, typename ParamsType>
ParamsWrapper<T, ParamsType>::ParamsWrapper(
        unsigned int nx,
        unsigned int ny
    ) {
    setValues(nx, ny);
}

template<typename T, typename ParamsType>
ParamsWrapper<T, ParamsType>::~ParamsWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::setValues(
        unsigned int nx,
        unsigned int ny
    ) {
    this->_hostParams.Nx = nx;
    this->_hostParams.Ny = ny;

    allocateAndCopyToDevice();
}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::allocateAndCopyToDevice() {
    // Cleanup device resources that may have been previously allocated
    cleanupDevice();

    // Prepare the host-side copy of Params with device pointers
    BaseParams paramsTemp = this->_hostParams; // Use a temporary host copy

    // Allocate memory for the Params struct on the device if not already allocated
    if (this->_deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->_deviceParams), sizeof(BaseParams)));
    }

    // Copy the prepared Params (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->_deviceParams, &paramsTemp, sizeof(BaseParams), cudaMemcpyHostToDevice));
}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::cleanupHost() {
    // Nothing to clean up for this derived class
}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::cleanupDevice() {
    // If _deviceParams was already freed and set to nullptr, return
    if(this->getDeviceParams() == nullptr) { return; }

    // Free the _deviceParams struct itself
    cudaFree(this->_deviceParams);
    this->_deviceParams = nullptr; // Ensure the pointer is marked as freed
}

template<typename T, typename ParamsType>
const ParamsType& ParamsWrapper<T, ParamsType>::getHostParams() const {
    return _hostParams;
}

template<typename T, typename ParamsType>
ParamsType* ParamsWrapper<T, ParamsType>::getDeviceParams() {
    return _deviceParams;
}

/*****************************************/
/***** Derived class 01:  BaseParams *****/
/*****************************************/
template<typename T>
BaseParamsWrapper<T>::BaseParamsWrapper(
        unsigned int nx,
        unsigned int ny
    ) : ParamsWrapper<T, BaseParams>(nx, ny) {
        ParamsWrapper<T, BaseParams>::allocateAndCopyToDevice();
    }

template<typename T>
void BaseParamsWrapper<T>::setValues(
        unsigned int nx,
        unsigned int ny
    ) {
    // Clean up existing data
    ParamsWrapper<T, BaseParams>::setValues(nx, ny);
}

/************************************************/
/***** Derived class 02: CollisionParamsBGK *****/
/************************************************/
template<typename T>
CollisionParamsBGKWrapper<T>::CollisionParamsBGKWrapper(
        unsigned int nx,
        unsigned int ny,
        T omegaShear
    ) {
    setValues(nx, ny, omegaShear);
}

template<typename T>
void CollisionParamsBGKWrapper<T>::setValues(
        unsigned int nx,
        unsigned int ny,
        T omegaShear
    ) {
    // Clean up existing data
    ParamsWrapper<T, CollisionParamsBGK<T>>::cleanupHost();

    // Assign new deep copies
    this->_hostParams.Nx                 = nx;
    this->_hostParams.Ny                 = ny;
    this->_hostParams.omegaShear         = omegaShear;

    ParamsWrapper<T, CollisionParamsBGK<T>>::allocateAndCopyToDevice();
}

/************************************************/
/***** Derived class 02: CollisionParamsCHM *****/
/************************************************/
template<typename T>
CollisionParamsCHMWrapper<T>::CollisionParamsCHMWrapper(
        unsigned int nx,
        unsigned int ny,
        T omegaShear,
        T omegaBulk
    ) {
    setValues(nx, ny, omegaShear, omegaBulk);
}

template<typename T>
void CollisionParamsCHMWrapper<T>::setValues(
        unsigned int nx,
        unsigned int ny,
        T omegaShear,
        T omegaBulk
    ) {
    // Clean up existing data
    ParamsWrapper<T, CollisionParamsCHM<T>>::cleanupHost();

    // Assign new deep copies
    this->_hostParams.Nx            = nx;
    this->_hostParams.Ny            = ny;
    this->_hostParams.omegaShear    = omegaShear;
    this->_hostParams.omegaBulk     = omegaBulk;

    ParamsWrapper<T, CollisionParamsCHM<T>>::allocateAndCopyToDevice();
}

/********************************************/
/***** Derived class 04: BoundaryParams *****/
/********************************************/
template<typename T>
BoundaryParamsWrapper<T>::BoundaryParamsWrapper() {}

template<typename T>
BoundaryParamsWrapper<T>::BoundaryParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        const T* WALL_VELOCITY,
        BoundaryLocation location
    ) {
    setValues(nx, ny, WALL_VELOCITY, location);
}

template<typename T>
BoundaryParamsWrapper<T>::~BoundaryParamsWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T>
void BoundaryParamsWrapper<T>::setValues(
        unsigned int nx,
        unsigned int ny,
        const T* WALL_VELOCITY,
        BoundaryLocation location
    ) {
    // Clean up existing data
    cleanupHost();

    // Assign new deep copies
    this->_hostParams.Nx         = nx;
    this->_hostParams.Ny         = ny;
    this->_hostParams.location   = location;

    if (WALL_VELOCITY != nullptr) {
        this->_hostParams.WALL_VELOCITY = new T[2];
        std::copy(WALL_VELOCITY, WALL_VELOCITY + 2, this->_hostParams.WALL_VELOCITY);
    } else {
        this->_hostParams.WALL_VELOCITY = nullptr; // Ensure it is nullptr if not used
    }

    allocateAndCopyToDevice();
}

template<typename T>
void BoundaryParamsWrapper<T>::setWallVelocity(const std::vector<T>& wallVelocity) {
    // Clean up existing data
    delete[] this->_hostParams.WALL_VELOCITY;

    // Store the size of the wallVelocity vector
    _D = wallVelocity.size();
    
    // Assign new deep copies
    this->_hostParams.WALL_VELOCITY = new T[_D];
    std::copy(wallVelocity.data(), wallVelocity.data() + _D, this->_hostParams.WALL_VELOCITY);

    allocateAndCopyToDevice();
}

template<typename T>
void BoundaryParamsWrapper<T>::allocateAndCopyToDevice() {
    // Cleanup device resources that may have been previously allocated
    cleanupDevice();

    // Allocate device memory for wall velocity and copy data, if wall velocity is specified, i.e., not nullptr
    T* deviceWallVelocity = nullptr;
    if (this->_hostParams.WALL_VELOCITY != nullptr) {
        size_t sizeWallVelocity = _D * sizeof(T);
        cudaErrorCheck(cudaMalloc(&deviceWallVelocity, sizeWallVelocity));
        cudaErrorCheck(cudaMemcpy(deviceWallVelocity, this->_hostParams.WALL_VELOCITY, sizeWallVelocity, cudaMemcpyHostToDevice));
    }

    // Prepare the host-side copy of Params with device pointers
    BoundaryParams<T> paramsTemp    = this->_hostParams; // Use a temporary host copy
    paramsTemp.WALL_VELOCITY        = deviceWallVelocity;

    // Allocate memory for the Params struct on the device if not already allocated
    if (this->_deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->_deviceParams), sizeof(BoundaryParams<T>)));
    }

    // Copy the prepared Params (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->_deviceParams, &paramsTemp, sizeof(BoundaryParams<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void BoundaryParamsWrapper<T>::cleanupHost() {
    if (this->_hostParams.WALL_VELOCITY != nullptr) {
        delete[] this->_hostParams.WALL_VELOCITY;
        this->_hostParams.WALL_VELOCITY = nullptr;
    }
}

template<typename T>
void BoundaryParamsWrapper<T>::cleanupDevice() {
    // If _deviceParams was already freed and set to nullptr, return
    if(this->getDeviceParams() == nullptr) { return; }

    // Temporary host-side BoundaryParams object to copy device pointers back to and then cudaFree them
    BoundaryParams<T> paramsTemp;

    // Copy _deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->_deviceParams, sizeof(BoundaryParams<T>), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.WALL_VELOCITY);

    // Finally, free the _deviceParams struct itself
    cudaFree(this->_deviceParams);
    this->_deviceParams = nullptr; // Ensure the pointer is marked as freed
}

#endif // KERNEL_PARAMETERS_HH