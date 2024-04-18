#ifndef KERNEL_PARAMETERS_HH
#define KERNEL_PARAMETERS_HH

#include <iostream>

#include "cuda/cudaErrorHandler.cuh"

/**********************************/
/***** Base class (templated) *****/
/**********************************/
template<typename T, typename ParamsType>
ParamsWrapper<T, ParamsType>::ParamsWrapper(unsigned int nx,unsigned int ny) {
    setValues(nx, ny);
    allocateAndCopyToDevice();
}

template<typename T, typename ParamsType>
ParamsWrapper<T, ParamsType>::~ParamsWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::setValues(unsigned int nx, unsigned int ny) {
    this->_hostParams.Nx = nx;
    this->_hostParams.Ny = ny;
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
    // No dynamically allocated memory in the base class
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

/****************************************/
/***** Derived class 01: BaseParams *****/
/****************************************/
template<typename T>
BaseParamsWrapper<T>::BaseParamsWrapper(unsigned int nx, unsigned int ny) : ParamsWrapper<T, BaseParams>(nx, ny) {}

template<typename T>
void BaseParamsWrapper<T>::setValues(unsigned int nx, unsigned int ny) {
    // Clean up existing data
    ParamsWrapper<T, BaseParams>::setValues(nx, ny);
    ParamsWrapper<T, BaseParams>::allocateAndCopyToDevice();
}

/************************************************/
/***** Derived class 02: CollisionParamsBGK *****/
/************************************************/
template<typename T>
CollisionParamsBGKWrapper<T>::CollisionParamsBGKWrapper(unsigned int nx, unsigned int ny, T omegaShear) {
    setValues(nx, ny, omegaShear);
}

template<typename T>
void CollisionParamsBGKWrapper<T>::setValues(unsigned int nx, unsigned int ny, T omegaShear) {
    // Clean up existing data
    ParamsWrapper<T, CollisionParamsBGK<T>>::cleanupHost();

    // Assign new deep copies
    ParamsWrapper<T, CollisionParamsBGK<T>>::setValues(nx, ny);
    this->_hostParams.omegaShear = omegaShear;

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
    ParamsWrapper<T, CollisionParamsCHM<T>>::setValues(nx, ny);
    this->_hostParams.omegaShear    = omegaShear;
    this->_hostParams.omegaBulk     = omegaBulk;

    ParamsWrapper<T, CollisionParamsCHM<T>>::allocateAndCopyToDevice();
}

/********************************************/
/***** Derived class 04: BoundaryParams *****/
/********************************************/
template<typename T>
BoundaryParamsWrapper<T>::BoundaryParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    ) {
    setValues(nx, ny, location);
}

template<typename T>
void BoundaryParamsWrapper<T>::setValues(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    ) {
    // Clean up existing data
    ParamsWrapper<T, BoundaryParams>::cleanupHost();

    // Assign new deep copies
    ParamsWrapper<T, BoundaryParams>::setValues(nx, ny);
    this->_hostParams.location   = location;

    ParamsWrapper<T, BoundaryParams>::allocateAndCopyToDevice();
}

/********************************************/
/***** Derived class 05: PeriodicParams *****/
/********************************************/
template<typename T>
PeriodicParamsWrapper<T>::PeriodicParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    ) {
    setValues(nx, ny, location);
}

template<typename T>
void PeriodicParamsWrapper<T>::setValues(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    ) {
    // Clean up existing data
    ParamsWrapper<T, PeriodicParams>::cleanupHost();

    // Assign new deep copies
    ParamsWrapper<T, PeriodicParams>::setValues(nx, ny);
    this->_hostParams.location   = location;

    ParamsWrapper<T, PeriodicParams>::allocateAndCopyToDevice();
}

/********************************************/
/***** Derived class 05: PeriodicParams *****/
/********************************************/
template<typename T>
BounceBackParamsWrapper<T>::BounceBackParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    ) {
    setValues(nx, ny, location);
}

template<typename T>
void BounceBackParamsWrapper<T>::setValues(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location
    ) {
    // Clean up existing data
    ParamsWrapper<T, BounceBackParams>::cleanupHost();

    // Assign new deep copies
    ParamsWrapper<T, PeriodicParams>::setValues(nx, ny);
    this->_hostParams.location   = location;

    ParamsWrapper<T, BounceBackParams>::allocateAndCopyToDevice();
}

/**********************************************/
/***** Derived class 06: MovingWallParams *****/
/**********************************************/
template<typename T>
MovingWallParamsWrapper<T>::MovingWallParamsWrapper(
        unsigned int nx,
        unsigned int ny,
        BoundaryLocation location,
        const T* wallVelocity
    ) {
    setValues(nx, ny, location, wallVelocity);
}

template<typename T>
void MovingWallParamsWrapper<T>::setWallVelocity(const std::vector<T>& wallVelocity) {
    // Clean up existing data
    delete[] this->_hostParams.wallVelocity;

    // Store the size of the wallVelocity vector
    _D = wallVelocity.size();
    
    // Assign new deep copies
    this->_hostParams.wallVelocity = new T[_D];
    std::copy(wallVelocity.data(), wallVelocity.data() + _D, this->_hostParams.wallVelocity);

    allocateAndCopyToDevice();
}

template<typename T>
void MovingWallParamsWrapper<T>::allocateAndCopyToDevice() {
    // Cleanup device resources that may have been previously allocated
    cleanupDevice();

    // Allocate device memory for wall velocity and copy data, if wall velocity is specified, i.e., not nullptr
    T* deviceWallVelocity = nullptr;
    if (this->_hostParams.wallVelocity != nullptr) {
        size_t sizeWallVelocity = _D * sizeof(T);
        cudaErrorCheck(cudaMalloc(&deviceWallVelocity, sizeWallVelocity));
        cudaErrorCheck(cudaMemcpy(deviceWallVelocity, this->_hostParams.wallVelocity, sizeWallVelocity, cudaMemcpyHostToDevice));
    }

    // Prepare the host-side copy of Params with device pointers
    MovingWallParams<T> paramsTemp  = this->_hostParams; // Use a temporary host copy
    paramsTemp.wallVelocity        = deviceWallVelocity;

    // Allocate memory for the Params struct on the device if not already allocated
    if (this->_deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->_deviceParams), sizeof(MovingWallParams<T>)));
    }

    // Copy the prepared Params (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->_deviceParams, &paramsTemp, sizeof(MovingWallParams<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void MovingWallParamsWrapper<T>::cleanupHost() {
    if (this->_hostParams.wallVelocity != nullptr) {
        delete[] this->_hostParams.wallVelocity;
        this->_hostParams.wallVelocity = nullptr;
    }
}

template<typename T>
void MovingWallParamsWrapper<T>::cleanupDevice() {
    // If _deviceParams was already freed and set to nullptr, return
    if(this->getDeviceParams() == nullptr) { return; }

    // Temporary host-side BoundaryParams object to copy device pointers back to and then cudaFree them
    MovingWallParams<T> paramsTemp;

    // Copy _deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->_deviceParams, sizeof(MovingWallParams<T>), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.wallVelocity);

    // Finally, free the _deviceParams struct itself
    cudaFree(this->_deviceParams);
    this->_deviceParams = nullptr; // Ensure the pointer is marked as freed
}

#endif // KERNEL_PARAMETERS_HH