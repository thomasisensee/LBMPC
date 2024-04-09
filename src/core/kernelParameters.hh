#ifndef KERNEL_PARAMETERS_HH
#define KERNEL_PARAMETERS_HH

#include <iostream>     // For std::cout

#include "cuda/cudaErrorHandler.cuh"

/**********************************/
/***** Base class (templated) *****/
/**********************************/
template<typename T, typename ParamsType>
ParamsWrapper<T, ParamsType>::ParamsWrapper() {}

template<typename T, typename ParamsType>
ParamsWrapper<T, ParamsType>::ParamsWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny
    ) {
    setValues(dim, nx, ny);
}

template<typename T, typename ParamsType>
ParamsWrapper<T, ParamsType>::~ParamsWrapper() {}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny
    ) {
    this->_hostParams.D                  = dim;
    this->_hostParams.Nx                 = nx;
    this->_hostParams.Ny                 = ny;

    allocateAndCopyToDevice();
}

template<typename T, typename ParamsType>
const ParamsType& ParamsWrapper<T, ParamsType>::getHostParams() const {
    return _hostParams;
}

template<typename T, typename ParamsType>
ParamsType* ParamsWrapper<T, ParamsType>::getDeviceParams() {
    return _deviceParams;
}

/**************************************/
/***** Derived class 01: LBParams *****/
/**************************************/
template<typename T>
LBParamsWrapper<T>::LBParamsWrapper() {}

template<typename T>
LBParamsWrapper<T>::LBParamsWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS
    ) {
    setValues(dim, nx, ny, q, LATTICE_VELOCITIES, LATTICE_WEIGHTS);
}

template<typename T>
LBParamsWrapper<T>::~LBParamsWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T>
void LBParamsWrapper<T>::setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS
    ) {
    // Clean up existing data
    cleanupHost();

    // Assign new deep copies
    this->_hostParams.D                  = dim;
    this->_hostParams.Nx                 = nx;
    this->_hostParams.Ny                 = ny;
    this->_hostParams.Q                  = q;

    this->_hostParams.LATTICE_VELOCITIES = new int[q * dim];
    std::copy(LATTICE_VELOCITIES, LATTICE_VELOCITIES + (q * dim), this->_hostParams.LATTICE_VELOCITIES);

    this->_hostParams.LATTICE_WEIGHTS = new T[q];
    std::copy(LATTICE_WEIGHTS, LATTICE_WEIGHTS + q, this->_hostParams.LATTICE_WEIGHTS);

    allocateAndCopyToDevice();
}

template<typename T>
void LBParamsWrapper<T>::allocateAndCopyToDevice() {
    // Cleanup device resources that may have been previously allocated
    cleanupDevice();

    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->_hostParams.Q * this->_hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, this->_hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->_hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, this->_hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of Params with device pointers
    LBParams<T> paramsTemp = this->_hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the Params struct on the device if not already allocated
    if (this->_deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->_deviceParams), sizeof(LBParams<T>)));
    }

    // Copy the prepared Params (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->_deviceParams, &paramsTemp, sizeof(LBParams<T>), cudaMemcpyHostToDevice));
}


template<typename T>
void LBParamsWrapper<T>::cleanupHost() {
    if (this->_hostParams.LATTICE_VELOCITIES != nullptr) {
        delete[] this->_hostParams.LATTICE_VELOCITIES;
        this->_hostParams.LATTICE_VELOCITIES = nullptr;
    }
    if (this->_hostParams.LATTICE_WEIGHTS != nullptr) {
        delete[] this->_hostParams.LATTICE_WEIGHTS;
        this->_hostParams.LATTICE_WEIGHTS = nullptr;
    }
}

template<typename T>
void LBParamsWrapper<T>::cleanupDevice() {
    // If _deviceParams was already freed and set to nullptr, return
    if(this->getDeviceParams() == nullptr) { return; }

    // Assuming _deviceParams has been properly allocated and initialized
    LBParams<T> paramsTemp;

    // Copy _deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->getDeviceParams(), sizeof(LBParams<T>), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.LATTICE_VELOCITIES);
    cudaFree(paramsTemp.LATTICE_WEIGHTS);

    // Finally, free the _deviceParams struct itself
    cudaFree(this->_deviceParams);
    this->_deviceParams = nullptr; // Ensure the pointer is marked as freed
}

/************************************************/
/***** Derived class 02: CollisionParamsBGK *****/
/************************************************/
template<typename T>
CollisionParamsBGKWrapper<T>::CollisionParamsBGKWrapper() {}

template<typename T>
CollisionParamsBGKWrapper<T>::CollisionParamsBGKWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
        T omegaShear
    ) {
    setValues(dim, nx, ny, q, LATTICE_VELOCITIES, LATTICE_WEIGHTS, omegaShear);
}

template<typename T>
CollisionParamsBGKWrapper<T>::~CollisionParamsBGKWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T>
void CollisionParamsBGKWrapper<T>::setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
        T omegaShear
    ) {
    // Clean up existing data
    cleanupHost();

    // Assign new deep copies
    this->_hostParams.D                  = dim;
    this->_hostParams.Nx                 = nx;
    this->_hostParams.Ny                 = ny;
    this->_hostParams.Q                  = q;
    this->_hostParams.omegaShear         = omegaShear;

    this->_hostParams.LATTICE_VELOCITIES = new int[q * dim];
    std::copy(LATTICE_VELOCITIES, LATTICE_VELOCITIES + (q * dim), this->_hostParams.LATTICE_VELOCITIES);

    this->_hostParams.LATTICE_WEIGHTS = new T[q];
    std::copy(LATTICE_WEIGHTS, LATTICE_WEIGHTS + q, this->_hostParams.LATTICE_WEIGHTS);

    allocateAndCopyToDevice();
}

template<typename T>
void CollisionParamsBGKWrapper<T>::allocateAndCopyToDevice() {
    // Cleanup device resources that may have been previously allocated
    cleanupDevice();

    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->_hostParams.Q * this->_hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, this->_hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->_hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, this->_hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of Params with device pointers
    CollisionParamsBGK<T> paramsTemp = this->_hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the Params struct on the device if not already allocated
    if (this->_deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->_deviceParams), sizeof(CollisionParamsBGK<T>)));
    }

    // Copy the prepared Params (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->_deviceParams, &paramsTemp, sizeof(CollisionParamsBGK<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void CollisionParamsBGKWrapper<T>::cleanupHost() {
    if (this->_hostParams.LATTICE_VELOCITIES != nullptr) {
        delete[] this->_hostParams.LATTICE_VELOCITIES;
        this->_hostParams.LATTICE_VELOCITIES = nullptr;
    }
    if (this->_hostParams.LATTICE_WEIGHTS != nullptr) {
        delete[] this->_hostParams.LATTICE_WEIGHTS;
        this->_hostParams.LATTICE_WEIGHTS = nullptr;
    }
}

template<typename T>
void CollisionParamsBGKWrapper<T>::cleanupDevice() {
    // If _deviceParams was already freed and set to nullptr, return
    if(this->getDeviceParams() == nullptr) { return; }

    // Assuming _deviceParams has been properly allocated and initialized
    CollisionParamsBGK<T> paramsTemp;

    // Copy _deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->getDeviceParams(), sizeof(CollisionParamsBGK<T>), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.LATTICE_VELOCITIES);
    cudaFree(paramsTemp.LATTICE_WEIGHTS);

    // Finally, free the _deviceParams struct itself
    cudaFree(this->_deviceParams);
    this->_deviceParams = nullptr; // Ensure the pointer is marked as freed
}

/************************************************/
/***** Derived class 02: CollisionParamsCHM *****/
/************************************************/
template<typename T>
CollisionParamsCHMWrapper<T>::CollisionParamsCHMWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T>
CollisionParamsCHMWrapper<T>::CollisionParamsCHMWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
        T omegaShear,
        T omegaBulk
    ) {
    setValues(dim, nx, ny, q, LATTICE_VELOCITIES, LATTICE_WEIGHTS, omegaShear, omegaBulk);
}

template<typename T>
CollisionParamsCHMWrapper<T>::~CollisionParamsCHMWrapper() {}

template<typename T>
void CollisionParamsCHMWrapper<T>::setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
        T omegaShear,
        T omegaBulk
    ) {
    // Clean up existing data
    cleanupHost();

    // Assign new deep copies
    this->_hostParams.D                  = dim;
    this->_hostParams.Nx                 = nx;
    this->_hostParams.Ny                 = ny;
    this->_hostParams.Q                  = q;
    this->_hostParams.omegaShear         = omegaShear;
    this->_hostParams.omegaBulk          = omegaBulk;

    this->_hostParams.LATTICE_VELOCITIES = new int[q * dim];
    std::copy(LATTICE_VELOCITIES, LATTICE_VELOCITIES + (q * dim), this->_hostParams.LATTICE_VELOCITIES);

    this->_hostParams.LATTICE_WEIGHTS = new T[q];
    std::copy(LATTICE_WEIGHTS, LATTICE_WEIGHTS + q, this->_hostParams.LATTICE_WEIGHTS);

    allocateAndCopyToDevice();
}

template<typename T>
void CollisionParamsCHMWrapper<T>::allocateAndCopyToDevice() {
    // Cleanup device resources that may have been previously allocated
    cleanupDevice();

    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->_hostParams.Q * this->_hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, this->_hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->_hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, this->_hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of Params with device pointers
    CollisionParamsCHM<T> paramsTemp = this->_hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the Params struct on the device if not already allocated
    if (this->_deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->_deviceParams), sizeof(CollisionParamsCHM<T>)));
    }

    // Copy the prepared Params (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->_deviceParams, &paramsTemp, sizeof(CollisionParamsCHM<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void CollisionParamsCHMWrapper<T>::cleanupHost() {
    if (this->_hostParams.LATTICE_VELOCITIES != nullptr) {
        delete[] this->_hostParams.LATTICE_VELOCITIES;
        this->_hostParams.LATTICE_VELOCITIES = nullptr;
    }
    if (this->_hostParams.LATTICE_WEIGHTS != nullptr) {
        delete[] this->_hostParams.LATTICE_WEIGHTS;
        this->_hostParams.LATTICE_WEIGHTS = nullptr;
    }
}

template<typename T>
void CollisionParamsCHMWrapper<T>::cleanupDevice() {
    // If _deviceParams was already freed and set to nullptr, return
    if(this->getDeviceParams() == nullptr) { return; }

    // Assuming _deviceParams has been properly allocated and initialized
    CollisionParamsCHM<T> paramsTemp;

    // Copy _deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->getDeviceParams(), sizeof(CollisionParamsCHM<T>), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.LATTICE_VELOCITIES);
    cudaFree(paramsTemp.LATTICE_WEIGHTS);

    // Finally, free the _deviceParams struct itself
    cudaFree(this->_deviceParams);
    this->_deviceParams = nullptr; // Ensure the pointer is marked as freed
}

/********************************************/
/***** Derived class 04: BoundaryParams *****/
/********************************************/
template<typename T>
BoundaryParamsWrapper<T>::BoundaryParamsWrapper() {}

template<typename T>
BoundaryParamsWrapper<T>::BoundaryParamsWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
        const unsigned int* POPULATION,
        const unsigned int* OPPOSITE_POPULATION,
        const T* WALL_VELOCITY,
        BoundaryLocation location
    ) {
    setValues(dim, nx, ny, q, LATTICE_VELOCITIES, LATTICE_WEIGHTS, POPULATION, OPPOSITE_POPULATION, WALL_VELOCITY, location);
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
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        const int* LATTICE_VELOCITIES,
        const T* LATTICE_WEIGHTS,
        const unsigned int* POPULATION,
        const unsigned int* OPPOSITE_POPULATION,
        const T* WALL_VELOCITY,
        BoundaryLocation location
    ) {
    // Clean up existing data
    cleanupHost();

    // Assign new deep copies
    this->_hostParams.D          = dim;
    this->_hostParams.Nx         = nx;
    this->_hostParams.Ny         = ny;
    this->_hostParams.Q          = q;
    this->_hostParams.location   = location;

    this->_hostParams.LATTICE_VELOCITIES = new int[q * dim];
    std::copy(LATTICE_VELOCITIES, LATTICE_VELOCITIES + (q * dim), this->_hostParams.LATTICE_VELOCITIES);

    this->_hostParams.LATTICE_WEIGHTS = new T[q];
    std::copy(LATTICE_WEIGHTS, LATTICE_WEIGHTS + q, this->_hostParams.LATTICE_WEIGHTS);

    unsigned int lengthPOPULATION;
    if      (dim == 2 && q == 9) { lengthPOPULATION = 3; }
    else if (dim == 2 && q == 5) { lengthPOPULATION = 1; }
    else { lengthPOPULATION = 0; }
    this->_hostParams.POPULATION = new unsigned int[lengthPOPULATION];
    std::copy(POPULATION, POPULATION + lengthPOPULATION, this->_hostParams.POPULATION);
    
    this->_hostParams.OPPOSITE_POPULATION = new unsigned int[q];
    std::copy(OPPOSITE_POPULATION, OPPOSITE_POPULATION + q, this->_hostParams.OPPOSITE_POPULATION);

    if (WALL_VELOCITY != nullptr) {
        this->_hostParams.WALL_VELOCITY = new T[dim];
        std::copy(WALL_VELOCITY, WALL_VELOCITY + dim, this->_hostParams.WALL_VELOCITY);
    } else {
        this->_hostParams.WALL_VELOCITY = nullptr; // Ensure it is nullptr if not used
    }

    allocateAndCopyToDevice();
}

template<typename T>
void BoundaryParamsWrapper<T>::setWallVelocity(const std::vector<T>& wallVelocity) {
    // Clean up existing data
    delete[] this->_hostParams.WALL_VELOCITY;
    
    // Assign new deep copies
    this->_hostParams.WALL_VELOCITY = new T[this->_hostParams.D];
    std::copy(wallVelocity.data(), wallVelocity.data() + this->_hostParams.D, this->_hostParams.WALL_VELOCITY);

    allocateAndCopyToDevice();
}

template<typename T>
void BoundaryParamsWrapper<T>::allocateAndCopyToDevice() {
    // Cleanup device resources that may have been previously allocated
    cleanupDevice();

    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->_hostParams.Q * this->_hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, this->_hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->_hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, this->_hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Allocate device memory for wall velocity and copy data, if wall velocity is specified, i.e., not nullptr
    T* deviceWallVelocity = nullptr;
    if (this->_hostParams.WALL_VELOCITY != nullptr) {
        size_t sizeWallVelocity = this->_hostParams.D * sizeof(T);
        cudaErrorCheck(cudaMalloc(&deviceWallVelocity, sizeWallVelocity));
        cudaErrorCheck(cudaMemcpy(deviceWallVelocity, this->_hostParams.WALL_VELOCITY, sizeWallVelocity, cudaMemcpyHostToDevice));
    }

    // Allocate device memory for opposite populations and copy data
    unsigned int* deviceOppositePopulation;
    size_t sizeOppositePopulation = this->_hostParams.Q * this->_hostParams.D * sizeof(unsigned int);
    cudaErrorCheck(cudaMalloc(&deviceOppositePopulation, sizeOppositePopulation));
    cudaErrorCheck(cudaMemcpy(deviceOppositePopulation, this->_hostParams.OPPOSITE_POPULATION, sizeOppositePopulation, cudaMemcpyHostToDevice));

    // Allocate device memory for opposite populations and copy data
    unsigned int* devicePopulation;
    size_t sizePopulation;
    if      (this->_hostParams.D == 2 && this->_hostParams.Q == 9)  { sizePopulation = 3 * sizeof(unsigned int); }
    else if (this->_hostParams.D == 2 && this->_hostParams.Q == 5)  { sizePopulation =     sizeof(unsigned int); }
    else                                                            { exit(EXIT_FAILURE); }
    cudaErrorCheck(cudaMalloc(&devicePopulation, sizePopulation));
    cudaErrorCheck(cudaMemcpy(devicePopulation, this->_hostParams.POPULATION, sizePopulation, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of Params with device pointers
    BoundaryParams<T> paramsTemp    = this->_hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES   = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS      = deviceLatticeWeights;
    paramsTemp.POPULATION           = devicePopulation;
    paramsTemp.OPPOSITE_POPULATION  = deviceOppositePopulation;
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
    if (this->_hostParams.LATTICE_VELOCITIES != nullptr) {
        delete[] this->_hostParams.LATTICE_VELOCITIES;
        this->_hostParams.LATTICE_VELOCITIES = nullptr;
    }
    if (this->_hostParams.LATTICE_WEIGHTS != nullptr) {
        delete[] this->_hostParams.LATTICE_WEIGHTS;
        this->_hostParams.LATTICE_WEIGHTS = nullptr;
    }
    if (this->_hostParams.OPPOSITE_POPULATION != nullptr) {
        delete[] this->_hostParams.OPPOSITE_POPULATION;
        this->_hostParams.OPPOSITE_POPULATION = nullptr;
    }
    if (this->_hostParams.POPULATION != nullptr) {
        delete[] this->_hostParams.POPULATION;
        this->_hostParams.POPULATION = nullptr;
    }
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
    cudaFree(paramsTemp.LATTICE_VELOCITIES);
    cudaFree(paramsTemp.LATTICE_WEIGHTS);
    cudaFree(paramsTemp.POPULATION);
    cudaFree(paramsTemp.OPPOSITE_POPULATION);
    cudaFree(paramsTemp.WALL_VELOCITY);

    // Finally, free the _deviceParams struct itself
    cudaFree(this->_deviceParams);
    this->_deviceParams = nullptr; // Ensure the pointer is marked as freed
}

#endif // KERNEL_PARAMETERS_HH