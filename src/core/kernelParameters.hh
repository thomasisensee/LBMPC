#ifndef KERNEL_PARAMETERS_HH
#define KERNEL_PARAMETERS_HH

/**********************************/
/***** Base class (templated) *****/
/**********************************/
template<typename T, typename ParamsType>
ParamsWrapper<T, ParamsType>::ParamsWrapper(unsigned int dim, unsigned int nx, unsigned int ny) {
    this->hostParams.D                  = dim;
    this->hostParams.Nx                 = nx;
    this->hostParams.Ny                 = ny;

    allocateAndCopyToDevice();
}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::allocateAndCopyToDevice() {
    // Prepare the host-side copy of LBParams with device pointers
    LBParams<T> paramsTemp = hostParams; // Use a temporary host copy

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&deviceParams, sizeof(LBParams<T>)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(deviceParams, &paramsTemp, sizeof(LBParams<T>), cudaMemcpyHostToDevice));
}

template<typename T, typename ParamsType>
ParamsType& ParamsWrapper<T, ParamsType>::getHostParams() {
    return hostParams;
}

template<typename T, typename ParamsType>
ParamsType* ParamsWrapper<T, ParamsType>::getDeviceParams() {
    return deviceParams;
}

/***************************************/
/***** Derived classe 01: LBParams *****/
/***************************************/
template<typename T>
LBParamsWrapper<T>::LBParamsWrapper(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights) {
    this->hostParams.D                  = dim;
    this->hostParams.Nx                 = nx;
    this->hostParams.Ny                 = ny;
    this->hostParams.Q                  = q;
    this->hostParams.LATTICE_VELOCITIES = latticeVelocities;
    this->hostParams.LATTICE_WEIGHTS    = latticeWeights;

    allocateAndCopyToDevice();
}

template<typename T>
LBParamsWrapper<T>::~LBParamsWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T>
void LBParamsWrapper<T>::allocateAndCopyToDevice() {
    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->hostParams.Q * this->hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, this->hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, this->hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of LBParams with device pointers
    LBParams<T> paramsTemp = this->hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (this->deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->deviceParams), sizeof(LBParams<T>)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->deviceParams, &paramsTemp, sizeof(LBParams<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void LBParamsWrapper<T>::cleanupHost() {
    delete[] this->hostParams.LATTICE_VELOCITIES;
    delete[] this->hostParams.LATTICE_WEIGHTS;
}

template<typename T>
void LBParamsWrapper<T>::cleanupDevice() {
    // Assuming deviceParams has been properly allocated and initialized
    LBParams<T> paramsTemp;

    // Copy deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->deviceParams, sizeof(LBParams<T>), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.LATTICE_VELOCITIES);
    cudaFree(paramsTemp.LATTICE_WEIGHTS);

    // Finally, free the deviceParams struct itself
    cudaFree(this->deviceParams);
    this->deviceParams = nullptr; // Ensure the pointer is marked as freed
}

/*************************************************/
/***** Derived classe 02: CollisionParamsBGK *****/
/*************************************************/
template<typename T>
CollisionParamsBGKWrapper<T>::CollisionParamsBGKWrapper(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights, T omegaShear) {
    this->hostParams.D                  = dim;
    this->hostParams.Nx                 = nx;
    this->hostParams.Ny                 = ny;
    this->hostParams.Q                  = q;
    this->hostParams.omegaShear         = omegaShear;
    this->hostParams.LATTICE_VELOCITIES = latticeVelocities;
    this->hostParams.LATTICE_WEIGHTS    = latticeWeights;

    allocateAndCopyToDevice();
}

template<typename T>
CollisionParamsBGKWrapper<T>::~CollisionParamsBGKWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T>
void CollisionParamsBGKWrapper<T>::allocateAndCopyToDevice() {
    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->hostParams.Q * this->hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, this->hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, this->hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of LBParams with device pointers
    CollisionParamsBGK<T> paramsTemp = this->hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (this->deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->deviceParams), sizeof(CollisionParamsBGK<T>)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->deviceParams, &paramsTemp, sizeof(CollisionParamsBGK<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void CollisionParamsBGKWrapper<T>::cleanupHost() {
    delete[] this->hostParams.LATTICE_VELOCITIES;
    delete[] this->hostParams.LATTICE_WEIGHTS;
}

template<typename T>
void CollisionParamsBGKWrapper<T>::cleanupDevice() {
    // Assuming deviceParams has been properly allocated and initialized
    CollisionParamsBGK<T> paramsTemp;

    // Copy deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->deviceParams, sizeof(CollisionParamsBGK<T>), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.LATTICE_VELOCITIES);
    cudaFree(paramsTemp.LATTICE_WEIGHTS);

    // Finally, free the deviceParams struct itself
    cudaFree(this->deviceParams);
    this->deviceParams = nullptr; // Ensure the pointer is marked as freed
}

/*************************************************/
/***** Derived classe 02: CollisionParamsCHM *****/
/*************************************************/
template<typename T>
CollisionParamsCHMWrapper<T>::CollisionParamsCHMWrapper(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights, T omegaShear, T omegaBulk) {
    this->hostParams.D                  = dim;
    this->hostParams.Nx                 = nx;
    this->hostParams.Ny                 = ny;
    this->hostParams.Q                  = q;
    this->hostParams.omegaShear         = omegaShear;
    this->hostParams.omegaBulk          = omegaBulk;
    this->hostParams.LATTICE_VELOCITIES = latticeVelocities;
    this->hostParams.LATTICE_WEIGHTS    = latticeWeights;

    allocateAndCopyToDevice();
}

template<typename T>
CollisionParamsCHMWrapper<T>::~CollisionParamsCHMWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T>
void CollisionParamsCHMWrapper<T>::allocateAndCopyToDevice() {
    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->hostParams.Q * this->hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, this->hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, this->hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Prepare the host-side copy of LBParams with device pointers
    CollisionParamsCHM<T> paramsTemp = this->hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES = deviceLatticeVelocities;
    paramsTemp.LATTICE_WEIGHTS = deviceLatticeWeights;

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (this->deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->deviceParams), sizeof(CollisionParamsCHM<T>)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->deviceParams, &paramsTemp, sizeof(CollisionParamsCHM<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void CollisionParamsCHMWrapper<T>::cleanupHost() {
    delete[] this->hostParams.LATTICE_VELOCITIES;
    delete[] this->hostParams.LATTICE_WEIGHTS;
}

template<typename T>
void CollisionParamsCHMWrapper<T>::cleanupDevice() {
    // Assuming deviceParams has been properly allocated and initialized
    CollisionParamsCHM<T> paramsTemp;

    // Copy deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->deviceParams, sizeof(CollisionParamsCHM<T>), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.LATTICE_VELOCITIES);
    cudaFree(paramsTemp.LATTICE_WEIGHTS);

    // Finally, free the deviceParams struct itself
    cudaFree(this->deviceParams);
    this->deviceParams = nullptr; // Ensure the pointer is marked as freed
}

/*********************************************/
/***** Derived classe 04: BoundaryParams *****/
/*********************************************/
template<typename T>
BoundaryParamsWrapper<T>::BoundaryParamsWrapper(unsigned int dim, unsigned int nx, unsigned int ny, unsigned int q, int* latticeVelocities, T* latticeWeights, unsigned int* OPPOSITE_POPULATION, T* wallVelocity, BoundaryLocation location) {
    this->hostParams.D                      = dim;
    this->hostParams.Nx                     = nx;
    this->hostParams.Ny                     = ny;
    this->hostParams.Q                      = q;
    this->hostParams.location               = location;
    this->hostParams.wallVelocity           = wallVelocity;
    this->hostParams.LATTICE_VELOCITIES     = latticeVelocities;
    this->hostParams.OPPOSITE_POPULATION    = OPPOSITE_POPULATION;
    this->hostParams.LATTICE_WEIGHTS        = latticeWeights;

    allocateAndCopyToDevice();
}

template<typename T>
BoundaryParamsWrapper<T>::~BoundaryParamsWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T>
void BoundaryParamsWrapper<T>::allocateAndCopyToDevice() {
    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->hostParams.Q * this->hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, this->hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for opposite populations and copy data
    int* deviceOppositePopulation;
    size_t sizeOppositePopulation = this->hostParams.Q * this->hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceOppositePopulation, sizeOppositePopulation));
    cudaErrorCheck(cudaMemcpy(deviceOppositePopulation, this->hostParams.OPPOSITE_POPULATION, sizeOppositePopulation, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, this->hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceWallVelocity;
    size_t sizeWallVelocity = this->hostParams.D * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceWallVelocity, sizeWallVelocity));
    cudaErrorCheck(cudaMemcpy(deviceWallVelocity, this->hostParams.wallVelocity, sizeWallVelocity, cudaMemcpyHostToDevice));


    // Prepare the host-side copy of LBParams with device pointers
    BoundaryParams<T> paramsTemp    = this->hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES   = deviceLatticeVelocities;
    paramsTemp.OPPOSITE_POPULATION  = deviceOppositePopulation;
    paramsTemp.LATTICE_WEIGHTS      = deviceLatticeWeights;
    paramsTemp.wallVelocity         = deviceWallVelocity;

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (this->deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->deviceParams), sizeof(BoundaryParams<T>)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->deviceParams, &paramsTemp, sizeof(BoundaryParams<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void BoundaryParamsWrapper<T>::cleanupHost() {
    delete[] this->hostParams.LATTICE_VELOCITIES;
    delete[] this->hostParams.OPPOSITE_POPULATION;
    delete[] this->hostParams.LATTICE_WEIGHTS;
    delete[] this->hostParams.wallVelocity;
}

template<typename T>
void BoundaryParamsWrapper<T>::cleanupDevice() {
    // Assuming deviceParams has been properly allocated and initialized
    CollisionParamsCHM<T> paramsTemp;

    // Copy deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->deviceParams, sizeof(BoundaryParams<T>), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.LATTICE_VELOCITIES);
    cudaFree(paramsTemp.OPPOSITE_POPULATION);
    cudaFree(paramsTemp.LATTICE_WEIGHTS);
    cudaFree(paramsTemp.wallVelocity);

    // Finally, free the deviceParams struct itself
    cudaFree(this->deviceParams);
    this->deviceParams = nullptr; // Ensure the pointer is marked as freed
}

#endif // KERNEL_PARAMETERS_HH
