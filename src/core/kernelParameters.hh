#ifndef KERNEL_PARAMETERS_HH
#define KERNEL_PARAMETERS_HH

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
ParamsWrapper<T, ParamsType>::~ParamsWrapper() {
    // Cleanup host resources
    cleanupHost();
    // Cleanup device resources
    cleanupDevice();
}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny
    ) {
    this->hostParams.D                  = dim;
    this->hostParams.Nx                 = nx;
    this->hostParams.Ny                 = ny;

    allocateAndCopyToDevice();
}


template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::setWallVelocity(T* WALL_VELOCITY) {
    this->hostParams.WALL_VELOCITY = WALL_VELOCITY;

    allocateAndCopyToDevice();
}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::allocateAndCopyToDevice() {
    // Prepare the host-side copy of LBParams with device pointers
    ParamsType paramsTemp = hostParams; // Use a temporary host copy

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&deviceParams, sizeof(ParamsType)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->getDeviceParams(), &paramsTemp, sizeof(ParamsType), cudaMemcpyHostToDevice));
}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::cleanupHost() {
    delete[] this->hostParams.LATTICE_VELOCITIES;
    delete[] this->hostParams.LATTICE_WEIGHTS;
}

template<typename T, typename ParamsType>
void ParamsWrapper<T, ParamsType>::cleanupDevice() {
    // If deviceParams was already freed and set to nullptr, return
    if(this->getDeviceParams() == nullptr) { return; }

    // Assuming deviceParams has been properly allocated and initialized
    ParamsType paramsTemp;

    // Copy deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->getDeviceParams(), sizeof(ParamsType), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.LATTICE_VELOCITIES);
    cudaFree(paramsTemp.LATTICE_WEIGHTS);

    // Finally, free the deviceParams struct itself
    cudaFree(this->deviceParams);
    this->deviceParams = nullptr; // Ensure the pointer is marked as freed
}

template<typename T, typename ParamsType>
const ParamsType& ParamsWrapper<T, ParamsType>::getHostParams() const {
    return hostParams;
}

template<typename T, typename ParamsType>
ParamsType* ParamsWrapper<T, ParamsType>::getDeviceParams() {
    return deviceParams;
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
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS
    ) {
    setValues(dim, nx, ny, q, LATTICE_VELOCITIES, LATTICE_WEIGHTS);
}

template<typename T>
LBParamsWrapper<T>::~LBParamsWrapper() {}

template<typename T>
void LBParamsWrapper<T>::setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS
    ) {

    // Clean up existing data
    delete[] this->hostParams.LATTICE_VELOCITIES;
    delete[] this->hostParams.LATTICE_WEIGHTS;

    // Assign new deep copies
    this->hostParams.D                  = dim;
    this->hostParams.Nx                 = nx;
    this->hostParams.Ny                 = ny;
    this->hostParams.Q                  = q;

    this->hostParams.LATTICE_VELOCITIES = new int[q * dim];
    std::copy(LATTICE_VELOCITIES, LATTICE_VELOCITIES + (q * dim), this->hostParams.LATTICE_VELOCITIES);

    this->hostParams.LATTICE_WEIGHTS = new T[q];
    std::copy(LATTICE_WEIGHTS, LATTICE_WEIGHTS + q, this->hostParams.LATTICE_WEIGHTS);

    allocateAndCopyToDevice();
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
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        T omegaShear
    ) {
    setValues(dim, nx, ny, q, LATTICE_VELOCITIES, LATTICE_WEIGHTS, omegaShear);
}

template<typename T>
CollisionParamsBGKWrapper<T>::~CollisionParamsBGKWrapper() {}

template<typename T>
void CollisionParamsBGKWrapper<T>::setValues(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        T omegaShear
    ) {
    // Clean up existing data
    delete[] this->hostParams.LATTICE_VELOCITIES;
    delete[] this->hostParams.LATTICE_WEIGHTS;

    // Assign new deep copies
    this->hostParams.D                  = dim;
    this->hostParams.Nx                 = nx;
    this->hostParams.Ny                 = ny;
    this->hostParams.Q                  = q;
    this->hostParams.omegaShear         = omegaShear;

    this->hostParams.LATTICE_VELOCITIES = new int[q * dim];
    std::copy(LATTICE_VELOCITIES, LATTICE_VELOCITIES + (q * dim), this->hostParams.LATTICE_VELOCITIES);

    this->hostParams.LATTICE_WEIGHTS = new T[q];
    std::copy(LATTICE_WEIGHTS, LATTICE_WEIGHTS + q, this->hostParams.LATTICE_WEIGHTS);

    allocateAndCopyToDevice();
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

/************************************************/
/***** Derived class 02: CollisionParamsCHM *****/
/************************************************/
template<typename T>
CollisionParamsCHMWrapper<T>::CollisionParamsCHMWrapper() {}

template<typename T>
CollisionParamsCHMWrapper<T>::CollisionParamsCHMWrapper(
        unsigned int dim,
        unsigned int nx,
        unsigned int ny,
        unsigned int q,
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
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
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        T omegaShear,
        T omegaBulk
    ) {
    // Clean up existing data
    delete[] this->hostParams.LATTICE_VELOCITIES;
    delete[] this->hostParams.LATTICE_WEIGHTS;

    // Assign new deep copies
    this->hostParams.D                  = dim;
    this->hostParams.Nx                 = nx;
    this->hostParams.Ny                 = ny;
    this->hostParams.Q                  = q;
    this->hostParams.omegaShear         = omegaShear;
    this->hostParams.omegaBulk          = omegaBulk;

    this->hostParams.LATTICE_VELOCITIES = new int[q * dim];
    std::copy(LATTICE_VELOCITIES, LATTICE_VELOCITIES + (q * dim), this->hostParams.LATTICE_VELOCITIES);

    this->hostParams.LATTICE_WEIGHTS = new T[q];
    std::copy(LATTICE_WEIGHTS, LATTICE_WEIGHTS + q, this->hostParams.LATTICE_WEIGHTS);

    allocateAndCopyToDevice();
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
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        unsigned int* OPPOSITE_POPULATION,
        T* WALL_VELOCITY,
        BoundaryLocation location
    ) {
    setValues(dim, nx, ny, q, LATTICE_VELOCITIES, LATTICE_WEIGHTS, OPPOSITE_POPULATION, WALL_VELOCITY, location);
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
        int* LATTICE_VELOCITIES,
        T* LATTICE_WEIGHTS,
        unsigned int* OPPOSITE_POPULATION,
        T* WALL_VELOCITY,
        BoundaryLocation location
    ) {

    // Clean up existing data
    delete[] this->hostParams.LATTICE_VELOCITIES;
    delete[] this->hostParams.LATTICE_WEIGHTS;
    delete[] this->hostParams.OPPOSITE_POPULATION;
    delete[] this->hostParams.WALL_VELOCITY;

    // Assign new deep copies
    this->hostParams.D          = dim;
    this->hostParams.Nx         = nx;
    this->hostParams.Ny         = ny;
    this->hostParams.Q          = q;
    this->hostParams.location   = location;

    this->hostParams.LATTICE_VELOCITIES = new int[q * dim];
    std::copy(LATTICE_VELOCITIES, LATTICE_VELOCITIES + (q * dim), this->hostParams.LATTICE_VELOCITIES);

    this->hostParams.LATTICE_WEIGHTS = new T[q];
    std::copy(LATTICE_WEIGHTS, LATTICE_WEIGHTS + q, this->hostParams.LATTICE_WEIGHTS);

    this->hostParams.OPPOSITE_POPULATION = new unsigned int[q];
    std::copy(OPPOSITE_POPULATION, OPPOSITE_POPULATION + q, this->hostParams.OPPOSITE_POPULATION);

    if (WALL_VELOCITY != nullptr) {
        this->hostParams.WALL_VELOCITY = new T[dim];
        std::copy(WALL_VELOCITY, WALL_VELOCITY + dim, this->hostParams.WALL_VELOCITY);
    } else {
        this->hostParams.WALL_VELOCITY = nullptr; // Ensure it's nullptr if not used
    }

    allocateAndCopyToDevice();
}

template<typename T>
void BoundaryParamsWrapper<T>::allocateAndCopyToDevice() {
    // Allocate device memory for lattice velocities and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = this->hostParams.Q * this->hostParams.D * sizeof(int);
    cudaErrorCheck(cudaMalloc(&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, this->hostParams.LATTICE_VELOCITIES, sizeLatticeVelocities, cudaMemcpyHostToDevice));

    // Allocate device memory for opposite populations and copy data
    unsigned int* deviceOppositePopulation;
    size_t sizeOppositePopulation = this->hostParams.Q * this->hostParams.D * sizeof(unsigned int);
    cudaErrorCheck(cudaMalloc(&deviceOppositePopulation, sizeOppositePopulation));
    cudaErrorCheck(cudaMemcpy(deviceOppositePopulation, this->hostParams.OPPOSITE_POPULATION, sizeOppositePopulation, cudaMemcpyHostToDevice));

    // Allocate device memory for lattice weights and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = this->hostParams.Q * sizeof(T);
    cudaErrorCheck(cudaMalloc(&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, this->hostParams.LATTICE_WEIGHTS, sizeLatticeWeights, cudaMemcpyHostToDevice));

    // Allocate device memory for wall velocity and copy data, if wall velocity is specified, i.e., not nullptr
    T* deviceWallVelocity = nullptr;
    if (this->hostParams.WALL_VELOCITY != nullptr) {
        size_t sizeWallVelocity = this->hostParams.D * sizeof(T);
        cudaErrorCheck(cudaMalloc(&deviceWallVelocity, sizeWallVelocity));
        cudaErrorCheck(cudaMemcpy(deviceWallVelocity, this->hostParams.WALL_VELOCITY, sizeWallVelocity, cudaMemcpyHostToDevice));
    }

    // Prepare the host-side copy of LBParams with device pointers
    BoundaryParams<T> paramsTemp    = this->hostParams; // Use a temporary host copy
    paramsTemp.LATTICE_VELOCITIES   = deviceLatticeVelocities;
    paramsTemp.OPPOSITE_POPULATION  = deviceOppositePopulation;
    paramsTemp.LATTICE_WEIGHTS      = deviceLatticeWeights;
    paramsTemp.WALL_VELOCITY        = deviceWallVelocity;

    // Allocate memory for the LBParams struct on the device if not already allocated
    if (this->deviceParams == nullptr) {
        cudaErrorCheck(cudaMalloc(&(this->deviceParams), sizeof(BoundaryParams<T>)));
    }

    // Copy the prepared LBParams (with device pointers) from the temporary host copy to the device
    cudaErrorCheck(cudaMemcpy(this->deviceParams, &paramsTemp, sizeof(BoundaryParams<T>), cudaMemcpyHostToDevice));
}

template<typename T>
void BoundaryParamsWrapper<T>::cleanupHost() {
    delete[] this->hostParams.OPPOSITE_POPULATION;
    delete[] this->hostParams.WALL_VELOCITY;
}

template<typename T>
void BoundaryParamsWrapper<T>::cleanupDevice() {
    // Assuming deviceParams has been properly allocated and initialized
    BoundaryParams<T> paramsTemp;

    // Copy deviceParams back to host to access the pointers
    cudaErrorCheck(cudaMemcpy(&paramsTemp, this->deviceParams, sizeof(BoundaryParams<T>), cudaMemcpyDeviceToHost));

    // Use the pointers from the temp copy to free device memory
    cudaFree(paramsTemp.OPPOSITE_POPULATION);
    cudaFree(paramsTemp.WALL_VELOCITY);

    // Finally, free the deviceParams struct itself
    cudaFree(this->deviceParams);
    this->deviceParams = nullptr; // Ensure the pointer is marked as freed
}

#endif // KERNEL_PARAMETERS_HH
