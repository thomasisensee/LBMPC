#ifndef LBM_Model_HH
#define LBM_Model_HH

#include <iostream>
#include <cuda_runtime.h>

#include "lbmModel.h"
#include "cuda/cudaErrorHandler.h"
#include "cuda/cudaKernels.h"

template<typename T>
__host__ __device__ unsigned int LBMModel<T>::getD() const
{
    return D;
}

template<typename T>
__host__ __device__ unsigned int LBMModel<T>::getQ() const
{
    return Q;
}

template<typename T>
__host__ __device__ int* LBMModel<T>::getLatticeVelocitiesPtr() const
{
    return LATTICE_VELOCITIES;
}

template<typename T>
__host__ __device__ T* LBMModel<T>::getLatticeWeightsPtr() const
{
    return LATTICE_WEIGHTS;
}

template<typename T>
void LBMModel<T>::print() const
{
    std::cout << "============================== LBM Model Details ==============================" << std::endl;
    std::cout << "==                                   D" << getD() << "Q" << getQ() << "                                    ==" << std::endl;
    std::cout << "== Cx ="; for(int i=0; i<Q; ++i) {std::cout << "\t" << LATTICE_VELOCITIES[i*D]; } std::cout << "    ==" << std::endl;
    std::cout << "== Cy ="; for(int i=0; i<Q; ++i) {std::cout << "\t" << LATTICE_VELOCITIES[i*D+1]; } std::cout << "   ==" << std::endl;
    std::cout << "== w  ="; for(int i=0; i<Q; ++i) {std::cout << "\t" << LATTICE_WEIGHTS[i]; } std::cout << "   ==" << std::endl;
    std::cout << "===============================================================================\n" << std::endl;
}

template<typename T>
__host__ __device__ D2Q9<T>::D2Q9()
{
    this->D = 2;
    this->Q = 9;
    this->LATTICE_WEIGHTS = new T[9];
    this->LATTICE_WEIGHTS[0] = 4.0/9.0;
    this->LATTICE_WEIGHTS[1] = 1.0/9.0;
    this->LATTICE_WEIGHTS[2] = 1.0/9.0;
    this->LATTICE_WEIGHTS[3] = 1.0/9.0;
    this->LATTICE_WEIGHTS[4] = 1.0/9.0;
    this->LATTICE_WEIGHTS[5] = 1.0/36.0;
    this->LATTICE_WEIGHTS[6] = 1.0/36.0;
    this->LATTICE_WEIGHTS[7] = 1.0/36.0;
    this->LATTICE_WEIGHTS[8] = 1.0/36.0;

    this->LATTICE_VELOCITIES = new int[18];
    int velocities[9][2] = {{0, 0},{1, 0},{0, 1},{-1, 0},{0, -1},{1, 1},{-1, 1},{-1, -1},{1, -1}};
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 2; ++j) {
            this->LATTICE_VELOCITIES[i*this->D+j] = velocities[i][j];
        }
    }
}

template<typename T>
__host__ __device__ D2Q9<T>::~D2Q9()
{
    delete[] this->LATTICE_VELOCITIES;
    delete[] this->LATTICE_WEIGHTS;
}

template<typename T>
__host__ __device__ int D2Q9<T>::getCX(unsigned int i) const 
{
    return this->LATTICE_VELOCITIES[i*2];
}

template<typename T>
__host__ __device__ int D2Q9<T>::getCY(unsigned int i) const 
{
    return this->LATTICE_VELOCITIES[i*2+1];
}

template<typename T>
__host__ __device__ T D2Q9<T>::getWEIGHT(unsigned int i) const 
{
    return this->LATTICE_WEIGHTS[i];
}

template<typename T>
__host__ LBMModel<T>* D2Q9<T>::getDerivedModel() const
{
    return new D2Q9<T>(*this); // Return a pointer to a new D2Q9 object
}

template<typename T, typename LBMModelClassType>
LBMModelWrapper<T,LBMModelClassType>::LBMModelWrapper(LBMModelClassType* lbmModel) : hostModel(lbmModel)//, deviceModel(nullptr)
{
    allocateAndCopyToDevice();
}

template<typename T, typename LBMModelClassType>
LBMModelWrapper<T,LBMModelClassType>::~LBMModelWrapper()
{
/*
    if(deviceModel)
    {
        cudaErrorCheck(cudaFree(deviceModel)); // even necessary?
    }
    */
}


template<typename T, typename LBMModelClassType>
void LBMModelWrapper<T,LBMModelClassType>::allocateAndCopyToDevice()
{
    // Allocate device version of LBMModel object and copy data
    cudaErrorCheck(cudaMalloc((void **)&deviceModel, sizeof(*hostModel)));
    cudaErrorCheck(cudaMemcpy(deviceModel, hostModel, sizeof(*hostModel), cudaMemcpyHostToDevice));

    // Allocate device memory for LATTICE_VELOCITIES and copy data
    int* deviceLatticeVelocities;
    size_t sizeLatticeVelocities = sizeof(int)*hostModel->getQ()*hostModel->getD();
    cudaErrorCheck(cudaMalloc((void**)&deviceLatticeVelocities, sizeLatticeVelocities));
    cudaErrorCheck(cudaMemcpy(deviceLatticeVelocities, hostModel->getLatticeVelocitiesPtr(), sizeLatticeVelocities, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(&(deviceModel->LATTICE_VELOCITIES), &deviceLatticeVelocities, sizeof(int*), cudaMemcpyHostToDevice));

    // Allocate device memory for LATTICE_WEIGHTS and copy data
    T* deviceLatticeWeights;
    size_t sizeLatticeWeights = sizeof(T)*hostModel->getQ();
    cudaErrorCheck(cudaMalloc((void**)&deviceLatticeWeights, sizeLatticeWeights));
    cudaErrorCheck(cudaMemcpy(deviceLatticeWeights, hostModel->getLatticeWeightsPtr(), sizeLatticeWeights, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(&(deviceModel->LATTICE_WEIGHTS), &deviceLatticeWeights, sizeof(T*), cudaMemcpyHostToDevice));



    test1<T, LBMModelClassType>(deviceModel);
    //test2(deviceLatticeWeights);
    
    
    /* // Different method: create object on device and copy pointer. Doesn't work yet.
    // Allocate memory for device-side object
    cudaErrorCheck(cudaMalloc((void**)&deviceModel, sizeof(*hostModel)));
    
    // Allocation of deviceModel on the device
    launchCreateDeviceModel<T, LBMModelClassType>(&(deviceModel));
    
    cudaErrorCheck(cudaMemcpy(deviceModel->getLatticeWeightsPtr(), hostModel->getLatticeWeightsPtr(), sizeLatticeWeights, cudaMemcpyHostToDevice));
*/
}

template<typename T, typename LBMModelClassType>
LBMModelClassType* LBMModelWrapper<T,LBMModelClassType>::getHostModel() const
{
    return hostModel;
}

template<typename T, typename LBMModelClassType>
LBMModelClassType* LBMModelWrapper<T,LBMModelClassType>::getDeviceModel() const
{
    return deviceModel;
}

template<typename T, typename LBMModelClassType>
LBMModelClassType* LBMModelWrapper<T,LBMModelClassType>::getDerivedDeviceModel() const
{
    return hostModel->getDerivedModel();
}

#endif
