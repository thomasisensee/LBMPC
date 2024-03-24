#ifndef LBM_Model_HH
#define LBM_Model_HH

#include <iostream>
#include <cuda_runtime.h>

#include "lbmModel.h"

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
__host__ int* LBMModel<T>::getLatticeVelocitiesPtr() const
{
    return LATTICE_VELOCITIES;
}

template<typename T>
__host__ T* LBMModel<T>::getLatticeWeightsPtr() const
{
    return LATTICE_WEIGHTS;
}

template<typename T>
void LBMModel<T>::print() const
{
    std::cout << "============================== LBM Model Details ==============================" << std::endl;
    std::cout << "==                                   D" << getD() << "Q" << getQ() << "                                    ==" << std::endl;
    std::cout << "== Cx ="; for(int i=0; i<Q; i++) {std::cout << "\t" << LATTICE_VELOCITIES[i*D]; } std::cout << "    ==" << std::endl;
    std::cout << "== Cy ="; for(int i=0; i<Q; i++) {std::cout << "\t" << LATTICE_VELOCITIES[i*D+1]; } std::cout << "   ==" << std::endl;
    std::cout << "== w  ="; for(int i=0; i<Q; i++) {std::cout << "\t" << LATTICE_WEIGHTS[i]; } std::cout << "   ==" << std::endl;
    std::cout << "===============================================================================\n" << std::endl;
}

template<typename T>
D2Q9<T>::D2Q9()
{
    this->D = 2;
    this->Q = 9;
    this->LATTICE_WEIGHTS = new T[9];
    this->LATTICE_WEIGHTS[0] = 4./9.;
    this->LATTICE_WEIGHTS[1] = 1./9.;
    this->LATTICE_WEIGHTS[2] = 1./9.;
    this->LATTICE_WEIGHTS[3] = 1./9.;
    this->LATTICE_WEIGHTS[4] = 1./9.;
    this->LATTICE_WEIGHTS[5] = 1./36.;
    this->LATTICE_WEIGHTS[6] = 1./36.;
    this->LATTICE_WEIGHTS[7] = 1./36.;
    this->LATTICE_WEIGHTS[8] = 1./36.;

    this->LATTICE_VELOCITIES = new int[this->Q*this->D];
    int velocities[9][2] = {{0, 0},{1, 0},{0, 1},{-1, 0},{0, -1},{1, 1},{-1, 1},{-1, -1},{1, -1}};
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 2; ++j) {
            this->LATTICE_VELOCITIES[i*this->D+j] = velocities[i][j];
        }
    }
}

template<typename T>
D2Q9<T>::~D2Q9()
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
LBMModelWrapper<T>::LBMModelWrapper(LBMModel<T>* lbmModel) : hostModel(lbmModel), deviceModel(nullptr)
{
    allocateAndCopyToDevice();  
}

template<typename T>
LBMModelWrapper<T>::~LBMModelWrapper()
{
    if(deviceModel)
    {
        cudaFree(deviceModel->getLatticeVelocitiesPtr());
        cudaFree(deviceModel->getLatticeWeightsPtr());
        cudaFree(deviceModel);
    }
}

template<typename T>
void LBMModelWrapper<T>::allocateAndCopyToDevice()
{
    // Allocate device version of LBMModel object and copy data
    cudaMalloc(&deviceModel, sizeof(LBMModel<T>));
    cudaMemcpy(deviceModel, hostModel, sizeof(LBMModel<T>), cudaMemcpyHostToDevice);

    // Allocate device memory for LATTICE_VELOCITIES and copy data
    int* latticeVelocitiesPtr = deviceModel->getLatticeVelocitiesPtr();
    cudaMalloc(&(latticeVelocitiesPtr), sizeof(int)*hostModel->getQ()*hostModel->getD());
    cudaMemcpy(deviceModel->getLatticeVelocitiesPtr(), hostModel->getLatticeVelocitiesPtr(), sizeof(int)*hostModel->getQ()*hostModel->getD(), cudaMemcpyHostToDevice);

    // Allocate device memory for LATTICE_WEIGHTS and copy data
    T* latticeWeightsPtr = deviceModel->getLatticeWeightsPtr();
    cudaMalloc(&(latticeWeightsPtr), sizeof(T)*hostModel->getQ());
    cudaMemcpy(deviceModel->getLatticeWeightsPtr(), hostModel->getLatticeWeightsPtr(), sizeof(T)*hostModel->getQ(), cudaMemcpyHostToDevice);
}

template<typename T>
LBMModel<T>* LBMModelWrapper<T>::getHostModel() const
{
    return hostModel;
}

template<typename T>
LBMModel<T>* LBMModelWrapper<T>::getDeviceModel() const
{
    return deviceModel;
}

#endif
