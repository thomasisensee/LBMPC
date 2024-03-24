#ifndef LBM_Model_H
#define LBM_Model_H

#include <cuda_runtime.h>

template<typename T>
class LBMModel
{
protected:
    /// Dimension
    unsigned int D;
    /// Number of velocities in velocity set
    unsigned int Q;
    /// pointer to array with lattice velocities
    int *LATTICE_VELOCITIES;
    /// pointer to array with lattice weights
    T *LATTICE_WEIGHTS;
public:
    /// get the dimension (D)
    __host__ __device__ unsigned int getD() const;
    /// get the number of velocities in velocity set (Q)
    __host__ __device__ unsigned int getQ() const;
    /// get the lattice velocity x-component corresponding to index i
    __host__ __device__ virtual int getCX(unsigned int i) const = 0;
    /// get the lattice velocity y-component corresponding to index i
    __host__ __device__ virtual int getCY(unsigned int i) const = 0;
    /// get the lattice weight corresponding to index i
    __host__ __device__ virtual T getWEIGHT(unsigned int i) const = 0;
    __host__  int* getLatticeVelocitiesPtr() const;
    __host__  T* getLatticeWeightsPtr() const;
    /// Prints LBM model details
    void print() const;
};

template<typename T>
class D2Q9 : public LBMModel<T>
{
public:
    D2Q9();
    ~D2Q9();
    __host__ __device__ int getCX(unsigned int i) const;
    __host__ __device__ int getCY(unsigned int i) const;
    __host__ __device__ T getWEIGHT(unsigned int i) const;
};

/// Wrapper class for duplication on GPU
template<typename T>
class LBMModelWrapper {
private:
    /// Host-side LBMModel object
    LBMModel<T>* hostModel;
    /// Device-side LBMModel object
    LBMModel<T>* deviceModel;

public:
    // Constructor
    LBMModelWrapper(LBMModel<T>* lbmModel);

    // Destructor
    ~LBMModelWrapper();

    // Allocate device memory and copy data
    void allocateAndCopyToDevice();
    
    /// Get pointer to the host LBMModel object
    LBMModel<T>* getHostModel() const;
    
    /// Get pointer to the device LBMModel object
    LBMModel<T>* getDeviceModel() const;
};

#include "lbmModel.hh"

#endif
