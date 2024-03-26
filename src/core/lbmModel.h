#ifndef LBM_Model_H
#define LBM_Model_H

#include <cuda_runtime.h>

template<typename T>
class LBMModel
{
public:
    /// Dimension
    unsigned int D;
    /// Number of velocities in velocity set
    unsigned int Q;
    /// pointer to array with lattice velocities
    int* LATTICE_VELOCITIES;
    /// pointer to array with lattice weights
    T* LATTICE_WEIGHTS;
public:
    /// get the dimension (D)
    unsigned int getD() const;
    /// get the number of velocities in velocity set (Q)
    unsigned int getQ() const;
    /// get the lattice velocity x-component corresponding to index i
    virtual int getCX(unsigned int i) const = 0;
    /// get the lattice velocity y-component corresponding to index i
    virtual int getCY(unsigned int i) const = 0;
    /// get the lattice weight corresponding to index i
    virtual T getWEIGHT(unsigned int i) const = 0;
    int* getLatticeVelocitiesPtr() const;
    T* getLatticeWeightsPtr() const;
    /// Prints LBM model details
    void print() const;
    /// Provides access to the specific derived class type
    virtual LBMModel<T>* getDerivedModel() const = 0;
};

template<typename T>
class D2Q9 : public LBMModel<T>
{
public:
    // Constructor
    D2Q9();
    // Destructor
    ~D2Q9();
    /// get the lattice velocity x-component corresponding to index i
    virtual int getCX(unsigned int i) const override;
    /// get the lattice velocity y-component corresponding to index i
    virtual int getCY(unsigned int i) const override;
    /// get the lattice weight corresponding to index i
    virtual T getWEIGHT(unsigned int i) const override;
    /// Provides access to the specific derived class type
    virtual LBMModel<T>* getDerivedModel() const override;
};

#include "lbmModel.hh"

#endif
