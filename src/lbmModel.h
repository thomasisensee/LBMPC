#ifndef LBM_Model_H
#define LBM_Model_H

template<typename T>
class lbmModel
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
    unsigned int getD() const;
    /// get the number of velocities in velocity set (Q)
    unsigned int getQ() const;
    /// get the lattice velocity x-component corresponding to index i
    virtual int getCX(unsigned int i) const = 0;
    /// get the lattice velocity y-component corresponding to index i
    virtual int getCY(unsigned int i) const = 0;
    /// get the lattice weight corresponding to index i
    virtual T getWEIGHT(unsigned int i) const = 0;
    /// Prints LBM model details
    void print() const;
};

template<typename T>
class D2Q9 : public lbmModel<T>
{
public:
    D2Q9();
    ~D2Q9();
    int getCX(unsigned int i) const;
    int getCY(unsigned int i) const;
    T getWEIGHT(unsigned int i) const;
};

#include "lbmModel.hh"

#endif
