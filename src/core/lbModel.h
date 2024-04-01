#ifndef LB_Model_H
#define LB_Model_H

#include <cuda_runtime.h>


/**********************/
/***** Base class *****/
/**********************/
template<typename T>
class LBModel {
public:
    /// Dimension
    const unsigned int D;

    /// Number of velocities in velocity set
    const unsigned int Q;

    /// pointer to array with lattice velocities
    int* LATTICE_VELOCITIES;
    
    unsigned int* OPPOSITE_POPULATION;

    /// pointer to array with lattice weights
    T* LATTICE_WEIGHTS;
public:
    /// Constructor
    LBModel(unsigned int d, unsigned int q);

    /// Destructor
    virtual ~LBModel();

    /// Get the dimension (D)
    unsigned int getD() const;

    /// Get the number of velocities in velocity set (Q)
    unsigned int getQ() const;

    /// Get the lattice velocity x-component corresponding to index i
    virtual int getCX(unsigned int i) const = 0;

    /// Get the lattice velocity y-component corresponding to index i
    virtual int getCY(unsigned int i) const = 0;

    /// Get the lattice weight corresponding to index i
    virtual T getWEIGHT(unsigned int i) const = 0;

    /// Get pointer to LATTICE_VELOCITIES
    int* getLatticeVelocitiesPtr() const;

    /// Get pointer to LATTICE_WEIGHTS
    T* getLatticeWeightsPtr() const;

    /// Prints LB model details
    void print() const;

    /// Provides access to the specific derived class type
    virtual LBModel<T>* getDerivedModel() const = 0;
};


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
class D2Q9 : public LBModel<T> {

///////////////////////////////////
///                             ///
/// D2Q9 Lattice configuration: ///
///                             ///
///       8   3   5             ///
///        \  |  /              ///
///         \ | /               ///
///          \|/                ///
///     2-----0-----1           ///
///          /|\                ///
///         / | \               ///
///        /  |  \              ///
///       6   4   7             ///
///                             ///
///////////////////////////////////

public:
    // Constructor
    D2Q9();

    /// get the lattice velocity x-component corresponding to index i
    virtual int getCX(unsigned int i) const override;

    /// get the lattice velocity y-component corresponding to index i
    virtual int getCY(unsigned int i) const override;

    /// get the lattice weight corresponding to index i
    virtual T getWEIGHT(unsigned int i) const override;

    /// Provides access to the specific derived class type
    virtual LBModel<T>* getDerivedModel() const override;
};

#include "lbModel.hh"

#endif
