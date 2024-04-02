#ifndef LB_Model_H
#define LB_Model_H

#include <cuda_runtime.h>


/**********************/
/***** Base class *****/
/**********************/
template<typename T, unsigned int Dim, unsigned int Q>
class LBModel {
protected:
    /// Dimension
    const unsigned int _D = Dim;

    /// Number of velocities in velocity set
    const unsigned int _Q = Q;

    /// pointer to array with lattice velocities
    int* _LATTICE_VELOCITIES;

    /// pointer to array with lattice weights
    T* _LATTICE_WEIGHTS;

    /// pointer to array with opposite populations 
    unsigned int* _OPPOSITE_POPULATION;

public:
    /// Constructor
    LBModel();

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
    
    /// Get the opposite population corresponding to index i
    virtual unsigned int getOppositePopualation(unsigned int i) const = 0;

    /// Get pointer to LATTICE_VELOCITIES
    int* getLatticeVelocitiesPtr() const;

    /// Get pointer to LATTICE_WEIGHTS
    T* getLatticeWeightsPtr() const;
    
    /// Get pointer to OPPOSITE_POPULATION
    unsigned int* getOppositePopualationPtr() const;

    /// Prints LB model details
    void print() const;

    /// Provides access to the specific derived class type
    virtual LBModel<T, Dim, Q>* getDerivedModel() const = 0;
};


/****************************************/
/***** Derived class 01: D2Q9 model *****/
/****************************************/
template<typename T>
class D2Q9 final : public LBModel<T, 2, 9> {

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
    /// Constructor
    D2Q9();
    
    /// Destructor
    ~D2Q9();

    /// get the lattice velocity x-component corresponding to index i
    virtual int getCX(unsigned int i) const override;

    /// get the lattice velocity y-component corresponding to index i
    virtual int getCY(unsigned int i) const override;

    /// get the lattice weight corresponding to index i
    virtual T getWEIGHT(unsigned int i) const override;
    
    /// Get the opposite population corresponding to index i
    virtual unsigned int getOppositePopualation(unsigned int i) const;

    /// Provides access to the specific derived class type
    virtual LBModel<T, 2, 9>* getDerivedModel() const override;
};

#include "lbModel.hh"

#endif
