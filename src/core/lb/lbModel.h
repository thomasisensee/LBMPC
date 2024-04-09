#ifndef LB_Model_H
#define LB_Model_H

#include <vector>
#include <map>

#include "core/constants.h"
#include "core/utilities.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
class LBModel {
protected:
    /// Dimension
    const unsigned int _D;

    /// Number of velocities in velocity set
    const unsigned int _Q;

    /// pointer to array with lattice velocities
    int* _LATTICE_VELOCITIES;

    /// pointer to array with lattice weights
    T* _LATTICE_WEIGHTS;

    /// pointer to array with opposite populations 
    unsigned int* _OPPOSITE_POPULATION;

    /// New: Maps boundary names to indices in the velocity set
    std::map<BoundaryLocation, std::vector<unsigned int>> _BOUNDARY_MAPPING;

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
    
    /// Get the opposite population corresponding to index i
    virtual unsigned int getOppositePopualation(unsigned int i) const = 0;

    /// Get pointer to LATTICE_VELOCITIES
    const int* getLatticeVelocitiesPtr() const;

    /// Get pointer to LATTICE_WEIGHTS
    const T* getLatticeWeightsPtr() const;

    /// Get pointer to POPULATION
    const unsigned int* getPopulationPtr(BoundaryLocation location) const;
    
    /// Get pointer to OPPOSITE_POPULATION
    const unsigned int* getOppositePopualationPtr() const;

    /// Prints LB model details
    void printParameters() const;
    
    /// Prints LB model details
    void printBoundaryMapping() const;

    /// Provides access to the specific derived class type
    virtual LBModel<T>* getDerivedModel() const = 0;
};


/****************************************/
/***** Derived class 01: D2Q9 model *****/
/****************************************/
template<typename T>
class D2Q9 final : public LBModel<T> {

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
    virtual LBModel<T>* getDerivedModel() const override;
};

/****************************************/
/***** Derived class 02: D2Q5 model *****/
/****************************************/
template<typename T>
class D2Q5 final : public LBModel<T> {

///////////////////////////////////
///                             ///
/// D2Q5 Lattice configuration: ///
///                             ///
///           3                 ///
///           |                 ///
///           |                 ///
///           |                 ///
///     2-----0-----1           ///
///           |                 ///
///           |                 ///
///           |                 ///
///           4                 ///
///                             ///
///////////////////////////////////

public:
    /// Constructor
    D2Q5();
    
    /// Destructor
    ~D2Q5();

    /// get the lattice velocity x-component corresponding to index i
    virtual int getCX(unsigned int i) const override;

    /// get the lattice velocity y-component corresponding to index i
    virtual int getCY(unsigned int i) const override;

    /// get the lattice weight corresponding to index i
    virtual T getWEIGHT(unsigned int i) const override;
    
    /// Get the opposite population corresponding to index i
    virtual unsigned int getOppositePopualation(unsigned int i) const;

    /// Provides access to the specific derived class type
    virtual LBModel<T>* getDerivedModel() const override;
};

#include "lbModel.hh"

#endif
