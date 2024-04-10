#ifndef LB_Model_H
#define LB_Model_H

#include <vector>
#include <map>

#include "core/constants.h"
#include "core/utilities.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T, typename LatticeDescriptor>
class LBModel {
protected:
    /// New: Maps boundary names to indices in the velocity set
    std::map<BoundaryLocation, std::vector<unsigned int>> _boundaryMapping;

public:
    /// Constructor
    LBModel();

    /// Get the dimension (D)
    unsigned int getD() const;

    /// Get the number of velocities in velocity set (Q)
    unsigned int getQ() const;

    /// Get the lattice velocity x-component corresponding to index i
    virtual int getCX(unsigned int i) const;

    /// Get the lattice velocity y-component corresponding to index i
    virtual int getCY(unsigned int i) const;

    /// Get the lattice weight corresponding to index i
    virtual T getWEIGHT(unsigned int i) const;

    /// Get pointer to LATTICE_VELOCITIES
    const int* getLatticeVelocitiesPtr() const;

    /// Get pointer to LATTICE_WEIGHTS
    const T* getLatticeWeightsPtr() const;

    /// Prints LB model details
    void printParameters() const;
};

#include "lbModel.hh"

#endif
