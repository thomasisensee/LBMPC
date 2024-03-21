#ifndef Cell_H
#define Cell_H

#include "lbmModel.h"

template<typename T, typename lbmModelType>
class Cell
{
private:
    /// LBM model providing dimensionality, velocity set, and weights
    lbmModelType *lbmModel;

public:
    // Constructor
    Cell(lbmModelType* model);
    /// Computes the 0th moment (density)
    T getZeroMoment(T *d_population) const;
    /// Computes the 1st moment X
    T getFirstMomentX(T *d_population) const;
    /// Computes the 1st moment Y
    T getFirstMomentY(T *d_population) const;
};

#endif
