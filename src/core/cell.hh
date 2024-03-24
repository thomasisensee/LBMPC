#ifndef Cell_HH
#define Cell_HH

#include "cell.h"
#include "lbmModel.h"

template<typename T, typename lbmModelType>
Cell<T,lbmModelType>::Cell(lbmModelType* model) : lbmModel(model) {}

template<typename T, typename lbmModelType>
T Cell<T,lbmModelType>::getZeroMoment(T* d_population) const
{
    T rho;
    for(int i=0; i<this->lbm.getQ; i++) { rho += d_population[i]; }
    return rho;
}

template<typename T, typename lbmModelType>
T Cell<T,lbmModelType>::getFirstMomentX(T* d_population) const
{
    T m1x;
    for(int i=0; i<this->lbm.getQ; i++) { m1x += d_population[i]*this->lbmModel.getCX(i); }
    return m1x;
}

template<typename T, typename lbmModelType>
T Cell<T,lbmModelType>::getFirstMomentY(T* d_population) const
{
    T m1y;
    for(int i=0; i<this->lbm.getQ; i++) { m1y += d_population[i]*this->lbmModel.getCY(i); }
    return m1y;
}

#endif
