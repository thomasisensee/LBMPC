#ifndef GRID_GEOMETRY_2D_HH
#define GRID_GEOMETRY_2D_HH

#include <iostream>

#include "gridGeometry2D.h"

/*************************************/
/***** Derived grid geometry: 2D *****/
/*************************************/
template<typename T>
GridGeometry2D<T>::GridGeometry2D(T delta, unsigned int nX, unsigned int nY) : GridGeometry1D<T>(delta, nX), _nY(nY) {}

template<typename T>
unsigned int GridGeometry2D<T>::getNy() const {
    return _nY;
}

template<typename T>
unsigned int GridGeometry2D<T>::getGhostNy() const {
    return _nY+2;
}

template<typename T>
unsigned int GridGeometry2D<T>::getVolume() const {
    return this->_nX*_nY;
}

template<typename T>
unsigned int GridGeometry2D<T>::getGhostVolume() const {
    return (this->_nX+2)*(_nY+2);
}

template<typename T>
void GridGeometry2D<T>::printParameters() const {
    std::cout << "============== Grid Details ==============" << std::endl;
    std::cout << "==\tExtent (Lx,Ly):"  << "\t" << "(" << this->getNx()*this->getDelta() << "/" << this->getNy()*this->getDelta() << ")" << "\t\t==" << std::endl;
    std::cout << "==\tExtent (Nx,Ny):"  << "\t" << "(" << this->getNx() << "/" << this->getNy() << ")" << "\t==" << std::endl;
    std::cout << "==\tΔx:" << "\t" << "\t" << this->getDelta() << "\t==" << std::endl;
    std::cout << "==========================================\n" << std::endl;
}

#endif // GRID_GEOMETRY_2D_HH
