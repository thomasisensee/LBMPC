#ifndef GRID_GEOMETRY_HH
#define GRID_GEOMETRY_HH

#include <iostream>

#include "gridGeometry.h"

/******************************/
/***** Base grid geometry *****/
/******************************/
template<typename T>
GridGeometry<T>::GridGeometry(T delta) : _delta(delta) {}

template<typename T>
T GridGeometry<T>::getDelta() const {
    return _delta;
}

/*************************************/
/***** Derived grid geometry: 1D *****/
/*************************************/
template<typename T>
GridGeometry1D<T>::GridGeometry1D(T delta, unsigned int nX) : GridGeometry<T>(delta), _nX(nX) {}

template<typename T>
unsigned int GridGeometry1D<T>::getNx() const {
    return _nX;
}

template<typename T>
unsigned int GridGeometry1D<T>::getGhostNx() const {
    return _nX+2;
}

template<typename T>
unsigned int GridGeometry1D<T>::getVolume() const {
    return _nX;
}

template<typename T>
unsigned int GridGeometry1D<T>::getGhostVolume() const {
    return _nX+2;
}

template<typename T>
void GridGeometry1D<T>::printParameters() const {
    std::cout << "============== Grid Details ==============" << std::endl;
    std::cout << "==\tExtent Lx:"  << "\t" << this->getNx()*this->getDelta() << "\t\t==" << std::endl;
    std::cout << "==\tExtent Nx:"  << "\t" << this->getNx()  << "\t==" << std::endl;
    std::cout << "==\tΔx:" << "\t" << "\t" << this->getDelta() << "\t==" << std::endl;
    std::cout << "==========================================\n" << std::endl;
}

/*************************************/
/***** Derived grid geometry: 2D *****/
/*************************************/
template<typename T>
GridGeometry2D<T>::GridGeometry2D(T delta, unsigned int nX, unsigned int nY) : GridGeometry<T>(delta), _nX(nX), _nY(nY) {}

template<typename T>
unsigned int GridGeometry2D<T>::getNx() const {
    return _nX;
}

template<typename T>
unsigned int GridGeometry2D<T>::getNy() const {
    return _nY;
}

template<typename T>
unsigned int GridGeometry2D<T>::getGhostNx() const {
    return _nX+2;
}

template<typename T>
unsigned int GridGeometry2D<T>::getGhostNy() const {
    return _nY+2;
}

template<typename T>
unsigned int GridGeometry2D<T>::getVolume() const {
    return _nX*_nY;
}

template<typename T>
unsigned int GridGeometry2D<T>::getGhostVolume() const {
    return (_nX+2)*(_nY+2);
}

template<typename T>
void GridGeometry2D<T>::printParameters() const {
    std::cout << "============== Grid Details ==============" << std::endl;
    std::cout << "==\tExtent (Lx,Ly):"  << "\t" << "(" << this->getNx()*this->getDelta() << "/" << this->getNy()*this->getDelta() << ")" << "\t\t==" << std::endl;
    std::cout << "==\tExtent (Nx,Ny):"  << "\t" << "(" << this->getNx() << "/" << this->getNy() << ")" << "\t==" << std::endl;
    std::cout << "==\tΔx:" << "\t" << "\t" << this->getDelta() << "\t==" << std::endl;
    std::cout << "==========================================\n" << std::endl;
}

#endif // GRID_GEOMETRY_HH
