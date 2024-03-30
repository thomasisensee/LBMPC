#ifndef GridGeometry_HH
#define GridGeometry_HH

#include <iostream>

#include "gridGeometry.h"

template<typename T>
GridGeometry2D<T>::GridGeometry2D(T delta, int nX, int nY) : _delta(delta), _nX(nX), _nY(nY) {}

template<typename T>
T GridGeometry2D<T>::getDelta() const {
    return _delta;
}

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
unsigned int GridGeometry2D<T>::getGhostVolume() const {
    return (_nX+2)*(_nY+2);
}

template<typename T>
void GridGeometry2D<T>::print() const {
    std::cout << "============== Grid Details ==============" << std::endl;
    std::cout << "==\tExtent (Lx,Ly):"  << "\t" << "(" << this->getNx()*this->getDelta() << "/" << this->getNy()*this->getDelta() << ")" << "\t\t==" << std::endl;
    std::cout << "==\tExtent (Nx,Ny):"  << "\t" << "(" << this->getNx() << "/" << this->getNy() << ")" << "\t==" << std::endl;
    std::cout << "==\tΔx:" << "\t" << "\t" << this->getDelta() << "\t==" << std::endl;
    std::cout << "==========================================\n" << std::endl;
}

#endif
