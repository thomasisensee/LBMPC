#ifndef GRID_GEOMETRY_1D_HH
#define GRID_GEOMETRY_1D_HH

#include "gridGeometry1D.h"

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
    return this->getNx();
}

template<typename T>
unsigned int GridGeometry1D<T>::getGhostVolume() const {
    return this->getGhostNx();
}

template<typename T>
void GridGeometry1D<T>::printParameters() const {
    std::cout << "============== Grid Details ==============" << std::endl;
    std::cout << "==\tExtent Lx:"  << "\t" << this->getNx()*this->getDelta() << "\t\t==" << std::endl;
    std::cout << "==\tExtent Nx:"  << "\t" << this->getNx()  << "\t==" << std::endl;
    std::cout << "==\tΔx:" << "\t" << "\t" << this->getDelta() << "\t==" << std::endl;
    std::cout << "==========================================\n" << std::endl;
}

#endif // GRID_GEOMETRY_1D_HH