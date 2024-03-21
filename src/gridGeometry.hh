#ifndef GridGeometry_HH
#define GridGeometry_HH

#include <iostream>
#include "gridGeometry.h"

template<typename T>
GridGeometry2D<T>::GridGeometry2D(T globPosX, T globPosY, T delta, int nX, int nY)
{
  init(globPosX, globPosY, delta, nX, nY);
}

template<typename T>
void GridGeometry2D<T>::init(T globPosX, T globPosY, T delta, int nX, int nY)
{
    _globPosX = globPosX;
    _globPosY = globPosY;
    _delta    = delta;
    _nX       = nX;
    _nY       = nY;
}

template<typename T>
T GridGeometry2D<T>::get_globPosX() const
{
    return _globPosX;
}

template<typename T>
T GridGeometry2D<T>::get_globPosY() const
{
    return _globPosY;
}

template<typename T>
T GridGeometry2D<T>::getDelta() const
{
    return _delta;
}

template<typename T>
unsigned int GridGeometry2D<T>::getNx() const
{
    return _nX;
}

template<typename T>
unsigned int GridGeometry2D<T>::getNy() const
{
    return _nY;
}

template<typename T>
unsigned int GridGeometry2D<T>::getGhostNx() const
{
    return _nX+2;
}

template<typename T>
unsigned int GridGeometry2D<T>::getGhostNy() const
{
    return _nY+2;
}

template<typename T>
void GridGeometry2D<T>::print() const
{
    std::cout << "============== Grid Details ==============" << std::endl;
    std::cout << "==\tOrigin (x,y):" << "\t" << "(" << this->get_globPosX() << "," << this->get_globPosY() << ")" << "\t\t==" << std::endl;
    std::cout << "==\tExtent (Nx,Ny):"  << "\t" << "(" << this->getNx() << "/" << this->getNy() << ")" << "\t==" << std::endl;
    std::cout << "==\tΔx:" << "\t" << "\t" << this->getDelta() << "\t\t==" << std::endl;
    std::cout << "==========================================\n" << std::endl;
}

#endif
