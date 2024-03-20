#ifndef GridGeometry_2D_HH
#define GridGeometry_2D_HH

#include <iostream>
#include "gridGeometry2D.h"

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
int GridGeometry2D<T>::getNx() const
{
    return _nX;
}

template<typename T>
int GridGeometry2D<T>::getNy() const
{
    return _nY;
}

template<typename T>
void GridGeometry2D<T>::print() const
{
    std::cout << "================== Grid Details ==================" << std::endl;
    std::cout << "==\tOrigin (x,y): " << "\t\t" << "(" << this->get_globPosX() << "," << this->get_globPosY() << ")" << "\t\t==" << std::endl;
    std::cout << "==\tExtent (Nx,Ny): "  << "\t" << "(" << this->getNx() << "/" << this->getNy() << ")" << "\t==" << std::endl;
    std::cout << "==\tΔx: " << "\t" << "\t" << "\t" << this->getDelta() << "\t\t==" << std::endl;
    std::cout << "==================================================" << std::endl;
}

#endif
