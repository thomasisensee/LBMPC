#ifndef LatticeGrid_HH
#define LatticeGrid_HH

#include <stdio.h>

#include "constants.h"

template<typename T, typename lbmModelType, typename gridGeometryType>
LatticeGrid<T,lbmModelType,gridGeometryType>::LatticeGrid(lbmModelType* model,gridGeometryType* grid) : lbmModel(model),gridGeometry(grid)
{
    this->h_distribution = new T[this->lbmModel->getQ() * this->gridGeometry->getGhostNx() * this->gridGeometry->getGhostNy()];
    InitializeCPU(h_distribution);
}

template<typename T, typename lbmModelType, typename gridGeometryType>
LatticeGrid<T,lbmModelType,gridGeometryType>::~LatticeGrid()
{
    delete[] this->h_distribution;
}

template<typename T, typename lbmModelType, typename gridGeometryType>
void LatticeGrid<T,lbmModelType,gridGeometryType>::InitializeCPU(T *h_data)
{
#define pos(x,y)		(Nx*(y)+(x))

    unsigned int Q = this->lbmModel->getQ();
    unsigned int Nx = this->gridGeometry->getGhostNx();
    unsigned int Ny = this->gridGeometry->getGhostNy();
	T first_order;
	T second_order;
	T third_order;
    T fourth_order;

	for(int l=0; l<Q; l++)
	{
	    for(int i=0; i<Nx; i++)
	    {
	        for(int j=0; j<Ny; j++)
	        {
                first_order = 0.;
                second_order = 0.;
                third_order = 0.;
                fourth_order = 0.;
   		        h_data[Q*pos(i,j)+l] = this->lbmModel->getWEIGHT(l)*(1. + first_order + second_order + third_order + fourth_order);
		    }
		}
	}

}

#endif
