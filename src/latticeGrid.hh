#ifndef LatticeGrid_HH
#define LatticeGrid_HH

#include <stdio.h>

#include "latticeGrid.h"

template<typename T>
LatticeGrid<T>::LatticeGrid(lbmModel<T>* model, GridGeometry2D<T>* grid, bool GPU) : LBMModel(model), gridGeometry(grid)
{
    GPU_ENABLED = GPU;
    this->h_distribution = new T[this->LBMModel->getQ() * this->gridGeometry->getGhostVolume()];
    
    if(GPU_ENABLED)
    {
        AllocateDeviceField<T>(&d_collision, this->LBMModel->getQ());
    }
    else
    {
        InitializeCPU(h_distribution);   
    }
}

template<typename T>
LatticeGrid<T>::~LatticeGrid()
{
    delete[] this->h_distribution;
    if(this->GPU_ENABLED)
    {
        FreeDeviceField<T>(d_collision);
    }
}

template<typename T>
void LatticeGrid<T>::InitializeCPU(T* h_data)
{
#define pos(x,y)		(Nx*(y)+(x))

    unsigned int Q = this->LBMModel->getQ();
    unsigned int Nx = this->gridGeometry->getGhostNx();
    unsigned int Ny = this->gridGeometry->getGhostNy();
	T first_order, second_order, third_order, fourth_order;

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
   		        h_data[Q*pos(i,j)+l] = this->LBMModel->getWEIGHT(l)*(1. + first_order + second_order + third_order + fourth_order);
		    }
		}
	}

}

#endif
