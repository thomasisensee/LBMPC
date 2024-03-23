#ifndef LatticeGrid_HH
#define LatticeGrid_HH

#include <stdio.h>

#include "latticeGrid.h"
#include "cuda.h"

template<typename T>
LatticeGrid<T>::LatticeGrid(LBMModel<T>* lbmModel, GridGeometry2D<T>* gridGeometry, bool GPU) : lbmModel(lbmModel), gridGeometry(gridGeometry)
{
    GPU_ENABLED = GPU;
    this->h_distribution = new T[this->lbmModel->getQ() * this->gridGeometry->getGhostVolume()];
    
    if(GPU_ENABLED)
    {
        allocateDeviceField<T>(&d_collision, this->lbmModel->getQ() * this->gridGeometry->getGhostVolume()*sizeof(T));
        KERNEL_CALLER_initializeLBMDistributions<T>(d_collision,lbmModel,gridGeometry);
    }
    else
    {
        initializeLBMDistributionsCPU(h_distribution);   
    }
}

template<typename T>
LatticeGrid<T>::~LatticeGrid()
{
    delete[] this->h_distribution;
    if(this->GPU_ENABLED)
    {
        freeDeviceField<T>(d_collision);
        freeDeviceField<T>(d_streaming);
    }
}

template<typename T>
void LatticeGrid<T>::initializeLBMDistributionsCPU(T* h_data)
{
#define pos(x,y)		(Nx*(y)+(x))

    unsigned int Q = this->lbmModel->getQ();
    unsigned int Nx = this->gridGeometry->getGhostNx();
    unsigned int Ny = this->gridGeometry->getGhostNy();
	T firstOrder, secondOrder, thirdOrder, fourthOrder;

	for(int l=0; l<Q; l++)
	{
	    for(int i=0; i<Nx; i++)
	    {
	        for(int j=0; j<Ny; j++)
	        {
                firstOrder = 0.;
                secondOrder = 0.;
                thirdOrder = 0.;
                fourthOrder = 0.;
   		        h_data[Q*pos(i,j)+l] = this->lbmModel->getWEIGHT(l)*(1. + firstOrder + secondOrder + thirdOrder + fourthOrder);
		    }
		}
	}

}

#endif
