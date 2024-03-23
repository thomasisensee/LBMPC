#ifndef Solver_HH
#define Solver_HH

#include "simulation.h"

template<typename T>
Simulation<T>::Simulation(LatticeGrid<T>* lGrid, bool GPU) : latGrid(lGrid)
{
    GPU_ENABLED = GPU;
}

template<typename T>
LBMFluidSimulation<T>::LBMFluidSimulation(LatticeGrid<T>* lGrid, bool GPU) : Simulation<T>(lGrid,GPU)
{

}

template<typename T>
void LBMFluidSimulation<T>::performTimeStep()
{

}

template<typename T>
void LBMFluidSimulation<T>::streamingStep()
{

}

template<typename T>
void LBMFluidSimulation<T>::collisionStep()
{
#define pos(x,y)		(Nx*(y)+(x))

    unsigned int Q = this->latGrid->lbmModel->getQ();
    unsigned int Nx = this->latGrid->gridGeometry->getGhostNx();
    unsigned int Ny = this->latGrid->gridGeometry->getGhostNy();
    
    
}

#endif

