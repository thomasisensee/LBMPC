#ifndef Solver_HH
#define Solver_HH

#include "simulation.h"

template<typename T>
Simulation<T>::Simulation(LatticeGrid<T>* lGrid, bool GPU) : latGrid(lGrid)
{
    GPU_ENABLED = GPU;
}

/*
template<typename T>
Simulation<T>::~Simulation()
{

}
*/

template<typename T>
LBMFluidSimulation<T>::LBMFluidSimulation(LatticeGrid<T>* lGrid, bool GPU) : Simulation<T>(lGrid,GPU)
{

}

template<typename T>
void LBMFluidSimulation<T>::PerformTimeStep()
{

}

template<typename T>
void LBMFluidSimulation<T>::Streaming()
{

}

template<typename T>
void LBMFluidSimulation<T>::Collision()
{
#define pos(x,y)		(Nx*(y)+(x))

    unsigned int Q = this->latGrid->LBMModel->getQ();
    unsigned int Nx = this->latGrid->gridGeometry->getGhostNx();
    unsigned int Ny = this->latGrid->gridGeometry->getGhostNy();
    
    
}

#endif

