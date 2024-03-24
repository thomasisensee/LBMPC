#ifndef Solver_HH
#define Solver_HH

#include "simulation.h"

template<typename T>
Simulation<T>::Simulation(LBMGridWrapper<T>* lbmGrid, bool GPU) : lbmGrid(lbmGrid), GPU_ENABLED(GPU) {}

template<typename T>
LBMFluidSimulation<T>::LBMFluidSimulation(LBMGridWrapper<T>* lbmGrid, bool GPU) : Simulation<T>(lbmGrid, GPU) {}

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

    //unsigned int Q = this->latGrid->lbmModel->getQ();
    //unsigned int Nx = this->latGrid->gridGeometry->getGhostNx();
    //unsigned int Ny = this->latGrid->gridGeometry->getGhostNy();
    
    
}

#endif

