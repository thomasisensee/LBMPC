#ifndef Solver_HH
#define Solver_HH

#include "simulation.h"

template<typename T>
Simulation<T>::Simulation(std::unique_ptr<LBMGrid<T>>&& lbmgrid) : totalIter(0), outputFrequency(1), lbmGrid(std::move(lbmgrid))  { }

template<typename T>
LBMFluidSimulation<T>::LBMFluidSimulation(std::unique_ptr<LBMGrid<T>>&& lbmgrid) : Simulation<T>(std::move(lbmgrid)) { }

template<typename T>
void LBMFluidSimulation<T>::run() {
    for (unsigned int iter = 0; iter < this->totalIter; ++iter) {
        this->lbmGrid->performStreamingStep();
        this->lbmGrid->performCollisionStep();
        //apply boundary conditions
    }

}



#endif
