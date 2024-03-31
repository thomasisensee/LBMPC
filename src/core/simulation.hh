#ifndef SIMULAITION_HH
#define SIMULAITION_HH

#include "simulation.h"


/**********************/
/***** Base class *****/
/**********************/
template<typename T>
Simulation<T>::Simulation(std::unique_ptr<LBGrid<T>>&& lbgrid) : totalIter(1), outputFrequency(1), lbGrid(std::move(lbgrid))  {}


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
LBFluidSimulation<T>::LBFluidSimulation(std::unique_ptr<LBGrid<T>>&& lbgrid) : Simulation<T>(std::move(lbgrid)) {}

template<typename T>
void LBFluidSimulation<T>::run() {
	
    for (unsigned int iter = 0; iter < this->totalIter; ++iter) {
        this->lbGrid->performStreamingStep();
        this->lbGrid->performCollisionStep();
        this->lbGrid->applyBoundaryConditions();
    }
}



#endif // SIMULAITION_HH
