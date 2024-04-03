#ifndef SIMULAITION_HH
#define SIMULAITION_HH

#include "simulation.h"


/**********************/
/***** Base class *****/
/**********************/
template<typename T>
Simulation<T>::Simulation(std::unique_ptr<LBGrid<T>>&& lbgrid) : _totalIter(10), _outputFrequency(1), _lbGrid(std::move(lbgrid))  {}

template<typename T>
Simulation<T>::~Simulation() {}


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
LBFluidSimulation<T>::LBFluidSimulation(std::unique_ptr<LBGrid<T>>&& lbgrid) : Simulation<T>(std::move(lbgrid)) {}

template<typename T>
void LBFluidSimulation<T>::run() {
	
    for (unsigned int iter = 0; iter < this->_totalIter; ++iter) {
        this->_lbGrid->applyBoundaryConditions();
        this->_lbGrid->performStreamingStep();
        this->_lbGrid->performCollisionStep();
    }
}



#endif // SIMULAITION_HH
