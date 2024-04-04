#ifndef SIMULAITION_HH
#define SIMULAITION_HH

#include "simulation.h"


/**********************/
/***** Base class *****/
/**********************/
template<typename T>
Simulation<T>::Simulation(
    std::unique_ptr<LBGrid<T>>&& lbgrid,
    std::unique_ptr<VTKWriter>&& vtkWriter)
    : _totalIter(10), _lbGrid(std::move(lbgrid)), _vtkWriter(std::move(vtkWriter))  {}

template<typename T>
Simulation<T>::~Simulation() {}


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
LBFluidSimulation<T>::LBFluidSimulation(
    std::unique_ptr<LBGrid<T>>&& lbgrid,
    std::unique_ptr<VTKWriter>&& vtkWriter)
    : Simulation<T>(std::move(lbgrid), std::move(vtkWriter)) {}

template<typename T>
void LBFluidSimulation<T>::run() {
	
    for (unsigned int iter = 0; iter < this->_totalIter; ++iter) {
        this->_lbGrid->applyBoundaryConditions();
        this->_lbGrid->performStreamingStep();
        this->_lbGrid->performCollisionStep();
        this->_vtkWriter->update(iter, this->_lbGrid->getHostDistributions(), this->_lbGrid->getDeviceCollision());
    }
}



#endif // SIMULAITION_HH
