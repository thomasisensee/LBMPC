#ifndef SIMULAITION_HH
#define SIMULAITION_HH

#include "simulation.h"


/**********************/
/***** Base class *****/
/**********************/
template<typename T>
Simulation<T>::Simulation(
    std::unique_ptr<LBGrid<T>>&& lbgrid,
    std::unique_ptr<VTKWriter>&& vtkWriter,
    unsigned int totalIter,
    unsigned int outputFrequency)
    : _totalIter(totalIter), _outputFrequency(outputFrequency),_lbGrid(std::move(lbgrid)), _vtkWriter(std::move(vtkWriter)) {
        _vtkWriter->setNonInherent(this->_lbGrid->getGridGeometry().getNx(), this->_lbGrid->getGridGeometry().getNy(), static_cast<float>(this->_lbGrid->getGridGeometry().getDelta()));
    }

template<typename T>
Simulation<T>::~Simulation() {}


template<typename T>
void Simulation<T>::checkOutput(unsigned int iter) {
    if (_outputFrequency && iter % _outputFrequency == 0) {
        _lbGrid->computeMoments();
        _vtkWriter->writeScalarField(_lbGrid->getHostZerothMoment(), "Rho", iter);
        _vtkWriter->writeVectorField(_lbGrid->getHostFirstMoment(), "Vel", iter);
    }
}


/********************************************************************************/
/***** Derived class 01: Simple fluid simulation without any external force *****/
/********************************************************************************/
template<typename T>
LBFluidSimulation<T>::LBFluidSimulation(
    std::unique_ptr<LBGrid<T>>&& lbgrid,
    std::unique_ptr<VTKWriter>&& vtkWriter,
    unsigned int totalIter,
    unsigned int outputFrequency)
    : Simulation<T>(std::move(lbgrid), std::move(vtkWriter), totalIter, outputFrequency) {}

template<typename T>
void LBFluidSimulation<T>::run() {
	this->checkOutput(0);        
    for (unsigned int iter = 1; iter <= this->_totalIter; ++iter) {
        this->_lbGrid->applyBoundaryConditions();
        this->_lbGrid->performStreamingStep();
        this->_lbGrid->performCollisionStep();
        this->checkOutput(iter);
    }
}



#endif // SIMULAITION_HH
