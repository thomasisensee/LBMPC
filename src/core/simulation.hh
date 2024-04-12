#ifndef SIMULAITION_HH
#define SIMULAITION_HH

#include "simulation.h"
#include "cuda/cudaUtilities.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T, unsigned int D, unsigned int Q>
Simulation<T,D,Q>::Simulation(
    std::unique_ptr<LBGrid<T,D,Q>>&& lbgrid,
    std::unique_ptr<VTKWriter>&& vtkWriter,
    T dt,
    T simTime,
    unsigned int numberOutput)
    : _lbGrid(std::move(lbgrid)), _vtkWriter(std::move(vtkWriter)), _dt(dt), _simTime(simTime), _totalIter(simTime / dt), _outputFrequency(simTime / dt / numberOutput), _outputCounter(0) {
        _vtkWriter->setNonInherent(this->_lbGrid->getGridGeometry().getNx(), this->_lbGrid->getGridGeometry().getNy(), static_cast<float>(this->_lbGrid->getGridGeometry().getDelta()));
    }

template<typename T, unsigned int D, unsigned int Q>
Simulation<T,D,Q>::~Simulation() {}

template<typename T, unsigned int D, unsigned int Q>
void Simulation<T,D,Q>::printOutput(unsigned int outputCounter) {
    std::ostringstream iterStr;
    iterStr.fill('0');  // Set the fill character for padding
    iterStr.width(5);   // Set the width. Adjust according to your needs
    iterStr << outputCounter;    // Insert the iteration number into the stream

    std::cout << "Output " << iterStr.str() << " written." << std::endl;
}

template<typename T, unsigned int D, unsigned int Q>
void Simulation<T,D,Q>::outputSimulationEndTime(float elapsedTimeMs) {
    std::string formattedTime = formatElapsedTime(elapsedTimeMs);
    std::cout << "==================================" << std::endl;
    std::cout << "== Simulation run time: " << formattedTime << "\t==" << std::endl;
    std::cout << "==================================" << std::endl;
}

template<typename T, unsigned int D, unsigned int Q>
void Simulation<T,D,Q>::checkOutput(unsigned int iter) {
    //if (_outputFrequency && iter % _outputFrequency == 0) {
     {
        _lbGrid->computeMoments();
        _vtkWriter->writeScalarField(_lbGrid->getHostZerothMoment(), "Rho", _outputCounter);
        _vtkWriter->writeVectorField(_lbGrid->getHostFirstMoment(), _lbGrid->getHostZerothMoment(), "Vel", _dt, _outputCounter);
        printOutput(_outputCounter);
        _outputCounter++;
    }
}

template<typename T, unsigned int D, unsigned int Q>
void Simulation<T,D,Q>::printParameters() {
    // Call member functions to print parameters
    _lbGrid->printParameters();

    // Print simulation specific parameters
    std::cout << "=========== Simulation ===========" << std::endl;
    std::cout << "== Simulation time: " << _simTime << "\t\t==" << std::endl;
    std::cout << "== Time step: " << _dt << "\t==" << std::endl;
    std::cout << "== Total iterations: " << _totalIter << "\t==" << std::endl;
    std::cout << "== Output frequency: " << _outputFrequency << "\t==" << std::endl;
    std::cout << "== Total outputs: " << _totalIter/_outputFrequency << "\t\t==" << std::endl;
    std::cout << "==================================" << std::endl << std::endl;
}

template<typename T, unsigned int D, unsigned int Q>
void Simulation<T,D,Q>::run() {
    CUDATimer timer;

    // Start the timer
    timer.startTimer();

	this->checkOutput(0);
 
    //for (unsigned int iter = 1; iter <= this->_totalIter; ++iter) {
    for (unsigned int iter = 1; iter <= 3; ++iter) {
        this->simulationSteps(iter);
    }

    // Stop the timer
    timer.stopTimer();

    // Get the elapsed time and output on screen
    float elapsedTimeMs = timer.getElapsedTime();
    this->outputSimulationEndTime(elapsedTimeMs);
}

template<typename T, unsigned int D, unsigned int Q>
void Simulation<T,D,Q>::simulationSteps(unsigned int iter) {
    std::cerr << "Simulation steps not implemented." << std::endl;
}

/********************************************************************************/
/***** Derived class 01: Simple fluid simulation without any external force *****/
/********************************************************************************/
template<typename T, unsigned int D, unsigned int Q>
LBFluidSimulation<T,D,Q>::LBFluidSimulation(
    std::unique_ptr<LBGrid<T,D,Q>>&& lbgrid,
    std::unique_ptr<VTKWriter>&& vtkWriter,
    T dt,
    T simTime,
    unsigned int numberOutput)
    : Simulation<T,D,Q>(std::move(lbgrid), std::move(vtkWriter), dt, simTime, numberOutput) {}

template<typename T, unsigned int D, unsigned int Q>
void LBFluidSimulation<T,D,Q>::simulationSteps(unsigned int iter) {
        this->_lbGrid->applyBoundaryConditions();
        this->_lbGrid->performStreamingStep();
        this->_lbGrid->performCollisionStep();
        this->checkOutput(iter);
}

#endif // SIMULAITION_HH