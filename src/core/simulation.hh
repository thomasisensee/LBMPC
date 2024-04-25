#ifndef SIMULAITION_HH
#define SIMULAITION_HH

#include <type_traits>

#include "simulation.h"
#include "cuda/cudaUtilities.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T,typename DESCRIPTOR>
Simulation<T,DESCRIPTOR>::Simulation(
    std::unique_ptr<LBGrid<T,DESCRIPTOR>>&& lbGrid,
    std::unique_ptr<VTKWriter>&& vtkWriter,
    T dt,
    T simTime,
    unsigned int numberOutput)
    : _lbGrid(std::move(lbGrid)), _vtkWriter(std::move(vtkWriter)), _dt(dt), _simTime(simTime), _totalIter(simTime / dt), _outputFrequency(simTime / dt / numberOutput), _outputCounter(0) {
        _vtkWriter->setNonInherent(this->_lbGrid->getGridGeometry().getNx(), this->_lbGrid->getGridGeometry().getNy(), static_cast<float>(this->_lbGrid->getGridGeometry().getDelta()));
    }

template<typename T,typename DESCRIPTOR>
void Simulation<T,DESCRIPTOR>::printOutput(unsigned int outputCounter) {
    std::ostringstream iterStr;
    iterStr.fill('0');  // Set the fill character for padding
    iterStr.width(5);   // Set the width. Adjust according to your needs
    iterStr << outputCounter;    // Insert the iteration number into the stream

    std::cout << "Output " << iterStr.str() << " written." << std::endl;
}

template<typename T,typename DESCRIPTOR>
void Simulation<T,DESCRIPTOR>::outputSimulationEndTime(float elapsedTimeMs) {
    std::string formattedTime = formatElapsedTime(elapsedTimeMs);
    std::cout << "==================================" << std::endl;
    std::cout << "== Simulation run time: " << formattedTime << "\t==" << std::endl;
    std::cout << "==================================" << std::endl;
}

template<typename T,typename DESCRIPTOR>
void Simulation<T,DESCRIPTOR>::checkOutput(unsigned int iter) {
    if (_outputFrequency && iter % _outputFrequency == 0) {
        _lbGrid->computeMoments();
        _lbGrid->fetchMoments();
        _vtkWriter->writeScalarField(_lbGrid->getHostZerothMoment(), "Rho", _outputCounter);

        // Check if DESCRIPTOR::TYPE is MomentumConservation
        if constexpr (std::is_same<typename DESCRIPTOR::TYPE, MomentumConservation>::value) {
            _vtkWriter->writeVectorField(_lbGrid->getHostFirstMoment(), _lbGrid->getHostZerothMoment(), "Vel", _dt, _outputCounter);
        }

        printOutput(_outputCounter);
        _outputCounter++;
    }
}

template<typename T,typename DESCRIPTOR>
void Simulation<T,DESCRIPTOR>::printGridParameters() {
    // Print grid parameters
    _lbGrid->printParameters();
}

template<typename T,typename DESCRIPTOR>
void Simulation<T,DESCRIPTOR>::printParameters() {
    // Call member functions to print parameters
    this->printGridParameters();

    // Print simulation specific parameters
    this->printSimulationParameters();
}

template<typename T,typename DESCRIPTOR>
void Simulation<T,DESCRIPTOR>::printSimulationParameters() {
    // Print simulation specific parameters
    std::cout << "=========== Simulation ===========" << std::endl;
    std::cout << "== Simulation time: " << _simTime << "\t\t==" << std::endl;
    std::cout << "== Time step: " << _dt << "\t==" << std::endl;
    std::cout << "== Total iterations: " << _totalIter << "\t==" << std::endl;
    std::cout << "== Output frequency: " << _outputFrequency << "\t==" << std::endl;
    std::cout << "== Total outputs: " << _totalIter/_outputFrequency << "\t\t==" << std::endl;
    std::cout << "==================================" << std::endl << std::endl;
}

template<typename T,typename DESCRIPTOR>
void Simulation<T,DESCRIPTOR>::run() {
    CUDATimer timer;

    // Start the timer
    timer.startTimer();

	this->checkOutput(0);
 
    for (unsigned int iter = 1; iter <= this->_totalIter; ++iter) {
        this->simulationSteps(iter);
    }

    // Stop the timer
    timer.stopTimer();

    // Get the elapsed time and output on screen
    float elapsedTimeMs = timer.getElapsedTime();
    this->outputSimulationEndTime(elapsedTimeMs);
}

template<typename T,typename DESCRIPTOR>
void Simulation<T,DESCRIPTOR>::simulationSteps(unsigned int iter) {
    std::cerr << "Simulation steps not implemented." << std::endl;
}

/*****************************************/
/***** Derived class 01: Single grid *****/
/*****************************************/
template<typename T,typename DESCRIPTOR>
LBFluidSimulation<T,DESCRIPTOR>::LBFluidSimulation(
    std::unique_ptr<LBGrid<T,DESCRIPTOR>>&& lbGrid,
    std::unique_ptr<VTKWriter>&& vtkWriter,
    T dt,
    T simTime,
    unsigned int numberOutput)
    : Simulation<T,DESCRIPTOR>(std::move(lbGrid), std::move(vtkWriter), dt, simTime, numberOutput) {}

template<typename T,typename DESCRIPTOR>
void LBFluidSimulation<T,DESCRIPTOR>::simulationSteps(unsigned int iter) {
    this->_lbGrid->applyBoundaryConditions();
    this->_lbGrid->performStreamingStep();
    this->_lbGrid->performCollisionStep();
    this->checkOutput(iter);
}

/******************************************************************************/
/***** Derived class 02: Coupled momentum - thermal simulation, two grids *****/
/******************************************************************************/
template<typename T,typename MOMENTUM_DESCRIPTOR,typename THERMAL_DESCRIPTOR>
LBCoupledSimulation<T,MOMENTUM_DESCRIPTOR,THERMAL_DESCRIPTOR>::LBCoupledSimulation(
    std::unique_ptr<LBGrid<T,MOMENTUM_DESCRIPTOR>>&& lbGridMomentum,
    std::unique_ptr<LBGrid<T,THERMAL_DESCRIPTOR>>&& lbGridThermal,
    std::unique_ptr<VTKWriter>&& vtkWriter,
    T dt,
    T simTime,
    unsigned int numberOutput)
    : Simulation<T,MOMENTUM_DESCRIPTOR>(std::move(lbGridMomentum), std::move(vtkWriter), dt, simTime, numberOutput), _lbGridThermal(std::move(lbGridThermal)) {}

template<typename T,typename MOMENTUM_DESCRIPTOR,typename THERMAL_DESCRIPTOR>
void LBCoupledSimulation<T,MOMENTUM_DESCRIPTOR,THERMAL_DESCRIPTOR>::simulationSteps(unsigned int iter) {
    this->_lbGrid->applyBoundaryConditions();
    this->_lbGridThermal->applyBoundaryConditions();
    this->_lbGrid->performStreamingStep();
    this->_lbGridThermal->performStreamingStep();
    this->_lbGrid->computeVelocity();
    this->_lbGridThermal->computeZerothMoment();
    this->_lbGrid->performCollisionStep();
    this->_lbGridThermal->performCollisionStep();
    this->checkOutput(iter);
}

template<typename T,typename MOMENTUM_DESCRIPTOR,typename THERMAL_DESCRIPTOR>
void LBCoupledSimulation<T,MOMENTUM_DESCRIPTOR,THERMAL_DESCRIPTOR>::printParameters() {
    // Call base class to print parameters of momentum grid
    Simulation<T,MOMENTUM_DESCRIPTOR>::printGridParameters();

    // Call member functions to print parameters of thermal grid
    _lbGridThermal->printParameters();

    // Call base class to print simulation parameters
    Simulation<T,MOMENTUM_DESCRIPTOR>::printSimulationParameters();
}

template<typename T,typename MOMENTUM_DESCRIPTOR,typename THERMAL_DESCRIPTOR>
void LBCoupledSimulation<T,MOMENTUM_DESCRIPTOR,THERMAL_DESCRIPTOR>::checkOutput(unsigned int iter) {
    if (this->_outputFrequency && iter % this->_outputFrequency == 0) {
        this->_lbGrid->computeMoments();
        this->_lbGrid->fetchMoments();
        this->_vtkWriter->writeScalarField(this->_lbGrid->getHostZerothMoment(), "Rho", this->_outputCounter);
        this->_vtkWriter->writeVectorField(this->_lbGrid->getHostFirstMoment(), this->_lbGrid->getHostZerothMoment(), "Vel", this->_dt, this->_outputCounter);

        _lbGridThermal->computeZerothMoment();
        _lbGridThermal->fetchZerothMoment();
        this->_vtkWriter->writeScalarField(_lbGridThermal->getHostZerothMoment(), "T", this->_outputCounter);

        this->printOutput(this->_outputCounter);
        this->_outputCounter++;
    }
}

#endif // SIMULAITION_HH