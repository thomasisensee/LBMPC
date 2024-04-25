#ifndef SIMULATION_H
#define SIMULATION_H

#include <stdio.h>

#include "lb/lbGrid.h"
#include "io/vtkWriter.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T,typename DESCRIPTOR>
class Simulation {
protected:
    /// constant members
    const T _dt;
    const T _simTime;
    const unsigned int _totalIter;
    const unsigned int _outputFrequency;
    std::unique_ptr<LBGrid<T,DESCRIPTOR>> _lbGrid;
    std::unique_ptr<VTKWriter> _vtkWriter;

    /// Output counter
    unsigned int _outputCounter;

    /// Print output iteration to screen
    void printOutput(unsigned int outputCounter);

    /// Print simulation run time to screen
    void outputSimulationEndTime(float elapsedTimeMs);

    /// Check for output and trigger moment computation and output if ncesseary
    virtual void checkOutput(unsigned int iter);

public:
    /// Constructor
    Simulation(std::unique_ptr<LBGrid<T,DESCRIPTOR>>&& lbgrid, std::unique_ptr<VTKWriter>&& vtkWriter, T dt, T simTime, unsigned int numberOutput);
  
    /// Destructor
    virtual ~Simulation() = default;

    /// Print parameters
    virtual void printParameters();

    /// Print parameters
    void printGridParameters();

    /// Print parameters
    void printSimulationParameters();

    /// Time loop
    void run();

    /// Simulation steps
    virtual void simulationSteps(unsigned int iter);
};


/*****************************************/
/***** Derived class 01: Single grid *****/
/*****************************************/
template<typename T,typename DESCRIPTOR>
class LBFluidSimulation final : public Simulation<T,DESCRIPTOR> {
public:
    /// Constructor
    LBFluidSimulation(std::unique_ptr<LBGrid<T,DESCRIPTOR>>&& lbGrid, std::unique_ptr<VTKWriter>&& vtkWriter, T dt, T simTime, unsigned int numberOutput);

    /// Simulation steps
    void simulationSteps(unsigned int iter) override;
};

/******************************************************************************/
/***** Derived class 02: Coupled momentum - thermal simulation, two grids *****/
/******************************************************************************/
template<typename T,typename MOMENTUM_DESCRIPTOR,typename THERMAL_DESCRIPTOR>
class LBCoupledSimulation final : public Simulation<T,MOMENTUM_DESCRIPTOR> {
protected:
    std::unique_ptr<LBGrid<T,THERMAL_DESCRIPTOR>> _lbGridThermal;

public:
    /// Constructor
    LBCoupledSimulation(std::unique_ptr<LBGrid<T,MOMENTUM_DESCRIPTOR>>&& lbGridMomentum, std::unique_ptr<LBGrid<T,THERMAL_DESCRIPTOR>>&& lbGridThermal, std::unique_ptr<VTKWriter>&& vtkWriter, T dt, T simTime, unsigned int numberOutput);

    /// Simulation steps
    void simulationSteps(unsigned int iter) override;

    /// Print parameters
    void printParameters() override;

    /// Check for output and trigger moment computation and output if ncesseary
    void checkOutput(unsigned int iter) override;
};

#include "simulation.hh"

#endif // SIMULATION_H
