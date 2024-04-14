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
    void checkOutput(unsigned int iter);

public:
    /// Constructor
    Simulation(std::unique_ptr<LBGrid<T,DESCRIPTOR>>&& lbgrid, std::unique_ptr<VTKWriter>&& vtkWriter, T dt, T simTime, unsigned int numberOutput);
  
    /// Destructor
    virtual ~Simulation();

    /// Print parameters
    void printParameters();

    /// Time loop
    void run();

    /// Simulation steps
    virtual void simulationSteps(unsigned int iter);
};


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T,typename DESCRIPTOR>
class LBFluidSimulation final : public Simulation<T,DESCRIPTOR> {
public:
    /// Constructor
    LBFluidSimulation(std::unique_ptr<LBGrid<T,DESCRIPTOR>>&& lbgrid, std::unique_ptr<VTKWriter>&& vtkWriter, T dt, T simTime, unsigned int numberOutput);

    /// Simulation steps
    void simulationSteps(unsigned int iter) override;
};

#include "simulation.hh"

#endif // SIMULATION_H
