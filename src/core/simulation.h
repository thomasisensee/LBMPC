#ifndef SIMULATION_H
#define SIMULATION_H

#include <stdio.h>

#include "lb/lbGrid.h"
#include "io/vtkWriter.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
class Simulation {
protected:
    const unsigned int _totalIter;
    const unsigned int _outputFrequency;
    std::unique_ptr<LBGrid<T>> _lbGrid; // For a single grid
    // std::vector<std::unique_ptr<LBGrid<T>>> _lbGrids; // For multiple grids

    std::unique_ptr<VTKWriter> _vtkWriter;

public:
    /// Constructor
    Simulation(std::unique_ptr<LBGrid<T>>&& lbgrid, std::unique_ptr<VTKWriter>&& vtkWriter, unsigned int totalIter, unsigned int outputFrequency);
  
    /// Destructor
    virtual ~Simulation();

    /// Time loop
    virtual void run() = 0;

    /// Check for output and trigger moment computation and output if ncesseary
    void checkOutput(unsigned int iter);
};


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
class LBFluidSimulation final : public Simulation<T> {
public:
    /// Constructor
    LBFluidSimulation(std::unique_ptr<LBGrid<T>>&& lbgrid, std::unique_ptr<VTKWriter>&& vtkWriter, unsigned int totalIter, unsigned int outputFrequency);

    /// Run simulation
    void run() override;
};

#include "simulation.hh"

#endif // SIMULATION_H
