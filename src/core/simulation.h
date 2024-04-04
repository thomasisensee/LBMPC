#ifndef SIMULATION_H
#define SIMULATION_H

#include <stdio.h>

#include "lbGrid.h"
#include "io/vtkWriter.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T>
class Simulation {
protected:
    const unsigned int _totalIter;
    std::unique_ptr<LBGrid<T>> _lbGrid; // For a single grid
    // std::vector<std::unique_ptr<LBGrid<T>>> _lbGrids; // For multiple grids

    std::unique_ptr<VTKWriter> _vtkWriter;

public:
    /// Constructor
    Simulation(std::unique_ptr<LBGrid<T>>&& lbgrid, std::unique_ptr<VTKWriter>&& vtkWriter);
  
    /// Destructor
    virtual ~Simulation();

    /// Time loop
    virtual void run() = 0;
};


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
class LBFluidSimulation final : public Simulation<T> {
public:
    /// Constructor
    LBFluidSimulation(std::unique_ptr<LBGrid<T>>&& lbgrid, std::unique_ptr<VTKWriter>&& vtkWriter);

    /// Collision step
    void run();
};

#include "simulation.hh"

#endif // SIMULATION_H
