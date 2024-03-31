#ifndef SIMULATION_H
#define SIMULATION_H

#include <stdio.h>
//#include <memory> // For std::unique_ptr and std::make_unique

#include "lbGrid.h"


/**********************/
/***** Base class *****/
/**********************/
template<typename T>
class Simulation {
protected:
    unsigned int totalIter;
    unsigned int outputFrequency;
    std::unique_ptr<LBGrid<T>> lbGrid; // For a single grid
    // std::vector<std::unique_ptr<LBGrid<T>>> lbGrids; // For multiple grids

public:
  /// Constructor
  Simulation(std::unique_ptr<LBGrid<T>>&& lbgrid);
  /// Time loop
  virtual void run() = 0;
};


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
class LBFluidSimulation : public Simulation<T> {
public:
    /// Constructor
    LBFluidSimulation(std::unique_ptr<LBGrid<T>>&& lbgrid);
    /// Collision step
    void run();
};

#include "simulation.hh"

#endif // SIMULATION_H
