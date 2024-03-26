#ifndef SIMULATION_H
#define SIMULATION_H

#include <stdio.h>
//#include <memory> // For std::unique_ptr and std::make_unique

#include "lbmGrid.h"

template<typename T>
class Simulation {
protected:
    unsigned int totalIter;
    unsigned int outputFrequency;
    std::unique_ptr<LBMGrid<T>> lbmGrid; // For a single grid
    // std::vector<std::unique_ptr<LBMGrid<T>>> lbmGrids; // For multiple grids

public:
  /// Constructor
  Simulation(std::unique_ptr<LBMGrid<T>>&& lbmgrid);
  /// Destructor
  //~Simulation();
  /// Time loop
  virtual void run() = 0;
};

template<typename T>
class LBMFluidSimulation : public Simulation<T> {
public:
    /// Constructor
    LBMFluidSimulation(std::unique_ptr<LBMGrid<T>>&& lbmgrid);
    /// Collision step
    void run();
};

#include "simulation.hh"

#endif
