#ifndef SIMULATION_H
#define SIMULATION_H

#include "gridGeometry.h"
#include "latticeGrid.h"

template<typename T>
class Simulation
{
protected:
    LatticeGrid<T>* latGrid;
    bool GPU_ENABLED;

public:
  /// Constructor
  Simulation(LatticeGrid<T>* latGrid, bool GPU=true);
  /// Destructor
  //~Simulation();
  /// Time loop
  virtual void performTimeStep() = 0;
  virtual void streamingStep() = 0;
  virtual void collisionStep() = 0;
};

template<typename T>
class LBMFluidSimulation : public Simulation<T>
{
public:
    /// Constructor
    LBMFluidSimulation(LatticeGrid<T>* latGrid, bool GPU=true);
    /// One time step
    void performTimeStep();
    /// Streaming of populations
    void streamingStep();
    /// Collision step
    void collisionStep();
};

#include "simulation.hh"

#endif
