#ifndef SIMULATION_H
#define SIMULATION_H

#include "gridGeometry.h"
#include "lbmGrid.h"

template<typename T>
class Simulation
{
protected:
    LBMGridWrapper<T>* lbmGrid;
    bool GPU_ENABLED;

public:
  /// Constructor
  Simulation(LBMGridWrapper<T>* lbmGrid, bool GPU=true);
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
    LBMFluidSimulation(LBMGridWrapper<T>* lbmGrid, bool GPU=true);
    /// One time step
    void performTimeStep();
    /// Streaming of populations
    void streamingStep();
    /// Collision step
    void collisionStep();
};

#include "simulation.hh"

#endif
