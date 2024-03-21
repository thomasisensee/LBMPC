#ifndef SIMULATION_H
#define SIMULATION_H

#include "gridGeometry.h"
#include "latticeGrid.h"
#include "latticeGrid.hh"

template<typename T>
class Simulation
{
protected:
    LatticeGrid<T>* latGrid;
    bool GPU_ENABLED;

public:
  /// Constructor
  Simulation(LatticeGrid<T>* latGrid,bool GPU=true);
  /// Destructor
  //~Simulation();
  /// Time loop
  virtual void PerformTimeStep() = 0;
  virtual void Streaming() = 0;
  virtual void Collision() = 0;
};

template<typename T>
class LBMFluidSimulation : public Simulation<T>
{
public:
    /// Constructor
    LBMFluidSimulation(LatticeGrid<T>* latGrid,bool GPU=true);
    /// One time step
    void PerformTimeStep();
    /// Streaming of populations
    void Streaming();
    /// Collision step
    void Collision();
};

#endif
