#ifndef Solver_H
#define Solver_H

##include "gridGeometry2D.h"

template<typename T>
class Solver {

private:
  /// Grid object
  Grid2D grid;
  /// Pointer to the collision model instance
  CollisionModel* collisionModel;
  /// LatticeGrid, containing distribution functions
  T *h_Collide;
  /// Device collide field pointer
  T *d_Collide=NULL;
  /// Device streaming field pointer
  T *d_Streaming=NULL;

public:
  /// Construction of a solver
  Solver(Grid2D grid);
  /// Destruction of a solver
  ~Solver();
  /// Initializes the solver
  void init(Grid2D grid);
  /// Initializes host fields
  InitializeHostFields();
  /// Initializes device fields
  InitializeHostFields();
  /// Output VTK
  OutputVTK();
  /// Time loop
  
  /// Prints solver details
  void print() const;
};

#endif
