#include <stdio.h>

#include "lbmModel.h"
#include "lbmModel.hh"
#include "gridGeometry.h"
#include "gridGeometry.hh"
#include "cell.h"
#include "cell.hh"
#include "latticeGrid.h"
#include "latticeGrid.hh"
#include "simulation.h"
#include "simulation.hh"

using FLOATING_POINT_TYPE = double;
using T = FLOATING_POINT_TYPE;

int main( int argc, char* argv[] )
{

// =======================
// === Read input file ===
// =======================

// ========================
// === Define LBM model ===
// ========================
D2Q9 lbm = D2Q9<T>();
lbm.print();

// ===============================
// === Prepare domain geometry ===
// ===============================
GridGeometry2D grid2D = GridGeometry2D<T>(0.,0.,.1,100,100);
grid2D.print();


Cell cell = Cell<T,D2Q9<T>>(&lbm);
LatticeGrid latticeGrid = LatticeGrid<T>(&lbm,&grid2D);
LatticeGrid latticeGridT = LatticeGrid<T>(&lbm,&grid2D);

LBMFluidSimulation sim = LBMFluidSimulation<T>(&latticeGrid);

// =========================
// === Initialize fields ===
// =========================
//Cell2D cell = Cell2D<T>();



//printf("Nx,Ny = (%d,%d)\n",grid.getNx(),grid.getNy());

}
