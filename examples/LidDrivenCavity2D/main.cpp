#include <stdio.h>


#include "lbmModel.h"
#include "lbmModel.hh"
#include "gridGeometry.h"
#include "gridGeometry.hh"
#include "cell.h"
#include "cell.hh"
#include "latticeGrid.h"
#include "latticeGrid.hh"

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
GridGeometry2D grid = GridGeometry2D<T>(0.,0.,.1,100,100);
grid.print();


Cell cell = Cell<T,D2Q9<T>>(&lbm);
LatticeGrid latticeGrid = LatticeGrid<T,D2Q9<T>,GridGeometry2D<T>>(&lbm,&grid);

// =========================
// === Initialize fields ===
// =========================
//Cell2D cell = Cell2D<T>();



//printf("Nx,Ny = (%d,%d)\n",grid.getNx(),grid.getNy());

}
