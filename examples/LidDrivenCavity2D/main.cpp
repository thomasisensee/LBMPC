#include <stdio.h>

#include "lbmModel.h"
#include "gridGeometry.h"
#include "lbmGrid.h"
//#include "simulation.h"

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
LBMModelWrapper lbmWrap = LBMModelWrapper<T>(&lbm);

// ===============================
// === Prepare domain geometry ===
// ===============================
//GridGeometry2D grid2D = GridGeometry2D<T>(0.,0.,.1,14,14);
//GridGeometry2DWrapper grid2DWrap = GridGeometry2DWrapper(&grid2D);
//grid2D.print();

//LBMGrid latticeGrid = LBMGrid<T>(&lbmWrap,&grid2DWrap);
//LBMGridWrapper latticeGridWrap = LBMGridWrapper<T>(&latticeGrid);

//LBMFluidSimulation sim = LBMFluidSimulation<T>(&latticeGridWrap);

// =========================
// === Initialize fields ===
// =========================
//Cell2D cell = Cell2D<T>();



//printf("Nx,Ny = (%d,%d)\n",grid.getNx(),grid.getNy());

return 0;
}
