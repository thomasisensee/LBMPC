#include <stdio.h>

#include "core/lbmModel.h"
#include "core/gridGeometry.h"
#include "core/lbmGrid.h"
#include "cuda/cudaUtilities.h"
//#include "simulation.h"

using FLOATING_POINT_TYPE = double;
using T = FLOATING_POINT_TYPE;

int main( int argc, char* argv[] )
{

// =======================
// === Read input file ===
// =======================
SetDevice();

// ========================
// === Define LBM model ===
// ========================
D2Q9<T> lbmModel;
lbmModel.print();

// ===============================
// === Prepare domain geometry ===
// ===============================
GridGeometry2D<T> grid2D(0.,0.,.1,14,14);
grid2D.print();

LBMGrid<T> latticeGrid(&lbmModel,&grid2D);

//LBMFluidSimulation<T> sim(&latticeGridWrap);

// =========================
// === Initialize fields ===
// =========================
//Cell2D<T> cell;



//printf("Nx,Ny = (%d,%d)\n",grid.getNx(),grid.getNy());

return 0;
}
