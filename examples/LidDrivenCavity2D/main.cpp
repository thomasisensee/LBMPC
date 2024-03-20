#include <stdio.h>
#include "gridGeometry2D.h"
#include "gridGeometry2D.hh"

using FLOATING_POINT_TYPE = double;
using T = FLOATING_POINT_TYPE;

int main( int argc, char* argv[] )
{

// =======================
// === Read input file ===
// =======================


// ======================
// === Prepare domain ===
// ======================
GridGeometry2D grid = GridGeometry2D<T>(0.,0.,.1,100,100);
grid.print();

// =========================
// === Initialize fields ===
// =========================


//printf("Nx,Ny = (%d,%d)\n",grid.getNx(),grid.getNy());

}
