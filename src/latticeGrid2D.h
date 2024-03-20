#ifndef LatticeGrid_2D_H
#define LatticeGrid_2D_H

#include "gridGeometry2D.h"

template<typename T>
class LatticeGrid {
private:
    /// Grid geometry (spacing, Nx, Ny, etc.)
    GridGeometry2D gridGeometry;
    /// Distribution functions f, i.e., a grid of cells
    std::vector<CellD2Q9> data; 

public:
    /// Construction of a lattice grid
    LatticeGrid(GridGeometry2D gridGeometry);

};

#endif
