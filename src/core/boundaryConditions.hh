#ifndef BOUNDARY_CONDITIONS_HH
#define BOUNDARY_CONDITIONS_HH

#include "boundaryConditions.h"

void BounceBack::applyBoundaryCondition(LBMGrid& grid, int x, int y)
{

}

void FixedVelocityBoundary::applyBoundaryCondition(LBMGrid& grid, int x, int y)
{

}

void PeriodicBoundary::applyBoundaryCondition(LBMGrid& grid, int x, int y)
{

}

template<typename T>
void BoundaryConditionManager<T>::addBoundaryCondition(const std::string& name, std::unique_ptr<BoundaryCondition> condition)
{
    boundaryConditions[name] = std::move(condition);
}

template<typename T>
void BoundaryConditionManager<T>::apply(const std::string& name, int x, int y)
{
    if (boundaryConditions.find(name) != boundaryConditions.end())
    {
        boundaryConditions[name]->applyBoundaryCondition(grid, x, y);
    }
}

#endif
