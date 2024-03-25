#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <stdio.h>

class BoundaryCondition {
public:
    virtual void applyBoundaryCondition(LBMGrid& grid, int x, int y) = 0;
    virtual ~BoundaryCondition() = default;
};

class BounceBack : public BoundaryCondition {
public:
    void applyBoundaryCondition(LBMGrid& grid, int x, int y) override;
};

template<typename T>
class FixedVelocityBoundary : public BoundaryCondition {
    std::vector<T> velocity;
public:
    FixedVelocityBoundary(const std::vector<T>& velocity) : velocity(velocity) {}
    void applyBoundaryCondition(LBMGrid& grid, int x, int y) override;
};

class PeriodicBoundary : public BoundaryCondition {
public:
    void applyBoundaryCondition(LBMGrid& grid, int x, int y) override;
};

class BoundaryConditionManager {
    std::map<std::string, std::unique_ptr<BoundaryCondition>> boundaryConditions;
    LBMGrid& grid;

public:
    BoundaryConditionManager(LBMGrid& grid) : grid(grid) {}
    void addBoundaryCondition(const std::string& name, std::unique_ptr<BoundaryCondition> condition);
    void apply(const std::string& name, int x, int y);
};

#include "boundaryConditions.hh"

#endif
