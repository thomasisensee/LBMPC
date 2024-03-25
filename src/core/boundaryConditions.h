#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <stdio.h>
#include <iostream>     // For std::cout
#include <map>          // For std::map
#include <memory>       // For std::unique_ptr
#include <string>       // Also include this if you're using std::string

enum class BoundaryType {
    North, South, East, West, Top, Bottom
};

template<typename T>
class BoundaryCondition {
public:
    virtual void applyBoundaryCondition(int x, int y) = 0; // suggestion was (LBMGrid& grid, int x, int y)
    virtual ~BoundaryCondition() = default;
};

template<typename T>
class BounceBack : public BoundaryCondition<T> {
public:
    void applyBoundaryCondition(int x, int y) override;
};

template<typename T>
class FixedVelocityBoundary : public BoundaryCondition<T> {
    std::vector<T> velocity;
public:
    FixedVelocityBoundary(const std::vector<T>& velocity);
    void applyBoundaryCondition(int x, int y) override;
};

template<typename T>
class PeriodicBoundary : public BoundaryCondition<T> {
public:
    void applyBoundaryCondition(int x, int y) override;
};

template<typename T>
class BoundaryConditionManager {
    std::map<BoundaryType, std::map<std::string, std::unique_ptr<BoundaryCondition<T>>>> boundaryConditions;
    //LBMGrid& grid;

public:
    /// Constructor
    BoundaryConditionManager();
    void addBoundaryCondition(BoundaryType boundary, const std::string& name, std::unique_ptr<BoundaryCondition<T>> condition);
    void apply(BoundaryType boundary/*,grid*/);
    void print() const;
};

#include "boundaryConditions.hh"

#endif
