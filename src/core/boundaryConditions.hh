#ifndef BOUNDARY_CONDITIONS_HH
#define BOUNDARY_CONDITIONS_HH

#include "boundaryConditions.h"

template<typename T>
void BounceBack<T>::applyBoundaryCondition(int x, int y)
{

}

template<typename T>
FixedVelocityBoundary<T>::FixedVelocityBoundary(const std::vector<T>& velocity) : velocity(velocity) {}

template<typename T>
void FixedVelocityBoundary<T>::applyBoundaryCondition(int x, int y) {

}

template<typename T>
void PeriodicBoundary<T>::applyBoundaryCondition(int x, int y) {

}

template<typename T>
BoundaryConditionManager<T>::BoundaryConditionManager() {

}

template<typename T>
void BoundaryConditionManager<T>::addBoundaryCondition(BoundaryType boundary, const std::string& name, std::unique_ptr<BoundaryCondition<T>> condition) {
    boundaryConditions[boundary][name] = std::move(condition);
}

template<typename T>
void BoundaryConditionManager<T>::apply(BoundaryType boundary/*,grid*/) {
/*
        auto& conditions = boundaryConditions[boundary];
        for (auto& conditionPair : conditions) {
            conditionPair.second->apply(grid);
        }
*/
}

template<typename T>
void BoundaryConditionManager<T>::print() const {
    std::cout << "====== Boundary Conditions ======" << std::endl;
    for (const auto& boundaryPair : boundaryConditions) {
        std::cout << "Boundary " << static_cast<int>(boundaryPair.first) << " Conditions:\n";
        for (const auto& conditionPair : boundaryPair.second) {
            std::cout << "- " << conditionPair.first << "\n";
        }
    }
    std::cout << "==================================" << std::endl;
    

}

#endif
