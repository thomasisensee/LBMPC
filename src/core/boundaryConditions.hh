#ifndef BOUNDARY_CONDITIONS_HH
#define BOUNDARY_CONDITIONS_HH

#include "boundaryConditions.h"


/***************************/
/***** Derived classes *****/
/***************************/
template<typename T>
void BounceBack<T>::apply(T* lbmField) {

}

template<typename T>
FixedVelocityBoundary<T>::FixedVelocityBoundary(const std::vector<T>& velocity) : velocity(velocity) {}

template<typename T>
void FixedVelocityBoundary<T>::apply(T* lbmField) {

}

template<typename T>
void PeriodicBoundary<T>::apply(T* lbmField) {

}


/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T>
BoundaryConditionManager<T>::BoundaryConditionManager() {

}

template<typename T>
void BoundaryConditionManager<T>::addBoundaryCondition(BoundaryLocation boundary, const std::string& name, std::unique_ptr<BoundaryCondition<T>> condition) {
    boundaryConditions[boundary][name] = std::move(condition);
}

template<typename T>
void BoundaryConditionManager<T>::apply(T* lbmField) {
    // Iterate over each boundary location
    for (auto& boundaryConditionsPair : boundaryConditions) {
        // Now iterate over each condition for this boundary
        for (auto& conditionPair : boundaryConditionsPair.second) {
            // Apply the boundary condition
            conditionPair.second->apply(lbmField);
        }
    }
}

template<typename T>
void BoundaryConditionManager<T>::print() const {
    std::cout << "====== Boundary conditions =======" << std::endl;
    for (const auto& boundaryConditionsPair : boundaryConditions) {
        std::cout << "== Boundary Location: " << boundaryLocationToString(boundaryConditionsPair.first) << "\t==" << std::endl;
        for (const auto& conditionPair : boundaryConditionsPair.second) {
            std::cout << "== Condition: " << conditionPair.first << "\t=="  << std::endl;
            std::cout << "==\t\t\t\t==\n";
            // If your BoundaryCondition class has more details to print, you can do so here
            // conditionPair.second->printDetails(); // Assuming such a method exists
        }
    }
    std::cout << "==================================\n" << std::endl;
}
#endif
