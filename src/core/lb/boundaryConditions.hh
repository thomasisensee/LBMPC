#ifndef BOUNDARY_CONDITIONS_HH
#define BOUNDARY_CONDITIONS_HH

#include <cuda_runtime.h>

#include "boundaryConditions.h"

#include "core/constants.h"
#include "core/utilities.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T, unsigned int D, unsigned int Q >
BoundaryCondition<T,D,Q>::BoundaryCondition(BoundaryLocation loc) : _location(loc) {}

template<typename T, unsigned int D, unsigned int Q >
BoundaryLocation BoundaryCondition<T,D,Q>::getLocation() const {
    return _location;
}

template<typename T, unsigned int D, unsigned int Q >
void BoundaryCondition<T,D,Q>::prepareKernelParams(const BaseParams& baseParams) {
    // Set kernel parameters (and duplicate on device)
    _params.setValues(
        baseParams.Nx,
        baseParams.Ny,
        nullptr,
        this->_location
    );

    // Set block and grid size for cuda kernel execution
    this->_threadsPerBlock = THREADS_PER_BLOCK_DIMENSION*THREADS_PER_BLOCK_DIMENSION;
    if (this->_location == BoundaryLocation::EAST || this->_location == BoundaryLocation::WEST) {
        this->_numBlocks = (baseParams.Ny + this->_threadsPerBlock - 1) / this->_threadsPerBlock;
    } else {
        this->_numBlocks = (baseParams.Nx + this->_threadsPerBlock - 1) / this->_threadsPerBlock;
    }
}

template<typename T, unsigned int D, unsigned int Q >
void BoundaryCondition<T,D,Q>::printBoundaryLocation() const {
    std::cout << "== Boundary Location: " << boundaryLocationToString(_location) << "\t==" << std::endl;
}

template<typename T, unsigned int D, unsigned int Q >
void BoundaryCondition<T,D,Q>::printParameters() const {
    printBoundaryLocation();
    std::cout << "== Condition: " << "Base" << "\t=="  << std::endl;
}


/**************************************/
/***** Derived class 01: Periodic *****/
/**************************************/
template<typename T, unsigned int D, unsigned int Q >
Periodic<T,D,Q>::Periodic(BoundaryLocation loc) : BoundaryCondition<T,D,Q>(loc) {}

template<typename T, unsigned int D, unsigned int Q >
void Periodic<T,D,Q>::apply(T* lbmField) {

}

template<typename T, unsigned int D, unsigned int Q >
void Periodic<T,D,Q>::printParameters() const {
    BoundaryCondition<T,D,Q>::printBoundaryLocation();
    std::cout << "== Condition: " << "Periodic" << "\t=="  << std::endl;
}

/*****************************************/
/***** Derived class 02: Bounce Back *****/
/*****************************************/
template<typename T, unsigned int D, unsigned int Q >
BounceBack<T,D,Q>::BounceBack(BoundaryLocation loc) : BoundaryCondition<T,D,Q>(loc) {}

template<typename T, unsigned int D, unsigned int Q >
void BounceBack<T,D,Q>::apply(T* lbmField) {
    dim3 blockSize(this->_threadsPerBlock);
    dim3 gridSize(this->_numBlocks);
    applyBounceBackCaller<T,D,Q>(lbmField, this->_params.getDeviceParams(), gridSize, blockSize);
}

template<typename T, unsigned int D, unsigned int Q >
void BounceBack<T,D,Q>::printParameters() const {
    BoundaryCondition<T,D,Q>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-back" << "\t=="  << std::endl;
}

/**********************************************************/
/***** Derived class 03: Fixed Velocity (Bounce Back) *****/
/**********************************************************/
template<typename T, unsigned int D, unsigned int Q >
MovingWall<T,D,Q>::MovingWall(BoundaryLocation loc, const std::vector<T>& velocity) : BounceBack<T,D,Q>(loc), _wallVelocity(velocity), _dxdt(0.0) {}

template<typename T, unsigned int D, unsigned int Q >
void MovingWall<T,D,Q>::prepareKernelParams(const BaseParams& baseParams) {
    BoundaryCondition<T,D,Q>::prepareKernelParams(baseParams);
    this->_params.setWallVelocity(getWallVelocity());
}

template<typename T, unsigned int D, unsigned int Q >
const std::vector<T>& MovingWall<T,D,Q>::getWallVelocity() const {
    return _wallVelocity;
}

template<typename T, unsigned int D, unsigned int Q >
void MovingWall<T,D,Q>::printParameters() const {
    BoundaryCondition<T,D,Q>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-Back with fixed velocity" << "\t=="  << std::endl;
    std::cout << "== Velocity = {" << getWallVelocity()[0] * _dxdt << ", " << getWallVelocity()[1] * _dxdt << "}\t=="  << std::endl;
}

template<typename T, unsigned int D, unsigned int Q >
void MovingWall<T,D,Q>::setDxdt(T dxdt) {
    _dxdt = dxdt;
}

/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T, unsigned int D, unsigned int Q >
BoundaryConditionManager<T,D,Q>::BoundaryConditionManager() {}

template<typename T, unsigned int D, unsigned int Q >
void BoundaryConditionManager<T,D,Q>::setDxdt(T dxdt) {
    _dxdt = dxdt;
}

template<typename T, unsigned int D, unsigned int Q >
void BoundaryConditionManager<T,D,Q>::addBoundaryCondition(std::unique_ptr<BoundaryCondition<T,D,Q>> condition) {
    for (const auto& existingCondition : boundaryConditions) {
        if (condition->getLocation() == existingCondition->getLocation()) {
            std::cerr << "Error: Boundary condition already exists for location " << boundaryLocationToString(condition->getLocation()) << std::endl;
            exit(EXIT_FAILURE);
        }
    }   

    // If MovingWall, then set dxdt
    MovingWall<T,D,Q>* MWCondition = dynamic_cast<MovingWall<T,D,Q>*>(condition.get());
    if (MWCondition != nullptr) {
        MWCondition->setDxdt(_dxdt);
    }

    boundaryConditions.push_back(std::move(condition));
}

template<typename T, unsigned int D, unsigned int Q >
void BoundaryConditionManager<T,D,Q>::prepareKernelParams(const BaseParams& baseParams) {
    for (const auto& condition : boundaryConditions) {
        condition->prepareKernelParams(baseParams);
    }
}

template<typename T, unsigned int D, unsigned int Q >
void BoundaryConditionManager<T,D,Q>::apply(T* lbmField) {
    for (const auto& condition : boundaryConditions) {
        condition->apply(lbmField);
    }
}

template<typename T, unsigned int D, unsigned int Q >
void BoundaryConditionManager<T,D,Q>::printParameters() const {
    std::cout << "====== Boundary conditions =======" << std::endl;
    for (const auto& condition : boundaryConditions) {
        condition->printParameters();
    }
    std::cout << "==================================\n" << std::endl;
}

#endif // BOUNDARY_CONDITIONS_HH