#ifndef BOUNDARY_CONDITIONS_HH
#define BOUNDARY_CONDITIONS_HH

#include <cuda_runtime.h>

#include "boundaryConditions.h"

#include "core/constants.h"
#include "core/utilities.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T, typename LatticeDescriptor>
BoundaryCondition<T, LatticeDescriptor>::BoundaryCondition(BoundaryLocation loc) : _location(loc) {}

template<typename T, typename LatticeDescriptor>
BoundaryLocation BoundaryCondition<T, LatticeDescriptor>::getLocation() const {
    return _location;
}

template<typename T, typename LatticeDescriptor>
void BoundaryCondition<T, LatticeDescriptor>::prepareKernelParams(const BaseParams& baseParams) {
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

template<typename T, typename LatticeDescriptor>
void BoundaryCondition<T, LatticeDescriptor>::printBoundaryLocation() const {
    std::cout << "== Boundary Location: " << boundaryLocationToString(_location) << "\t==" << std::endl;
}

template<typename T, typename LatticeDescriptor>
void BoundaryCondition<T, LatticeDescriptor>::printParameters() const {
    printBoundaryLocation();
    std::cout << "== Condition: " << "Base" << "\t=="  << std::endl;
}


/**************************************/
/***** Derived class 01: Periodic *****/
/**************************************/
template<typename T, typename LatticeDescriptor>
Periodic<T, LatticeDescriptor>::Periodic(BoundaryLocation loc) : BoundaryCondition<T, LatticeDescriptor>(loc) {}

template<typename T, typename LatticeDescriptor>
void Periodic<T, LatticeDescriptor>::apply(T* lbmField) {

}

template<typename T, typename LatticeDescriptor>
void Periodic<T, LatticeDescriptor>::printParameters() const {
    BoundaryCondition<T, LatticeDescriptor>::printBoundaryLocation();
    std::cout << "== Condition: " << "Periodic" << "\t=="  << std::endl;
}

/*****************************************/
/***** Derived class 02: Bounce Back *****/
/*****************************************/
template<typename T, typename LatticeDescriptor>
BounceBack<T, LatticeDescriptor>::BounceBack(BoundaryLocation loc) : BoundaryCondition<T, LatticeDescriptor>(loc) {}

template<typename T, typename LatticeDescriptor>
void BounceBack<T, LatticeDescriptor>::apply(T* lbmField) {
    dim3 blockSize(this->_threadsPerBlock);
    dim3 gridSize(this->_numBlocks);
    applyBounceBackCaller<T, LatticeDescriptor>(lbmField, this->_params.getDeviceParams(), gridSize, blockSize);
}

template<typename T, typename LatticeDescriptor>
void BounceBack<T, LatticeDescriptor>::printParameters() const {
    BoundaryCondition<T, LatticeDescriptor>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-back" << "\t=="  << std::endl;
}

/**********************************************************/
/***** Derived class 03: Fixed Velocity (Bounce Back) *****/
/**********************************************************/
template<typename T, typename LatticeDescriptor>
MovingWall<T, LatticeDescriptor>::MovingWall(BoundaryLocation loc, const std::vector<T>& velocity) : BounceBack<T, LatticeDescriptor>(loc), _wallVelocity(velocity), _dxdt(0.0) {}

template<typename T, typename LatticeDescriptor>
void MovingWall<T, LatticeDescriptor>::prepareKernelParams(const BaseParams& baseParams) {
    BoundaryCondition<T, LatticeDescriptor>::prepareKernelParams(baseParams);
    this->_params.setWallVelocity(getWallVelocity());
}

template<typename T, typename LatticeDescriptor>
const std::vector<T>& MovingWall<T, LatticeDescriptor>::getWallVelocity() const {
    return _wallVelocity;
}

template<typename T, typename LatticeDescriptor>
void MovingWall<T, LatticeDescriptor>::printParameters() const {
    BoundaryCondition<T, LatticeDescriptor>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-Back with fixed velocity" << "\t=="  << std::endl;
    std::cout << "== Velocity = {" << getWallVelocity()[0] * _dxdt << ", " << getWallVelocity()[1] * _dxdt << "}\t=="  << std::endl;
}

template<typename T, typename LatticeDescriptor>
void MovingWall<T, LatticeDescriptor>::setDxdt(T dxdt) {
    _dxdt = dxdt;
}

/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T, typename LatticeDescriptor>
BoundaryConditionManager<T, LatticeDescriptor>::BoundaryConditionManager() {}

template<typename T, typename LatticeDescriptor>
void BoundaryConditionManager<T, LatticeDescriptor>::setDxdt(T dxdt) {
    _dxdt = dxdt;
}

template<typename T, typename LatticeDescriptor>
void BoundaryConditionManager<T, LatticeDescriptor>::addBoundaryCondition(std::unique_ptr<BoundaryCondition<T, LatticeDescriptor>> condition) {
    for (const auto& existingCondition : boundaryConditions) {
        if (condition->getLocation() == existingCondition->getLocation()) {
            std::cerr << "Error: Boundary condition already exists for location " << boundaryLocationToString(condition->getLocation()) << std::endl;
            exit(EXIT_FAILURE);
        }
    }   

    // If MovingWall, then set dxdt
    MovingWall<T, LatticeDescriptor>* MWCondition = dynamic_cast<MovingWall<T, LatticeDescriptor>*>(condition.get());
    if (MWCondition != nullptr) {
        MWCondition->setDxdt(_dxdt);
    }

    boundaryConditions.push_back(std::move(condition));
}

template<typename T, typename LatticeDescriptor>
void BoundaryConditionManager<T, LatticeDescriptor>::prepareKernelParams(const BaseParams& baseParams) {
    for (const auto& condition : boundaryConditions) {
        condition->prepareKernelParams(baseParams);
    }
}

template<typename T, typename LatticeDescriptor>
void BoundaryConditionManager<T, LatticeDescriptor>::apply(T* lbmField) {
    for (const auto& condition : boundaryConditions) {
        condition->apply(lbmField);
    }
}

template<typename T, typename LatticeDescriptor>
void BoundaryConditionManager<T, LatticeDescriptor>::printParameters() const {
    std::cout << "====== Boundary conditions =======" << std::endl;
    for (const auto& condition : boundaryConditions) {
        condition->printParameters();
    }
    std::cout << "==================================\n" << std::endl;
}

#endif // BOUNDARY_CONDITIONS_HH