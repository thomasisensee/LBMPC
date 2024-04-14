#ifndef BOUNDARY_CONDITIONS_HH
#define BOUNDARY_CONDITIONS_HH

#include <cuda_runtime.h>

#include "boundaryConditions.h"

#include "core/constants.h"
#include "core/utilities.h"

/**********************/
/***** Base class *****/
/**********************/
template<typename T,typename LATTICE_DESCRIPTOR>
BoundaryCondition<T,LATTICE_DESCRIPTOR>::BoundaryCondition(BoundaryLocation loc) : _location(loc) {}

template<typename T,typename LATTICE_DESCRIPTOR>
BoundaryLocation BoundaryCondition<T,LATTICE_DESCRIPTOR>::getLocation() const {
    return _location;
}

template<typename T,typename LATTICE_DESCRIPTOR>
void BoundaryCondition<T,LATTICE_DESCRIPTOR>::prepareKernelParams(const BaseParams& baseParams) {
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

template<typename T,typename LATTICE_DESCRIPTOR>
void BoundaryCondition<T,LATTICE_DESCRIPTOR>::printBoundaryLocation() const {
    std::cout << "== Boundary Location: " << boundaryLocationToString(_location) << "\t==" << std::endl;
}

template<typename T,typename LATTICE_DESCRIPTOR>
void BoundaryCondition<T,LATTICE_DESCRIPTOR>::printParameters() const {
    printBoundaryLocation();
    std::cout << "== Condition: " << "Base" << "\t=="  << std::endl;
}


/**************************************/
/***** Derived class 01: Periodic *****/
/**************************************/
template<typename T,typename LATTICE_DESCRIPTOR>
Periodic<T,LATTICE_DESCRIPTOR>::Periodic(BoundaryLocation loc) : BoundaryCondition<T,LATTICE_DESCRIPTOR>(loc) {}

template<typename T,typename LATTICE_DESCRIPTOR>
void Periodic<T,LATTICE_DESCRIPTOR>::apply(T* lbmField) {

}

template<typename T,typename LATTICE_DESCRIPTOR>
void Periodic<T,LATTICE_DESCRIPTOR>::printParameters() const {
    BoundaryCondition<T,LATTICE_DESCRIPTOR>::printBoundaryLocation();
    std::cout << "== Condition: " << "Periodic" << "\t=="  << std::endl;
}

/*****************************************/
/***** Derived class 02: Bounce Back *****/
/*****************************************/
template<typename T,typename LATTICE_DESCRIPTOR>
BounceBack<T,LATTICE_DESCRIPTOR>::BounceBack(BoundaryLocation loc) : BoundaryCondition<T,LATTICE_DESCRIPTOR>(loc) {}

template<typename T,typename LATTICE_DESCRIPTOR>
void BounceBack<T,LATTICE_DESCRIPTOR>::apply(T* lbmField) {
    dim3 blockSize(this->_threadsPerBlock);
    dim3 gridSize(this->_numBlocks);
    applyBounceBackCaller<T,LATTICE_DESCRIPTOR>(lbmField, this->_params.getDeviceParams(), gridSize, blockSize);
}

template<typename T,typename LATTICE_DESCRIPTOR>
void BounceBack<T,LATTICE_DESCRIPTOR>::printParameters() const {
    BoundaryCondition<T,LATTICE_DESCRIPTOR>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-back" << "\t=="  << std::endl;
}

/**********************************************************/
/***** Derived class 03: Fixed Velocity (Bounce Back) *****/
/**********************************************************/
template<typename T,typename LATTICE_DESCRIPTOR>
MovingWall<T,LATTICE_DESCRIPTOR>::MovingWall(BoundaryLocation loc, const std::vector<T>& velocity) : BounceBack<T,LATTICE_DESCRIPTOR>(loc), _wallVelocity(velocity), _dxdt(0.0) {}

template<typename T,typename LATTICE_DESCRIPTOR>
void MovingWall<T,LATTICE_DESCRIPTOR>::prepareKernelParams(const BaseParams& baseParams) {
    BoundaryCondition<T,LATTICE_DESCRIPTOR>::prepareKernelParams(baseParams);
    this->_params.setWallVelocity(getWallVelocity());
}

template<typename T,typename LATTICE_DESCRIPTOR>
const std::vector<T>& MovingWall<T,LATTICE_DESCRIPTOR>::getWallVelocity() const {
    return _wallVelocity;
}

template<typename T,typename LATTICE_DESCRIPTOR>
void MovingWall<T,LATTICE_DESCRIPTOR>::printParameters() const {
    BoundaryCondition<T,LATTICE_DESCRIPTOR>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-Back with fixed velocity" << "\t=="  << std::endl;
    std::cout << "== Velocity = {" << getWallVelocity()[0] * _dxdt << ", " << getWallVelocity()[1] * _dxdt << "}\t=="  << std::endl;
}

/**********************************************************/
/***** Derived class 03: Fixed Velocity (Bounce Back) *****/
/**********************************************************/
template<typename T,typename LATTICE_DESCRIPTOR>
AntiBounceBack<T,LATTICE_DESCRIPTOR>::AntiBounceBack(BoundaryLocation loc, T wallValue) : BounceBack<T,LATTICE_DESCRIPTOR>(loc), _wallValue(wallValue) {}

template<typename T,typename LATTICE_DESCRIPTOR>
void AntiBounceBack<T,LATTICE_DESCRIPTOR>::prepareKernelParams(const BaseParams& baseParams) {
    BoundaryCondition<T,LATTICE_DESCRIPTOR>::prepareKernelParams(baseParams);
    this->_params.setWallValue(getWallValue());
}

template<typename T,typename LATTICE_DESCRIPTOR>
const T AntiBounceBack<T,LATTICE_DESCRIPTOR>::getWallValue() const {
    return _wallValue;
}

template<typename T,typename LATTICE_DESCRIPTOR>
void AntiBounceBack<T,LATTICE_DESCRIPTOR>::printParameters() const {
    BoundaryCondition<T,LATTICE_DESCRIPTOR>::printBoundaryLocation();
    std::cout << "== Condition: " << "Bounce-Back with fixed velocity" << "\t=="  << std::endl;
    std::cout << "== Value = {" << getWallValue()  << "\t=="  << std::endl;
}

template<typename T,typename LATTICE_DESCRIPTOR>
void MovingWall<T,LATTICE_DESCRIPTOR>::setDxdt(T dxdt) {
    _dxdt = dxdt;
}

/*************************/
/***** Wrapper class *****/
/*************************/
template<typename T,typename LATTICE_DESCRIPTOR>
BoundaryConditionManager<T,LATTICE_DESCRIPTOR>::BoundaryConditionManager() {}

template<typename T,typename LATTICE_DESCRIPTOR>
void BoundaryConditionManager<T,LATTICE_DESCRIPTOR>::setDxdt(T dxdt) {
    _dxdt = dxdt;
}

template<typename T,typename LATTICE_DESCRIPTOR>
void BoundaryConditionManager<T,LATTICE_DESCRIPTOR>::addBoundaryCondition(std::unique_ptr<BoundaryCondition<T,LATTICE_DESCRIPTOR>> condition) {
    for (const auto& existingCondition : boundaryConditions) {
        if (condition->getLocation() == existingCondition->getLocation()) {
            std::cerr << "Error: Boundary condition already exists for location " << boundaryLocationToString(condition->getLocation()) << std::endl;
            exit(EXIT_FAILURE);
        }
    }   

    // If MovingWall, then set dxdt
    MovingWall<T,LATTICE_DESCRIPTOR>* MWCondition = dynamic_cast<MovingWall<T,LATTICE_DESCRIPTOR>*>(condition.get());
    if (MWCondition != nullptr) {
        MWCondition->setDxdt(_dxdt);
    }

    boundaryConditions.push_back(std::move(condition));
}

template<typename T,typename LATTICE_DESCRIPTOR>
void BoundaryConditionManager<T,LATTICE_DESCRIPTOR>::prepareKernelParams(const BaseParams& baseParams) {
    for (const auto& condition : boundaryConditions) {
        condition->prepareKernelParams(baseParams);
    }
}

template<typename T,typename LATTICE_DESCRIPTOR>
void BoundaryConditionManager<T,LATTICE_DESCRIPTOR>::apply(T* lbmField) {
    for (const auto& condition : boundaryConditions) {
        condition->apply(lbmField);
    }
}

template<typename T,typename LATTICE_DESCRIPTOR>
void BoundaryConditionManager<T,LATTICE_DESCRIPTOR>::printParameters() const {
    std::cout << "====== Boundary conditions =======" << std::endl;
    for (const auto& condition : boundaryConditions) {
        condition->printParameters();
    }
    std::cout << "==================================\n" << std::endl;
}

#endif // BOUNDARY_CONDITIONS_HH